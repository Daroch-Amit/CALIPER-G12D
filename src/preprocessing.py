"""Preprocessing and feature selection.

FeaturePreprocessor is fit once on training data and then applied unchanged to
new molecules, so the transform is identical at training and inference time.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectFromModel,
    VarianceThreshold,
    f_regression,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kras.preprocessing")

SEED = 42


class FeaturePreprocessor:
    """Impute -> variance filter -> correlation filter -> scale -> select.

    The default selection_method is a consensus vote across five independent
    selectors; a feature is kept only if at least `min_votes` of them nominate
    it. Set selection_method="none" when the downstream model is expected to do
    its own feature weighting, as during cross-validated model comparison.
    """

    def __init__(
        self,
        variance_threshold: float = 1e-4,
        correlation_threshold: float = 0.95,
        scaler: str = "robust",
        selection_method: str = "consensus",
        select_fraction: float = 0.25,
        min_votes: int = 2,
        max_features: int | None = 300,
        min_features: int | None = 10,
        use_lightgbm: bool = True,
        use_rfecv: bool = False,
        use_boruta: bool = False,
        use_pca: bool = False,
        pca_variance: float = 0.99,
        seed: int = SEED,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.scaler_kind = scaler
        self.selection_method = selection_method
        self.select_fraction = select_fraction
        self.min_votes = min_votes
        self.max_features = max_features
        self.min_features = min_features
        self.use_lightgbm = use_lightgbm
        self.use_rfecv = use_rfecv
        self.use_boruta = use_boruta
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.seed = seed

        self.imputer_ = None
        self.var_mask_ = None
        self.corr_mask_ = None
        self.scaler_ = None
        self.select_mask_ = None
        self.pca_ = None
        self.input_names_: list[str] | None = None
        self.selected_features_: list[str] | None = None
        self.report_: dict = {}

    # --- helpers ----------------------------------------------------------

    def _correlation_mask(self, X: np.ndarray) -> np.ndarray:
        """Drop the lower-variance member of every strongly correlated pair."""
        if X.shape[1] < 2:
            return np.ones(X.shape[1], dtype=bool)

        variance = X.var(axis=0)
        corr = np.nan_to_num(np.corrcoef(X, rowvar=False), nan=0.0)
        upper = np.triu(np.abs(corr), k=1)

        keep = np.ones(X.shape[1], dtype=bool)
        for j in range(1, X.shape[1]):
            for i in np.where(upper[:, j] > self.correlation_threshold)[0]:
                if not keep[i]:
                    continue
                if variance[j] >= variance[i]:
                    keep[i] = False
                else:
                    keep[j] = False
                    break
        return keep & (variance > 0)

    @staticmethod
    def _top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
        mask = np.zeros(len(scores), dtype=bool)
        if k > 0:
            order = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1][:k]
            mask[order] = True
        return mask

    def _consensus_votes(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Let several selectors vote; return the vote count per feature."""
        n_features = X.shape[1]
        k = max(1, int(self.select_fraction * n_features))
        votes = np.zeros(n_features, dtype=int)
        log.info("consensus selection: each selector nominates top %d of %d", k, n_features)

        votes += self._top_k_mask(
            mutual_info_regression(X, y, random_state=self.seed), k
        ).astype(int)

        f_stat, _ = f_regression(X, y)
        votes += self._top_k_mask(f_stat, k).astype(int)

        forest = RandomForestRegressor(
            n_estimators=300, max_features="sqrt", random_state=self.seed, n_jobs=1
        )
        forest.fit(X, y)
        votes += self._top_k_mask(forest.feature_importances_, k).astype(int)

        if self.use_lightgbm:
            try:
                from lightgbm import LGBMRegressor

                booster = LGBMRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    random_state=self.seed,
                    n_jobs=1,
                    verbose=-1,
                )
                booster.fit(X, y)
                gains = np.asarray(booster.feature_importances_, dtype=float)
                votes += self._top_k_mask(gains, k).astype(int)
            except Exception as exc:
                log.warning("skipping LightGBM vote: %s", exc)

        try:
            lasso = SelectFromModel(
                LassoCV(cv=5, random_state=self.seed, n_jobs=1, max_iter=20000)
            )
            lasso.fit(X, y)
            votes += lasso.get_support().astype(int)
        except Exception as exc:
            log.warning("skipping Lasso vote: %s", exc)

        return votes

    def _apply_bounds(self, mask: np.ndarray, votes: np.ndarray) -> np.ndarray:
        """Relax or tighten the vote threshold to respect min/max feature counts."""
        if self.min_features and mask.sum() < self.min_features:
            order = np.argsort(votes)[::-1][: self.min_features]
            mask = np.zeros_like(mask)
            mask[order] = True
        if self.max_features and mask.sum() > self.max_features:
            chosen = np.where(mask)[0]
            keep = chosen[np.argsort(votes[chosen])[::-1][: self.max_features]]
            mask = np.zeros_like(mask)
            mask[keep] = True
        return mask

    def _boruta_mask(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        from boruta import BorutaPy

        forest = RandomForestRegressor(n_estimators=200, random_state=self.seed, n_jobs=1)
        selector = BorutaPy(forest, n_estimators="auto", random_state=self.seed)
        selector.fit(X, y)
        return selector.support_

    # --- fit / transform --------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.input_names_ = list(feature_names)
        names = np.asarray(feature_names, dtype=object)
        report = {"n_input": int(X.shape[1])}

        # RDKit and Mordred both emit +/-inf for some descriptors; treat those
        # as missing. keep_empty_features matters here: without it SimpleImputer
        # silently drops all-NaN columns and the mask/name arrays desync.
        X = np.where(np.isfinite(X), X, np.nan)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        X = self.imputer_.fit_transform(X)

        selector = VarianceThreshold(self.variance_threshold).fit(X)
        self.var_mask_ = selector.get_support()
        X, names = X[:, self.var_mask_], names[self.var_mask_]
        report["after_variance"] = int(X.shape[1])
        log.info("after variance filter: %d features", X.shape[1])

        self.corr_mask_ = self._correlation_mask(X)
        X, names = X[:, self.corr_mask_], names[self.corr_mask_]
        report["after_correlation"] = int(X.shape[1])
        log.info("after correlation filter: %d features", X.shape[1])

        if self.scaler_kind == "standard":
            self.scaler_ = StandardScaler()
        elif self.scaler_kind == "robust":
            self.scaler_ = RobustScaler()
        else:
            self.scaler_ = None
        if self.scaler_ is not None:
            X = self.scaler_.fit_transform(X)

        if self.selection_method == "none":
            mask = np.ones(X.shape[1], dtype=bool)
        elif self.use_boruta:
            mask = self._boruta_mask(X, y)
        elif self.selection_method == "consensus":
            votes = self._consensus_votes(X, y)
            report["max_votes"] = int(votes.max()) if len(votes) else 0
            mask = self._apply_bounds(votes >= self.min_votes, votes)
        elif self.selection_method == "mutual_info":
            scores = mutual_info_regression(X, y, random_state=self.seed)
            k = self.max_features or max(1, int(self.select_fraction * X.shape[1]))
            mask = self._top_k_mask(scores, k)
        elif self.selection_method == "model":
            forest = RandomForestRegressor(n_estimators=400, random_state=self.seed, n_jobs=1)
            forest.fit(X, y)
            k = self.max_features or max(1, int(self.select_fraction * X.shape[1]))
            mask = self._top_k_mask(forest.feature_importances_, k)
        elif self.selection_method == "rfe":
            k = self.max_features or max(10, X.shape[1] // 4)
            estimator = RandomForestRegressor(
                n_estimators=200, random_state=self.seed, n_jobs=1
            )
            mask = RFE(estimator, n_features_to_select=k, step=0.1).fit(X, y).get_support()
        else:
            raise ValueError(f"unknown selection_method: {self.selection_method}")

        if self.use_rfecv and mask.sum() > (self.min_features or 5):
            index = np.where(mask)[0]
            estimator = RandomForestRegressor(
                n_estimators=200, random_state=self.seed, n_jobs=1
            )
            rfecv = RFECV(
                estimator,
                step=0.1,
                cv=KFold(5, shuffle=True, random_state=self.seed),
                scoring="r2",
                min_features_to_select=self.min_features or 10,
                n_jobs=1,
            )
            rfecv.fit(X[:, index], y)
            refined = np.zeros_like(mask)
            refined[index[rfecv.get_support()]] = True
            mask = refined
            report["rfecv_optimal"] = int(mask.sum())

        self.select_mask_ = mask
        X, names = X[:, mask], names[mask]
        report["after_selection"] = int(X.shape[1])
        log.info("after selection: %d features", X.shape[1])

        if self.use_pca:
            self.pca_ = PCA(n_components=self.pca_variance, random_state=self.seed)
            X = self.pca_.fit_transform(X)
            names = np.asarray([f"PC{i + 1}" for i in range(X.shape[1])], dtype=object)
            report["after_pca"] = int(X.shape[1])

        self.selected_features_ = list(names)
        report["n_output"] = int(X.shape[1])
        self.report_ = report
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.imputer_ is None:
            raise RuntimeError("call fit() before transform()")
        X = np.asarray(X, dtype=float)
        X = np.where(np.isfinite(X), X, np.nan)
        X = self.imputer_.transform(X)
        X = X[:, self.var_mask_]
        X = X[:, self.corr_mask_]
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        X = X[:, self.select_mask_]
        if self.pca_ is not None:
            X = self.pca_.transform(X)
        return X

    def fit_transform(self, X, y, feature_names) -> np.ndarray:
        return self.fit(X, y, feature_names).transform(X)

    def save(self, path) -> None:
        import joblib

        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path) -> "FeaturePreprocessor":
        import joblib

        return joblib.load(path)
