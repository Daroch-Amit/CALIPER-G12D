"""Model comparison, hyperparameter tuning and export of the deployable bundle.

A scaffold-disjoint test set is split off first and never touched during model
selection or tuning, so the reported hold-out numbers are honest.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from .featurization import FeatureGenerator
from .preprocessing import FeaturePreprocessor

# Both of these are cosmetic: LightGBM complains about unnamed arrays and
# LassoCV reports a soft tolerance miss that does not affect the selection.
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

RDLogger.DisableLog("rdApp.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kras.train")

SEED = 42


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES, or empty string for acyclic/unparseable input."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold) if scaffold.GetNumAtoms() else ""
    except Exception:
        return ""


def scaffold_groups(smiles_list: list[str]) -> np.ndarray:
    """Integer group id per molecule. Acyclic molecules each get their own group."""
    mapping: dict[str, int] = {}
    codes = np.empty(len(smiles_list), dtype=np.int64)
    for i, smi in enumerate(smiles_list):
        scaffold = murcko_scaffold(smi)
        key = scaffold if scaffold else f"__acyclic_{i}"
        if key not in mapping:
            mapping[key] = len(mapping)
        codes[i] = mapping[key]
    return codes


def load_representation(artifacts: Path, tag: str, target_col: str):
    X = np.load(artifacts / f"features_{tag}.npz")["X"].astype(np.float64)
    names = json.loads((artifacts / f"feature_names_{tag}.json").read_text())
    aligned = pd.read_csv(artifacts / f"aligned_{tag}.csv")

    y = aligned[target_col].to_numpy(dtype=float)
    smiles_col = (
        "canonical_smiles" if "canonical_smiles" in aligned.columns else aligned.columns[0]
    )
    smiles = aligned[smiles_col].astype(str).tolist()

    if not (X.shape[0] == len(y) == len(smiles)):
        raise ValueError(
            f"[{tag}] row mismatch: X={X.shape[0]} y={len(y)} smiles={len(smiles)}"
        )
    return X, names, y, smiles


def build_model_zoo(seed: int = SEED) -> dict:
    """Every regressor available in this environment, keyed by name."""
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR

    zoo = {
        "Ridge": Ridge(alpha=1.0, random_state=seed),
        "KNN": KNeighborsRegressor(n_neighbors=7, weights="distance", n_jobs=1),
        "SVR": SVR(C=10.0, gamma="scale"),
        "RandomForest": RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=500, random_state=seed, n_jobs=1),
        "GradientBoosting": GradientBoostingRegressor(random_state=seed),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=seed),
    }

    try:
        from xgboost import XGBRegressor

        zoo["XGBoost"] = XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            verbosity=0,
            objective="reg:squarederror",
        )
    except ImportError:
        log.warning("xgboost not installed, skipping")

    try:
        from lightgbm import LGBMRegressor

        zoo["LightGBM"] = LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
    except ImportError:
        log.warning("lightgbm not installed, skipping")

    try:
        from catboost import CatBoostRegressor

        zoo["CatBoost"] = CatBoostRegressor(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            random_state=seed,
            verbose=0,
            thread_count=1,
        )
    except ImportError:
        log.warning("catboost not installed, skipping")

    return zoo


def resolve_model_name(requested: str, zoo: dict) -> str:
    lookup = {name.lower(): name for name in zoo}
    key = requested.strip().lower()
    if key not in lookup:
        raise SystemExit(f"--model '{requested}' unavailable. options: {sorted(zoo)}")
    return lookup[key]


def regression_metrics(y_true, y_pred) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
    }


def scaffold_holdout_split(smiles: list[str], test_size: float, seed: int):
    """Assign whole scaffold groups to the test set until it is large enough."""
    groups = defaultdict(list)
    for i, smi in enumerate(smiles):
        scaffold = murcko_scaffold(smi)
        groups[scaffold if scaffold else f"__acyclic_{i}"].append(i)

    rng = np.random.default_rng(seed)
    buckets = list(groups.values())
    rng.shuffle(buckets)

    target = int(round(test_size * len(smiles)))
    test_idx: list[int] = []
    for bucket in buckets:
        if len(test_idx) >= target:
            break
        test_idx.extend(bucket)

    mask = np.zeros(len(smiles), dtype=bool)
    mask[test_idx] = True
    return np.where(~mask)[0], np.where(mask)[0]


def cv_compare(X, y, groups, names, zoo, n_folds, preprocessor_kwargs, seed):
    """Score every model in the zoo under a shared scaffold-grouped CV split."""
    cv = GroupKFold(n_splits=n_folds)
    scores = {name: [] for name in zoo}

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups), start=1):
        preprocessor = FeaturePreprocessor(**preprocessor_kwargs)
        preprocessor.fit(X[train_idx], y[train_idx], names)
        X_train = preprocessor.transform(X[train_idx])
        X_valid = preprocessor.transform(X[valid_idx])
        log.info("fold %d/%d: %d features", fold, n_folds, X_train.shape[1])

        for name, template in zoo.items():
            model = clone(template)
            model.fit(X_train, y[train_idx])
            rmse = np.sqrt(mean_squared_error(y[valid_idx], model.predict(X_valid)))
            scores[name].append(rmse)

    rows = [
        {"model": name, "cv_rmse": float(np.mean(v)), "cv_rmse_sd": float(np.std(v))}
        for name, v in scores.items()
    ]
    return pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)


def suggest_model(name: str, trial, seed: int):
    """Optuna search space per family, also used to rebuild the tuned model."""
    if name in ("RandomForest", "ExtraTrees"):
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

        cls = RandomForestRegressor if name == "RandomForest" else ExtraTreesRegressor
        return cls(
            n_estimators=trial.suggest_int("n_estimators", 300, 800, step=100),
            max_depth=trial.suggest_int("max_depth", 6, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_float("max_features", 0.2, 1.0),
            random_state=seed,
            n_jobs=1,
        )

    if name == "XGBoost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 300, 900, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 9),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )

    if name == "LightGBM":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=trial.suggest_int("n_estimators", 300, 900, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )

    if name == "CatBoost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=trial.suggest_int("iterations", 300, 900, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            depth=trial.suggest_int("depth", 4, 8),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            random_state=seed,
            verbose=0,
            thread_count=1,
        )

    if name == "SVR":
        from sklearn.svm import SVR

        return SVR(
            C=trial.suggest_float("C", 1e-1, 1e3, log=True),
            gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
            epsilon=trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
        )

    if name == "Ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=trial.suggest_float("alpha", 1e-2, 100, log=True), random_state=seed)

    if name == "KNN":
        from sklearn.neighbors import KNeighborsRegressor

        return KNeighborsRegressor(
            n_neighbors=trial.suggest_int("n_neighbors", 3, 25),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            p=trial.suggest_int("p", 1, 2),
            n_jobs=1,
        )

    if name == "GradientBoosting":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            n_estimators=trial.suggest_int("n_estimators", 150, 500, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 5),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            random_state=seed,
        )

    if name == "HistGradientBoosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_iter=trial.suggest_int("max_iter", 200, 600, step=100),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
            random_state=seed,
        )

    raise ValueError(f"no search space defined for {name}")


def tune_model(name, X, y, groups, n_folds, n_trials, seed):
    """Optimise one model family against scaffold-grouped CV RMSE."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = GroupKFold(n_splits=n_folds)

    def objective(trial):
        candidate = suggest_model(name, trial, seed)
        fold_rmse = []
        for train_idx, valid_idx in cv.split(X, y, groups):
            model = clone(candidate)
            model.fit(X[train_idx], y[train_idx])
            fold_rmse.append(
                np.sqrt(mean_squared_error(y[valid_idx], model.predict(X[valid_idx])))
            )
        return float(np.mean(fold_rmse))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("tuning %s: best CV RMSE=%.3f", name, study.best_value)

    tuned = suggest_model(name, optuna.trial.FixedTrial(study.best_params), seed)
    return tuned, study.best_params


def main() -> None:
    import joblib

    parser = argparse.ArgumentParser(description="Train and export the final model")
    parser.add_argument("--representation", default="ecfp")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--target-col", default="pIC50")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--scaler", default="robust", choices=["standard", "robust", "none"])
    parser.add_argument("--max-features", type=int, default=300)
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument(
        "--compare-selection", default="none", choices=["none", "model", "consensus"]
    )
    parser.add_argument("--model", default=None, help="force a model family, e.g. xgboost")
    parser.add_argument("--no-compare", action="store_true", help="skip the CV comparison")
    parser.add_argument("--tune", type=int, default=0, help="Optuna trials (0 disables tuning)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--outdir", default="models")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = args.representation.replace("+", "_")

    X, names, y, smiles = load_representation(artifacts, tag, args.target_col)
    log.info("loaded %s: %d molecules x %d features", tag, *X.shape)

    train_idx, test_idx = scaffold_holdout_split(smiles, args.test_size, args.seed)
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = scaffold_groups([smiles[i] for i in train_idx])
    log.info("scaffold split: %d train / %d test", len(train_idx), len(test_idx))

    compare_kwargs = dict(
        scaler=args.scaler,
        selection_method=args.compare_selection,
        max_features=args.max_features,
        use_lightgbm=True,
        seed=args.seed,
    )
    deploy_kwargs = dict(
        scaler=args.scaler,
        selection_method="consensus",
        min_votes=args.min_votes,
        max_features=args.max_features,
        use_lightgbm=True,
        seed=args.seed,
    )

    zoo = build_model_zoo(args.seed)
    if args.no_compare and not args.model:
        raise SystemExit("--no-compare requires --model")

    table = None
    if not args.no_compare:
        log.info("comparing %d model families under scaffold CV", len(zoo))
        table = cv_compare(
            X_train_raw, y_train, groups_train, names, zoo, args.folds, compare_kwargs, args.seed
        )
        table.to_csv(artifacts / f"model_comparison_{tag}.csv", index=False)
        print(f"\n=== model comparison ({args.folds}-fold scaffold CV) ===")
        with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
            print(table.to_string(index=False))

    if args.model:
        best_name = resolve_model_name(args.model, zoo)
        log.info("using requested model: %s", best_name)
    else:
        best_name = table.iloc[0]["model"]
        log.info("best family: %s (CV RMSE=%.3f)", best_name, table.iloc[0]["cv_rmse"])

    log.info("fitting deployment preprocessor on the full training partition")
    preprocessor = FeaturePreprocessor(**deploy_kwargs)
    preprocessor.fit(X_train_raw, y_train, names)
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    log.info("deployment features: %d", X_train.shape[1])

    best_params = None
    if args.tune:
        log.info("tuning %s with %d Optuna trials", best_name, args.tune)
        model, best_params = tune_model(
            best_name, X_train, y_train, groups_train, args.folds, args.tune, args.seed
        )
    else:
        model = clone(zoo[best_name])
    model.fit(X_train, y_train)

    test_metrics = regression_metrics(y_test, model.predict(X_test))
    log.info(
        "hold-out: R2=%.3f RMSE=%.3f MAE=%.3f",
        test_metrics["r2"],
        test_metrics["rmse"],
        test_metrics["mae"],
    )

    cv_rmse = None
    if table is not None:
        matched = table.loc[table["model"] == best_name, "cv_rmse"]
        cv_rmse = float(matched.iloc[0]) if len(matched) else None

    bundle = {
        "feature_generator": FeatureGenerator(args.representation),
        "preprocessor": preprocessor,
        "model": model,
        "meta": {
            "representation": args.representation,
            "model": best_name,
            "best_params": best_params,
            "seed": args.seed,
            "cv_rmse": cv_rmse,
            "test_metrics": test_metrics,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_features": int(X_train.shape[1]),
            # Stored so the uncertainty step can rebuild the identical split.
            "test_size": args.test_size,
            "folds": args.folds,
            "trained_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    joblib.dump(bundle, outdir / f"model_bundle_{tag}.joblib", compress=3)
    pd.DataFrame({"y_true": y_test, "y_pred": model.predict(X_test)}).to_csv(
        outdir / f"test_predictions_{tag}.csv", index=False
    )
    (outdir / f"metrics_{tag}.json").write_text(
        json.dumps(bundle["meta"], indent=2, default=str)
    )

    print(
        f"\nbest model: {best_name} | hold-out R2={test_metrics['r2']:.3f} "
        f"RMSE={test_metrics['rmse']:.3f}"
    )
    print(f"saved {outdir}/model_bundle_{tag}.joblib")


if __name__ == "__main__":
    main()
