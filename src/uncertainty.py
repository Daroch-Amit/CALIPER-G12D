"""Conformal prediction intervals and a Tanimoto applicability domain.

Both objects are written back into the model bundle so the app only ever has to
load a single file.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as rfg
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from .train import load_representation, scaffold_groups, scaffold_holdout_split

RDLogger.DisableLog("rdApp.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kras.uncertainty")


@dataclass
class ConformalRegressor:
    """Holds the calibrated half-width; intervals are simply prediction +/- q_hat."""

    q_hat: float
    alpha: float

    def interval(self, predictions: np.ndarray):
        predictions = np.asarray(predictions, dtype=float)
        return predictions - self.q_hat, predictions + self.q_hat


def cv_plus_qhat(model, X, y, groups, n_folds: int, alpha: float, seed: int) -> float:
    """Quantile of out-of-fold absolute residuals, i.e. the interval half-width."""
    cv = GroupKFold(n_splits=n_folds)
    out_of_fold = np.full(len(y), np.nan)

    for train_idx, valid_idx in cv.split(X, y, groups):
        fold_model = clone(model)
        fold_model.fit(X[train_idx], y[train_idx])
        out_of_fold[valid_idx] = fold_model.predict(X[valid_idx])

    residuals = np.abs(y - out_of_fold)
    n = len(residuals)
    # Finite-sample correction; without the (n+1)/n factor the interval is
    # slightly too narrow and coverage undershoots the nominal level.
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(residuals, level))


@dataclass
class ApplicabilityDomain:
    """Flags query molecules that sit outside the training chemical space."""

    threshold: float
    radius: int = 2
    n_bits: int = 2048
    _fingerprints: list = field(default_factory=list, repr=False)

    @staticmethod
    def _fingerprint(smiles: str, radius: int, n_bits: int):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        generator = rfg.GetMorganGenerator(radius=radius, fpSize=n_bits)
        return generator.GetFingerprint(mol)

    @classmethod
    def fit(
        cls,
        smiles: list[str],
        percentile: float = 5.0,
        radius: int = 2,
        n_bits: int = 2048,
    ) -> "ApplicabilityDomain":
        generator = rfg.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fingerprints = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fingerprints.append(generator.GetFingerprint(mol))

        nearest = []
        for i, fp in enumerate(fingerprints):
            similarities = DataStructs.BulkTanimotoSimilarity(fp, fingerprints)
            similarities[i] = -1.0  # ignore the molecule matching itself
            nearest.append(max(similarities))

        return cls(
            threshold=float(np.percentile(nearest, percentile)),
            radius=radius,
            n_bits=n_bits,
            _fingerprints=fingerprints,
        )

    def max_similarity(self, smiles: str) -> float:
        fp = self._fingerprint(smiles, self.radius, self.n_bits)
        if fp is None or not self._fingerprints:
            return 0.0
        return float(max(DataStructs.BulkTanimotoSimilarity(fp, self._fingerprints)))

    def in_domain(self, smiles: str):
        similarity = self.max_similarity(smiles)
        return similarity >= self.threshold, similarity


def main() -> None:
    import joblib

    parser = argparse.ArgumentParser(description="Add conformal intervals and an AD flag")
    parser.add_argument("--bundle", required=True, help="models/model_bundle_<tag>.joblib")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--target-col", default="pIC50")
    parser.add_argument("--alpha", type=float, default=0.1, help="0.1 gives 90% intervals")
    parser.add_argument("--ad-percentile", type=float, default=5.0)
    args = parser.parse_args()

    bundle = joblib.load(args.bundle)
    meta = bundle["meta"]
    tag = meta["representation"].replace("+", "_")
    seed = meta.get("seed", 42)
    test_size = meta.get("test_size", 0.15)
    folds = meta.get("folds", 5)
    log.info("loaded bundle: model=%s representation=%s", meta["model"], meta["representation"])

    X, _, y, smiles = load_representation(Path(args.artifacts), tag, args.target_col)
    train_idx, test_idx = scaffold_holdout_split(smiles, test_size, seed)

    preprocessor = bundle["preprocessor"]
    X_train = preprocessor.transform(X[train_idx])
    X_test = preprocessor.transform(X[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]
    smiles_train = [smiles[i] for i in train_idx]
    smiles_test = [smiles[i] for i in test_idx]
    groups_train = scaffold_groups(smiles_train)

    log.info("calibrating conformal intervals (%d folds, alpha=%.2f)", folds, args.alpha)
    q_hat = cv_plus_qhat(
        clone(bundle["model"]), X_train, y_train, groups_train, folds, args.alpha, seed
    )
    conformal = ConformalRegressor(q_hat=q_hat, alpha=args.alpha)

    predictions = bundle["model"].predict(X_test)
    lower, upper = conformal.interval(predictions)
    coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
    log.info(
        "q_hat=%.3f pIC50 | test coverage=%.3f (target %.2f) | width=%.2f",
        q_hat,
        coverage,
        1 - args.alpha,
        2 * q_hat,
    )

    log.info("fitting applicability domain (percentile=%.1f)", args.ad_percentile)
    domain = ApplicabilityDomain.fit(smiles_train, args.ad_percentile)
    fraction_in = float(np.mean([domain.in_domain(s)[0] for s in smiles_test]))
    log.info(
        "AD threshold=%.3f | %.0f%% of test molecules in-domain",
        domain.threshold,
        100 * fraction_in,
    )

    bundle["conformal"] = conformal
    bundle["applicability_domain"] = domain
    meta["conformal"] = {
        "alpha": args.alpha,
        "q_hat": q_hat,
        "test_coverage": coverage,
        "interval_width": 2 * q_hat,
    }
    meta["applicability_domain"] = {
        "tanimoto_threshold": domain.threshold,
        "test_fraction_in_domain": fraction_in,
    }

    joblib.dump(bundle, args.bundle, compress=3)
    Path(args.bundle).with_name(f"uncertainty_report_{tag}.json").write_text(
        json.dumps(
            {
                "conformal": meta["conformal"],
                "applicability_domain": meta["applicability_domain"],
            },
            indent=2,
        )
    )

    print(f"\nconformal 90% interval: +/-{q_hat:.2f} pIC50 (measured coverage {coverage:.0%})")
    print(f"applicability domain: Tanimoto >= {domain.threshold:.2f} ({fraction_in:.0%} in-domain)")
    print(f"updated {args.bundle}")


if __name__ == "__main__":
    main()
