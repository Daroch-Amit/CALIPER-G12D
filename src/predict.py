"""Inference wrapper around the trained bundle.

Used by the Streamlit app, but works fine from a plain script or notebook too.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# These imports look unused, but they register the classes that joblib needs to
# resolve when unpickling the bundle. Do not remove them.
from .featurization import FeatureGenerator, standardize_smiles  # noqa: F401
from .preprocessing import FeaturePreprocessor  # noqa: F401

try:
    from .uncertainty import ApplicabilityDomain, ConformalRegressor  # noqa: F401
except ImportError:
    ApplicabilityDomain = ConformalRegressor = None

DEFAULT_BUNDLE = "models/model_bundle_ecfp.joblib"


def pic50_to_ic50_nm(pic50: float) -> float:
    """IC50 in nanomolar from pIC50."""
    return float(np.power(10.0, 9.0 - pic50))


@dataclass
class Prediction:
    smiles_input: str
    valid: bool
    canonical_smiles: str | None = None
    pic50: float | None = None
    pic50_low: float | None = None
    pic50_high: float | None = None
    ic50_nm: float | None = None
    ic50_nm_low: float | None = None
    ic50_nm_high: float | None = None
    in_domain: bool | None = None
    similarity: float | None = None
    message: str = "ok"

    def as_dict(self) -> dict:
        return asdict(self)


class Predictor:
    """Loads the bundle once and scores molecules on demand."""

    def __init__(self, bundle_path: str | Path = DEFAULT_BUNDLE) -> None:
        import joblib

        path = Path(bundle_path)
        if not path.exists():
            raise FileNotFoundError(f"no bundle at {path}; train a model first")

        bundle = joblib.load(path)
        self.generator = bundle["feature_generator"]
        self.preprocessor = bundle["preprocessor"]
        self.model = bundle["model"]
        self.conformal = bundle.get("conformal")
        self.domain = bundle.get("applicability_domain")
        self.meta = bundle.get("meta", {})

    def _predict_pic50(self, canonical: list[str]) -> np.ndarray:
        X, _ = self.generator.transform(canonical, standardize=False)
        return self.model.predict(self.preprocessor.transform(X))

    def _assemble(self, raw: str, canonical: str, pic50: float) -> Prediction:
        low = high = None
        if self.conformal is not None:
            low_arr, high_arr = self.conformal.interval(np.array([pic50]))
            low, high = float(low_arr[0]), float(high_arr[0])

        in_domain = similarity = None
        if self.domain is not None:
            flag, sim = self.domain.in_domain(canonical)
            in_domain, similarity = bool(flag), float(sim)

        # Potency and concentration run in opposite directions, so the lower
        # pIC50 bound corresponds to the higher IC50 bound.
        return Prediction(
            smiles_input=raw,
            valid=True,
            canonical_smiles=canonical,
            pic50=pic50,
            pic50_low=low,
            pic50_high=high,
            ic50_nm=pic50_to_ic50_nm(pic50),
            ic50_nm_low=pic50_to_ic50_nm(high) if high is not None else None,
            ic50_nm_high=pic50_to_ic50_nm(low) if low is not None else None,
            in_domain=in_domain,
            similarity=similarity,
        )

    def predict_one(self, smiles: str) -> Prediction:
        canonical = standardize_smiles(smiles)
        if canonical is None:
            return Prediction(smiles, valid=False, message="could not parse this SMILES")
        pic50 = float(self._predict_pic50([canonical])[0])
        return self._assemble(smiles, canonical, pic50)

    def predict_batch(self, smiles_list: list[str]) -> list[Prediction]:
        canonical = [standardize_smiles(s) for s in smiles_list]
        results: list[Prediction | None] = [None] * len(smiles_list)

        valid_positions = [i for i, c in enumerate(canonical) if c is not None]
        for i, c in enumerate(canonical):
            if c is None:
                results[i] = Prediction(
                    smiles_list[i], valid=False, message="could not parse this SMILES"
                )

        if valid_positions:
            predictions = self._predict_pic50([canonical[i] for i in valid_positions])
            for position, index in enumerate(valid_positions):
                results[index] = self._assemble(
                    smiles_list[index], canonical[index], float(predictions[position])
                )

        return [r for r in results if r is not None]
