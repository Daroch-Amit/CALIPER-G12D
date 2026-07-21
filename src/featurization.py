"""Molecular featurisation for the KRAS G12D dataset.

Turns SMILES into numeric feature matrices. One picklable FeatureGenerator does
the work so that training and inference always use the identical encoding.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator as rfg
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kras.featurization")

SEED = 42

# Grab the descriptor list once at import; Descriptors._descList is a fairly
# expensive attribute to touch inside a loop over thousands of molecules.
RDKIT_DESCRIPTORS = list(Descriptors._descList)

# Trailing CXSMILES block, e.g. "...c12 |r|". RDKit usually parses these, but
# a fair number of BindingDB records need the block stripped before they will.
CX_SUFFIX = re.compile(r"\s*\|[^|]*\|\s*$")

_largest_fragment = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def standardize_smiles(smiles: str, remove_salts: bool = True) -> str | None:
    """Canonicalise a SMILES string, or return None if RDKit cannot parse it."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(CX_SUFFIX.sub("", smiles))
    if mol is None:
        return None

    try:
        if remove_salts:
            mol = _largest_fragment.choose(mol)
        mol = _uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True) or None
    except Exception:
        # Sanitisation can fail on a handful of pathological structures.
        # Treat those as unparseable rather than letting them poison a run.
        return None


def embed_3d(mol: Chem.Mol, seed: int = SEED) -> Chem.Mol | None:
    """Generate and minimise a single conformer (only needed for 3D descriptors)."""
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass  # unminimised conformer is still usable
    return mol


def fp_to_array(fp, n_bits: int) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


class FeatureGenerator:
    """SMILES -> feature matrix, for one representation or a '+'-joined combination.

    Supported blocks: rdkit2d, morgan/ecfp, fcfp, maccs, atompair, topotorsion,
    avalon, mordred2d, mordred3d, padel.
    """

    def __init__(
        self,
        representation: str = "ecfp",
        n_bits: int = 2048,
        radius: int = 2,
        remove_salts: bool = True,
    ) -> None:
        self.representation = representation.lower()
        self.n_bits = n_bits
        self.radius = radius
        self.remove_salts = remove_salts
        self.feature_names_: list[str] | None = None
        self._mordred_calc = None

    # --- individual blocks ------------------------------------------------

    def _rdkit2d(self, mols):
        rows = []
        for mol in mols:
            if mol is None:
                rows.append([np.nan] * len(RDKIT_DESCRIPTORS))
                continue
            values = []
            for _, fn in RDKIT_DESCRIPTORS:
                try:
                    values.append(float(fn(mol)))
                except Exception:
                    values.append(np.nan)
            rows.append(values)
        names = [f"RDKit_{name}" for name, _ in RDKIT_DESCRIPTORS]
        return np.asarray(rows, dtype=float), names

    def _morgan(self, mols, feature_based: bool):
        invariants = rfg.GetMorganFeatureAtomInvGen() if feature_based else None
        gen = rfg.GetMorganGenerator(
            radius=self.radius,
            fpSize=self.n_bits,
            includeChirality=True,
            atomInvariantsGenerator=invariants,
        )
        prefix = "FCFP" if feature_based else "ECFP"
        rows = [
            fp_to_array(gen.GetFingerprint(m), self.n_bits)
            if m is not None
            else np.zeros(self.n_bits, dtype=np.int8)
            for m in mols
        ]
        return np.asarray(rows), [f"{prefix}_{i}" for i in range(self.n_bits)]

    def _maccs(self, mols):
        rows = [
            fp_to_array(MACCSkeys.GenMACCSKeys(m), 167)
            if m is not None
            else np.zeros(167, dtype=np.int8)
            for m in mols
        ]
        return np.asarray(rows), [f"MACCS_{i}" for i in range(167)]

    def _hashed(self, mols, kind: str):
        if kind == "atompair":
            gen = rfg.GetAtomPairGenerator(fpSize=self.n_bits)
            prefix = "AP"
        else:
            gen = rfg.GetTopologicalTorsionGenerator(fpSize=self.n_bits)
            prefix = "TT"
        rows = [
            fp_to_array(gen.GetFingerprint(m), self.n_bits)
            if m is not None
            else np.zeros(self.n_bits, dtype=np.int8)
            for m in mols
        ]
        return np.asarray(rows), [f"{prefix}_{i}" for i in range(self.n_bits)]

    def _avalon(self, mols, n_bits: int = 1024):
        rows = [
            fp_to_array(pyAvalonTools.GetAvalonFP(m, nBits=n_bits), n_bits)
            if m is not None
            else np.zeros(n_bits, dtype=np.int8)
            for m in mols
        ]
        return np.asarray(rows), [f"AVALON_{i}" for i in range(n_bits)]

    def _mordred(self, mols, use_3d: bool):
        from mordred import Calculator, descriptors

        if self._mordred_calc is None:
            self._mordred_calc = Calculator(descriptors, ignore_3D=not use_3d)

        prepared = []
        for mol in mols:
            if mol is None:
                prepared.append(Chem.MolFromSmiles("C"))  # placeholder, dropped later
            else:
                prepared.append(embed_3d(mol) if use_3d else mol)
        prepared = [m if m is not None else Chem.MolFromSmiles("C") for m in prepared]

        frame = self._mordred_calc.pandas(prepared, quiet=True, nproc=1)
        # Mordred returns error objects for descriptors it could not compute.
        values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        tag = "Mordred3D" if use_3d else "Mordred2D"
        return values, [f"{tag}_{c}" for c in frame.columns]

    def _padel(self, smiles_list):
        from padelpy import from_smiles  # needs a working Java install

        rows, names = [], None
        for smi in smiles_list:
            try:
                desc = from_smiles(smi, fingerprints=False, descriptors=True, timeout=60)
                if names is None:
                    names = list(desc.keys())
                rows.append([float(desc.get(k) or "nan") for k in names])
            except Exception:
                rows.append([np.nan] * (len(names) if names else 0))
        return np.asarray(rows, dtype=float), [f"PaDEL_{n}" for n in (names or [])]

    def _block(self, name: str, mols, smiles):
        if name in ("morgan", "ecfp"):
            return self._morgan(mols, feature_based=False)
        if name == "fcfp":
            return self._morgan(mols, feature_based=True)
        if name == "rdkit2d":
            return self._rdkit2d(mols)
        if name == "maccs":
            return self._maccs(mols)
        if name in ("atompair", "topotorsion"):
            return self._hashed(mols, name)
        if name == "avalon":
            return self._avalon(mols)
        if name == "mordred2d":
            return self._mordred(mols, use_3d=False)
        if name == "mordred3d":
            return self._mordred(mols, use_3d=True)
        if name == "padel":
            return self._padel(smiles)
        raise ValueError(f"unknown representation block: {name!r}")

    # --- public API -------------------------------------------------------

    def transform(
        self, smiles: Iterable[str], standardize: bool = True
    ) -> tuple[np.ndarray, list[str]]:
        """Featurise SMILES and return (X, feature_names)."""
        smiles = list(smiles)
        canonical = [
            standardize_smiles(s, self.remove_salts) if standardize else s for s in smiles
        ]
        mols = [Chem.MolFromSmiles(c) if c else None for c in canonical]

        blocks, names = [], []
        for part in self.representation.split("+"):
            values, block_names = self._block(part.strip(), mols, canonical)
            blocks.append(values)
            names.extend(block_names)

        X = np.hstack(blocks) if len(blocks) > 1 else blocks[0]
        self.feature_names_ = names
        return X, names

    def transform_df(self, smiles: Iterable[str], standardize: bool = True) -> pd.DataFrame:
        X, names = self.transform(smiles, standardize)
        return pd.DataFrame(X, columns=names)

    def __getstate__(self):
        # The Mordred calculator is heavy and not reliably picklable; rebuild it
        # lazily after loading instead of dragging it into the saved bundle.
        state = self.__dict__.copy()
        state["_mordred_calc"] = None
        return state


def main() -> None:
    import joblib

    parser = argparse.ArgumentParser(description="Compute molecular features")
    parser.add_argument("--input", required=True, help="CSV with SMILES and pIC50 columns")
    parser.add_argument("--representation", default="ecfp")
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--target-col", default="pIC50")
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--outdir", default="artifacts")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    log.info("loaded %d rows from %s", len(df), args.input)

    df["_canonical"] = df[args.smiles_col].apply(standardize_smiles)
    n_dropped = int(df["_canonical"].isna().sum())
    df = df.dropna(subset=["_canonical"]).reset_index(drop=True)
    log.info("standardised structures, dropped %d, kept %d", n_dropped, len(df))

    generator = FeatureGenerator(args.representation, args.n_bits, args.radius)
    log.info("computing '%s' features", args.representation)
    X, names = generator.transform(df["_canonical"], standardize=False)
    log.info("feature matrix: %d molecules x %d features", *X.shape)

    # Mordred occasionally emits values above the float32 ceiling (~3.4e38).
    # Clip finite values and send inf/NaN to NaN so the cast below cannot
    # silently manufacture infinities that break the downstream imputer.
    X = np.asarray(X, dtype=np.float64)
    X[~np.isfinite(X)] = np.nan
    finite = np.isfinite(X)
    X[finite] = np.clip(X[finite], -3.0e38, 3.0e38)

    tag = args.representation.replace("+", "_")
    np.savez_compressed(outdir / f"features_{tag}.npz", X=X.astype(np.float32))
    (outdir / f"feature_names_{tag}.json").write_text(json.dumps(names))
    df[[args.smiles_col, "_canonical", args.target_col]].rename(
        columns={"_canonical": "canonical_smiles"}
    ).to_csv(outdir / f"aligned_{tag}.csv", index=False)
    joblib.dump(generator, outdir / f"feature_generator_{tag}.joblib", compress=3)

    log.info("wrote features, names, aligned targets and generator to %s/", outdir)


if __name__ == "__main__":
    main()
