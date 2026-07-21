# KRAS G12D IC50 Predictor

Machine learning pipeline that predicts the inhibitory potency (pIC50, reported
also as IC50 in nM) of small molecules against the KRAS G12D mutant, with
conformal prediction intervals, a Tanimoto applicability domain, and a Streamlit
front end.

## Layout

```
.
├── data/                  input CSV (SMILES, pIC50)
├── artifacts/             features and intermediate files (generated)
├── models/                trained bundle and metrics (generated)
├── src/
│   ├── featurization.py   SMILES -> features, plus structure standardisation
│   ├── preprocessing.py   filtering, scaling, consensus feature selection
│   ├── train.py           scaffold splitting, model comparison, tuning, export
│   ├── uncertainty.py     conformal intervals + applicability domain
│   └── predict.py         inference wrapper used by the app
├── run_featurization.py
├── run_train.py
├── run_uncertainty.py
├── app.py                 Streamlit interface
└── requirements.txt
```

The package is imported as `src.*` and that name is load-bearing: saved bundles
store fully qualified class paths, so renaming the package invalidates every
model trained before the rename. Always run the scripts from the project root.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.10-3.13.

## Usage

```bash
# 1. featurise the dataset
python run_featurization.py --input data/GTPaseKRAS_G12D_pIC50.csv \
    --representation ecfp --outdir artifacts

# 2. train, tune and export the bundle
python run_train.py --representation ecfp --folds 5 --tune 40
#    retrain a known family without the full model sweep:
python run_train.py --representation ecfp --model xgboost --no-compare --tune 40

# 3. calibrate intervals and fit the applicability domain (edits the bundle)
python run_uncertainty.py --bundle models/model_bundle_ecfp.joblib \
    --artifacts artifacts --alpha 0.1 --ad-percentile 5

# 4. launch the app
streamlit run app.py
```

Available representations: `ecfp`, `fcfp`, `rdkit2d`, `maccs`, `atompair`,
`topotorsion`, `avalon`, `mordred2d`, `mordred3d`, `padel`, and any `+`-joined
combination such as `rdkit2d+ecfp`. ECFP is the default because it gave the
lowest cross-validated error on this dataset; Mordred is slow on a single core
and PaDEL needs a working Java install.

## Programmatic use

```python
from src.predict import Predictor

predictor = Predictor("models/model_bundle_ecfp.joblib")
result = predictor.predict_one("C=CC(=O)N1CCN(c2nc(=O)n(-c3scnc3C(C)C)c3nc(-c4c(O)cccc4F)c(Cl)cc23)C(C)C1")
print(result.pic50, result.ic50_nm, result.in_domain, result.similarity)
```

## Notes on the evaluation

Train/test splitting and cross-validation are scaffold-grouped throughout, so no
Bemis-Murcko scaffold appears on both sides of a split. Random splits scatter
close analogues across the partition and give optimistic numbers on congeneric
series like this one. Supervised feature selection is fit inside each fold; the
hold-out set is not touched during model selection or tuning.

Accuracy on this data is bounded by the labels themselves. Near-identical
molecules (Tanimoto >= 0.95) in the public sources disagree by a median of about
0.43 pIC50, roughly 2.7-fold in IC50, so predictions should be read as a ranking
aid rather than a substitute for assay values, and the conformal interval and
in-domain flag are there to make that explicit.

## Licence

MIT.

For research use only. Computational estimates require experimental validation.
