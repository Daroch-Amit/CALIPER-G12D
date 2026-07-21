"""Streamlit front end for the KRAS G12D potency predictor."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from src.predict import Predictor

# Single artefact holding the featuriser, preprocessor, model, conformal
# calibration and applicability domain.
BUNDLE = "models/model_bundle_ecfp.joblib"

st.set_page_config(
    page_title="KRAS G12D Potency Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .main .block-container {max-width: 1080px; padding-top: 1.4rem;}
      .card {background:#fff; border:1px solid #e6e8eb; border-radius:14px;
             padding:1.05rem 1.25rem; box-shadow:0 1px 3px rgba(0,0,0,.05);}
      .big {font-size:2rem; font-weight:700; color:#0f172a; line-height:1.1;}
      .lbl {font-size:.78rem; color:#64748b; text-transform:uppercase; letter-spacing:.04em;}
      .pill {display:inline-block; padding:.28rem .8rem; border-radius:999px;
             font-weight:600; font-size:.85rem;}
      .green {background:#dcfce7; color:#166534;}
      .blue  {background:#dbeafe; color:#1e40af;}
      .amber {background:#fef3c7; color:#92400e;}
      .red   {background:#fee2e2; color:#991b1b;}
    </style>
    """,
    unsafe_allow_html=True,
)


# cache_resource keeps the bundle in memory across reruns; without it every
# widget interaction would reload roughly 20 MB from disk.
@st.cache_resource(show_spinner="Loading prediction engine...")
def load_predictor() -> Predictor:
    return Predictor(BUNDLE)


def potency_band(ic50_nm: float) -> tuple[str, str]:
    """Rough medicinal-chemistry bands, used only for the coloured label."""
    if ic50_nm < 10:
        return "Highly potent", "green"
    if ic50_nm < 100:
        return "Potent", "blue"
    if ic50_nm < 1000:
        return "Moderate", "amber"
    return "Weak", "red"


def draw_molecule(smiles: str, size: int = 320):
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=(size, size)) if mol else None


st.sidebar.title("🧬 KRAS G12D")
st.sidebar.caption("Potency Predictor")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Single Prediction", "Batch Prediction", "Visualizations"],
    label_visibility="collapsed",
)

engine_ready = True
try:
    predictor = load_predictor()
except Exception:
    engine_ready = False
    st.sidebar.error("Model bundle not found. Train a model first.")

if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = None


if page == "Home":
    st.title("KRAS G12D Potency Predictor")
    st.markdown(
        "Estimate the inhibitory potency of a small molecule against the "
        "**KRAS G12D** oncogenic mutant. Enter a structure and the tool returns "
        "a predicted **pIC50** and **IC50 (nM)** with a confidence range, and "
        "checks whether the molecule resembles the chemistry behind the model."
    )
    columns = st.columns(3)
    blurbs = [
        ("1 - Enter a molecule", "Paste a SMILES on *Single Prediction*, or upload a CSV on *Batch Prediction*."),
        ("2 - Read the potency", "Get pIC50, IC50 in nM, a confidence range and a potency band."),
        ("3 - Check the domain", "An in-domain flag shows whether the estimate is well supported."),
    ]
    for column, (title, body) in zip(columns, blurbs):
        with column:
            st.markdown(
                f"<div class='card'><div class='lbl'>{title}</div>"
                f"<div style='margin-top:.4rem;color:#334155'>{body}</div></div>",
                unsafe_allow_html=True,
            )
    st.info("For research use only. Computational estimates require experimental validation.")


elif page == "Single Prediction":
    st.title("Single Molecule Prediction")
    example = "C=CC(=O)N1CCN(c2nc(=O)n(-c3scnc3C(C)C)c3nc(-c4c(O)cccc4F)c(Cl)cc23)C(C)C1"

    smiles = st.text_input("Enter a SMILES string", placeholder=f"e.g. {example}")
    columns = st.columns([1, 1, 4])
    predict = columns[0].button("Predict", type="primary", use_container_width=True)
    if columns[1].button("Example", use_container_width=True):
        smiles, predict = example, True

    if predict and engine_ready:
        if not smiles.strip():
            st.warning("Please enter a SMILES string.")
        else:
            result = predictor.predict_one(smiles.strip())
            if not result.valid:
                st.error(f"Could not parse this structure. {result.message}")
            else:
                left, right = st.columns([1, 1.3])
                with left:
                    image = draw_molecule(result.canonical_smiles)
                    if image is not None:
                        st.image(image, caption="Standardised structure")
                with right:
                    label, css = potency_band(result.ic50_nm)
                    st.markdown(f"<span class='pill {css}'>{label}</span>", unsafe_allow_html=True)

                    metric_left, metric_right = st.columns(2)
                    ic50_range = (
                        f"range {result.ic50_nm_low:,.1f} - {result.ic50_nm_high:,.1f} nM"
                        if result.ic50_nm_low is not None
                        else ""
                    )
                    metric_left.markdown(
                        f"<div class='card'><div class='lbl'>Predicted IC50</div>"
                        f"<div class='big'>{result.ic50_nm:,.1f} nM</div>"
                        f"<div style='color:#64748b'>{ic50_range}</div></div>",
                        unsafe_allow_html=True,
                    )
                    pic50_range = (
                        f"90% interval {result.pic50_low:.2f} - {result.pic50_high:.2f}"
                        if result.pic50_low is not None
                        else ""
                    )
                    metric_right.markdown(
                        f"<div class='card'><div class='lbl'>Predicted pIC50</div>"
                        f"<div class='big'>{result.pic50:.2f}</div>"
                        f"<div style='color:#64748b'>{pic50_range}</div></div>",
                        unsafe_allow_html=True,
                    )

                    if result.in_domain is not None:
                        if result.in_domain:
                            st.success(
                                f"In applicability domain "
                                f"(similarity to known actives: {result.similarity:.2f})"
                            )
                        else:
                            st.warning(
                                f"Outside applicability domain (similarity "
                                f"{result.similarity:.2f}). Treat this estimate with caution."
                            )


elif page == "Batch Prediction":
    st.title("Batch Prediction")
    st.markdown(
        "Upload a CSV containing a column of SMILES. The column is detected "
        "automatically where possible."
    )
    upload = st.file_uploader("Upload CSV", type=["csv"])

    if upload is not None and engine_ready:
        frame = pd.read_csv(upload)
        # Most exports name the column something containing "smiles"; fall back
        # to asking the user when that guess fails.
        smiles_column = next((c for c in frame.columns if "smiles" in c.lower()), None)
        if smiles_column is None:
            smiles_column = st.selectbox("Select the SMILES column", frame.columns)
        st.caption(f"Using column **{smiles_column}** - {len(frame)} rows")

        if st.button("Run predictions", type="primary"):
            with st.spinner("Scoring molecules..."):
                predictions = predictor.predict_batch(
                    frame[smiles_column].astype(str).tolist()
                )
                results = pd.DataFrame(
                    [
                        {
                            "input_smiles": p.smiles_input,
                            "valid": p.valid,
                            "pIC50": round(p.pic50, 3) if p.pic50 is not None else None,
                            "pIC50_low": round(p.pic50_low, 3) if p.pic50_low is not None else None,
                            "pIC50_high": round(p.pic50_high, 3) if p.pic50_high is not None else None,
                            "IC50_nM": round(p.ic50_nm, 2) if p.ic50_nm is not None else None,
                            "IC50_nM_low": round(p.ic50_nm_low, 2) if p.ic50_nm_low is not None else None,
                            "IC50_nM_high": round(p.ic50_nm_high, 2) if p.ic50_nm_high is not None else None,
                            "in_domain": p.in_domain,
                            "similarity": round(p.similarity, 3) if p.similarity is not None else None,
                        }
                        for p in predictions
                    ]
                )
                st.session_state["batch_results"] = results

            n_valid = int(results["valid"].sum())
            st.success(f"Scored {n_valid} molecules ({len(results) - n_valid} could not be parsed).")
            st.dataframe(results, use_container_width=True, height=380)

            buffer = io.StringIO()
            results.to_csv(buffer, index=False)
            st.download_button(
                "Download results (CSV)",
                buffer.getvalue(),
                file_name="kras_g12d_predictions.csv",
                mime="text/csv",
            )


elif page == "Visualizations":
    st.title("Prediction Visualizations")
    results = st.session_state.get("batch_results")

    if results is None or results.empty:
        st.info("Run a batch prediction first to see distributions here.")
    else:
        valid = results[results["valid"]].copy()
        left, middle, right = st.columns(3)
        left.metric("Molecules scored", len(valid))
        middle.metric("Median IC50 (nM)", f"{valid['IC50_nM'].median():,.1f}")
        if valid["in_domain"].notna().any():
            right.metric("In applicability domain", f"{valid['in_domain'].mean() * 100:.0f}%")

        st.subheader("Predicted pIC50 distribution")
        st.bar_chart(valid["pIC50"].round(1).value_counts().sort_index())

        st.subheader("Potency breakdown")
        bands = valid["IC50_nM"].apply(lambda v: potency_band(v)[0])
        st.bar_chart(bands.value_counts())

        st.subheader("Most potent candidates")
        st.dataframe(
            valid.nsmallest(10, "IC50_nM")[
                ["input_smiles", "pIC50", "IC50_nM", "in_domain"]
            ],
            use_container_width=True,
        )

st.sidebar.markdown("---")
st.sidebar.caption("For research use only. Estimates require experimental validation.")
