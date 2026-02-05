import streamlit as st
import os, tempfile, zipfile, time
from typing import List, Optional

import pandas as pd
import numpy as np
import joblib
import py3Dmol
import requests

# ==================== Import Feature Extractor ====================
try:
    import Features_RNALig as FR
except ImportError as e:
    FR = None
    FEATURE_ERR = str(e)
else:
    FEATURE_ERR = None

# ==================== Page Config ====================
st.set_page_config(
    page_title="RNALig – RNA–Ligand Binding Affinity Predictor",
    layout="wide",
)

# ==================== Styling ====================
st.markdown("""
<style>
.main { background-color: #f4f6fb; }
.block-container { max-width: 95% !important; padding: 2rem; }

.header-wrap {
    background: white;
    padding: 12px 25px;
    border-radius: 0 0 18px 18px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

.header-title { font-size: 30px; font-weight: 800; }
.header-subtitle { color: #6b7280; font-size: 14px; }

.content-card {
    background: white;
    padding: 2.2rem;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
}

.movie-card {
    background: #f9fafb;
    border-radius: 16px;
    padding: 0.6rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
}

.small-muted { font-size: 0.85rem; color: #6b7280; }

.footer-wrap {
    border-top: 1px solid #e5e7eb;
    padding-top: 1rem;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ==================== Load Model ====================
@st.cache_resource
def load_model():
    try:
        bundle = joblib.load("RNALig_training_model.pkl")
    except Exception:
        st.error("RNALig model file not found.")
        return None, None

    if isinstance(bundle, dict):
        return bundle["model"], bundle.get("features")
    return bundle, None

# ==================== Header ====================
def render_header():
    st.markdown('<div class="header-wrap">', unsafe_allow_html=True)
    col1, col2 = st.columns([0.18, 0.82])

    with col1:
        for f in ["RNALig_logo.png", "logo.png"]:
            if os.path.exists(f):
                st.image(f, width=150)
                break

    with col2:
        st.markdown("""
        <div class="header-title">RNALig – RNA–Ligand Binding Affinity Predictor</div>
        <div class="header-subtitle">
        AI-driven scoring and interpretability for RNA–ligand complexes
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ==================== 3D Viewer ====================
def show_3d_structure(pdb_str, spin=False):
    view = py3Dmol.view(width=350, height=260)
    view.addModel(pdb_str, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    if spin:
        view.spin(True)
    st.components.v1.html(view._make_html(), height=280)

# ==================== Demo Finder ====================
def find_demo_pdbs():
    return sorted([f for f in os.listdir(".") if f.lower().startswith("demo") and f.endswith(".pdb")])

# ==================== Home Page ====================
def render_home_content():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)

    col_text, col_demo = st.columns([2, 1.4])

    with col_text:
        st.markdown("## Overview")
        st.write("""
RNALig is an AI-driven scoring function that estimates RNA–ligand binding
affinities directly from three-dimensional complexes. The tool automatically
cleans raw PDB/mmCIF files, standardizes ligands, detects RNA binding pockets,
and extracts a comprehensive set of structural and physicochemical features.
These descriptors are used by a trained Random Forest regression model to
predict binding affinity in kcal/mol.
""")

        st.markdown(
            "Use the **Run Predictions** page to upload or fetch your own complexes "
            "and execute the full prediction pipeline."
        )

    with col_demo:
        demo_files = find_demo_pdbs()
        if demo_files:
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            placeholder = st.empty()
            for fname in demo_files:
                with open(fname) as f:
                    pdb = f.read()
                with placeholder.container():
                    show_3d_structure(pdb, spin=True)
                time.sleep(1.0)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
<p class="small-muted">
RNALig is intended for research use only. Predictions should be interpreted
alongside structural inspection and experimental data.
</p>
""", unsafe_allow_html=True)

# ==================== Run Pipeline ====================
def render_run_pipeline():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.header("Run Predictions")

    if FR is None:
        st.error("Features_RNALig could not be imported.")
        st.code(FEATURE_ERR)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown("Upload or fetch RNA–ligand complexes and run the full RNALig pipeline.")

    uploads = st.file_uploader(
        "Upload PDB/mmCIF files",
        type=["pdb", "cif", "mmcif"],
        accept_multiple_files=True,
    )

    pdb_paths = []
    if uploads:
        tmp = tempfile.mkdtemp(prefix="rnalig_")
        for up in uploads:
            path = os.path.join(tmp, up.name)
            with open(path, "wb") as f:
                f.write(up.getbuffer())
            pdb_paths.append(path)

    if st.button("Run full pipeline", type="primary"):
        if not pdb_paths:
            st.error("No input structures provided.")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        rows = []
        for p in pdb_paths:
            rows.append(FR.process_one_pdb(p, FR.default_args()))

        df = pd.DataFrame(rows)
        model, feats = load_model()
        X = df.select_dtypes(include=[np.number]).fillna(0)
        y = model.predict(X)
        df["Predicted_binding_affinity_kcal_mol"] = np.round(y, 3)

        st.subheader("Results")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download results (CSV)",
            df.to_csv(index=False).encode(),
            "RNALig_results.csv",
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ==================== Tutorial ====================
def render_tutorial():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("""
### Quick Start

1. Go to **Run Predictions**
2. Upload RNA–ligand PDB/mmCIF files
3. Click **Run full pipeline**
4. Inspect predicted affinities and features
5. Download results
""")
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== FAQ ====================
def render_faq():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### Frequently Asked Questions")

    with st.expander("Does RNALig perform docking?"):
        st.write("No. RNALig scores pre-formed RNA–ligand complexes only.")

    with st.expander("What does a more negative affinity mean?"):
        st.write("More negative values indicate stronger predicted binding.")

    st.markdown('</div>', unsafe_allow_html=True)

# ==================== Footer ====================
def render_footer():
    st.markdown('<div class="footer-wrap">', unsafe_allow_html=True)
    col1, col2 = st.columns([0.18, 0.82])

    with col1:
        if os.path.exists("Lab_Logo.png"):
            st.image("Lab_Logo.png", width=90)

    with col2:
        st.markdown("""
**Computational BioLab**  
Email: computationalbiolab@gmail.com  

**Citation:**  
Sharma P. *et al.* RNALig: An AI-driven RNA–ligand binding affinity predictor.  
*Nucleic Acids Research*, 2026.  

**Version:** RNALig v1.0 | **Last updated:** Feb 2026
""")

    st.markdown('</div>', unsafe_allow_html=True)

# ==================== Main ====================
def main():
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Run Predictions", "Tutorial", "FAQ"],
        index=0,
    )

    render_header()

    if page == "Home":
        render_home_content()
    elif page == "Run Predictions":
        render_run_pipeline()
    elif page == "Tutorial":
        render_tutorial()
    else:
        render_faq()

    render_footer()

if __name__ == "__main__":
    main()
