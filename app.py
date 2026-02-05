import streamlit as st
import os, tempfile, zipfile, time
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import joblib
import py3Dmol
import requests

# -------------------- Import RNALig feature extractor --------------------
try:
    import Features_RNALig as FR
except ImportError as e:
    FR = None
    _feature_import_error = str(e)
else:
    _feature_import_error = None

# -------------------- Page config --------------------
st.set_page_config(
    page_title="RNALig – RNA–Ligand Binding Affinity Predictor",
    layout="wide",
)

# -------------------- Styling --------------------
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

.small-muted { font-size: 0.85rem; color: #6b7280; }

.footer-wrap {
    border-top: 1px solid #e5e7eb;
    padding-top: 1rem;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Model loading --------------------
@st.cache_resource
def load_model_bundle():
    try:
        bundle = joblib.load("RNALig_training_model.pkl")
    except Exception:
        st.error("RNALig model file not found.")
        return None, None

    if isinstance(bundle, dict):
        return bundle["model"], bundle.get("features")
    return bundle, None

# -------------------- Header --------------------
def render_header():
    st.markdown('<div class="header-wrap">', unsafe_allow_html=True)
    col1, col2 = st.columns([0.2, 0.8])

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

        st.markdown("""
        <div style="display:flex; gap:30px; margin-top:12px;">
          <div><b>Input</b><br>RNA–ligand 3D structures</div>
          <div><b>Method</b><br>Structure-based ML</div>
          <div><b>Output</b><br>Affinity (kcal/mol)</div>
          <div><b>Extras</b><br>Features + 3D view</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Home Page --------------------
def render_home():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("## Overview")

    st.write("""
RNALig is an AI-driven scoring function that estimates RNA–ligand binding
affinities directly from three-dimensional complexes. The tool automatically
cleans raw PDB/mmCIF structures, standardizes ligands, detects RNA binding
pockets, and extracts a comprehensive set of structural and physicochemical
features. These descriptors are used by a trained Random Forest regression
model to predict binding affinity in kcal/mol.
""")

    st.markdown("### Model and Training Data")
    st.write("""
The RNALig model was trained on experimentally determined RNA–ligand binding
affinities curated from public databases and literature sources. Structure-based
features derived from cleaned RNA–ligand complexes were used for training and
evaluation via cross-validation and independent testing.
""")

    st.markdown("### Limitations")
    st.write("""
RNALig assumes a pre-formed RNA–ligand complex and does not perform
conformational sampling. Predictions may be less reliable for highly flexible
RNAs, metal-ion mediated binding, or ligands outside the training chemical
space.
""")

    if st.button("▶ Try RNALig now (Run Predictions)", type="primary"):
        st.session_state.page = "Run Predictions"

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
<p class="small-muted">
RNALig is intended for research use only. Predictions should be interpreted
alongside experimental and structural evidence.
</p>
""", unsafe_allow_html=True)

# -------------------- FAQ --------------------
def render_faq():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("## Frequently Asked Questions")

    with st.expander("What input formats are supported?"):
        st.write("PDB and mmCIF RNA–ligand complexes.")

    with st.expander("Does RNALig perform docking?"):
        st.write("No. RNALig scores pre-formed complexes only.")

    with st.expander("What does a more negative affinity mean?"):
        st.write("More negative values indicate stronger predicted binding.")

    with st.expander("Can I use RNALig for virtual screening?"):
        st.write("Yes, via batch upload or ZIP processing.")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Footer --------------------
def render_footer():
    st.markdown('<div class="footer-wrap">', unsafe_allow_html=True)
    col1, col2 = st.columns([0.2, 0.8])

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

# -------------------- Main --------------------
def main():
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Run Predictions", "Tutorial", "FAQ"],
        index=["Home", "Run Predictions", "Tutorial", "FAQ"].index(st.session_state.page)
    )

    render_header()

    if page == "Home":
        render_home()
    elif page == "FAQ":
        render_faq()
    else:
        st.info("Pipeline code unchanged – keep your existing Run Predictions & Tutorial sections here.")

    render_footer()

if __name__ == "__main__":
    main()
