import streamlit as st
import os, tempfile, zipfile, time
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import joblib
import py3Dmol

# -------------------- Import your feature extractor --------------------
try:
    import Features_RNALig as FR
except ImportError as e:
    FR = None
    _feature_import_error = str(e)
else:
    _feature_import_error = None

# -------------------- Page config + CSS --------------------
st.set_page_config(
    page_title="RNALig – RNA–Ligand Binding Affinity Pipeline",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        max-width: 1000px;
    }

    /* Center and space the tab bar */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        margin-top: 10px !important;
        margin-bottom: 20px !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
    }

    h1, h2, h3 {
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", sans-serif;
    }
    .hero-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        background: #EEF6FF;
        color: #1D4ED8;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.6rem;
        margin-bottom: 0.4rem;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: #555;
        max-width: 720px;
        margin: 0 auto 0.8rem auto;
    }
    .small-muted {
        font-size: 0.85rem;
        color: #777;
    }
    .hero-center {
        text-align: center;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Model loading --------------------
@st.cache_resource
def load_model_bundle() -> Tuple[Optional[object], Optional[List[str]]]:
    """
    Load RNALig_training_model.pkl.
    Expected:
      - {"model": <sklearn_estimator>, "features": [feat1, ...]}
      - or plain sklearn estimator.
    """
    try:
        with open("RNALig_training_model.pkl", "rb") as f:
            bundle = joblib.load(f)
    except FileNotFoundError:
        st.error("❌ Model file `RNALig_training_model.pkl` not found in this folder.")
        return None, None
    except Exception as e:
        st.error(f"❌ Failed to load model bundle: {e}")
        return None, None

    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"], bundle.get("features")
    return bundle, None

# -------------------- Args for Features_RNALig --------------------
def build_default_args(outdir: str):
    class Args:
        pass

    args = Args()
    args.outdir = outdir

    # ligand detection
    args.cutoff = 4.0
    args.min_heavy = 8
    args.require_carbon = True
    args.keep_ions = False

    # interaction metrics
    args.vdw_mode = "shell"
    args.vdw_legacy_cutoff = 4.0
    args.hbond_cutoff = 3.5
    args.hydroph_cutoff = 4.5

    # electrostatics
    args.elec_mode = "charged"
    args.elec_targets = "phosphate"
    args.elec_qthr = 0.2
    args.elec_dmin = 3.0
    args.elec_dmax = 10.0
    args.elec_include_negative = False

    # visualization flags
    args.viz_rna = False
    args.viz_ligand = False
    args.viz_complex = False
    args.pocket_cutoff = 5.0
    args.pocket_sasa = 0.05
    args.rna_label_topk = 5

    args.lig_viz_dir = None
    args.rna_viz_dir = None

    return args

# -------------------- Core pipeline functions --------------------
def run_feature_extraction(pdb_paths: List[str]):
    if FR is None or not hasattr(FR, "process_one_pdb"):
        raise RuntimeError(
            "Could not import Features_RNALig or missing process_one_pdb(). "
            "Check that Features_RNALig.py is in this folder and imports correctly."
        )

    outdir = tempfile.mkdtemp(prefix="rnalig_feat_")
    args = build_default_args(outdir)

    rows = []
    cleaned_map = {}

    for path in pdb_paths:
        base = os.path.basename(path)
        st.write(f"🔬 Processing: `{base}` ...")
        row = FR.process_one_pdb(path, args)
        rows.append(row)

        pdb_id = row.get("PDB_ID", os.path.splitext(base)[0])
        clean_name = os.path.splitext(base)[0] + "_clean.pdb"
        clean_path = os.path.join(outdir, clean_name)
        if os.path.exists(clean_path):
            cleaned_map[pdb_id] = clean_path

    df = pd.DataFrame(rows)
    if "PDB_ID" in df.columns:
        cols = ["PDB_ID"] + [c for c in df.columns if c != "PDB_ID"]
        df = df[cols]

    return df, cleaned_map


def predict_binding_affinity(df_features: pd.DataFrame):
    model, feat_names = load_model_bundle()
    if model is None:
        return None, None

    id_col = None
    for c in df_features.columns:
        if "pdb" in c.lower() or "id" in c.lower() or "name" in c.lower():
            id_col = c
            break

    numeric = df_features.select_dtypes(include=[np.number]).copy()

    if feat_names:
        for f in feat_names:
            if f not in numeric.columns:
                numeric[f] = np.nan
        X = numeric[feat_names].astype(float)
    else:
        X = numeric

    X = X.fillna(X.median())
    y_pred = model.predict(X)

    if id_col is not None:
        df_pred = pd.DataFrame(
            {"PDB_ID": df_features[id_col], "Predicted_binding_affinity_kcal_mol": y_pred}
        )
    else:
        df_pred = pd.DataFrame(
            {"Index": np.arange(len(df_features)), "Predicted_binding_affinity_kcal_mol": y_pred}
        )

    df_combined = df_features.copy()
    df_combined["Predicted_binding_affinity_kcal_mol"] = y_pred

    return df_pred, df_combined

# -------------------- 3D viewer helpers --------------------
def show_3d_structure(pdb_str: str, width: int = 450, height: int = 350, spin: bool = False):
    """Render a PDB string with py3Dmol."""
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_str, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.addStyle({"and": [{"resn": "LIG"}]}, {"stick": {"colorscheme": "cyanCarbon"}})
    view.zoomTo()
    if spin:
        view.spin(True)
    html = view._make_html()
    st.components.v1.html(html, height=height + 15)

def show_feature_panel(row: pd.Series, cleaned_path: Optional[str] = None):
    pdb_id = row.get("PDB_ID", "Unknown")
    pred = row.get("Predicted_binding_affinity_kcal_mol", None)

    st.markdown(f"### 🧾 {pdb_id}")
    if pred is not None:
        st.markdown(f"**Predicted binding affinity:** `{pred:.3f} kcal/mol`")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("**Feature values (all)**")
        df_single = row.to_frame(name="Value")
        st.dataframe(df_single, use_container_width=True)

        num_series = row.select_dtypes(include=[np.number])
        if len(num_series) > 0:
            st.markdown("**Numeric features (bar chart)**")
            st.bar_chart(num_series)

    with col_right:
        if cleaned_path is not None:
            try:
                with open(cleaned_path, "r") as f:
                    pdb_block = f.read()
                st.markdown("**Cleaned complex (3D view)**")
                show_3d_structure(pdb_block, width=320, height=260, spin=False)
            except Exception as e:
                st.warning(f"Could not render cleaned PDB: {e}")
        else:
            st.info("No cleaned PDB found to display.")

# -------------------- Demo helpers --------------------
def find_demo_pdbs() -> List[str]:
    """Return sorted list of demo*.pdb files."""
    demos = []
    for fname in os.listdir("."):
        if fname.lower().endswith(".pdb") and fname.lower().startswith("demo"):
            demos.append(fname)
    demos.sort()
    return demos

# -------------------- Home tab --------------------
def render_home():
    # Hero: centered logo + title
    logo_path = None
    for candidate in ["rnalig_logo.png", "RNALig_logo.png", "logo.png"]:
        if os.path.exists(candidate):
            logo_path = candidate
            break

    st.markdown('<div class="hero-center">', unsafe_allow_html=True)

    if logo_path:
        st.image(logo_path, width=150)

    st.markdown(
        '<div class="hero-badge">AI-driven scoring for RNA–ligand complexes</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-title">RNALig – RNA–Ligand Binding Affinity Pipeline</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Overview + demo side-by-side (like RSApred style)
    st.markdown("### Overview")
    col_text, col_demo = st.columns([2, 1.4])

    with col_text:
        st.write(
            "RNALig is an AI-driven scoring function that estimates RNA–ligand binding "
            "affinities directly from 3D complexes. It automatically cleans raw PDB/mmCIF "
            "files, standardises ligands, and detects the RNA binding pocket. A rich set "
            "of structural and physicochemical descriptors—including SASA, non-covalent "
            "contacts, hydrogen bonds, stacking interactions and electrostatics—is "
            "extracted for each complex. These features are fed into a trained Random "
            "Forest model to predict binding affinity in kcal/mol. The interface is "
            "designed as an end-to-end pipeline that exposes both the feature table and "
            "final scores for every structure, supporting interpretability, screening and "
            "method benchmarking."
        )
        st.markdown("")
        st.markdown(
            "👉 Use the **“Run RNALig”** tab to upload your own complexes and run the full pipeline."
        )

    with col_demo:
        st.subheader("Demo RNA–ligand movie", anchor=False)

        demo_files = find_demo_pdbs()
        if not demo_files:
            st.info(
                "Place one or more demo PDB files in this folder with names like "
                "`demo1.pdb`, `demo2.pdb`, ... to show an animated example here."
            )
        else:
            st.caption("The viewer will cycle through all demo complexes.")

            placeholder = st.empty()

            if st.button("▶ Play demo animation"):
                # Loop over all demo structures twice
                for _ in range(2):
                    for fname in demo_files:
                        try:
                            with open(fname, "r") as f:
                                pdb_block = f.read()
                        except Exception:
                            continue
                        with placeholder.container():
                            st.write(f"Showing: `{fname}`")
                            show_3d_structure(pdb_block, spin=True)
                        time.sleep(1.5)

            # Show first demo statically so viewer is not empty
            try:
                with open(demo_files[0], "r") as f:
                    pdb_block0 = f.read()
                with placeholder.container():
                    st.write(f"Showing: `{demo_files[0]}`")
                    show_3d_structure(pdb_block0, spin=True)
            except Exception:
                pass

    st.markdown(
        '<p class="small-muted">RNALig is intended for research use only. Predictions should be '
        'interpreted alongside structural inspection and experimental data.</p>',
        unsafe_allow_html=True,
    )

# -------------------- Run RNALig tab --------------------
def render_run_pipeline():
    st.header("Run RNALig pipeline")

    if FR is None:
        st.error(
            "Could not import `Features_RNALig`. Make sure `Features_RNALig.py` "
            "is in this folder and all its dependencies (rdkit, freesasa, RNA, etc.) "
            "are installed in your conda environment."
        )
        if _feature_import_error:
            with st.expander("Import error details"):
                st.code(_feature_import_error)
        return

    st.markdown(
        """
This tab performs the full **clean → feature extraction → prediction** workflow
for each RNA–ligand complex you upload.
        """
    )

    st.subheader("Input mode")

    mode = st.radio(
        "Choose how to load structures:",
        (
            "Option 1: Upload up to 5 PDB/mmCIF files",
            "Option 2: Upload a ZIP with many PDB/mmCIF files",
        ),
    )

    pdb_paths: List[str] = []

    if mode.startswith("Option 1"):
        uploads = st.file_uploader(
            "Upload PDB/mmCIF files",
            type=["pdb", "cif", "mmcif"],
            accept_multiple_files=True,
        )
        if uploads:
            if len(uploads) > 5:
                st.warning("You uploaded more than 5 files; only the first 5 will be processed.")
                uploads = uploads[:5]
            tmp_in = tempfile.mkdtemp(prefix="rnalig_in_")
            for up in uploads:
                out_path = os.path.join(tmp_in, up.name)
                with open(out_path, "wb") as f:
                    f.write(up.getbuffer())
                pdb_paths.append(out_path)

    else:  # ZIP mode
        zfile = st.file_uploader(
            "Upload a ZIP containing PDB/mmCIF files",
            type=["zip"],
        )
        if zfile is not None:
            tmp_in = tempfile.mkdtemp(prefix="rnalig_zip_")
            zip_path = os.path.join(tmp_in, "input.zip")
            with open(zip_path, "wb") as f:
                f.write(zfile.getbuffer())

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_in)

            for root, _, files in os.walk(tmp_in):
                for fn in files:
                    if fn.lower().endswith((".pdb", ".cif", ".mmcif")):
                        pdb_paths.append(os.path.join(root, fn))

            if pdb_paths:
                st.info(f"Found {len(pdb_paths)} structure file(s) inside the ZIP.")
            else:
                st.error("No .pdb/.cif/.mmcif files found in the ZIP.")

    if st.button("🚀 Run full pipeline (features + prediction)", type="primary"):
        if not pdb_paths:
            st.error("No structures to process. Please upload files or a ZIP first.")
            return

        with st.spinner("Running feature extraction for all structures..."):
            try:
                df_features, cleaned_map = run_feature_extraction(pdb_paths)
            except Exception as e:
                st.error(f"❌ Feature extraction failed: {e}")
                return

        st.success(f"✅ Extracted features for {len(df_features)} structure(s).")

        with st.spinner("Predicting binding affinities..."):
            df_pred, df_combined = predict_binding_affinity(df_features)
        if df_pred is None:
            st.error("❌ Prediction step failed due to model issues.")
            return

        st.subheader("Global summary")
        st.markdown("**All predictions**")
        st.dataframe(df_pred, use_container_width=True)

        st.markdown("#### 📥 Download results")
        st.download_button(
            "Download all features (CSV)",
            data=df_features.to_csv(index=False).encode("utf-8"),
            file_name="RNALig_features.csv",
        )
        st.download_button(
            "Download predictions only (CSV)",
            data=df_pred.to_csv(index=False).encode("utf-8"),
            file_name="RNALig_predictions_only.csv",
        )
        st.download_button(
            "Download features + predictions (CSV)",
            data=df_combined.to_csv(index=False).encode("utf-8"),
            file_name="RNALig_features_with_predictions.csv",
        )

        st.markdown("---")
        st.subheader("Per-complex feature & structure views")

        id_col = "PDB_ID" if "PDB_ID" in df_combined.columns else None

        for idx, row in df_combined.iterrows():
            if id_col:
                pdb_id = row[id_col]
                clean_path = cleaned_map.get(pdb_id)
                label = f"📁 {pdb_id}"
            else:
                pdb_id = f"row_{idx}"
                clean_path = None
                label = f"📁 Complex {idx}"

            with st.expander(label, expanded=False):
                show_feature_panel(row, cleaned_path=clean_path)

# -------------------- Docs tab --------------------
def render_docs():
    st.header("Quick usage guide")

    st.markdown(
        """
### 1. Prepare input structures

- RNA–ligand complexes in **PDB** or **mmCIF** format  
- Each file should contain at least one RNA chain and one bound small-molecule ligand.

### 2. Run the pipeline

1. Go to the **“Run RNALig”** tab  
2. Choose upload mode (individual files or ZIP)  
3. Click **“Run full pipeline (features + prediction)”**  
4. RNALig will:
   - Clean the complex  
   - Detect the ligand pocket  
   - Compute structural & physicochemical features  
   - Apply the trained Random Forest model  

### 3. Interpret the results

- **Global table**: overview of all complexes and predicted binding affinities  
- **Per-complex panels**:
  - Full feature vector (table)
  - Bar chart of numeric features
  - 3D view of cleaned complex (if available)

> RNALig is a research tool. Predictions should be interpreted together with
> structural inspection and experimental data where available.
        """
    )

# -------------------- Main --------------------
def main():
    tabs = st.tabs(["🏠 Home", "📊 Run RNALig", "📖 Docs"])

    with tabs[0]:
        render_home()

    with tabs[1]:
        render_run_pipeline()

    with tabs[2]:
        render_docs()


if __name__ == "__main__":
    main()
