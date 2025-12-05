# app.py

import os
import time
import tempfile
import zipfile
from typing import List, Tuple, Optional

import requests
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import py3Dmol

# ============================================================
#  Try to import your feature extractor
# ============================================================
try:
    import Features_RNALig as FR
except ImportError as e:
    FR = None
    _feature_import_error = str(e)
else:
    _feature_import_error = None


# ============================================================
#  Streamlit page configuration (light theme only)
# ============================================================
st.set_page_config(
    page_title="RNALig – RNA–Ligand Binding Affinity Predictor",
    layout="wide",
)

# Global CSS
st.markdown(
    """
    <style>
    /* Light grey global background */
    .main {
        background-color: #f4f6fb;
    }

    .block-container {
        max-width: 96% !important;
        padding-top: 1.0rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
        padding-bottom: 1.8rem !important;
    }

    h1, h2, h3 {
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", sans-serif;
    }
    p, li {
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", sans-serif;
        font-size: 0.96rem;
    }

    .small-muted {
        font-size: 0.85rem;
        color: #777777;
    }

    /* Header bar */
    .header-wrap {
        background: #ffffff;
        border-radius: 0 0 18px 18px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
        padding: 12px 22px 14px 22px;
        margin-bottom: 18px;
        border-bottom: 1px solid #e5e7eb;
    }

    .header-title {
        font-size: 28px;
        font-weight: 800;
        font-family: 'Inter', sans-serif;
        color: #1f2933;
        margin-bottom: 4px;
    }

    .header-subtitle {
        font-size: 14px;
        color: #6b7280;
    }

    .content-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 2.0rem 2.4rem 2.2rem 2.4rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.10);
        margin-bottom: 1.5rem;
    }

    .movie-card {
        background: #f9fafb;
        border-radius: 16px;
        padding: 0.9rem 0.9rem 0.3rem 0.9rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.10);
    }

    .footer-logo {
        max-height: 72px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
#  Model loading
# ============================================================
@st.cache_resource
def load_model_bundle() -> Tuple[Optional[object], Optional[List[str]]]:
    """
    Load RNALig_training_model.pkl.

    The file may contain either:
      - {"model": <sklearn_estimator>, "features": [feat1, ...]}
      - or a plain sklearn estimator.
    """
    try:
        with open("RNALig_training_model.pkl", "rb") as f:
            bundle = joblib.load(f)
    except FileNotFoundError:
        st.error("Model file `RNALig_training_model.pkl` not found in this folder.")
        return None, None
    except Exception as e:
        st.error(f"Failed to load model bundle: {e}")
        return None, None

    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"], bundle.get("features")
    return bundle, None


# ============================================================
#  Build default args for Features_RNALig
# ============================================================
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

    # visualization flags (we handle visualization ourselves)
    args.viz_rna = False
    args.viz_ligand = False
    args.viz_complex = False
    args.pocket_cutoff = 5.0
    args.pocket_sasa = 0.05
    args.rna_label_topk = 5

    args.lig_viz_dir = None
    args.rna_viz_dir = None

    return args


# ============================================================
#  Core feature extraction pipeline
# ============================================================
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
        row_dict = FR.process_one_pdb(path, args)
        rows.append(row_dict)

        pdb_id = row_dict.get("PDB_ID", os.path.splitext(base)[0])
        clean_name = os.path.splitext(base)[0] + "_clean.pdb"
        clean_path = os.path.join(outdir, clean_name)
        if os.path.exists(clean_path):
            cleaned_map[pdb_id] = clean_path

    df = pd.DataFrame(rows)
    if "PDB_ID" in df.columns:
        cols = ["PDB_ID"] + [c for c in df.columns if c != "PDB_ID"]
        df = df[cols]

    # round numeric features to 2 decimals
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].round(2)

    return df, cleaned_map


# ============================================================
#  Prediction helper
# ============================================================
def predict_binding_affinity(df_features: pd.DataFrame):
    model, feat_names = load_model_bundle()
    if model is None:
        return None, None

    id_col = None
    for c in df_features.columns:
        low = c.lower()
        if "pdb" in low or "id" in low or "name" in low:
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

    # round predictions
    y_pred = np.round(y_pred, 2)

    if id_col is not None:
        df_pred = pd.DataFrame(
            {"PDB_ID": df_features[id_col], "Predicted_binding_affinity_kcal_mol": y_pred}
        )
    else:
        df_pred = pd.DataFrame(
            {
                "Index": np.arange(len(df_features)),
                "Predicted_binding_affinity_kcal_mol": y_pred,
            }
        )

    df_combined = df_features.copy()
    df_combined["Predicted_binding_affinity_kcal_mol"] = y_pred

    return df_pred, df_combined


# ============================================================
#  3D visualization helpers
# ============================================================
def show_3d_structure(
    pdb_str: str,
    width: int = 430,
    height: int = 320,
    spin: bool = False,
):
    """
    Render a PDB string with py3Dmol.

    - RNA shown as cartoon (spectrum)
    - Ligands (hetero atoms) as sticks
    - Pocket surface around ligands (vdW surface with opacity)
    """
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_str, "pdb")

    # Cartoon for everything (RNA)
    view.setStyle({"cartoon": {"color": "spectrum"}})

    # Ligand sticks: any hetero atom
    view.addStyle(
        {"hetflag": True},
        {"stick": {"colorscheme": "cyanCarbon", "radius": 0.25}},
    )

    # Pocket / surface around ligand
    try:
        view.addSurface(
            py3Dmol.VDW,
            {"opacity": 0.35, "color": "white"},
            {"hetflag": True},
        )
    except Exception:
        pass

    view.zoomTo()
    if spin:
        view.spin(True)

    html = view._make_html()
    st.components.v1.html(html, height=height + 20)


def show_feature_panel(row: pd.Series, cleaned_path: Optional[str] = None):
    pdb_id = row.get("PDB_ID", "Unknown")
    pred = row.get("Predicted_binding_affinity_kcal_mol", None)

    st.markdown(f"##### 📁 {pdb_id}")
    if pred is not None:
        st.markdown(
            f"**Predicted binding affinity:** "
            f"`{float(pred):.2f} kcal/mol`"
        )

    col_left, col_right = st.columns([2.0, 1.2])

    with col_left:
        st.markdown("**Feature values (all)**")
        df_single = row.to_frame(name="Value")
        st.dataframe(df_single, use_container_width=True)

        numeric = row.select_dtypes(include=[np.number])
        if len(numeric) > 0:
            st.markdown("**Numeric features (bar chart)**")
            st.bar_chart(numeric)

    with col_right:
        if cleaned_path is not None and os.path.exists(cleaned_path):
            try:
                with open(cleaned_path, "r") as f:
                    pdb_block = f.read()
                st.markdown("**Cleaned complex (3D view)**")
                show_3d_structure(pdb_block, width=360, height=280, spin=False)
            except Exception as e:
                st.warning(f"Could not render cleaned PDB: {e}")
        else:
            st.info("No cleaned PDB found to display.")


# ============================================================
#  Demo helpers for home page
# ============================================================
def find_demo_pdbs() -> List[str]:
    demos = []
    for fname in os.listdir("."):
        low = fname.lower()
        if low.endswith(".pdb") and low.startswith("demo"):
            demos.append(fname)
    demos.sort()
    return demos


# ============================================================
#  Header (RNALig logo + title)
# ============================================================
def render_header():
    st.markdown('<div class="header-wrap">', unsafe_allow_html=True)

    col_logo, col_text = st.columns([0.16, 0.84])

    # Try to find RNALig logo
    logo_path = None
    for fname in os.listdir("."):
        low = fname.lower()
        if low.startswith("rnalig") and low.endswith((".png", ".jpg", ".jpeg")):
            logo_path = fname
            break

    with col_logo:
        if logo_path:
            st.image(logo_path, use_column_width=True)
        else:
            st.markdown("**RNALig**")

    with col_text:
        st.markdown(
            """
            <div class="header-title">
                RNALig – RNA–Ligand Binding Affinity Predictor
            </div>
            <div class="header-subtitle">
                AI-driven scoring &amp; interpretability for RNA–ligand complexes
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
#  Footer (Computational BioLab + logo)
# ============================================================
def render_footer():
    st.markdown(
        "<hr style='margin-top:2.0rem;margin-bottom:0.8rem;'>",
        unsafe_allow_html=True,
    )

    col_logo, col_text = st.columns([0.12, 0.88])

    # Robust search for any logo whose filename contains "lab" and "logo"
    lab_logo_path = None
    for root, _, files in os.walk("."):
        for fname in files:
            low = fname.lower()
            if "lab" in low and "logo" in low and low.endswith(
                (".png", ".jpg", ".jpeg")
            ):
                lab_logo_path = os.path.join(root, fname)
                break
        if lab_logo_path:
            break

    with col_logo:
        if lab_logo_path:
            st.image(lab_logo_path, use_column_width=True)
        else:
            st.write("")

    with col_text:
        st.markdown(
            """
**Computational BioLab**  
Email: [computationalbiolab@gmail.com](mailto:computationalbiolab@gmail.com)  
            """
        )


# ============================================================
#  Home page
# ============================================================
def render_home_content():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)

    st.markdown("### Overview")

    col_text, col_demo = st.columns([2.0, 1.4])

    with col_text:
        st.write(
            "RNALig is an AI-driven scoring function that estimates RNA–ligand "
            "binding affinities directly from 3D complexes. It automatically "
            "cleans raw PDB/mmCIF files, standardises ligands and detects the "
            "RNA binding pocket. A rich set of structural and physicochemical "
            "descriptors such as SASA, non-covalent contacts, hydrogen bonds, "
            "stacking interactions and electrostatics, is extracted for each "
            "complex. These features are fed into a trained Random Forest model "
            "to predict binding affinity in kcal/mol. The interface is designed "
            "as an end-to-end pipeline that exposes both the feature table and "
            "final scores for every structure, supporting interpretability, "
            "virtual screening and method benchmarking."
        )
        st.markdown(
            "Use the **“Run Predictions”** page to upload or fetch your own "
            "complexes and run the full pipeline."
        )

    with col_demo:
        demo_files = find_demo_pdbs()
        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
        placeholder = st.empty()

        if not demo_files:
            st.info(
                "Place one or more demo PDB files in this folder with names like "
                "`demo1.pdb`, `demo2.pdb`, ... to show an animated example here."
            )
        else:
            # autoplay through all demos once
            for fname in demo_files:
                try:
                    with open(fname, "r") as f:
                        pdb_block = f.read()
                except Exception:
                    continue
                with placeholder.container():
                    show_3d_structure(pdb_block, spin=True)
                time.sleep(1.0)

            # keep the last demo spinning
            try:
                with open(demo_files[-1], "r") as f:
                    pdb_last = f.read()
                with placeholder.container():
                    show_3d_structure(pdb_last, spin=True)
            except Exception:
                pass

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
<p class="small-muted">
RNALig is intended for research use only. Predictions should be interpreted
alongside structural inspection and experimental data.
</p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
#  Fetch PDB from RCSB
# ============================================================
def fetch_pdb_from_rcsb(pdb_id: str, out_dir: str) -> Optional[str]:
    pdb_id_clean = pdb_id.strip().lower()
    if len(pdb_id_clean) != 4:
        return None

    url = f"https://files.rcsb.org/download/{pdb_id_clean.upper()}.pdb"
    dest = os.path.join(out_dir, f"{pdb_id_clean}.pdb")
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200 and "ATOM" in r.text:
            with open(dest, "w") as f:
                f.write(r.text)
            return dest
    except Exception:
        return None
    return None


# ============================================================
#  Run Predictions page
# ============================================================
def render_run_pipeline():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.header("Run Predictions")

    if FR is None:
        st.error(
            "Could not import `Features_RNALig`. "
            "Make sure `Features_RNALig.py` is in this folder and all its "
            "dependencies (freesasa, rdkit, RNA, etc.) are installed."
        )
        if _feature_import_error:
            with st.expander("Import error details"):
                st.code(_feature_import_error)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown(
        """
This page performs the full **clean → feature extraction → prediction** workflow
for each RNA–ligand complex you provide.
        """
    )

    st.subheader("Input mode")

    mode = st.radio(
        "Choose how to load structures:",
        (
            "Option 1: Fetch PDB IDs from RCSB",
            "Option 2: Upload up to 5 PDB/mmCIF files",
            "Option 3: Upload a ZIP with many PDB/mmCIF files",
        ),
    )

    pdb_paths: List[str] = []

    # ------------- Option 1: fetch from PDB ----------------
    if mode.startswith("Option 1"):
        st.markdown(
            "Enter one or more **PDB IDs** (4 characters) separated by spaces or commas."
        )
        pdb_text = st.text_input("PDB IDs", value="")

        if pdb_text.strip():
            tmp_dir = tempfile.mkdtemp(prefix="rnalig_fetch_")
            ids = [
                token.strip().lower()
                for token in pdb_text.replace(",", " ").split()
                if token.strip()
            ]
            if ids:
                st.info(f"Will fetch: {', '.join(i.upper() for i in ids)}")
                for pid in ids:
                    path = fetch_pdb_from_rcsb(pid, tmp_dir)
                    if path:
                        pdb_paths.append(path)
                    else:
                        st.warning(f"Could not fetch a valid PDB for ID `{pid}`.")

    # ------------- Option 2: upload up to 5 ----------------
    elif mode.startswith("Option 2"):
        uploads = st.file_uploader(
            "Upload PDB/mmCIF files",
            type=["pdb", "cif", "mmcif"],
            accept_multiple_files=True,
        )
        if uploads:
            if len(uploads) > 5:
                st.warning(
                    "You uploaded more than 5 files; only the first 5 will be processed."
                )
                uploads = uploads[:5]
            tmp_in = tempfile.mkdtemp(prefix="rnalig_in_")
            for up in uploads:
                out_path = os.path.join(tmp_in, up.name)
                with open(out_path, "wb") as f:
                    f.write(up.getbuffer())
                pdb_paths.append(out_path)

    # ------------- Option 3: ZIP with many -----------------
    else:
        zfile = st.file_uploader(
            "Upload a ZIP containing PDB/mmCIF files", type=["zip"]
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

    # ------------- Run pipeline button ---------------------
    run_btn = st.button(
        " Run full pipeline (features + prediction)",
        type="primary",
        use_container_width=False,
    )

    if run_btn:
        if not pdb_paths:
            st.error("No structures to process. Please provide PDB IDs or upload files.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        with st.spinner("Running feature extraction for all structures..."):
            try:
                df_features, cleaned_map = run_feature_extraction(pdb_paths)
            except Exception as e:
                st.error(f"Feature extraction failed: {e}")
                st.markdown("</div>", unsafe_allow_html=True)
                return

        st.success(f"Extracted features for {len(df_features)} structure(s).")

        with st.spinner("Predicting binding affinities..."):
            df_pred, df_combined = predict_binding_affinity(df_features)

        if df_pred is None:
            st.error("Prediction step failed due to model issues.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        # ---------------- Global summary ----------------
        st.subheader("Global summary")
        st.markdown("**All predictions**")
        st.dataframe(df_pred, use_container_width=True)

        # ΔG vs best binder (most negative)
        st.markdown("###### Δ Affinity vs best binder")
        if "Predicted_binding_affinity_kcal_mol" in df_pred.columns:
            best_val = df_pred["Predicted_binding_affinity_kcal_mol"].min()
            df_delta = df_pred.copy()
            df_delta["ΔG_vs_best"] = (df_delta["Predicted_binding_affinity_kcal_mol"] - best_val).round(3)
            st.dataframe(df_delta, use_container_width=True)

        # ---------------- Downloads ----------------
        st.markdown("#### Download results")
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

        # ---------------- Optional compact heatmap ----------------
        st.markdown("#### Feature patterns across complexes")
        num_df = df_combined.select_dtypes(include=[np.number])
        # Only draw heatmap for reasonably small data to keep app responsive
        if 1 < len(num_df) <= 30 and 1 < num_df.shape[1] <= 30:
            import seaborn as sns
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(min(10, 0.5 * num_df.shape[1] + 4),
                                            min(8, 0.3 * len(num_df) + 3)))
            sns.heatmap(num_df, cmap="viridis", ax=ax)
            ax.set_xlabel("Features")
            ax.set_ylabel("Complex index")
            st.pyplot(fig)
        # If too big, we silently skip (no "skipped" message).

        st.markdown("---")
        st.subheader("Per-complex feature & structure views")

        id_col = "PDB_ID" if "PDB_ID" in df_combined.columns else None

        for idx, row in df_combined.iterrows():
            if id_col:
                pdb_id = row[id_col]
                clean_path = cleaned_map.get(pdb_id)
                label = f"{pdb_id}"
            else:
                pdb_id = f"row_{idx}"
                clean_path = None
                label = f"Complex {idx}"

            with st.expander(label, expanded=False):
                show_feature_panel(row, cleaned_path=clean_path)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
#  Tutorial Page
# ============================================================
def render_tutorial():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.header("Tutorial")

    st.markdown(
        """
### 1. Prepare input structures

- RNA–ligand complexes in **PDB** or **mmCIF** format, or valid **PDB IDs** at RCSB.  
- Each file should contain at least one RNA chain and one bound small-molecule ligand.

### 2. Run the pipeline

1. Open the **Run Predictions** page.  
2. Choose one input mode:
   - Fetch PDB IDs directly from RCSB.
   - Upload up to 5 structures.
   - Upload a ZIP archive of many complexes.  
3. Click **“Run full pipeline (features + prediction)”**.  
4. RNALig will:
   - Clean the complex.
   - Detect the ligand pocket.
   - Compute structural & physicochemical features.
   - Apply the trained Random Forest model.

### 3. Interpret the results

- **Global summary table**: overview of complexes and predicted binding affinities.  
- **ΔG_vs_best table**: relative affinity compared to the best binder in the batch.  
- **Per-complex panels**:
  - Full feature vector (interactive table).
  - Bar chart of numeric features.
  - 3D view of the cleaned complex with RNA cartoon, ligand sticks and pocket surface.

> RNALig is a research tool. Predictions should be interpreted together with
> structural inspection and, whenever possible, experimental data.
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
#  Main
# ============================================================
def main():
    # Sidebar navigation (no theme toggle)
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Run Predictions", "Tutorial"],
        index=0,
    )

    render_header()

    if page == "Home":
        render_home_content()
    elif page == "Run Predictions":
        render_run_pipeline()
    else:
        render_tutorial()

    render_footer()


if __name__ == "__main__":
    main()
