import os
import time
import tempfile
import zipfile
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import py3Dmol
import requests
import streamlit as st

# -------------------------------------------------------------------
# Try to import feature extractor
# -------------------------------------------------------------------
try:
    import Features_RNALig as FR
except ImportError as e:
    FR = None
    _feature_import_error = str(e)
else:
    _feature_import_error = None

# -------------------------------------------------------------------
# Streamlit page config + global CSS
# -------------------------------------------------------------------
st.set_page_config(
    page_title="RNALig – RNA–Ligand Binding Affinity Predictor",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Global background */
    .main {
        background-color: #f4f6fb;
    }

    /* Wider content area */
    .block-container {
        max-width: 95% !important;
        padding-top: 0.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-bottom: 1.5rem !important;
    }

    h1, h2, h3 {
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", sans-serif;
    }
    p, li, label, span {
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", sans-serif;
        font-size: 0.96rem;
    }

    .small-muted {
        font-size: 0.85rem;
        color: #777;
    }

    /* HEADER BAR */
    .header-wrap {
        background: #ffffff;
        border-radius: 0 0 18px 18px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
        padding: 10px 24px 16px 24px;
        margin-bottom: 18px;
        border-bottom: 1px solid #e5e7eb;
    }

    .header-title {
        font-size: 27px;
        font-weight: 800;
        font-family: 'Inter', sans-serif;
        color: #1f2933;
        margin-bottom: 4px;
    }

    .header-subtitle {
        font-size: 14px;
        color: #6b7280;
    }

    /* Main white card for page content */
    .content-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 1.8rem 2.3rem 2.2rem 2.3rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.10);
        margin: 0 auto 1.5rem auto;
        width: 100%;
    }

    /* Demo movie card on home page */
    .movie-card {
        background: #f9fafb;
        border-radius: 16px;
        padding: 0.8rem 0.8rem 0.2rem 0.8rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.10);
    }

    /* Hide built-in theme switcher */
    div[data-testid="stThemeSwitcher"] {
        display: none !important;
    }

    /* Footer styling */
    .footer-wrap {
        margin-top: 1.5rem;
        padding-top: 0.8rem;
        border-top: 1px solid #e5e7eb;
        font-size: 0.9rem;
        color: #374151;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------
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
        st.error("Model file `RNALig_training_model.pkl` not found in this folder.")
        return None, None
    except Exception as e:
        st.error(f"Failed to load model bundle: {e}")
        return None, None

    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"], bundle.get("features")
    return bundle, None


# -------------------------------------------------------------------
# Default arguments for Features_RNALig
# -------------------------------------------------------------------
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

    # visualization flags (we only need cleaned PDBs; pocket viz here)
    args.viz_rna = False
    args.viz_ligand = False
    args.viz_complex = False
    args.pocket_cutoff = 5.0
    args.pocket_sasa = 0.05
    args.rna_label_topk = 5

    args.lig_viz_dir = None
    args.rna_viz_dir = None

    return args


# -------------------------------------------------------------------
# Core pipeline functions
# -------------------------------------------------------------------
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

        # Try to find cleaned pdb
        pdb_id = row.get("PDB_ID", os.path.splitext(base)[0])
        # Assume cleaned file named something like "<pdbid>_clean.pdb"
        candidate_names = [
            f"{pdb_id}_clean.pdb",
            f"{os.path.splitext(base)[0]}_clean.pdb",
        ]
        clean_path = None
        for c in candidate_names:
            cp = os.path.join(outdir, c)
            if os.path.exists(cp):
                clean_path = cp
                break
        if clean_path is not None:
            cleaned_map[pdb_id] = clean_path

    df = pd.DataFrame(rows)

    # Reorder to show PDB_ID first if present
    if "PDB_ID" in df.columns:
        cols = ["PDB_ID"] + [c for c in df.columns if c != "PDB_ID"]
        df = df[cols]

    # Round numeric features to 2 decimals
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].round(2)

    return df, cleaned_map


def predict_binding_affinity(df_features: pd.DataFrame):
    model, feat_names = load_model_bundle()
    if model is None:
        return None, None

    # Try to identify ID column
    id_col = None
    for c in df_features.columns:
        cl = c.lower()
        if "pdb" in cl or "id" in cl or "name" in cl:
            id_col = c
            break

    numeric = df_features.select_dtypes(include=[np.number]).copy()

    if feat_names:
        # Ensure all features present
        for f in feat_names:
            if f not in numeric.columns:
                numeric[f] = np.nan
        X = numeric[feat_names].astype(float)
    else:
        X = numeric

    X = X.fillna(X.median())
    y_pred = model.predict(X)
    y_pred = np.round(y_pred.astype(float), 2)

    if id_col is not None:
        df_pred = pd.DataFrame(
            {
                "PDB_ID": df_features[id_col],
                "Predicted_binding_affinity_kcal_mol": y_pred,
            }
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


# -------------------------------------------------------------------
# 3D viewer helpers
# -------------------------------------------------------------------
def show_3d_structure(
    pdb_str: str,
    width: int = 430,
    height: int = 320,
    spin: bool = False,
):
    """
    Render a PDB string with py3Dmol.
    - RNA as cartoon
    - Ligand as sticks (any HETATM)
    - Pocket as translucent surface around non-ligand atoms
    """
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_str, "pdb")

    # RNA/protein backbone
    view.setStyle({"cartoon": {"color": "spectrum"}})

    # Ligand (HETATM) as sticks
    view.setStyle({"hetflag": True}, {"stick": {"colorscheme": "cyanCarbon"}})

    # Pocket surface (everything except ligand)
    try:
        view.addSurface(
            py3Dmol.SAS,
            {"opacity": 0.25, "color": "white"},
            {"hetflag": True, "invert": True},
        )
    except Exception:
        # If surface fails, ignore quietly
        pass

    view.zoomTo()
    if spin:
        view.spin(True)
    html = view._make_html()
    st.components.v1.html(html, height=height + 15)


def show_feature_panel(row: pd.Series, cleaned_path: Optional[str] = None):
    pdb_id = row.get("PDB_ID", "Unknown")
    pred = row.get("Predicted_binding_affinity_kcal_mol", None)

    st.markdown(f"### 🧾 {pdb_id}")
    if isinstance(pred, (float, int)):
        st.markdown(f"**Predicted binding affinity:** `{pred:.2f} kcal/mol`")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("**Feature values (all)**")
        df_single = row.to_frame(name="Value")
        st.dataframe(df_single, use_container_width=True)

        # Numeric bar chart
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


# -------------------------------------------------------------------
# Demo helpers
# -------------------------------------------------------------------
def find_demo_pdbs() -> List[str]:
    """Return sorted list of demo*.pdb files in the current folder."""
    demos = []
    for fname in os.listdir("."):
        if fname.lower().endswith(".pdb") and fname.lower().startswith("demo"):
            demos.append(fname)
    demos.sort()
    return demos


# -------------------------------------------------------------------
# Header (RNALig logo + title)
# -------------------------------------------------------------------
def render_header():
    st.markdown('<div class="header-wrap">', unsafe_allow_html=True)

    col_logo, col_text = st.columns([0.16, 0.84])

    logo_path = "logo.png"  # RNALig logo

    with col_logo:
        try:
            if os.path.exists(logo_path):
                st.image(logo_path, width=130)
            else:
                st.markdown("**RNALig**")
        except Exception:
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


# -------------------------------------------------------------------
# Footer (Computational BioLab)
# -------------------------------------------------------------------
def render_footer():
    st.markdown('<div class="footer-wrap">', unsafe_allow_html=True)
    col_logo, col_text = st.columns([0.2, 0.8])

    with col_logo:
        lab_logo = "Lab_Logo.png"
        if os.path.exists(lab_logo):
            st.image(lab_logo, width=110)

    with col_text:
        st.markdown("**Computational BioLab**")
        st.markdown(
            "Email: "
            "[computationalbiolab@gmail.com]"
            "(mailto:computationalbiolab@gmail.com)"
        )
        st.markdown("All rights reserved.")
    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Page contents
# -------------------------------------------------------------------
def render_home_content():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)

    st.markdown("### Overview")
    col_text, col_demo = st.columns([2, 1.4])

    with col_text:
        st.write(
            "RNALig is an AI-driven scoring function that estimates RNA–ligand "
            "binding affinities directly from 3D complexes. It automatically "
            "cleans raw PDB/mmCIF files, standardises ligands and detects the "
            "RNA binding pocket. A rich set of structural and physicochemical "
            "descriptors—including SASA, non-covalent contacts, hydrogen bonds, "
            "stacking interactions and electrostatics—is extracted for each "
            "complex. These features are fed into a trained Random Forest model "
            "to predict binding affinity in kcal/mol. The interface is designed "
            "as an end-to-end pipeline that exposes both the feature table and "
            "final scores for every structure, supporting interpretability, "
            "virtual screening and method benchmarking."
        )
        st.markdown("")
        st.markdown(
            "Use the **“Run Predictions”** page to upload or fetch your own "
            "complexes and run the full pipeline."
        )

    with col_demo:
        demo_files = find_demo_pdbs()
        if not demo_files:
            st.info(
                "Place one or more demo PDB files in this folder with names like "
                "`demo1.pdb`, `demo2.pdb`, ... to show an animated example here."
            )
        else:
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            placeholder = st.empty()

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

            # Finally keep last demo visible
            try:
                with open(demo_files[-1], "r") as f:
                    pdb_last = f.read()
                with placeholder.container():
                    show_3d_structure(pdb_last, spin=True)
            except Exception:
                pass

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<p class="small-muted">RNALig is intended for research use only. '
        'Predictions should be interpreted alongside structural inspection '
        'and experimental data.</p>',
        unsafe_allow_html=True,
    )


def fetch_pdb_from_rcsb(pdb_id: str, outdir: str) -> Optional[str]:
    """
    Download PDB from RCSB by 4-letter ID.
    Tries .pdb first, then .cif.
    """
    pdb_id = pdb_id.lower().strip()
    if len(pdb_id) != 4:
        return None

    urls = [
        f"https://files.rcsb.org/download/{pdb_id}.pdb",
        f"https://files.rcsb.org/download/{pdb_id}.cif",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and len(r.text) > 100:
                ext = ".pdb" if url.endswith(".pdb") else ".cif"
                out_path = os.path.join(outdir, f"{pdb_id}{ext}")
                with open(out_path, "w") as f:
                    f.write(r.text)
                return out_path
        except Exception:
            continue
    return None


def render_run_pipeline():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.header("Run Predictions")

    if FR is None:
        st.error(
            "Could not import `Features_RNALig`. Make sure `Features_RNALig.py` "
            "is in this folder and all its dependencies (rdkit, freesasa, RNA, etc.) "
            "are installed in your conda environment."
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
            "Option 1: Fetch PDB from RCSB",
            "Option 2: Upload up to 5 PDB/mmCIF files",
            "Option 3: Upload a ZIP with many PDB/mmCIF files",
        ),
    )

    pdb_paths: List[str] = []

    # ---------------------------------------------------------
    # Option 1: Fetch PDBs from RCSB
    # ---------------------------------------------------------
    if mode.startswith("Option 1"):
        st.markdown(
            "Enter one or more 4-letter PDB IDs (comma-separated). "
            "Example: `1F27, 4JF2`"
        )
        pdb_input = st.text_input("PDB IDs")
        if pdb_input:
            tmp_in = tempfile.mkdtemp(prefix="rnalig_fetch_")
            ids = [p.strip() for p in pdb_input.split(",") if p.strip()]
            for pid in ids:
                path = fetch_pdb_from_rcsb(pid, tmp_in)
                if path is not None:
                    pdb_paths.append(path)
                else:
                    st.warning(f"Could not fetch structure for `{pid}` from RCSB.")

    # ---------------------------------------------------------
    # Option 2: Upload up to 5 PDB/mmCIF files
    # ---------------------------------------------------------
    elif mode.startswith("Option 2"):
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

    # ---------------------------------------------------------
    # Option 3: Upload ZIP
    # ---------------------------------------------------------
    else:
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

    # ---------------------------------------------------------
    # Run pipeline
    # ---------------------------------------------------------
    if st.button(" Run full pipeline (features + prediction)", type="primary"):
        if not pdb_paths:
            st.error("No structures to process. Please fetch or upload files first.")
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

        st.subheader("Global summary")
        st.markdown("**All predictions**")
        st.dataframe(df_pred, use_container_width=True)

        # Downloads
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

        # -----------------------------------------------------
        # Feature patterns across complexes (numeric snapshot)
        # -----------------------------------------------------
        st.markdown("---")
        st.subheader("Feature patterns across complexes")

        num_cols = df_combined.select_dtypes(include=[np.number]).columns
        # Remove ID-like numeric columns if any
        num_cols = [c for c in num_cols if not ("id" in c.lower())]

        if len(num_cols) > 0 and len(df_combined) > 1:
            df_for_view = df_combined[num_cols].copy()
            if df_for_view.shape[1] > 30:
                df_for_view = df_for_view.iloc[:, :30]
            if df_for_view.shape[0] > 40:
                df_for_view = df_for_view.iloc[:40, :]
            st.dataframe(
                df_for_view,
                use_container_width=True,
            )
            st.info(
                "Above table shows a compact snapshot of numeric feature patterns "
                "across complexes. For deeper analysis or custom plots, download "
                "the CSV files above and explore in your preferred tool."
            )
        else:
            st.info(
                "At least 2 complexes and some numeric features are required "
                "to show cross-complex patterns. Download CSVs above for "
                "further analysis if needed."
            )

        # -----------------------------------------------------
        # Per-complex views
        # -----------------------------------------------------
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


def render_tutorial():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.header("Tutorial")

    st.markdown(
        """
### 1. Prepare input structures

- RNA–ligand complexes in **PDB** or **mmCIF** format  
- Each file should contain at least one RNA chain and one bound small-molecule ligand.

### 2. Run the pipeline

1. Go to the **Run Predictions** page  
2. Choose upload mode:  
   - Fetch PDBs from RCSB (Option 1)  
   - Upload individual PDB/mmCIF files (Option 2)  
   - Upload a ZIP of many structures (Option 3)  
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
  - 3D view of cleaned complex with ligand + pocket surface

> RNALig is a research tool. Predictions should be interpreted together with
> structural inspection and experimental data where available.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Run Predictions", "Tutorial"],
        index=0,
    )

    # Header (logo + title)
    render_header()

    # Page contents
    if page == "Home":
        render_home_content()
    elif page == "Run Predictions":
        render_run_pipeline()
    else:
        render_tutorial()

    # Footer
    render_footer()


if __name__ == "__main__":
    main()
