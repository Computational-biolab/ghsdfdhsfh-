import streamlit as st
import os, io, tempfile, zipfile
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import joblib

import py3Dmol  # make sure py3Dmol is in your environment.yml

# ------------------ Try to import your feature extractor ------------------
try:
    import Features_RNALig as FR
except ImportError as e:
    FR = None
    _feature_import_error = str(e)
else:
    _feature_import_error = None


# ------------------ Model loading ------------------

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


# ------------------ Args for Features_RNALig ------------------

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

    # visualization flags (you can toggle if Features_RNALig uses them)
    args.viz_rna = False
    args.viz_ligand = False
    args.viz_complex = False
    args.pocket_cutoff = 5.0
    args.pocket_sasa = 0.05
    args.rna_label_topk = 5

    args.lig_viz_dir = None
    args.rna_viz_dir = None

    return args


# ------------------ Core pipeline functions ------------------

def run_feature_extraction(pdb_paths: List[str]):
    """
    For each PDB file, call FR.process_one_pdb and collect:
      - features DataFrame
      - mapping PDB_ID -> cleaned PDB path ( *_clean.pdb )
    """
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

        # guess ID
        pdb_id = row.get("PDB_ID", os.path.splitext(base)[0])

        # Features_RNALig usually writes <basename>_clean.pdb in outdir
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
    """
    Use RNALig model to predict binding affinity for each row.
    Returns:
      - df_pred_only (ID + prediction)
      - df_combined (features + prediction)
    """
    model, feat_names = load_model_bundle()
    if model is None:
        return None, None

    # choose ID column if present
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


# ------------------ Visualization helpers ------------------

def show_3d_structure(pdb_path: str):
    """Embed a simple 3D viewer for a PDB file using py3Dmol."""
    try:
        with open(pdb_path, "r") as f:
            pdb_block = f.read()
    except Exception as e:
        st.warning(f"Could not load cleaned structure for 3D view: {e}")
        return

    view = py3Dmol.view(width=500, height=400)
    view.addModel(pdb_block, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.addStyle({"resn": "LIG"}, {"stick": {}})  # if your ligand residue name is LIG
    view.zoomTo()

    html = view._make_html()
    st.components.v1.html(html, height=420)


def show_feature_panel(row: pd.Series, cleaned_path: Optional[str] = None):
    """
    Show: prediction card + feature table + bar chart (numeric features)
    for a single complex.
    """
    pdb_id = row.get("PDB_ID", "Unknown")
    pred = row.get("Predicted_binding_affinity_kcal_mol", None)

    # Header card
    st.markdown(
        f"### 🧾 {pdb_id}"
    )
    if pred is not None:
        st.markdown(
            f"**Predicted binding affinity:** `{pred:.3f} kcal/mol`"
        )

    cols = st.columns([2, 1])

    with cols[0]:
        st.markdown("**Feature values (all)**")
        df_single = row.to_frame(name="Value")
        st.dataframe(df_single, use_container_width=True)

        # numeric-only bar chart
        num_series = row.select_dtypes(include=[np.number])
        if len(num_series) > 0:
            st.markdown("**Numeric features (bar chart)**")
            st.bar_chart(num_series)

    with cols[1]:
        if cleaned_path is not None:
            st.markdown("**Cleaned complex (3D view)**")
            show_3d_structure(cleaned_path)
        else:
            st.info("No cleaned PDB available for 3D view.")


# ------------------ Streamlit UI ------------------

def main():
    st.set_page_config(page_title="RNALig – Full Pipeline", layout="wide")
    st.title("🧬 RNALig – RNA–Ligand Binding Affinity Pipeline")

    st.markdown(
        """
This interface performs the **full RNALig workflow** for each input structure:

1. **PDB/mmCIF input** (single or multiple)  
2. **Cleaning + feature extraction** via `Features_RNALig.py`  
3. **Binding affinity prediction** using your trained model  
4. For **each complex**, you see:
   - Cleaned 3D structure  
   - Full feature vector  
   - Bar plot of numeric features  
   - Predicted binding affinity
        """
    )

    if FR is None:
        st.error(
            "Could not import `Features_RNALig`. Make sure `Features_RNALig.py` "
            "is in this folder and all its dependencies (rdkit, freesasa, RNA, etc.) "
            "are installed in your conda env."
        )
        if _feature_import_error:
            with st.expander("Import error details"):
                st.code(_feature_import_error)
        return

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
                st.warning("You uploaded more than 5 files; only the first 5 will be used.")
                uploads = uploads[:5]
            tmp_in = tempfile.mkdtemp(prefix="rnalig_in_")
            for up in uploads:
                out_path = os.path.join(tmp_in, up.name)
                with open(out_path, "wb") as f:
                    f.write(up.getbuffer())
                pdb_paths.append(out_path)

    else:  # ZIP
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
                st.info(f"Found {len(pdb_paths)} structure files inside the ZIP.")
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
                st.error(f"Feature extraction failed: {e}")
                return

        st.success(f"Extracted features for {len(df_features)} structure(s).")

        # Predict
        with st.spinner("Predicting binding affinities..."):
            df_pred, df_combined = predict_binding_affinity(df_features)
        if df_pred is None:
            st.error("Prediction step failed due to model issues.")
            return

        st.subheader("Global summary")
        st.markdown("**All predictions (table)**")
        st.dataframe(df_pred, use_container_width=True)

        # Downloads
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
        st.subheader("Per-complex details")

        # Show one panel per complex
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


if __name__ == "__main__":
    main()
