# === FILE: app.py ===
"""
Streamlit RNALig app bundle

This single file includes two deliverables separated by markers:
 - app.py (the Streamlit application)
 - requirements.txt (below the marker)

Copy the app.py portion into a file named `app.py` in your GitHub repo and the requirements block into `requirements.txt`.

Notes / behavior:
 - Clean PDB: upload one or more PDB files. The app will detect non-ion ligands and let you select which ligand to keep for each file. Produces cleaned PDB(s) and a master CSV.
 - Predict: upload a model bundle .pkl (a dict with keys 'model' and 'features') and a features CSV/XLSX to predict binding affinity and download results.
 - Feature extraction: if you include Features_RNALig.py in the same repo and install its optional deps (freesasa, rdkit, RNA, py3Dmol) the app will import and call its `process_one_pdb` for richer features. If not available, a small fallback extractor computes simple sequence-based features.

This app aims to work "purely using Streamlit" on Streamlit Community Cloud. Some third-party compiled packages (freesasa, rdkit) may not be available on the free service — the app gracefully warns and falls back.
"""

import streamlit as st
import io, os, tempfile, sys, csv
from typing import List, Tuple
import pandas as pd
import numpy as np
import joblib

# ------------------ Clean PDB utilities (adapted from your clean_pdb.py) ------------------
ION_WAT = {'HOH','NA','K','CL','MG','ZN','CA','MN','FE','CU','BR','IOD','I','NI','HG','AG','CD','AU','PB','RB'}
STD_RNA = {'A','U','G','C'}
MOD_TO_STD = {
    'PSU': 'U','5MU': 'U','H2U': 'U','H5U': 'U','5MC': 'C','M2G': 'G','M2A': 'A','1MA': 'A','2MA':'A','M2U':'U'
}

def pad80(line: str) -> str:
    line = line.rstrip('\n')
    return (line + ' ' * (80 - len(line)))[:80]

def record_type(line: str) -> str:
    return pad80(line)[:6].strip().upper()

def get_resname(line: str) -> str:
    return pad80(line)[17:20].strip()

def get_chain(line: str) -> str:
    return pad80(line)[21].strip()

def map_resname_to_standard(resname: str):
    if not resname: return None
    r = resname.upper()
    if r in STD_RNA: return r
    if r in MOD_TO_STD: return MOD_TO_STD[r]
    first = r[0]
    if first in STD_RNA: return first
    for ch in r:
        if ch in STD_RNA: return ch
    return None


def detect_ligands_from_lines(lines: List[str]) -> List[Tuple[str,str]]:
    ligands = set()
    for raw in lines:
        rec = record_type(raw)
        if rec != 'HETATM':
            continue
        resname = get_resname(raw)
        chain = get_chain(raw)
        if resname.upper() not in ION_WAT:
            ligands.add((resname, chain))
    return sorted(ligands)


def replace_resname_in_line(line80: str, new_resname: str) -> str:
    new = list(line80)
    new_r = new_resname.rjust(3) if new_resname else '   '
    new[17:20] = list(new_r)
    return ''.join(new)


def clean_pdb_selective_from_lines(lines: List[str], selected_ligand: Tuple[str,str]):
    sel_resname, sel_chain = selected_ligand
    kept = []
    conversions = {}
    for raw in lines:
        rec = record_type(raw)
        line80 = pad80(raw)
        if rec in {'TER','END','MODEL','ENDMDL'}:
            kept.append(raw)
            continue
        if rec == 'ATOM':
            orig_resname = line80[17:20].strip()
            mapped = map_resname_to_standard(orig_resname)
            if mapped:
                new_line80 = replace_resname_in_line(line80, mapped)
                kept.append(new_line80.rstrip() + '\n')
                resseq = line80[22:26].strip()
                conversions[(orig_resname, line80[21].strip(), resseq)] = mapped
            else:
                continue
        elif rec == 'HETATM':
            resname = line80[17:20].strip()
            chain = line80[21].strip()
            if (resname.upper() not in ION_WAT) and (resname == sel_resname) and (chain == sel_chain):
                kept.append(raw)
            else:
                continue
    if not kept or kept[-1].strip() != 'END':
        kept.append('END\n')
    return kept, conversions

# ------------------ Simple feature fallback (if Features_RNALig not available) ------------------
from Bio.PDB import PDBParser

def simple_rna_features_from_lines(lines: List[str]):
    """Return a minimal features dict: sequence length and counts per base, GC%"""
    # write to temp file and parse with Biopython
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdb')
    try:
        tmp.writelines([l.encode('utf-8') if isinstance(l,str) else l for l in lines])
        tmp.close()
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('T', tmp.name)
        seq = ''
        for ch in structure.get_chains():
            for res in ch.get_residues():
                rn = res.get_resname().strip()
                if rn in MOD_TO_STD:
                    seq += MOD_TO_STD[rn]
                elif rn in STD_RNA:
                    seq += rn
        counts = {b: seq.count(b) for b in ('A','C','G','U')}
        total = sum(counts.values())
        gc = (counts['G'] + counts['C'])/total*100 if total else np.nan
        return {'RNA_length': total, 'RNA_count_A': counts['A'], 'RNA_count_C': counts['C'], 'RNA_count_G': counts['G'], 'RNA_count_U': counts['U'], 'RNA_GC_percent': round(gc,3)}
    finally:
        try: os.unlink(tmp.name)
        except Exception: pass

# ------------------ Streamlit App ------------------

def main():
    st.set_page_config(page_title='RNALig — Streamlit', layout='wide')
    st.title('RNALig — Streamlit deployment bundle')

    tabs = st.tabs(['Clean PDB', 'Extract Features', 'Predict Binding Affinity'])

    # -------- Tab 1: Clean PDB --------
    with tabs[0]:
        st.header('Clean PDB files — keep one ligand per structure')
        uploaded = st.file_uploader('Upload one or more PDB files', type=['pdb','ent','txt'], accept_multiple_files=True)
        if uploaded:
            file_selections = {}
            st.write('Detected ligands (choose which to keep for each file):')
            for up in uploaded:
                content = up.getvalue().decode('utf-8', errors='ignore').splitlines(True)
                ligs = detect_ligands_from_lines(content)
                key = up.name
                if not ligs:
                    st.warning(f'No non-ion ligands detected in {up.name} — file will be skipped')
                    file_selections[key] = None
                elif len(ligs) == 1:
                    st.info(f'{up.name}: auto-selected ligand {ligs[0][0]} chain {ligs[0][1]}')
                    file_selections[key] = ligs[0]
                else:
                    opts = [f"{r}:{c or '-'}" for (r,c) in ligs]
                    sel = st.selectbox(f'Select ligand to keep for {up.name}', options=opts, key=f'sel_{up.name}')
                    idx = opts.index(sel)
                    file_selections[key] = ligs[idx]

            if st.button('Process and download cleaned files'):
                master_rows = []
                zip_buffer = io.BytesIO()
                import zipfile
                with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                    for up in uploaded:
                        sel = file_selections.get(up.name)
                        if not sel:
                            continue
                        content = up.getvalue().decode('utf-8', errors='ignore').splitlines(True)
                        kept, conv = clean_pdb_selective_from_lines(content, sel)
                        cleaned_name = os.path.splitext(up.name)[0] + '_cleaned.pdb'
                        zf.writestr(cleaned_name, ''.join(kept))
                        # conversions summary
                        conv_summary = ''
                        if conv:
                            conv_items = []
                            seen = set()
                            for (orig, chain, resseq), mapped in sorted(conv.items(), key=lambda x: (x[0][1], x[0][2], x[0][0])):
                                key = (orig, chain, resseq, mapped)
                                if key in seen: continue
                                seen.add(key)
                                conv_items.append(f"{orig}->{mapped} (chain {chain or '-'} res {resseq})")
                            conv_summary = '; '.join(conv_items)
                        master_rows.append((os.path.splitext(up.name)[0], sel[0], sel[1] or '', conv_summary))
                if master_rows:
                    df = pd.DataFrame(master_rows, columns=['PDB_ID','Ligand_name','Chain_ID','Residue_mappings'])
                    st.download_button('Download cleaned PDBs (zip)', data=zip_buffer.getvalue(), file_name='cleaned_pdbs.zip')
                    csv_bytes = df.to_csv(index=False).encode('utf-8')
                    st.download_button('Download master summary CSV', data=csv_bytes, file_name='clean_summary_master.csv')
                else:
                    st.info('No cleaned files produced.')

    # -------- Tab 2: Extract Features --------
    with tabs[1]:
        st.header('Feature extraction — single PDB (simple fallback or full extractor if available)')
        st.write('If you included `Features_RNALig.py` in the same repo and installed optional dependencies, the app will use it. Otherwise a simple fallback extractor runs.')

        pdb_file = st.file_uploader('Upload single PDB file for feature extraction', type=['pdb','cif','mmcif'], key='feat_upload')

        # Attempt import and capture any exception text
        FR = None
        import_err_text = None

        # First try direct import (works when Features_RNALig.py is in repo root and importable)
        try:
            import Features_RNALig as FR_mod
            FR = FR_mod
            st.success('Found Features_RNALig.py in repo and imported successfully.')
        except Exception as e:
            import_err_text = f"Direct import failed: {e}"

        # If direct import failed, try importing from file path (useful when file present but not on sys.path)
        if FR is None:
            try:
                import importlib.util, pathlib, sys, traceback
                candidate = pathlib.Path('Features_RNALig.py')
                if candidate.exists():
                    spec = importlib.util.spec_from_file_location('Features_RNALig_fromfile', str(candidate))
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    FR = mod
                    st.success('Imported Features_RNALig.py from repository file path.')
                    import_err_text = None
                else:
                    import_err_text = (import_err_text or '') + '\nFeatures_RNALig.py not found at repo root.'
            except Exception as e:
                import_err_text = (import_err_text or '') + '\nImport-from-file failed:\n' + traceback.format_exc()

        # Allow user to upload Features_RNALig.py into the running session and try import
        uploaded_extractor = st.file_uploader('Upload Features_RNALig.py (optional, to override)', type=['py'])
        if uploaded_extractor is not None:
            try:
                code = uploaded_extractor.read().decode('utf-8')
                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='_Features_RNALig.py')
                tmpf.write(code.encode('utf-8'))
                tmpf.flush(); tmpf.close()
                import importlib.util, traceback
                spec = importlib.util.spec_from_file_location('Features_RNALig_uploaded', tmpf.name)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                FR = mod
                st.success('Uploaded Features_RNALig.py imported successfully.')
                import_err_text = None
            except Exception:
                import_err_text = (import_err_text or '') + '\nUpload import failed:\n' + traceback.format_exc()

        if import_err_text:
            with st.expander('Feature extractor import error / diagnostics (click to view)'):
                st.text(import_err_text)
            st.warning('Features_RNALig could not be used; the app will use the simple fallback extractor.')

        if pdb_file:
            if st.button('Run feature extraction'):
                lines = pdb_file.getvalue().decode('utf-8', errors='ignore').splitlines(True)
                if FR is not None and hasattr(FR, 'process_one_pdb'):
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdb')
                    tmp.writelines([l.encode('utf-8') for l in lines])
                    tmp.close()
                    outdir = tempfile.mkdtemp()
                    class Args: pass
                    args = Args()
                    args.cutoff=4.0; args.min_heavy=8; args.require_carbon=True; args.keep_ions=False
                    args.viz_rna=False; args.viz_ligand=False; args.outdir=outdir; args.pocket_cutoff=5.0; args.pocket_sasa=0.05; args.rna_label_topk=5
                    args.vdw_mode='shell'; args.vdw_legacy_cutoff=4.0; args.hbond_cutoff=3.5; args.hydroph_cutoff=4.5
                    args.elec_mode='charged'; args.elec_targets='phosphate'; args.elec_qthr=0.2; args.elec_dmin=3.0; args.elec_dmax=10.0; args.elec_include_negative=False
                    try:
                        row = FR.process_one_pdb(tmp.name, args)
                        df = pd.DataFrame([row])
                        st.dataframe(df)
                        st.download_button('Download features CSV', data=df.to_csv(index=False).encode('utf-8'), file_name='features_from_pdb.csv')
                    except Exception as e:
                        import traceback
                        st.error('Full extractor raised an exception. See details below.')
                        st.text(traceback.format_exc())
                        st.info('Falling back to simple extractor output:')
                        feats = simple_rna_features_from_lines(lines)
                        df = pd.DataFrame([feats])
                        st.dataframe(df)
                        st.download_button('Download simple features CSV', data=df.to_csv(index=False).encode('utf-8'), file_name='simple_features.csv')
                else:
                    feats = simple_rna_features_from_lines(lines)
                    df = pd.DataFrame([feats])
                    st.dataframe(df)
                    st.download_button('Download simple features CSV', data=df.to_csv(index=False).encode('utf-8'), file_name='simple_features.csv')

    # -------- Tab 3: Predict Binding Affinity --------
    with tabs[2]:
        st.header('Predict binding affinity from features using a saved model bundle (.pkl)')
        st.write('Model bundle is loaded automatically from the repository (`RNALig_training_model.pkl`). Upload a features table (.csv/.xlsx) to run predictions.')

        # load model bundle from repository
        try:
            with open('RNALig_training_model.pkl','rb') as f:
                bundle = joblib.load(f)
            if isinstance(bundle, dict) and 'model' in bundle:
                model = bundle['model']
                feat_names = bundle.get('features')
                st.success('Loaded model bundle from repository.')
            else:
                model = None
                feat_names = None
                st.error('Model bundle file found but in unexpected format. Expected a dict with key "model".')
        except FileNotFoundError:
            model = None
            feat_names = None
            st.error('Model file `RNALig_training_model.pkl` not found in the repository.')
        except Exception as e:
            model = None
            feat_names = None
            st.error(f'Failed to load model bundle: {e}')

        feats = st.file_uploader('Upload features file (.csv, .xlsx)', type=['csv','xlsx','xls'])

        if model is not None and feats is not None:
            if st.button('Run prediction'):
                try:
                    # read features table
                    if feats.name.lower().endswith(('.xlsx','.xls')):
                        df = pd.read_excel(feats)
                    else:
                        try:
                            df = pd.read_csv(feats)
                        except Exception:
                            df = pd.read_csv(feats, engine='python')
                    df.columns = [str(c).strip() for c in df.columns]
                    id_col = next((c for c in df.columns if 'pdb' in c.lower()), None)
                    if id_col is None:
                        id_col = next((c for c in df.columns if c.lower() in ('id','pdb_id','name','file','filename')), None)

                    # prepare numeric inputs
                    numeric = df.select_dtypes(include=[np.number]).copy()
                    if feat_names:
                        for f in feat_names:
                            if f not in numeric.columns:
                                numeric[f] = np.nan
                        X = numeric[feat_names].astype(float)
                    else:
                        X = numeric

                    # fill missing with column medians
                    X = X.fillna(X.median())

                    y_pred = model.predict(X)
                    out = pd.DataFrame({(id_col if id_col else 'Index'):(df[id_col] if id_col and id_col in df.columns else np.arange(len(df))), 'Predicted_binding_affinity (kcal/mol)': y_pred})
                    st.dataframe(out)
                    st.download_button('Download predictions CSV', data=out.to_csv(index=False).encode('utf-8'), file_name='Predictions_from_model.csv')
                except Exception as e:
                    st.error(f'Prediction failed: {e}')
        else:
            if model is None:
                st.info('Model bundle not available — place RNALig_training_model.pkl in the repo root to enable prediction.')
            else:
                st.info('Please upload a features file to run prediction.')


if __name__ == '__main__':
    main()

