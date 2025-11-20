#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNALig Pro - Full RNA–Ligand Feature Extractor (Publication-Grade)
==================================================================

• Processes every PDB/mmCIF in --indir (no skips) and writes one row per PDB.
• Handles huge structures (> 99,999 atoms) using true mmCIF subset fallback.
• Computes RNA, ligand, and complex-level features (biophysically meaningful).
• Fills NaN if any sub-metric fails; never drops a PDB.
• Writes RNA pocket-depth and ligand HTML visualizations (optional).
• Shows a progress bar via tqdm (falls back to simple iteration if not installed).
• Logs status for each file to log.txt.

Example:
python Features_RNALig_Pro_Full.py   --indir Training   --outdir Training/Results   --outcsv All_Features.csv   --viz_rna --viz_ligand   --pocket_cutoff 5.0 --pocket_sasa 0.05 --rna_label_topk 5   --min_heavy 4 --no-require_carbon --keep_ions --cutoff 5.0
"""

import os, sys, math, argparse, tempfile, traceback
from typing import List, Set, Dict, Tuple

import numpy as np
import pandas as pd

from Bio.PDB import (
    PDBParser, MMCIFParser,
    PDBIO, MMCIFIO,
    Select, Model, Chain, Residue, Atom
)
from Bio.PDB.NeighborSearch import NeighborSearch

# tqdm optional; fall back if absent
try:
    from tqdm import tqdm as _tqdm
    def TQDM(x, **k): return _tqdm(x, **k)
except Exception:
    def TQDM(x, **k): return x

# Optional dependencies
_HAS_RDKIT = True
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
except Exception:
    _HAS_RDKIT = False

_HAS_VIENNA = True
try:
    import RNA
except Exception:
    _HAS_VIENNA = False

_HAS_PY3DMOL = True
try:
    import py3Dmol
except Exception:
    _HAS_PY3DMOL = False

_HAS_SCIPY = True
try:
    from scipy.spatial import cKDTree
except Exception:
    _HAS_SCIPY = False

import freesasa


# -------------------- Constants --------------------

AA3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","MSE","SEC","PYL"}
WATERS = {"HOH","WAT"}
IONS   = {"NA","K","MG","MN","CA","ZN","CL","CU","FE","CO","CD","SR"}
NON_LIGAND_MISC = {
    "GOL","EDO","PG4","PEG","PGE","MPD","TRS","MES","BME","DMS","DMSO","IPA","ACT","ACE","ACY",
    "EOH","FMT","FA","HEP","EPE","HEZ","PEG1","PEG2","SO4","PO4","UNL","UNX",
    "NAG","NDG","BGC","MAN","GAL","GLC","FUC","SIA","XYS","BMA"
}
MOD2CANON = {
    "U":"U","PSU":"U","H2U":"U","4SU":"U","5MU":"U","5BU":"U","OMU":"U","UR3":"U",
    "A":"A","1MA":"A","M1A":"A","I6A":"A","RIA":"A","AET":"A","A2M":"A","M6A":"A",
    "G":"G","2MG":"G","M2G":"G","OMG":"G","7MG":"G","G7M":"G","YYG":"G",
    "C":"C","5MC":"C","M5C":"C","OMC":"C","1SC":"C",
    "I":"G","HYP":"A"
}
RNA_NAMES: Set[str] = set(MOD2CANON.keys()) | {"A","C","G","U"}
PHOSPHATE_O_NAMES = {"OP1","OP2","OP3","O1P","O2P","O3P","O5'"}


# -------------------- Helpers --------------------

class _KeepOnly(Select):
    def __init__(self, keep): self.keep=set(keep)
    def accept_residue(self, res): return (res.get_parent().id, res.id) in self.keep

def _is_rna_res(res) -> bool:
    return res.get_resname().strip() in RNA_NAMES

def _res_heavy_hasC(res):
    heavy = 0; hasC = False
    for a in res.get_atoms():
        el = (a.element or a.get_name()[0]).upper()
        if el != "H":
            heavy += 1
            if el == "C":
                hasC = True
    return heavy, hasC

def _com(coords: np.ndarray) -> np.ndarray:
    return coords.mean(axis=0) if (isinstance(coords, np.ndarray) and coords.size) else np.array([np.nan, np.nan, np.nan])

def _com_atoms(atoms):
    arr = np.array([a.coord for a in atoms if a.coord is not None], float)
    return arr.mean(axis=0) if arr.size else np.array([float("inf")]*3)

def detect_rna_chains(structure) -> List[str]:
    chains = []
    for ch in structure.get_chains():
        for res in ch:
            if _is_rna_res(res):
                chains.append(ch.id); break
    return sorted(set(chains))

def chain_to_sequence(structure, chain_id: str) -> str:
    model = structure[0]
    if chain_id not in [c.id for c in model.get_chains()]:
        return ""
    chain = model[chain_id]
    seq=[]
    for res in chain.get_residues():
        if not _is_rna_res(res): continue
        rn = res.get_resname().strip()
        seq.append(MOD2CANON.get(rn, "N"))
    return "".join(seq)

def nucleotide_composition(seq: str):
    counts = {b: seq.count(b) for b in ("A","C","G","U")}
    total = sum(counts.values())
    gc = (counts["G"] + counts["C"]) / total * 100.0 if total else 0.0
    return counts, total, gc

def vienna_mfe(seq: str) -> float:
    if not _HAS_VIENNA or not seq:
        return float("nan")
    _, mfe = RNA.fold(seq.replace("N","U"))
    return float(mfe)

def freesasa_total(path: str) -> float:
    st = freesasa.Structure(path, options={'hetatm': True})
    return float(freesasa.calc(st).totalArea())

def shape_metrics_from_structure_path(path: str):
    parser = MMCIFParser(QUIET=True) if path.lower().endswith(('.cif','.mmcif')) else PDBParser(QUIET=True)
    s = parser.get_structure("R", path)
    coords = [a.coord for a in s.get_atoms() if a.coord is not None]
    if len(coords) == 0:
        return np.nan, np.nan, np.nan, np.nan
    X = np.asarray(coords)
    center = X.mean(axis=0)
    Xc = X - center
    S = (Xc.T @ Xc) / Xc.shape[0]
    S = 0.5*(S+S.T)
    eig = np.linalg.eigvalsh(S)
    l1, l2, l3 = float(eig[2]), float(eig[1]), float(eig[0])
    rg = float(math.sqrt(l1 + l2 + l3))
    asph = float(l1 - 0.5*(l2 + l3))
    acyl = float(l2 - l3)
    kappa2 = 1.0 - 3.0*((l1*l2 + l2*l3 + l3*l1)/((l1+l2+l3)**2 + 1e-12))
    return rg, asph, acyl, float(kappa2)

def _subset_structure(structure, keep:Set[Tuple[str,Tuple]]):
    # Build a new Structure with only residues in keep
    new_model = Model.Model(0)
    chain_map: Dict[str, Chain.Chain] = {}
    for ch in structure.get_chains():
        for res in ch:
            key = (ch.id, res.id)
            if key not in keep:
                continue
            if ch.id not in chain_map:
                nc = Chain.Chain(ch.id)
                new_model.add(nc)
                chain_map[ch.id] = nc
            else:
                nc = chain_map[ch.id]
            nr = Residue.Residue(res.id, res.get_resname(), res.segid)
            for a in res.get_atoms():
                na = Atom.Atom(a.get_name(), a.coord, a.bfactor, a.occupancy,
                               a.altloc, a.fullname, a.serial_number, element=a.element)
                nr.add(na)
            nc.add(nr)
    from Bio.PDB.Structure import Structure
    nS = Structure("SUBSET")
    nS.add(new_model)
    return nS

def _write_subset(structure, keep:Set[Tuple[str,Tuple]], outp:str):
    # Write subset to PDB; if atom count > 99,999, write true mmCIF subset
    atom_count = 0
    for ch in structure.get_chains():
        for res in ch:
            if (ch.id, res.id) in keep:
                atom_count += sum(1 for _ in res.get_atoms())

    if atom_count > 99999:
        subset = _subset_structure(structure, keep)
        cif_path = os.path.splitext(outp)[0] + ".cif"
        io = MMCIFIO(); io.set_structure(subset); io.save(cif_path)
        return cif_path
    else:
        io = PDBIO(); io.set_structure(structure); io.save(outp, select=_KeepOnly(keep))
        return outp

def _depth_to_hex(d:float,dmin:float,dmax:float)->str:
    if not np.isfinite(d): d=dmin
    if dmax<=dmin+1e-9: t=0.0
    else: t=(d-dmin)/(dmax-dmin)
    t=max(0.0,min(1.0,t))
    r=int(255*t); g=0; b=int(255*(1.0-t))
    return f"#{r:02X}{g:02X}{b:02X}"

def render_pocket_depth_html(clean_path, biopy_struct, chain_id, res_depth_map, out_html, topk=5):
    if not _HAS_PY3DMOL:return
    with open(clean_path) as fh: model_str=fh.read()
    fmt = "cif" if clean_path.lower().endswith((".cif",".mmcif")) else "pdb"
    v=py3Dmol.view(width=1100,height=780)
    v.addModel(model_str, fmt)
    v.setBackgroundColor("0xFFFFFF")
    v.setStyle({"chain":chain_id},{"cartoon":{"color":"forestgreen","opacity":0.95}})
    v.addSurface(py3Dmol.VDW,{"opacity":0.15,"color":"0x6BAED6"},{"chain":chain_id})
    if res_depth_map:
        vals=list(res_depth_map.values()); dmin,dmax=min(vals),max(vals)
        model=biopy_struct[0]
        if chain_id not in [c.id for c in model.get_chains()]:
            with open(out_html,"w") as f:f.write(v._make_html()); return
        chain=model[chain_id]
        for res in chain.get_residues():
            if not _is_rna_res(res): continue
            i=res.id[1]
            if i in res_depth_map:
                col=_depth_to_hex(res_depth_map[i],dmin,dmax)
                v.setStyle({"chain":chain_id,"resi":[i]},{"stick":{"radius":0.26,"color":col}})
        deep=sorted(res_depth_map.items(),key=lambda x:x[1],reverse=True)[:max(0,int(topk))]
        for resi,d in deep:
            try:
                res=next(r for r in chain if _is_rna_res(r) and r.id[1]==resi)
                pos = res["P"].coord if "P" in res else next(res.get_atoms()).coord
                v.addLabel(f"{resi} ({d:.2f} A)",
                           {"fontSize":12,"backgroundOpacity":0.7,"backgroundColor":"white","fontColor":"black",
                            "position":{"x":float(pos[0]),"y":float(pos[1]),"z":float(pos[2])}})
            except Exception: pass
        v.addLabel(f"Pocket depth: shallow ({dmin:.2f}) -> deep ({dmax:.2f})",
                   {"fontSize":14,"backgroundOpacity":0.0,"fontColor":"black"})
    v.zoomTo({"chain":chain_id})
    with open(out_html,"w") as f:f.write(v._make_html())


# -------------------- Pocket helpers --------------------

def _atoms_from_residues(structure, chain_id, resi_list: List[int]):
    model = structure[0]
    if chain_id not in [c.id for c in model.get_chains()]:
        return np.zeros((0,3))
    resimap = {res.id[1]: res for res in model[chain_id].get_residues() if _is_rna_res(res)}
    atoms=[]
    for i in resi_list:
        r=resimap.get(i)
        if r is None: continue
        for a in r.get_atoms():
            if a.coord is not None: atoms.append(a.coord)
    return np.asarray(atoms,float) if atoms else np.zeros((0,3))

def _surface_atoms_coords(rna_only_path, sasa_thresh=0.05):
    st=freesasa.Structure(rna_only_path, options={'hetatm': False})
    calc=freesasa.calc(st)
    parser = MMCIFParser(QUIET=True) if rna_only_path.lower().endswith(('.cif','.mmcif')) else PDBParser(QUIET=True)
    s=parser.get_structure("R", rna_only_path)
    allcoords=[a.coord for a in s.get_atoms() if a.coord is not None]
    per_atom_area=[calc.atomArea(i) for i in range(st.nAtoms())]
    surf=[c for c,area in zip(allcoords, per_atom_area) if area>sasa_thresh]
    return np.asarray(surf,float) if surf else np.zeros((0,3))

def pocket_residues_for_ligand(structure, rna_chain_id, ligand_residue, cutoff=5.0):
    model=structure[0]
    if rna_chain_id not in [c.id for c in model.get_chains()]:
        return []
    rna_atoms=[]
    for res in model[rna_chain_id].get_residues():
        if _is_rna_res(res):
            rna_atoms.extend(list(res.get_atoms()))
    if not rna_atoms: return []
    lig_atoms=list(ligand_residue.get_atoms())
    ns=NeighborSearch(rna_atoms)
    resi=set()
    for la in lig_atoms:
        for a in ns.search(la.coord, cutoff):
            rid=a.get_parent().id
            resi.add(int(rid[1]))
    return sorted(resi)

def pocket_depth_and_stats(clean_struct, chain_id, pocket_resi_list, rna_only_path, sasa_thresh=0.05):
    pocket_xyz=_atoms_from_residues(clean_struct, chain_id, pocket_resi_list)
    if pocket_xyz.shape[0]==0:
        return 0, 0, np.nan, np.nan, {}
    surf_xyz=_surface_atoms_coords(rna_only_path, sasa_thresh)
    if surf_xyz.shape[0]==0:
        return len(pocket_resi_list), pocket_xyz.shape[0], np.nan, np.nan, {}
    if _HAS_SCIPY:
        tree=cKDTree(surf_xyz)
        dists,_=tree.query(pocket_xyz,k=1,workers=-1)
    else:
        dists=np.min(np.linalg.norm(pocket_xyz[:,None,:]-surf_xyz[None,:,:],axis=2),axis=1)
    depth_mean=float(np.mean(dists)); depth_max=float(np.max(dists))
    model=clean_struct[0]
    if chain_id not in [c.id for c in model.get_chains()]:
        return len(pocket_resi_list), pocket_xyz.shape[0], depth_mean, depth_max, {}
    chain=model[chain_id]
    coords=[]; res_index=[]
    for res in chain.get_residues():
        if not _is_rna_res(res): continue
        i=res.id[1]
        if i in pocket_resi_list:
            for a in res.get_atoms():
                if a.coord is not None:
                    coords.append(a.coord); res_index.append(i)
    coords=np.asarray(coords,float) if coords else np.zeros((0,3))
    res_mean={}
    if coords.size:
        if _HAS_SCIPY:
            tree=cKDTree(surf_xyz); dd,_=tree.query(coords,k=1,workers=-1)
        else:
            dd=np.min(np.linalg.norm(coords[:,None,:]-surf_xyz[None,:,:],axis=2),axis=2).min(axis=1)
        for di,ri in zip(dd,res_index):
            res_mean.setdefault(ri,[]).append(float(di))
    res_mean={k:(float(np.mean(v)) if v else 0.0) for k,v in res_mean.items()}
    return len(pocket_resi_list), pocket_xyz.shape[0], depth_mean, depth_max, res_mean


# -------------------- Ligand features --------------------

def _asphericity_from_coords(coords: np.ndarray) -> float:
    if coords.shape[0] < 3: return float("nan")
    X = coords - coords.mean(0)
    cov = (X.T @ X) / X.shape[0]
    cov = 0.5*(cov+cov.T)
    vals = np.linalg.eigvalsh(cov)
    l1,l2,l3 = float(vals[0]), float(vals[1]), float(vals[2])
    return l3 - 0.5*(l1 + l2)

def _ligand_from_path(path):
    if not _HAS_RDKIT: return None
    if path.lower().endswith(('.cif','.mmcif')):
        return None
    mol = Chem.MolFromPDBFile(path, removeHs=False, sanitize=True)
    if mol is None:
        mol = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False)
        if mol: Chem.SanitizeMol(mol)
    return mol

def _write_ligand_only(structure, res, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, f"{tag}_ligand.pdb")
    return _write_subset(structure, {(res.get_parent().id, res.id)}, outp)

def ligand_features_for_residue(structure, lig_tuple, outdir, make_viz, lig_viz_subdir=None):
    chain_id, resn, resi, icode, res = lig_tuple
    tag = f"{resn}_{chain_id}{resi}{icode}".replace(" ","")
    target_dir = lig_viz_subdir or outdir

    lig_path = _write_ligand_only(structure, res, target_dir, tag)

    mw=logp=tpsa=hbd=hba=rotb=arom=rg=pmi1=pmi2=pmi3=asph=np.nan
    smiles=""; chiral=0; atoms_total=atoms_heavy=0
    ch_mean=ch_std=ch_min=ch_max=np.nan
    mmff_energy=np.nan

    mol = _ligand_from_path(lig_path) if _HAS_RDKIT else None
    if mol is not None:
        molH = Chem.AddHs(mol, addCoords=True)
        if molH.GetNumConformers() == 0:
            AllChem.EmbedMolecule(molH, randomSeed=42)
        try:
            mp = AllChem.MMFFGetMoleculeProperties(molH, mmffVariant="MMFF94")
            ff = AllChem.MMFFGetMoleculeForceField(molH, mp)
            mmff_energy = float(ff.CalcEnergy())
        except Exception:
            mmff_energy = float("nan")

        mw   = Descriptors.MolWt(molH)
        logp = Descriptors.MolLogP(molH)
        tpsa = Descriptors.TPSA(molH)
        hbd  = Descriptors.NumHDonors(molH)
        hba  = Descriptors.NumHAcceptors(molH)
        rotb = Descriptors.NumRotatableBonds(molH)
        arom = rdMolDescriptors.CalcNumAromaticRings(molH)
        smiles = Chem.MolToSmiles(Chem.RemoveHs(molH), isomericSmiles=True)
        chiral = len(Chem.FindMolChiralCenters(molH, includeUnassigned=True))
        atoms_total = molH.GetNumAtoms()
        atoms_heavy = molH.GetNumHeavyAtoms()

        conf = molH.GetConformer()
        coords = np.array([[conf.GetAtomPosition(i).x,
                            conf.GetAtomPosition(i).y,
                            conf.GetAtomPosition(i).z] for i in range(molH.GetNumAtoms())], float)
        rg  = rdMolDescriptors.CalcRadiusOfGyration(molH)
        pmi1 = rdMolDescriptors.CalcPMI1(molH)
        pmi2 = rdMolDescriptors.CalcPMI2(molH)
        pmi3 = rdMolDescriptors.CalcPMI3(molH)
        asph = _asphericity_from_coords(coords)

        try:
            Chem.rdPartialCharges.ComputeGasteigerCharges(molH)
            charges = [float(a.GetDoubleProp("_GasteigerCharge")) for a in molH.GetAtoms()]
            ch_mean, ch_std = float(np.mean(charges)), float(np.std(charges))
            ch_min, ch_max = float(np.min(charges)), float(np.max(charges))
        except Exception:
            ch_mean = ch_std = ch_min = ch_max = float("nan")

    # FreeSASA (works for both PDB/CIF)
    try:
        fs_struct = freesasa.Structure(lig_path, options={'hetatm': True})
        fs_calc = freesasa.calc(fs_struct)
        sasa_total = float(fs_calc.totalArea())
        per_atom = [fs_calc.atomArea(i) for i in range(fs_struct.nAtoms())]
        elements=[]
        with open(lig_path) as fh:
            for line in fh:
                if line.startswith(("ATOM","HETATM")):
                    el = line[76:78].strip() or line[12:14].strip()
                    elements.append(el.upper())
        sasa_polar = float(sum(a for a,e in zip(per_atom,elements) if e in ("O","N")))
        sasa_nonpolar = float(sum(a for a,e in zip(per_atom,elements) if e not in ("O","N")))
    except Exception:
        sasa_total = sasa_polar = sasa_nonpolar = float("nan")

    # Ligand HTML
    html_path = ""
    if make_viz and _HAS_PY3DMOL:
        try:
            with open(lig_path, "r") as fh:
                model_str = fh.read()
            fmt = "cif" if lig_path.lower().endswith((".cif",".mmcif")) else "pdb"
            view = py3Dmol.view(width=900, height=700)
            view.addModel(model_str, fmt)
            view.setBackgroundColor("0xFFFFFF")
            view.addSurface(py3Dmol.VDW, {"opacity":0.15, "color":"0xBBBBBB"})
            if mol is not None:
                try:
                    Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
                    charges = [float(a.GetDoubleProp("_GasteigerCharge")) for a in mol.GetAtoms()]
                    qmin, qmax = float(np.min(charges)), float(np.max(charges))
                    def _col(q):
                        if not np.isfinite(q): q=0.0
                        t=0.5 if qmax<=qmin+1e-9 else (q-qmin)/(qmax-qmin)
                        t=min(1.0,max(0.0,t))
                        r=int(255*t); g=0; b=int(255*(1.0-t))
                        return f"#{r:02X}{g:02X}{b:02X}"
                    for idx, q in enumerate(charges, start=1):
                        view.setStyle({"serial": idx}, {"stick":{"radius":0.24, "color": _col(q)}})
                    view.addLabel(f"Atom charges: blue (neg) -> red (pos) [{qmin:.2f} -> {qmax:.2f}]",
                                  {"fontSize":14,"backgroundOpacity":0.7,"backgroundColor":"white","fontColor":"black"})
                except Exception:
                    view.setStyle({}, {"stick":{"radius":0.24}})
            else:
                view.setStyle({}, {"stick":{"radius":0.24}})
            view.zoomTo()
            html_path = os.path.join(target_dir, f"{tag}_ligand.html")
            with open(html_path,"w") as f:
                f.write(view._make_html())
        except Exception:
            html_path = ""

    return {
        "Ligand_chain": chain_id, "Ligand_resname": resn, "Ligand_resseq": resi, "Ligand_icode": icode or "",
        "Ligand_SMILES": smiles,
        "Ligand_MW_Da": mw, "Ligand_LogP": logp, "Ligand_TPSA_A2": tpsa,
        "Ligand_HBD": hbd, "Ligand_HBA": hba, "Ligand_RotBonds": rotb, "Ligand_AromaticRings": arom, "Ligand_ChiralCenters": chiral,
        "Ligand_AtomCount_total": atoms_total, "Ligand_AtomCount_heavy": atoms_heavy,
        "Ligand_Rg_A": rg, "Ligand_PMI1": pmi1, "Ligand_PMI2": pmi2, "Ligand_PMI3": pmi3,
        "Ligand_Asphericity": asph,
        "Ligand_Charge_mean": ch_mean, "Ligand_Charge_std": ch_std, "Ligand_Charge_min": ch_min, "Ligand_Charge_max": ch_max,
        "Ligand_SASA_total_A2": sasa_total, "Ligand_SASA_polar_A2": sasa_polar, "Ligand_SASA_nonpolar_A2": sasa_nonpolar,
        "Ligand_MMFF94_Energy_kcal_mol": mmff_energy,
        "Ligand_File": os.path.basename(lig_path),
        "Ligand_HTML": os.path.basename(html_path) if html_path else ""
    }


# -------------------- Complex features --------------------

def _contacts_core(rna_atoms, lig_atoms, cutoff=4.0):
    d=[]
    for a in lig_atoms:
        if (a.element or a.get_name()[0]).upper()=="H": continue
        for b in rna_atoms:
            if (b.element or b.get_name()[0]).upper()=="H": continue
            dist=np.linalg.norm(a.coord-b.coord)
            if dist<=cutoff: d.append(dist)
    if d:
        arr=np.asarray(d,float)
        return len(arr), float(arr.mean()), float(arr.std())
    return 0, np.nan, np.nan

def _vdw_shell(rna_atoms, lig_atoms, rmin=4.0, rmax=6.0):
    d=[]
    for a in lig_atoms:
        if (a.element or a.get_name()[0]).upper()=="H": continue
        for b in rna_atoms:
            if (b.element or b.get_name()[0]).upper()=="H": continue
            dist=np.linalg.norm(a.coord-b.coord)
            if rmin<=dist<=rmax: d.append(dist)
    if d:
        arr=np.asarray(d,float)
        return len(arr), float(arr.mean()), float(arr.std())
    return 0, np.nan, np.nan

def _hb_pairs(rna_atoms, lig_atoms, cutoff=3.5):
    out=[]
    for a in lig_atoms:
        if (a.element or a.get_name()[0]).upper() not in ("N","O"): continue
        for b in rna_atoms:
            if (b.element or b.get_name()[0]).upper() not in ("N","O"): continue
            if np.linalg.norm(a.coord-b.coord)<=cutoff:
                out.append((a,b))
    return out

def _hydroph_pairs(rna_atoms, lig_atoms, cutoff=4.5):
    out=[]
    for a in lig_atoms:
        if (a.element or a.get_name()[0]).upper()!="C": continue
        for b in rna_atoms:
            if (b.element or b.get_name()[0]).upper()!="C": continue
            if np.linalg.norm(a.coord-b.coord)<=cutoff:
                out.append((a,b))
    return out

def _rna_targets(structure, mode="phosphate"):
    atoms=[]
    for ch in structure.get_chains():
        for res in ch:
            if not _is_rna_res(res): continue
            if mode in ("phosphate","all"):
                for an,at in res.child_dict.items():
                    if an.strip().upper() in PHOSPHATE_O_NAMES:
                        atoms.append(at)
            if mode in ("bases","all"):
                for an,at in res.child_dict.items():
                    el=(at.element or at.get_name()[0]).upper()
                    if el in ("N","O"):
                        atoms.append(at)
    return atoms

def _electrostatics_charged_count(lig_path, structure, dmin, dmax, qthr, include_negative, target_mode):
    if not _HAS_RDKIT:
        return 0
    mol=_ligand_from_path(lig_path)
    if mol is None or mol.GetNumConformers()==0: return 0
    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        charges=[float(a.GetDoubleProp("_GasteigerCharge")) for a in mol.GetAtoms()]
    except Exception:
        return 0

    conf=mol.GetConformer()
    lig_pos=np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                      for i in range(mol.GetNumAtoms())], float)
    pos_mask=[q>=qthr for q in charges]
    neg_mask=[q<=-qthr for q in charges] if include_negative else [False]*len(charges)
    tgt = _rna_targets(structure, target_mode)
    if not tgt: return 0
    tgt_pos=np.array([a.coord for a in tgt], float)

    dists_count=0
    for i,use in enumerate(pos_mask):
        if use:
            dv=np.linalg.norm(tgt_pos - lig_pos[i], axis=1)
            sel=(dv>=dmin)&(dv<=dmax)
            dists_count += int(np.count_nonzero(sel))
    for i,use in enumerate(neg_mask):
        if use:
            dv=np.linalg.norm(tgt_pos - lig_pos[i], axis=1)
            sel=(dv>=dmin)&(dv<=dmax)
            dists_count += int(np.count_nonzero(sel))
    return int(dists_count)

def _electrostatics_legacy_count(rna_atoms, lig_atoms, cutoff=5.0):
    d_count=0
    rna_N=[a for a in rna_atoms if (a.element or a.get_name()[0]).upper()=="N"]
    lig_O=[a for a in lig_atoms if (a.element or a.get_name()[0]).upper()=="O"]
    for a in rna_N:
        for b in lig_O:
            dist=np.linalg.norm(a.coord-b.coord)
            if dist<=cutoff: d_count += 1
    return int(d_count)

def _sasa_total(path):
    fs=freesasa.Structure(path, options={'hetatm':True})
    return float(freesasa.calc(fs).totalArea())

def _buried_surface_area_A2(pC,pR,pL):
    return max(0.0, _sasa_total(pR)+_sasa_total(pL)-_sasa_total(pC))

def _residue_centroid(res):
    pts=[a.coord for a in res.get_atoms()]
    return np.mean(pts,axis=0) if pts else np.array([np.nan,np.nan,np.nan])

def _dbscan_clusters(X, eps=6.0):
    n=len(X)
    if n==0: return []
    visited=[False]*n; labels=[-1]*n; cid=0
    for i in range(n):
        if visited[i]: continue
        visited[i]=True
        d=np.linalg.norm(X-X[i],axis=1); neigh=np.where(d<=eps)[0].tolist()
        labels[i]=cid; queue=neigh[:]
        while queue:
            j=queue.pop()
            if not visited[j]:
                visited[j]=True
                dj=np.linalg.norm(X-X[j],axis=1); neigh_j=np.where(dj<=eps)[0].tolist()
                queue.extend([k for k in neigh_j if labels[k]==-1])
            labels[j]=cid
        cid+=1
    groups={}
    for idx,l in enumerate(labels): groups.setdefault(l,[]).append(idx)
    return list(groups.values())

def complex_features_for_ligand(structure, lig_res, args, outdir):
    rna_atoms=[a for ch in structure.get_chains() for r in ch if _is_rna_res(r) for a in r.get_atoms()]
    lig_atoms=list(lig_res.get_atoms())

    c_total, c_mean, c_std = _contacts_core(rna_atoms, lig_atoms, cutoff=args.cutoff)
    if args.vdw_mode == "legacy":
        vdw_cnt, vdw_mean, vdw_std = _contacts_core(rna_atoms, lig_atoms, cutoff=args.vdw_legacy_cutoff)
    else:
        vdw_cnt, vdw_mean, vdw_std = _vdw_shell(rna_atoms, lig_atoms, rmin=4.0, rmax=6.0)

    hbonds = _hb_pairs(rna_atoms, lig_atoms, cutoff=args.hbond_cutoff)
    hydros = _hydroph_pairs(rna_atoms, lig_atoms, cutoff=args.hydroph_cutoff)

    ns=NeighborSearch(rna_atoms)
    pocket_atoms=[]; seen=set()
    for a in lig_atoms:
        for b in ns.search(a.coord,5.0):
            if b.serial_number not in seen:
                pocket_atoms.append(b); seen.add(b.serial_number)
    pocket_residues=sorted({a.get_parent() for a in pocket_atoms if _is_rna_res(a.get_parent())},
                           key=lambda r:(r.get_parent().id, r.id[1], r.id[2]))

    contact_res: Set[Tuple[str,int,str]] = set()
    for a in lig_atoms:
        for b in ns.search(a.coord,args.cutoff):
            res=b.get_parent()
            if _is_rna_res(res):
                ch=res.get_parent().id
                resi=res.id[1]; icode=res.id[2] if len(res.id)>2 else ""
                contact_res.add((ch,resi,icode))
    contact_ids = ";".join([f"{c}:{i}{ic if ic else ''}" for (c, i, ic) in sorted(contact_res, key=lambda x:(x[0],x[1],x[2]))])
    n_contact_res=len(contact_res)

    if pocket_residues:
        cents=np.array([_residue_centroid(r) for r in pocket_residues],float)
        clusters=_dbscan_clusters(cents, eps=6.0)
        n_sites=len(clusters)
    else:
        n_sites=0

    tmp=tempfile.mkdtemp()
    lig_path=os.path.join(tmp,"ligand.pdb")
    rna_path=os.path.join(tmp,"rna.pdb")
    complex_path=os.path.join(tmp,"complex.pdb")
    _write_subset(structure, {(lig_res.get_parent().id, lig_res.id)}, lig_path)
    keep_rna={(r.get_parent().id, r.id) for ch in structure.get_chains() for r in ch if _is_rna_res(r)}
    _write_subset(structure, keep_rna, rna_path)
    _write_subset(structure, keep_rna | {(lig_res.get_parent().id, lig_res.id)}, complex_path)

    if args.elec_mode == "legacy":
        elec_cnt = _electrostatics_legacy_count(rna_atoms, lig_atoms, cutoff=args.vdw_legacy_cutoff)
    else:
        elec_cnt = _electrostatics_charged_count(
            lig_path, structure,
            dmin=args.elec_dmin, dmax=args.elec_dmax,
            qthr=args.elec_qthr, include_negative=args.elec_include_negative,
            target_mode=args.elec_targets
        )

    try:
        bsa = _buried_surface_area_A2(complex_path, rna_path, lig_path)
    except Exception:
        bsa = np.nan

    lig_coords=np.array([a.coord for a in lig_atoms],float)
    pocket_coords=np.array([a.coord for a in pocket_atoms],float)
    lig_com = _com(lig_coords)
    pocket_com = _com(pocket_coords)
    d_com = float(np.linalg.norm(lig_com - pocket_com)) if pocket_coords.size else np.nan

    return {
        "Complex_contacts_le_4A": c_total, "Complex_contacts_mean_A": c_mean, "Complex_contacts_std_A": c_std,
        "Complex_vdw_count": vdw_cnt, "Complex_vdw_mean_A": vdw_mean, "Complex_vdw_std_A": vdw_std,
        "Complex_hbond_count": len(hbonds), "Complex_hydrophobic_count": len(hydros),
        "Complex_electrostatic_count": elec_cnt,
        "Complex_BSA_A2": bsa,
        "Complex_pocket_COM_to_ligand_COM_A": d_com,
        "Complex_n_contact_residues": n_contact_res,
        "Complex_contact_residue_ids": contact_ids,
        "Complex_n_binding_sites": n_sites
    }


# -------------------- Assignment + IO --------------------

def chain_contact_stats_with_ligand(structure, chain_id, lig_residue, cutoff=4.0):
    model = structure[0]
    if chain_id not in [c.id for c in model.get_chains()]:
        return (0,0,float('inf'),0.0,float('inf'))
    rna_atoms = []
    for res in model[chain_id].get_residues():
        if _is_rna_res(res):
            rna_atoms.extend(list(res.get_atoms()))
    if not rna_atoms: return (0,0,float('inf'),0.0,float('inf'))
    lig_atoms = list(lig_residue.get_atoms())
    if not lig_atoms: return (0,0,float('inf'),0.0,float('inf'))

    ns = NeighborSearch(rna_atoms)
    res_contacts = set(); atom_contacts = 0; min_d = float('inf'); lig_atoms_with_contacts = set()
    for li, la in enumerate(lig_atoms):
        neigh = ns.search(la.coord, cutoff)
        if neigh:
            atom_contacts += len(neigh)
            lig_atoms_with_contacts.add(li)
            for a in neigh:
                res_contacts.add(a.get_parent().id)
        for a in rna_atoms:
            d = la - a
            if d < min_d: min_d = d

    com_chain = _com_atoms(rna_atoms); com_lig = _com_atoms(lig_atoms)
    com_distance = float(np.linalg.norm(com_chain - com_lig))
    frac_lig_atoms = len(lig_atoms_with_contacts)/len(lig_atoms) if lig_atoms else 0.0
    return (len(res_contacts), atom_contacts, min_d, frac_lig_atoms, com_distance)

def pick_primary_chain_for_ligand(structure, candidate_chains, lig_residue,
                                  cutoff=4.0, min_res_contacts=2, min_frac_lig_atoms=0.2):
    best_key = None; best_chain = None
    nearest_chain = None; nearest_min_d = float('inf')
    for ch in candidate_chains:
        nres, natm, mind, frac, comd = chain_contact_stats_with_ligand(structure, ch, lig_residue, cutoff)
        if mind < nearest_min_d:
            nearest_min_d, nearest_chain = mind, ch
        key = (nres, frac, natm, -mind, -comd)
        if (best_key is None) or (key > best_key):
            best_key, best_chain = key, ch
    if best_chain is None:
        return None
    nres, natm, mind, frac, comd = chain_contact_stats_with_ligand(structure, best_chain, lig_residue, cutoff)
    if (nres >= min_res_contacts) or (frac >= min_frac_lig_atoms):
        return best_chain
    return nearest_chain

def _load_structure(path):
    parser = MMCIFParser(QUIET=True) if path.lower().endswith(('.cif','.mmcif')) else PDBParser(QUIET=True)
    return parser.get_structure(os.path.basename(path), path)

def detect_contacting_ligands(structure, rna_atoms, cutoff=4.0, min_heavy=8,
                              require_carbon=True, keep_ions=False):
    ns = NeighborSearch(rna_atoms) if rna_atoms else None
    ligs = []
    for ch in structure.get_chains():
        for res in ch:
            rn = res.get_resname().strip()
            if rn in RNA_NAMES or rn in AA3 or rn in WATERS or rn in NON_LIGAND_MISC:
                continue
            if rn in IONS and not keep_ions:
                continue
            heavy, hasC = _res_heavy_hasC(res)
            if heavy < min_heavy: continue
            if require_carbon and not hasC: continue
            if ns is not None and not any(ns.search(a.coord, cutoff) for a in res.get_atoms()):
                continue
            ligs.append((ch.id, rn, res.id[1], res.id[2] if len(res.id)>2 else "", res))
    return ligs

def process_one_pdb(path: str, args):
    s = _load_structure(path)

    all_rna_chains = detect_rna_chains(s)
    rna_atoms_all = [a for ch in s.get_chains() for r in ch if _is_rna_res(r) for a in r.get_atoms()]
    ligs = detect_contacting_ligands(s, rna_atoms_all, cutoff=args.cutoff,
                                     min_heavy=args.min_heavy, require_carbon=args.require_carbon,
                                     keep_ions=args.keep_ions)

    # One-ligand-per-PDB: if >1 found, pick the one with most contacts
    chosen = None
    if len(ligs) == 0:
        chosen = None
    elif len(ligs) == 1:
        chosen = ligs[0]
    else:
        best = None; best_contacts = -1
        for L in ligs:
            contacts = 0
            lig_atoms=list(L[-1].get_atoms())
            ns=NeighborSearch(rna_atoms_all)
            for a in lig_atoms:
                contacts += len(ns.search(a.coord, args.cutoff))
            if contacts > best_contacts:
                best_contacts = contacts; best = L
        chosen = best

    # Clean subset structure (RNA + chosen ligand if present)
    ligand_keep = {(chosen[-1].get_parent().id, chosen[-1].id)} if chosen else set()
    rna_keep = {(r.get_parent().id, r.id) for ch in s.get_chains() for r in ch if _is_rna_res(r)}
    keep_set = ligand_keep | rna_keep
    base_clean = os.path.join(args.outdir, os.path.splitext(os.path.basename(path))[0] + "_clean.pdb")
    clean_path = _write_subset(s, keep_set, base_clean)
    sc = _load_structure(clean_path)  # re-parse

    # Assign RNA chain to ligand (if any)
    assigned_chain = None
    if chosen is not None:
        assigned_chain = pick_primary_chain_for_ligand(
            s, all_rna_chains, chosen[-1], cutoff=args.cutoff,
            min_res_contacts=2, min_frac_lig_atoms=0.2
        )
    if assigned_chain is None and all_rna_chains:
        assigned_chain = all_rna_chains[0]

    # RNA-only file (for SASA & shape)
    rna_only_tmp = tempfile.mkdtemp()
    rna_only = os.path.join(rna_only_tmp, "rna_only.pdb")
    if assigned_chain and (assigned_chain in [c.id for c in sc[0].get_chains()]):
        keep = set((assigned_chain, res.id) for res in sc[0][assigned_chain] if _is_rna_res(res))
        _write_subset(sc, keep, rna_only)
    else:
        open(rna_only,"w").close()

    # RNA features
    seq = chain_to_sequence(s, assigned_chain) if assigned_chain else ""
    comp, length, gc = nucleotide_composition(seq)
    mfe_val = vienna_mfe(seq)
    try: sasa_total = freesasa_total(rna_only)
    except Exception: sasa_total = float("nan")
    rg, asph, acyl, kappa2 = shape_metrics_from_structure_path(rna_only)

    # Pocket depth & stats
    pocket_res_all=set()
    if chosen is not None and assigned_chain is not None:
        pocket_res_all.update(pocket_residues_for_ligand(sc, assigned_chain, chosen[-1], cutoff=args.pocket_cutoff))

    n_res, n_atoms, d_mean, d_max, res_depth_map = pocket_depth_and_stats(
        sc, assigned_chain, sorted(pocket_res_all), rna_only, sasa_thresh=args.pocket_sasa
    ) if assigned_chain else (0,0,np.nan,np.nan,{})

    # RNA HTML
    if args.viz_rna and _HAS_PY3DMOL and res_depth_map and assigned_chain:
        base = os.path.splitext(os.path.basename(path))[0]
        os.makedirs(args.rna_viz_dir or args.outdir, exist_ok=True)
        out_html = os.path.join(args.rna_viz_dir or args.outdir, f"{base}_chain{assigned_chain}_pocket.html")
        render_pocket_depth_html(clean_path, sc, assigned_chain, res_depth_map, out_html, topk=args.rna_label_topk)

    # Ligand + complex features (if ligand present)
    lig_feats = {}; cx_feats = {}
    if chosen is not None:
        try:
            lig_feats = ligand_features_for_residue(s, chosen, args.outdir, args.viz_ligand, lig_viz_subdir=args.lig_viz_dir)
        except Exception as e:
            sys.stderr.write(f"[Ligand] {os.path.basename(path)}: {e}\n")
            lig_feats = {}
        try:
            cx_feats = complex_features_for_ligand(s, chosen[-1], args, args.outdir)
        except Exception as e:
            sys.stderr.write(f"[Complex] {os.path.basename(path)}: {e}\n")
            cx_feats = {}

    # One row per PDB
    base_row = {
        "PDB_ID": os.path.basename(path),
        "Assigned_RNA_chain": assigned_chain or "",
        "Ligand_tag": (f"{chosen[1]}_{chosen[0]}{chosen[2]}{chosen[3] or ''}".replace(' ','')
                       if chosen else ""),
        "RNA_length": length,
        "RNA_count_A": comp.get("A",0),
        "RNA_count_C": comp.get("C",0),
        "RNA_count_G": comp.get("G",0),
        "RNA_count_U": comp.get("U",0),
        "RNA_GC_percent": round(gc,3),
        "RNA_MFE_kcal_mol": mfe_val,
        "RNA_SASA_total_A2": sasa_total,
        "RNA_Rg_A": rg,
        "RNA_asphericity": asph,
        "RNA_acylindricity": acyl,
        "RNA_kappa2": kappa2,
        "RNA_pocket_res_count": n_res,
        "RNA_pocket_atom_count": n_atoms,
        "RNA_pocket_depth_mean_A": d_mean,
        "RNA_pocket_depth_max_A": d_max
    }
    row = {**base_row, **lig_feats, **cx_feats}
    return row

def iter_structs(indir):
    for fn in sorted(os.listdir(indir)):
        if fn.lower().endswith((".pdb",".cif",".mmcif")):
            yield os.path.join(indir, fn)


# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="RNALig Pro - Full Feature Extractor (one row per PDB)")
    ap.add_argument("--pdb", help="Single PDB/mmCIF file")
    ap.add_argument("--indir", help="Directory of PDB/mmCIF files")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--outcsv", default="All_Features.csv", help="Unified CSV filename")

    # Ligand detection
    ap.add_argument("--cutoff", type=float, default=4.0, help="RNA–ligand heavy-atom contact cutoff (A)")
    ap.add_argument("--min_heavy", type=int, default=8, help="Minimum heavy atoms to treat residue as ligand")
    ap.add_argument("--require_carbon", dest="require_carbon", action="store_true"); ap.set_defaults(require_carbon=True)
    ap.add_argument("--no-require_carbon", dest="require_carbon", action="store_false")
    ap.add_argument("--keep_ions", action="store_true", help="Include ions as ligands (off by default)")

    # Interaction metrics
    ap.add_argument("--vdw_mode", choices=["shell","legacy"], default="shell", help="vdW metric")
    ap.add_argument("--vdw_legacy_cutoff", type=float, default=4.0, help="Legacy vdW cutoff (A)")
    ap.add_argument("--hbond_cutoff", type=float, default=3.5, help="H-bond N/O···N/O distance cutoff (A)")
    ap.add_argument("--hydroph_cutoff", type=float, default=4.5, help="Hydrophobic C···C contact cutoff (A)")

    # Electrostatics
    ap.add_argument("--elec_mode", choices=["charged","legacy"], default="charged", help="Electrostatics mode")
    ap.add_argument("--elec_targets", choices=["phosphate","bases","all"], default="phosphate", help="RNA target atoms")
    ap.add_argument("--elec_qthr", type=float, default=0.2, help="|q| threshold for charged atom selection (Gasteiger)")
    ap.add_argument("--elec_dmin", type=float, default=3.0, help="Electrostatics min distance (A)")
    ap.add_argument("--elec_dmax", type=float, default=10.0, help="Electrostatics max distance (A)")
    ap.add_argument("--elec_include_negative", action="store_true", help="Also count negative charges (default off)")

    # Visualization
    ap.add_argument("--viz_rna", action="store_true", help="Write RNA pocket HTMLs with legend/labels")
    ap.add_argument("--viz_ligand", action="store_true", help="Write per-ligand charge-colored HTMLs")
    ap.add_argument("--viz_complex", action="store_true", help="(reserved)")

    # Pocket fine-tuning
    ap.add_argument("--pocket_cutoff", type=float, default=5.0, help="Residue inclusion cutoff around ligand (A)")
    ap.add_argument("--pocket_sasa", type=float, default=0.05, help="Per-atom SASA threshold for surface (A^2)")
    ap.add_argument("--rna_label_topk", type=int, default=5, help="Top-K deepest residues to label")

    args = ap.parse_args()
    if not args.pdb and not args.indir:
        raise SystemExit("Provide --pdb or --indir")

    os.makedirs(args.outdir, exist_ok=True)
    args.lig_viz_dir = os.path.join(args.outdir, "Ligand") if args.viz_ligand else None
    args.rna_viz_dir = os.path.join(args.outdir, "RNA") if args.viz_rna else None
    if args.lig_viz_dir: os.makedirs(args.lig_viz_dir, exist_ok=True)
    if args.rna_viz_dir: os.makedirs(args.rna_viz_dir, exist_ok=True)

    paths = [args.pdb] if args.pdb else list(iter_structs(args.indir))

    all_rows = []
    log_lines = []
    print(f"Processing {len(paths)} complexes ...\n")

    for p in TQDM(paths, desc="Extracting", unit="pdb"):
        status="OK"
        try:
            row = process_one_pdb(p, args)
        except Exception as e:
            status=f"ERROR: {e.__class__.__name__}"
            row={"PDB_ID": os.path.basename(p)}
            # Fill common columns to keep CSV stable
            safe_cols = [
                "Assigned_RNA_chain","Ligand_tag","RNA_length","RNA_count_A","RNA_count_C","RNA_count_G","RNA_count_U",
                "RNA_GC_percent","RNA_MFE_kcal_mol","RNA_SASA_total_A2","RNA_Rg_A","RNA_asphericity","RNA_acylindricity",
                "RNA_kappa2","RNA_pocket_res_count","RNA_pocket_atom_count","RNA_pocket_depth_mean_A","RNA_pocket_depth_max_A"
            ]
            for k in safe_cols: row.setdefault(k, np.nan)
            sys.stderr.write(f"[{os.path.basename(p)}] {e}\n")
            traceback.print_exc(limit=1)
        all_rows.append(row)
        log_lines.append(f"{os.path.basename(p)}\t{status}")

    df = pd.DataFrame(all_rows)
    cols = ["PDB_ID"] + [c for c in df.columns if c != "PDB_ID"]
    df = df[cols]

    out_csv_path = os.path.join(args.outdir, args.outcsv)
    df.to_csv(out_csv_path, index=False)

    with open(os.path.join(args.outdir, "log.txt"), "w") as f:
        f.write("\n".join(log_lines))

    print("\nDone.")
    print(f"CSV: {out_csv_path}")
    print(f"Log: {os.path.join(args.outdir, 'log.txt')}")

if __name__ == "__main__":
    main()
