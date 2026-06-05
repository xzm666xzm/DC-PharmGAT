#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pharmacophore feature extraction script (Pharmacophore Feature Extraction)

success text: 
- support reading from mol2 files or SMILES file(.ism/.smi)read molecule
- define key pharmacophore features: hydrogen-bond donor, hydrogen-bond acceptor, aromatic ring, positive charge center, negative charge center, hydrophobic group
- generate Target Pharmacophore Profile (TPP) vector
- save as JSON and.pt file

usage: 
    # mol2 file
    python extract_pharmacophore.py --input path/to/crystal_ligand.mol2 --output output_dir
    
    # SMILES file (for cdk2 target)
    python extract_pharmacophore.py --input dude/cdk2/actives_final.ism --output cdk2 --name cdk2
"""

import argparse
import json
import os
from collections import defaultdict

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


def _get_ascii_temp_dir():
    """get single ASCII path of details directory, use RDKit file read"""
    import tempfile
    # prefer the default temporary directory(result details ASCII)
    default_tmp = tempfile.gettempdir()
    try:
        default_tmp.encode('ascii')
        return default_tmp
    except UnicodeEncodeError:
        pass
    # Windows details use C:/temp as a fallback
    for fallback in ['C:/temp/rdkit_tmp', 'D:/temp/rdkit_tmp']:
        try:
            os.makedirs(fallback, exist_ok=True)
            return fallback
        except OSError:
            continue
    # final fallback
    return default_tmp


def load_mol2(mol2_path):
    """load mol2 file(support non-ASCII paths, process kekulize error)"""
    import shutil
    
    # normalize the path(handle mixed slash separators)
    mol2_path = os.path.normpath(mol2_path)
    
    # text test path details ASCII details(also check the absolute path)
    needs_tmp = False
    try:
        mol2_path.encode('ascii')
        # details path can details ASCII, text the absolute path may contain non-ASCII directories
        os.path.abspath(mol2_path).encode('ascii')
    except UnicodeEncodeError:
        needs_tmp = True
    
    tmp_path = None
    safe_path = mol2_path
    
    if needs_tmp:
        # path information ASCII details, copy details ASCII details directory
        tmp_dir = _get_ascii_temp_dir()
        tmp_path = os.path.join(tmp_dir, 'ligand_tmp.mol2')
        shutil.copy2(mol2_path, tmp_path)
        safe_path = tmp_path

    
    try:
        mol = Chem.MolFromMol2File(safe_path, removeHs=False)
        if mol is None:
            mol = Chem.MolFromMol2File(safe_path, removeHs=True)
        if mol is None:
            # text test sanitize=False + details kekulize(solve aromatic bond kekulize failed issue)
            mol = _load_mol2_skip_kekulize(safe_path)
        if mol is None:
            mol = load_mol2_with_cleanup(mol2_path)
        if mol is None:
            # details process after of file details test details kekulize
            mol = _load_mol2_cleanup_skip_kekulize(mol2_path)
    except OSError:
        # RDKit C++ information Bad input file, text test summary file load
        if tmp_path is None:
            tmp_dir = _get_ascii_temp_dir()
            tmp_path = os.path.join(tmp_dir, 'ligand_tmp.mol2')
            shutil.copy2(mol2_path, tmp_path)
        mol = Chem.MolFromMol2File(tmp_path, removeHs=False)
        if mol is None:
            mol = Chem.MolFromMol2File(tmp_path, removeHs=True)
        if mol is None:
            mol = _load_mol2_skip_kekulize(tmp_path)
        if mol is None:
            mol = load_mol2_with_cleanup(mol2_path)
        if mol is None:
            mol = _load_mol2_cleanup_skip_kekulize(mol2_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    
    # summary: mol2 file middle extract summary, text test use SMILES text build molecule
    if mol is None:
        mol = _rebuild_mol_from_mol2(mol2_path)
    
    if mol is None:
        raise ValueError(f"none load molecule file: {mol2_path}")
    return mol




def _rebuild_mol_from_mol2(mol2_path):

    """

    summary: mol2 file details atom details and details, text test use DetermineBonds text build molecule. 

    use mol2 file bond type text attention text error(text AA2AR), details unable to sanitize of details. 

    """

    try:

        atoms = []

        coords = []

        in_atom_section = False

        

        with open(mol2_path, 'r') as f:

            for line in f:

                line_stripped = line.strip()

                if line_stripped.startswith('@<TRIPOS>ATOM'):

                    in_atom_section = True

                    continue

                elif line_stripped.startswith('@<TRIPOS>'):

                    in_atom_section = False

                    continue

                

                if in_atom_section and line_stripped:

                    parts = line_stripped.split()

                    if len(parts) >= 6:

                        atom_type = parts[5]

                        element = atom_type.split('.')[0]

                        if element == 'H':

                            continue

                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])

                        atoms.append(element)

                        coords.append((x, y, z))

        

        if not atoms:

            return None

        

        # details1: use RDKit DetermineBonds details bond details

        try:

            from rdkit.Chem import rdDetermineBonds

            from rdkit.Geometry import Point3D

            

            rwmol = Chem.RWMol()

            conf = Chem.Conformer(len(atoms))

            

            for i, (elem, (x, y, z)) in enumerate(zip(atoms, coords)):

                atom = Chem.Atom(elem)

                rwmol.AddAtom(atom)

                conf.SetAtomPosition(i, Point3D(x, y, z))

            

            rwmol.AddConformer(conf, assignId=True)

            rdDetermineBonds.DetermineBonds(rwmol, charge=0)

            

            mol = rwmol.GetMol()

            try:

                Chem.SanitizeMol(mol)

            except Exception:

                try:

                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)

                except Exception:

                    pass

            

            print(f'  [rebuild] DetermineBonds succeeded (heavy atoms: {len(atoms)})')

            return mol

        except (ImportError, AttributeError):

            pass

        except Exception as e:

            print(f'  [rebuild] DetermineBonds failed: {e}')

        

        # details2: load text sanitize of molecule, use pharmacophore features extract

        mol = Chem.MolFromMol2File(mol2_path, removeHs=True, sanitize=False)

        if mol is not None:

            print('  [rebuild] Using unsanitized mol for pharmacophore extraction')

            return mol

        

        return None

    except Exception as e:

        print(f'  [rebuild] Failed: {e}')

        return None





def _load_mol2_skip_kekulize(path):
    """
    load mol2 file, details kekulize step. 
    details mol2 file(text FEN1)of aromatic bond text attention summary RDKit kekulize failed, 
    molecule information have effect of, details kekulize after can positive details use. 
    """
    try:
        mol = Chem.MolFromMol2File(path, removeHs=False, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                return mol
            except Exception:
                # details check lossed(text AA2AR of bond type text note error details), 
                # text test details kekulize + properties
                try:
                    mol2 = Chem.MolFromMol2File(path, removeHs=False, sanitize=False)
                    if mol2 is not None:
                        Chem.SanitizeMol(mol2, sanitizeOps=(
                            Chem.SanitizeFlags.SANITIZE_ALL 
                            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE 
                            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
))
                        return mol2
                except Exception:
                    pass
    except Exception:
        pass
    return None


def _load_mol2_cleanup_skip_kekulize(mol2_path):
    """
    process mol2 file in <0> details base information after, details kekulize load. 
    """
    import re
    try:
        with open(mol2_path, 'r') as f:
            content = f.read()
        content = re.sub(r'\s+<\d+>\s+', ' LIG ', content)
        tmp_dir = _get_ascii_temp_dir()
        tmp_path = os.path.join(tmp_dir, 'ligand_cleanup_nk.mol2')
        with open(tmp_path, 'w') as f:
            f.write(content)
        mol = Chem.MolFromMol2File(tmp_path, removeHs=False, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                return mol
            except Exception:
                # details check lossed, details kekulize + properties
                try:
                    mol2 = Chem.MolFromMol2File(tmp_path, removeHs=False, sanitize=False)
                    if mol2 is not None:
                        Chem.SanitizeMol(mol2, sanitizeOps=(
                            Chem.SanitizeFlags.SANITIZE_ALL 
                            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE 
                            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
))
                        return mol2
                except Exception:
                    pass
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    except Exception:
        pass
    return None


def load_mol2_with_cleanup(mol2_path):
    """
    process mol2 file in issue format after load
    process text <0> details of nitro group information
    """
    import re
    
    try:
        with open(mol2_path, 'r') as f:
            content = f.read()
        
        # process nitro group information <0>, <1> text
        content = re.sub(r'\s+<\d+>\s+', ' LIG ', content)
        
        # create details file(use ASCII path)
        tmp_dir = _get_ascii_temp_dir()
        tmp_path = os.path.join(tmp_dir, 'ligand_cleanup.mol2')
        with open(tmp_path, 'w') as f:
            f.write(content)
        
        # text test load process after of file
        mol = Chem.MolFromMol2File(tmp_path, removeHs=False)
        if mol is None:
            mol = Chem.MolFromMol2File(tmp_path, removeHs=True)
        
        # remove details file
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        
        return mol
    except Exception as e:
        print(f"process mol2 file details error: {e}")
        return None


def load_smiles_file(smiles_path):
    """
    load SMILES file (.ism text.smi)
    format: SMILES ID NAME (split text)
    return molecule list
    """
    molecules = []
    with open(smiles_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                smiles = parts[0]
                mol_id = parts[1] if len(parts) > 1 else f"mol_{len(molecules)}"
                mol_name = parts[2] if len(parts) > 2 else mol_id
                
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mol.SetProp("_Name", mol_name)
                    mol.SetProp("_ID", mol_id)
                    molecules.append(mol)
    
    return molecules


def load_molecules(input_path):
    """data file information load molecule"""
    ext = os.path.splitext(input_path)[1].lower()
    
    if ext == '.mol2':
        return [load_mol2(input_path)]
    elif ext in ['.ism', '.smi', '.smiles']:
        return load_smiles_file(input_path)
    else:
        raise ValueError(f"information of file format: {ext}, details.mol2,.ism,.smi")


def get_hbond_donors(mol):
    """get hydrogen-bond donor atom"""
    donors = []
    # hydrogen-bond donor: N-H, O-H, S-H
    pattern_nh = Chem.MolFromSmarts('[N;!H0]')
    pattern_oh = Chem.MolFromSmarts('[O;!H0]')
    pattern_sh = Chem.MolFromSmarts('[S;!H0]')
    
    for pattern in [pattern_nh, pattern_oh, pattern_sh]:
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            donors.extend(matches)
    
    return donors


def get_hbond_acceptors(mol):
    """get hydrogen-bond acceptor atom"""
    acceptors = []
    # hydrogen-bond acceptor: has details charge molecule of N, O, S (exclude positively charged)
    pattern_n = Chem.MolFromSmarts('[N;!$([N+]);!$([n+])]')
    pattern_o = Chem.MolFromSmarts('[O;!$([O+])]')
    pattern_s = Chem.MolFromSmarts('[S;!$([S+])]')
    
    for pattern in [pattern_n, pattern_o, pattern_s]:
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            acceptors.extend(matches)
    
    return acceptors


def get_aromatic_rings(mol):
    """get aromatic ring"""
    ring_info = mol.GetRingInfo()
    aromatic_rings = []
    
    for ring in ring_info.AtomRings():
        is_aromatic = all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
        if is_aromatic:
            aromatic_rings.append(ring)
    
    return aromatic_rings


def get_positive_centers(mol):
    """get positive charge center"""
    positive = []
    # positive charge: details, nitro group, summary
    patterns = [
        '[N+;!$([N+]-[O-])]',  # details (exclude nitro group)
        '[n+]',                 # aromatic successfully molecule
        '[NH2+]',               # molecule details
        '[NH3+]',               # molecule information
]
    
    for smarts in patterns:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            positive.extend(matches)
    
    return positive


def get_negative_centers(mol):
    """get negative charge center"""
    negative = []
    # negative charge: information, information, summary
    patterns = [
        '[O-]',                 # text anion
        '[S-]',                 # text anion
        '[n-]',                 # aromatic text anion
        '[C;$(C(=O)[O-])]',     # summary
]
    
    for smarts in patterns:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            negative.extend(matches)
    
    return negative


def get_hydrophobic_centers(mol):
    """get hydrophobic group"""
    hydrophobic = []
    # hydrophobic group: information, information
    patterns = [
        '[C;!$(C=O);!$(C-[N,O,S])]',  # information (summary atom)
        '[F,Cl,Br,I]',                 # details
        '[c;!$(c-[N,O,S])]',           # aromatic text (summary atom)
]
    
    for smarts in patterns:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            hydrophobic.extend(matches)
    
    return hydrophobic


def extract_pharmacophore_features(mol):
    """extract has pharmacophore features"""
    features = {
        'hbond_donors': get_hbond_donors(mol),
        'hbond_acceptors': get_hbond_acceptors(mol),
        'aromatic_rings': get_aromatic_rings(mol),
        'positive_centers': get_positive_centers(mol),
        'negative_centers': get_negative_centers(mol),
        'hydrophobic_centers': get_hydrophobic_centers(mol),
    }
    return features


def compute_additional_descriptors(mol):
    """calculation details of molecule information"""
    descriptors = {
        'molecular_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol),
        'num_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'num_heavy_atoms': mol.GetNumHeavyAtoms(),
        'num_rings': rdMolDescriptors.CalcNumRings(mol),
        'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
        'fraction_csp3': rdMolDescriptors.CalcFractionCSP3(mol),
    }
    return descriptors


def generate_tpp_vector(features, descriptors):
    """
    generate Target Pharmacophore Profile (TPP) vector
    
    return one details, details: 
    - details features (has_xxx)
    - data features (xxx_count)
    - molecule information
    """
    tpp = {}
    
    # details features
    tpp['has_hbond_donor'] = 1 if len(features['hbond_donors']) > 0 else 0
    tpp['has_hbond_acceptor'] = 1 if len(features['hbond_acceptors']) > 0 else 0
    tpp['has_aromatic'] = 1 if len(features['aromatic_rings']) > 0 else 0
    tpp['has_positive_charge'] = 1 if len(features['positive_centers']) > 0 else 0
    tpp['has_negative_charge'] = 1 if len(features['negative_centers']) > 0 else 0
    tpp['has_hydrophobic'] = 1 if len(features['hydrophobic_centers']) > 0 else 0
    
    # data features
    tpp['hbond_donor_count'] = len(features['hbond_donors'])
    tpp['hbond_acceptor_count'] = len(features['hbond_acceptors'])
    tpp['aromatic_ring_count'] = len(features['aromatic_rings'])
    tpp['positive_charge_count'] = len(features['positive_centers'])
    tpp['negative_charge_count'] = len(features['negative_centers'])
    tpp['hydrophobic_count'] = len(features['hydrophobic_centers'])
    
    # molecule information
    tpp.update(descriptors)
    
    return tpp


def tpp_to_tensor(tpp):
    """will TPP summary as PyTorch count"""
    # details features order
    feature_order = [
        # details features
        'has_hbond_donor', 'has_hbond_acceptor', 'has_aromatic',
        'has_positive_charge', 'has_negative_charge', 'has_hydrophobic',
        # data features
        'hbond_donor_count', 'hbond_acceptor_count', 'aromatic_ring_count',
        'positive_charge_count', 'negative_charge_count', 'hydrophobic_count',
        # molecule information
        'molecular_weight', 'logp', 'tpsa', 'num_rotatable_bonds',
        'num_heavy_atoms', 'num_rings', 'num_aromatic_rings',
        'num_heteroatoms', 'fraction_csp3',
]
    
    values = [float(tpp[key]) for key in feature_order]
    tensor = torch.tensor(values, dtype=torch.float32)
    
    return tensor, feature_order


def print_summary(features, tpp):
    """details pharmacophore features needed"""
    print("\n" + "=" * 60)
    print("pharmacophore features extract results (Pharmacophore Feature Summary)")
    print("=" * 60)
    
    print("\n[features details]")
    print(f"  hydrogen-bond donor (H-bond Donors):     {tpp['hbond_donor_count']} text")
    print(f"  hydrogen-bond acceptor (H-bond Acceptors):  {tpp['hbond_acceptor_count']} text")
    print(f"  aromatic ring (Aromatic Rings):      {tpp['aromatic_ring_count']} text")
    print(f"  positive charge center (Positive):        {tpp['positive_charge_count']} text")
    print(f"  negative charge center (Negative):        {tpp['negative_charge_count']} text")
    print(f"  hydrophobic group (Hydrophobic):       {tpp['hydrophobic_count']} text")
    
    print("\n[molecule information]")
    print(f"  molecule amount (MW):                  {tpp['molecular_weight']:.2f}")
    print(f"  LogP:                         {tpp['logp']:.2f}")
    print(f"  TPSA:                         {tpp['tpsa']:.2f}")
    print(f"  can details bonds:                   {tpp['num_rotatable_bonds']}")
    print(f"  atoms:                     {tpp['num_heavy_atoms']}")
    print(f"  ring data:                         {tpp['num_rings']}")
    print(f"  aromatic ring data:                     {tpp['num_aromatic_rings']}")
    print(f"  atoms:                     {tpp['num_heteroatoms']}")
    print(f"  Fsp3:                         {tpp['fraction_csp3']:.3f}")
    
    print("\n" + "=" * 60)


def save_results(tpp, tensor, feature_order, output_dir, base_name="pharmacophore"):
    """save results text JSON and.pt file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # save JSON
    json_path = os.path.join(output_dir, f"{base_name}_tpp.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(tpp, f, indent=2, ensure_ascii=False)
    print(f"\nTPP vector saved to: {json_path}")
    
    # save PyTorch count
    pt_path = os.path.join(output_dir, f"{base_name}_tpp.pt")
    torch.save({
        'tpp_vector': tensor,
        'feature_names': feature_order,
        'tpp_dict': tpp,
    }, pt_path)
    print(f"PyTorch count saved to: {pt_path}")
    
    return json_path, pt_path


def aggregate_tpp_from_molecules(molecules):
    """
    details molecule details generate details TPP vector
    information have active molecules of pharmacophore features split text
    """
    all_features = []
    all_descriptors = []
    
    for mol in molecules:
        features = extract_pharmacophore_features(mol)
        descriptors = compute_additional_descriptors(mol)
        tpp = generate_tpp_vector(features, descriptors)
        all_features.append(features)
        all_descriptors.append(tpp)
    
    # summary
    aggregated_tpp = {}
    
    # details features: calculation have details molecule have features
    bool_keys = ['has_hbond_donor', 'has_hbond_acceptor', 'has_aromatic',
                 'has_positive_charge', 'has_negative_charge', 'has_hydrophobic']
    for key in bool_keys:
        count = sum(1 for tpp in all_descriptors if tpp[key] == 1)
        aggregated_tpp[key] = 1 if count > len(molecules) * 0.5 else 0  # details50%of molecule have features
        aggregated_tpp[f'{key}_ratio'] = count / len(molecules)
    
    # data features: calculation details value, details value, details value
    count_keys = ['hbond_donor_count', 'hbond_acceptor_count', 'aromatic_ring_count',
                  'positive_charge_count', 'negative_charge_count', 'hydrophobic_count']
    for key in count_keys:
        values = [tpp[key] for tpp in all_descriptors]
        aggregated_tpp[f'{key}_mean'] = sum(values) / len(values)
        aggregated_tpp[f'{key}_min'] = min(values)
        aggregated_tpp[f'{key}_max'] = max(values)
    
    # molecule information: calculation details value
    desc_keys = ['molecular_weight', 'logp', 'tpsa', 'num_rotatable_bonds',
                 'num_heavy_atoms', 'num_rings', 'num_aromatic_rings',
                 'num_heteroatoms', 'fraction_csp3']
    for key in desc_keys:
        values = [tpp[key] for tpp in all_descriptors]
        aggregated_tpp[f'{key}_mean'] = sum(values) / len(values)
    
    aggregated_tpp['num_molecules'] = len(molecules)
    
    return aggregated_tpp, all_descriptors


def aggregated_tpp_to_tensor(aggregated_tpp):
    """will details of TPP summary as PyTorch count"""
    # details features order (use model details of details features)
    feature_order = [
        # details features
        'has_hbond_donor', 'has_hbond_acceptor', 'has_aromatic',
        'has_positive_charge', 'has_negative_charge', 'has_hydrophobic',
        # compare features
        'has_hbond_donor_ratio', 'has_hbond_acceptor_ratio', 'has_aromatic_ratio',
        'has_positive_charge_ratio', 'has_negative_charge_ratio', 'has_hydrophobic_ratio',
        # data features threshold
        'hbond_donor_count_mean', 'hbond_acceptor_count_mean', 'aromatic_ring_count_mean',
        'positive_charge_count_mean', 'negative_charge_count_mean', 'hydrophobic_count_mean',
        # molecule summary value
        'molecular_weight_mean', 'logp_mean', 'tpsa_mean', 'num_rotatable_bonds_mean',
        'num_heavy_atoms_mean', 'num_rings_mean', 'num_aromatic_rings_mean',
        'num_heteroatoms_mean', 'fraction_csp3_mean',
]
    
    values = [float(aggregated_tpp[key]) for key in feature_order]
    tensor = torch.tensor(values, dtype=torch.float32)
    
    return tensor, feature_order


def print_aggregated_summary(aggregated_tpp):
    """summary pharmacophore features needed"""
    print("\n" + "=" * 70)
    print("details pharmacophore features (Aggregated Pharmacophore Profile)")
    print("=" * 70)
    print(f"molecule count: {aggregated_tpp['num_molecules']}")
    
    print("\n[features details compare text]")
    print(f"  hydrogen-bond donor (H-bond Donors):     {aggregated_tpp['has_hbond_donor_ratio']*100:.1f}%")
    print(f"  hydrogen-bond acceptor (H-bond Acceptors):  {aggregated_tpp['has_hbond_acceptor_ratio']*100:.1f}%")
    print(f"  aromatic ring (Aromatic Rings):      {aggregated_tpp['has_aromatic_ratio']*100:.1f}%")
    print(f"  positive charge center (Positive):        {aggregated_tpp['has_positive_charge_ratio']*100:.1f}%")
    print(f"  negative charge center (Negative):        {aggregated_tpp['has_negative_charge_ratio']*100:.1f}%")
    print(f"  hydrophobic group (Hydrophobic):       {aggregated_tpp['has_hydrophobic_ratio']*100:.1f}%")
    
    print("\n[features count details (details value)]")
    print(f"  hydrogen-bond donor:   {aggregated_tpp['hbond_donor_count_mean']:.2f}")
    print(f"  hydrogen-bond acceptor:   {aggregated_tpp['hbond_acceptor_count_mean']:.2f}")
    print(f"  aromatic ring:     {aggregated_tpp['aromatic_ring_count_mean']:.2f}")
    print(f"  positive charge center: {aggregated_tpp['positive_charge_count_mean']:.2f}")
    print(f"  negative charge center: {aggregated_tpp['negative_charge_count_mean']:.2f}")
    print(f"  hydrophobic group:   {aggregated_tpp['hydrophobic_count_mean']:.2f}")
    
    print("\n[molecule information (details value)]")
    print(f"  molecule amount (MW):    {aggregated_tpp['molecular_weight_mean']:.2f}")
    print(f"  LogP:           {aggregated_tpp['logp_mean']:.2f}")
    print(f"  TPSA:           {aggregated_tpp['tpsa_mean']:.2f}")
    print(f"  can details bonds:     {aggregated_tpp['num_rotatable_bonds_mean']:.1f}")
    print(f"  atoms:       {aggregated_tpp['num_heavy_atoms_mean']:.1f}")
    print(f"  ring data:           {aggregated_tpp['num_rings_mean']:.1f}")
    print(f"  aromatic ring data:       {aggregated_tpp['num_aromatic_rings_mean']:.1f}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='molecule file extract pharmacophore features generate TPP vector'
)
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='details file path (details.mol2,.ism,.smi)'
)
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='.',
        help='details directory (details: from the current directory)'
)
    parser.add_argument(
        '--name', '-n',
        type=str,
        default='pharmacophore',
        help='details file of base information (details: pharmacophore)'
)
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='silent mode, summary need'
)
    
    args = parser.parse_args()
    
    # load molecule
    print(f"positive load molecule: {args.input}")
    molecules = load_molecules(args.input)
    print(f"success load {len(molecules)} molecule")
    
    if len(molecules) == 0:
        print("error: no success load details molecule!")
        return None, None
    
    if len(molecules) == 1:
        # molecule mode (mol2 file)
        mol = molecules[0]
        print("successfully extract pharmacophore features...")
        features = extract_pharmacophore_features(mol)
        descriptors = compute_additional_descriptors(mol)
        tpp = generate_tpp_vector(features, descriptors)
        tensor, feature_order = tpp_to_tensor(tpp)
        
        if not args.quiet:
            print_summary(features, tpp)
        
        save_results(tpp, tensor, feature_order, args.output, args.name)
    else:
        # molecule mode (SMILES file) - generate details TPP
        print("successfully extract pharmacophore features information...")
        aggregated_tpp, all_tpps = aggregate_tpp_from_molecules(molecules)
        tensor, feature_order = aggregated_tpp_to_tensor(aggregated_tpp)
        
        if not args.quiet:
            print_aggregated_summary(aggregated_tpp)
        
        # save details results
        os.makedirs(args.output, exist_ok=True)
        
        json_path = os.path.join(args.output, f"{args.name}_tpp.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated_tpp, f, indent=2, ensure_ascii=False)
        print(f"\nTPP vector saved to: {json_path}")
        
        pt_path = os.path.join(args.output, f"{args.name}_tpp.pt")
        torch.save({
            'tpp_vector': tensor,
            'feature_names': feature_order,
            'tpp_dict': aggregated_tpp,
            'all_molecule_tpps': all_tpps,  # save to have molecule of TPP
        }, pt_path)
        print(f"PyTorch count saved to: {pt_path}")
        
        tpp = aggregated_tpp
    
    print("\ncompleted!")
    return tpp, tensor


if __name__ == '__main__':
    main()
