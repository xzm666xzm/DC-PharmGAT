# atom_features: encode atom symbol, num bonded atoms, num hydrogens, implicit valence, if aromatic
# new text: pharmacophore property features (is_h_bond_donor, is_h_bond_acceptor, is_aromatic, is_positive, is_negative, is_hydrophobic)

import numpy as np
from rdkit import Chem


def encoding(feat, featArray):
    if feat not in featArray:
        feat = featArray[0]
    return list(map(lambda f: int(feat == f), featArray))


# ============================================================
# pharmacophore property details test data (and extract_pharmacophore.py save to single)
# ============================================================

# information SMARTS mode (extract high effect rate)
_SMARTS_PATTERNS = {}

def _get_smarts_pattern(smarts):
    """get text create SMARTS mode object"""
    if smarts not in _SMARTS_PATTERNS:
        _SMARTS_PATTERNS[smarts] = Chem.MolFromSmarts(smarts)
    return _SMARTS_PATTERNS[smarts]


def get_hbond_donor_atoms(mol):
    """
    get hydrogen-bond donor atom summary
    hydrogen-bond donor: N-H, O-H, S-H (has hydrogen atom of N, O, S)
    """
    donor_atoms = set()
    patterns = ['[N;!H0]', '[O;!H0]', '[S;!H0]']
    
    for smarts in patterns:
        pattern = _get_smarts_pattern(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                donor_atoms.update(match)
    
    return donor_atoms


def get_hbond_acceptor_atoms(mol):
    """
    get hydrogen-bond acceptor atom summary
    hydrogen-bond acceptor: has details charge molecule of N, O, S (exclude positively charged)
    """
    acceptor_atoms = set()
    patterns = [
        '[N;!$([N+]);!$([n+])]',  # text (exclude positive charge)
        '[O;!$([O+])]',           # text (exclude positive charge)
        '[S;!$([S+])]',           # text (exclude positive charge)
]
    
    for smarts in patterns:
        pattern = _get_smarts_pattern(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                acceptor_atoms.update(match)
    
    return acceptor_atoms


def get_positive_atoms(mol):
    """
    get positive charge center atom summary
    positive charge: details, molecule details, aromatic successfully molecule
    """
    positive_atoms = set()
    patterns = [
        '[N+;!$([N+]-[O-])]',  # details (exclude nitro group)
        '[n+]',                 # aromatic successfully molecule
        '[NH2+]',               # molecule details
        '[NH3+]',               # molecule information
]
    
    for smarts in patterns:
        pattern = _get_smarts_pattern(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                positive_atoms.update(match)
    
    return positive_atoms


def get_negative_atoms(mol):
    """
    get negative charge center atom summary
    negative charge: information, information, summary
    """
    negative_atoms = set()
    patterns = [
        '[O-]',                 # text anion
        '[S-]',                 # text anion
        '[n-]',                 # aromatic text anion
        '[C;$(C(=O)[O-])]',     # summary
]
    
    for smarts in patterns:
        pattern = _get_smarts_pattern(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                negative_atoms.update(match)
    
    return negative_atoms


def get_hydrophobic_atoms(mol):
    """
    get hydrophobic hydrophobic atoms summary
    hydrophobic group: information, details, summary atom of aromatic text
    """
    hydrophobic_atoms = set()
    patterns = [
        '[C;!$(C=O);!$(C-[N,O,S])]',  # information (summary atom)
        '[F,Cl,Br,I]',                 # details
        '[c;!$(c-[N,O,S])]',           # aromatic text (summary atom)
]
    
    for smarts in patterns:
        pattern = _get_smarts_pattern(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                hydrophobic_atoms.update(match)
    
    return hydrophobic_atoms


# cache molecule of pharmacophore atom details (information recalculation)
_mol_pharmacophore_cache = {}

def get_pharmacophore_sets(mol):
    """
    get molecule of has pharmacophore atom details
    use cache information recalculation
    """
    # use molecule of SMILES as cache bond
    try:
        mol_key = Chem.MolToSmiles(mol)
    except:
        mol_key = id(mol)
    
    if mol_key not in _mol_pharmacophore_cache:
        _mol_pharmacophore_cache[mol_key] = {
            'donors': get_hbond_donor_atoms(mol),
            'acceptors': get_hbond_acceptor_atoms(mol),
            'positive': get_positive_atoms(mol),
            'negative': get_negative_atoms(mol),
            'hydrophobic': get_hydrophobic_atoms(mol),
        }
    
    return _mol_pharmacophore_cache[mol_key]


def clear_pharmacophore_cache():
    """text remove pharmacophore cache (process count molecule can use)"""
    global _mol_pharmacophore_cache
    _mol_pharmacophore_cache = {}


def getPharmacophoreFeatures(atom, mol=None, pharmacophore_sets=None):
    """
    get atom of pharmacophore property features
    
    argument:
        atom: RDKit Atom object
        mol: RDKit Mol object (use calculate pharmacophore details)
        pharmacophore_sets: calculate of pharmacophore atom details (can, extract high effect rate)
    
    return:
        numpy array: [is_donor, is_acceptor, is_aromatic, is_positive, is_negative, is_hydrophobic]
    """
    atom_idx = atom.GetIdx()
    
    # result extract donor details calculation of details, directly use
    if pharmacophore_sets is not None:
        donors = pharmacophore_sets['donors']
        acceptors = pharmacophore_sets['acceptors']
        positive = pharmacophore_sets['positive']
        negative = pharmacophore_sets['negative']
        hydrophobic = pharmacophore_sets['hydrophobic']
    elif mol is not None:
        # molecule calculation pharmacophore details
        sets = get_pharmacophore_sets(mol)
        donors = sets['donors']
        acceptors = sets['acceptors']
        positive = sets['positive']
        negative = sets['negative']
        hydrophobic = sets['hydrophobic']
    else:
        # none calculate, return details
        return np.array([0, 0, 0, 0, 0, 0])
    
    return np.array([
        int(atom_idx in donors),       # is_h_bond_donor
        int(atom_idx in acceptors),    # is_h_bond_acceptor
        int(atom.GetIsAromatic()),     # is_aromatic (information atom get)
        int(atom_idx in positive),     # is_positive
        int(atom_idx in negative),     # is_negative
        int(atom_idx in hydrophobic),  # is_hydrophobic
])


# ============================================================
# has features data (save information, new text pharmacophore features)
# ============================================================

def getAtomFeatures(atom, mol=None, pharmacophore_sets=None):
    """
    get atom features vector
    
    has features:
        - atom details one-hot (44dimensions)
        - degree one-hot (6dimensions)
        - hydrogen atoms one-hot (5dimensions)
        - details one-hot (6dimensions)
        - details aromatic (1dimensions)
    
    new text pharmacophore features:
        - is_h_bond_donor (1dimensions)
        - is_h_bond_acceptor (1dimensions)
        - is_aromatic (1dimensions) - and has restore, save to single property
        - is_positive (1dimensions)
        - is_negative (1dimensions)
        - is_hydrophobic (1dimensions)
    
    details: 62 + 6 = 68 dimensions (text62dimensions + new6dimensions)
    """
    symbolArray = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                   'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                   'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
                   'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                   'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    degArray = [x for x in range(6)]
    
    # has features
    base_features = np.array([
        *encoding(atom.GetSymbol(), symbolArray),
        *encoding(atom.GetDegree(), degArray),
        *encoding(atom.GetTotalNumHs(), degArray[:-1]),
        *encoding(atom.GetValence(Chem.ValenceType.IMPLICIT), degArray),
        int(atom.GetIsAromatic())
])
    
    # new text pharmacophore features
    pharmacophore_features = getPharmacophoreFeatures(atom, mol, pharmacophore_sets)
    
    # details features
    return np.concatenate([base_features, pharmacophore_features])


def getBondFeatures(bond):
    """get bond features vector (save information)"""
    bondType = bond.GetBondType()
    return np.array(list(map(int, [
        bondType == Chem.rdchem.BondType.SINGLE,
        bondType == Chem.rdchem.BondType.DOUBLE,
        bondType == Chem.rdchem.BondType.TRIPLE,
        bondType == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
])))


def num_atom_features():
    """return atom features dimension"""
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(getAtomFeatures(a, mol=m))


def num_bond_features():
    """return bond features dimension"""
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(getBondFeatures(simple_mol.GetBonds()[0]))


# ============================================================
# information data: get molecule of has atom features
# ============================================================

def getMolAtomFeatures(mol):
    """
    get molecule middle has atom of features details
    
    argument:
        mol: RDKit Mol object
    
    return:
        numpy array: shape (num_atoms, num_features)
    """
    # calculate pharmacophore details (calculate single)
    pharmacophore_sets = get_pharmacophore_sets(mol)
    
    atom_features = []
    for atom in mol.GetAtoms():
        features = getAtomFeatures(atom, mol=mol, pharmacophore_sets=pharmacophore_sets)
        atom_features.append(features)
    
    return np.array(atom_features)


# ============================================================
# pharmacophore features details (use details test and can details)
# ============================================================

PHARMACOPHORE_FEATURE_NAMES = [
    'is_h_bond_donor',
    'is_h_bond_acceptor', 
    'is_aromatic',
    'is_positive',
    'is_negative',
    'is_hydrophobic',
]


def get_feature_names():
    """get has features information table"""
    symbolArray = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                   'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                   'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
                   'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                   'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    
    names = []
    # atom details
    names.extend([f'symbol_{s}' for s in symbolArray])
    # degree
    names.extend([f'degree_{i}' for i in range(6)])
    # hydrogen atoms
    names.extend([f'num_Hs_{i}' for i in range(5)])
    # details
    names.extend([f'implicit_valence_{i}' for i in range(6)])
    # details aromatic (has)
    names.append('is_aromatic_base')
    # pharmacophore features
    names.extend(PHARMACOPHORE_FEATURE_NAMES)
    
    return names
