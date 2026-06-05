#!/usr/bin/env python3
"""
read mol2 file, details Gasteiger charge charge, summary training format of mol2 file
summary of Sybyl atom type (N.4, N.pl3, N.am, S.o2, S.o text) and details bond(am)type
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import sys


def get_sybyl_atom_type(atom, mol):
    """data RDKit atom details of Sybyl atom type"""
    sym = atom.GetSymbol()
    hyb = atom.GetHybridization()
    arom = atom.GetIsAromatic()
    fc = atom.GetFormalCharge()
    degree = atom.GetDegree()  # details of atoms(summaryH)

    if sym == 'C':
        if arom:
            return 'C.ar'
        elif hyb == Chem.HybridizationType.SP2:
            return 'C.2'
        elif hyb == Chem.HybridizationType.SP:
            return 'C.1'
        elif atom.GetIsAromatic():
            return 'C.ar'
        else:
            return 'C.3'

    elif sym == 'N':
        if arom:
            return 'N.ar'
        # positively charged four information -> N.4
        if fc > 0 and degree == 4:
            return 'N.4'
        if fc > 0 and hyb == Chem.HybridizationType.SP3:
            return 'N.4'
        if hyb == Chem.HybridizationType.SP2:
            # check details (N-C=O)
            is_amide = False
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'C':
                    for n2 in neighbor.GetNeighbors():
                        if n2.GetSymbol() == 'O':
                            bond = mol.GetBondBetweenAtoms(neighbor.GetIdx(), n2.GetIdx())
                            if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                                is_amide = True
                                break
                    # text check details base (N-S(=O)=O) -> as N.am text N.pl3
                    if not is_amide:
                        for n2 in neighbor.GetNeighbors():
                            if n2.GetSymbol() == 'S':
                                is_amide = True
                                break
                if is_amide:
                    break
            if is_amide:
                return 'N.am'
            # check details N.pl3 (details base, details)
            # result SP2 details text, information3textHatom has total text
            if degree >= 3:
                return 'N.pl3'
            return 'N.2'
        elif hyb == Chem.HybridizationType.SP3:
            if fc > 0:
                return 'N.4'
            return 'N.3'
        elif hyb == Chem.HybridizationType.SP:
            return 'N.1'
        else:
            return 'N.3'

    elif sym == 'O':
        if arom:
            return 'O.ar'
        # check details as nitro group negative (O.co2)
        if fc < 0:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'C':
                    o_count = 0
                    for n2 in neighbor.GetNeighbors():
                        if n2.GetSymbol() == 'O':
                            o_count += 1
                    if o_count >= 2:
                        return 'O.co2'
        if hyb == Chem.HybridizationType.SP2:
            return 'O.2'
        else:
            return 'O.3'

    elif sym == 'S':
        # details andSdetails of atoms amount(text bond text)
        double_bond_o_count = 0
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'O':
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                    double_bond_o_count += 1
        if double_bond_o_count >= 2:
            return 'S.o2'  # details base -SO2-
        elif double_bond_o_count == 1:
            return 'S.o'   # details -SO-
        elif arom:
            return 'S.ar'
        elif hyb == Chem.HybridizationType.SP2:
            return 'S.2'
        else:
            return 'S.3'

    elif sym == 'P':
        return 'P.3'

    elif sym == 'H':
        return 'H'
    elif sym == 'F':
        return 'F'
    elif sym == 'Cl':
        return 'Cl'
    elif sym == 'Br':
        return 'Br'
    elif sym == 'I':
        return 'I'
    else:
        return sym


def get_sybyl_bond_type(bond, mol):
    """data RDKit bond summary Sybyl bond type, summary bond(am)"""
    if bond.GetIsAromatic():
        return 'ar'
    bt = bond.GetBondType()
    if bt == Chem.BondType.DOUBLE:
        return '2'
    elif bt == Chem.BondType.TRIPLE:
        return '3'
    elif bt == Chem.BondType.SINGLE:
        # check details bond (C(=O)-N)
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        c_atom, n_atom = None, None
        if a1.GetSymbol() == 'C' and a2.GetSymbol() == 'N':
            c_atom, n_atom = a1, a2
        elif a1.GetSymbol() == 'N' and a2.GetSymbol() == 'C':
            c_atom, n_atom = a2, a1
        if c_atom is not None and n_atom is not None:
            # check C details have =O(details of nitro group)
            for neighbor in c_atom.GetNeighbors():
                if neighbor.GetSymbol() == 'O':
                    c_o_bond = mol.GetBondBetweenAtoms(c_atom.GetIdx(), neighbor.GetIdx())
                    if c_o_bond and c_o_bond.GetBondType() == Chem.BondType.DOUBLE:
                        return 'am'
        return '1'
    else:
        return '1'


def detect_overlapping_atoms(mol):
    """text test molecule middle details store details of atom"""
    conf = mol.GetConformer()
    overlapping = []
    n = mol.GetNumAtoms()

    for i in range(n):
        for j in range(i + 1, n):
            pos_i = conf.GetAtomPosition(i)
            pos_j = conf.GetAtomPosition(j)
            dist = ((pos_i.x - pos_j.x)**2 + (pos_i.y - pos_j.y)**2 +
                    (pos_i.z - pos_j.z)**2) ** 0.5
            if dist < 0.01:  # details < 0.01 Å as details
                overlapping.append((i, j, dist))

    return overlapping


def clean_molecule(mol):
    """
    process molecule in issue atom:
    1. text test summary of atom
    2. has details, text remove has hydrogen atom details new details
    3. use summary hydrogen atom details
    """
    overlapping = detect_overlapping_atoms(mol)

    if not overlapping:
        print("  [OK] details test information atom")
        return mol

    print(f"  [WARN] text test text {len(overlapping)} information atom:")
    for i, j, dist in overlapping:
        sym_i = mol.GetAtomWithIdx(i).GetSymbol()
        sym_j = mol.GetAtomWithIdx(j).GetSymbol()
        print(f"    atom {i+1}({sym_i}) and atom {j+1}({sym_j}), details: {dist:.4f} A")

    print("  ==> text remove has hydrogen atom details new details...")

    # save to atom details
    conf = mol.GetConformer()
    heavy_coords = {}
    for atom in mol.GetAtoms():
        if atom.GetSymbol()!= 'H':
            idx = atom.GetIdx()
            pos = conf.GetAtomPosition(idx)
            heavy_coords[idx] = (pos.x, pos.y, pos.z)

    # text remove hydrogen atom
    mol_no_h = Chem.RemoveHs(mol)
    if mol_no_h is None:
        print("  [FAIL] text remove hydrogen atom failed, text test directly use original molecule")
        return mol

    # text new details hydrogen atom
    mol_with_h = Chem.AddHs(mol_no_h, addCoords=True)
    if mol_with_h is None:
        print("  [FAIL] details hydrogen atom failed")
        return mol

    # text test use summary hydrogen atom details(information atom)
    try:
        mp = AllChem.MMFFGetMoleculeProperties(mol_with_h)
        if mp is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol_with_h, mp)
            if ff is not None:
                # information has atom
                for atom in mol_with_h.GetAtoms():
                    if atom.GetSymbol()!= 'H':
                        ff.AddFixedPoint(atom.GetIdx())
                ff.Minimize(maxIts=500)
                print("  [OK] hydrogen atom details information")
            else:
                print("  [INFO] MMFF information can use, use details hydrogen atom details")
        else:
            print("  [INFO] MMFF text property calculation failed, use details hydrogen atom details")
    except Exception as e:
        print(f"  [INFO] summary warning: {e}")

    # validation process results
    new_overlapping = detect_overlapping_atoms(mol_with_h)
    if new_overlapping:
        print(f"  [WARN] process after has {len(new_overlapping)} information atom")
    else:
        print("  [OK] process completed, none details atom")

    print(f"  process before: {mol.GetNumAtoms()} atom, {mol.GetNumBonds()} bond")
    print(f"  process after: {mol_with_h.GetNumAtoms()} atom, {mol_with_h.GetNumBonds()} bond")

    return mol_with_h


def process_mol2(input_file, output_file=None):
    """read mol2 file, process issue atom, details charge charge, information training format text new details"""

    if output_file is None:
        output_file = input_file.replace('.mol2', '_charged.mol2')

    print(f"read: {input_file}")

    # read mol2 file
    mol = Chem.MolFromMol2File(input_file, removeHs=False)

    if mol is None:
        print("error: none read mol2 file!")
        return False

    print(f"atoms: {mol.GetNumAtoms()}")
    print(f"bonds: {mol.GetNumBonds()}")

    # text test details process details atom
    print("\ncheck atom details...")
    mol = clean_molecule(mol)

    # calculation Gasteiger charge charge
    AllChem.ComputeGasteigerCharges(mol)
    print("calculate Gasteiger charge charge")

    # get molecule
    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else input_file.replace('.mol2', '')
    if mol_name == "****" or not mol_name:
        mol_name = input_file.replace('.mol2', '').split('/')[-1].split('\\')[-1]

    # generate mol2 details
    conf = mol.GetConformer()

    lines = ["@<TRIPOS>MOLECULE", mol_name]
    lines.append(f"   {mol.GetNumAtoms()}    {mol.GetNumBonds()}     0     0     0")
    lines.extend(["SMALL", "USER_CHARGES", "", "@<TRIPOS>ATOM"])

    # details atom type split text(use details test details)
    type_counts = {}

    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        sym = atom.GetSymbol()

        # use details of Sybyl type additional
        sybyl = get_sybyl_atom_type(atom, mol)

        # details type
        type_counts[sybyl] = type_counts.get(sybyl, 0) + 1

        # get charge charge
        try:
            charge = float(atom.GetProp('_GasteigerCharge'))
            if np.isnan(charge):
                charge = 0.0
        except:
            charge = 0.0

        atom_name = f"{sym}{i+1}"
        lines.append(
            f"{i+1:6d} {atom_name:<4s}    {pos.x:8.4f}   {pos.y:8.4f}   "
            f"{pos.z:8.4f} {sybyl:<9s} 1 <0>        {charge:7.4f}"
)

    # BOND text split
    lines.append("@<TRIPOS>BOND")
    bond_type_counts = {}
    for i, bond in enumerate(mol.GetBonds()):
        a1 = bond.GetBeginAtomIdx() + 1
        a2 = bond.GetEndAtomIdx() + 1

        # use details of bond type additional
        bt = get_sybyl_bond_type(bond, mol)
        bond_type_counts[bt] = bond_type_counts.get(bt, 0) + 1

        lines.append(f"{i+1:6d}   {a1:2d}   {a2:2d} {bt}")

    lines.append("")

    # write file (use Unix run text \n, and training format single)
    with open(output_file, 'w', newline='\n') as f:
        f.write('\n'.join(lines))

    # details
    print(f"\natom type split text:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t:<8s}: {c}")
    print(f"\nbond type split text:")
    for t, c in sorted(bond_type_counts.items()):
        print(f"  {t:<4s}: {c}")

    print(f"\ndetails: {output_file}")
    print("completed!")
    return True


if __name__ == "__main__":
    # details process 5h84_ligand.mol2
    input_file = "crystal_ligand.mol2"
    output_file = "crystal_ligand1.mol2"

    # can summary run argument get
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    process_mol2(input_file, output_file)
