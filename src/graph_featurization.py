"""
Molecular graph tensor construction for DC-PharmGAT.

The model consumes padded atom, bond, and neighbor-index tensors. This module
keeps that conversion in one place and includes the pharmacophore-aware atom
features used by the DC-PharmGAT architecture.
"""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem

from features import (
    clear_pharmacophore_cache,
    getAtomFeatures,
    getBondFeatures,
    get_pharmacophore_sets,
    num_atom_features,
    num_bond_features,
)


def _pad_axis(array: np.ndarray, target_size: int, axis: int, fill_value: int = 0) -> np.ndarray:
    if target_size <= array.shape[axis]:
        return array
    padding = [(0, 0)] * array.ndim
    padding[axis] = (0, target_size - array.shape[axis])
    return np.pad(array, pad_width=padding, mode="constant", constant_values=fill_value)


def build_molecular_graph_tensors(
    smiles_list: list[str],
    max_neighbors: int = 5,
    max_atoms: int = 200,
    dataset_name: str = "unknown",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    atom_feature_count = num_atom_features()
    bond_feature_count = num_bond_features()
    molecule_count = len(smiles_list)

    atom_tensor = np.zeros((molecule_count, max_atoms, atom_feature_count), dtype=float)
    bond_tensor = np.zeros((molecule_count, max_atoms, max_neighbors, bond_feature_count), dtype=float)
    neighbor_tensor = -np.ones((molecule_count, max_atoms, max_neighbors), dtype=int)

    print(f"building graph tensors for {dataset_name} dataset")
    for molecule_index, smiles in enumerate(smiles_list):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            print(f"Warning: could not parse SMILES: {smiles}")
            continue

        pharmacophore_sets = get_pharmacophore_sets(molecule)
        atoms = molecule.GetAtoms()
        bonds = molecule.GetBonds()
        atom_index_map: dict[int, int] = {}
        neighbors: list[list[int]] = [[] for _ in range(len(atoms))]

        if len(atoms) > atom_tensor.shape[1]:
            atom_tensor = _pad_axis(atom_tensor, len(atoms), axis=1)
            bond_tensor = _pad_axis(bond_tensor, len(atoms), axis=1)
            neighbor_tensor = _pad_axis(neighbor_tensor, len(atoms), axis=1, fill_value=-1)

        for local_atom_index, atom in enumerate(atoms):
            atom_tensor[molecule_index, local_atom_index, :atom_feature_count] = getAtomFeatures(
                atom,
                mol=molecule,
                pharmacophore_sets=pharmacophore_sets,
            )
            atom_index_map[atom.GetIdx()] = local_atom_index

        for bond in bonds:
            atom_a = atom_index_map[bond.GetBeginAtom().GetIdx()]
            atom_b = atom_index_map[bond.GetEndAtom().GetIdx()]
            neighbor_slot_a = len(neighbors[atom_a])
            neighbor_slot_b = len(neighbors[atom_b])
            required_neighbors = max(neighbor_slot_a, neighbor_slot_b) + 1

            if required_neighbors > bond_tensor.shape[2]:
                bond_tensor = _pad_axis(bond_tensor, required_neighbors, axis=2)
                neighbor_tensor = _pad_axis(neighbor_tensor, required_neighbors, axis=2, fill_value=-1)

            bond_features = np.asarray(getBondFeatures(bond), dtype=int)
            bond_tensor[molecule_index, atom_a, neighbor_slot_a, :] = bond_features
            bond_tensor[molecule_index, atom_b, neighbor_slot_b, :] = bond_features
            neighbors[atom_a].append(atom_b)
            neighbors[atom_b].append(atom_a)

        for atom_index, atom_neighbors in enumerate(neighbors):
            neighbor_tensor[molecule_index, atom_index, : len(atom_neighbors)] = atom_neighbors

        if molecule_index > 0 and molecule_index % 1000 == 0:
            clear_pharmacophore_cache()

    clear_pharmacophore_cache()
    return (
        torch.from_numpy(atom_tensor).float(),
        torch.from_numpy(bond_tensor).float(),
        torch.from_numpy(neighbor_tensor).long(),
    )
