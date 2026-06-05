#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check random-split SMILES and scaffold leakage for active/decoy files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split


RDLogger.DisableLog("rdApp.*")


def load_ism(path: Path, label: int) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 2:
                rows.append({"smiles": parts[0], "id": parts[1], "label": label})
    return pd.DataFrame(rows)


def scaffold_from_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol))
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check leakage between random train/test splits.")
    parser.add_argument("--actives", required=True, help="Active molecule .ism file.")
    parser.add_argument("--decoys", required=True, help="Decoy/inactive molecule .ism file.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    active_df = load_ism(Path(args.actives), label=1)
    decoy_df = load_ism(Path(args.decoys), label=0)
    dataset = pd.concat([active_df, decoy_df], ignore_index=True)

    train_df, test_df = train_test_split(
        dataset,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=dataset["label"],
    )

    train_smiles = set(train_df["smiles"])
    test_smiles = set(test_df["smiles"])
    smiles_overlap = train_smiles & test_smiles

    train_active_scaffolds = {
        scaffold
        for scaffold in (scaffold_from_smiles(smi) for smi in train_df[train_df["label"] == 1]["smiles"])
        if scaffold
    }
    test_active_scaffolds = {
        scaffold
        for scaffold in (scaffold_from_smiles(smi) for smi in test_df[test_df["label"] == 1]["smiles"])
        if scaffold
    }
    scaffold_overlap = train_active_scaffolds & test_active_scaffolds

    test_actives = int(test_df["label"].sum())
    leaked_test_actives = sum(
        1
        for smi in test_df[test_df["label"] == 1]["smiles"]
        if (scaffold := scaffold_from_smiles(smi)) and scaffold in train_active_scaffolds
    )

    print(f"Total molecules: {len(dataset)}")
    print(f"Train molecules: {len(train_df)}")
    print(f"Test molecules: {len(test_df)}")
    print(f"Exact SMILES overlap: {len(smiles_overlap)}")
    print(f"Train active scaffolds: {len(train_active_scaffolds)}")
    print(f"Test active scaffolds: {len(test_active_scaffolds)}")
    print(f"Active scaffold overlap: {len(scaffold_overlap)}")
    if test_actives:
        print(f"Test actives with train-overlapping scaffold: {leaked_test_actives}/{test_actives}")


if __name__ == "__main__":
    main()
