#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic receptor preparation for Smina docking.

Given a PDB file and a ligand residue name, this script extracts the crystal
ligand, removes waters and selected hetero residues from the receptor, converts
the receptor to PDBQT with OpenBabel, and writes a docking_config.txt file.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_residue_set(value: str) -> set[str]:
    return {item.strip().upper() for item in value.split(",") if item.strip()}


def extract_ligand(lines: list[str], ligand_resname: str, output_pdb: Path) -> tuple[float, float, float, float, float, float]:
    ligand_lines: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    for line in lines:
        if line.startswith("HETATM") and line[17:20].strip().upper() == ligand_resname:
            ligand_lines.append(line)
            xs.append(float(line[30:38]))
            ys.append(float(line[38:46]))
            zs.append(float(line[46:54]))

    if not ligand_lines:
        raise ValueError(f"Ligand residue {ligand_resname} was not found in the PDB file.")

    with output_pdb.open("w", encoding="utf-8") as handle:
        handle.writelines(ligand_lines)
        handle.write("END\n")

    center = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
    size = (max(xs) - min(xs) + 10.0, max(ys) - min(ys) + 10.0, max(zs) - min(zs) + 10.0)
    return (*center, *size)


def clean_receptor(
    lines: list[str],
    ligand_resname: str,
    remove_residues: set[str],
    output_pdb: Path,
    chain: str | None,
) -> None:
    cleaned: list[str] = []
    for line in lines:
        record = line[:6].strip()
        if record == "ATOM":
            if chain is None or line[21].strip() in {chain, ""}:
                cleaned.append(line)
        elif record == "HETATM":
            resname = line[17:20].strip().upper()
            if resname == ligand_resname or resname in remove_residues:
                continue
        elif record == "TER":
            cleaned.append(line)
        elif record in {"HEADER", "TITLE", "REMARK", "CRYST1", "SCALE1", "SCALE2", "SCALE3"}:
            cleaned.append(line)

    cleaned.append("END\n")
    with output_pdb.open("w", encoding="utf-8") as handle:
        handle.writelines(cleaned)


def run_obabel(obabel: str, input_path: Path, output_path: Path, extra_args: list[str]) -> None:
    cmd = [obabel, str(input_path), "-O", str(output_path), *extra_args]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"OpenBabel failed: {result.stderr.strip()}")


def write_config(path: Path, center_size: tuple[float, float, float, float, float, float]) -> None:
    cx, cy, cz, sx, sy, sz = center_size
    path.write_text(
        "\n".join(
            [
                "receptor = receptor.pdbqt",
                f"center_x = {cx:.3f}",
                f"center_y = {cy:.3f}",
                f"center_z = {cz:.3f}",
                f"size_x = {sx:.1f}",
                f"size_y = {sy:.1f}",
                f"size_z = {sz:.1f}",
                "exhaustiveness = 8",
                "num_modes = 5",
                "energy_range = 3",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a receptor PDBQT and docking box from a crystal PDB.")
    parser.add_argument("--pdb", required=True, help="Input crystal structure PDB file.")
    parser.add_argument("--ligand-resname", required=True, help="Ligand residue name used to define the docking box.")
    parser.add_argument("--obabel", default="obabel", help="Path to OpenBabel executable.")
    parser.add_argument("--output-dir", default=".", help="Directory for generated receptor and ligand files.")
    parser.add_argument("--remove-residues", default="HOH", help="Comma-separated HETATM residue names to remove.")
    parser.add_argument("--chain", help="Optional protein chain ID to keep.")
    parser.add_argument("--ph", type=float, default=7.4, help="pH used by OpenBabel protonation.")
    parser.add_argument("--skip-ligand-mol2", action="store_true", help="Only write crystal_ligand.pdb.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ligand_resname = args.ligand_resname.upper()
    lines = Path(args.pdb).read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)

    ligand_pdb = output_dir / "crystal_ligand.pdb"
    ligand_mol2 = output_dir / "crystal_ligand.mol2"
    cleaned_pdb = output_dir / "cleaned_receptor.pdb"
    receptor_pdbqt = output_dir / "receptor.pdbqt"
    docking_config = output_dir / "docking_config.txt"

    center_size = extract_ligand(lines, ligand_resname, ligand_pdb)
    clean_receptor(
        lines,
        ligand_resname,
        parse_residue_set(args.remove_residues),
        cleaned_pdb,
        args.chain,
    )

    if not args.skip_ligand_mol2:
        run_obabel(args.obabel, ligand_pdb, ligand_mol2, [])
    run_obabel(args.obabel, cleaned_pdb, receptor_pdbqt, ["-h", "-p", str(args.ph), "-xr"])
    write_config(docking_config, center_size)

    print(f"Saved ligand PDB: {ligand_pdb}")
    if ligand_mol2.exists():
        print(f"Saved ligand MOL2: {ligand_mol2}")
    print(f"Saved cleaned receptor: {cleaned_pdb}")
    print(f"Saved receptor PDBQT: {receptor_pdbqt}")
    print(f"Saved docking config: {docking_config}")


if __name__ == "__main__":
    main()
