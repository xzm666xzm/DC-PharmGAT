#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic ligand extraction utility.

Use this when only the co-crystal ligand needs to be extracted from a PDB file.
It writes a ligand PDB and can optionally convert it to MOL2 with OpenBabel.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def extract_ligand_lines(pdb_path: Path, ligand_resname: str) -> list[str]:
    ligand_resname = ligand_resname.upper()
    lines: list[str] = []
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("HETATM") and line[17:20].strip().upper() == ligand_resname:
                lines.append(line)
    if not lines:
        raise ValueError(f"Ligand residue {ligand_resname} was not found in {pdb_path}")
    return lines


def write_ligand_pdb(lines: list[str], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        handle.writelines(lines)
        handle.write("END\n")


def convert_to_mol2(obabel: str, ligand_pdb: Path, output_mol2: Path) -> None:
    result = subprocess.run(
        [obabel, str(ligand_pdb), "-O", str(output_mol2)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"OpenBabel failed: {result.stderr.strip()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a ligand residue from a PDB file.")
    parser.add_argument("--pdb", required=True, help="Input PDB file.")
    parser.add_argument("--ligand-resname", required=True, help="Ligand residue name, for example OHT or ZDG.")
    parser.add_argument("--output-pdb", default="crystal_ligand.pdb")
    parser.add_argument("--output-mol2", default="crystal_ligand.mol2")
    parser.add_argument("--obabel", default="obabel")
    parser.add_argument("--skip-mol2", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ligand_lines = extract_ligand_lines(Path(args.pdb), args.ligand_resname)
    output_pdb = Path(args.output_pdb)
    output_mol2 = Path(args.output_mol2)

    write_ligand_pdb(ligand_lines, output_pdb)
    print(f"Saved {len(ligand_lines)} ligand atoms to {output_pdb}")

    if not args.skip_mol2:
        convert_to_mol2(args.obabel, output_pdb, output_mol2)
        print(f"Saved MOL2 ligand to {output_mol2}")


if __name__ == "__main__":
    main()
