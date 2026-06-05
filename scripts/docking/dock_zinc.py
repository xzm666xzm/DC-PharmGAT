#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic ZINC docking workflow using RDKit and Smina.

The previous dock_zinc_<target>.py scripts only differed by receptor path,
binding-box coordinates, output name, and a few runtime options. This file keeps
those differences as command-line arguments so the repository does not need one
copy per receptor or PDB ID.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import subprocess
import tempfile
import time
from pathlib import Path


def smiles_to_sdf(smiles: str, output_sdf: Path, seed: int = 42) -> bool:
    """Convert a SMILES string to a 3D SDF ligand file."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        result = AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True)
        if result == -1:
            return False

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass

    writer = Chem.SDWriter(str(output_sdf))
    writer.write(mol)
    writer.close()
    return True


def parse_smina_score(stdout: str) -> float | None:
    """Extract the best affinity score from Smina stdout."""
    after_separator = False
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("-----+"):
            after_separator = True
            continue
        if after_separator and stripped:
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    return None
        if stripped.startswith("1 "):
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    return None
    return None


def dock_one(job: tuple[int, str, str, dict[str, object]]) -> tuple[int, str, float | None, str]:
    index, smiles, molecule_id, config = job
    tmp_dir = Path(str(config["tmp_dir"]))
    ligand_sdf = tmp_dir / f"mol_{os.getpid()}_{index}.sdf"

    try:
        if not smiles_to_sdf(smiles, ligand_sdf, seed=int(config["seed"])):
            return index, molecule_id, None, "3D_FAILED"

        center = config["center"]
        size = config["size"]
        cmd = [
            str(config["smina"]),
            "--receptor",
            str(config["receptor"]),
            "--ligand",
            str(ligand_sdf),
            "--center_x",
            str(center[0]),
            "--center_y",
            str(center[1]),
            "--center_z",
            str(center[2]),
            "--size_x",
            str(size[0]),
            "--size_y",
            str(size[1]),
            "--size_z",
            str(size[2]),
            "--exhaustiveness",
            str(config["exhaustiveness"]),
            "--num_modes",
            str(config["num_modes"]),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=int(config["timeout"]),
            check=False,
        )
        score = parse_smina_score(result.stdout)
        if score is None:
            stderr = result.stderr.strip()[:120] if result.stderr else ""
            return index, molecule_id, None, f"PARSE_FAILED|rc={result.returncode}|{stderr}"
        return index, molecule_id, score, "OK"
    except subprocess.TimeoutExpired:
        return index, molecule_id, None, "TIMEOUT"
    except Exception as exc:
        return index, molecule_id, None, f"ERROR:{str(exc)[:120]}"
    finally:
        try:
            ligand_sdf.unlink(missing_ok=True)
        except Exception:
            pass


def read_smiles(path: Path, limit: int | None = None) -> list[tuple[int, str, str]]:
    molecules: list[tuple[int, str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            smiles = parts[0]
            molecule_id = parts[1] if len(parts) > 1 else str(index)
            molecules.append((index, smiles, molecule_id))
            if limit is not None and len(molecules) >= limit:
                break
    return molecules


def read_completed(output_csv: Path) -> set[int]:
    completed: set[int] = set()
    if not output_csv.exists():
        return completed
    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                completed.add(int(row["index"]))
            except (KeyError, ValueError):
                continue
    return completed


def append_rows(output_csv: Path, rows: list[tuple[int, str, float | None, str]]) -> None:
    exists = output_csv.exists()
    with output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["index", "molecule_id", "score", "status"])
        for index, molecule_id, score, status in rows:
            writer.writerow([index, molecule_id, "" if score is None else score, status])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dock a SMILES library against a prepared receptor with Smina.")
    parser.add_argument("--smina", required=True, help="Path to the smina executable.")
    parser.add_argument("--receptor", required=True, help="Prepared receptor PDBQT file.")
    parser.add_argument("--smiles", required=True, help="Input SMILES file. First column must be SMILES.")
    parser.add_argument("--output", default="docking_results.csv", help="Output CSV path.")
    parser.add_argument("--center", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--size", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--num-modes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--limit", type=int, help="Dock only the first N molecules.")
    parser.add_argument("--tmp-dir", default=str(Path(tempfile.gettempdir()) / "dc_pharmgat_docking"))
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing output rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    output_csv = Path(args.output)
    molecules = read_smiles(Path(args.smiles), limit=args.limit)
    completed = set() if args.no_resume else read_completed(output_csv)
    jobs = [
        (
            index,
            smiles,
            molecule_id,
            {
                "smina": Path(args.smina),
                "receptor": Path(args.receptor),
                "center": tuple(args.center),
                "size": tuple(args.size),
                "exhaustiveness": args.exhaustiveness,
                "num_modes": args.num_modes,
                "timeout": args.timeout,
                "tmp_dir": tmp_dir,
                "seed": args.seed,
            },
        )
        for index, smiles, molecule_id in molecules
        if index not in completed
    ]

    print(f"Input molecules: {len(molecules)}")
    print(f"Already completed: {len(completed)}")
    print(f"Queued: {len(jobs)}")
    if not jobs:
        return

    start = time.time()
    buffer: list[tuple[int, str, float | None, str]] = []
    with mp.Pool(processes=args.workers) as pool:
        for done, row in enumerate(pool.imap_unordered(dock_one, jobs), start=1):
            buffer.append(row)
            if len(buffer) >= args.save_interval:
                append_rows(output_csv, buffer)
                buffer.clear()
            if done % args.save_interval == 0:
                elapsed = time.time() - start
                print(f"Processed {done}/{len(jobs)} in {elapsed / 60:.1f} min")

    if buffer:
        append_rows(output_csv, buffer)
    print(f"Saved docking results to {output_csv}")


if __name__ == "__main__":
    main()
