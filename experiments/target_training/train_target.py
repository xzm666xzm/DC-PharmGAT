#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic target-training workflow for DC-PharmGAT.

This script replaces the target-specific train_<target>_*.py copies. It keeps
the workflow configurable by command-line arguments so the repository can show
representative training code without hard-coding a particular biological target.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


TARGET_TRAINING_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = TARGET_TRAINING_DIR.parent
REPO_ROOT = EXPERIMENTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"

HYPERPARAMETERS = [
    (1, 0.001, 0.0001, 128),
    (2, 0.001, 0.0001, 256),
    (3, 0.001, 0.0, 128),
    (4, 0.001, 0.0, 256),
    (5, 0.001, 0.001, 128),
    (6, 0.001, 0.001, 256),
    (7, 0.0001, 0.0001, 128),
    (8, 0.0001, 0.0001, 256),
    (9, 0.0001, 0.0, 128),
    (10, 0.0001, 0.0, 256),
    (11, 0.0001, 0.001, 128),
    (12, 0.0001, 0.001, 256),
    (13, 0.0003, 0.0001, 128),
    (14, 0.0003, 0.0001, 256),
    (15, 0.0003, 0.0, 128),
    (16, 0.0003, 0.0, 256),
    (17, 0.0003, 0.001, 128),
    (18, 0.0003, 0.001, 256),
]

HP_RESULTS_HEADER = (
    "model number,oversampled size,batch size,learning rate,dropout rate,gfe threshold,"
    "fingerprint length,validation auc,validation prauc,validation precision,"
    "validation recall,validation f1,validation hits,tr,test auc,test prauc,"
    "test precision,test recall,test f1,test hits,avg gfe,t1enrichment,"
    "t5enrichment,t10enrichment,t50enrichment,t100enrichment\n"
)


def parse_model_selection(value: str) -> set[int]:
    selected: set[int] = set()
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(chunk))
    return selected


def hyperparameters_for_model(model_number: int) -> tuple[int, float, float, int]:
    for params in HYPERPARAMETERS:
        if params[0] == model_number:
            return params
    raise ValueError(f"No hyperparameters found for model {model_number}")


def resolve_path(path: str | None, base: Path = TARGET_TRAINING_DIR) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    local = base / candidate
    if local.exists():
        return local
    return REPO_ROOT / candidate


def as_cli_path(path: Path) -> str:
    try:
        return str(path.relative_to(TARGET_TRAINING_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    extra_paths = [str(SRC_DIR), str(EXPERIMENTS_DIR)]
    existing = env.get("PYTHONPATH")
    if existing:
        extra_paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(extra_paths)
    return env


def run_command(cmd: list[str], cwd: Path, dry_run: bool = False) -> subprocess.CompletedProcess[str]:
    print(" ".join(cmd))
    if dry_run:
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=command_env(),
        text=True,
        capture_output=True,
        check=True,
    )


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def find_source_dir(data_root: Path, target: str, target_folder: str | None, required: list[str]) -> Path:
    target_variants = [target, target.lower(), target.upper()]
    folder_variants = [target_folder] if target_folder else []
    folder_variants += target_variants

    candidates: list[Path] = []
    for folder in folder_variants:
        if folder:
            for target_name in target_variants:
                candidates.append(data_root / folder / target_name)
            candidates.append(data_root / folder)
    for target_name in target_variants:
        candidates.append(data_root / target_name)
    candidates.append(data_root)

    for candidate in candidates:
        if all((candidate / name).exists() for name in required):
            return candidate

    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find a data directory containing "
        f"{', '.join(required)}. Searched:\n  - {searched}"
    )


def copy_required_files(source_dir: Path, output_dir: Path, filenames: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        shutil.copy2(source_dir / filename, output_dir / filename)


def prepare_complete_dataset(args: argparse.Namespace, run_name: str) -> str:
    if args.dataset == "dude":
        source_dir = find_source_dir(
            args.data_root,
            args.target,
            args.target_folder,
            ["actives_final.ism", "decoys_final.ism"],
        )
        output_dir = TARGET_TRAINING_DIR / "dude" / run_name
        copy_required_files(source_dir, output_dir, ["actives_final.ism", "decoys_final.ism"])
        print(f"Prepared DUD-E data in {output_dir}")
        return "dude"

    source_dir = find_source_dir(
        args.data_root,
        args.target,
        args.target_folder,
        ["actives.smi", "inactives.smi"],
    )
    output_dir = TARGET_TRAINING_DIR / "lit-pcba" / run_name.upper()
    copy_required_files(source_dir, output_dir, ["actives.smi", "inactives.smi"])
    print(f"Prepared LIT-PCBA data in {output_dir}")
    return "lit-pcba"


def prepare_scaffold_dataset(args: argparse.Namespace, run_name: str) -> str:
    source_dir = find_source_dir(
        args.data_root,
        args.target,
        args.target_folder,
        ["train.ism", "valid.ism", "test.ism"],
    )
    output_dir = TARGET_TRAINING_DIR / "dude" / run_name
    copy_required_files(source_dir, output_dir, ["train.ism", "valid.ism", "test.ism"])

    actives_source = first_existing(
        [
            source_dir / "actives_dedup.ism",
            source_dir / "actives_final.ism",
            source_dir / "actives.ism",
        ]
    )
    if actives_source:
        shutil.copy2(actives_source, output_dir / "actives_final.ism")

    print(f"Prepared scaffold-split data in {output_dir}")
    return "scaffold"


def extract_pharmacophore(args: argparse.Namespace, run_name: str, data_source_dir: Path | None) -> None:
    output_dir = TARGET_TRAINING_DIR / run_name
    tpp_pt = output_dir / f"{run_name}_tpp.pt"
    if tpp_pt.exists() and not args.force:
        print(f"TPP file already exists: {tpp_pt}")
        return

    candidates: list[Path] = []
    if args.pharmacophore_input:
        candidates.append(args.pharmacophore_input)
    if data_source_dir:
        candidates += [
            data_source_dir / "crystal_ligand.mol2",
            data_source_dir / "actives_dedup.ism",
            data_source_dir / "actives_final.ism",
            data_source_dir / "actives.smi",
        ]

    input_path = first_existing(candidates)
    if input_path is None:
        print("No pharmacophore input found; training will use train.py defaults.")
        return

    cmd = [
        args.python,
        str(SRC_DIR / "extract_pharmacophore.py"),
        "--input",
        as_cli_path(input_path),
        "--output",
        run_name,
        "--name",
        run_name,
    ]
    result = run_command(cmd, cwd=TARGET_TRAINING_DIR, dry_run=args.dry_run)
    if result.stdout:
        print(result.stdout)


def clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def initialize_training_workspace(args: argparse.Namespace, run_name: str) -> None:
    """Create the output layout expected by experiments/train.py."""
    run_dir = TARGET_TRAINING_DIR / run_name
    generated_dirs = [
        run_dir / "trainingJobs",
        run_dir / "logs",
        run_dir / "models",
        run_dir / "res",
    ]

    if args.dry_run:
        print(f"Would initialize training workspace: {run_dir}")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    for directory in generated_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        if args.force:
            clear_directory(directory)

    hp_results = run_dir / "hpResults.csv"
    if args.force or not hp_results.exists():
        hp_results.write_text(HP_RESULTS_HEADER, encoding="utf-8")


def run_training(args: argparse.Namespace, run_name: str, train_dataset: str) -> list[int]:
    selected = parse_model_selection(args.models)
    training_jobs_dir = TARGET_TRAINING_DIR / run_name / "trainingJobs"
    successes: list[int] = []

    for model_num, lr, wd, batch_size in HYPERPARAMETERS:
        if model_num not in selected:
            continue

        cmd = [
            args.python,
            str(EXPERIMENTS_DIR / "train.py"),
            "-dropout",
            str(args.dropout),
            "-learn_rate",
            str(lr),
            "-os",
            str(args.oversampling),
            "-bs",
            str(batch_size),
            "-protein",
            run_name,
            "-fplen",
            str(args.fingerprint_length),
            "-wd",
            str(wd),
            "-mnum",
            str(model_num),
            "-dataset",
            train_dataset,
            "--skip_test_eval",
        ]

        print(f"Training model {model_num}: lr={lr}, wd={wd}, batch_size={batch_size}")
        start = time.time()
        try:
            result = run_command(cmd, cwd=training_jobs_dir, dry_run=args.dry_run)
            if result.stdout and args.verbose:
                print(result.stdout)
            successes.append(model_num)
            print(f"Model {model_num} finished in {(time.time() - start) / 60:.1f} min")
        except subprocess.CalledProcessError as exc:
            print(f"Model {model_num} failed with exit code {exc.returncode}")
            if exc.stdout:
                print(exc.stdout)
            if exc.stderr:
                print(exc.stderr)
            if not args.keep_going:
                raise

    return successes


def summarize_results(run_name: str) -> int | None:
    results_file = TARGET_TRAINING_DIR / run_name / "hpResults.csv"
    if not results_file.exists():
        print(f"No result file found: {results_file}")
        return None

    df = pd.read_csv(results_file)
    if df.empty or "validation auc" not in df.columns:
        print(f"Result file has no usable validation auc column: {results_file}")
        return None

    df_valid = df.dropna(subset=["validation auc"])
    if df_valid.empty:
        print(f"Result file has no validation AUC values: {results_file}")
        return None

    df_sorted = df_valid.sort_values("validation auc", ascending=False)
    print("\nTop models by validation AUC:")
    for _, row in df_sorted.head(5).iterrows():
        print(
            f"  model {int(row['model number'])}: "
            f"val_auc={float(row['validation auc']):.4f}, "
            f"val_prauc={float(row.get('validation prauc', 0.0)):.4f}"
        )

    best = df_sorted.iloc[0]
    best_model = int(best["model number"])
    _, best_lr, best_wd, best_batch_size = hyperparameters_for_model(best_model)
    best_json = TARGET_TRAINING_DIR / run_name / "best_stage1.json"
    with best_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_number": best_model,
                "selection_metric": "validation_auc",
                "selection_validation_auc": float(best["validation auc"]),
                "validation_auc": float(best["validation auc"]),
                "validation_prauc": float(best.get("validation prauc", 0.0)),
                "final_test_auc": None,
                "final_test_prauc": None,
                "lr": best_lr,
                "wd": best_wd,
                "bs": best_batch_size,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            handle,
            indent=2,
        )

    best_model_path = TARGET_TRAINING_DIR / run_name / "models" / f"model{best_model}.pth"
    best_stage_path = TARGET_TRAINING_DIR / run_name / "best_stage1.pth"
    if best_model_path.exists():
        shutil.copy2(best_model_path, best_stage_path)

    print(f"Best validation-selected model: {best_model}")
    return best_model


def run_final_test(args: argparse.Namespace, run_name: str, train_dataset: str, best_model: int) -> bool:
    model_num, lr, wd, batch_size = hyperparameters_for_model(best_model)
    training_jobs_dir = TARGET_TRAINING_DIR / run_name / "trainingJobs"
    cmd = [
        args.python,
        str(EXPERIMENTS_DIR / "train.py"),
        "-dropout",
        str(args.dropout),
        "-learn_rate",
        str(lr),
        "-os",
        str(args.oversampling),
        "-bs",
        str(batch_size),
        "-protein",
        run_name,
        "-fplen",
        str(args.fingerprint_length),
        "-wd",
        str(wd),
        "-mnum",
        str(model_num),
        "-dataset",
        train_dataset,
        "--final_test_only",
    ]

    print(f"\nFinal held-out test evaluation for validation-selected model {model_num}")
    try:
        result = run_command(cmd, cwd=training_jobs_dir, dry_run=args.dry_run)
        if result.stdout and args.verbose:
            print(result.stdout)
    except subprocess.CalledProcessError as exc:
        print(f"Final test evaluation failed with exit code {exc.returncode}")
        if exc.stdout:
            print(exc.stdout)
        if exc.stderr:
            print(exc.stderr)
        if not args.keep_going:
            raise
        return False

    if args.dry_run:
        return True

    results_file = TARGET_TRAINING_DIR / run_name / "hpResults.csv"
    best_json = TARGET_TRAINING_DIR / run_name / "best_stage1.json"
    df = pd.read_csv(results_file)
    rows = df[(df["model number"] == model_num) & df["test auc"].notna()]
    if rows.empty:
        print("Final test row was not found in hpResults.csv")
        return False

    final_row = rows.iloc[-1]
    record: dict[str, object] = {}
    if best_json.exists():
        with best_json.open("r", encoding="utf-8") as handle:
            record = json.load(handle)

    record.update(
        {
            "model_number": model_num,
            "selection_metric": record.get("selection_metric", "validation_auc"),
            "final_test_auc": float(final_row["test auc"]),
            "final_test_prauc": float(final_row["test prauc"]),
            "test_auc": float(final_row["test auc"]),
            "test_prauc": float(final_row["test prauc"]),
            "final_test_updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    with best_json.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)

    print(
        f"Final test metrics: auc={record['final_test_auc']:.4f}, "
        f"prauc={record['final_test_prauc']:.4f}"
    )
    return True


def build_parser(default_split_mode: str = "complete") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a generic DC-PharmGAT target-training workflow.")
    parser.add_argument("--target", required=True, help="Target identifier in the dataset directory.")
    parser.add_argument(
        "--target-folder",
        help="Dataset subfolder name when it differs from --target, for example uppercase LIT-PCBA folders.",
    )
    parser.add_argument(
        "--output-name",
        help="Run/output folder name. Defaults to --target for complete and <target>_scaffold for scaffold.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["complete", "scaffold"],
        default=default_split_mode,
        help="Use random complete data or precomputed scaffold-split data.",
    )
    parser.add_argument(
        "--dataset",
        choices=["dude", "lit-pcba"],
        default="dude",
        help="Complete-split dataset format. Scaffold mode always trains with scaffold data.",
    )
    parser.add_argument(
        "--data-root",
        default="DUD-E",
        help="Dataset root. Relative paths are resolved from experiments/target_training, then repository root.",
    )
    parser.add_argument("--pharmacophore-input", help="Optional ligand or active-molecule file for TPP extraction.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for subprocesses.")
    parser.add_argument("--models", default="1-18", help="Model numbers to train, e.g. 1-3,8,12.")
    parser.add_argument("--fingerprint-length", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--oversampling", type=int, default=25)
    parser.add_argument("--skip-tpp", action="store_true", help="Skip pharmacophore extraction.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--force", action="store_true", help="Overwrite reusable generated files when possible.")
    parser.add_argument("--keep-going", action="store_true", help="Continue when one model fails.")
    parser.add_argument("--verbose", action="store_true", help="Print full training subprocess output.")
    return parser


def main(default_split_mode: str = "complete") -> None:
    parser = build_parser(default_split_mode)
    args = parser.parse_args()
    args.data_root = resolve_path(args.data_root) or TARGET_TRAINING_DIR
    args.pharmacophore_input = resolve_path(args.pharmacophore_input)

    run_name = args.output_name
    if not run_name:
        run_name = args.target if args.split_mode == "complete" else f"{args.target}_scaffold"
    run_name = run_name.lower()

    if args.split_mode == "scaffold":
        train_dataset = prepare_scaffold_dataset(args, run_name)
    else:
        train_dataset = prepare_complete_dataset(args, run_name)

    data_source_dir = None
    try:
        if args.split_mode == "scaffold":
            data_source_dir = find_source_dir(
                args.data_root,
                args.target,
                args.target_folder,
                ["train.ism", "valid.ism", "test.ism"],
            )
        elif args.dataset == "dude":
            data_source_dir = find_source_dir(
                args.data_root,
                args.target,
                args.target_folder,
                ["actives_final.ism", "decoys_final.ism"],
            )
        else:
            data_source_dir = find_source_dir(
                args.data_root,
                args.target,
                args.target_folder,
                ["actives.smi", "inactives.smi"],
            )
    except FileNotFoundError:
        data_source_dir = None

    if not args.skip_tpp:
        extract_pharmacophore(args, run_name, data_source_dir)
    initialize_training_workspace(args, run_name)

    successes = run_training(args, run_name, train_dataset)
    print(f"\nSuccessful models: {len(successes)}")
    best_model = summarize_results(run_name)
    if best_model is not None:
        run_final_test(args, run_name, train_dataset, best_model)


if __name__ == "__main__":
    main()
