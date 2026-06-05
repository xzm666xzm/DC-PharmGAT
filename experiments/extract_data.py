#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract scaffold-split preprocessing statistics from report files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_TARGETS = [
    "MAPK1",
    "PKM2",
    "KAT2A",
    "FEN1",
    "ALDH1",
    "GBA",
    "OPRK1",
    "TP53",
    "PPARG",
    "ESR1_ant",
    "MTORC1",
    "IDH1",
]


def extract(pattern: str, content: str) -> str:
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).strip() if match else "N/A"


def parse_report(report_path: Path) -> dict[str, str]:
    content = report_path.read_text(encoding="utf-8", errors="ignore")
    return {
        "original active": extract(r"active molecules \(Actives\).*?original count:\s+(\d+)", content),
        "original inactive": extract(r"inactive molecules \(Decoys\).*?original count:\s+(\d+)", content),
        "after deduplication active": extract(r"active molecules \(Actives\).*?after deduplication:\s+(\d+)", content),
        "after deduplication inactive": extract(r"inactive molecules \(Decoys\).*?after deduplication:\s+(\d+)", content),
        "train active": extract(r"training set:.*?active:\s*(\d+)", content),
        "valid active": extract(r"validation set:.*?active:\s*(\d+)", content),
        "test active": extract(r"test set:.*?active:\s*(\d+)", content),
        "scaffold overlap": extract(r"training-test scaffold overlap:\s*(\d+)", content),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LIT-PCBA preprocessing reports.")
    parser.add_argument("--data-root", required=True, help="Directory containing target/processed/preprocessing_report.txt.")
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    columns = [
        "Target",
        "original active",
        "original inactive",
        "after deduplication active",
        "after deduplication inactive",
        "train active",
        "valid active",
        "test active",
        "scaffold overlap",
    ]
    print(" | ".join(columns))
    print("-" * 140)
    for target in args.targets:
        report_path = data_root / target / "processed" / "preprocessing_report.txt"
        if not report_path.exists():
            print(f"{target} | report not found")
            continue
        row = {"Target": target, **parse_report(report_path)}
        print(" | ".join(row[col] for col in columns))


if __name__ == "__main__":
    main()
