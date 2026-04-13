#!/usr/bin/env python3
"""Download official LongBench subsets from Hugging Face and export as JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


DEFAULT_LONG_BENCH_SUBSETS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download LongBench from Hugging Face and dump each subset to JSONL."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="THUDM/LongBench",
        help="Dataset repo id on Hugging Face. Default: THUDM/LongBench",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to download. Default: test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/triattention/data/longbench_raw"),
        help="Output directory for exported jsonl files.",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default="",
        help="Comma-separated subset names. Empty means official LongBench full subset list.",
    )
    parser.add_argument(
        "--max-samples-per-subset",
        type=int,
        default=0,
        help="Limit samples per subset for quick smoke download. 0 means all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    return parser.parse_args()


def resolve_subsets(raw: str) -> List[str]:
    if raw.strip():
        return [x.strip() for x in raw.split(",") if x.strip()]
    return DEFAULT_LONG_BENCH_SUBSETS


def main() -> None:
    args = parse_args()
    subsets = resolve_subsets(args.subsets)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "Missing dependency `datasets`. Install it first:\n"
            "  pip install datasets\n"
            f"Import error: {e}"
        )

    print(f"[INFO] repo_id={args.repo_id}")
    print(f"[INFO] split={args.split}")
    print(f"[INFO] subsets={subsets}")
    print(f"[INFO] output_dir={args.output_dir.resolve()}")

    total_rows = 0
    for subset in subsets:
        out_path = args.output_dir / f"{subset}.{args.split}.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] exists: {out_path}")
            continue

        print(f"[LOAD] subset={subset}")
        try:
            ds = load_dataset(args.repo_id, subset, split=args.split)
        except RuntimeError as e:
            msg = str(e)
            if "Dataset scripts are no longer supported" in msg:
                raise SystemExit(
                    "Current `datasets` version does not support script-based LongBench loading.\n"
                    "Please reinstall to a compatible version, then rerun:\n"
                    "  pip uninstall -y datasets\n"
                    "  pip install datasets==3.6.0\n"
                    f"Original error: {e}"
                )
            raise
        if args.max_samples_per_subset > 0:
            keep = min(args.max_samples_per_subset, len(ds))
            ds = ds.select(range(keep))

        with out_path.open("w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        count = len(ds)
        total_rows += count
        print(f"[DONE] {subset}: {count} rows -> {out_path}")

    print(f"[SUMMARY] total rows exported: {total_rows}")


if __name__ == "__main__":
    main()
