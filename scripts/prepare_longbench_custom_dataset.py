#!/usr/bin/env python3
"""Convert official LongBench data into sglang bench_serving custom JSONL format.

Output format (one JSON object per line):
{
  "id": "...",
  "source_dataset": "...",
  "conversations": [
    {"from": "human", "content": "<prompt>"},
    {"from": "gpt", "content": "<reference answer or placeholder>"}
  ]
}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare LongBench data for `python -m sglang.bench_serving --dataset-name custom`."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="LongBench source path. Supports a JSON/JSONL file or a directory of JSON/JSONL files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output custom JSONL path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum output rows. 0 means no limit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260413,
        help="Random seed for shuffling.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples before writing output.",
    )
    parser.add_argument(
        "--require-context",
        action="store_true",
        help="Keep only rows with non-empty `context` field.",
    )
    parser.add_argument(
        "--dataset-filter",
        type=str,
        default="",
        help="Comma-separated dataset names to keep (match LongBench `dataset` field or input filename stem).",
    )
    return parser.parse_args()


def iter_source_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    files = sorted(
        [
            p
            for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jsonl", ".json"}
        ]
    )
    if not files:
        raise FileNotFoundError(f"No json/jsonl files found under: {input_path}")
    return files


def load_records(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
        return

    # .json path
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
    elif isinstance(obj, dict):
        yield obj


def stringify_answer(item: Dict[str, Any]) -> str:
    candidates = item.get("answers")
    if isinstance(candidates, list):
        for c in candidates:
            if c is None:
                continue
            text = str(c).strip()
            if text:
                return text

    for key in ("answer", "output", "target"):
        if key in item and item[key] is not None:
            text = str(item[key]).strip()
            if text:
                return text
    # For throughput-only benchmarking, placeholder answer is acceptable.
    return "N/A"


def build_prompt(item: Dict[str, Any], source_dataset: str) -> str:
    context = str(item.get("context", "") or "").strip()
    question = str(item.get("input", "") or "").strip()
    instruction = str(item.get("instruction", "") or "").strip()

    parts: List[str] = []
    parts.append(
        "You are given a long context and a question. Read the context and answer the question."
    )
    parts.append(f"[Dataset]\n{source_dataset}")
    if instruction:
        parts.append(f"[Instruction]\n{instruction}")
    if context:
        parts.append(f"[Context]\n{context}")
    if question:
        parts.append(f"[Question]\n{question}")
    parts.append("[Answer]\n")
    return "\n\n".join(parts)


def convert_one(
    item: Dict[str, Any],
    source_file: Path,
    idx: int,
) -> Tuple[Dict[str, Any], str, bool]:
    dataset_name = str(item.get("dataset", "") or source_file.stem)
    has_context = bool(str(item.get("context", "") or "").strip())
    prompt = build_prompt(item, dataset_name)
    answer = stringify_answer(item)
    output = {
        "id": f"{dataset_name}-{idx}",
        "source_dataset": dataset_name,
        "conversations": [
            {"from": "human", "content": prompt},
            {"from": "gpt", "content": answer},
        ],
    }
    return output, dataset_name, has_context


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    dataset_filter = {
        x.strip() for x in args.dataset_filter.split(",") if x.strip()
    } or None

    source_files = iter_source_files(args.input_path)
    converted: List[Dict[str, Any]] = []
    kept_per_dataset: Dict[str, int] = {}
    dropped_context = 0
    scanned = 0

    for source in source_files:
        for idx, item in enumerate(load_records(source)):
            scanned += 1
            row, dataset_name, has_context = convert_one(item, source, idx)

            if dataset_filter and dataset_name not in dataset_filter:
                continue
            if args.require_context and not has_context:
                dropped_context += 1
                continue

            converted.append(row)
            kept_per_dataset[dataset_name] = kept_per_dataset.get(dataset_name, 0) + 1

    if args.shuffle:
        random.shuffle(converted)
    if args.max_samples > 0:
        converted = converted[: args.max_samples]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        for row in converted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[DONE] wrote {len(converted)} rows to {args.output_path}")
    print(f"[INFO] scanned rows: {scanned}")
    if args.require_context:
        print(f"[INFO] dropped by empty context: {dropped_context}")
    print("[INFO] kept rows by dataset:")
    for name in sorted(kept_per_dataset.keys()):
        print(f"  - {name}: {kept_per_dataset[name]}")


if __name__ == "__main__":
    main()
