#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class BenchCase:
    context: str
    concurrency: int
    input_len: int
    output_len: int
    num_prompts: int


METRICS = [
    "request_throughput",
    "output_throughput",
    "mean_ttft_ms",
    "mean_itl_ms",
    "mean_e2e_latency_ms",
]


def _parse_concurrency_list(value: str) -> List[int]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    vals = [int(x) for x in items]
    if not vals or any(v <= 0 for v in vals):
        raise ValueError("--concurrency-list must be comma-separated positive ints")
    return vals


def _write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _to_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _run_command(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed with exit code {proc.returncode}")


def _load_last_jsonl(path: Path) -> Dict:
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        raise RuntimeError(f"No records in {path}")
    return json.loads(lines[-1])


def _build_cases(args: argparse.Namespace) -> List[BenchCase]:
    cs = _parse_concurrency_list(args.concurrency_list)
    cases: List[BenchCase] = []
    for c in cs:
        n = c * args.request_multiplier
        cases.append(BenchCase("short", c, args.short_input_len, args.short_output_len, n))
        cases.append(BenchCase("long", c, args.long_input_len, args.long_output_len, n))
    return cases


def _run_mode(args: argparse.Namespace) -> None:
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("benchmark/context_compare") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    for case in _build_cases(args):
        raw = out_dir / "raw" / f"{case.context}_c{case.concurrency}.jsonl"
        cmd = [
            sys.executable,
            "-m",
            "sglang.bench_serving",
            "--backend",
            args.backend,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--dataset-name",
            "random",
            "--num-prompts",
            str(case.num_prompts),
            "--random-input-len",
            str(case.input_len),
            "--random-output-len",
            str(case.output_len),
            "--request-rate",
            args.request_rate,
            "--max-concurrency",
            str(case.concurrency),
            "--output-file",
            str(raw),
            "--tag",
            f"sparseattention_{case.context}_c{case.concurrency}",
        ]
        if args.model:
            cmd += ["--model", args.model]
        _run_command(cmd)
        result = _load_last_jsonl(raw)
        row = {
            "context": case.context,
            "concurrency": case.concurrency,
            "input_len": case.input_len,
            "output_len": case.output_len,
            "num_prompts": case.num_prompts,
            "raw_file": str(raw),
        }
        row.update(result)
        rows.append(row)
    summary = out_dir / "runs_summary.csv"
    _write_csv(summary, rows)
    print(f"Saved run summary: {summary}")


def _compare_mode(args: argparse.Namespace) -> None:
    baseline = Path(args.baseline_dir)
    optimized = Path(args.optimized_dir)
    b_rows = _read_csv(baseline / "runs_summary.csv")
    o_rows = _read_csv(optimized / "runs_summary.csv")

    def key_of(r: Dict[str, str]) -> Tuple[str, str]:
        return (r.get("context", ""), r.get("concurrency", ""))

    b_map = {key_of(r): r for r in b_rows}
    o_map = {key_of(r): r for r in o_rows}
    keys = sorted(set(b_map.keys()) & set(o_map.keys()))
    if not keys:
        raise RuntimeError("No matched (context, concurrency) rows between baseline and optimized")

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else optimized / f"ab_vs_{baseline.name}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    comp_rows: List[Dict] = []
    for k in keys:
        b = b_map[k]
        o = o_map[k]
        row: Dict[str, object] = {
            "context": k[0],
            "concurrency": int(k[1]),
        }
        for m in METRICS:
            bv = _to_float(b.get(m))
            ov = _to_float(o.get(m))
            row[f"baseline_{m}"] = bv
            row[f"{args.optimized_label}_{m}"] = ov
            row[f"delta_{m}"] = None if (bv is None or ov is None) else (ov - bv)
            if bv is None or ov is None or bv == 0:
                row[f"ratio_{m}"] = None
            else:
                row[f"ratio_{m}"] = ov / bv
        comp_rows.append(row)

    _write_csv(out_dir / "ab_comparison.csv", comp_rows)

    md_lines = [
        "# Baseline vs " + args.optimized_label,
        "",
        "| context | c | req/s ratio | out tok/s ratio | ttft ratio | itl ratio | e2e ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in comp_rows:
        md_lines.append(
            "| "
            + f"{r['context']} | {r['concurrency']} | "
            + f"{(r.get('ratio_request_throughput') or 0):.3f} | "
            + f"{(r.get('ratio_output_throughput') or 0):.3f} | "
            + f"{(r.get('ratio_mean_ttft_ms') or 0):.3f} | "
            + f"{(r.get('ratio_mean_itl_ms') or 0):.3f} | "
            + f"{(r.get('ratio_mean_e2e_latency_ms') or 0):.3f} |"
        )
    (out_dir / "ab_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved compare results: {out_dir / 'ab_comparison.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark and compare SparseAttention runs")
    sub = parser.add_subparsers(dest="mode", required=True)

    run = sub.add_parser("run")
    run.add_argument("--backend", default="sglang")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=30000)
    run.add_argument("--model", default=None)
    run.add_argument("--request-rate", default="inf")
    run.add_argument("--concurrency-list", default="1,10,20")
    run.add_argument("--request-multiplier", type=int, default=10)
    run.add_argument("--short-input-len", type=int, default=1024)
    run.add_argument("--short-output-len", type=int, default=100)
    run.add_argument("--long-input-len", type=int, default=30000)
    run.add_argument("--long-output-len", type=int, default=1000)
    run.add_argument("--output-dir", default=None)

    compare = sub.add_parser("compare")
    compare.add_argument("--baseline-dir", required=True)
    compare.add_argument("--optimized-dir", required=True)
    compare.add_argument("--optimized-label", default="sparseattention")
    compare.add_argument("--output-dir", default=None)

    args = parser.parse_args()
    if args.mode == "run":
        _run_mode(args)
    else:
        _compare_mode(args)


if __name__ == "__main__":
    main()
