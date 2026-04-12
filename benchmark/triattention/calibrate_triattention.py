#!/usr/bin/env python3
"""Simple coarse-to-fine calibration script for TriAttention parameters.

This script is intended for fast parameter range discovery on one GPU.
It compares each TriAttention config against a baseline (TriAttention disabled),
then reports:
1) accuracy drop (%),
2) serving throughput/latency metrics.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests


@dataclass(frozen=True)
class TriConfig:
    window_size: int
    notable_budget: int
    selection_interval: int

    def tag(self) -> str:
        return (
            f"w{self.window_size}_b{self.notable_budget}_i{self.selection_interval}"
        )


def parse_int_list(raw: str) -> List[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError(f"Invalid int list: {raw!r}")
    return values


def run_cmd(cmd: Sequence[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fout:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            stdout=fout,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}; see {log_path}"
        )


def wait_server_ready(host: str, port: int, timeout_s: int) -> None:
    url = f"http://{host}:{port}/v1/models"
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"Server did not become ready within {timeout_s}s: {url}")


def build_server_cmd(args: argparse.Namespace, cfg: Optional[TriConfig]) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.extra_server_args:
        cmd.extend(shlex.split(args.extra_server_args, posix=False))
    if cfg is not None:
        cmd.extend(
            [
                "--enable-triattention",
                "--triattention-window-size",
                str(cfg.window_size),
                "--triattention-notable-budget",
                str(cfg.notable_budget),
                "--triattention-selection-interval",
                str(cfg.selection_interval),
            ]
        )
    return cmd


def launch_server(
    args: argparse.Namespace, cfg: Optional[TriConfig], run_dir: Path
) -> subprocess.Popen:
    cmd = build_server_cmd(args, cfg)
    server_log = run_dir / "server.log"
    fout = server_log.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(args.workspace),
        stdout=fout,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_server_ready(args.host, args.port, args.server_timeout_s)
    except Exception:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def read_last_jsonl(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        raise ValueError(f"Empty jsonl file: {path}")
    return json.loads(lines[-1])


def eval_accuracy(args: argparse.Namespace, run_dir: Path) -> Optional[float]:
    if args.accuracy_mode == "none":
        return None
    result_jsonl = run_dir / "accuracy_result.jsonl"
    cmd = [
        sys.executable,
        "benchmark/gsm8k/bench_sglang.py",
        "--backend",
        "srt",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--num-questions",
        str(args.gsm8k_num_questions),
        "--num-shots",
        str(args.gsm8k_num_shots),
        "--parallel",
        str(args.gsm8k_parallel),
        "--temperature",
        "0",
        "--top-p",
        "1",
        "--result-file",
        str(result_jsonl),
    ]
    if args.gsm8k_platinum:
        cmd.append("--platinum")
    run_cmd(cmd, args.workspace, run_dir / "accuracy.log")
    payload = read_last_jsonl(result_jsonl)
    return float(payload["accuracy"])


def eval_perf(args: argparse.Namespace, run_dir: Path) -> Dict[str, float]:
    output_jsonl = run_dir / "perf_result.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dataset-name",
        args.perf_dataset,
        "--num-prompts",
        str(args.perf_num_prompts),
        "--output-file",
        str(output_jsonl),
        "--disable-tqdm",
    ]
    if args.perf_model:
        cmd.extend(["--model", args.perf_model])
    if args.perf_dataset == "random":
        cmd.extend(
            [
                "--random-input",
                str(args.perf_random_input),
                "--random-output",
                str(args.perf_random_output),
                "--random-range-ratio",
                str(args.perf_random_range_ratio),
            ]
        )
    if args.perf_request_rate > 0:
        cmd.extend(["--request-rate", str(args.perf_request_rate)])
    run_cmd(cmd, args.workspace, run_dir / "perf.log")
    payload = read_last_jsonl(output_jsonl)
    return {
        "request_throughput": float(payload["request_throughput"]),
        "output_throughput": float(payload["output_throughput"]),
        "median_ttft_ms": float(payload["median_ttft_ms"]),
        "median_itl_ms": float(payload["median_itl_ms"]),
    }


def accuracy_drop_pct(base_acc: Optional[float], acc: Optional[float]) -> Optional[float]:
    if base_acc is None or acc is None:
        return None
    if base_acc <= 0:
        return None
    return max(0.0, (base_acc - acc) / base_acc * 100.0)


def evaluate_one(
    args: argparse.Namespace, cfg: Optional[TriConfig], stage: str, run_idx: int
) -> Dict[str, Any]:
    run_name = "baseline" if cfg is None else f"{stage}_{run_idx:03d}_{cfg.tag()}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    proc = launch_server(args, cfg, run_dir)
    try:
        acc = eval_accuracy(args, run_dir)
        perf = eval_perf(args, run_dir)
    finally:
        stop_server(proc)

    return {
        "run_name": run_name,
        "stage": stage,
        "is_baseline": cfg is None,
        "config": None
        if cfg is None
        else {
            "window_size": cfg.window_size,
            "notable_budget": cfg.notable_budget,
            "selection_interval": cfg.selection_interval,
        },
        "accuracy": acc,
        **perf,
    }


def dedup_configs(configs: Iterable[TriConfig]) -> List[TriConfig]:
    seen = set()
    ret: List[TriConfig] = []
    for c in configs:
        key = (c.window_size, c.notable_budget, c.selection_interval)
        if key in seen:
            continue
        seen.add(key)
        ret.append(c)
    return ret


def expand_fine_configs(
    seeds: Sequence[TriConfig],
    window_deltas: Sequence[int],
    budget_deltas: Sequence[int],
    interval_deltas: Sequence[int],
) -> List[TriConfig]:
    fine: List[TriConfig] = []
    for seed in seeds:
        for dw, db, di in itertools.product(
            window_deltas, budget_deltas, interval_deltas
        ):
            w = seed.window_size + dw
            b = seed.notable_budget + db
            itv = seed.selection_interval + di
            if w <= 0 or b <= 0 or itv <= 0:
                continue
            fine.append(TriConfig(w, b, itv))
    return dedup_configs(fine)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coarse-to-fine calibration for TriAttention params."
    )
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--server-timeout-s", type=int, default=180)
    parser.add_argument("--extra-server-args", type=str, default="")

    parser.add_argument(
        "--accuracy-mode",
        type=str,
        default="gsm8k",
        choices=["none", "gsm8k"],
        help="Use gsm8k benchmark for accuracy gate, or skip with 'none'.",
    )
    parser.add_argument("--gsm8k-num-questions", type=int, default=200)
    parser.add_argument("--gsm8k-num-shots", type=int, default=5)
    parser.add_argument("--gsm8k-parallel", type=int, default=32)
    parser.add_argument("--gsm8k-platinum", action="store_true")
    parser.add_argument(
        "--max-accuracy-drop-pct",
        type=float,
        default=1.0,
        help="Maximum allowed relative accuracy drop (%) versus baseline.",
    )

    parser.add_argument("--perf-dataset", type=str, default="random")
    parser.add_argument("--perf-model", type=str, default="")
    parser.add_argument("--perf-num-prompts", type=int, default=500)
    parser.add_argument("--perf-random-input", type=int, default=4096)
    parser.add_argument("--perf-random-output", type=int, default=256)
    parser.add_argument("--perf-random-range-ratio", type=float, default=1.0)
    parser.add_argument("--perf-request-rate", type=float, default=0.0)

    parser.add_argument("--window-candidates", type=str, default="64,128,256")
    parser.add_argument("--budget-candidates", type=str, default="128,256,384")
    parser.add_argument("--interval-candidates", type=str, default="16,32,64")
    parser.add_argument("--fine-around-topk", type=int, default=2)
    parser.add_argument("--fine-window-deltas", type=str, default="0,-32,32")
    parser.add_argument("--fine-budget-deltas", type=str, default="0,-64,64")
    parser.add_argument("--fine-interval-deltas", type=str, default="0,-16,16")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/triattention/results"),
    )

    args = parser.parse_args()
    args.workspace = args.workspace.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coarse_configs = dedup_configs(
        TriConfig(w, b, i)
        for w, b, i in itertools.product(
            parse_int_list(args.window_candidates),
            parse_int_list(args.budget_candidates),
            parse_int_list(args.interval_candidates),
        )
    )
    print(f"[INFO] Coarse configs: {len(coarse_configs)}")

    results: List[Dict[str, Any]] = []

    baseline = evaluate_one(args, cfg=None, stage="baseline", run_idx=0)
    results.append(baseline)
    base_acc = baseline.get("accuracy")
    print(
        "[BASELINE] "
        f"accuracy={base_acc}, output_tps={baseline['output_throughput']:.2f}, "
        f"median_itl_ms={baseline['median_itl_ms']:.2f}"
    )

    for idx, cfg in enumerate(coarse_configs, start=1):
        row = evaluate_one(args, cfg=cfg, stage="coarse", run_idx=idx)
        row["accuracy_drop_pct"] = accuracy_drop_pct(base_acc, row.get("accuracy"))
        drop = row["accuracy_drop_pct"]
        row["accuracy_gate_pass"] = (
            True if drop is None else drop <= args.max_accuracy_drop_pct
        )
        results.append(row)
        print(
            f"[COARSE {idx}/{len(coarse_configs)}] {cfg.tag()} "
            f"acc={row.get('accuracy')} drop={drop} "
            f"out_tps={row['output_throughput']:.2f} gate={row['accuracy_gate_pass']}"
        )

    coarse_pass = [r for r in results if r["stage"] == "coarse" and r["accuracy_gate_pass"]]
    coarse_pass = sorted(coarse_pass, key=lambda x: x["output_throughput"], reverse=True)
    topk = coarse_pass[: max(0, args.fine_around_topk)]
    seed_cfgs = [
        TriConfig(
            x["config"]["window_size"],
            x["config"]["notable_budget"],
            x["config"]["selection_interval"],
        )
        for x in topk
    ]

    fine_configs = expand_fine_configs(
        seed_cfgs,
        parse_int_list(args.fine_window_deltas),
        parse_int_list(args.fine_budget_deltas),
        parse_int_list(args.fine_interval_deltas),
    )
    coarse_set = {
        (c.window_size, c.notable_budget, c.selection_interval) for c in coarse_configs
    }
    fine_configs = [
        c
        for c in fine_configs
        if (c.window_size, c.notable_budget, c.selection_interval) not in coarse_set
    ]
    print(f"[INFO] Fine configs: {len(fine_configs)}")

    for idx, cfg in enumerate(fine_configs, start=1):
        row = evaluate_one(args, cfg=cfg, stage="fine", run_idx=idx)
        row["accuracy_drop_pct"] = accuracy_drop_pct(base_acc, row.get("accuracy"))
        drop = row["accuracy_drop_pct"]
        row["accuracy_gate_pass"] = (
            True if drop is None else drop <= args.max_accuracy_drop_pct
        )
        results.append(row)
        print(
            f"[FINE {idx}/{len(fine_configs)}] {cfg.tag()} "
            f"acc={row.get('accuracy')} drop={drop} "
            f"out_tps={row['output_throughput']:.2f} gate={row['accuracy_gate_pass']}"
        )

    feasible = [
        x for x in results if (not x["is_baseline"]) and x.get("accuracy_gate_pass", True)
    ]
    feasible = sorted(feasible, key=lambda x: x["output_throughput"], reverse=True)
    best = feasible[0] if feasible else None

    summary = {
        "baseline": baseline,
        "num_runs": len(results),
        "num_feasible": len(feasible),
        "best_feasible": best,
    }

    results_path = args.output_dir / "all_results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[DONE] wrote: {results_path}")
    print(f"[DONE] wrote: {summary_path}")
    if best:
        print(
            "[BEST] "
            f"{best['config']} output_tps={best['output_throughput']:.2f}, "
            f"median_itl_ms={best['median_itl_ms']:.2f}, "
            f"accuracy_drop_pct={best.get('accuracy_drop_pct')}"
        )
    else:
        print("[BEST] no feasible TriAttention config under current accuracy threshold.")


if __name__ == "__main__":
    main()

