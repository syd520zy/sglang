"""Compare SenseNova SRT tuning runs and select by concurrency-2 throughput."""

import argparse
import json
from pathlib import Path


def build_report(root: Path):
    runs = []
    for run_dir in sorted(root.glob("graph-*")):
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            runs.append(
                {
                    "directory": str(run_dir),
                    "runtime_checks_passed": False,
                    "thinking_failures": None,
                    "c2_throughput_req_s": 0.0,
                    "error": "run did not produce summary.json",
                }
            )
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runtime_path = summary_path.with_name("srt-runtime-summary.json")
        if not runtime_path.is_file():
            runs.append(
                {
                    "directory": str(run_dir),
                    "runtime_checks_passed": False,
                    "thinking_failures": None,
                    "c2_throughput_req_s": 0.0,
                    "error": "run did not produce srt-runtime-summary.json",
                }
            )
            continue
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        cases = summary["cases"]
        think_cases = {
            name: value for name, value in cases.items() if "_think_" in name
        }
        failures = sum(case.get("failures", 0) for case in think_cases.values())
        c2_cases = [
            case
            for name, case in think_cases.items()
            if name.startswith("c2_") and case.get("thinking_backends") == ["srt"]
        ]
        throughput = min(
            (case.get("throughput_req_s_mean", 0.0) for case in c2_cases),
            default=0.0,
        )
        environment = {}
        for line in (
            summary_path.with_name("environment.txt")
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()
        ):
            if "=" in line:
                key, value = line.split("=", 1)
                environment[key] = value
        runs.append(
            {
                "directory": str(run_dir),
                "cuda_graph_max_bs": int(environment["thinking_cuda_graph_max_bs"]),
                "mem_fraction": float(environment["thinking_mem_fraction"]),
                "max_running_requests": int(
                    environment["thinking_max_running_requests"]
                ),
                "runtime_checks_passed": runtime["passed"],
                "thinking_failures": failures,
                "c2_throughput_req_s": throughput,
                "cases": think_cases,
            }
        )
    eligible = [
        run
        for run in runs
        if run["runtime_checks_passed"]
        and run["thinking_failures"] == 0
        and run["c2_throughput_req_s"] > 0
    ]
    best = max(eligible, key=lambda run: run["c2_throughput_req_s"], default=None)
    return {"runs": runs, "best": best}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.root)
    if not report["runs"]:
        parser.error(f"no tuning runs found below {args.root}")
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["best"] is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
