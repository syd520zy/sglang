#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import re
from statistics import mean


FILE_PATTERN = re.compile(r"round(?P<round>\d+)_(?P<case>short|long)_c(?P<conc>\d+)\.jsonl$")


def parse_one_file(path: str):
    name = os.path.basename(path)
    m = FILE_PATTERN.match(name)
    if not m:
        return None

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return None

    # Keep the latest line in case the file has multiple runs appended.
    obj = json.loads(lines[-1])
    return {
        "round": int(m.group("round")),
        "case": m.group("case"),
        "concurrency": int(m.group("conc")),
        "num_prompts": obj.get("num_prompts"),
        "mean_ttft_ms": obj.get("mean_ttft_ms"),
        "mean_tpot_ms": obj.get("mean_tpot_ms"),
        "mean_e2e_latency_ms": obj.get("mean_e2e_latency_ms"),
        "output_throughput_toks": obj.get("output_throughput"),
        "request_throughput_reqs": obj.get("request_throughput"),
        "input_throughput_toks": obj.get("input_throughput"),
        "total_throughput_toks": obj.get("total_throughput"),
        "p99_ttft_ms": obj.get("p99_ttft_ms"),
        "p99_tpot_ms": obj.get("p99_tpot_ms"),
        "p99_e2e_latency_ms": obj.get("p99_e2e_latency_ms"),
        "completed": obj.get("completed"),
    }


def write_csv(rows, output_csv: str):
    fieldnames = [
        "round",
        "case",
        "concurrency",
        "num_prompts",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_e2e_latency_ms",
        "output_throughput_toks",
        "request_throughput_reqs",
        "input_throughput_toks",
        "total_throughput_toks",
        "p99_ttft_ms",
        "p99_tpot_ms",
        "p99_e2e_latency_ms",
        "completed",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["case"], x["concurrency"], x["round"])):
            w.writerow(r)


def aggregate(rows):
    groups = {}
    for r in rows:
        key = (r["case"], r["concurrency"])
        groups.setdefault(key, []).append(r)

    agg_rows = []
    for (case, conc), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        def m(k):
            vals = [x[k] for x in items if isinstance(x.get(k), (int, float))]
            return mean(vals) if vals else None

        agg_rows.append(
            {
                "case": case,
                "concurrency": conc,
                "rounds": len(items),
                "avg_mean_ttft_ms": m("mean_ttft_ms"),
                "avg_mean_tpot_ms": m("mean_tpot_ms"),
                "avg_mean_e2e_latency_ms": m("mean_e2e_latency_ms"),
                "avg_output_throughput_toks": m("output_throughput_toks"),
                "avg_request_throughput_reqs": m("request_throughput_reqs"),
                "avg_p99_ttft_ms": m("p99_ttft_ms"),
                "avg_p99_tpot_ms": m("p99_tpot_ms"),
                "avg_p99_e2e_latency_ms": m("p99_e2e_latency_ms"),
            }
        )
    return agg_rows


def write_agg_csv(rows, output_csv: str):
    fieldnames = [
        "case",
        "concurrency",
        "rounds",
        "avg_mean_ttft_ms",
        "avg_mean_tpot_ms",
        "avg_mean_e2e_latency_ms",
        "avg_output_throughput_toks",
        "avg_request_throughput_reqs",
        "avg_p99_ttft_ms",
        "avg_p99_tpot_ms",
        "avg_p99_e2e_latency_ms",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--output-agg-csv", default=None)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "round*_*.jsonl")))
    rows = []
    for path in files:
        item = parse_one_file(path)
        if item is not None:
            rows.append(item)

    if not rows:
        raise SystemExit(f"No valid jsonl result files found in: {args.input_dir}")

    write_csv(rows, args.output_csv)
    print(f"Wrote detailed csv: {args.output_csv}")

    agg_csv = args.output_agg_csv or os.path.splitext(args.output_csv)[0] + "_agg.csv"
    agg_rows = aggregate(rows)
    write_agg_csv(agg_rows, agg_csv)
    print(f"Wrote aggregated csv: {agg_csv}")


if __name__ == "__main__":
    main()

