"""Compare native and SRT SenseNova concurrency runs.

Reads two `summary.json` files produced by
`profile_sensenova_thinking_concurrency.py` with identical settings and reports
throughput, latency and the overlap evidence that decides whether the SRT batch
was ever shared.
"""

import argparse
import json
from pathlib import Path


def load_summary(directory: Path):
    return json.loads((directory / "summary.json").read_text(encoding="utf-8"))


def thinking_backends(summary):
    return {
        backend
        for case, entry in summary["cases"].items()
        if not case.endswith("_off")
        for backend in entry.get("thinking_backends", [])
    }


def compare_case(native, srt):
    native_throughput = native.get("throughput_req_s_mean")
    srt_throughput = srt.get("throughput_req_s_mean")
    comparison = {
        "concurrency": native.get("concurrency"),
        "native_throughput_req_s": native_throughput,
        "srt_throughput_req_s": srt_throughput,
        "native_latency_ms_p50": native.get("latency_ms_p50"),
        "native_latency_ms_p95": native.get("latency_ms_p95"),
        "srt_latency_ms_p50": srt.get("latency_ms_p50"),
        "srt_latency_ms_p95": srt.get("latency_ms_p95"),
        "native_parallelism_ratio_max": native.get("parallelism_ratio_max"),
        "srt_parallelism_ratio_max": srt.get("parallelism_ratio_max"),
        "srt_overlapped_beyond_serial": srt.get("overlapped_beyond_serial"),
        "srt_waves_with_think_overlap": srt.get("waves_with_think_overlap"),
        "native_think_decode_ms_per_token": native.get("think_decode_ms_per_token"),
        "srt_think_decode_ms_per_token": srt.get("think_decode_ms_per_token"),
        "native_failures": native.get("failures"),
        "srt_failures": srt.get("failures"),
        "srt_error_samples": srt.get("error_samples"),
    }
    if native_throughput and srt_throughput:
        comparison["throughput_speedup"] = round(srt_throughput / native_throughput, 3)
    if (
        comparison["native_think_decode_ms_per_token"]
        and comparison["srt_think_decode_ms_per_token"]
    ):
        comparison["think_decode_speedup"] = round(
            comparison["native_think_decode_ms_per_token"]
            / comparison["srt_think_decode_ms_per_token"],
            3,
        )
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--srt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    native_summary = load_summary(args.native_dir)
    srt_summary = load_summary(args.srt_dir)
    comparable_settings = (
        "model",
        "prompt",
        "resolution",
        "steps",
        "guidance_scale",
        "budgets",
        "concurrency",
        "repeats",
        "seed",
    )
    mismatches = {
        key: {
            "native": native_summary["settings"].get(key),
            "srt": srt_summary["settings"].get(key),
        }
        for key in comparable_settings
        if native_summary["settings"].get(key) != srt_summary["settings"].get(key)
    }
    if mismatches:
        raise RuntimeError(f"The two runs used different settings: {mismatches}")
    native_cases = set(native_summary["cases"])
    srt_cases = set(srt_summary["cases"])
    if native_cases != srt_cases:
        raise RuntimeError(
            "The two runs contain different cases: "
            f"native_only={sorted(native_cases - srt_cases)}, "
            f"srt_only={sorted(srt_cases - native_cases)}"
        )

    native_backends = thinking_backends(native_summary)
    srt_backends = thinking_backends(srt_summary)
    if native_backends != {"native"}:
        raise RuntimeError(f"Native run used unexpected backends: {native_backends}")
    if srt_backends != {"srt"}:
        raise RuntimeError(
            f"SRT run used {srt_backends}; a native fallback is not SRT data"
        )

    comparisons = {}
    for case in sorted(native_cases):
        if case.endswith("_off"):
            continue
        comparisons[case] = compare_case(
            native_summary["cases"][case], srt_summary["cases"][case]
        )
    for case in sorted(native_cases):
        if not case.endswith("_off"):
            continue
        comparisons[case] = compare_case(
            native_summary["cases"][case], srt_summary["cases"][case]
        )

    concurrent_thinking = {
        case: entry
        for case, entry in comparisons.items()
        if entry["concurrency"] > 1 and not case.endswith("_off")
    }
    concurrent_off = {
        case: entry
        for case, entry in comparisons.items()
        if entry["concurrency"] > 1 and case.endswith("_off")
    }
    verdicts = {
        "native_backends": sorted(native_backends),
        "srt_backends": sorted(srt_backends),
        "no_request_failures": all(
            not entry["native_failures"] and not entry["srt_failures"]
            for entry in comparisons.values()
        ),
        # Control: the same server batches two non-thinking requests, so the
        # batching machinery itself works.
        "concurrent_off_requests_overlap": all(
            entry["srt_overlapped_beyond_serial"] for entry in concurrent_off.values()
        )
        if concurrent_off
        else None,
        # The P1-1 question: does a concurrent thinking pair share the SRT batch?
        "concurrent_thinking_requests_overlap": all(
            entry["srt_overlapped_beyond_serial"]
            for entry in concurrent_thinking.values()
        )
        if concurrent_thinking
        else None,
        "concurrent_thinking_think_phases_overlap": all(
            (entry["srt_waves_with_think_overlap"] or 0) > 0
            for entry in concurrent_thinking.values()
        )
        if concurrent_thinking
        else None,
        "concurrent_thinking_throughput_speedup": {
            case: entry.get("throughput_speedup")
            for case, entry in concurrent_thinking.items()
        },
    }
    report = {
        "settings": native_summary["settings"],
        "comparisons": comparisons,
        "verdicts": verdicts,
        "limitations": [
            (
                "Overlap is derived from client intervals and stage durations, not from the SRT "
                "scheduler; the SRT server does not report its decode batch size to this client."
            ),
            (
                "Read the 'Dynamic batch stats' lines in the main server log for the "
                "dispatch-level evidence (merged_rate and top_rejects)."
            ),
        ],
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
