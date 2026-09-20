"""Print a pasteable summary of SenseNova thinking result directories.

Handles the three result layouts the benchmark scripts produce: the staged
profiling profile, the concurrency comparison and the lifecycle acceptance.
The output is bounded, so it can be copied back verbatim instead of moving the
whole directory.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

MAX_LISTED = 8
MAX_DEPTH = 3

REPORT_NAMES = ("summary.json", "comparison.json", "environment.txt")


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def head_lines(path, limit=3):
    try:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [line.strip() for line in text[:limit] if line.strip()]


def number(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else value


def emit_profile(directory, summary):
    cases = summary.get("cases") or {}
    for case, entry in sorted(cases.items()):
        per_token = entry.get("think_decode_ms_per_token")
        total = (entry.get("stage_timings_ms_mean") or {}).get("total")
        print(
            f"  case {case}: think_decode={number(per_token)} ms/token "
            f"total={number(total)} ms"
        )
        timings = entry.get("stage_timings_ms_mean") or {}
        if timings:
            print(
                "    stages: "
                + " ".join(
                    f"{name}={number(value, 1)}" for name, value in timings.items()
                )
            )
    records = load_json(directory / "records.json") or []
    backends = sorted(
        {r.get("thinking_backend") for r in records if r.get("case") != "off"}
    )
    if backends:
        print(f"  backends: {backends}")


def emit_concurrency(summary):
    for key, entry in sorted((summary.get("cases") or {}).items()):
        print(
            f"  case {key} (concurrency={entry.get('concurrency')}): "
            f"wall={number(entry.get('wall_ms_mean'), 1)} ms "
            f"throughput={number(entry.get('throughput_req_s_mean'))} req/s "
            f"parallelism mean/max={number(entry.get('parallelism_ratio_mean'))}/"
            f"{number(entry.get('parallelism_ratio_max'))}"
        )
        print(
            f"    latency p50/p95={number(entry.get('latency_ms_p50'), 1)}/"
            f"{number(entry.get('latency_ms_p95'), 1)} ms "
            f"failures={entry.get('failures')} backends={entry.get('thinking_backends')}"
        )
        print(
            f"    overlapped_beyond_serial={entry.get('overlapped_beyond_serial')} "
            f"waves_with_think_overlap={entry.get('waves_with_think_overlap')} "
            f"reasoning_tokens={number(entry.get('reasoning_tokens_mean'), 1)}"
        )
        for error in (entry.get("error_samples") or [])[:MAX_LISTED]:
            print(f"    error sample: {error}")


def emit_comparison(report):
    verdicts = report.get("verdicts") or {}
    print("  verdicts:")
    for name, value in verdicts.items():
        print(f"    {name}: {value}")
    for case, entry in sorted((report.get("comparisons") or {}).items()):
        print(
            f"  {case}: native={number(entry.get('native_think_decode_ms_per_token'))} "
            f"-> srt={number(entry.get('srt_think_decode_ms_per_token'))} ms/token "
            f"speedup={number(entry.get('think_decode_speedup'))} "
            f"throughput_speedup={number(entry.get('throughput_speedup'))}"
        )
    for limitation in report.get("limitations") or []:
        print(f"  limitation: {limitation}")


def emit_lifecycle(directory):
    for name in ("startup", "after-kill"):
        report = load_json(directory / f"lifecycle-{name}.json")
        if report is None:
            continue
        print(f"  phase {name}: {'PASS' if report.get('passed') else 'FAIL'}")
        for check, entry in (report.get("checks") or {}).items():
            status = "PASS" if entry.get("passed") else "FAIL"
            print(f"    [{status}] {check}: {entry.get('evidence')}")
        evidence = report.get("evidence") or {}
        for probe in evidence.get("probes") or []:
            if probe.get("ok"):
                print(
                    f"    probe {probe['label']}: ok backend={probe.get('thinking_backend')} "
                    f"elapsed={number(probe.get('elapsed_ms'), 1)} ms"
                )
            else:
                print(f"    probe {probe['label']}: error={probe.get('error')}")
    residue = directory / "residue.txt"
    if residue.exists():
        print("  residue:")
        for line in head_lines(residue, limit=MAX_LISTED):
            print(f"    {line}")
    info = load_json(directory / "server-info.json")
    if info is not None:
        backend = info.get("thinking_backend")
        if backend is not None:
            print(f"  server_info.thinking_backend: {backend}")
    for name in ("lifecycle-after-kill.json",):
        report = load_json(directory / name)
        if report and report.get("evidence", {}).get("server_log_errors"):
            print("  logged errors:")
            for line in report["evidence"]["server_log_errors"][:MAX_LISTED]:
                print(f"    {line}")


def emit_environment(directory):
    environment = directory / "environment.txt"
    if not environment.exists():
        return
    print("  environment:")
    for line in head_lines(environment, limit=MAX_LISTED):
        print(f"    {line}")


def is_concurrency_summary(summary):
    return any(
        "concurrency" in entry or "parallelism_ratio_mean" in entry
        for entry in (summary.get("cases") or {}).values()
    )


def emit_directory(directory):
    emit_environment(directory)
    summary = load_json(directory / "summary.json")
    if summary is not None:
        if is_concurrency_summary(summary):
            emit_concurrency(summary)
        else:
            emit_profile(directory, summary)
    comparison = load_json(directory / "comparison.json")
    if comparison is not None:
        emit_comparison(comparison)
    if any(directory.glob("lifecycle-*.json")) or (directory / "residue.txt").exists():
        emit_lifecycle(directory)


def has_report(directory):
    """True when a directory holds files this viewer knows how to print."""
    if any((directory / name).exists() for name in REPORT_NAMES):
        return True
    return (
        any(directory.glob("lifecycle-*.json")) or (directory / "residue.txt").exists()
    )


def describe(directory, depth=0):
    print(f"--- {directory}")
    content = io.StringIO()
    with contextlib.redirect_stdout(content):
        emit_directory(directory)
        # A run nests its results under the round and the mode, so descend while
        # a directory still has something to show.
        if depth < MAX_DEPTH:
            for child in sorted(p for p in directory.iterdir() if p.is_dir()):
                if has_report(child):
                    describe(child, depth + 1)
    text = content.getvalue().rstrip()
    print(text if text else unexpected(directory))


def unexpected(directory):
    """Report a directory that printed nothing, so a result is never silent."""
    entries = sorted(path.name for path in directory.iterdir())
    if not entries:
        return "  (the directory is empty)"
    listed = ", ".join(entries[:MAX_LISTED])
    if len(entries) > MAX_LISTED:
        listed += f", ... ({len(entries)} entries)"
    return f"  (no result files this viewer recognizes; contains: {listed})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="also write the summary to this file",
    )
    args = parser.parse_args()

    lines = []
    for directory in args.result_dirs:
        if not directory.is_dir():
            print(f"--- {directory}: not a directory")
            continue
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            describe(directory)
        text = buffer.getvalue().rstrip()
        print(text)
        lines.append(text)

    if args.output is not None:
        args.output.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
