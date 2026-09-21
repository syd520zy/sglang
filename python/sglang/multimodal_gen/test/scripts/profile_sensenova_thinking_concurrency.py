"""Measure SenseNova thinking throughput under concurrent text-to-image requests.

The wave is the unit of measurement: `concurrency` requests are released
together through a barrier and their client-side intervals are kept, so a case
that serializes inside the server is visible as `parallelism_ratio` near 1.0
instead of `concurrency`.
"""

import argparse
import base64
import binascii
import hashlib
import json
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STAGES = (
    "input_prepare",
    "condition_prefill",
    "think_decode",
    "think_replay_prefill",
    "cfg_prefill",
    "denoise_prepare",
    "denoise_loop",
    "total",
)

# Stages that run before think_decode, so their sum locates the think window
# inside a request's client interval.
STAGES_BEFORE_THINK = ("input_prepare", "condition_prefill")

# Arbitrary: a ratio this high cannot come from a server that ran the wave one
# request at a time, whose ratio stays near 1.0.
PARALLEL_THRESHOLD = 1.5

DEFAULT_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "::1")


def resolve_base_url(raw_url, allowed_hosts):
    """Return the request origin, restricted to the declared hosts.

    The lifecycle scripts always point this at the local server they started; a
    remote host has to be named through --allowed-host.
    """
    parts = urllib.parse.urlsplit(raw_url)
    if parts.scheme not in {"http", "https"}:
        raise ValueError(
            f"--base-url scheme must be http or https, got {parts.scheme!r}"
        )
    if not parts.hostname:
        raise ValueError(f"--base-url has no host: {raw_url!r}")
    if parts.hostname not in allowed_hosts:
        raise ValueError(
            f"--base-url host {parts.hostname!r} is not allowed; "
            "add it with --allowed-host"
        )
    if parts.query or parts.fragment:
        raise ValueError(f"--base-url must not carry a query or fragment: {raw_url!r}")
    return f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"


def percentile(values, percent):
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def validate_usage(usage, *, think_mode, max_think_tokens, expected_backend):
    if not isinstance(usage, dict):
        return "usage is missing from the response"
    timings = usage.get("stage_timings_ms")
    if not isinstance(timings, dict) or any(stage not in timings for stage in STAGES):
        return f"incomplete stage timings: {timings!r}"
    reasoning_tokens = usage.get("reasoning_tokens", 0)
    if think_mode and not 1 <= reasoning_tokens <= max_think_tokens:
        return f"invalid reasoning token count: {reasoning_tokens!r}"
    if think_mode and not isinstance(usage.get("think_text"), str):
        return "thinking text is missing from usage"
    backend = usage.get("thinking_backend")
    if think_mode and backend not in {"srt", "native"}:
        return f"invalid thinking backend: {backend!r}"
    if not think_mode and reasoning_tokens:
        return "thinking tokens were returned while thinking was disabled"
    if think_mode and expected_backend != "any" and backend != expected_backend:
        return f"expected backend {expected_backend}, got {backend!r}"
    return None


def request_one(args, *, case, think_mode, max_think_tokens, seed, request_index, wave):
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "n": 1,
        "response_format": "b64_json",
        "output_format": "png",
        "seed": seed,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "think_mode": think_mode,
        "profile_stages": True,
    }
    if think_mode:
        payload["max_think_tokens"] = max_think_tokens
    request = urllib.request.Request(
        f"{args.base_url}/v1/images/generations",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    record = {
        "case": case,
        "wave": wave,
        "request_index": request_index,
        "seed": seed,
        "concurrency": args.current_concurrency,
    }
    started_at = time.perf_counter()
    error = None
    result = None
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as http_error:
        detail = http_error.read().decode(errors="replace")[:400]
        error = f"HTTP {http_error.code}: {detail}"
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    completed_at = time.perf_counter()
    record["started_at"] = started_at
    record["completed_at"] = completed_at
    record["client_elapsed_ms"] = round((completed_at - started_at) * 1000, 3)
    if error is not None:
        record["error"] = error
        return record

    if len(result.get("data", [])) != 1:
        record["error"] = "server did not return exactly one image"
        return record
    encoded_image = result["data"][0].get("b64_json")
    try:
        image = base64.b64decode(encoded_image, validate=True)
    except (binascii.Error, TypeError, ValueError):
        record["error"] = "server returned an invalid b64_json image"
        return record
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = (
        image_dir / f"{case}-wave-{wave}-request-{request_index}-seed-{seed}.png"
    )
    image_path.write_bytes(image)
    usage = result.get("usage") or {}
    error = validate_usage(
        usage,
        think_mode=think_mode,
        max_think_tokens=max_think_tokens,
        expected_backend=args.expected_backend,
    )
    if error is not None:
        record["error"] = error
        return record

    timings = usage["stage_timings_ms"]
    record.update(
        {
            "reasoning_tokens": usage.get("reasoning_tokens", 0),
            "thinking_backend": usage.get("thinking_backend") if think_mode else None,
            "image_path": str(image_path),
            "image_sha256": hashlib.sha256(image).hexdigest(),
            "think_text_sha256": (
                hashlib.sha256(usage["think_text"].encode("utf-8")).hexdigest()
                if think_mode
                else None
            ),
            "outside_model_ms": round(
                record["client_elapsed_ms"] - timings["total"], 3
            ),
            "stage_timings_ms": timings,
        }
    )
    return record


def run_wave(args, *, case, concurrency, budget, base_seed, think_mode, wave):
    barrier = threading.Barrier(concurrency)

    def launch(request_index):
        # Every request carries its own seed, so a wave never overwrites one image
        # with another and the per-request work stays identical across cases.
        barrier.wait()
        return request_one(
            args,
            case=case,
            think_mode=think_mode,
            max_think_tokens=budget,
            seed=base_seed + request_index,
            request_index=request_index,
            wave=wave,
        )

    args.current_concurrency = concurrency
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(launch, index) for index in range(concurrency)]
        records = [future.result() for future in futures]

    origin = min(record["started_at"] for record in records)
    for record in records:
        record["start_offset_ms"] = round((record["started_at"] - origin) * 1000, 3)
        record["end_offset_ms"] = round((record["completed_at"] - origin) * 1000, 3)
    return records


def think_window_ms(record):
    """Estimated think-phase interval, derived from the stage durations.

    The SRT server owns the real batch, so this only shows whether the think
    phases were in flight together, which is what batching requires.
    """
    timings = record.get("stage_timings_ms")
    if not timings or not timings.get("think_decode"):
        return None
    # Concurrent clients start together even when the scheduler executes them
    # serially. Anchor the server work at response completion so queueing time
    # is not mistaken for overlapping model execution.
    model_started_at = record["end_offset_ms"] - timings["total"]
    thought_at = model_started_at + sum(timings[stage] for stage in STAGES_BEFORE_THINK)
    return thought_at, thought_at + timings["think_decode"]


def model_window_ms(record):
    timings = record.get("stage_timings_ms")
    if not timings:
        return None
    return record["end_offset_ms"] - timings["total"], record["end_offset_ms"]


def pair_overlap_ms(left, right):
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def summarize_wave(records):
    ok = [record for record in records if "error" not in record]
    wave_start = min(record["start_offset_ms"] for record in records)
    wave_end = max(record["end_offset_ms"] for record in records)
    wall_ms = max(wave_end - wave_start, 1e-9)
    summary = {
        "requests": len(records),
        "failures": len(records) - len(ok),
        "wall_ms": round(wall_ms, 3),
        "throughput_req_s": round(len(ok) / (wall_ms / 1000.0), 4),
        "errors": [record["error"] for record in records if "error" in record],
    }
    if not ok:
        return summary

    model_windows = [model_window_ms(record) for record in ok]
    model_windows = [window for window in model_windows if window is not None]
    if model_windows:
        model_wall_ms = max(window[1] for window in model_windows) - min(
            window[0] for window in model_windows
        )
        model_busy_ms = sum(window[1] - window[0] for window in model_windows)
        summary["parallelism_ratio"] = round(
            model_busy_ms / max(model_wall_ms, 1e-9), 3
        )
    summary["latency_ms"] = {
        "mean": round(statistics.mean(r["client_elapsed_ms"] for r in ok), 3),
        "p50": round(percentile([r["client_elapsed_ms"] for r in ok], 50.0), 3),
        "p95": round(percentile([r["client_elapsed_ms"] for r in ok], 95.0), 3),
    }

    windows = [think_window_ms(record) for record in ok]
    windows = [window for window in windows if window is not None]
    if len(windows) > 1:
        overlaps = [
            pair_overlap_ms(windows[left], windows[right])
            for left in range(len(windows))
            for right in range(left + 1, len(windows))
        ]
        summary["think_window_pairs"] = len(overlaps)
        summary["think_window_overlap_pairs"] = sum(
            1 for overlap in overlaps if overlap > 0.0
        )
        summary["think_window_overlap_ms_max"] = round(max(overlaps), 3)
    return summary


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["case"], record["wave"])].append(record)

    waves = defaultdict(list)
    concurrencies = {}
    for (case, _wave), wave_records in grouped.items():
        waves[case].append(summarize_wave(wave_records))
        concurrencies[case] = wave_records[0]["concurrency"]

    cases = {}
    for case, wave_summaries in waves.items():
        usable = [
            summary
            for summary in wave_summaries
            if summary["requests"] > summary["failures"]
        ]
        entry = {
            "concurrency": concurrencies[case],
            "waves": len(wave_summaries),
            "failures": sum(summary["failures"] for summary in wave_summaries),
            "error_samples": sorted(
                {error for summary in wave_summaries for error in summary["errors"]}
            )[:5],
        }
        if not usable:
            cases[case] = entry
            continue

        entry["throughput_req_s_mean"] = round(
            statistics.mean(summary["throughput_req_s"] for summary in usable), 4
        )
        entry["wall_ms_mean"] = round(
            statistics.mean(summary["wall_ms"] for summary in usable), 3
        )
        for key in ("mean", "p50", "p95"):
            entry[f"latency_ms_{key}"] = round(
                statistics.mean(summary["latency_ms"][key] for summary in usable), 3
            )
        entry["parallelism_ratio_mean"] = round(
            statistics.mean(summary["parallelism_ratio"] for summary in usable), 3
        )
        entry["parallelism_ratio_max"] = round(
            max(summary["parallelism_ratio"] for summary in usable), 3
        )
        entry["overlapped_beyond_serial"] = (
            entry["parallelism_ratio_max"] >= PARALLEL_THRESHOLD
        )
        entry["waves_with_think_overlap"] = sum(
            1 for summary in usable if summary.get("think_window_overlap_pairs", 0) > 0
        )

        case_records = [
            record
            for record in records
            if record["case"] == case and "error" not in record
        ]
        entry["thinking_backends"] = sorted(
            {
                record["thinking_backend"]
                for record in case_records
                if record["thinking_backend"]
            }
        )
        tokens = [record["reasoning_tokens"] for record in case_records]
        if tokens and any(tokens):
            entry["reasoning_tokens_mean"] = round(statistics.mean(tokens), 3)
        stage_means = {
            stage: round(
                statistics.mean(
                    record["stage_timings_ms"][stage] for record in case_records
                ),
                3,
            )
            for stage in STAGES
        }
        entry["stage_timings_ms_mean"] = stage_means
        if entry.get("reasoning_tokens_mean"):
            entry["think_decode_ms_per_token"] = round(
                stage_means["think_decode"] / entry["reasoning_tokens_mean"], 3
            )
        cases[case] = entry
    return cases


def warmup(args):
    for budget in [0, *args.budgets]:
        think_mode = budget > 0
        label = f"think_{budget}" if think_mode else "off"
        print(f"Warmup: {label}", flush=True)
        run_wave(
            args,
            case=f"warmup_{label}",
            concurrency=1,
            budget=budget,
            base_seed=args.seed,
            think_mode=think_mode,
            wave=0,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--allowed-host",
        action="append",
        dest="allowed_hosts",
        help="Host the client may target; repeatable, defaults to the local hosts.",
    )
    parser.add_argument("--model", default="sensenova/SenseNova-U1.5-8B-MoT")
    parser.add_argument(
        "--prompt",
        default="A realistic photo of three red apples arranged to the left of a blue bowl.",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--budgets", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument(
        "--expected-backend",
        choices=["srt", "native", "any"],
        default="srt",
        help="Fail a thinking request served by another backend; fallback is not SRT data.",
    )
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if not args.budgets or any(not 1 <= budget <= 1024 for budget in args.budgets):
        parser.error("--budgets values must be between 1 and 1024")
    if not args.concurrency or any(value < 1 for value in args.concurrency):
        parser.error("--concurrency values must be at least 1")
    args.base_url = resolve_base_url(
        args.base_url, tuple(args.allowed_hosts or DEFAULT_ALLOWED_HOSTS)
    )

    args.current_concurrency = 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_warmup:
        warmup(args)

    records = []
    cases = [(False, 0), *((True, budget) for budget in args.budgets)]
    for concurrency in args.concurrency:
        for repeat in range(args.repeats):
            ordered = list(cases)
            if repeat % 2:
                ordered.reverse()
            for think_mode, budget in ordered:
                label = f"think_{budget}" if think_mode else "off"
                case = f"c{concurrency}_{label}"
                print(
                    f"Concurrency {concurrency}, repeat {repeat + 1}/{args.repeats}: {label}",
                    flush=True,
                )
                wave_records = run_wave(
                    args,
                    case=case,
                    concurrency=concurrency,
                    budget=budget,
                    base_seed=args.seed + repeat * 1000,
                    think_mode=think_mode,
                    wave=repeat,
                )
                records.extend(wave_records)
                wave_summary = summarize_wave(wave_records)
                line = (
                    f"  wall={wave_summary['wall_ms']:.1f} ms "
                    f"throughput={wave_summary['throughput_req_s']:.3f} req/s"
                )
                if "parallelism_ratio" in wave_summary:
                    line += f" parallelism={wave_summary['parallelism_ratio']:.2f}"
                if wave_summary["failures"]:
                    line += f" failures={wave_summary['failures']}"
                print(line, flush=True)
                for error in wave_summary["errors"]:
                    print(f"    error: {error}", flush=True)

    report = {
        "settings": {
            "model": args.model,
            "prompt": args.prompt,
            "resolution": [args.width, args.height],
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "budgets": args.budgets,
            "concurrency": args.concurrency,
            "repeats": args.repeats,
            "seed": args.seed,
            "expected_backend": args.expected_backend,
            "parallel_threshold": PARALLEL_THRESHOLD,
        },
        "cases": summarize(records),
    }
    (args.output_dir / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
