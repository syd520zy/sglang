"""Profile SenseNova text-to-image stages with and without thinking."""

import argparse
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
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


def request_one(args, *, think_mode, max_think_tokens, seed, profile_stages=True):
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
        "profile_stages": profile_stages,
    }
    if think_mode:
        payload["max_think_tokens"] = max_think_tokens
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/images/generations",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"HTTP {error.code}: {error.read().decode(errors='replace')}"
        ) from error
    client_elapsed_ms = (time.perf_counter() - started) * 1000

    if len(result.get("data", [])) != 1:
        raise ValueError("Server did not return exactly one image")
    if not profile_stages:
        return {}
    usage = result.get("usage") or {}
    timings = usage.get("stage_timings_ms")
    if not isinstance(timings, dict) or any(stage not in timings for stage in STAGES):
        raise ValueError(f"Incomplete stage timings: {timings!r}")
    reasoning_tokens = usage.get("reasoning_tokens", 0)
    if think_mode and not 1 <= reasoning_tokens <= max_think_tokens:
        raise ValueError(f"Invalid reasoning token count: {reasoning_tokens!r}")
    if think_mode and not isinstance(usage.get("think_text"), str):
        raise ValueError("Thinking text is missing from usage")
    thinking_backend = usage.get("thinking_backend")
    if think_mode and thinking_backend not in {"srt", "native"}:
        raise ValueError(f"Invalid thinking backend: {thinking_backend!r}")
    if not think_mode and reasoning_tokens:
        raise ValueError("Thinking tokens were returned while thinking was disabled")

    return {
        "case": f"think_{max_think_tokens}" if think_mode else "off",
        "seed": seed,
        "reasoning_tokens": reasoning_tokens,
        "thinking_backend": thinking_backend,
        "think_text_sha256": (
            hashlib.sha256(usage["think_text"].encode("utf-8")).hexdigest()
            if think_mode
            else None
        ),
        "client_elapsed_ms": round(client_elapsed_ms, 3),
        "outside_model_ms": round(client_elapsed_ms - timings["total"], 3),
        "stage_timings_ms": timings,
    }


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["case"]].append(record)

    cases = {}
    for name, rows in grouped.items():
        reasoning_tokens = statistics.mean(row["reasoning_tokens"] for row in rows)
        stage_means = {
            stage: round(
                statistics.mean(row["stage_timings_ms"][stage] for row in rows), 3
            )
            for stage in STAGES
        }
        cases[name] = {
            "samples": len(rows),
            "thinking_backends": sorted(
                {row["thinking_backend"] for row in rows if row["thinking_backend"]}
            ),
            "reasoning_tokens_mean": reasoning_tokens,
            "client_elapsed_ms_mean": round(
                statistics.mean(row["client_elapsed_ms"] for row in rows), 3
            ),
            "outside_model_ms_mean": round(
                statistics.mean(row["outside_model_ms"] for row in rows), 3
            ),
            "stage_timings_ms_mean": stage_means,
            "stage_share_pct": {
                stage: round(stage_means[stage] / stage_means["total"] * 100, 2)
                for stage in STAGES
                if stage != "total"
            },
        }
        if reasoning_tokens:
            cases[name]["think_decode_ms_per_token"] = round(
                stage_means["think_decode"] / reasoning_tokens, 3
            )

    off = cases["off"]
    comparisons = {}
    for name, row in cases.items():
        if name == "off":
            continue
        total_delta = (
            row["stage_timings_ms_mean"]["total"]
            - off["stage_timings_ms_mean"]["total"]
        )
        denoise_delta = (
            row["stage_timings_ms_mean"]["denoise_loop"]
            - off["stage_timings_ms_mean"]["denoise_loop"]
        )
        comparisons[name] = {
            "total_delta_vs_off_ms": round(total_delta, 3),
            "denoise_delta_vs_off_ms": round(denoise_delta, 3),
        }
        if total_delta > 0:
            comparisons[name].update(
                think_decode_share_of_total_delta_pct=round(
                    row["stage_timings_ms_mean"]["think_decode"] / total_delta * 100,
                    2,
                ),
                denoise_share_of_total_delta_pct=round(
                    denoise_delta / total_delta * 100, 2
                ),
            )
    return {"cases": cases, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="sensenova/SenseNova-U1.5-8B-MoT")
    parser.add_argument(
        "--prompt",
        default="A realistic photo of three red apples arranged to the left of a blue bowl.",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--budgets", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if not args.budgets or any(not 1 <= budget <= 1024 for budget in args.budgets):
        parser.error("--budgets values must be between 1 and 1024")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_warmup:
        print("Warmup: thinking off", flush=True)
        request_one(
            args,
            think_mode=False,
            max_think_tokens=0,
            seed=args.seed,
            profile_stages=False,
        )

    records = []
    for repeat in range(args.repeats):
        seed = args.seed + repeat
        cases = [(False, 0), *((True, budget) for budget in args.budgets)]
        if repeat % 2:
            cases.reverse()
        for think_mode, budget in cases:
            label = f"think_{budget}" if think_mode else "off"
            print(f"Run {repeat + 1}/{args.repeats}: {label}", flush=True)
            record = request_one(
                args,
                think_mode=think_mode,
                max_think_tokens=budget,
                seed=seed,
            )
            records.append(record)
            print(
                f"  total={record['stage_timings_ms']['total']:.1f} ms "
                f"think={record['stage_timings_ms']['think_decode']:.1f} ms "
                f"denoise={record['stage_timings_ms']['denoise_loop']:.1f} ms "
                f"tokens={record['reasoning_tokens']}",
                f"backend={record['thinking_backend']}",
                flush=True,
            )

    report = {
        "settings": {
            "model": args.model,
            "prompt": args.prompt,
            "resolution": [args.width, args.height],
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "budgets": args.budgets,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        **summarize(records),
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
