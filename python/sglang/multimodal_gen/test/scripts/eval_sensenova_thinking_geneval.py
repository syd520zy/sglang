"""Generate paired SenseNova images for the official GenEval scorer and compare scores."""

import argparse
import base64
import hashlib
import json
import random
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image


def read_metadata(path):
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows or any(not row.get("prompt") or not row.get("tag") for row in rows):
        raise ValueError("GenEval metadata must contain prompt and tag on every line")
    return rows


def selected_indices(rows, tags, max_per_tag, selection_seed, smoke):
    rng = random.Random(selection_seed)
    selected = []
    for tag in tags:
        candidates = [index for index, row in enumerate(rows) if row["tag"] == tag]
        if not candidates:
            raise ValueError(f"Unknown GenEval task: {tag}")
        count = 1 if smoke else max_per_tag or len(candidates)
        selected.extend(rng.sample(candidates, min(count, len(candidates))))
    return sorted(selected)


def check_png(data, width, height):
    with Image.open(BytesIO(data)) as image:
        if image.format != "PNG" or image.size != (width, height):
            raise ValueError(
                f"Expected a {width}x{height} PNG, received {image.format} {image.size}"
            )
        image.verify()


def generate_one(base_url, prompt, seed, mode, args):
    payload = {
        "model": args.model,
        "prompt": prompt,
        "width": args.width,
        "height": args.height,
        "n": 1,
        "response_format": "b64_json",
        "output_format": "png",
        "seed": seed,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "think_mode": mode == "on",
    }
    if mode == "on":
        payload["max_think_tokens"] = args.max_think_tokens
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/images/generations",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"HTTP {error.code}: {error.read().decode(errors='replace')}"
        ) from error
    elapsed = time.perf_counter() - start
    images = result.get("data", [])
    if len(images) != 1 or "b64_json" not in images[0]:
        raise ValueError("Server did not return exactly one base64 image")
    image = base64.b64decode(images[0]["b64_json"], validate=True)
    check_png(image, args.width, args.height)
    usage = result.get("usage") or {}
    reasoning_tokens = usage.get("reasoning_tokens") or 0
    think_text = usage.get("think_text") or ""
    if mode == "on":
        if (
            not isinstance(reasoning_tokens, int)
            or not 1 <= reasoning_tokens <= args.max_think_tokens
        ):
            raise ValueError(f"Invalid thinking token count: {reasoning_tokens}")
        if not isinstance(think_text, str) or not think_text.endswith("</think>"):
            raise ValueError("Thinking response is missing its closing tag")
    elif reasoning_tokens or think_text:
        raise ValueError("Thinking was returned while think_mode was disabled")
    return image, {
        "seed": seed,
        "elapsed_seconds": elapsed,
        "reasoning_tokens": reasoning_tokens,
        "think_text": think_text,
        "image_sha256": hashlib.sha256(image).hexdigest(),
    }


def generate(args):
    metadata_bytes = args.metadata_file.read_bytes()
    rows = read_metadata(args.metadata_file)
    config = {
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "metadata_rows": len(rows),
        "model": args.model,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "max_think_tokens": args.max_think_tokens,
        "samples_per_prompt": args.samples_per_prompt,
        "seed": args.seed,
        "tags": args.tags,
        "max_per_tag": args.max_per_tag,
        "selection_seed": args.selection_seed,
        "smoke": args.smoke,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != config:
            raise ValueError(
                "Existing output directory was generated with different settings"
            )
    else:
        manifest_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    copied_metadata = args.output_dir / "evaluation_metadata.jsonl"
    if copied_metadata.exists() and copied_metadata.read_bytes() != metadata_bytes:
        raise ValueError(
            "Existing output directory contains different GenEval metadata"
        )
    copied_metadata.write_bytes(metadata_bytes)

    indices = selected_indices(
        rows, args.tags, args.max_per_tag, args.selection_seed, args.smoke
    )
    print(f"Selected {len(indices)} prompts from {args.tags}", flush=True)
    for index in indices:
        row = rows[index]
        for sample in range(args.samples_per_prompt):
            for mode in ("off", "on"):
                folder = args.output_dir / mode / f"{index:05d}"
                samples = folder / "samples"
                samples.mkdir(parents=True, exist_ok=True)
                (folder / "metadata.jsonl").write_text(
                    json.dumps(row, ensure_ascii=False), encoding="utf-8"
                )
                image_path = samples / f"{sample:04d}.png"
                record_path = samples / f"{sample:04d}.json"
                if image_path.exists() and record_path.exists():
                    try:
                        record = json.loads(record_path.read_text(encoding="utf-8"))
                        image = image_path.read_bytes()
                        check_png(image, args.width, args.height)
                    except (OSError, ValueError):
                        pass
                    else:
                        if hashlib.sha256(image).hexdigest() == record.get(
                            "image_sha256"
                        ):
                            continue
                image, record = generate_one(
                    args.base_url, row["prompt"], args.seed + sample, mode, args
                )
                image_path.write_bytes(image)
                record_path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(
                    f"{index + 1}/{len(rows)} {row['tag']} sample={sample} {mode} {record['elapsed_seconds']:.1f}s",
                    flush=True,
                )
    print(f"Generation complete: {args.output_dir}")


def read_results(path, root, mode):
    results = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        image_path = Path(row["filename"])
        key = (int(image_path.parent.parent.name), int(image_path.stem))
        if image_path.parents[2].name != mode or not isinstance(
            row.get("correct"), bool
        ):
            raise ValueError(f"Invalid GenEval result for {mode}: {image_path}")
        if key in results:
            raise ValueError(f"Duplicate GenEval result: {mode} {key}")
        record_path = root / mode / f"{key[0]:05d}" / "samples" / f"{key[1]:04d}.json"
        if not record_path.exists():
            raise ValueError(f"Missing generation record: {record_path}")
        results[key] = (row, json.loads(record_path.read_text(encoding="utf-8")))
    return results


def compare(args):
    config = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    metadata = read_metadata(args.output_dir / "evaluation_metadata.jsonl")
    expected = {
        (index, sample)
        for index in selected_indices(
            metadata,
            config["tags"],
            config["max_per_tag"],
            config["selection_seed"],
            config["smoke"],
        )
        for sample in range(config["samples_per_prompt"])
    }
    off = read_results(args.off_results, args.output_dir, "off")
    on = read_results(args.on_results, args.output_dir, "on")
    if set(off) != expected or set(on) != expected:
        raise ValueError(
            f"Incomplete evaluation: expected {len(expected)} pairs, got off={len(off)}, on={len(on)}"
        )
    by_tag = defaultdict(list)
    by_tag_prompt = defaultdict(lambda: defaultdict(list))
    durations = defaultdict(list)
    reasoning = []
    wins = losses = 0
    for key in sorted(expected):
        off_row, off_record = off[key]
        on_row, on_record = on[key]
        target = metadata[key[0]]
        if any(
            row["prompt"] != target["prompt"] or row["tag"] != target["tag"]
            for row in (off_row, on_row)
        ):
            raise ValueError(f"Prompt or task mismatch for pair {key}")
        if (
            off_record["seed"] != on_record["seed"]
            or off_record["seed"] != config["seed"] + key[1]
        ):
            raise ValueError(f"Image seed mismatch for pair {key}")
        a, b = bool(off_row["correct"]), bool(on_row["correct"])
        by_tag[target["tag"]].append((a, b))
        by_tag_prompt[target["tag"]][key[0]].append((a, b))
        wins += b and not a
        losses += a and not b
        durations["off"].append(off_record["elapsed_seconds"])
        durations["on"].append(on_record["elapsed_seconds"])
        reasoning.append(on_record["reasoning_tokens"])

    scores = {
        tag: {
            "pairs": len(pairs),
            "off": sum(a for a, _ in pairs) / len(pairs),
            "on": sum(b for _, b in pairs) / len(pairs),
        }
        for tag, pairs in by_tag.items()
    }
    for row in scores.values():
        row["delta"] = row["on"] - row["off"]
    tags = sorted(scores)
    rng = random.Random(42)
    bootstrap = []
    for _ in range(2000):
        deltas = []
        for tag in tags:
            prompts = list(by_tag_prompt[tag].values())
            sampled = [
                pair for _ in prompts for pair in prompts[rng.randrange(len(prompts))]
            ]
            deltas.append(sum(b - a for a, b in sampled) / len(sampled))
        bootstrap.append(statistics.mean(deltas))
    bootstrap.sort()
    report = {
        "benchmark": "GenEval paired subset A/B; not an official full-set leaderboard score",
        "smoke": config["smoke"],
        "prompts": len({index for index, _ in expected}),
        "pairs": len(expected),
        "settings": config,
        "macro_off": statistics.mean(scores[tag]["off"] for tag in tags),
        "macro_on": statistics.mean(scores[tag]["on"] for tag in tags),
        "macro_delta": statistics.mean(scores[tag]["delta"] for tag in tags),
        "delta_95pct_bootstrap_ci": [bootstrap[49], bootstrap[1949]],
        "paired_on_wins": wins,
        "paired_on_losses": losses,
        "latency_seconds_mean": {
            mode: statistics.mean(values) for mode, values in durations.items()
        },
        "reasoning_tokens_mean": statistics.mean(reasoning),
        "thinking_budget_hit_rate_proxy": sum(
            n == config["max_think_tokens"] for n in reasoning
        )
        / len(reasoning),
        "by_tag": scores,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generation = sub.add_parser("generate")
    generation.add_argument("--metadata-file", type=Path, required=True)
    generation.add_argument("--output-dir", type=Path, required=True)
    generation.add_argument("--base-url", default="http://127.0.0.1:30000")
    generation.add_argument("--model", default="sensenova/SenseNova-U1.5-8B-MoT")
    generation.add_argument("--width", type=int, default=1024)
    generation.add_argument("--height", type=int, default=1024)
    generation.add_argument("--steps", type=int, default=50)
    generation.add_argument("--guidance-scale", type=float, default=4.0)
    generation.add_argument("--max-think-tokens", type=int, default=256)
    generation.add_argument("--samples-per-prompt", type=int, default=1)
    generation.add_argument("--seed", type=int, default=42)
    generation.add_argument(
        "--tags", nargs="+", default=["counting", "position", "color_attr"]
    )
    generation.add_argument("--max-per-tag", type=int, default=30)
    generation.add_argument("--selection-seed", type=int, default=20260917)
    generation.add_argument("--timeout", type=int, default=1800)
    generation.add_argument(
        "--smoke", action="store_true", help="One prompt from each GenEval task"
    )
    comparison = sub.add_parser("compare")
    comparison.add_argument("--output-dir", type=Path, required=True)
    comparison.add_argument("--off-results", type=Path, required=True)
    comparison.add_argument("--on-results", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        if (
            args.samples_per_prompt < 1
            or args.max_per_tag < 0
            or not 1 <= args.max_think_tokens <= 1024
            or len(set(args.tags)) != len(args.tags)
        ):
            parser.error(
                "Invalid sample count, task list, task limit, or thinking token budget"
            )
        generate(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
