"""Capture a bounded SenseNova CUDA serving trace for B1 or B2."""

import argparse
import base64
import concurrent.futures
import importlib.util
import json
import time
import urllib.request
from pathlib import Path

from validate_npu_batching import PROMPTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, choices=[1, 2], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.size < 1 or args.steps < 2:
        parser.error("size must be positive and steps must be at least 2")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    old_traces = set(output.glob("*.trace.json.gz"))
    prompts = [PROMPTS[1]] if args.batch_size == 1 else PROMPTS[:2]

    def request(index):
        payload = dict(
            model=args.model,
            prompt=prompts[index],
            seed=1000 + index,
            n=1,
            size=f"{args.size}x{args.size}",
            num_inference_steps=args.steps,
            guidance_scale=4.0,
            response_format="b64_json",
            output_format="png",
            generator_device="cuda",
            profile=True,
            num_profiled_timesteps=1,
            profile_all_stages=False,
        )
        req = urllib.request.Request(
            args.url + "/v1/images/generations",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=3600) as response:
            body = json.load(response)
        if len(body.get("data", [])) != 1:
            raise RuntimeError("expected exactly one generated image")
        base64.b64decode(body["data"][0]["b64_json"], validate=True)
        return time.perf_counter() - start

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        latencies = list(executor.map(request, range(args.batch_size)))

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        new_traces = set(output.glob("*.trace.json.gz")) - old_traces
        if new_traces:
            break
        time.sleep(0.25)
    else:
        raise RuntimeError(
            f"no new trace appeared in {output}; verify the server profiler directory"
        )

    if len(new_traces) != 1:
        raise RuntimeError(
            f"expected one grouped trace, found {len(new_traces)}: "
            f"{sorted(str(path) for path in new_traces)}"
        )

    trace = new_traces.pop()
    metadata = dict(
        batch_size=args.batch_size,
        prompts=prompts,
        seeds=list(range(1000, 1000 + args.batch_size)),
        size=args.size,
        steps=args.steps,
        cfg=4.0,
        profile_all_stages=False,
        num_profiled_timesteps=1,
        request_latencies_s=latencies,
        trace=trace.name,
        trace_bytes=trace.stat().st_size,
        flash_attn_installed=importlib.util.find_spec("flash_attn") is not None,
    )
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    total_bytes = trace.stat().st_size + metadata_path.stat().st_size
    if total_bytes >= 14_000_000:
        raise RuntimeError(
            f"profile is {total_bytes} bytes, above the 14 MB per-run budget; "
            "repeat both B1 and B2 with --size 1024"
        )
    print(json.dumps(metadata, indent=2))
    print(f"Saved {total_bytes} bytes in {output}")


if __name__ == "__main__":
    main()
