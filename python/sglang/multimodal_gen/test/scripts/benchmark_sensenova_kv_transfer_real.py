"""Benchmark KV handoff from a live SenseNova SRT pool through shared memory."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import requests
import torch

MIB = 1024**2
_TRANSFER_RID_PREFIX = "sensenova-kvxfer-"


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",")]
    if not values or any(item <= 0 for item in values):
        raise ValueError("expected comma-separated positive integers")
    return values


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * quantile))]


def get_srt_url(base_url: str, timeout: float) -> str:
    response = requests.get(f"{base_url.rstrip('/')}/server_info", timeout=timeout)
    response.raise_for_status()
    backend = response.json().get("thinking_backend") or {}
    if backend.get("backend") != "srt" or backend.get("state") != "ready":
        raise RuntimeError(f"managed SRT is not ready: {backend}")
    url = backend.get("url")
    if not isinstance(url, str):
        raise RuntimeError(f"managed SRT has no URL: {backend}")
    return url.rstrip("/")


def request_export(
    srt_url: str,
    transfer_dir: Path,
    token_ids: list[int],
    dump_id: str,
    timeout: float,
) -> tuple[dict, float]:
    started = time.perf_counter()
    response = requests.post(
        f"{srt_url}/generate",
        json={
            "rid": f"{_TRANSFER_RID_PREFIX}{dump_id}",
            "input_ids": token_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
                "skip_special_tokens": False,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    request_wall_ms = (time.perf_counter() - started) * 1000
    metadata_path = transfer_dir / f"{dump_id}.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"SRT response completed without {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(",".join(map(str, token_ids)).encode()).hexdigest()
    if metadata.get("token_sha256") != expected_hash:
        raise RuntimeError("SRT export token hash does not match the request")
    return metadata, request_wall_ms


def import_export(metadata: dict, transfer_dir: Path) -> tuple[dict, list]:
    if metadata.get("dtype") != "torch.bfloat16":
        raise RuntimeError(f"unsupported transfer dtype: {metadata.get('dtype')}")
    shape = tuple(metadata["shape"])
    if len(shape) != 5 or shape[1] != 2:
        raise RuntimeError(f"unexpected transfer shape: {shape}")
    data_path = transfer_dir / metadata["data_file"]
    expected_bytes = int(metadata["data_bytes"])
    if data_path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"transfer file has {data_path.stat().st_size} bytes, expected "
            f"{expected_bytes}"
        )
    mapped = torch.from_file(
        str(data_path),
        shared=False,
        size=shape[0] * shape[1] * shape[2] * shape[3] * shape[4],
        dtype=torch.bfloat16,
    ).view(shape)
    layer_shape = shape[2:]
    pinned = torch.empty((2, *layer_shape), dtype=torch.bfloat16, pin_memory=True)
    target = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    shared_to_pinned_ms = 0.0
    h2d_ms = 0.0
    wall_started = time.perf_counter()
    for layer in range(shape[0]):
        stage_started = time.perf_counter()
        pinned.copy_(mapped[layer])
        shared_to_pinned_ms += (time.perf_counter() - stage_started) * 1000
        start_event.record()
        target[layer].copy_(pinned, non_blocking=True)
        end_event.record()
        end_event.synchronize()
        h2d_ms += start_event.elapsed_time(end_event)
    wall_ms = (time.perf_counter() - wall_started) * 1000
    # Synchronizing one value proves the final layer reached the persistent target.
    expected_sample = float(mapped[-1, -1, -1, -1, -1].float().item())
    target_sample = float(target[-1, -1, -1, -1, -1].float().item())
    return {
        "shared_to_pinned": shared_to_pinned_ms,
        "h2d": h2d_ms,
        "wall": wall_ms,
        "target_mib": target.numel() * target.element_size() / MIB,
        "cpu_staging_mib": pinned.numel() * pinned.element_size() / MIB,
        "sample_match": target_sample == expected_sample,
    }, [mapped, pinned, target]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--transfer-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=parse_int_list, default=[256, 512, 1024])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--expected-layers", type=int, default=42)
    parser.add_argument("--token-id", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--replay-reference-ms", type=float, default=164.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats <= 0 or args.expected_layers <= 0 or args.token_id < 0:
        parser.error(
            "repeats and expected-layers must be positive; token-id must be "
            "non-negative"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.transfer_dir.mkdir(parents=True, exist_ok=True)
    srt_url = get_srt_url(args.base_url, args.timeout)

    cases = {}
    for token_count in args.tokens:
        samples = []
        token_ids = [args.token_id] * token_count
        for repeat in range(args.repeats):
            dump_id = f"t{token_count}r{repeat}n{time.time_ns():x}"
            metadata, request_wall_ms = request_export(
                srt_url, args.transfer_dir, token_ids, dump_id, args.timeout
            )
            if metadata.get("token_count") != token_count:
                raise RuntimeError(f"unexpected token count: {metadata}")
            if len(metadata.get("layer_ids", [])) != args.expected_layers:
                raise RuntimeError(f"unexpected layer count: {metadata}")
            imported, buffers = import_export(metadata, args.transfer_dir)
            if not imported["sample_match"]:
                raise RuntimeError("receiving-process H2D sample does not match")
            export = metadata["timings_ms"]
            samples.append(
                {
                    "request_wall": request_wall_ms,
                    "export_gather": export["gather"],
                    "export_d2h": export["d2h"],
                    "export_shared_write": export["shared_write"],
                    "export_wall": export["wall"],
                    "import_shared_to_pinned": imported["shared_to_pinned"],
                    "import_h2d": imported["h2d"],
                    "import_wall": imported["wall"],
                    "handoff_wall": export["wall"] + imported["wall"],
                }
            )
            target_mib = imported["target_mib"]
            cpu_staging_mib = imported["cpu_staging_mib"]
            del buffers
            (args.transfer_dir / metadata["data_file"]).unlink()
            (args.transfer_dir / f"{dump_id}.json").unlink()
            torch.cuda.empty_cache()

        means = {
            name: statistics.mean(sample[name] for sample in samples)
            for name in samples[0]
        }
        p95 = {
            name: percentile([sample[name] for sample in samples], 0.95)
            for name in samples[0]
        }
        cases[f"tokens-{token_count}"] = {
            "tokens": token_count,
            "repeats": args.repeats,
            "shape": metadata["shape"],
            "data_mib": metadata["data_bytes"] / MIB,
            "target_mib": target_mib,
            "cpu_staging_mib": cpu_staging_mib,
            "sample_match": True,
            "timings_ms_mean": means,
            "timings_ms_p95": p95,
            "handoff_vs_replay_speedup": args.replay_reference_ms
            / means["handoff_wall"],
            "samples": samples,
        }
        print(
            f"tokens={token_count} handoff={means['handoff_wall']:.2f} ms "
            f"request={means['request_wall']:.2f} ms",
            flush=True,
        )

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "srt_url": srt_url,
        "method": {
            "source": "live managed SRT NHD KV pool",
            "transport": "layer-streamed pinned D2H and /dev/shm mapped file",
            "target": "persistent BNSD CUDA tensor in the receiving process",
            "request_wall_includes": "diagnostic prefill, one token, export, and HTTP",
            "handoff_wall_includes": "export and receiving-process import only",
        },
        "settings": {
            "tokens": args.tokens,
            "repeats": args.repeats,
            "expected_layers": args.expected_layers,
            "replay_reference_ms": args.replay_reference_ms,
        },
        "cases": cases,
        "summary": {
            "all_handoffs_faster_than_replay": all(
                case["timings_ms_mean"]["handoff_wall"] < args.replay_reference_ms
                for case in cases.values()
            ),
            "max_handoff_ms": max(
                case["timings_ms_mean"]["handoff_wall"] for case in cases.values()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
