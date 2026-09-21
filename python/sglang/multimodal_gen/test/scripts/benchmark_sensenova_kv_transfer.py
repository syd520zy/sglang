"""Benchmark layer-streamed SenseNova KV transfer through shared CPU memory."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

MIB = 1024**2
GIB = 1024**3


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",")]
    if not values or any(item <= 0 for item in values):
        raise ValueError("expected comma-separated positive integers")
    return values


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * quantile))
    return ordered[index]


def benchmark_case(
    *,
    tokens: int,
    batch_size: int,
    layers: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    repeats: int,
) -> dict:
    device = torch.device("cuda")
    total_slots = tokens * batch_size
    tensor_shape = (total_slots, kv_heads, head_dim)
    cache_shape = (batch_size, kv_heads, tokens, head_dim)
    element_size = torch.empty((), dtype=dtype).element_size()
    layer_kv_bytes = 2 * total_slots * kv_heads * head_dim * element_size
    total_kv_bytes = layers * layer_kv_bytes

    buffers = {
        "source_keys": [
            torch.empty(tensor_shape, dtype=dtype, device=device) for _ in range(layers)
        ],
        "source_values": [
            torch.empty(tensor_shape, dtype=dtype, device=device) for _ in range(layers)
        ],
        "slots": torch.randperm(total_slots, device=device),
    }
    torch.cuda.synchronize()
    source_baseline = torch.cuda.memory_allocated(device)

    outbound_shape = (batch_size, tokens, kv_heads, head_dim)
    buffers.update(
        target_keys=[
            torch.empty(cache_shape, dtype=dtype, device=device) for _ in range(layers)
        ],
        target_values=[
            torch.empty(cache_shape, dtype=dtype, device=device) for _ in range(layers)
        ],
        pinned_out_k=torch.empty(outbound_shape, dtype=dtype, pin_memory=True),
        pinned_out_v=torch.empty(outbound_shape, dtype=dtype, pin_memory=True),
        shared_k=torch.empty(cache_shape, dtype=dtype).share_memory_(),
        shared_v=torch.empty(cache_shape, dtype=dtype).share_memory_(),
        pinned_in_k=torch.empty(cache_shape, dtype=dtype, pin_memory=True),
        pinned_in_v=torch.empty(cache_shape, dtype=dtype, pin_memory=True),
    )
    events = {
        name: torch.cuda.Event(enable_timing=True)
        for name in (
            "gather_start",
            "gather_end",
            "d2h_start",
            "d2h_end",
            "h2d_start",
            "h2d_end",
        )
    }
    torch.cuda.synchronize()

    def one_layer(layer: int) -> dict[str, float]:
        events["gather_start"].record()
        gathered_k = (
            buffers["source_keys"][layer]
            .index_select(0, buffers["slots"])
            .view(batch_size, tokens, kv_heads, head_dim)
        )
        gathered_v = (
            buffers["source_values"][layer]
            .index_select(0, buffers["slots"])
            .view(batch_size, tokens, kv_heads, head_dim)
        )
        events["gather_end"].record()

        events["d2h_start"].record()
        buffers["pinned_out_k"].copy_(gathered_k, non_blocking=True)
        buffers["pinned_out_v"].copy_(gathered_v, non_blocking=True)
        events["d2h_end"].record()
        events["d2h_end"].synchronize()
        gather_ms = events["gather_start"].elapsed_time(events["gather_end"])
        d2h_ms = events["d2h_start"].elapsed_time(events["d2h_end"])
        gathered_k = gathered_v = None

        started = time.perf_counter()
        buffers["shared_k"].copy_(buffers["pinned_out_k"].permute(0, 2, 1, 3))
        buffers["shared_v"].copy_(buffers["pinned_out_v"].permute(0, 2, 1, 3))
        shared_layout_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        buffers["pinned_in_k"].copy_(buffers["shared_k"])
        buffers["pinned_in_v"].copy_(buffers["shared_v"])
        shared_to_pinned_ms = (time.perf_counter() - started) * 1000

        events["h2d_start"].record()
        buffers["target_keys"][layer].copy_(buffers["pinned_in_k"], non_blocking=True)
        buffers["target_values"][layer].copy_(buffers["pinned_in_v"], non_blocking=True)
        events["h2d_end"].record()
        events["h2d_end"].synchronize()
        h2d_ms = events["h2d_start"].elapsed_time(events["h2d_end"])
        return {
            "gather": gather_ms,
            "d2h": d2h_ms,
            "shared_layout": shared_layout_ms,
            "shared_to_pinned": shared_to_pinned_ms,
            "h2d": h2d_ms,
        }

    one_layer(0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    samples = []
    for _ in range(repeats):
        totals = {
            "gather": 0.0,
            "d2h": 0.0,
            "shared_layout": 0.0,
            "shared_to_pinned": 0.0,
            "h2d": 0.0,
        }
        started = time.perf_counter()
        for layer in range(layers):
            for stage, duration in one_layer(layer).items():
                totals[stage] += duration
        torch.cuda.synchronize()
        totals["wall"] = (time.perf_counter() - started) * 1000
        totals["stage_sum"] = sum(
            totals[stage]
            for stage in ("gather", "d2h", "shared_layout", "shared_to_pinned", "h2d")
        )
        samples.append(totals)

    peak_increment = torch.cuda.max_memory_allocated(device) - source_baseline
    means = {
        stage: statistics.mean(sample[stage] for sample in samples)
        for stage in samples[0]
    }
    p95 = {
        stage: percentile([sample[stage] for sample in samples], 0.95)
        for stage in samples[0]
    }
    stage_bytes = {
        "gather": total_kv_bytes,
        "d2h": total_kv_bytes,
        "shared_layout": total_kv_bytes,
        "shared_to_pinned": total_kv_bytes,
        "h2d": total_kv_bytes,
    }
    bandwidth = {
        stage: stage_bytes[stage] / GIB / (means[stage] / 1000) for stage in stage_bytes
    }
    result = {
        "batch_size": batch_size,
        "tokens": tokens,
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "dtype": str(dtype),
        "layer_kv_mib": layer_kv_bytes / MIB,
        "total_kv_mib": total_kv_bytes / MIB,
        "cpu_memory": {
            "shared_mib": layer_kv_bytes / MIB,
            "pinned_mib": 2 * layer_kv_bytes / MIB,
            "total_staging_mib": 3 * layer_kv_bytes / MIB,
        },
        "timings_ms_mean": means,
        "timings_ms_p95": p95,
        "bandwidth_gib_s": bandwidth,
        "gpu_memory": {
            "source_cache_mib": total_kv_bytes / MIB,
            "target_cache_mib": total_kv_bytes / MIB,
            "peak_increment_mib": peak_increment / MIB,
            "temporary_overhead_mib": max(0.0, (peak_increment - total_kv_bytes) / MIB),
        },
    }

    buffers.clear()
    events.clear()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", type=parse_int_list, default=parse_int_list("256,512,1024,4096")
    )
    parser.add_argument(
        "--batch-sizes", type=parse_int_list, default=parse_int_list("1,2")
    )
    parser.add_argument("--layers", type=int, default=42)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--replay-reference-ms", type=float, default=164.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.layers <= 0
        or args.kv_heads <= 0
        or args.head_dim <= 0
        or args.repeats <= 0
    ):
        parser.error("layers, kv-heads, head-dim and repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    cases = {}
    for batch_size in args.batch_sizes:
        for tokens in args.tokens:
            name = f"batch-{batch_size}_tokens-{tokens}"
            print(f"Running {name}...", flush=True)
            case = benchmark_case(
                tokens=tokens,
                batch_size=batch_size,
                layers=args.layers,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                dtype=torch.bfloat16,
                repeats=args.repeats,
            )
            case["replay_reference_ms"] = args.replay_reference_ms
            case["transfer_vs_replay_speedup"] = (
                args.replay_reference_ms / case["timings_ms_mean"]["wall"]
            )
            cases[name] = case
            print(
                f"  wall={case['timings_ms_mean']['wall']:.2f} ms "
                f"temporary_gpu={case['gpu_memory']['temporary_overhead_mib']:.1f} MiB",
                flush=True,
            )

    common_cases = [
        case
        for case in cases.values()
        if case["tokens"] <= 512 and case["batch_size"] <= 2
    ]
    common_transfer_times = [case["timings_ms_mean"]["wall"] for case in common_cases]
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "method": {
            "source_layout": "synthetic NHD pool with randomized token slots",
            "target_layout": "persistent BNSD cache",
            "execution": "serialized layer stream without transfer overlap",
            "included": [
                "GPU slot gather",
                "pinned D2H",
                "shared-memory layout conversion",
                "shared-memory to pinned ingress",
                "pinned H2D into the persistent target cache",
            ],
            "excluded": [
                "model execution contention",
                "cross-process control-plane latency",
            ],
        },
        "settings": {
            "tokens": args.tokens,
            "batch_sizes": args.batch_sizes,
            "layers": args.layers,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "repeats": args.repeats,
            "replay_reference_ms": args.replay_reference_ms,
        },
        "cases": cases,
        "summary": {
            "common_cases_faster_than_replay": (
                all(
                    transfer_ms < args.replay_reference_ms
                    for transfer_ms in common_transfer_times
                )
                if common_transfer_times
                else None
            ),
            "max_common_transfer_ms": (
                max(common_transfer_times) if common_transfer_times else None
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
