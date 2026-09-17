# SPDX-License-Identifier: Apache-2.0
"""SenseNova denoising KV microbenchmark; does not measure end-to-end images/s."""

import argparse
import json
import random
import statistics
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.testing

from sglang.kernels.ops.attention.sensenova_int8 import (
    _int8_prefix_attention,
    int8_image_attention,
    int8_prefix_attention,
    quantize_image_kv,
)
from sglang.multimodal_gen.runtime.layers.kvcache.sensenova import (
    quantize_prefix,
    validate_int8_kv_device,
)


def measure_interleaved(functions, rounds):
    """Compile/warm every path before timing; retain each round for paired analysis."""
    for fn in functions.values():
        triton.testing.do_bench(fn, warmup=100, rep=100, return_mode="median")
    samples = {name: [] for name in functions}
    rng = random.Random(42)
    for _ in range(rounds):
        order = list(functions)
        rng.shuffle(order)
        for name in order:
            samples[name].append(
                float(
                    triton.testing.do_bench(
                        functions[name], warmup=100, rep=300, return_mode="median"
                    )
                )
            )
    return {
        name: {
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "round_medians_ms": values,
        }
        for name, values in samples.items()
    }


def profile_image_paths(functions, output_dir, iterations):
    """Capture warmed paths separately; profiler times are diagnostic only."""
    from torch.profiler import ProfilerActivity, profile, record_function

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, fn in functions.items():
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            for _ in range(iterations):
                with record_function("sensenova/" + name):
                    fn()
            torch.cuda.synchronize()
        trace = output / f"{name}.trace.json"
        prof.export_chrome_trace(str(trace))
        (output / f"{name}.operators.txt").write_text(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50),
            encoding="utf-8",
        )
        # Read actual device events so a CPU-only capture cannot look successful.
        events = json.loads(trace.read_text(encoding="utf-8"))["traceEvents"]
        kernels = {}
        for event in events:
            if event.get("cat") != "kernel" or "dur" not in event:
                continue
            entry = kernels.setdefault(event["name"], {"calls": 0, "total_us": 0.0})
            entry["calls"] += 1
            entry["total_us"] += event["dur"]
        if not kernels:
            raise RuntimeError(
                f"No CUDA kernel events in {trace}; check CUPTI/profiler availability"
            )
        for entry in kernels.values():
            entry["mean_us"] = entry["total_us"] / entry["calls"]
        (output / f"{name}.kernels.json").write_text(
            json.dumps(kernels, indent=2), encoding="utf-8"
        )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefix-length", type=int, default=256)
    parser.add_argument("--image-tokens", type=int, default=1024)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--baseline", choices=("flash", "sdpa", "sdpa-gqa"), default="sdpa"
    )
    parser.add_argument(
        "--sweep-kernel",
        action="store_true",
        help="Measure experimental tile sizes; production defaults stay unchanged",
    )
    parser.add_argument("--profile-dir")
    parser.add_argument("--profile-iterations", type=int, default=10)
    parser.add_argument("--quantize-image", action="store_true")
    parser.add_argument("--resolution", type=int, choices=(512, 1024, 2048, 4096))
    parser.add_argument("--effective-patch-size", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    if args.profile_iterations <= 0:
        parser.error("profile-iterations must be positive")
    if args.profile_dir and not args.quantize_image:
        parser.error("profile-dir requires --quantize-image")
    if args.effective_patch_size <= 0:
        parser.error("effective-patch-size must be positive")
    if args.resolution is not None:
        if args.resolution % args.effective_patch_size:
            parser.error("Resolution must be divisible by effective-patch-size")
        args.image_tokens = (args.resolution // args.effective_patch_size) ** 2
    b, p, s, h, hk, d = (
        args.batch_size,
        args.prefix_length,
        args.image_tokens,
        args.query_heads,
        args.kv_heads,
        args.head_dim,
    )
    if min(b, s, h, hk, d, args.steps, args.rounds) <= 0 or p < 0 or h % hk or d > 256:
        parser.error("Invalid dimensions, GQA ratio, or step count")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    validate_int8_kv_device(torch.device("cuda"), dtype)
    if args.baseline == "flash":
        try:
            from flash_attn import flash_attn_func
        except ImportError:
            parser.error(
                "flash_attn_func unavailable; use --baseline sdpa (no FA3 required)"
            )

    torch.manual_seed(42)
    q = torch.randn(b, s, h, d, device="cuda", dtype=dtype).transpose(1, 2)
    k = torch.randn(b, s, hk, d, device="cuda", dtype=dtype).transpose(1, 2)
    v = torch.randn_like(k)
    # Same prompt, multiple images: one shared prefix, as in current main.
    pk = torch.randn(1, hk, p, d, device="cuda", dtype=dtype)
    pv = torch.randn_like(pk)

    def prepare_baseline():
        kb = torch.empty(b, p + s, hk, d, device="cuda", dtype=dtype)
        vb = torch.empty_like(kb)
        kb[:, :p].copy_(pk.transpose(1, 2))
        vb[:, :p].copy_(pv.transpose(1, 2))
        return kb, vb

    def prepare_int8():
        return *quantize_prefix(pk), *quantize_prefix(pv)

    kb, vb = prepare_baseline()
    ik, ks, iv, vs = prepare_int8()

    def baseline(mode=args.baseline):
        qb = q.transpose(1, 2).contiguous()
        kb[:, p:].copy_(k.transpose(1, 2).contiguous())
        vb[:, p:].copy_(v.transpose(1, 2).contiguous())
        if mode == "flash":
            return flash_attn_func(qb, kb, vb, causal=False)
        keys, values = kb.transpose(1, 2), vb.transpose(1, 2)
        if mode == "sdpa" and h != hk:
            keys = keys.repeat_interleave(h // hk, dim=1)
            values = values.repeat_interleave(h // hk, dim=1)
        return (
            F.scaled_dot_product_attention(
                qb.transpose(1, 2),
                keys,
                values,
                enable_gqa=mode == "sdpa-gqa",
            )
            .transpose(1, 2)
            .contiguous()
        )

    def quantized():
        return int8_prefix_attention(q, k, v, ik, iv, ks, vs, d**-0.5)

    def separated(quantized_prefix=False, bm=32, bn=64, warps=4):
        out = torch.empty((b, s, h, d), device=q.device, dtype=q.dtype)
        _int8_prefix_attention[(triton.cdiv(s, bm), h, b)](
            q,
            k,
            v,
            ik if quantized_prefix else pk,
            iv if quantized_prefix else pv,
            ks,
            vs,
            out,
            q.stride(),
            k.stride(),
            v.stride(),
            h,
            hk,
            s,
            p,
            d,
            True,
            True,
            d**-0.5,
            bm,
            bn,
            max(16, triton.next_power_of_2(d)),
            num_warps=warps,
            num_stages=2,
        )
        return out

    if args.quantize_image:
        image_k, image_v, image_ks, image_vs = quantize_image_kv(k, v)

        def image_attention():
            return int8_image_attention(
                q, image_k, image_v, pk, pv, image_ks, image_vs, d**-0.5
            )

        def image_total():
            new_k, new_v, new_ks, new_vs = quantize_image_kv(k, v)
            return int8_image_attention(
                q, new_k, new_v, pk, pv, new_ks, new_vs, d**-0.5
            )

        functions = {
            "model_sdpa": partial(baseline, "sdpa"),
            "native_gqa": partial(baseline, "sdpa-gqa"),
            "unquantized_separated": partial(separated, False, 64, 32, 4),
            "image_int8_attention": image_attention,
            "image_int8_total": image_total,
        }
        ref = baseline("sdpa-gqa").float()
        errors = {}
        for name, fn in functions.items():
            result = fn().float()
            error = ((result - ref).norm() / ref.norm().clamp_min(1e-12)).item()
            if not torch.isfinite(result).all() or not 0 <= error <= 0.025:
                raise AssertionError(f"{name} relative L2 error: {error}")
            errors[name] = error
        del ref, result
        functions["image_quantization"] = partial(quantize_image_kv, k, v)
        timings = measure_interleaved(functions, args.rounds)
        if args.profile_dir:
            profile_image_paths(functions, args.profile_dir, args.profile_iterations)
        # All comparison buffers remain resident. This measures only the
        # additional live allocations of each invocation, not model peak memory.
        extra_peak = {}
        for name, fn in functions.items():
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            result = fn()
            torch.cuda.synchronize()
            extra_peak[name] = torch.cuda.max_memory_allocated() - before
            del result
        print(
            json.dumps(
                {
                    "benchmark_version": 4,
                    "quantization_target": "image_kv",
                    "device": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "triton": triton.__version__,
                    **vars(args),
                    "timings": timings,
                    "relative_l2_errors": errors,
                    "extra_peak_allocated_bytes": extra_peak,
                    "image_fp_bytes_per_layer": k.nbytes + v.nbytes,
                    "image_int8_bytes_per_layer": sum(
                        t.nbytes for t in (image_k, image_v, image_ks, image_vs)
                    ),
                    "shared_prefix_bytes_per_layer": pk.nbytes + pv.nbytes,
                    "speedup_vs_native_gqa": timings["native_gqa"]["median_ms"]
                    / timings["image_int8_total"]["median_ms"],
                    "speedup_vs_model_sdpa": timings["model_sdpa"]["median_ms"]
                    / timings["image_int8_total"]["median_ms"],
                },
                indent=2,
            )
        )
        return

    ref = baseline("sdpa").float()
    functions = {
        "model_sdpa": partial(baseline, "sdpa"),
        "native_gqa": partial(baseline, "sdpa-gqa"),
        "unquantized_separated": separated,
        "int8": quantized,
    }
    if args.baseline == "flash":
        functions["flash"] = baseline
    sweep = []
    if args.sweep_kernel:
        for bm, bn, warps in ((32, 64, 4), (64, 32, 4), (64, 64, 4), (64, 64, 8)):
            row = {"block_m": bm, "block_n": bn, "num_warps": warps}
            for quant in (False, True):
                name = "int8" if quant else "unquantized"
                key = f"{name}_{bm}_{bn}_{warps}"
                fn = partial(separated, quant, bm, bn, warps)
                try:
                    fn()  # Compile before accepting a candidate for measurement.
                    torch.cuda.synchronize()
                except triton.OutOfResources as exc:
                    row[name + "_unsupported"] = str(exc)
                    print(f"Skipping {key}: {exc}", file=sys.stderr)
                    continue
                functions[key] = fn
                row[name + "_measurement"] = key
            sweep.append(row)
    errors = {}
    for name, fn in functions.items():
        result = fn().float()
        error = ((result - ref).norm() / ref.norm().clamp_min(1e-12)).item()
        if not torch.isfinite(result).all() or not 0 <= error <= 0.025:
            raise AssertionError(f"{name} relative L2 error: {error}")
        errors[name] = error
    del result, ref
    timings = measure_interleaved(functions, args.rounds)
    preparation = measure_interleaved(
        {"baseline": prepare_baseline, "int8": prepare_int8}, args.rounds
    )
    for row in sweep:
        for name in ("unquantized", "int8"):
            key = row.get(name + "_measurement")
            if key is not None:
                row[name + "_relative_l2_error"] = errors[key]
                row[name + "_attention_ms"] = timings[key]["median_ms"]
    baseline_key = {"sdpa": "model_sdpa", "sdpa-gqa": "native_gqa", "flash": "flash"}[
        args.baseline
    ]
    baseline_ms = timings[baseline_key]["median_ms"]
    int8_ms = timings["int8"]["median_ms"]
    baseline_prepare_ms = preparation["baseline"]["median_ms"]
    int8_prepare_ms = preparation["int8"]["median_ms"]
    print(
        json.dumps(
            {
                "benchmark_version": 3,
                "timings": timings,
                "preparation_timings": preparation,
                "relative_l2_errors": errors,
                "model_sdpa_attention_ms": timings["model_sdpa"]["median_ms"],
                "native_gqa_attention_ms": timings["native_gqa"]["median_ms"],
                "unquantized_separated_attention_ms": timings["unquantized_separated"][
                    "median_ms"
                ],
                "unquantized_separated_cache_bytes_per_layer": pk.nbytes + pv.nbytes,
                "kernel_sweep": sweep,
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "triton": triton.__version__,
                **vars(args),
                "baseline_cache_bytes_per_layer": sum(
                    t.nbytes for t in (pk, pv, kb, vb)
                ),
                "int8_cache_bytes_per_layer": sum(t.nbytes for t in (ik, iv, ks, vs)),
                "baseline_prepare_ms": baseline_prepare_ms,
                "int8_prepare_ms": int8_prepare_ms,
                "baseline_attention_ms": baseline_ms,
                "int8_attention_ms": int8_ms,
                "amortized_attention_speedup": (
                    baseline_prepare_ms + args.steps * baseline_ms
                )
                / (int8_prepare_ms + args.steps * int8_ms),
                "relative_l2_error": errors["int8"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
