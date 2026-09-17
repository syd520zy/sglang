# SPDX-License-Identifier: Apache-2.0
"""SenseNova denoising KV microbenchmark; does not measure end-to-end images/s."""

import argparse
import json
import sys

import torch
import torch.nn.functional as F
import triton
import triton.testing

from sglang.kernels.ops.attention.sensenova_int8 import (
    _int8_prefix_attention,
    int8_prefix_attention,
)
from sglang.multimodal_gen.runtime.layers.kvcache.sensenova import (
    quantize_prefix,
    validate_int8_kv_device,
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
    args = parser.parse_args()
    b, p, s, h, hk, d = (
        args.batch_size,
        args.prefix_length,
        args.image_tokens,
        args.query_heads,
        args.kv_heads,
        args.head_dim,
    )
    if min(b, s, h, hk, d, args.steps) <= 0 or p < 0 or h % hk or d > 256:
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

    ref = baseline().float()
    torch.testing.assert_close(separated().float(), ref, atol=0.015, rtol=0.015)
    native_gqa_ms = triton.testing.do_bench(lambda: baseline("sdpa-gqa"))
    model_sdpa_ms = triton.testing.do_bench(lambda: baseline("sdpa"))
    separated_ms = triton.testing.do_bench(separated)
    sweep = []
    if args.sweep_kernel:
        for bm, bn, warps in ((32, 64, 4), (64, 32, 4), (64, 64, 4), (64, 64, 8)):
            row = {"block_m": bm, "block_n": bn, "num_warps": warps}
            try:
                for quant in (False, True):
                    name = "int8" if quant else "unquantized"
                    fn = lambda: separated(quant, bm, bn, warps)
                    result = fn().float()
                    relative_error = ((result - ref).norm() / ref.norm()).item()
                    if not torch.isfinite(result).all() or relative_error > 0.025:
                        raise AssertionError(
                            f"{name} relative L2 error: {relative_error}"
                        )
                    row[name + "_relative_l2_error"] = relative_error
                    row[name + "_attention_ms"] = triton.testing.do_bench(fn)
            except triton.OutOfResources as exc:
                row["unsupported"] = str(exc)
                print(
                    f"Skipping resource-limited configuration: {row}", file=sys.stderr
                )
            sweep.append(row)
    error = ((quantized().float() - ref).norm() / ref.norm()).item()
    baseline_ms = triton.testing.do_bench(baseline)
    int8_ms = triton.testing.do_bench(quantized)
    baseline_prepare_ms = triton.testing.do_bench(prepare_baseline)
    int8_prepare_ms = triton.testing.do_bench(prepare_int8)
    print(
        json.dumps(
            {
                "benchmark_version": 2,
                "model_sdpa_attention_ms": model_sdpa_ms,
                "native_gqa_attention_ms": native_gqa_ms,
                "unquantized_separated_attention_ms": separated_ms,
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
                "relative_l2_error": error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
