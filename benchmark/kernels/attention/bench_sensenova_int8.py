# SPDX-License-Identifier: Apache-2.0
"""SenseNova denoising KV microbenchmark; does not measure end-to-end images/s."""

import argparse
import json

import torch
import torch.nn.functional as F
import triton
import triton.testing

from sglang.kernels.ops.attention.sensenova_int8 import int8_prefix_attention
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
    parser.add_argument("--baseline", choices=("flash", "sdpa"), default="flash")
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
        from flash_attn import flash_attn_func

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

    def baseline():
        qb = q.transpose(1, 2).contiguous()
        kb[:, p:].copy_(k.transpose(1, 2).contiguous())
        vb[:, p:].copy_(v.transpose(1, 2).contiguous())
        if args.baseline == "flash":
            return flash_attn_func(qb, kb, vb, causal=False)
        return F.scaled_dot_product_attention(
            qb.transpose(1, 2),
            kb.transpose(1, 2),
            vb.transpose(1, 2),
            enable_gqa=True,
        ).transpose(1, 2)

    def quantized():
        return int8_prefix_attention(q, k, v, ik, iv, ks, vs, d**-0.5)

    ref = baseline().float()
    error = ((quantized().float() - ref).norm() / ref.norm()).item()
    baseline_ms = triton.testing.do_bench(baseline)
    int8_ms = triton.testing.do_bench(quantized)
    baseline_prepare_ms = triton.testing.do_bench(prepare_baseline)
    int8_prepare_ms = triton.testing.do_bench(prepare_int8)
    print(
        json.dumps(
            {
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
