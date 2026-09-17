# SPDX-License-Identifier: Apache-2.0
"""Bidirectional image attention over an INT8 prefix and FP16/BF16 image K/V.

Dequantization is tile-local. Both segments share one online softmax, rather than
normalizing the prefix and image attention separately. No full-precision prefix
or concatenated KV tensor is written to device memory.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _int8_prefix_attention(
    Q,
    K,
    V,
    PK,
    PV,
    KS,
    VS,
    O,
    q_stride: tl.constexpr,
    k_stride: tl.constexpr,
    v_stride: tl.constexpr,
    H: tl.constexpr,
    HKV: tl.constexpr,
    S: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    SHARED_K: tl.constexpr,
    SHARED_V: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    kv_head = head // (H // HKV)
    rows = block * BM + tl.arange(0, BM)
    dims = tl.arange(0, BD)
    cols = tl.arange(0, BN)
    q = tl.load(
        Q
        + batch * q_stride[0]
        + head * q_stride[1]
        + rows[:, None] * q_stride[2]
        + dims[None, :] * q_stride[3],
        (rows[:, None] < S) & (dims[None, :] < D),
        other=0,
    )
    maximum = tl.full((BM,), float("-inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    accumulator = tl.zeros((BM, BD), tl.float32)

    for segment in tl.static_range(2):
        if segment == 0:
            length = P
        else:
            length = S
        for start in range(0, length, BN):
            tokens = start + cols
            if segment == 0:
                kb = 0 if SHARED_K else batch
                vb = 0 if SHARED_V else batch
                ki = (kb * HKV + kv_head) * P + tokens
                vi = (vb * HKV + kv_head) * P + tokens
                key = tl.load(
                    PK + ki[None, :] * D + dims[:, None],
                    (tokens[None, :] < P) & (dims[:, None] < D),
                    other=0,
                )
                value = tl.load(
                    PV + vi[:, None] * D + dims[None, :],
                    (tokens[:, None] < P) & (dims[None, :] < D),
                    other=0,
                )
                # Full-precision specialization is used only by the benchmark
                # to separate storage-layout savings from quantization costs.
                if PK.dtype.element_ty == tl.int8:
                    key_scale = tl.load(KS + ki, tokens < P, other=0)
                    value_scale = tl.load(VS + vi, tokens < P, other=0)
                    key = (key.to(tl.float32) * key_scale[None, :]).to(q.dtype)
                    value = (value.to(tl.float32) * value_scale[:, None]).to(q.dtype)
            else:
                key = tl.load(
                    K
                    + batch * k_stride[0]
                    + kv_head * k_stride[1]
                    + tokens[None, :] * k_stride[2]
                    + dims[:, None] * k_stride[3],
                    (tokens[None, :] < S) & (dims[:, None] < D),
                    other=0,
                )
                value = tl.load(
                    V
                    + batch * v_stride[0]
                    + kv_head * v_stride[1]
                    + tokens[:, None] * v_stride[2]
                    + dims[None, :] * v_stride[3],
                    (tokens[:, None] < S) & (dims[None, :] < D),
                    other=0,
                )
            scores = tl.dot(q, key) * (SCALE * 1.4426950408889634)
            scores = tl.where(tokens[None, :] < length, scores, float("-inf"))
            new_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
            probabilities = tl.exp2(scores - new_maximum[:, None])
            correction = tl.exp2(maximum - new_maximum)
            accumulator = accumulator * correction[:, None]
            accumulator = tl.dot(probabilities.to(q.dtype), value, accumulator)
            denominator = denominator * correction + tl.sum(probabilities, axis=1)
            maximum = new_maximum

    result = (accumulator / denominator[:, None]).to(O.dtype.element_ty)
    tl.store(
        O + ((batch * S + rows[:, None]) * H + head) * D + dims[None, :],
        result,
        (rows[:, None] < S) & (dims[None, :] < D),
    )


def int8_prefix_attention(q, k, v, prefix_k, prefix_v, k_scale, v_scale, softmax_scale):
    """Inference-only [B,H,S,D] inputs, [B,S,H,D] output; no mask/dropout."""
    tensors = (q, k, v, prefix_k, prefix_v, k_scale, v_scale)
    if not q.is_cuda or torch.version.hip is not None:
        raise ValueError("INT8 prefix attention requires NVIDIA CUDA")
    if any(t.device != q.device for t in tensors):
        raise ValueError("All attention inputs must be on the same device")
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("Expected [B,H,S,D] Q/K/V with matching K/V shapes")
    b, h, s, d = q.shape
    if min(b, h, s, d) <= 0 or d > 256:
        raise ValueError("Nonempty Q with head dimension <= 256 is required")
    if k.shape[1] == 0 or h % k.shape[1] or (k.shape[0], *k.shape[2:]) != (b, s, d):
        raise ValueError(
            "Current K/V must match Q batch/length/dimension and support GQA"
        )
    if (
        q.dtype not in (torch.float16, torch.bfloat16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise ValueError("Q and current K/V must have the same FP16/BF16 dtype")
    for prefix, scale in ((prefix_k, k_scale), (prefix_v, v_scale)):
        if prefix.ndim != 4 or prefix.shape[0] not in (1, b):
            raise ValueError("Prefix must have batch size 1 or match Q")
        if prefix.shape[1] != k.shape[1] or prefix.shape[3] != d:
            raise ValueError("Prefix head count/dimension must match current K/V")
        if prefix.dtype != torch.int8 or not prefix.is_contiguous():
            raise ValueError("Prefix must be contiguous INT8")
        if (
            scale.shape != prefix.shape[:3]
            or scale.dtype != torch.float32
            or not scale.is_contiguous()
        ):
            raise ValueError("Prefix scales must be contiguous FP32 [B,Hkv,S]")
    if prefix_k.shape[2] != prefix_v.shape[2]:
        raise ValueError("Prefix K and V must have the same length")

    out = torch.empty((b, s, h, d), device=q.device, dtype=q.dtype)
    _int8_prefix_attention[(triton.cdiv(s, 32), h, b)](
        q,
        k,
        v,
        prefix_k,
        prefix_v,
        k_scale,
        v_scale,
        out,
        q.stride(),
        k.stride(),
        v.stride(),
        h,
        k.shape[1],
        s,
        prefix_k.shape[2],
        d,
        prefix_k.shape[0] == 1,
        prefix_v.shape[0] == 1,
        softmax_scale,
        32,
        64,
        max(16, triton.next_power_of_2(d)),
        num_warps=4,
        num_stages=2,
    )
    return out
