# SPDX-License-Identifier: Apache-2.0
"""Bidirectional SenseNova attention with tile-local INT8 KV dequantization.

Prefix and image segments share one online softmax. The production entry point
quantizes the prefix; the experimental image entry point keeps the prefix in
FP16/BF16 and quantizes current image K/V every step.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


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
    IMAGE_KS=None,
    IMAGE_VS=None,
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
                if K.dtype.element_ty == tl.int8:
                    si = (batch * HKV + kv_head) * S + tokens
                    key_scale = tl.load(IMAGE_KS + si, tokens < S, other=0)
                    value_scale = tl.load(IMAGE_VS + si, tokens < S, other=0)
                    key = (key.to(tl.float32) * key_scale[None, :]).to(q.dtype)
                    value = (value.to(tl.float32) * value_scale[:, None]).to(q.dtype)
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


@triton.jit
def _quantize_image_kv(
    K,
    V,
    IK,
    IV,
    KS,
    VS,
    k_stride: tl.constexpr,
    v_stride: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    dims = tl.arange(0, BD)
    ki = batch * k_stride[0] + head * k_stride[1] + token * k_stride[2]
    vi = batch * v_stride[0] + head * v_stride[1] + token * v_stride[2]
    key = tl.load(K + ki + dims * k_stride[3], dims < D, other=0).to(tl.float32)
    value = tl.load(V + vi + dims * v_stride[3], dims < D, other=0).to(tl.float32)
    ks = tl.maximum(tl.max(tl.abs(key), 0), 1e-12) / 127.0
    vs = tl.maximum(tl.max(tl.abs(value), 0), 1e-12) / 127.0
    index = (batch * H + head) * S + token
    tl.store(KS + index, ks)
    tl.store(VS + index, vs)
    # Round-to-nearest-even, matching the prefix quantizer.
    qk = libdevice.nearbyint(key / ks)
    qv = libdevice.nearbyint(value / vs)
    tl.store(
        IK + index * D + dims,
        tl.minimum(tl.maximum(qk, -127), 127).to(tl.int8),
        dims < D,
    )
    tl.store(
        IV + index * D + dims,
        tl.minimum(tl.maximum(qv, -127), 127).to(tl.int8),
        dims < D,
    )


def quantize_image_kv(k, v):
    """Quantize current [B,Hkv,S,D] K/V together; call again every denoising step.

    K must already include normalization and RoPE. Inputs must be finite.
    Returns contiguous INT8 K/V and FP32 per-token/head scales.
    """
    if not k.is_cuda or torch.version.hip is not None:
        raise ValueError("Image KV quantization requires NVIDIA CUDA")
    if k.ndim != 4 or v.shape != k.shape or min(k.shape) <= 0 or k.shape[-1] > 256:
        raise ValueError("Expected matching nonempty [B,Hkv,S,D] K/V with D <= 256")
    if (
        k.dtype not in (torch.float16, torch.bfloat16)
        or v.dtype != k.dtype
        or v.device != k.device
    ):
        raise ValueError("K/V must share FP16/BF16 dtype and device")
    ik = torch.empty(k.shape, device=k.device, dtype=torch.int8)
    iv = torch.empty_like(ik)
    ks = torch.empty(k.shape[:3], device=k.device, dtype=torch.float32)
    vs = torch.empty_like(ks)
    b, h, s, d = k.shape
    _quantize_image_kv[(s, h, b)](
        k,
        v,
        ik,
        iv,
        ks,
        vs,
        k.stride(),
        v.stride(),
        h,
        s,
        d,
        triton.next_power_of_2(d),
        num_warps=4,
    )
    return ik, iv, ks, vs


def int8_image_attention(q, k, v, prefix_k, prefix_v, k_scale, v_scale, softmax_scale):
    """Experimental inference attention: INT8 image KV, full-precision prefix.

    Inputs use [B,H,S,D]; output uses [B,S,H,D]. No mask or dropout.
    Quantization is separate so callers can measure its per-step cost.
    """
    if not q.is_cuda or torch.version.hip is not None:
        raise ValueError("Image INT8 attention requires NVIDIA CUDA")
    if q.ndim != 4 or min(q.shape) <= 0 or q.shape[-1] > 256:
        raise ValueError("Expected nonempty [B,H,S,D] Q with D <= 256")
    b, h, s, d = q.shape
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Q must be FP16/BF16")
    if k.ndim != 4 or v.shape != k.shape or k.shape[1] <= 0:
        raise ValueError("Expected matching [B,Hkv,S,D] image K/V")
    hk = k.shape[1]
    if (k.shape[0], *k.shape[2:]) != (b, s, d) or h % hk:
        raise ValueError("Image K/V must match Q dimensions and support GQA")
    for kv, scale in ((k, k_scale), (v, v_scale)):
        if kv.dtype != torch.int8 or not kv.is_contiguous():
            raise ValueError("Image K/V must be contiguous INT8")
        if (
            scale.shape != kv.shape[:3]
            or scale.dtype != torch.float32
            or not scale.is_contiguous()
        ):
            raise ValueError("Image scales must be contiguous FP32 [B,Hkv,S]")
    for prefix in (prefix_k, prefix_v):
        if (
            prefix.ndim != 4
            or prefix.shape[0] not in (1, b)
            or prefix.shape[1] != hk
            or prefix.shape[3] != d
        ):
            raise ValueError(
                "Prefix must match KV heads/dimension and have batch 1 or B"
            )
        if prefix.dtype != q.dtype or not prefix.is_contiguous():
            raise ValueError("Prefix must be contiguous with Q dtype")
    if prefix_k.shape[2] != prefix_v.shape[2]:
        raise ValueError("Prefix K/V lengths must match")
    if any(t.device != q.device for t in (k, v, prefix_k, prefix_v, k_scale, v_scale)):
        raise ValueError("All inputs must share a device")
    out = torch.empty((b, s, h, d), device=q.device, dtype=q.dtype)
    _int8_prefix_attention[(triton.cdiv(s, 64), h, b)](
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
        hk,
        s,
        prefix_k.shape[2],
        d,
        prefix_k.shape[0] == 1,
        prefix_v.shape[0] == 1,
        softmax_scale,
        64,
        32,
        max(16, triton.next_power_of_2(d)),
        IMAGE_KS=k_scale,
        IMAGE_VS=v_scale,
        num_warps=4,
        num_stages=2,
    )
    return out
