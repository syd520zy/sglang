# SPDX-License-Identifier: Apache-2.0
"""Read-only INT8 prefixes for SenseNova image denoising.

Current-image K/V are intentionally not cached: every layer recomputes them on
every denoising step. Prefixes use [B, Hkv, S, D], with FP32 scales [B, Hkv, S].
"""

from dataclasses import dataclass
from importlib.util import find_spec

import torch


def validate_int8_kv_device(device, dtype):
    if torch.device(device).type != "cuda" or torch.version.hip is not None:
        raise ValueError("SenseNova INT8 KV requires an NVIDIA CUDA device")
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("SenseNova INT8 KV requires FP16 or BF16 model precision")
    if torch.cuda.get_device_capability(device)[0] < 8:
        raise ValueError("SenseNova INT8 KV requires SM80 or newer (A800/RTX 4090)")
    if find_spec("triton") is None:
        raise RuntimeError("SenseNova INT8 KV requires Triton")


def quantize_prefix(x):
    """Symmetric per-token/head quantization; preserve shared batch storage."""
    if x.ndim != 4 or x.shape[-1] == 0:
        raise ValueError("Expected a [B, Hkv, S, D] prefix with nonzero head dimension")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Prefix dtype must be FP16 or BF16")
    if x.stride(0) == 0:
        x = x[:1]
    x = x.float()
    scale = x.abs().amax(dim=-1).clamp_min(1e-12) / 127
    quantized = (x / scale[..., None]).round().clamp_(-127, 127).to(torch.int8)
    return quantized.contiguous(), scale.contiguous()


@dataclass
class Int8Prefix:
    keys: torch.Tensor
    values: torch.Tensor
    key_scale: torch.Tensor
    value_scale: torch.Tensor


class Int8DenoisingCache:
    """Denoising-only cache, consumed after prefill/Think has finished.

    This is not an autoregressive Transformers Cache. SenseNova's image-only
    forward reads ``layers`` and ``get_seq_length`` with a precomputed mask.
    Appending tokens is deliberately unsupported.
    """

    def __init__(self, layers):
        self.layers = layers

    @classmethod
    @torch.no_grad()
    def from_cache(cls, cache):
        # Validate before consuming anything, including layers after the first.
        for layer in cache.layers:
            k, v = layer.keys, layer.values
            if k is None or v is None or k.shape != v.shape:
                raise ValueError(
                    "Prefix K and V must be populated and have the same shape"
                )
            if k.ndim != 4 or not 0 < k.shape[-1] <= 256:
                raise ValueError("INT8 KV requires [B, Hkv, S, D] with 0 < D <= 256")
            if k.dtype not in (torch.float16, torch.bfloat16) or k.dtype != v.dtype:
                raise ValueError("Prefix K and V must have the same FP16 or BF16 dtype")
            if k.device != v.device:
                raise ValueError("Prefix K and V must be on the same device")
        layers = []
        for layer in cache.layers:
            k, ks = quantize_prefix(layer.keys)
            v, vs = quantize_prefix(layer.values)
            layers.append(Int8Prefix(k, v, ks, vs))
            # The caller transfers ownership at the prefill -> denoise boundary.
            layer.keys = None
            layer.values = None
        return cls(layers)

    def get_seq_length(self, layer_idx=0):
        return self.layers[layer_idx].keys.shape[2] if self.layers else 0

    def update(self, *args, **kwargs):
        raise RuntimeError(
            "INT8 denoising prefixes are read-only; finish prefill first"
        )

    def attention(self, layer_idx, q, k, v, softmax_scale):
        from sglang.kernels.ops.attention.sensenova_int8 import int8_prefix_attention

        prefix = self.layers[layer_idx]
        return int8_prefix_attention(
            q,
            k,
            v,
            prefix.keys,
            prefix.values,
            prefix.key_scale,
            prefix.value_scale,
            softmax_scale,
        )
