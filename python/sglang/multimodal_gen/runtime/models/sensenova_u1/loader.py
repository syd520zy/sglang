# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc
import logging
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models import (  # noqa: F401
    sensenova_u1 as _sensenova_u1,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.srt_thinking import (
    compact_mode_enabled,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision_types import PRECISION_TO_TYPE

logger = logging.getLogger(__name__)

_DENSE_LAYER_MODULES = ("mlp", "input_layernorm", "post_attention_layernorm")
_DENSE_ATTENTION_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "q_norm",
    "q_norm_hw",
    "k_norm",
    "k_norm_hw",
)


def _model_bytes(model: torch.nn.Module) -> int:
    seen = set()
    total = 0
    for tensor in (*model.parameters(), *model.buffers()):
        if id(tensor) not in seen:
            seen.add(id(tensor))
            total += tensor.numel() * tensor.element_size()
    return total


def _prune_for_compact_t2i(model: torch.nn.Module) -> int:
    before = _model_bytes(model)
    language_model = model.language_model
    for layer in language_model.model.layers:
        for name in _DENSE_LAYER_MODULES:
            setattr(layer, name, None)
        for name in _DENSE_ATTENTION_MODULES:
            setattr(layer.self_attn, name, None)
    language_model.model.norm = None
    language_model.lm_head = None
    model._sensenova_compact_mode = True
    gc.collect()
    return before - _model_bytes(model)


def load_model_and_tokenizer(
    model_path: str,
    server_args: ServerArgs,
) -> dict[str, Any]:
    dtype = PRECISION_TO_TYPE.get(
        server_args.pipeline_config.model_precision, torch.bfloat16
    )
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if server_args.trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    if server_args.revision is not None:
        model_kwargs["revision"] = server_args.revision

    tokenizer_kwargs: dict[str, Any] = {}
    if server_args.trust_remote_code:
        tokenizer_kwargs["trust_remote_code"] = True
    if server_args.revision is not None:
        tokenizer_kwargs["revision"] = server_args.revision

    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    model = AutoModel.from_pretrained(model_path, **model_kwargs).eval()
    if compact_mode_enabled():
        released = _prune_for_compact_t2i(model)
        logger.info(
            "SenseNova compact T2I removed %.3f GiB of dense main-process weights",
            released / (1 << 30),
        )
    device = get_local_torch_device()
    current_platform.set_device(device)
    model = model.to(device)
    return {"model": model, "tokenizer": tokenizer}
