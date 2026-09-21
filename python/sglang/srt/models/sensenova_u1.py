# SPDX-License-Identifier: Apache-2.0
"""SRT text-only runtime for SenseNova-U1 thinking decode."""

import hashlib
import json
import os
import time
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.models.qwen2 import Qwen2MLP, Qwen2Model
from sglang.srt.models.qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
)
from sglang.srt.runtime_context import get_parallel, get_stream
from sglang.srt.utils import add_prefix, is_cuda

_KV_DIAGNOSTIC_DIR = "SGLANG_SENSENOVA_KV_DIAGNOSTIC_DIR"
_KV_DIAGNOSTIC_RID_PREFIX = "sensenova-kvdiag-"
_KV_TRANSFER_DIR = "SGLANG_SENSENOVA_KV_TRANSFER_DIR"
_KV_TRANSFER_RID_PREFIX = "sensenova-kvxfer-"
_KV_TRANSFER_REUSE_BUFFER = "SGLANG_SENSENOVA_KV_TRANSFER_REUSE_BUFFER"
_KV_TRANSFER_MAX_TOKENS = "SGLANG_SENSENOVA_KV_TRANSFER_MAX_TOKENS"


def _atomic_json_dump(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)


def _export_prefix_kv_impl(
    output_dir: str,
    dump_id: str,
    token_ids: list[int],
    slots: torch.Tensor,
    kv_pool,
) -> None:
    """Stream real paged KV through one pinned layer into a shared-memory file."""
    layer_ids = list(
        range(kv_pool.start_layer, kv_pool.start_layer + len(kv_pool.k_buffer))
    )
    first_keys, first_values = kv_pool.get_kv_buffer(layer_ids[0])
    if first_keys.shape[1:] != first_values.shape[1:]:
        raise RuntimeError("SenseNova KV transfer requires matching K/V shapes")
    length = len(token_ids)
    layer_shape = (first_keys.shape[1], length, first_keys.shape[2])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    total_elements = len(layer_ids) * 2 * layer_shape[0] * length * layer_shape[2]
    reuse_buffer = os.environ.get(_KV_TRANSFER_REUSE_BUFFER) == "1"
    lock_path = None
    if reuse_buffer:
        max_tokens = int(os.environ.get(_KV_TRANSFER_MAX_TOKENS, "4096"))
        if length > max_tokens:
            raise RuntimeError(
                f"SenseNova KV transfer length {length} exceeds {max_tokens}"
            )
        lock_path = output / "buffer.lock"
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            # Batch-1 compact mode uses the reusable buffer. A concurrent request
            # keeps correctness by falling back to its own one-shot file.
            reuse_buffer = False
            lock_path = None
        else:
            with os.fdopen(lock_fd, "w", encoding="utf-8") as handle:
                handle.write(dump_id)
            data_path = output / "buffer.bin"
            buffer_elements = (
                len(layer_ids) * 2 * layer_shape[0] * max_tokens * layer_shape[2]
            )
            buffer_bytes = buffer_elements * first_keys.element_size()
            if not data_path.exists() or data_path.stat().st_size < buffer_bytes:
                with open(data_path, "ab") as handle:
                    handle.truncate(buffer_bytes)
            mapping_path = data_path
    if not reuse_buffer:
        data_path = output / f"{dump_id}.bin"
        mapping_path = output / f".{dump_id}.{os.getpid()}.bin.tmp"
        with open(mapping_path, "wb") as handle:
            handle.truncate(total_elements * first_keys.element_size())
    mapped = torch.from_file(
        str(mapping_path),
        shared=True,
        size=total_elements,
        dtype=first_keys.dtype,
    ).view(len(layer_ids), 2, *layer_shape)
    pin_memory = first_keys.is_cuda
    pinned_keys = torch.empty(
        layer_shape, dtype=first_keys.dtype, pin_memory=pin_memory
    )
    pinned_values = torch.empty(
        layer_shape, dtype=first_values.dtype, pin_memory=pin_memory
    )
    timings = {"gather": 0.0, "d2h": 0.0, "shared_write": 0.0}
    events = None
    if first_keys.is_cuda:
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

    wall_started = time.perf_counter()
    for output_layer, layer_id in enumerate(layer_ids):
        keys, values = kv_pool.get_kv_buffer(layer_id)
        if events is not None:
            events[0].record()
        else:
            stage_started = time.perf_counter()
        gathered_keys = keys.index_select(0, slots).transpose(0, 1)
        gathered_values = values.index_select(0, slots).transpose(0, 1)
        if events is not None:
            events[1].record()
            events[2].record()
        else:
            timings["gather"] += (time.perf_counter() - stage_started) * 1000
            stage_started = time.perf_counter()
        pinned_keys.copy_(gathered_keys, non_blocking=first_keys.is_cuda)
        pinned_values.copy_(gathered_values, non_blocking=first_keys.is_cuda)
        if events is not None:
            events[3].record()
            events[3].synchronize()
            timings["gather"] += events[0].elapsed_time(events[1])
            timings["d2h"] += events[2].elapsed_time(events[3])
        else:
            timings["d2h"] += (time.perf_counter() - stage_started) * 1000

        stage_started = time.perf_counter()
        mapped[output_layer, 0].copy_(pinned_keys)
        mapped[output_layer, 1].copy_(pinned_values)
        timings["shared_write"] += (time.perf_counter() - stage_started) * 1000

    wall_ms = (time.perf_counter() - wall_started) * 1000
    del mapped
    if not reuse_buffer:
        os.replace(mapping_path, data_path)
    metadata = {
        "source": "srt",
        "dump_id": dump_id,
        "token_sha256": hashlib.sha256(
            ",".join(map(str, token_ids)).encode()
        ).hexdigest(),
        "token_count": length,
        "layer_ids": layer_ids,
        "shape": [len(layer_ids), 2, *layer_shape],
        "dtype": str(first_keys.dtype),
        "data_file": data_path.name,
        "data_bytes": total_elements * first_keys.element_size(),
        "reuse_buffer": reuse_buffer,
        "lock_file": lock_path.name if lock_path is not None else None,
        "timings_ms": {**timings, "wall": wall_ms},
    }
    _atomic_json_dump(metadata, output / f"{dump_id}.json")


def _export_prefix_kv(
    output_dir: str,
    dump_id: str,
    token_ids: list[int],
    slots: torch.Tensor,
    kv_pool,
) -> None:
    try:
        _export_prefix_kv_impl(output_dir, dump_id, token_ids, slots, kv_pool)
    except Exception:
        output = Path(output_dir)
        lock_path = output / "buffer.lock"
        try:
            if lock_path.read_text(encoding="utf-8") == dump_id:
                lock_path.unlink()
        except OSError:
            pass
        (output / f".{dump_id}.{os.getpid()}.bin.tmp").unlink(missing_ok=True)
        raise


def _understanding_weights(weights: Iterable[tuple[str, torch.Tensor]]):
    """Yield only the dense understanding branch under Qwen3-compatible names."""
    prefix = "language_model."
    for name, tensor in weights:
        if name.startswith(prefix) and "_mot_gen" not in name:
            yield name[len(prefix) :], tensor


class SenseNovaU1Attention(Qwen3Attention):
    """SenseNova text attention with temporal-half RoPE and split Q/K norms."""

    def __init__(
        self,
        config,
        layer_id: int,
        start_layer: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        rope_parameters = getattr(config, "rope_parameters", None)
        rope_theta = (
            rope_parameters["rope_theta"]
            if rope_parameters and "rope_theta" in rope_parameters
            else getattr(config, "rope_theta", 1000000)
        )
        rope_scaling = (
            rope_parameters
            if rope_parameters and "rope_theta" in rope_parameters
            else getattr(config, "rope_scaling", None)
        )
        super().__init__(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            start_layer=start_layer,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=getattr(config, "head_dim", None),
            max_position_embeddings=getattr(config, "max_position_embeddings", 32768),
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            prefix=prefix,
            alt_stream=alt_stream,
        )
        assert self.head_dim % 2 == 0
        self.half_head_dim = self.head_dim // 2
        self.q_norm = RMSNorm(self.half_head_dim, eps=config.rms_norm_eps)
        self.q_norm_hw = RMSNorm(self.half_head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.half_head_dim, eps=config.rms_norm_eps)
        self.k_norm_hw = RMSNorm(self.half_head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            self.half_head_dim,
            rotary_dim=self.half_head_dim,
            max_position=getattr(config, "max_position_embeddings", 32768),
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.use_fused_qk_norm_mrope = False

    def _split_norm(self, states: torch.Tensor, temporal_norm, spatial_norm):
        token_count = states.shape[0]
        states = states.view(token_count, -1, self.head_dim)
        temporal, spatial = states.split(self.half_head_dim, dim=-1)
        temporal = temporal_norm(temporal.reshape(-1, self.half_head_dim)).view(
            token_count, -1, self.half_head_dim
        )
        spatial = spatial_norm(spatial.reshape(-1, self.half_head_dim)).view(
            token_count, -1, self.half_head_dim
        )
        return temporal, spatial

    def forward_prepare_native(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_t, q_hw = self._split_norm(q, self.q_norm, self.q_norm_hw)
        k_t, k_hw = self._split_norm(k, self.k_norm, self.k_norm_hw)
        q_t, k_t = self.rotary_emb(
            positions,
            q_t.reshape(q_t.shape[0], -1),
            k_t.reshape(k_t.shape[0], -1),
        )
        q = torch.cat(
            (q_t.view(q_hw.shape[0], -1, self.half_head_dim), q_hw), dim=-1
        ).reshape(q.shape[0], -1)
        k = torch.cat(
            (k_t.view(k_hw.shape[0], -1, self.half_head_dim), k_hw), dim=-1
        ).reshape(k.shape[0], -1)
        return q, k, v


class SenseNovaU1DecoderLayer(Qwen3DecoderLayer):
    def __init__(
        self,
        config,
        layer_id: int = 0,
        start_layer: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = SenseNovaU1Attention(
            config,
            layer_id=layer_id,
            start_layer=start_layer,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )
        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=False,
            is_previous_layer_sparse=False,
            is_next_layer_sparse=False,
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )


class SenseNovaU1TextModel(Qwen2Model):
    def __init__(
        self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ) -> None:
        alt_stream = get_stream("alt") if is_cuda() else None
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=SenseNovaU1DecoderLayer,
            alt_stream=alt_stream,
        )


class NEOChatModel(Qwen3ForCausalLM):
    """Expose only SenseNova's understanding branch to the SRT scheduler."""

    def __init__(self, config, quant_config: QuantizationConfig | None = None) -> None:
        nn.Module.__init__(self)
        text_config = config.llm_config
        self.pp_group = get_pp_group()
        self.config = text_config
        self.quant_config = quant_config
        self.model = SenseNovaU1TextModel(
            text_config, quant_config=quant_config, prefix="model"
        )
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and text_config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    text_config.vocab_size,
                    text_config.hidden_size,
                    quant_config=quant_config,
                    use_attn_tp_group=get_parallel().enable_dp_lm_head,
                    prefix="lm_head",
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(text_config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.capture_aux_hidden_states = False

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return super().load_weights(_understanding_weights(weights))

    def prepare_for_kv_cache_release(
        self, req, req_to_token_pool, token_to_kv_pool_allocator
    ) -> None:
        """Export finalized prefix KV for explicitly marked diagnostic requests."""
        if not isinstance(req.rid, str):
            return
        diagnostic = req.rid.startswith(_KV_DIAGNOSTIC_RID_PREFIX)
        transfer = req.rid.startswith(_KV_TRANSFER_RID_PREFIX)
        if not diagnostic and not transfer:
            return
        output_dir = os.environ.get(
            _KV_DIAGNOSTIC_DIR if diagnostic else _KV_TRANSFER_DIR
        )
        if not output_dir:
            return
        prefix = _KV_DIAGNOSTIC_RID_PREFIX if diagnostic else _KV_TRANSFER_RID_PREFIX
        dump_id = req.rid.removeprefix(prefix)
        if not dump_id or not dump_id.isascii() or not dump_id.isalnum():
            raise ValueError(f"invalid SenseNova KV export id: {dump_id!r}")
        length = len(req.origin_input_ids)
        if not req.kv.holds_kv or length <= 0 or req.kv.kv_committed_len < length:
            raise RuntimeError("SenseNova KV export request has no committed KV")

        kv_pool = token_to_kv_pool_allocator.get_kvcache()
        if getattr(kv_pool, "kv_cache_layout", None) != "nhd":
            raise RuntimeError("SenseNova KV export requires an NHD KV cache")
        if getattr(kv_pool, "is_quantized_kv_cache", False):
            raise RuntimeError("SenseNova KV export requires an unquantized cache")

        slots = req_to_token_pool.req_to_token[req.kv.req_pool_idx, :length]
        token_ids = [int(token_id) for token_id in req.origin_input_ids]
        if transfer:
            _export_prefix_kv(output_dir, dump_id, token_ids, slots, kv_pool)
            return

        layer_id = kv_pool.start_layer
        keys, values = kv_pool.get_kv_buffer(layer_id)
        token_sha256 = hashlib.sha256(
            ",".join(map(str, token_ids)).encode()
        ).hexdigest()
        payload = {
            "source": "srt",
            "layer_id": layer_id,
            "token_ids": token_ids,
            "token_sha256": token_sha256,
            "keys": keys[slots].transpose(0, 1).contiguous().cpu(),
            "values": values[slots].transpose(0, 1).contiguous().cpu(),
        }
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path / f"{dump_id}-srt.pt")


EntryClass = NEOChatModel
