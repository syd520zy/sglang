# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import time
from collections import deque
from types import SimpleNamespace

import pytest
import requests
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from sglang.multimodal_gen.configs.pipeline_configs.sensenova_u1 import (
    SenseNovaU1PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.configs.sample.sensenova_u1 import (
    SenseNovaU1SamplingParams,
)
from sglang.multimodal_gen.configs.sensenova_u1 import (
    DEFAULT_MAX_THINK_TOKENS,
    SENSENOVA_U1_REQUEST_EXTRA_KEY,
)
from sglang.multimodal_gen.registry import (
    _get_config_info,
    get_model_info,
    get_non_diffusers_pipeline_name,
    is_registered_diffusion_model_path,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.image_api import (
    _image_request_model_kwargs,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
    ImageUsage,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    process_generation_batch,
)
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.models.sensenova_u1 import srt_thinking
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.configuration_neo_chat import (
    NEOLLMConfig,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.configuration_neo_vit import (
    NEOVisionConfig,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.conversation import (
    get_conv_template,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_neo_chat import (
    NEOChatModel,
    _copy_right_aligned_prefix_bnsd,
    _preallocate_think_cache,
    _randn_with_seed,
    prepare_flash_kv_cache,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    _flash_or_sdpa,
    _sdpa_attn_func,
    create_block_causal_mask,
    make_qwen3_rms_norm,
    npu_fia_available,
    position_ids_from_indexes,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.srt_thinking import (
    ManagedSRTThinkingServer,
    SRTThinkingClient,
    ThinkingBackendStatus,
    normalize_thinking_output_ids,
    prepare_managed_srt_thinking,
    runtime_files,
    thinking_backend_info,
    thinking_strict_enabled,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1 import (
    SenseNovaU1GenerationStage,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.perf_logger import MemorySnapshot
from sglang.multimodal_gen.test.scripts.compare_sensenova_kv_diagnostic import (
    error_metrics,
    numerically_compatible,
)
from sglang.multimodal_gen.test.scripts.inspect_sensenova_srt_runtime import (
    inspect_runtime_log,
)
from sglang.multimodal_gen.test.scripts.profile_sensenova_thinking_concurrency import (
    summarize_wave,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models.sensenova_u1 import (
    _KV_TRANSFER_RID_PREFIX,
)
from sglang.srt.models.sensenova_u1 import NEOChatModel as SRTNEOChatModel
from sglang.srt.models.sensenova_u1 import (
    _understanding_weights,
)


class _FakeSenseNovaModel:
    def __init__(self):
        self.call_kwargs = None
        self.call_kwargs_list = []

    def t2i_generate(self, tokenizer, prompt, **kwargs):
        self.call_kwargs = {"tokenizer": tokenizer, "prompt": prompt, **kwargs}
        self.call_kwargs_list.append(self.call_kwargs)
        if kwargs["profile_stages"]:
            self.last_profile_timings_ms = {
                "input_prepare": 1.0,
                "condition_prefill": 2.0,
                "think_decode": 3.0 if kwargs["think_mode"] else 0.0,
                "think_replay_prefill": 0.0,
                "cfg_prefill": 4.0,
                "denoise_prepare": 5.0,
                "denoise_loop": 6.0,
                "total": 18.0 if kwargs["think_mode"] else 15.0,
            }
        sample = torch.tensor(
            [
                [
                    [[-1.0, 0.0], [0.5, 1.0]],
                    [[-1.0, 0.0], [0.5, 1.0]],
                    [[-1.0, 0.0], [0.5, 1.0]],
                ]
            ]
        )
        image = sample.repeat(kwargs["batch_size"], 1, 1, 1)
        if kwargs["think_mode"]:
            backend = "srt" if kwargs.get("thinking_backend") is not None else "native"
            self.last_think_token_count = 3
            self.last_think_token_counts = [3] * kwargs["batch_size"]
            self.last_thinking_backend = backend
            self.last_thinking_backends = [backend] * kwargs["batch_size"]
            think_text = ["draft</think>"] * kwargs["batch_size"]
            return image, think_text[0] if kwargs["batch_size"] == 1 else think_text
        return image


def _think_logits(token_id):
    logits = torch.full((1, 1, 32), float("-inf"))
    logits[0, 0, token_id] = 0
    return logits


class _FakeThinkLanguageModel:
    def __init__(self, next_tokens):
        self.model = SimpleNamespace(current_index=None)
        self.next_tokens = list(next_tokens)
        self.calls = []

    def __call__(self, *, past_key_values, **_kwargs):
        self.calls.append(_kwargs)
        next_token = self.next_tokens.pop(0) if self.next_tokens else 0
        return SimpleNamespace(
            logits=_think_logits(next_token),
            past_key_values=past_key_values,
        )


class _FakeThinkTokenizer:
    _token_ids = {"</s>": 8, "</think>": 9}

    def convert_tokens_to_ids(self, token):
        return self._token_ids[token]

    def __call__(self, text, **_kwargs):
        assert text == "\n\n<img>"
        return {"input_ids": torch.tensor([[20, 21]])}

    def decode(self, token_ids, **_kwargs):
        pieces = {7: "draft", 9: "</think>"}
        return "".join(pieces[token_id] for token_id in token_ids)


class _FakeTokenizer:
    pad_token_id = None
    eos_token_id = 2

    def __call__(self, text, return_tensors):
        del return_tensors
        token_count = len(text.split()) + 1
        return {"input_ids": torch.arange(1, token_count + 1).unsqueeze(0)}


class _RecordingTraceContext:
    tracing_enable = True

    def __init__(self):
        self.finish_count = 0
        self.started_slices = []
        self.finished_slices = []

    def trace_req_finish(self):
        self.finish_count += 1

    def trace_slice_start(self, name, level=0):
        self.started_slices.append((name, level))

    def trace_slice_end(self, name, level=0, **kwargs):
        self.finished_slices.append((name, level))


class _SequentialTestExecutor(PipelineExecutor):
    def __init__(self, server_args, *, fail=False, fail_request_ids=None):
        super().__init__(server_args)
        self.fail = fail
        self.fail_request_ids = set(fail_request_ids or [])
        self.executed_requests = []

    def execute_group(self, stages, batches, server_args):
        for batch in batches:
            batch.metrics.record_stage("InputValidationStage", 0.125)
            batch.metrics.record_memory_snapshot(
                "after_validation",
                MemorySnapshot(
                    allocated_mb=100.0,
                    reserved_mb=200.0,
                    peak_allocated_mb=300.0,
                    peak_reserved_mb=400.0,
                ),
            )
        return batches

    def execute(self, stages, batch, server_args):
        self.executed_requests.append(batch)
        if self.fail or batch.request_id in self.fail_request_ids:
            raise RuntimeError(f"generation failed for {batch.request_id}")
        return OutputBatch(
            output_file_paths=[batch.output_file_name],
            metrics=batch.metrics,
        )


class _SequentialTestPipeline:
    def __init__(self, server_args, *, fail=False, fail_request_ids=None):
        self.input_stage = InputValidationStage()
        self.executor = _SequentialTestExecutor(
            server_args,
            fail=fail,
            fail_request_ids=fail_request_ids,
        )

    def forward_batch_sequentially(self, batches, server_args):
        return self.executor.execute_group_sequentially(
            [self.input_stage, object()], batches, server_args
        )


class _WorkerBackedSchedulerClient:
    def __init__(self, worker):
        self.worker = worker

    async def forward(self, batches):
        return next(self.worker.execute_forward_sequentially(batches))


@pytest.mark.parametrize(
    ("template_name", "expected_system_message"),
    [
        (
            "Hermes-2",
            "\u4f60\u662f\u7531\u4e0a\u6d77\u4eba\u5de5\u667a\u80fd\u5b9e\u9a8c\u5ba4\u8054\u5408\u5546\u6c64\u79d1\u6280\u5f00\u53d1\u7684\u4e66\u751f\u591a\u6a21\u6001\u5927\u6a21\u578b\uff0c\u82f1\u6587\u540d\u53ebInternVL, \u662f\u4e00\u4e2a\u6709\u7528\u65e0\u5bb3\u7684\u4eba\u5de5\u667a\u80fd\u52a9\u624b\u3002",
        ),
        (
            "internlm2-chat",
            "\u4f60\u662f\u7531\u4e0a\u6d77\u4eba\u5de5\u667a\u80fd\u5b9e\u9a8c\u5ba4\u8054\u5408\u5546\u6c64\u79d1\u6280\u5f00\u53d1\u7684\u4e66\u751f\u591a\u6a21\u6001\u5927\u6a21\u578b\uff0c\u82f1\u6587\u540d\u53ebInternVL, \u662f\u4e00\u4e2a\u6709\u7528\u65e0\u5bb3\u7684\u4eba\u5de5\u667a\u80fd\u52a9\u624b\u3002",
        ),
        (
            "phi3-chat",
            "\u4f60\u662f\u7531\u4e0a\u6d77\u4eba\u5de5\u667a\u80fd\u5b9e\u9a8c\u5ba4\u8054\u5408\u5546\u6c64\u79d1\u6280\u5f00\u53d1\u7684\u4e66\u751f\u591a\u6a21\u6001\u5927\u6a21\u578b\uff0c\u82f1\u6587\u540d\u53ebInternVL, \u662f\u4e00\u4e2a\u6709\u7528\u65e0\u5bb3\u7684\u4eba\u5de5\u667a\u80fd\u52a9\u624b\u3002",
        ),
        (
            "internvl2_5",
            "\u4f60\u662f\u4e66\u751f\xb7\u4e07\u8c61\uff0c\u82f1\u6587\u540d\u662fInternVL\uff0c\u662f\u7531\u4e0a\u6d77\u4eba\u5de5\u667a\u80fd\u5b9e\u9a8c\u5ba4\u3001\u6e05\u534e\u5927\u5b66\u53ca\u591a\u5bb6\u5408\u4f5c\u5355\u4f4d\u8054\u5408\u5f00\u53d1\u7684\u591a\u6a21\u6001\u5927\u8bed\u8a00\u6a21\u578b\u3002",
        ),
    ],
)
def test_sensenova_u1_conversation_preserves_upstream_system_prompt(
    template_name, expected_system_message
):
    assert get_conv_template(template_name).system_message == expected_system_message


def _force_generator_fallback(monkeypatch, device_type):
    original_generator = torch.Generator

    def unsupported_device_generator(device="cpu"):
        if torch.device(device).type == device_type:
            raise RuntimeError(f"Generator is unsupported on {device_type}")
        return original_generator(device)

    monkeypatch.setattr(torch, "Generator", unsupported_device_generator)


def test_sensenova_u1_randn_fallback_preserves_cpu_rng(monkeypatch):
    _force_generator_fallback(monkeypatch, "cpu")
    rng_state = torch.get_rng_state().clone()

    first = _randn_with_seed((2, 3), device="cpu", dtype=torch.float32, seed=17)
    second = _randn_with_seed((2, 3), device="cpu", dtype=torch.float32, seed=17)

    assert torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_sensenova_u1_randn_supports_per_sample_seeds():
    actual = _randn_with_seed(
        (2, 3, 4), device=torch.device("cpu"), dtype=torch.float32, seed=[7, 19]
    )
    expected = torch.cat(
        [
            _randn_with_seed(
                (1, 3, 4),
                device=torch.device("cpu"),
                dtype=torch.float32,
                seed=seed,
            )
            for seed in (7, 19)
        ]
    )

    assert torch.equal(actual, expected)


def test_sensenova_u1_builds_padded_batched_text_inputs():
    model = SimpleNamespace(device=torch.device("cpu"))

    input_ids, indexes, attention_mask, valid_mask, prefix_lengths = (
        NEOChatModel._build_t2i_text_inputs(
            model, _FakeTokenizer(), ["short", "a much longer prompt"]
        )
    )

    assert input_ids.shape == (2, 5)
    assert indexes.shape == (2, 3, 5)
    assert prefix_lengths.tolist() == [2, 5]
    assert valid_mask.tolist() == [
        [True, True, False, False, False],
        [True, True, True, True, True],
    ]
    mask = attention_mask["full_attention"]
    assert mask.shape == (2, 1, 5, 5)
    assert torch.isneginf(mask[0, :, :, 2:]).all()
    assert torch.isfinite(mask[0, :, :, :2]).any()


def test_sensenova_u1_position_indexes_support_batched_inputs():
    indexes = torch.tensor(
        [
            [[0, 1], [0, 0], [0, 0]],
            [[4, 4], [0, 1], [0, 0]],
        ]
    )

    assert torch.equal(position_ids_from_indexes(indexes, 0), indexes[:, 0])
    assert torch.equal(
        position_ids_from_indexes(indexes[0], 1), indexes[0, 1].unsqueeze(0)
    )


def test_sensenova_u1_singleton_text_matches_valid_batched_tokens():
    model = SimpleNamespace(device=torch.device("cpu"))
    tokenizer = _FakeTokenizer()
    batched = NEOChatModel._build_t2i_text_inputs(
        model, tokenizer, ["short", "a much longer prompt"]
    )
    for i, prompt in enumerate(["short", "a much longer prompt"]):
        single = NEOChatModel._build_t2i_text_inputs(model, tokenizer, prompt)
        length = single[0].shape[1]
        assert torch.equal(batched[0][i, :length], single[0][0])
        assert torch.equal(batched[1][i, :, :length], single[1])
        assert torch.equal(
            batched[2]["full_attention"][i, :, :length, :length],
            single[2]["full_attention"][0],
        )


def test_sensenova_u1_block_causal_mask_rejects_padded_keys():
    indexes = torch.tensor([[0, 1, 2], [0, 1, 2]])
    valid = torch.tensor([[True, True, False], [True, True, True]])

    mask = create_block_causal_mask(indexes, valid)

    assert mask.shape == (2, 1, 3, 3)
    assert torch.isneginf(mask[0, :, :, 2]).all()
    assert mask[1, 0, 2, 2] == 0


def test_sensenova_u1_builds_per_sample_image_indexes():
    indexes = NEOChatModel._build_t2i_image_indexes(
        SimpleNamespace(),
        token_h=2,
        token_w=2,
        text_len=torch.tensor([2, 5]),
        device=torch.device("cpu"),
    )

    assert indexes.shape == (2, 3, 4)
    assert indexes[:, 0].tolist() == [[2, 2, 2, 2], [5, 5, 5, 5]]
    assert indexes[:, 1].tolist() == [[0, 0, 1, 1], [0, 0, 1, 1]]
    assert indexes[:, 2].tolist() == [[0, 1, 0, 1], [0, 1, 0, 1]]


def test_sensenova_u1_compacts_variable_length_kv_before_attention():
    generator = torch.Generator().manual_seed(29)
    q = torch.randn(2, 3, 4, 8, generator=generator)
    k = torch.randn(2, 8, 2, 8, generator=generator)
    v = torch.randn(2, 8, 2, 8, generator=generator)
    actual = _flash_or_sdpa(
        q,
        k,
        v,
        actual_seq_lengths_kv=[5, 8],
    )

    expected_short = _sdpa_attn_func(
        q[:1],
        torch.cat((k[:1, :2], k[:1, 5:]), dim=1),
        torch.cat((v[:1, :2], v[:1, 5:]), dim=1),
    )
    expected_long = _sdpa_attn_func(q[1:], k[1:], v[1:])
    torch.testing.assert_close(actual, torch.cat((expected_short, expected_long)))


def test_sensenova_u1_sdpa_masks_padded_prefix_keys():
    q = torch.tensor([[[[1.0, 0.0]]]])
    k = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]])
    v = torch.tensor([[[[2.0, 0.0]], [[0.0, 4.0]], [[100.0, 100.0]]]])
    key_mask = torch.tensor([[[[True, True, False]]]])

    actual = _sdpa_attn_func(q, k, v, attention_mask=key_mask)
    expected = _sdpa_attn_func(q, k[:, :2], v[:, :2])

    torch.testing.assert_close(actual, expected)


def test_sensenova_u1_right_aligns_bnsd_prefix_for_npu_fia():
    source = torch.tensor(
        [
            [[[1], [2], [99], [99], [99]]],
            [[[3], [4], [5], [6], [7]]],
        ]
    )
    destination = torch.zeros(2, 1, 8, 1, dtype=source.dtype)

    _copy_right_aligned_prefix_bnsd(destination, source, [2, 5])

    assert destination[:, 0, :5, 0].tolist() == [
        [0, 0, 0, 1, 2],
        [3, 4, 5, 6, 7],
    ]
    assert destination[:, :, 5:].eq(0).all()


@pytest.mark.parametrize("available", [False, True])
def test_sensenova_u1_npu_fia_checks_operator_availability(monkeypatch, available):
    namespace = SimpleNamespace()
    if available:
        namespace.npu_fused_infer_attention_score = object()
    monkeypatch.setattr(torch.ops, "npu", namespace, raising=False)

    assert npu_fia_available() is available


@pytest.mark.parametrize(
    ("is_npu", "uses_native"),
    [(False, True), (True, False)],
)
def test_sensenova_u1_shared_rmsnorm_dispatch(monkeypatch, is_npu, uses_native):
    monkeypatch.setattr(current_platform, "is_npu", lambda: is_npu)

    norm = make_qwen3_rms_norm(64, eps=1e-6)

    assert isinstance(norm, RMSNorm)
    assert norm.cast_x_before_out_mul
    assert (norm._forward_method == norm.forward_native) is uses_native


@torch.no_grad()
def test_sensenova_u1_fused_dense_mlp_matches_original(monkeypatch):
    config = SimpleNamespace(
        hidden_size=16,
        intermediate_size=24,
        hidden_act="silu",
    )
    with torch.random.fork_rng():
        torch.manual_seed(37)
        mlp = Qwen3MLP(config).eval()
        hidden_states = torch.randn(2, 5, config.hidden_size)
        expected = mlp(hidden_states)

    monkeypatch.setattr(mlp, "_use_npu_fused_mlp", lambda _x: True)
    monkeypatch.setattr(
        torch.ops,
        "npu",
        SimpleNamespace(
            npu_swiglu=lambda x, dim=-1: (
                F.silu(x.chunk(2, dim=dim)[0]) * x.chunk(2, dim=dim)[1]
            )
        ),
        raising=False,
    )
    actual = mlp(hidden_states)

    torch.testing.assert_close(actual, expected)
    assert set(mlp.state_dict()) == {
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    }
    assert (
        mlp.gate_proj.weight.untyped_storage().data_ptr()
        == mlp.up_proj.weight.untyped_storage().data_ptr()
    )


def test_sensenova_u1_batched_gqa_matches_unpadded_singletons():
    generator = torch.Generator().manual_seed(17)
    q = torch.randn(2, 3, 4, 8, generator=generator)
    k = torch.randn(2, 8, 2, 8, generator=generator)
    v = torch.randn(2, 8, 2, 8, generator=generator)
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[0, 2:5] = False
    attention_mask = valid[:, None, None, :].expand(-1, -1, q.shape[1], -1)

    actual = _sdpa_attn_func(q, k, v, attention_mask=attention_mask)

    for i in range(2):
        expected = _sdpa_attn_func(
            q[i : i + 1], k[i : i + 1, valid[i]], v[i : i + 1, valid[i]]
        )
        torch.testing.assert_close(actual[i : i + 1], expected)


@torch.no_grad()
def test_sensenova_u1_prefix_and_denoise_attention_match_singletons():
    config = NEOLLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng():
        torch.manual_seed(23)
        attention = Qwen3Attention(config, layer_idx=0).eval()
    generator = torch.Generator().manual_seed(31)
    text = torch.randn(2, 5, 64, generator=generator)
    image = torch.randn(2, 3, 64, generator=generator)
    helper = SimpleNamespace(device=torch.device("cpu"))

    def run(prefix, lengths, image_states):
        batch_size, width, _ = prefix.shape
        positions = torch.arange(width).expand(batch_size, -1)
        indexes = torch.stack(
            [positions, torch.zeros_like(positions), torch.zeros_like(positions)], dim=1
        )
        valid = positions < torch.tensor(lengths)[:, None]
        cache = DynamicCache(config=config)
        attention.forward_und(
            prefix, indexes, create_block_causal_mask(positions, valid), cache
        )
        prefix_keys = cache.layers[0].keys.clone()
        prepare_flash_kv_cache(
            cache,
            current_len=3,
            batch_size=batch_size,
            prefix_lengths=torch.tensor(lengths),
        )
        image_indexes = NEOChatModel._build_t2i_image_indexes(
            helper, 1, 3, torch.tensor(lengths), torch.device("cpu")
        )
        outputs = []
        for _ in range(2):
            image_states, _ = attention.forward_gen(
                image_states,
                image_indexes,
                None,
                cache,
                update_cache=False,
            )
            outputs.append(image_states)
        torch.testing.assert_close(cache.layers[0].keys, prefix_keys)
        return prefix_keys, outputs

    keys, batched = run(text, [2, 5], image)
    for i, length in enumerate([2, 5]):
        single_keys, single = run(text[i : i + 1, :length], [length], image[i : i + 1])
        torch.testing.assert_close(keys[i : i + 1, :, :length], single_keys)
        for step in range(2):
            torch.testing.assert_close(
                batched[step][i : i + 1], single[step], atol=1e-5, rtol=1e-4
            )


def test_sensenova_u1_randn_fallback_preserves_device_rng(monkeypatch):
    device_type = current_platform.device_type
    if not device_type or device_type == "cpu":
        pytest.skip("No accelerator is available")

    device = torch.device(device_type, 0)
    device_module = torch.get_device_module(device)
    if not device_module.is_available():
        pytest.skip(f"{device_type} is not available")

    _force_generator_fallback(monkeypatch, device_type)
    cpu_rng_state = torch.get_rng_state().clone()
    device_rng_state = device_module.get_rng_state(device).clone()

    first = _randn_with_seed((2, 3), device=device, dtype=torch.float32, seed=17)
    second = _randn_with_seed((2, 3), device=device, dtype=torch.float32, seed=17)

    assert torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), cpu_rng_state)
    assert torch.equal(device_module.get_rng_state(device), device_rng_state)


def test_sensenova_u1_registry_resolves_local_and_hf_paths(tmp_path):
    _get_config_info.cache_clear()
    get_model_info.cache_clear()

    local_path = tmp_path / "checkpoint-revision-abc123"
    local_path.mkdir()
    (local_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["NEOChatModel"],
                "model_type": "neo_chat",
            }
        )
    )

    assert is_registered_diffusion_model_path(str(local_path))
    assert get_non_diffusers_pipeline_name(str(local_path)) == "SenseNovaU1Pipeline"

    local_model_info = get_model_info(str(local_path))
    assert local_model_info is not None
    assert local_model_info.pipeline_config_cls is SenseNovaU1PipelineConfig
    assert local_model_info.sampling_param_cls is SenseNovaU1SamplingParams

    model_info = get_model_info("sensenova/SenseNova-U1.5-8B-MoT")
    assert model_info is not None
    assert model_info.pipeline_config_cls is SenseNovaU1PipelineConfig
    assert model_info.sampling_param_cls is SenseNovaU1SamplingParams

    modelscope_id = "SenseNova/SenseNova-U1.5-8B-MoT"
    assert is_registered_diffusion_model_path(modelscope_id)
    assert get_non_diffusers_pipeline_name(modelscope_id) == "SenseNovaU1Pipeline"
    assert get_model_info(modelscope_id) is not None
    get_model_info.cache_clear()


def test_sensenova_u1_registry_requires_exact_hub_id(monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.registry.maybe_download_model_index",
        lambda _: {},
    )
    _get_config_info.cache_clear()
    get_model_info.cache_clear()

    unrelated_repo = "acme/SenseNova-U1.5-8B-MoT"
    assert not is_registered_diffusion_model_path(unrelated_repo)
    assert get_non_diffusers_pipeline_name(unrelated_repo) is None
    assert _get_config_info(unrelated_repo) is None

    _get_config_info.cache_clear()
    get_model_info.cache_clear()


def test_sensenova_u1_registry_does_not_route_lora_only_repositories(tmp_path):
    lora_repo = "sensenova/SenseNova-U1.5-8B-MoT-LoRA"
    lora_path = tmp_path / "SenseNova-U1.5-8B-MoT-LoRA"
    lora_path.mkdir()
    (lora_path / "adapter_config.json").write_text("{}")

    assert get_non_diffusers_pipeline_name(lora_repo) is None
    assert get_non_diffusers_pipeline_name(str(lora_path)) is None
    assert not is_registered_diffusion_model_path(lora_repo)
    assert not is_registered_diffusion_model_path(str(lora_path))


@pytest.mark.parametrize("backend", ["auto", "sglang", "diffusers"])
def test_sensenova_u1_known_adapter_only_repo_rejected_before_backend_resolution(
    monkeypatch, backend
):
    def fail_model_index_download(_):
        raise AssertionError("adapter-only repo should not download model_index")

    def fail_diffusers_resolution(**_kwargs):
        raise AssertionError("adapter-only repo should not resolve diffusers info")

    monkeypatch.setattr(
        "sglang.multimodal_gen.registry.maybe_download_model_index",
        fail_model_index_download,
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.registry._get_diffusers_model_info",
        fail_diffusers_resolution,
    )
    get_model_info.cache_clear()

    loras_repo = "sensenova/SenseNova-U1.5-8B-MoT-LoRAs"

    assert get_non_diffusers_pipeline_name(loras_repo) is None
    assert get_model_info(loras_repo, backend=backend) is None
    get_model_info.cache_clear()


def test_sensenova_u1_sampling_params_keep_private_defaults_internal():
    params = SenseNovaU1SamplingParams(prompt="hello", width=2304, height=4096)

    assert params.guidance_scale == 4.0
    assert params.num_inference_steps == 50
    assert params.num_outputs_per_prompt == 1
    assert params.cfg_norm == "none"
    assert params.timestep_shift == 3.0

    extra = params.build_request_extra()[SENSENOVA_U1_REQUEST_EXTRA_KEY]
    assert extra == {
        "cfg_norm": "none",
        "timestep_shift": 3.0,
        "enable_timestep_shift": True,
        "cfg_interval": (0.0, 1.0),
        "t_eps": 0.02,
        "think_mode": False,
        "max_think_tokens": DEFAULT_MAX_THINK_TOKENS,
        "profile_stages": False,
    }


def test_sensenova_u1_rejects_unaligned_resolution():
    with pytest.raises(ValueError, match="divisible by 32"):
        SenseNovaU1SamplingParams(width=2160, height=3840)


def test_sensenova_u1_accepts_openai_image_api_num_frames():
    params = SenseNovaU1SamplingParams(
        prompt="hello",
        width=2048,
        height=2048,
        num_frames=1,
    )

    assert params.num_frames == 1
    assert params.data_type == DataType.IMAGE


def test_sensenova_u1_scheduler_capabilities():
    config = SenseNovaU1PipelineConfig()

    assert config.supports_dynamic_batching()
    assert config.supports_sequential_multi_output_inference()


def _make_sensenova_u1_scheduler_request(
    request_id: str, prompt: str, seed: int | list[int], **sampling_overrides
) -> Req:
    sampling = SenseNovaU1SamplingParams(
        prompt=prompt,
        seed=seed,
        **sampling_overrides,
    )
    return Req(
        request_id=request_id,
        prompt=prompt,
        seed=seed,
        sampling_params=sampling,
        extra=sampling.build_request_extra(),
    )


def test_sensenova_u1_batch_cost_tracks_resolution_steps_and_cfg():
    config = SenseNovaU1PipelineConfig()
    batch = SimpleNamespace(
        width=1024,
        height=1024,
        num_inference_steps=5,
        guidance_scale=4.0,
        num_outputs_per_prompt=1,
    )

    assert config.estimate_request_cost(batch) == 32 * 32 * 5 * 2


def test_sensenova_u1_multi_output_request_is_not_dynamically_batched():
    scheduler = object.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(pipeline_config=SenseNovaU1PipelineConfig())
    sampling = SenseNovaU1SamplingParams(
        prompt="a mountain lake", num_outputs_per_prompt=2
    )
    request = SimpleNamespace(
        is_warmup=False,
        realtime_session_id=None,
        session=None,
        prompt=sampling.prompt,
        image_path=None,
        return_file_paths_only=False,
        num_outputs_per_prompt=2,
        sampling_params=sampling,
    )

    assert not scheduler._can_dynamic_batch(request, request)
    assert (
        scheduler._get_dynamic_batch_reject_reason(request, request)
        == "sequential_multi_output"
    )


def test_sensenova_u1_think_mode_batching_requires_srt():
    scheduler = object.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(pipeline_config=SenseNovaU1PipelineConfig())
    scheduler._batch_admission = SimpleNamespace(enabled=True)
    scheduler._batch_metrics_enabled = False
    request = _make_sensenova_u1_scheduler_request(
        "request-0", "a mountain lake", 7, think_mode=True
    )
    scheduler.waiting_queue = deque([(b"identity", request, time.monotonic())])

    assert not scheduler._can_dynamic_batch(request, request)
    assert (
        scheduler._get_dynamic_batch_reject_reason(request, request)
        == "pipeline_request_unsupported"
    )
    items = scheduler.get_next_batch_to_run()
    assert items is not None
    assert items[0][0] == b"identity"
    assert items[0][1] is request
    assert not scheduler.waiting_queue

    scheduler.server_args.pipeline_config.srt_thinking_dynamic_batching = True
    assert scheduler._can_dynamic_batch(request, request)


def test_sensenova_u1_external_srt_enables_thinking_batching(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_SENSENOVA_THINKING_RUNTIME_DIR", str(tmp_path))
    config = SenseNovaU1PipelineConfig()
    url = "http://127.0.0.1:31000"
    ThinkingBackendStatus.for_url(url).mark_unavailable("previous run failed")
    server_args = SimpleNamespace(
        pipeline_config=config,
        srt_encoder_url=url,
    )

    assert prepare_managed_srt_thinking(server_args) is None
    assert config.srt_thinking_dynamic_batching
    assert ThinkingBackendStatus.for_url(url).state.value == "starting"


@pytest.mark.parametrize(
    "sampling_overrides",
    [
        {"guidance_scale": 1.0},
        {"num_inference_steps": 25},
        {"cfg_norm": "global"},
        {"timestep_shift": 2.0},
        {"t_eps": 0.01},
    ],
)
def test_sensenova_u1_scheduler_rejects_heterogeneous_generation_options(
    sampling_overrides,
):
    scheduler = object.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(pipeline_config=SenseNovaU1PipelineConfig())
    base = _make_sensenova_u1_scheduler_request("request-0", "first", 7)
    candidate = _make_sensenova_u1_scheduler_request(
        "request-1", "second", 19, **sampling_overrides
    )

    assert not scheduler._can_dynamic_batch(base, candidate)


def test_sensenova_u1_scheduler_normalizes_single_output_seed_lists():
    scheduler = object.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(pipeline_config=SenseNovaU1PipelineConfig())
    requests = [
        _make_sensenova_u1_scheduler_request("request-0", "first", [7]),
        _make_sensenova_u1_scheduler_request("request-1", "second", 19),
    ]

    merged = scheduler._try_merge_generation_reqs(requests)

    assert merged.extra["dynamic_batch_seeds"] == [7, 19]


def test_sensenova_u1_scheduler_merge_and_split_preserve_request_order():
    scheduler = object.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(pipeline_config=SenseNovaU1PipelineConfig())
    requests = [
        _make_sensenova_u1_scheduler_request(
            "request-0",
            "short",
            7,
            output_path="/tmp/first",
            output_file_name="first.png",
        ),
        _make_sensenova_u1_scheduler_request(
            "request-1",
            "a longer prompt",
            19,
            output_path="/tmp/second",
            output_file_name="second.png",
        ),
    ]
    merged = scheduler._try_merge_generation_reqs(requests)
    assert merged.prompt == ["short", "a longer prompt"]
    assert merged.extra["dynamic_batch_seeds"] == [7, 19]
    expected_paths = [request.output_file_path() for request in requests]
    assert merged.extra["dynamic_batch_output_paths"] == expected_paths
    assert requests[0].prompt == "short"
    outputs = scheduler._split_batched_output(
        OutputBatch(
            output=[torch.tensor([7]), torch.tensor([19])],
            output_file_paths=expected_paths,
            usage_list=[
                {"think_text": "first", "reasoning_tokens": 7},
                {"think_text": "second", "reasoning_tokens": 19},
            ],
        ),
        requests,
    )
    assert [output.output[0].item() for output in outputs] == [7, 19]
    assert [output.output_file_paths for output in outputs] == [
        [path] for path in expected_paths
    ]
    assert [output.usage for output in outputs] == [
        {"think_text": "first", "reasoning_tokens": 7},
        {"think_text": "second", "reasoning_tokens": 19},
    ]
    assert outputs[0].usage is not outputs[1].usage
    assert (
        scheduler._split_batched_output(
            OutputBatch(output=[torch.tensor([7])]), requests
        )
        is None
    )
    requests[1].sampling_params.width = 1024
    del requests[1]._dynamic_batch_sig
    assert scheduler._try_merge_generation_reqs(requests) is None


def test_sensenova_thinking_concurrency_summary_does_not_count_queueing_as_overlap():
    def record(request_index, completed_at, total_ms):
        timings = {
            stage: 0.0
            for stage in (
                "input_prepare",
                "condition_prefill",
                "think_decode",
                "think_replay_prefill",
                "cfg_prefill",
                "denoise_prepare",
                "denoise_loop",
                "total",
            )
        }
        timings["total"] = total_ms
        return {
            "request_index": request_index,
            "start_offset_ms": 0.0,
            "end_offset_ms": completed_at,
            "client_elapsed_ms": completed_at,
            "stage_timings_ms": timings,
        }

    serial = summarize_wave([record(0, 1000.0, 1000.0), record(1, 2000.0, 1000.0)])
    parallel = summarize_wave([record(0, 1000.0, 1000.0), record(1, 1000.0, 1000.0)])

    assert serial["parallelism_ratio"] == 1.0
    assert not serial.get("think_window_overlap_pairs")
    assert parallel["parallelism_ratio"] == 2.0


@pytest.mark.parametrize(
    "kv_log",
    [
        "Post-capture KV sizing: KV cache allocated.",
        "KV Cache is allocated.",
    ],
)
def test_sensenova_srt_runtime_inspection_checks_graph_and_kv_capacity(kv_log):
    report = inspect_runtime_log(
        f"""
Capture target decode CUDA graph begin. backend=piecewise, num_tokens_per_req=1, bs=[1, 2], avail mem=10.00 GB
Capture target decode CUDA graph end. elapsed=1.00 s, mem usage=1.00 GB, avail mem=9.00 GB.
{kv_log} dtype: torch.bfloat16, #tokens: 12288, KV size: 1.00 GB, avail mem=8.00 GB
""",
        context_length=4096,
        max_concurrency=2,
        cuda_graph_max_bs=2,
    )

    assert report["passed"]
    assert report["cuda_graph"]["captured_batch_sizes"] == [1, 2]
    assert report["kv_pool"]["required_tokens"] == 8192


def test_sensenova_u1_rejects_multi_gpu_during_arg_validation():
    config = SenseNovaU1PipelineConfig()

    with pytest.raises(ValueError, match="num_gpus=1"):
        config.validate_server_args(
            SimpleNamespace(
                num_gpus=2,
                enable_torch_compile=False,
                lora_path=None,
                attention_backend=None,
                component_attention_backends={},
            )
        )


def test_sensenova_u1_clears_auto_tuned_runtime_defaults():
    config = SenseNovaU1PipelineConfig()
    args = SimpleNamespace(
        num_gpus=1,
        enable_torch_compile=False,
        lora_path=None,
        component_residency={"transformer": "layerwise-offload"},
        cpu_offload_components=["transformer"],
        dit_cpu_offload=True,
        text_encoder_cpu_offload=True,
        image_encoder_cpu_offload=True,
        vae_cpu_offload=True,
        dit_layerwise_offload=True,
        layerwise_offload_components=["transformer"],
        quantization=None,
        quantization_ignored_layers=None,
        transformer_weights_path=None,
        component_paths={"model": "/tmp/component"},
        component_weights_paths={"model": "/tmp/model.safetensors"},
        component_quantizations={},
        component_quantization_ignored_layers={},
        component_precisions={},
        attention_backend="aiter",
        component_attention_backends={"text_encoder": "torch_sdpa"},
        attention_backend_config={"foo": "bar"},
        is_arg_explicitly_set=lambda _name: False,
    )

    config.validate_server_args(args)

    assert args.component_residency is None
    assert args.cpu_offload_components is None
    assert args.dit_cpu_offload is False
    assert args.text_encoder_cpu_offload is False
    assert args.image_encoder_cpu_offload is False
    assert args.vae_cpu_offload is False
    assert args.dit_layerwise_offload is False
    assert args.layerwise_offload_components is None
    assert args.component_paths == {}
    assert args.component_weights_paths == {}
    assert args.attention_backend is None
    assert args.component_attention_backends == {}
    assert args.attention_backend_config is None


def test_sensenova_u1_allows_explicit_resident_component_residency():
    config = SenseNovaU1PipelineConfig()
    args = SimpleNamespace(
        num_gpus=1,
        enable_torch_compile=False,
        lora_path=None,
        component_residency={"transformer": "resident"},
        cpu_offload_components=None,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        image_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        dit_layerwise_offload=False,
        layerwise_offload_components=None,
        quantization=None,
        quantization_ignored_layers=None,
        transformer_weights_path=None,
        component_paths={},
        component_weights_paths={},
        component_quantizations={},
        component_quantization_ignored_layers={},
        component_precisions={},
        attention_backend=None,
        component_attention_backends={},
        attention_backend_config={},
        is_arg_explicitly_set=lambda name: name == "component_residency",
    )

    config.validate_server_args(args)

    assert args.component_residency == {"transformer": "resident"}


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"enable_torch_compile": True}, "torch.compile"),
        ({"lora_path": "sensenova/SenseNova-U1.5-8B-MoT-LoRAs"}, "LoRA adapters"),
        (
            {"component_residency": {"transformer": "component-offload"}},
            "component residency offload",
        ),
        ({"cpu_offload_components": ["transformer"]}, "CPU offload"),
        ({"dit_cpu_offload": True}, "DiT CPU offload"),
        ({"text_encoder_cpu_offload": True}, "text encoder CPU offload"),
        ({"image_encoder_cpu_offload": True}, "image encoder CPU offload"),
        ({"vae_cpu_offload": True}, "VAE CPU offload"),
        ({"dit_layerwise_offload": True}, "DiT layerwise offload"),
        ({"layerwise_offload_components": ["transformer"]}, "layerwise offload"),
        ({"quantization": "fp8"}, "quantization"),
        ({"quantization_ignored_layers": ["foo"]}, "quantization ignored layers"),
        (
            {"transformer_weights_path": "/tmp/transformer.safetensors"},
            "pre-quantized transformer weights",
        ),
        ({"component_paths": {"model": "/tmp/component"}}, "component path overrides"),
        (
            {"component_weights_paths": {"model": "/tmp/model.safetensors"}},
            "component weight path overrides",
        ),
        ({"component_quantizations": {"transformer": "fp8"}}, "component quantization"),
        (
            {"component_quantization_ignored_layers": {"transformer": ["foo"]}},
            "component quantization ignored layers",
        ),
        ({"component_precisions": {"transformer": "fp16"}}, "component precision"),
        ({"attention_backend": "fa"}, "custom attention backends"),
        (
            {"component_attention_backends": {"text_encoder": "torch_sdpa"}},
            "component attention backends",
        ),
        ({"attention_backend_config": {"foo": "bar"}}, "attention backend config"),
    ],
)
def test_sensenova_u1_rejects_unsupported_runtime_modes(override, expected):
    config = SenseNovaU1PipelineConfig()
    args = {
        "num_gpus": 1,
        "enable_torch_compile": False,
        "lora_path": None,
        "component_residency": None,
        "cpu_offload_components": None,
        "dit_cpu_offload": None,
        "text_encoder_cpu_offload": None,
        "image_encoder_cpu_offload": None,
        "vae_cpu_offload": False,
        "dit_layerwise_offload": None,
        "layerwise_offload_components": None,
        "quantization": None,
        "quantization_ignored_layers": None,
        "transformer_weights_path": None,
        "component_paths": {},
        "component_weights_paths": {},
        "component_quantizations": {},
        "component_quantization_ignored_layers": {},
        "component_precisions": {},
        "attention_backend": None,
        "component_attention_backends": {},
        "attention_backend_config": {},
    }
    args.update(override)

    with pytest.raises(ValueError, match=expected):
        config.validate_server_args(SimpleNamespace(**args))


def test_sensenova_u1_rejects_direct_server_args_quantization():
    config = SenseNovaU1PipelineConfig()

    with pytest.raises(ValueError, match="quantization"):
        ServerArgs(
            model_path="sensenova/SenseNova-U1.5-8B-MoT",
            pipeline_config=config,
            quantization="fp8",
        )


def test_sensenova_u1_rejects_file_valued_component_paths(tmp_path):
    config = SenseNovaU1PipelineConfig()

    with pytest.raises(ValueError, match="component weight path overrides"):
        ServerArgs(
            model_path="sensenova/SenseNova-U1.5-8B-MoT",
            pipeline_config=config,
            component_paths={"model": str(tmp_path / "model.safetensors")},
        )


def test_sensenova_u1_vision_config_round_trips_sequence_fields(tmp_path):
    config = NEOVisionConfig(llm_hidden_size=2048, downsample_ratio=0.5)
    config.save_pretrained(tmp_path)

    loaded = NEOVisionConfig.from_pretrained(tmp_path)

    assert loaded.llm_hidden_size == (2048,)
    assert loaded.downsample_ratio == (0.5,)


def test_sensenova_u1_vision_config_normalizes_nested_singletons():
    config = NEOVisionConfig(llm_hidden_size=[[2048]], downsample_ratio=[[0.5]])

    assert config.llm_hidden_size == (2048,)
    assert config.downsample_ratio == (0.5,)


def test_sensenova_u1_rejects_video_frame_count():
    with pytest.raises(ValueError, match="num_frames=1"):
        SenseNovaU1SamplingParams(width=2048, height=2048, num_frames=2)


def test_sensenova_u1_cli_args_expose_only_sglang_compatible_fields():
    args = SimpleNamespace(
        prompt="hello",
        width=2304,
        height=4096,
        guidance_scale=4.5,
        num_inference_steps=30,
        num_outputs_per_prompt=2,
        cfg_norm="global",
        timestep_shift=9.0,
        think_mode=True,
        max_think_tokens=128,
        profile_stages=True,
    )

    cli_args = SenseNovaU1SamplingParams.get_cli_args(args)

    assert cli_args["prompt"] == "hello"
    assert cli_args["width"] == 2304
    assert cli_args["height"] == 4096
    assert cli_args["guidance_scale"] == 4.5
    assert cli_args["num_inference_steps"] == 30
    assert cli_args["num_outputs_per_prompt"] == 2
    assert "cfg_norm" not in cli_args
    assert "timestep_shift" not in cli_args
    assert cli_args["think_mode"] is True
    assert cli_args["max_think_tokens"] == 128
    assert cli_args["profile_stages"] is True


def test_sensenova_u1_thinking_fields_remain_model_specific():
    request = ImageGenerationsRequest(
        prompt="a mountain lake",
        extra_body={
            "think_mode": True,
            "max_think_tokens": 128,
            "profile_stages": True,
        },
    )

    assert _image_request_model_kwargs(request, SenseNovaU1SamplingParams) == {
        "think_mode": True,
        "max_think_tokens": 128,
        "profile_stages": True,
    }


@pytest.mark.parametrize("max_think_tokens", [0, 1025])
def test_sensenova_u1_rejects_invalid_think_token_budget(max_think_tokens):
    with pytest.raises(ValueError, match="max_think_tokens"):
        SenseNovaU1SamplingParams(max_think_tokens=max_think_tokens)


@pytest.mark.parametrize("max_think_tokens", [True, 1.5])
def test_sensenova_u1_rejects_non_integer_think_token_budget(max_think_tokens):
    with pytest.raises(TypeError, match="max_think_tokens"):
        SenseNovaU1SamplingParams(max_think_tokens=max_think_tokens)


def test_sensenova_u1_rejects_non_boolean_think_mode():
    with pytest.raises(TypeError, match="think_mode"):
        SenseNovaU1SamplingParams(think_mode="true")


def test_sensenova_u1_rejects_non_boolean_stage_profiling():
    with pytest.raises(TypeError, match="profile_stages"):
        SenseNovaU1SamplingParams(profile_stages="true")


def test_sensenova_u1_generation_stage_uses_sglang_params_and_single_model_batch():
    sampling = SenseNovaU1SamplingParams(
        prompt="a mountain lake",
        width=2304,
        height=4096,
        guidance_scale=4.5,
        num_inference_steps=30,
        seed=123,
    )
    batch = SimpleNamespace(
        prompt=sampling.prompt,
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=sampling.seed,
        num_outputs_per_prompt=sampling.num_outputs_per_prompt,
        extra=sampling.build_request_extra(),
        metrics=None,
    )
    model = _FakeSenseNovaModel()
    stage = SenseNovaU1GenerationStage(model=model, tokenizer="tok")

    output = stage.forward(batch, server_args=SimpleNamespace())

    assert len(output.output) == 1
    assert torch.allclose(
        output.output[0],
        torch.tensor(
            [
                [[0.0, 0.5], [0.75, 1.0]],
                [[0.0, 0.5], [0.75, 1.0]],
                [[0.0, 0.5], [0.75, 1.0]],
            ]
        ),
    )
    assert model.call_kwargs["tokenizer"] == "tok"
    assert model.call_kwargs["prompt"] == "a mountain lake"
    assert model.call_kwargs["image_size"] == (2304, 4096)
    assert model.call_kwargs["cfg_scale"] == 4.5
    assert model.call_kwargs["num_steps"] == 30
    assert model.call_kwargs["batch_size"] == 1
    assert model.call_kwargs["seed"] == 123
    assert model.call_kwargs["think_mode"] is False
    assert model.call_kwargs["max_think_tokens"] == DEFAULT_MAX_THINK_TOKENS
    assert model.call_kwargs["profile_stages"] is False
    assert output.usage is None


def test_sensenova_u1_generation_stage_returns_profile_timings():
    sampling = SenseNovaU1SamplingParams(
        prompt="a mountain lake",
        width=1024,
        height=1024,
        profile_stages=True,
    )
    batch = SimpleNamespace(
        prompt=sampling.prompt,
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=sampling.seed,
        num_outputs_per_prompt=1,
        extra=sampling.build_request_extra(),
        metrics=None,
    )
    model = _FakeSenseNovaModel()

    output = SenseNovaU1GenerationStage(model=model, tokenizer="tok").forward(
        batch, server_args=SimpleNamespace()
    )

    assert model.call_kwargs["profile_stages"] is True
    assert output.usage == {"stage_timings_ms": model.last_profile_timings_ms}
    assert (
        ImageUsage.model_validate(output.usage).stage_timings_ms["denoise_loop"] == 6.0
    )


def test_sensenova_u1_generation_stage_returns_thinking_usage():
    sampling = SenseNovaU1SamplingParams(
        prompt="a mountain lake",
        width=2048,
        height=2048,
        think_mode=True,
        max_think_tokens=128,
    )
    batch = SimpleNamespace(
        prompt=sampling.prompt,
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=sampling.seed,
        num_outputs_per_prompt=1,
        extra=sampling.build_request_extra(),
        metrics=None,
    )
    model = _FakeSenseNovaModel()
    thinking_backend = object()

    output = SenseNovaU1GenerationStage(
        model=model, tokenizer="tok", thinking_backend=thinking_backend
    ).forward(batch, server_args=SimpleNamespace())

    assert model.call_kwargs["think_mode"] is True
    assert model.call_kwargs["max_think_tokens"] == 128
    assert model.call_kwargs["thinking_backend"] is thinking_backend
    assert output.usage == {
        "think_text": "draft</think>",
        "reasoning_tokens": 3,
        "thinking_backend": "srt",
    }
    usage = ImageUsage.model_validate(output.usage)
    assert usage.think_text == "draft</think>"
    assert usage.thinking_backend == "srt"


def test_sensenova_u1_generation_stage_reports_all_transferred_prefixes():
    sampling = SenseNovaU1SamplingParams(
        prompt="a mountain lake",
        width=1024,
        height=1024,
        think_mode=False,
    )
    batch = SimpleNamespace(
        prompt=sampling.prompt,
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=sampling.seed,
        num_outputs_per_prompt=1,
        extra=sampling.build_request_extra(),
        metrics=None,
    )
    model = _FakeSenseNovaModel()
    model.last_srt_kv_transferred_prefixes = ["condition", "uncondition"]
    model.last_srt_kv_transfer_timings = {
        "session_reused_tokens": 0,
        "srt_cached_tokens": 0,
        "import_wall": 1.0,
    }

    output = SenseNovaU1GenerationStage(
        model=model, tokenizer="tok", thinking_backend=object()
    ).forward(batch, server_args=SimpleNamespace())

    assert output.usage == {
        "srt_kv_transfer_used": True,
        "srt_kv_transferred_prefixes": ["condition", "uncondition"],
        "srt_kv_transfer_timings_ms": {"import_wall": 1.0},
        "srt_kv_session_reused_tokens": 0,
        "srt_kv_cached_tokens": 0,
    }
    assert ImageUsage.model_validate(output.usage).srt_kv_transferred_prefixes == [
        "condition",
        "uncondition",
    ]


@pytest.mark.parametrize(
    ("first_token", "next_tokens", "max_think_tokens", "expected_suffix"),
    [
        (7, [9, 0], 4, [20, 21]),
        (8, [], 4, [9, 20, 21]),
        (7, [6], 2, [9, 20, 21]),
    ],
)
def test_sensenova_u1_thinking_always_closes_within_budget(
    monkeypatch,
    first_token,
    next_tokens,
    max_think_tokens,
    expected_suffix,
):
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify."
        "modeling_neo_chat.get_conv_template",
        lambda _template: SimpleNamespace(sep="</s>"),
    )
    appended_ids = []
    model = SimpleNamespace(
        template="test",
        language_model=_FakeThinkLanguageModel(next_tokens),
        device=torch.device("cpu"),
        last_think_token_count=0,
    )

    def append_text_tokens(_cache, index, token_ids):
        appended_ids.append(token_ids[0].tolist())
        return index + token_ids.shape[1]

    model._append_text_tokens_to_cache = append_text_tokens

    _, _, think_text = NEOChatModel._generate_think(
        model,
        _FakeThinkTokenizer(),
        SimpleNamespace(logits=_think_logits(first_token)),
        past_key_values=object(),
        t_idx=5,
        IMG_START_TOKEN="<img>",
        max_think_tokens=max_think_tokens,
    )

    assert think_text.endswith("</think>")
    assert model.last_think_token_count <= max_think_tokens
    assert appended_ids == [expected_suffix]
    assert all(call["use_cache"] for call in model.language_model.calls)


def test_sensenova_think_cache_keeps_only_written_tokens_visible():
    cache = DynamicCache()
    initial_keys = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4)
    initial_values = initial_keys + 100
    cache.update(initial_keys, initial_values, 0)
    _preallocate_think_cache(cache, additional_tokens=3)
    layer = cache.layers[0]
    key_buffer_ptr = layer._key_buffer.data_ptr()

    assert cache.get_seq_length() == 2
    for token in range(3):
        keys = torch.full((1, 2, 1, 4), float(token + 20))
        values = keys + 100
        cache.update(keys, values, 0)
        assert cache.get_seq_length() == 3 + token
        assert layer._key_buffer.data_ptr() == key_buffer_ptr
        assert layer.keys.data_ptr() == key_buffer_ptr
        torch.testing.assert_close(layer.keys[..., -1:, :], keys)
        torch.testing.assert_close(layer.values[..., -1:, :], values)

    torch.testing.assert_close(layer.keys[..., :2, :], initial_keys)
    torch.testing.assert_close(layer.values[..., :2, :], initial_values)


@pytest.mark.parametrize(
    ("output_ids", "expected"),
    [
        ([7, 9, 6], [7, 9]),
        ([7, 8, 6], [7, 9]),
        ([7, 6, 5], [7, 6, 9]),
    ],
)
def test_sensenova_srt_thinking_output_closes_within_budget(output_ids, expected):
    assert (
        normalize_thinking_output_ids(
            output_ids,
            max_think_tokens=3,
            eos_token_id=8,
            think_end_token_id=9,
        )
        == expected
    )


def test_sensenova_srt_thinking_client_uses_token_api(monkeypatch):
    request = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"output_ids": [7, 9]}

    def post(url, **kwargs):
        request.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr("requests.post", post)
    output_ids = SRTThinkingClient("http://127.0.0.1:1234", 3, 90).generate_batch(
        [[1, 2]], max_think_tokens=4, eos_token_id=8, think_end_token_id=9
    )[0]

    assert output_ids == [7, 9]
    assert request == {
        "url": "http://127.0.0.1:1234/generate",
        "json": {
            "input_ids": [1, 2],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 3,
                "stop_token_ids": [8, 9],
                "no_stop_trim": True,
                "skip_special_tokens": False,
            },
        },
        "timeout": (3, 90),
    }


def test_sensenova_srt_thinking_client_batches_token_requests(monkeypatch):
    request = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"output_ids": [7, 9]}, {"output_ids": [6, 5]}]

    def post(url, **kwargs):
        request.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr("requests.post", post)
    output_ids = SRTThinkingClient("http://127.0.0.1:1234", 3, 90).generate_batch(
        [[1, 2], [3]],
        max_think_tokens=3,
        eos_token_id=8,
        think_end_token_id=9,
    )

    assert output_ids == [[7, 9], [6, 5, 9]]
    assert request["json"]["input_ids"] == [[1, 2], [3]]


@pytest.mark.parametrize(
    ("raw_output", "expected_output", "expected_offset", "expected_suffix"),
    [
        ([7, 9], [7, 9], 4, [20]),
        ([7, 8], [7, 9], 3, [9, 20]),
        ([7, 6], [7, 6, 9], 4, [9, 20]),
    ],
)
def test_sensenova_srt_thinking_client_reuses_session_for_kv_transfer(
    monkeypatch,
    tmp_path,
    raw_output,
    expected_output,
    expected_offset,
    expected_suffix,
):
    requests_seen = []

    class Response:
        def __init__(self, payload=None):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    def post(url, **kwargs):
        requests_seen.append((url, kwargs["json"]))
        if url.endswith("/open_session"):
            return Response("session-1")
        if url.endswith("/close_session"):
            return Response()
        if kwargs["json"]["rid"].startswith("sensenova-think-"):
            return Response({"output_ids": raw_output, "meta_info": {}})

        dump_id = kwargs["json"]["rid"].removeprefix("sensenova-kvxfer-")
        (tmp_path / f"{dump_id}.json").write_text(
            json.dumps({"dump_id": dump_id}), encoding="utf-8"
        )
        return Response(
            {
                "output_ids": [3],
                "meta_info": {"id": kwargs["json"]["rid"], "cached_tokens": 3},
            }
        )

    monkeypatch.setenv("SGLANG_SENSENOVA_KV_TRANSFER_DIR", str(tmp_path))
    monkeypatch.setattr("requests.post", post)
    client = SRTThinkingClient("http://127.0.0.1:1234", 3, 90)

    output_ids, context = client.generate_batch_for_kv_transfer(
        [[1, 2]], max_think_tokens=3, eos_token_id=8, think_end_token_id=9
    )
    final_prefix = [1, 2, *expected_output, 20]
    metadata = client.transfer_prefix_kv(final_prefix, "abc123", context)

    assert output_ids == [expected_output]
    continuation = requests_seen[2][1]
    assert continuation["input_ids"] == expected_suffix
    assert continuation["session_params"]["id"] == "session-1"
    assert continuation["session_params"]["rid"].startswith("sensenova-think-")
    assert continuation["session_params"]["offset"] == expected_offset
    assert requests_seen[3] == (
        "http://127.0.0.1:1234/close_session",
        {"session_id": "session-1"},
    )
    assert metadata["session_reused_tokens"] == expected_offset
    assert metadata["request_meta_info"]["cached_tokens"] == 3


def test_sensenova_srt_client_requests_diagnostic_prefix(monkeypatch, tmp_path):
    request = {}

    class Response:
        def raise_for_status(self):
            pass

    def post(url, **kwargs):
        request.update(url=url, **kwargs)
        return Response()

    monkeypatch.setenv("SGLANG_SENSENOVA_KV_DIAGNOSTIC_DIR", str(tmp_path))
    monkeypatch.setattr("requests.post", post)
    SRTThinkingClient("http://127.0.0.1:1234", 3, 90).dump_prefix_kv(
        [1, 2, 3], "abc123"
    )

    assert request == {
        "url": "http://127.0.0.1:1234/generate",
        "json": {
            "rid": "sensenova-kvdiag-abc123",
            "input_ids": [1, 2, 3],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
                "skip_special_tokens": False,
            },
        },
        "timeout": (3, 90),
    }


def test_sensenova_srt_worker_dumps_committed_nhd_kv(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_SENSENOVA_KV_DIAGNOSTIC_DIR", str(tmp_path))
    keys = torch.arange(60, dtype=torch.bfloat16).reshape(10, 2, 3)
    values = keys + 100
    pool = SimpleNamespace(
        kv_cache_layout="nhd",
        is_quantized_kv_cache=False,
        start_layer=0,
        get_kv_buffer=lambda _layer_id: (keys, values),
    )
    req = SimpleNamespace(
        rid="sensenova-kvdiag-abc123",
        origin_input_ids=[11, 12],
        output_ids=[13],
        kv=SimpleNamespace(holds_kv=True, kv_committed_len=3, req_pool_idx=0),
    )
    req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor([[3, 5, 0]], dtype=torch.long)
    )
    allocator = SimpleNamespace(get_kvcache=lambda: pool)

    SRTNEOChatModel.prepare_for_kv_cache_release(
        None, req, req_to_token_pool, allocator
    )

    payload = torch.load(tmp_path / "abc123-srt.pt", weights_only=True)
    assert payload["token_ids"] == [11, 12]
    torch.testing.assert_close(payload["keys"], keys[[3, 5]].transpose(0, 1))
    torch.testing.assert_close(payload["values"], values[[3, 5]].transpose(0, 1))


def test_sensenova_srt_worker_streams_all_kv_layers(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_SENSENOVA_KV_TRANSFER_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_SENSENOVA_KV_TRANSFER_REUSE_BUFFER", "1")
    monkeypatch.setenv("SGLANG_SENSENOVA_KV_TRANSFER_MAX_TOKENS", "8")
    keys = [
        torch.arange(60, dtype=torch.bfloat16).reshape(10, 2, 3) + layer * 100
        for layer in range(2)
    ]
    values = [key + 50 for key in keys]
    pool = SimpleNamespace(
        kv_cache_layout="nhd",
        is_quantized_kv_cache=False,
        start_layer=0,
        k_buffer=keys,
        get_kv_buffer=lambda layer_id: (keys[layer_id], values[layer_id]),
    )
    req = SimpleNamespace(
        rid=f"{_KV_TRANSFER_RID_PREFIX}abc123",
        origin_input_ids=[11, 12],
        kv=SimpleNamespace(holds_kv=True, kv_committed_len=2, req_pool_idx=0),
    )
    req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor([[3, 5]], dtype=torch.long)
    )

    SRTNEOChatModel.prepare_for_kv_cache_release(
        None,
        req,
        req_to_token_pool,
        SimpleNamespace(get_kvcache=lambda: pool),
    )

    metadata = json.loads((tmp_path / "abc123.json").read_text())
    assert metadata["shape"] == [2, 2, 2, 2, 3]
    assert metadata["data_file"] == "buffer.bin"
    assert metadata["reuse_buffer"] is True
    assert (tmp_path / "buffer.lock").read_text() == "abc123"
    assert (tmp_path / "buffer.bin").stat().st_size == 384
    exported = torch.from_file(
        str(tmp_path / metadata["data_file"]),
        shared=False,
        size=48,
        dtype=torch.bfloat16,
    ).view(tuple(metadata["shape"]))
    for layer in range(2):
        torch.testing.assert_close(
            exported[layer, 0], keys[layer][[3, 5]].transpose(0, 1)
        )
        torch.testing.assert_close(
            exported[layer, 1], values[layer][[3, 5]].transpose(0, 1)
        )


def test_sensenova_kv_error_metrics_report_outliers():
    metrics = error_metrics(
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.04, 1.01]),
        atol=0.03,
        rtol=0.0,
    )

    assert metrics["out_of_tolerance_count"] == 1
    assert metrics["out_of_tolerance_pct"] == 50.0
    assert metrics["allclose"] is False
    assert numerically_compatible(
        metrics,
        max_abs_error=0.05,
        max_mean_abs_error=0.03,
        max_out_of_tolerance_pct=50.0,
    )
    assert not numerically_compatible(
        metrics,
        max_abs_error=0.03,
        max_mean_abs_error=0.03,
        max_out_of_tolerance_pct=50.0,
    )


def test_sensenova_srt_replay_pads_after_each_complete_prefix():
    captured = {}

    def prefix_forward(input_ids, indexes, attention_mask):
        captured.update(
            input_ids=input_ids,
            indexes=indexes,
            attention_mask=attention_mask,
        )
        return object(), torch.zeros((*input_ids.shape, 4))

    model = SimpleNamespace(
        device=torch.device("cpu"),
        _t2i_prefix_forward=prefix_forward,
    )

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.tensor([[20, 21]])}

    tokenizer = Tokenizer()

    _, _, indexes, key_valid_mask, lengths = NEOChatModel._replay_srt_think_prefix(
        model,
        tokenizer,
        torch.tensor([[1, 2, 3], [4, 5, 0]]),
        torch.tensor([3, 2]),
        [[7, 9], [6, 9]],
        "<img>",
    )

    assert captured["input_ids"].tolist() == [
        [1, 2, 3, 7, 9, 20, 21],
        [4, 5, 6, 9, 20, 21, 0],
    ]
    assert indexes.shape == (2, 3, 7)
    assert key_valid_mask.tolist() == [
        [True, True, True, True, True, True, True],
        [True, True, True, True, True, True, False],
    ]
    assert lengths.tolist() == [7, 6]


def test_sensenova_srt_replay_builds_cache_from_transfer(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_SENSENOVA_USE_SRT_KV_TRANSFER", "1")
    monkeypatch.setenv("SGLANG_SENSENOVA_KV_TRANSFER_DIR", str(tmp_path))
    config = NEOLLMConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
    )
    exported = torch.arange(96, dtype=torch.bfloat16).reshape(1, 2, 2, 6, 4)

    class Backend:
        def transfer_prefix_kv(self, token_ids, dump_id, session_context):
            assert session_context == {"session_id": "session-1"}
            data_path = tmp_path / "buffer.bin"
            with open(data_path, "wb") as handle:
                handle.truncate(exported.numel() * exported.element_size())
            mapped = torch.from_file(
                str(data_path),
                shared=True,
                size=exported.numel(),
                dtype=exported.dtype,
            ).view(exported.shape)
            mapped.copy_(exported)
            del mapped
            (tmp_path / "buffer.lock").write_text(dump_id)
            metadata = {
                "dump_id": dump_id,
                "token_sha256": hashlib.sha256(
                    ",".join(map(str, token_ids)).encode()
                ).hexdigest(),
                "layer_ids": [0],
                "shape": list(exported.shape),
                "dtype": "torch.bfloat16",
                "data_file": "buffer.bin",
                "data_bytes": exported.numel() * exported.element_size(),
                "reuse_buffer": True,
                "lock_file": "buffer.lock",
                "timings_ms": {"wall": 1.0},
            }
            (tmp_path / f"{dump_id}.json").write_text(json.dumps(metadata))
            return metadata

    def prefix_forward(*_args):
        raise AssertionError("native replay must be skipped")

    model = SimpleNamespace(
        device=torch.device("cpu"),
        language_model=SimpleNamespace(config=config),
        _t2i_prefix_forward=prefix_forward,
        last_srt_kv_transferred_prefixes=[],
    )

    def import_prefix(backend, token_ids, context):
        return NEOChatModel._import_srt_prefix_tokens(
            model, backend, token_ids, context
        )

    model._import_srt_prefix_tokens = import_prefix

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.tensor([[20, 21]])}

    cache, hidden, *_ = NEOChatModel._replay_srt_think_prefix(
        model,
        Tokenizer(),
        torch.tensor([[1, 2]]),
        torch.tensor([2]),
        [[7, 9]],
        "<img>",
        Backend(),
        {"session_id": "session-1"},
    )

    assert hidden is None
    assert model.last_srt_kv_transfer_used is True
    torch.testing.assert_close(cache.layers[0].keys, exported[:, 0])
    torch.testing.assert_close(cache.layers[0].values, exported[:, 1])
    assert not (tmp_path / "buffer.lock").exists()


def test_sensenova_srt_imports_finalized_text_prefix_without_session(monkeypatch):
    captured = {}
    expected_cache = object()
    expected_timings = {"import_wall": 1.0}

    class Backend:
        def transfer_prefix_kv(self, token_ids, dump_id, session_context):
            captured.update(
                token_ids=token_ids,
                dump_id=dump_id,
                session_context=session_context,
            )
            return {"dump_id": dump_id}

    def load_prefix(metadata, token_ids, config, device):
        captured.update(
            metadata=metadata,
            load_token_ids=token_ids,
            config=config,
            device=device,
        )
        return expected_cache, expected_timings

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify."
        "modeling_neo_chat._load_srt_prefix_kv",
        load_prefix,
    )
    config = object()
    model = SimpleNamespace(
        language_model=SimpleNamespace(config=config),
        device=torch.device("cpu"),
    )

    cache, timings = NEOChatModel._import_srt_prefix_tokens(model, Backend(), [1, 2, 3])

    assert cache is expected_cache
    assert timings is expected_timings
    assert timings["request_wall"] >= 0
    assert timings["total_wall"] >= timings["request_wall"]
    assert captured["token_ids"] == [1, 2, 3]
    assert captured["load_token_ids"] == [1, 2, 3]
    assert captured["session_context"] is None
    assert captured["metadata"] == {"dump_id": captured["dump_id"]}
    assert captured["config"] is config
    assert captured["device"] == torch.device("cpu")


def test_sensenova_srt_text_prefix_transfer_uses_valid_tokens_only():
    captured = {}
    expected_cache = object()
    expected_timings = {"import_wall": 1.0}
    model = SimpleNamespace()

    def import_prefix(_backend, token_ids):
        captured["token_ids"] = token_ids
        return expected_cache, expected_timings

    model._import_srt_prefix_tokens = import_prefix
    cache, timings = NEOChatModel._try_import_srt_text_prefix(
        model,
        object(),
        torch.tensor([[1, 2, 3, 0]]),
        torch.tensor(3),
        "condition",
    )

    assert cache is expected_cache
    assert timings is expected_timings
    assert captured["token_ids"] == [1, 2, 3]


def test_sensenova_srt_text_prefix_transfer_failure_allows_native_fallback():
    model = SimpleNamespace(
        _import_srt_prefix_tokens=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("transfer failed")
        )
    )

    cache, timings = NEOChatModel._try_import_srt_text_prefix(
        model,
        object(),
        torch.tensor([[1, 2, 3]]),
        torch.tensor(3),
        "CFG",
    )

    assert cache is None
    assert timings == {}


def test_sensenova_srt_kv_transfer_failure_replays_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_SENSENOVA_USE_SRT_KV_TRANSFER", "1")
    monkeypatch.setenv("SGLANG_SENSENOVA_KV_TRANSFER_DIR", str(tmp_path))
    expected_cache = object()

    class Backend:
        def transfer_prefix_kv(self, _token_ids, _dump_id, _session_context):
            raise RuntimeError("transfer failed")

    model = SimpleNamespace(
        device=torch.device("cpu"),
        language_model=SimpleNamespace(config=NEOLLMConfig()),
        _t2i_prefix_forward=lambda *_args: (
            expected_cache,
            torch.zeros((1, 6, 4)),
        ),
    )

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.tensor([[20, 21]])}

    cache, hidden, *_ = NEOChatModel._replay_srt_think_prefix(
        model,
        Tokenizer(),
        torch.tensor([[1, 2]]),
        torch.tensor([2]),
        [[7, 9]],
        "<img>",
        Backend(),
        {"session_id": "session-1"},
    )

    assert cache is expected_cache
    assert hidden.shape == (1, 6, 4)
    assert model.last_srt_kv_transfer_used is False


def test_sensenova_srt_replay_dumps_matching_native_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_SENSENOVA_KV_DIAGNOSTIC_DIR", str(tmp_path))
    captured = {}
    cache = SimpleNamespace(
        layers=[
            SimpleNamespace(
                keys=torch.arange(36, dtype=torch.bfloat16).reshape(1, 2, 6, 3),
                values=torch.arange(36, dtype=torch.bfloat16).reshape(1, 2, 6, 3) + 100,
            )
        ]
    )

    def prefix_forward(input_ids, _indexes, _attention_mask):
        return cache, torch.zeros((*input_ids.shape, 4))

    class Backend:
        def dump_prefix_kv(self, input_ids, dump_id):
            captured.update(input_ids=input_ids, dump_id=dump_id)

    model = SimpleNamespace(
        device=torch.device("cpu"),
        _t2i_prefix_forward=prefix_forward,
    )

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.tensor([[20, 21]])}

    tokenizer = Tokenizer()

    NEOChatModel._replay_srt_think_prefix(
        model,
        tokenizer,
        torch.tensor([[1, 2]]),
        torch.tensor([2]),
        [[7, 9]],
        "<img>",
        Backend(),
    )

    assert captured["input_ids"] == [1, 2, 7, 9, 20, 21]
    payload = torch.load(
        tmp_path / f"{captured['dump_id']}-native.pt", weights_only=True
    )
    assert payload["token_ids"] == captured["input_ids"]
    assert payload["keys"].shape == (2, 6, 3)


class _DeadSRTProcess:
    """Stands in for an SRT subprocess that exited before it served."""

    pid = 4242
    exitcode = 1

    def start(self):
        pass

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass


class _UnstartableSRTProcess(_DeadSRTProcess):
    def start(self):
        raise OSError("spawn failed")


class _RecordingHandler(logging.Handler):
    """Collects the backend logger's own records, whatever the root config is."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def _capture_thinking_log():
    handler = _RecordingHandler()
    srt_thinking.logger.addHandler(handler)
    try:
        yield handler
    finally:
        srt_thinking.logger.removeHandler(handler)


def _thinking_status(tmp_path, *, strict, url="http://127.0.0.1:1234"):
    return ThinkingBackendStatus(
        url,
        strict=strict,
        status_file=str(tmp_path / "srt-thinking-test.json"),
        log_file=str(tmp_path / "srt-thinking-test.log"),
    )


def _refuse_connection(*args, **kwargs):
    raise requests.ConnectionError("connection refused")


def test_sensenova_thinking_status_records_failure_once(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.get", _refuse_connection)
    status = _thinking_status(tmp_path, strict=False)
    with _capture_thinking_log() as handler:
        assert not ManagedSRTThinkingServer(
            process=_DeadSRTProcess(), url="http://127.0.0.1:1234", status=status
        ).start()
        # The client reports the same failure on every later request, but the
        # operator only needs to read it once.
        assert status.mark_unavailable("ConnectionError: refused") is False

    state = json.loads((tmp_path / "srt-thinking-test.json").read_text())
    assert state["state"] == "fallback"
    assert state["backend"] == "native"
    assert "exited with code 1" in state["reason"]
    failure_messages = [
        message
        for message in handler.messages
        if "thinking backend unavailable" in message
    ]
    assert len(failure_messages) == 1
    assert "exited with code 1" in failure_messages[0]


def test_sensenova_thinking_strict_mode_stops_instead_of_falling_back(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("requests.get", _refuse_connection)
    status = _thinking_status(tmp_path, strict=True)

    with pytest.raises(RuntimeError, match="requires SRT"):
        ManagedSRTThinkingServer(
            process=_DeadSRTProcess(), url="http://127.0.0.1:1234", status=status
        ).start()

    state = json.loads((tmp_path / "srt-thinking-test.json").read_text())
    assert state["state"] == "failed"
    assert state["backend"] == "native"
    assert state["strict"] is True


def test_sensenova_thinking_records_a_process_spawn_failure(tmp_path):
    status = _thinking_status(tmp_path, strict=False)

    assert not ManagedSRTThinkingServer(
        process=_UnstartableSRTProcess(),
        url="http://127.0.0.1:1234",
        status=status,
    ).start()

    state = json.loads((tmp_path / "srt-thinking-test.json").read_text())
    assert state["state"] == "fallback"
    assert "OSError: spawn failed" in state["reason"]


def test_sensenova_thinking_client_never_retries_a_dead_backend(tmp_path, monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr("requests.post", post)
    status = _thinking_status(tmp_path, strict=False)
    status.mark_ready()
    client = SRTThinkingClient("http://127.0.0.1:1234", 3, 90, status=status)

    assert client.status.state.value == "ready"
    with _capture_thinking_log() as handler:
        with pytest.raises(requests.ConnectionError):
            client.generate_batch(
                [[1]], max_think_tokens=4, eos_token_id=8, think_end_token_id=9
            )
        assert client.status.state.value == "fallback"
        assert status.snapshot()["state"] == "fallback"

        with pytest.raises(RuntimeError, match="unavailable"):
            client.generate_batch(
                [[1]], max_think_tokens=4, eos_token_id=8, think_end_token_id=9
            )

    assert calls == ["http://127.0.0.1:1234/generate"]
    assert len(handler.messages) == 1
    assert "ConnectionError: connection refused" in handler.messages[0]


def test_sensenova_thinking_client_observes_parent_startup_failure(
    tmp_path, monkeypatch
):
    url = "http://127.0.0.1:1234"
    monkeypatch.setenv("SGLANG_SENSENOVA_THINKING_RUNTIME_DIR", str(tmp_path))
    monkeypatch.delenv("SGLANG_SENSENOVA_THINKING_LOG_FILE", raising=False)
    monkeypatch.delenv("SGLANG_SENSENOVA_THINKING_STRICT", raising=False)
    client = SRTThinkingClient(url, 3, 90, status=ThinkingBackendStatus.for_url(url))
    parent_status = ThinkingBackendStatus.for_url(url)
    parent_status.mark_unavailable("the internal SRT process exited")

    def unexpected_post(*args, **kwargs):
        raise AssertionError("the client must not call a backend already marked failed")

    monkeypatch.setattr("requests.post", unexpected_post)
    with _capture_thinking_log() as handler:
        with pytest.raises(RuntimeError, match="unavailable"):
            client.generate_batch(
                [[1]], max_think_tokens=4, eos_token_id=8, think_end_token_id=9
            )
        assert client.fail(RuntimeError("unavailable")) is False

    assert client.status.state.value == "fallback"
    assert handler.messages == []


def test_sensenova_thinking_backend_info_reports_the_state(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_SENSENOVA_THINKING_RUNTIME_DIR", str(tmp_path))
    monkeypatch.delenv("SGLANG_SENSENOVA_THINKING_LOG_FILE", raising=False)
    monkeypatch.delenv("SGLANG_SENSENOVA_THINKING_STRICT", raising=False)
    # thinking_backend_info matches the pipeline config by class name.
    pipeline_config = type("SenseNovaU1PipelineConfig", (), {})()
    url = "http://127.0.0.1:1234"

    server_args = SimpleNamespace(pipeline_config=pipeline_config, srt_encoder_url=None)
    assert thinking_backend_info(server_args) == {
        "state": "disabled",
        "backend": "native",
        "url": None,
        "strict": False,
        "reason": "SGLANG_SENSENOVA_THINKING_BACKEND=native",
        "log_file": None,
        "pid": os.getpid(),
    }
    assert (
        thinking_backend_info(SimpleNamespace(pipeline_config=SimpleNamespace()))
        is None
    )

    server_args.srt_encoder_url = url
    assert thinking_backend_info(server_args)["state"] == "starting"

    ThinkingBackendStatus.for_url(url).mark_ready()
    info = thinking_backend_info(server_args)
    assert info["state"] == "ready"
    assert info["backend"] == "srt"
    assert info["log_file"].startswith(str(tmp_path))

    # The state is shared through the file, but strict is a live setting: a
    # record written by a run with another setting must not report it.
    monkeypatch.setenv("SGLANG_SENSENOVA_THINKING_STRICT", "1")
    assert thinking_backend_info(server_args)["strict"] is True
    monkeypatch.setenv("SGLANG_SENSENOVA_THINKING_STRICT", "0")
    assert thinking_backend_info(server_args)["strict"] is False


def test_sensenova_thinking_runtime_files_honor_the_log_override(tmp_path, monkeypatch):
    runtime_dir = tmp_path / "runtime"
    log_file = tmp_path / "logs" / "srt.log"
    monkeypatch.setenv("SGLANG_SENSENOVA_THINKING_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("SGLANG_SENSENOVA_THINKING_LOG_FILE", str(log_file))

    status_file, actual_log_file = runtime_files("http://127.0.0.1:1234")

    assert os.path.dirname(status_file) == str(runtime_dir)
    assert status_file.endswith(".json")
    assert actual_log_file == str(log_file)


def test_sensenova_thinking_strict_mode_reads_the_environment(monkeypatch):
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("SGLANG_SENSENOVA_THINKING_STRICT", value)
        assert thinking_strict_enabled()
    for value in ("", "0", "false", "off"):
        monkeypatch.setenv("SGLANG_SENSENOVA_THINKING_STRICT", value)
        assert not thinking_strict_enabled()


def test_sensenova_srt_loads_only_understanding_weights():
    weights = [
        ("language_model.model.layers.0.self_attn.q_proj.weight", 1),
        ("language_model.model.layers.0.self_attn.q_proj_mot_gen.weight", 2),
        ("fm_modules.fm_head.weight", 3),
    ]

    assert list(_understanding_weights(weights)) == [
        ("model.layers.0.self_attn.q_proj.weight", 1)
    ]


def test_sensenova_u1_generation_stage_passes_dynamic_batch_inputs():
    sampling = SenseNovaU1SamplingParams(
        prompt="first prompt",
        width=1024,
        height=1024,
        guidance_scale=4.0,
        num_inference_steps=5,
        seed=7,
    )
    batch = SimpleNamespace(
        prompt=["first prompt", "a longer second prompt"],
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=sampling.seed,
        num_outputs_per_prompt=1,
        extra={
            **sampling.build_request_extra(),
            "dynamic_batch_seeds": [7, 19],
        },
        metrics=None,
    )
    model = _FakeSenseNovaModel()
    stage = SenseNovaU1GenerationStage(model=model, tokenizer="tok")

    output = stage.forward(batch, server_args=SimpleNamespace())

    assert len(output.output) == 2
    assert model.call_kwargs["prompt"] == [
        "first prompt",
        "a longer second prompt",
    ]
    assert model.call_kwargs["batch_size"] == 2
    assert model.call_kwargs["seed"] == [7, 19]


def test_sensenova_u1_generation_stage_batches_srt_thinking_usage():
    sampling = SenseNovaU1SamplingParams(
        prompt="first prompt",
        width=1024,
        height=1024,
        think_mode=True,
    )
    batch = SimpleNamespace(
        prompt=["first prompt", "second prompt"],
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=sampling.seed,
        num_outputs_per_prompt=1,
        extra={
            **sampling.build_request_extra(),
            "dynamic_batch_seeds": [7, 19],
        },
        metrics=None,
    )

    model = _FakeSenseNovaModel()
    output = SenseNovaU1GenerationStage(
        model=model, tokenizer="tok", thinking_backend=object()
    ).forward(batch, server_args=SimpleNamespace())

    assert len(output.output) == 2
    assert model.call_kwargs["batch_size"] == 2
    assert output.usage is None
    assert output.usage_list == [
        {
            "think_text": "draft</think>",
            "reasoning_tokens": 3,
            "thinking_backend": "srt",
        },
        {
            "think_text": "draft</think>",
            "reasoning_tokens": 3,
            "thinking_backend": "srt",
        },
    ]


def test_sensenova_u1_generation_stage_runs_native_fallback_sequentially():
    sampling = SenseNovaU1SamplingParams(
        prompt="first prompt", width=1024, height=1024, think_mode=True
    )
    batch = SimpleNamespace(
        prompt=["first prompt", "second prompt"],
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=sampling.seed,
        num_outputs_per_prompt=1,
        extra={
            **sampling.build_request_extra(),
            "dynamic_batch_seeds": [7, 19],
        },
        metrics=None,
    )
    model = _FakeSenseNovaModel()

    output = SenseNovaU1GenerationStage(model=model, tokenizer="tok").forward(
        batch, server_args=SimpleNamespace()
    )

    assert [call["prompt"] for call in model.call_kwargs_list] == [
        "first prompt",
        "second prompt",
    ]
    assert [call["seed"] for call in model.call_kwargs_list] == [7, 19]
    assert all(call["batch_size"] == 1 for call in model.call_kwargs_list)
    assert [usage["thinking_backend"] for usage in output.usage_list] == [
        "native",
        "native",
    ]


def test_sensenova_u1_multi_output_request_expands_before_generation_stage():
    sampling = SenseNovaU1SamplingParams(
        prompt="a mountain lake",
        width=2304,
        height=4096,
        num_outputs_per_prompt=2,
    )
    batch = Req(
        request_id="req-0",
        prompt=sampling.prompt,
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=42,
        sampling_params=sampling,
        extra=sampling.build_request_extra(),
        output_file_name="sample.png",
    )
    server_args = SimpleNamespace(pipeline_config=SenseNovaU1PipelineConfig())
    input_stage = InputValidationStage()
    stage = SenseNovaU1GenerationStage(model=_FakeSenseNovaModel(), tokenizer="tok")
    batch.metrics.record_stage("InputValidationStage", 0.125)
    batch.metrics.record_memory_snapshot(
        "after_validation",
        MemorySnapshot(
            allocated_mb=100.0,
            reserved_mb=200.0,
            peak_allocated_mb=300.0,
            peak_reserved_mb=400.0,
        ),
    )

    expanded = list(input_stage.iter_sequential_requests(batch, server_args))

    assert [req.num_outputs_per_prompt for req in expanded] == [1, 1]
    assert [req.seed for req in expanded] == [42, 43]
    assert [req.request_id for req in expanded] == ["req-0:0", "req-0:1"]
    assert [req.output_file_name for req in expanded] == [
        "sample_0.png",
        "sample_1.png",
    ]
    assert [req.metrics.request_id for req in expanded] == ["req-0:0", "req-0:1"]
    assert all(req.trace_ctx is batch.trace_ctx for req in expanded)
    assert all(req.metrics is not batch.metrics for req in expanded)
    assert expanded[0].metrics is not expanded[1].metrics
    assert all(
        req.metrics.stages == {"InputValidationStage": 125.0} for req in expanded
    )
    assert all(
        req.metrics.memory_snapshots["after_validation"].peak_reserved_mb == 400.0
        for req in expanded
    )
    assert (
        expanded[0].metrics.memory_snapshots["after_validation"]
        is not expanded[1].metrics.memory_snapshots["after_validation"]
    )

    expanded[0].metrics.record_stage("child-only", 0.5)
    expanded[0].metrics.memory_snapshots["after_validation"].peak_reserved_mb = 999.0
    assert "child-only" not in expanded[1].metrics.stages
    assert "child-only" not in batch.metrics.stages
    assert (
        expanded[1].metrics.memory_snapshots["after_validation"].peak_reserved_mb
        == 400.0
    )
    assert batch.metrics.memory_snapshots["after_validation"].peak_reserved_mb == 400.0

    for req in expanded:
        output = stage.forward(req, server_args=SimpleNamespace())
        assert len(output.output) == 1


def test_sensenova_u1_multi_output_rejects_short_seed_list():
    sampling = SenseNovaU1SamplingParams(
        prompt="a mountain lake",
        width=2304,
        height=4096,
        num_outputs_per_prompt=2,
        seed=[7],
    )
    batch = Req(
        request_id="req-0",
        prompt=sampling.prompt,
        width=sampling.width,
        height=sampling.height,
        guidance_scale=sampling.guidance_scale,
        num_inference_steps=sampling.num_inference_steps,
        seed=sampling.seed,
        sampling_params=sampling,
        extra=sampling.build_request_extra(),
        output_file_name="sample.png",
    )
    server_args = SimpleNamespace(pipeline_config=SenseNovaU1PipelineConfig())

    with pytest.raises(ValueError, match="seed list length"):
        list(InputValidationStage().iter_sequential_requests(batch, server_args))


def _make_sensenova_u1_sequential_entrypoint(*, fail=False, fail_request_ids=None):
    sampling = SenseNovaU1SamplingParams(
        prompt="a mountain lake",
        width=2304,
        height=4096,
        num_outputs_per_prompt=2,
        save_output=False,
        suppress_logs=True,
    )
    trace_ctx = _RecordingTraceContext()
    batch = Req(
        request_id="req-0",
        prompt=sampling.prompt,
        width=sampling.width,
        height=sampling.height,
        seed=42,
        sampling_params=sampling,
        extra=sampling.build_request_extra(),
        output_file_name="sample.png",
        trace_ctx=trace_ctx,
    )
    server_args = SimpleNamespace(pipeline_config=SenseNovaU1PipelineConfig())
    pipeline = _SequentialTestPipeline(
        server_args, fail=fail, fail_request_ids=fail_request_ids
    )
    worker = GPUWorker.__new__(GPUWorker)
    worker.pipeline = pipeline
    worker.server_args = server_args
    worker.is_output_rank = True
    worker._runtime_peak_reserved_mb = 0.0
    worker._release_warmup_pool_before_serving = False
    worker._realtime_sessions = SimpleNamespace(attach=lambda _req: None)
    return batch, trace_ctx, pipeline.executor, _WorkerBackedSchedulerClient(worker)


def _force_cpu_entrypoint(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cpu", lambda: True)
    monkeypatch.setattr(current_platform, "is_mps", lambda: False)
    monkeypatch.setattr(current_platform, "is_npu", lambda: False)
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.entrypoints.openai.utils.get_global_server_args",
        lambda: SimpleNamespace(batching_max_size=1),
    )


def test_sensenova_u1_multi_output_entrypoint_success(monkeypatch):
    _force_cpu_entrypoint(monkeypatch)
    batch, trace_ctx, executor, scheduler_client = (
        _make_sensenova_u1_sequential_entrypoint()
    )

    paths, result = asyncio.run(process_generation_batch(scheduler_client, batch))

    assert paths == ["sample_0.png", "sample_1.png"]
    assert result.error is None
    assert [req.request_id for req in executor.executed_requests] == [
        "req-0:0",
        "req-0:1",
    ]
    assert [req.seed for req in executor.executed_requests] == [42, 43]
    assert result.metrics_list is not None
    assert [metrics.request_id for metrics in result.metrics_list] == [
        "req-0:0",
        "req-0:1",
    ]
    assert all(
        "InputValidationStage" in metrics.stages
        and "PipelineExecutor.sequential_wait" in metrics.stages
        and metrics.memory_snapshots["after_validation"].peak_reserved_mb == 400.0
        for metrics in result.metrics_list
    )
    assert all(req.trace_ctx is trace_ctx for req in executor.executed_requests)
    assert trace_ctx.started_slices == [("gpu_forward", 2)]
    assert trace_ctx.finished_slices == [("gpu_forward", 2)]
    assert trace_ctx.finish_count == 1


def test_sensenova_u1_multi_output_entrypoint_failure(monkeypatch):
    _force_cpu_entrypoint(monkeypatch)
    batch, trace_ctx, executor, scheduler_client = (
        _make_sensenova_u1_sequential_entrypoint(fail=True)
    )

    with pytest.raises(RuntimeError, match="generation failed for req-0:0"):
        asyncio.run(process_generation_batch(scheduler_client, batch))

    assert [req.request_id for req in executor.executed_requests] == [
        "req-0:0",
        "req-0:1",
    ]
    assert all(
        "InputValidationStage" in req.metrics.stages
        and "PipelineExecutor.sequential_wait" in req.metrics.stages
        and req.metrics.memory_snapshots["after_validation"].peak_reserved_mb == 400.0
        for req in executor.executed_requests
    )
    assert all(req.trace_ctx is trace_ctx for req in executor.executed_requests)
    assert trace_ctx.started_slices == [("gpu_forward", 2)]
    assert trace_ctx.finished_slices == [("gpu_forward", 2)]
    assert trace_ctx.finish_count == 1


@pytest.mark.parametrize("failed_request_id", ["req-0:0", "req-0:1"])
def test_sensenova_u1_multi_output_entrypoint_mixed_failure_fails_parent(
    monkeypatch, failed_request_id
):
    _force_cpu_entrypoint(monkeypatch)
    batch, trace_ctx, executor, scheduler_client = (
        _make_sensenova_u1_sequential_entrypoint(fail_request_ids={failed_request_id})
    )

    with pytest.raises(
        RuntimeError, match=f"generation failed for {failed_request_id}"
    ):
        asyncio.run(process_generation_batch(scheduler_client, batch))

    assert [req.request_id for req in executor.executed_requests] == [
        "req-0:0",
        "req-0:1",
    ]
    assert all(req.trace_ctx is trace_ctx for req in executor.executed_requests)
    assert trace_ctx.started_slices == [("gpu_forward", 2)]
    assert trace_ctx.finished_slices == [("gpu_forward", 2)]
    assert trace_ctx.finish_count == 1
