# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Run the model with npu graph and torch.compile."""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

import numpy as np
import torch

import sglang
from sglang.srt.configs.model_config import AttentionArch, is_deepseek_nsa
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.utils import (
    empty_context,
    get_bool_env_var,
    get_compiler_backend,
    get_int_env_var,
    is_npu,
)

is_npu = is_npu()

if is_npu:
    import torch_npu
    from torch_npu.profiler import ProfilerActivity, profile

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors


@contextmanager
def patch_model_npu(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
):
    if enable_compile:
        backend = get_compiler_backend("npugraph_ex")
        yield torch.compile(
            torch.no_grad()(model.forward),
            fullgraph=True,
            dynamic=False,
            backend=backend,
        )
    else:
        yield model.forward


class NPUGraphRunner(CudaGraphRunner):
    """A NPUGraphRunner runs the forward pass of a model with npu graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        sglang.srt.model_executor.cuda_graph_runner.patch_model = patch_model_npu
        super().__init__(model_runner)
        self.update_attr_name = None
        self.update_attr_type = None
        self.model_runner = model_runner
        self._init_arch_map()
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        # NPU CSAttention strategy:
        # - default: keep decode on graph (enables layer-level local graph policy in indexer)
        # - optional legacy mode: bypass graph for long-context decode batches
        #   via SGLANG_NSA_CSATTENTION_NPU_GRAPH_OUTSIDE=1.
        self.nsa_csattention_enabled = get_bool_env_var("SGLANG_NSA_USE_CSATTENTION", "false")
        self.nsa_cs_graph_outside = get_bool_env_var(
            "SGLANG_NSA_CSATTENTION_NPU_GRAPH_OUTSIDE", "false"
        )
        self.nsa_cs_long_context_threshold = max(
            0, get_int_env_var("SGLANG_NSA_CS_LONG_CONTEXT_THRESHOLD", 16384)
        )
        self.is_nsa_model = is_deepseek_nsa(self.model_runner.model_config.hf_config)
        self._nsa_cs_bypass_graph_logged = False

    def _init_arch_map(self):
        if self.is_dllm:
            self.attr_name: Dict[str, str] = {
                AttentionArch.MLA: "actual_seq_lengths_kv",
                AttentionArch.MHA: "actual_seq_lengths_kv",
            }
        else:
            self.attr_name: Dict[str, str] = {
                AttentionArch.MLA: "actual_seq_lengths_kv",
                AttentionArch.MHA: "context_lens",
            }
        self.attr_type: Dict[str, Union[list, torch.Tensor]] = {
            AttentionArch.MLA: [],
            AttentionArch.MHA: torch.Tensor(),
        }

    def _create_device_graph(self):
        return torch.npu.NPUGraph()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        if self.enable_torch_compile:
            skip_guard_context = torch.compiler.set_stance(skip_guard_eval_unsafe=True)
        else:
            skip_guard_context = empty_context()

        with skip_guard_context, torch.npu.graph(
            graph,
            pool=pool,
            stream=stream,
            auto_dispatch_capture=True,
        ):
            out = run_once_fn()
        return out

    def _get_update_attr_name(self):
        return self.attr_name[AttentionArch.MLA]

    def _get_update_attr_type(self):
        return self.attr_type[AttentionArch.MLA]

    def _update_inputs(self, seq_lens):
        if isinstance(self.update_attr_type, torch.Tensor):
            seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))

        self.graphs[self.bs].update(
            cpu_update_input=[{self.update_attr_name: seq_lens}]
        )

    def _cache_loc_dtype(self):
        return torch.int32

    def _init_profile_context_and_memory_record(self):
        output_dir = os.path.join(
            os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp"), "graph_capture_profile"
        )
        if not Path(output_dir).exists():
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Profiling starts for graph capture for NPU. Traces will be saved to: {output_dir}"
        )
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=[torch_npu.profiler.ExportType.Text],
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        )
        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                output_dir, async_mode=True
            ),
            experimental_config=experimental_config,
        )
        return profile_context

    def _post_process_after_profile(self, prof_context):
        # for NPU, profile data will be saved to disk for further analysis.
        pass

    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        if not skip_attn_backend_init:
            self.replay_prepare(forward_batch, pp_proxy_tensors)
        else:
            # In speculative decoding, these two fields are still needed.
            self.buffers.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.buffers.positions[: self.raw_num_token].copy_(forward_batch.positions)

        self.update_attr_name = self._get_update_attr_name()
        self.update_attr_type = self._get_update_attr_type()
        # Replay
        if not is_deepseek_nsa(self.model_runner.model_config.hf_config):
            if forward_batch.forward_mode.is_target_verify():
                seq_lens_cpu = forward_batch.seq_lens.cpu() + self.num_tokens_per_bs
                seq_lens = seq_lens_cpu.tolist() + [0] * (self.bs - self.raw_bs)
            else:
                seq_lens = forward_batch.seq_lens.cpu().tolist() + [0] * (
                    self.bs - self.raw_bs
                )
            thread = threading.Thread(target=self._update_inputs, args=(seq_lens,))
            thread.start()
            self.graphs[self.bs].replay()
            thread.join()
        else:
            self.graphs[self.bs].replay()

        output = self.output_buffers[self.bs]
        if isinstance(output, LogitsProcessorOutput):
            if self.is_dllm:
                next_token_logits = None
                full_logits = output.full_logits[: self.raw_num_token]
            else:
                full_logits = None
                next_token_logits = output.next_token_logits[: self.raw_num_token]
            return LogitsProcessorOutput(
                next_token_logits=next_token_logits,
                full_logits=full_logits,
                hidden_states=(
                    output.hidden_states[: self.raw_num_token]
                    if output.hidden_states is not None
                    else None
                ),
            )
        else:
            assert isinstance(output, PPProxyTensors)
            return PPProxyTensors({k: v[: self.bs] for k, v in output.tensors.items()})

    def can_run(self, forward_batch: ForwardBatch):
        if not super().can_run(forward_batch):
            return False

        if (
            not self.nsa_csattention_enabled
            or not self.nsa_cs_graph_outside
            or not self.is_nsa_model
            or not forward_batch.forward_mode.is_decode_or_idle()
        ):
            return True

        max_seq_len = None
        if forward_batch.seq_lens_cpu is not None:
            max_seq_len = int(forward_batch.seq_lens_cpu.max().item())
        elif forward_batch.seq_lens is not None and forward_batch.seq_lens.numel() > 0:
            max_seq_len = int(forward_batch.seq_lens.max().item())

        if max_seq_len is None:
            return True

        if max_seq_len >= self.nsa_cs_long_context_threshold:
            if not self._nsa_cs_bypass_graph_logged:
                logger.info(
                    "Bypass NPU graph for long-context CSAttention batch "
                    "(legacy full-outside mode): "
                    "max_seq_len=%s threshold=%s",
                    max_seq_len,
                    self.nsa_cs_long_context_threshold,
                )
                self._nsa_cs_bypass_graph_logged = True
            return False

        return True
