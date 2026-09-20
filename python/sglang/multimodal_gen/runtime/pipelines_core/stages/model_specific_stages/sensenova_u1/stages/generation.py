# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from sglang.multimodal_gen.configs.sensenova_u1 import (
    DEFAULT_CFG_INTERVAL,
    DEFAULT_CFG_NORM,
    DEFAULT_ENABLE_TIMESTEP_SHIFT,
    DEFAULT_MAX_THINK_TOKENS,
    DEFAULT_T_EPS,
    DEFAULT_THINK_MODE,
    DEFAULT_TIMESTEP_SHIFT,
    SENSENOVA_U1_REQUEST_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.models.sensenova_u1.srt_thinking import (
    BatchedSRTThinkingFallbackRequired,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _denorm_sensenova_output(x: torch.Tensor) -> torch.Tensor:
    """Convert SenseNova's normalized image tensor from [-1, 1] to [0, 1]."""
    return ((x.float() + 1.0) * 0.5).clamp(0, 1)


@dataclass(frozen=True)
class SenseNovaU1GenerationOptions:
    cfg_norm: str = DEFAULT_CFG_NORM
    timestep_shift: float = DEFAULT_TIMESTEP_SHIFT
    enable_timestep_shift: bool = DEFAULT_ENABLE_TIMESTEP_SHIFT
    cfg_interval: tuple[float, float] = DEFAULT_CFG_INTERVAL
    t_eps: float = DEFAULT_T_EPS
    think_mode: bool = DEFAULT_THINK_MODE
    max_think_tokens: int = DEFAULT_MAX_THINK_TOKENS
    profile_stages: bool = False

    @classmethod
    def from_batch(cls, batch: Req) -> SenseNovaU1GenerationOptions:
        extra = batch.extra.get(SENSENOVA_U1_REQUEST_EXTRA_KEY, {})
        return cls(
            cfg_norm=extra.get("cfg_norm", DEFAULT_CFG_NORM),
            timestep_shift=float(extra.get("timestep_shift", DEFAULT_TIMESTEP_SHIFT)),
            enable_timestep_shift=bool(
                extra.get("enable_timestep_shift", DEFAULT_ENABLE_TIMESTEP_SHIFT)
            ),
            cfg_interval=tuple(extra.get("cfg_interval", DEFAULT_CFG_INTERVAL)),
            t_eps=float(extra.get("t_eps", DEFAULT_T_EPS)),
            think_mode=bool(extra.get("think_mode", DEFAULT_THINK_MODE)),
            max_think_tokens=int(
                extra.get("max_think_tokens", DEFAULT_MAX_THINK_TOKENS)
            ),
            profile_stages=bool(extra.get("profile_stages", False)),
        )


class SenseNovaU1GenerationStage(PipelineStage):
    def __init__(self, model: torch.nn.Module, tokenizer: Any, thinking_backend=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.thinking_backend = thinking_backend

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        del server_args
        options = SenseNovaU1GenerationOptions.from_batch(batch)
        if int(batch.num_outputs_per_prompt) != 1:
            raise ValueError(
                "SenseNova-U1 expects output expansion before generation; "
                f"got num_outputs_per_prompt={batch.num_outputs_per_prompt}."
            )
        prompts = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]
        batch_size = len(prompts)
        if batch_size == 0:
            raise ValueError(
                "SenseNova-U1 dynamic batch must contain at least one prompt"
            )
        dynamic_seeds = batch.extra.get("dynamic_batch_seeds")
        if dynamic_seeds is None:
            dynamic_seeds = batch.seed if isinstance(batch.seed, list) else [batch.seed]
        elif not isinstance(dynamic_seeds, list):
            dynamic_seeds = [dynamic_seeds]
        seeds = []
        for seed in dynamic_seeds:
            if isinstance(seed, list):
                if len(seed) != 1:
                    raise ValueError(
                        "SenseNova-U1 dynamic batching requires one seed per request"
                    )
                seed = seed[0]
            seeds.append(int(seed))
        if len(seeds) != batch_size:
            raise ValueError(
                "SenseNova-U1 dynamic batch requires one seed per prompt; "
                f"got {len(seeds)} seeds for {batch_size} prompts"
            )
        seed = seeds[0] if batch_size == 1 else seeds

        generation_kwargs = {
            "image_size": (int(batch.width), int(batch.height)),
            "cfg_scale": float(batch.guidance_scale),
            "cfg_norm": options.cfg_norm,
            "timestep_shift": options.timestep_shift,
            "enable_timestep_shift": options.enable_timestep_shift,
            "cfg_interval": options.cfg_interval,
            "num_steps": int(batch.num_inference_steps),
            "t_eps": options.t_eps,
            "think_mode": options.think_mode,
            "max_think_tokens": options.max_think_tokens,
            "profile_stages": options.profile_stages,
        }

        def generate(prompt, item_batch_size, item_seed, thinking_backend):
            return self.model.t2i_generate(
                self.tokenizer,
                prompt,
                batch_size=item_batch_size,
                thinking_backend=thinking_backend,
                seed=item_seed,
                **generation_kwargs,
            )

        use_native_fallback = (
            batch_size > 1 and options.think_mode and self.thinking_backend is None
        )
        if not use_native_fallback:
            try:
                out = generate(batch.prompt, batch_size, seed, self.thinking_backend)
            except BatchedSRTThinkingFallbackRequired:
                use_native_fallback = True

        per_request_timings = None
        if use_native_fallback:
            image_batches = []
            think_text = []
            think_token_counts = []
            thinking_backends = []
            per_request_timings = []
            for prompt, item_seed in zip(prompts, seeds):
                item_images, item_think_text = generate(prompt, 1, item_seed, None)
                image_batches.append(item_images)
                think_text.append(item_think_text)
                think_token_counts.append(
                    int(getattr(self.model, "last_think_token_count", 0))
                )
                thinking_backends.append(
                    getattr(self.model, "last_thinking_backend", "native")
                )
                per_request_timings.append(
                    dict(getattr(self.model, "last_profile_timings_ms", {}))
                )
            images = torch.cat(image_batches, dim=0)
        elif options.think_mode:
            images, think_text = out
            think_text = think_text if isinstance(think_text, list) else [think_text]
            think_token_counts = list(
                getattr(
                    self.model,
                    "last_think_token_counts",
                    [getattr(self.model, "last_think_token_count", 0)] * batch_size,
                )
            )
            thinking_backends = list(
                getattr(
                    self.model,
                    "last_thinking_backends",
                    [getattr(self.model, "last_thinking_backend", "native")]
                    * batch_size,
                )
            )
        else:
            images = out
            think_text = None

        images = _denorm_sensenova_output(images)
        samples = [sample.contiguous() for sample in images]
        usage_list = []
        for index in range(batch_size):
            usage = {}
            if think_text is not None:
                usage.update(
                    think_text=think_text[index],
                    reasoning_tokens=int(think_token_counts[index]),
                    thinking_backend=thinking_backends[index],
                )
            if options.profile_stages:
                usage["stage_timings_ms"] = (
                    per_request_timings[index]
                    if per_request_timings is not None
                    else dict(getattr(self.model, "last_profile_timings_ms", {}))
                )
            usage_list.append(usage or None)
        return OutputBatch(
            output=samples,
            metrics=batch.metrics,
            usage=usage_list[0] if batch_size == 1 else None,
            usage_list=usage_list if batch_size > 1 else None,
        )
