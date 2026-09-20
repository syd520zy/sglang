# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import os
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

_MANAGED_SERVERS: list[ManagedSRTThinkingServer] = []


def normalize_thinking_output_ids(
    output_ids: list[int],
    *,
    max_think_tokens: int,
    eos_token_id: int,
    think_end_token_id: int,
) -> list[int]:
    normalized = []
    for token_id in output_ids:
        if token_id == eos_token_id:
            break
        normalized.append(int(token_id))
        if token_id == think_end_token_id:
            break

    if normalized and normalized[-1] == think_end_token_id:
        return normalized[:max_think_tokens]
    return normalized[: max_think_tokens - 1] + [think_end_token_id]


class SRTThinkingClient:
    def __init__(self, url: str, connect_timeout: float, timeout: float):
        self.url = url.rstrip("/")
        self.connect_timeout = connect_timeout
        self.timeout = timeout
        self.available = True

    def generate(
        self,
        input_ids: list[int],
        *,
        max_think_tokens: int,
        eos_token_id: int,
        think_end_token_id: int,
    ) -> list[int]:
        if not self.available:
            raise RuntimeError("SenseNova thinking SRT backend is unavailable")
        try:
            response = requests.post(
                f"{self.url}/generate",
                json={
                    "input_ids": input_ids,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": max(1, max_think_tokens - 1),
                        "stop_token_ids": [eos_token_id, think_end_token_id],
                        "no_stop_trim": True,
                        "skip_special_tokens": False,
                    },
                },
                timeout=(self.connect_timeout, self.timeout),
            )
            response.raise_for_status()
            payload = response.json()
            output_ids = (
                payload.get("output_ids") if isinstance(payload, dict) else None
            )
        except (requests.RequestException, ValueError):
            self.available = False
            raise
        if not isinstance(output_ids, list):
            self.available = False
            raise TypeError("SenseNova thinking SRT response has no output_ids")
        return normalize_thinking_output_ids(
            output_ids,
            max_think_tokens=max_think_tokens,
            eos_token_id=eos_token_id,
            think_end_token_id=think_end_token_id,
        )


def _run_srt_server(server_args) -> None:
    from sglang.srt.entrypoints.http_server import launch_server

    launch_server(server_args)


@dataclass
class ManagedSRTThinkingServer:
    process: mp.Process
    url: str

    def start(self) -> bool:
        self.process.start()
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            try:
                if (
                    requests.get(f"{self.url}/health_generate", timeout=2).status_code
                    == 200
                ):
                    _MANAGED_SERVERS.append(self)
                    atexit.register(self.shutdown)
                    logger.info(
                        "SenseNova managed SRT thinking backend ready at %s", self.url
                    )
                    return True
            except requests.RequestException:
                pass
            if not self.process.is_alive():
                logger.warning(
                    "SenseNova managed SRT thinking backend failed to start; "
                    "requests will use the native fallback"
                )
                return False
            time.sleep(2)

        self.shutdown()
        logger.warning(
            "SenseNova managed SRT thinking backend timed out; "
            "requests will use the native fallback"
        )
        return False

    def shutdown(self) -> None:
        if not self.process.is_alive():
            return
        from sglang.srt.utils import kill_process_tree

        kill_process_tree(self.process.pid, wait_timeout=60)


def _is_sensenova_pipeline(server_args) -> bool:
    return type(server_args.pipeline_config).__name__ == "SenseNovaU1PipelineConfig"


def prepare_managed_srt_thinking(server_args):
    """Reserve the internal text runtime before diffusion workers are spawned."""
    if not _is_sensenova_pipeline(server_args) or server_args.srt_encoder_url:
        return None
    backend = os.environ.get("SGLANG_SENSENOVA_THINKING_BACKEND", "srt").lower()
    if backend == "native":
        logger.info("SenseNova thinking uses the native fallback backend")
        return None
    if backend != "srt":
        raise ValueError("SGLANG_SENSENOVA_THINKING_BACKEND must be 'srt' or 'native'")

    from sglang.srt.server_args import ServerArgs as SRTServerArgs
    from sglang.srt.utils.network import get_free_port

    host = "127.0.0.1"
    port = get_free_port()
    srt_args = SRTServerArgs(
        model_path=server_args.model_path,
        host=host,
        port=port,
        trust_remote_code=True,
        dtype="bfloat16",
        attention_backend="triton",
        context_length=4096,
        language_model_only=True,
        base_gpu_id=server_args.base_gpu_id,
        tp_size=1,
        mem_fraction_static=float(
            os.environ.get("SGLANG_SENSENOVA_THINKING_MEM_FRACTION", "0.3")
        ),
        max_running_requests=8,
        cuda_graph_max_bs_decode=2,
    )
    process = mp.get_context("spawn").Process(
        target=_run_srt_server,
        args=(srt_args,),
        name="sensenova-srt-thinking",
    )
    url = f"http://{host}:{port}"
    server_args.srt_encoder_url = url
    return ManagedSRTThinkingServer(process=process, url=url)
