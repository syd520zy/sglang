# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import atexit
import hashlib
import json
import logging
import multiprocessing as mp
import os
import sys
import tempfile
import time
from enum import Enum

import requests

logger = logging.getLogger(__name__)


class BatchedSRTThinkingFallbackRequired(RuntimeError):
    """Ask the generation stage to preserve native fallback per request."""


_ENV_BACKEND = "SGLANG_SENSENOVA_THINKING_BACKEND"
_ENV_STRICT = "SGLANG_SENSENOVA_THINKING_STRICT"
_ENV_RUNTIME_DIR = "SGLANG_SENSENOVA_THINKING_RUNTIME_DIR"
_ENV_LOG_FILE = "SGLANG_SENSENOVA_THINKING_LOG_FILE"
_DEFAULT_RUNTIME_DIRNAME = "sglang-sensenova-thinking"
_STARTUP_DEADLINE_S = 600.0
_UNSAFE_PATH_CHARS = ("/", "\\", "\x00")

_MANAGED_SERVERS: list[ManagedSRTThinkingServer] = []


class ThinkingBackendState(str, Enum):
    """Lifecycle of the internal SRT thinking backend.

    disabled: thinking uses the native decode by configuration.
    starting: the SRT subprocess is spawned, its HTTP surface is not confirmed yet.
    ready: SRT answers, thinking requests are served by it.
    failed: SRT is required (strict mode) and unusable, so requests fail.
    fallback: SRT is unusable and not required, so thinking uses the native decode.
    stopped: the subprocess was reclaimed on shutdown.
    """

    DISABLED = "disabled"
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    FALLBACK = "fallback"
    STOPPED = "stopped"


_SRT_SERVING_STATES = (ThinkingBackendState.STARTING, ThinkingBackendState.READY)


def thinking_strict_enabled() -> bool:
    return os.environ.get(_ENV_STRICT, "").strip().lower() in ("1", "true", "yes", "on")


def checked_dir(path: str) -> str:
    """Normalize the configured runtime directory, refusing traversal components."""
    if not path or "\x00" in path or os.pardir in path.replace("\\", "/").split("/"):
        raise ValueError(f"refusing an unsafe thinking runtime dir: {path!r}")
    return os.path.abspath(path)


def checked_name(name: str) -> str:
    """Accept a single path component only, so it cannot leave its directory."""
    if not name or name in (".", os.pardir) or "\x00" in name:
        raise ValueError(f"refusing an unsafe thinking runtime file name: {name!r}")
    if any(separator in name for separator in _UNSAFE_PATH_CHARS):
        raise ValueError(f"refusing an unsafe thinking runtime file name: {name!r}")
    return name


def contained_path(root: str, name: str) -> str:
    """Join a checked name onto a checked directory and confirm containment."""
    directory = checked_dir(root)
    path = os.path.join(directory, checked_name(name))
    if os.path.commonpath([directory, path]) != directory:
        raise ValueError(f"refusing a path outside {directory!r}: {name!r}")
    return path


def runtime_files(url: str) -> tuple[str, str, str]:
    """Return the checked (runtime dir, status name, log name) for one SRT url."""
    root = checked_dir(
        os.environ.get(_ENV_RUNTIME_DIR)
        or os.path.join(tempfile.gettempdir(), _DEFAULT_RUNTIME_DIRNAME)
    )
    explicit_log = os.environ.get(_ENV_LOG_FILE)
    if explicit_log:
        name = checked_name(os.path.basename(checked_dir(explicit_log)))
    else:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        name = f"srt-thinking-{digest}"
    return root, f"{name}.json", f"{name}.log"


def read_thinking_status(url: str) -> dict | None:
    """The state recorded for this SRT url, or None when nothing usable exists."""
    root, status_name, _ = runtime_files(url)
    try:
        with open(contained_path(root, status_name), encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


class ThinkingBackendStatus:
    """Thinking backend state shared by the server parent and its pipeline workers.

    The parent owns the subprocess and records the state while starting and stopping
    it; the workers load the same file when they build the client and rewrite it once
    a request proves SRT unusable. That file is the cross-process view /server_info
    reports, so the two backends can be told apart without reading logs.
    """

    def __init__(
        self,
        url: str,
        *,
        strict: bool,
        runtime_dir: str | None = None,
        status_name: str | None = None,
        log_name: str | None = None,
    ):
        self.url = url
        self.strict = strict
        self.runtime_dir = checked_dir(runtime_dir) if runtime_dir else None
        self.status_name = checked_name(status_name) if status_name else None
        self.log_name = checked_name(log_name) if log_name else None
        self.state = ThinkingBackendState.STARTING
        self.reason = ""
        self._failure_reported = False

    @classmethod
    def for_url(cls, url: str) -> ThinkingBackendStatus:
        runtime_dir, status_name, log_name = runtime_files(url)
        status = cls(
            url,
            strict=thinking_strict_enabled(),
            runtime_dir=runtime_dir,
            status_name=status_name,
            log_name=log_name,
        )
        recorded = read_thinking_status(url)
        if recorded is not None:
            status.adopt(recorded)
        return status

    def adopt(self, recorded: dict) -> None:
        try:
            self.state = ThinkingBackendState(recorded["state"])
        except (KeyError, ValueError):
            return
        self.reason = str(recorded.get("reason") or "")
        self._failure_reported = self.state in (
            ThinkingBackendState.FAILED,
            ThinkingBackendState.FALLBACK,
        )

    def refresh(self) -> None:
        """Refresh state written by the parent or another pipeline process."""
        if not self.tracks_files():
            return
        recorded = read_thinking_status(self.url)
        if recorded is not None:
            self.adopt(recorded)

    def tracks_files(self) -> bool:
        return self.runtime_dir is not None and self.status_name is not None

    def log_path(self) -> str | None:
        if self.runtime_dir is None or self.log_name is None:
            return None
        return contained_path(self.runtime_dir, self.log_name)

    def snapshot(self) -> dict:
        return {
            "state": self.state.value,
            "backend": "srt" if self.state in _SRT_SERVING_STATES else "native",
            "url": self.url,
            "strict": self.strict,
            "reason": self.reason,
            "log_file": self.log_path(),
            # Which process wrote this record, so a stale one is identifiable.
            "pid": os.getpid(),
        }

    def write(self) -> None:
        if not self.tracks_files():
            return
        target = contained_path(self.runtime_dir, self.status_name)
        temporary = f"{target}.{os.getpid()}.{time.time_ns()}.tmp"
        try:
            os.makedirs(self.runtime_dir, exist_ok=True)
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(self.snapshot(), handle)
            os.replace(temporary, target)
        except OSError as exc:
            logger.warning("Could not write the thinking backend status file: %s", exc)
        finally:
            try:
                os.remove(temporary)
            except OSError as exc:
                if os.path.exists(temporary):
                    logger.debug("Could not remove thinking status temp file: %s", exc)

    def mark_starting(self) -> None:
        self.state = ThinkingBackendState.STARTING
        self.reason = ""
        self._failure_reported = False
        self.write()
        logger.info(
            "SenseNova thinking backend starting: srt at %s; internal SRT logs: %s",
            self.url,
            self.log_path(),
        )

    def mark_ready(self) -> None:
        if self.state is ThinkingBackendState.READY:
            return
        self.state = ThinkingBackendState.READY
        self.reason = ""
        self._failure_reported = False
        self.write()
        logger.info("SenseNova thinking backend ready: srt at %s", self.url)

    def mark_unavailable(self, reason: str) -> bool:
        """Record that SRT cannot serve; True only for its first failure here."""
        previous = self.state
        first = not self._failure_reported
        self._failure_reported = True
        self.state = (
            ThinkingBackendState.FAILED
            if self.strict
            else ThinkingBackendState.FALLBACK
        )
        if first:
            self.reason = reason
            logger.error(
                "SenseNova thinking backend unavailable (%s): %s; %s",
                self.state.value,
                reason,
                (
                    "strict mode requires SRT, so the request fails instead of falling back"
                    if self.strict
                    else "thinking requests use the native decode from now on"
                ),
            )
        if first or self.state is not previous:
            self.write()
        return first

    def mark_stopped(self) -> None:
        if self.state is ThinkingBackendState.STOPPED:
            return
        self.state = ThinkingBackendState.STOPPED
        self.write()


class SRTThinkingClient:
    def __init__(
        self,
        url: str,
        connect_timeout: float,
        timeout: float,
        status: ThinkingBackendStatus | None = None,
    ):
        self.url = url.rstrip("/")
        self.connect_timeout = connect_timeout
        self.timeout = timeout
        # Without an explicit status the client records nothing on disk.
        self.status = (
            status
            if status is not None
            else ThinkingBackendStatus(self.url, strict=thinking_strict_enabled())
        )
        self.available = self.status.state in _SRT_SERVING_STATES

    @classmethod
    def for_server_args(cls, server_args) -> SRTThinkingClient:
        url = server_args.srt_encoder_url
        return cls(
            url,
            server_args.srt_encoder_connect_timeout,
            server_args.srt_encoder_timeout,
            status=ThinkingBackendStatus.for_url(url),
        )

    @property
    def strict(self) -> bool:
        return self.status.strict

    def generate(
        self,
        input_ids: list[int],
        *,
        max_think_tokens: int,
        eos_token_id: int,
        think_end_token_id: int,
    ) -> list[int]:
        return self.generate_batch(
            [input_ids],
            max_think_tokens=max_think_tokens,
            eos_token_id=eos_token_id,
            think_end_token_id=think_end_token_id,
        )[0]

    def generate_batch(
        self,
        input_ids: list[list[int]],
        *,
        max_think_tokens: int,
        eos_token_id: int,
        think_end_token_id: int,
    ) -> list[list[int]]:
        # The managed server is started by the parent after pipeline workers are
        # spawned. It can also fail there, so re-read that cross-process result
        # before waiting on an address the parent already declared unavailable.
        self.status.refresh()
        self.available = self.status.state in _SRT_SERVING_STATES
        if not self.available:
            # The backend died earlier in this process; never wait on it again.
            raise RuntimeError("SenseNova thinking SRT backend is unavailable")
        try:
            response = requests.post(
                f"{self.url}/generate",
                json={
                    "input_ids": input_ids[0] if len(input_ids) == 1 else input_ids,
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
        except (requests.RequestException, ValueError) as exc:
            self.fail(exc)
            raise
        responses = [payload] if isinstance(payload, dict) else payload
        if not isinstance(responses, list) or len(responses) != len(input_ids):
            exc = TypeError(
                "SenseNova thinking SRT response does not match the request batch"
            )
            self.fail(exc)
            raise exc
        output_ids = [
            item.get("output_ids") if isinstance(item, dict) else None
            for item in responses
        ]
        if not all(isinstance(item, list) for item in output_ids):
            exc = TypeError("SenseNova thinking SRT response has no output_ids")
            self.fail(exc)
            raise exc
        self.status.mark_ready()
        return [
            normalize_thinking_output_ids(
                item,
                max_think_tokens=max_think_tokens,
                eos_token_id=eos_token_id,
                think_end_token_id=think_end_token_id,
            )
            for item in output_ids
        ]

    def fail(self, exc: BaseException) -> bool:
        """Mark SRT unusable; True when this was its first failure on this client."""
        self.available = False
        return self.status.mark_unavailable(f"{type(exc).__name__}: {exc}")


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


def _run_srt_server(
    server_args, runtime_dir: str | None = None, log_name: str | None = None
) -> None:
    if runtime_dir and log_name:
        # Keep the SRT runtime out of the diffusion log so both stay readable.
        target = contained_path(runtime_dir, log_name)
        os.makedirs(checked_dir(runtime_dir), exist_ok=True)
        handle = open(target, "a", encoding="utf-8", errors="replace", buffering=1)
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
        sys.stdout = handle
        sys.stderr = handle

    from sglang.srt.entrypoints.http_server import launch_server

    launch_server(server_args)


class ManagedSRTThinkingServer:
    """Owns the internal SRT subprocess: its state, its port and its CUDA context."""

    def __init__(
        self,
        process: mp.Process,
        url: str,
        status: ThinkingBackendStatus,
    ):
        self.process = process
        self.url = url
        self.status = status
        self._shutdown = False

    def start(self) -> bool:
        """Spawn SRT and wait for it; raises in strict mode when it cannot serve."""
        self.status.mark_starting()
        try:
            self.process.start()
        except Exception as exc:
            # No child owns the reserved port in this case. Publish the failure
            # before pipeline workers can try that address themselves.
            self._shutdown = True
            reason = (
                f"the internal SRT process could not start: {type(exc).__name__}: {exc}"
            )
            self.status.mark_unavailable(reason)
            if self.status.strict:
                raise RuntimeError(
                    f"SenseNova thinking requires SRT: {reason}"
                ) from exc
            return False
        failure = self._wait_until_ready()
        if failure is None:
            _MANAGED_SERVERS.append(self)
            atexit.register(self.shutdown)
            self.status.mark_ready()
            return True

        self.shutdown()
        self.status.mark_unavailable(failure)
        if self.status.strict:
            raise RuntimeError(f"SenseNova thinking requires SRT: {failure}")
        return False

    def _wait_until_ready(self) -> str | None:
        """Return the reason SRT never served, or None once it answers."""
        deadline = time.monotonic() + _STARTUP_DEADLINE_S
        while time.monotonic() < deadline:
            try:
                if (
                    requests.get(f"{self.url}/health_generate", timeout=2).status_code
                    == 200
                ):
                    return None
            except requests.RequestException:
                pass
            if not self.process.is_alive():
                return (
                    "the internal SRT process exited with code "
                    f"{self.process.exitcode} before it served requests"
                )
            time.sleep(2)
        return (
            f"the internal SRT server was not ready within {_STARTUP_DEADLINE_S:.0f}s"
        )

    def shutdown(self) -> None:
        """Reclaim the subprocess, its port and its CUDA context."""
        if self._shutdown:
            return
        self._shutdown = True
        if self.process.is_alive():
            from sglang.srt.utils import kill_process_tree

            kill_process_tree(self.process.pid, wait_timeout=60)
        self.process.join(timeout=10)
        self.status.mark_stopped()
        logger.info(
            "SenseNova internal SRT thinking backend stopped (pid %s)", self.process.pid
        )


def _is_sensenova_pipeline(server_args) -> bool:
    return type(server_args.pipeline_config).__name__ == "SenseNovaU1PipelineConfig"


def thinking_backend_info(server_args) -> dict | None:
    """Thinking backend state for /server_info, or None for other pipelines."""
    if not _is_sensenova_pipeline(server_args):
        return None
    strict = thinking_strict_enabled()
    url = server_args.srt_encoder_url
    if url is None:
        return {
            "state": ThinkingBackendState.DISABLED.value,
            "backend": "native",
            "url": None,
            "strict": strict,
            "reason": f"{_ENV_BACKEND}=native",
            "log_file": None,
            "pid": os.getpid(),
        }

    recorded = read_thinking_status(url)
    if recorded is not None:
        # The record carries the state the workers share; the configuration is
        # read live, so a file left by an earlier run cannot misreport it.
        return dict(recorded, url=url, strict=strict)
    runtime_dir, _, log_name = runtime_files(url)
    return {
        "state": ThinkingBackendState.STARTING.value,
        "backend": "srt",
        "url": url,
        "strict": strict,
        "reason": "no state recorded yet",
        "log_file": contained_path(runtime_dir, log_name),
        "pid": os.getpid(),
    }


def prepare_managed_srt_thinking(server_args):
    """Reserve the internal text runtime before diffusion workers are spawned."""
    if not _is_sensenova_pipeline(server_args):
        return None
    pipeline_config = server_args.pipeline_config
    pipeline_config.srt_thinking_dynamic_batching = bool(server_args.srt_encoder_url)
    if server_args.srt_encoder_url:
        return None
    backend = os.environ.get(_ENV_BACKEND, "srt").lower()
    if backend == "native":
        logger.info("SenseNova thinking uses the native fallback backend")
        return None
    if backend != "srt":
        raise ValueError(f"{_ENV_BACKEND} must be 'srt' or 'native'")

    from sglang.srt.server_args import ServerArgs as SRTServerArgs
    from sglang.srt.utils.network import get_free_port

    max_running_requests = int(
        os.environ.get("SGLANG_SENSENOVA_THINKING_MAX_RUNNING_REQUESTS", "8")
    )
    cuda_graph_max_bs = int(
        os.environ.get("SGLANG_SENSENOVA_THINKING_CUDA_GRAPH_MAX_BS", "2")
    )
    if max_running_requests < 1 or cuda_graph_max_bs < 1:
        raise ValueError(
            "SenseNova thinking max running requests and CUDA graph max batch "
            "size must both be positive"
        )
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
            os.environ.get("SGLANG_SENSENOVA_THINKING_MEM_FRACTION", "0.45")
        ),
        max_running_requests=max_running_requests,
        cuda_graph_max_bs_decode=cuda_graph_max_bs,
    )
    url = f"http://{host}:{port}"
    status = ThinkingBackendStatus.for_url(url)
    process = mp.get_context("spawn").Process(
        target=_run_srt_server,
        args=(srt_args, status.runtime_dir, status.log_name),
        name="sensenova-srt-thinking",
    )
    server_args.srt_encoder_url = url
    pipeline_config.srt_thinking_dynamic_batching = True
    return ManagedSRTThinkingServer(process=process, url=url, status=status)
