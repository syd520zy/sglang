# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import atexit
import hashlib
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import requests

logger = logging.getLogger(__name__)


class BatchedSRTThinkingFallbackRequired(RuntimeError):
    """Ask the generation stage to preserve native fallback per request."""


_ENV_BACKEND = "SGLANG_SENSENOVA_THINKING_BACKEND"
_ENV_STRICT = "SGLANG_SENSENOVA_THINKING_STRICT"
_ENV_RUNTIME_DIR = "SGLANG_SENSENOVA_THINKING_RUNTIME_DIR"
_ENV_LOG_FILE = "SGLANG_SENSENOVA_THINKING_LOG_FILE"
_ENV_KV_DIAGNOSTIC_DIR = "SGLANG_SENSENOVA_KV_DIAGNOSTIC_DIR"
_ENV_KV_TRANSFER_DIR = "SGLANG_SENSENOVA_KV_TRANSFER_DIR"
_ENV_USE_KV_TRANSFER = "SGLANG_SENSENOVA_USE_SRT_KV_TRANSFER"
_ENV_COMPACT_MODE = "SGLANG_SENSENOVA_COMPACT_MODE"
_ENV_COMPACT_OWNS_KV_TRANSFER_DIR = "SGLANG_SENSENOVA_COMPACT_OWNS_KV_TRANSFER_DIR"
_DEFAULT_RUNTIME_DIRNAME = "sglang-sensenova-thinking"
_STARTUP_DEADLINE_S = 600.0
_KV_DIAGNOSTIC_RID_PREFIX = "sensenova-kvdiag-"
_KV_TRANSFER_RID_PREFIX = "sensenova-kvxfer-"


class ThinkingBackendState(str, Enum):
    DISABLED = "disabled"
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    FALLBACK = "fallback"
    STOPPED = "stopped"


_SRT_SERVING_STATES = (ThinkingBackendState.STARTING, ThinkingBackendState.READY)


def thinking_strict_enabled() -> bool:
    return compact_mode_enabled() or os.environ.get(
        _ENV_STRICT, ""
    ).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def compact_mode_enabled() -> bool:
    return os.environ.get(_ENV_COMPACT_MODE, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _configure_compact_mode(server_args) -> None:
    if not compact_mode_enabled():
        return
    if server_args.srt_encoder_url:
        raise ValueError(
            "SenseNova compact mode currently requires the managed local SRT backend"
        )
    if os.environ.get(_ENV_BACKEND, "srt").lower() != "srt":
        raise ValueError(f"{_ENV_COMPACT_MODE}=1 requires {_ENV_BACKEND}=srt")
    runtime_root = os.path.abspath(
        os.environ.get(_ENV_RUNTIME_DIR)
        or os.path.join(tempfile.gettempdir(), _DEFAULT_RUNTIME_DIRNAME)
    )
    os.environ[_ENV_USE_KV_TRANSFER] = "1"
    if _ENV_KV_TRANSFER_DIR not in os.environ:
        shared_root = "/dev/shm" if os.path.isdir("/dev/shm") else runtime_root
        os.environ[_ENV_KV_TRANSFER_DIR] = os.path.join(
            shared_root, f"{_DEFAULT_RUNTIME_DIRNAME}-{os.getpid()}"
        )
        os.environ[_ENV_COMPACT_OWNS_KV_TRANSFER_DIR] = "1"
    else:
        os.environ.pop(_ENV_COMPACT_OWNS_KV_TRANSFER_DIR, None)


def runtime_files(url: str) -> tuple[str, str]:
    """Return absolute status and log paths for one SRT endpoint."""
    root = os.path.abspath(
        os.environ.get(_ENV_RUNTIME_DIR)
        or os.path.join(tempfile.gettempdir(), _DEFAULT_RUNTIME_DIRNAME)
    )
    name = f"srt-thinking-{hashlib.sha256(url.encode()).hexdigest()[:12]}"
    status_file = os.path.join(root, f"{name}.json")
    log_file = os.path.abspath(
        os.environ.get(_ENV_LOG_FILE) or os.path.join(root, f"{name}.log")
    )
    return status_file, log_file


def read_thinking_status(url: str) -> dict | None:
    """The state recorded for this SRT url, or None when nothing usable exists."""
    status_file, _ = runtime_files(url)
    try:
        with open(status_file, encoding="utf-8") as handle:
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
        status_file: str | None = None,
        log_file: str | None = None,
    ):
        self.url = url
        self.strict = strict
        self.status_file = os.path.abspath(status_file) if status_file else None
        self.log_file = os.path.abspath(log_file) if log_file else None
        self.state = ThinkingBackendState.STARTING
        self.reason = ""
        self._failure_reported = False

    @classmethod
    def for_url(cls, url: str) -> ThinkingBackendStatus:
        status_file, log_file = runtime_files(url)
        status = cls(
            url,
            strict=thinking_strict_enabled(),
            status_file=status_file,
            log_file=log_file,
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
        if self.status_file is None:
            return
        recorded = read_thinking_status(self.url)
        if recorded is not None:
            self.adopt(recorded)

    def snapshot(self) -> dict:
        return {
            "state": self.state.value,
            "backend": "srt" if self.state in _SRT_SERVING_STATES else "native",
            "url": self.url,
            "strict": self.strict,
            "reason": self.reason,
            "log_file": self.log_file,
            # Which process wrote this record, so a stale one is identifiable.
            "pid": os.getpid(),
        }

    def write(self) -> None:
        if self.status_file is None:
            return
        target = self.status_file
        temporary = f"{target}.{os.getpid()}.{time.time_ns()}.tmp"
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
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
            self.log_file,
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
        if self._failure_reported:
            return False
        self._failure_reported = True
        self.state = (
            ThinkingBackendState.FAILED
            if self.strict
            else ThinkingBackendState.FALLBACK
        )
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
        self.write()
        return True

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
        if self.status.state not in _SRT_SERVING_STATES:
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

    def generate_batch_for_kv_transfer(
        self,
        input_ids: list[list[int]],
        *,
        max_think_tokens: int,
        eos_token_id: int,
        think_end_token_id: int,
    ) -> tuple[list[list[int]], dict | list[dict]]:
        """Generate responses while retaining one SRT session per KV export."""
        self.status.refresh()
        if self.status.state not in _SRT_SERVING_STATES:
            raise RuntimeError("SenseNova thinking SRT backend is unavailable")

        session_ids = []
        try:
            for _ in input_ids:
                response = requests.post(
                    f"{self.url}/open_session",
                    json={"capacity_of_str_len": 16384},
                    timeout=(self.connect_timeout, self.timeout),
                )
                response.raise_for_status()
                session_id = response.json()
                if not isinstance(session_id, str) or not session_id:
                    raise TypeError(
                        "SenseNova thinking SRT returned an invalid session id"
                    )
                session_ids.append(session_id)

            rids = [
                f"sensenova-think-{time.time_ns():x}-{index}"
                for index in range(len(input_ids))
            ]

            def generate(index):
                response = requests.post(
                    f"{self.url}/generate",
                    json={
                        "rid": rids[index],
                        "input_ids": input_ids[index],
                        "session_params": {"id": session_ids[index], "rid": None},
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
                return response.json()

            with ThreadPoolExecutor(max_workers=len(input_ids)) as executor:
                responses = list(executor.map(generate, range(len(input_ids))))
            output_ids = [
                item.get("output_ids") if isinstance(item, dict) else None
                for item in responses
            ]
            if not all(isinstance(item, list) for item in output_ids):
                raise TypeError("SenseNova thinking SRT response has no session result")
        except (requests.RequestException, ValueError, TypeError) as exc:
            for session_id in session_ids:
                self._close_session(session_id)
            self.fail(exc)
            raise

        normalized = [
            normalize_thinking_output_ids(
                item,
                max_think_tokens=max_think_tokens,
                eos_token_id=eos_token_id,
                think_end_token_id=think_end_token_id,
            )
            for item in output_ids
        ]
        contexts = [
            {
                "session_id": session_id,
                "parent_rid": rid,
                "input_ids": list(row),
                "output_ids": [int(token_id) for token_id in output],
            }
            for session_id, rid, row, output in zip(
                session_ids, rids, input_ids, output_ids
            )
        ]
        self.status.mark_ready()
        return normalized, contexts[0] if len(contexts) == 1 else contexts

    def dump_prefix_kv(self, input_ids: list[int], dump_id: str) -> None:
        """Prefill one finalized replay prefix so SRT can dump diagnostic KV."""
        if not os.environ.get(_ENV_KV_DIAGNOSTIC_DIR):
            return
        try:
            response = requests.post(
                f"{self.url}/generate",
                json={
                    "rid": f"{_KV_DIAGNOSTIC_RID_PREFIX}{dump_id}",
                    "input_ids": input_ids,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 1,
                        "skip_special_tokens": False,
                    },
                },
                timeout=(self.connect_timeout, self.timeout),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            self.fail(exc)
            raise

    def transfer_prefix_kv(
        self, input_ids: list[int], dump_id: str, session_context: dict | None = None
    ) -> dict:
        """Ask SRT to export a finalized prefix and return its transfer metadata."""
        output_dir = os.environ.get(_ENV_KV_TRANSFER_DIR)
        if not output_dir:
            self.close_transfer_context(session_context)
            raise RuntimeError(f"{_ENV_KV_TRANSFER_DIR} is not configured")
        request = {
            "rid": f"{_KV_TRANSFER_RID_PREFIX}{dump_id}",
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 1,
                "skip_special_tokens": False,
            },
        }
        session_id = None
        reused_tokens = 0
        if session_context is not None:
            session_id = session_context["session_id"]
            previous_ids = session_context["input_ids"] + session_context["output_ids"]
            reused_tokens = next(
                (
                    index
                    for index, (previous, current) in enumerate(
                        zip(previous_ids, input_ids)
                    )
                    if previous != current
                ),
                min(len(previous_ids), len(input_ids)),
            )
            if reused_tokens <= 0:
                self.close_transfer_context(session_context)
                raise RuntimeError(
                    "SenseNova SRT session does not share the expected prefix"
                )
            request["input_ids"] = input_ids[reused_tokens:]
            request["session_params"] = {
                "id": session_id,
                "rid": session_context["parent_rid"],
                "offset": reused_tokens,
            }
        try:
            response = requests.post(
                f"{self.url}/generate",
                json=request,
                timeout=(self.connect_timeout, self.timeout),
            )
            response.raise_for_status()
            response_payload = response.json()
        finally:
            self.close_transfer_context(session_context)
        metadata_path = os.path.join(output_dir, f"{dump_id}.json")
        try:
            with open(metadata_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
            meta_info = (
                response_payload.get("meta_info")
                if isinstance(response_payload, dict)
                else None
            )
            metadata["session_reused_tokens"] = reused_tokens
            metadata["request_meta_info"] = meta_info or {}
            logger.info(
                "SenseNova SRT KV session reused %d prefix tokens; SRT reported "
                "cached_tokens=%s",
                reused_tokens,
                (meta_info or {}).get("cached_tokens"),
            )
            return metadata
        except (OSError, ValueError) as exc:
            lock_path = os.path.join(output_dir, "buffer.lock")
            try:
                with open(lock_path, encoding="utf-8") as handle:
                    owns_lock = handle.read() == dump_id
                if owns_lock:
                    os.remove(lock_path)
            except OSError:
                pass
            try:
                os.remove(os.path.join(output_dir, f"{dump_id}.bin"))
            except OSError:
                pass
            raise RuntimeError(
                f"SenseNova SRT response completed without valid KV metadata: "
                f"{metadata_path}"
            ) from exc

    def close_transfer_context(self, session_context: dict | list[dict] | None) -> None:
        if not session_context:
            return
        if isinstance(session_context, list):
            for context in session_context:
                self.close_transfer_context(context)
            return
        session_id = session_context.pop("session_id", None)
        if session_id is not None:
            self._close_session(session_id)

    def _close_session(self, session_id: str) -> None:
        try:
            response = requests.post(
                f"{self.url}/close_session",
                json={"session_id": session_id},
                timeout=(self.connect_timeout, self.timeout),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("SenseNova thinking SRT session cleanup failed: %s", exc)

    def fail(self, exc: BaseException) -> bool:
        """Mark SRT unusable; True when this was its first failure on this client."""
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


def _run_srt_server(server_args, log_file: str | None = None) -> None:
    if log_file:
        # Keep the SRT runtime out of the diffusion log so both stay readable.
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handle = open(  # noqa: SIM115
            log_file, "a", encoding="utf-8", errors="replace", buffering=1
        )
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
        if os.environ.get(_ENV_COMPACT_OWNS_KV_TRANSFER_DIR) == "1":
            shutil.rmtree(os.environ.get(_ENV_KV_TRANSFER_DIR, ""), ignore_errors=True)
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
    _, log_file = runtime_files(url)
    return {
        "state": ThinkingBackendState.STARTING.value,
        "backend": "srt",
        "url": url,
        "strict": strict,
        "reason": "no state recorded yet",
        "log_file": log_file,
        "pid": os.getpid(),
    }


def prepare_managed_srt_thinking(server_args):
    """Reserve the internal text runtime before diffusion workers are spawned."""
    if not _is_sensenova_pipeline(server_args):
        return None
    _configure_compact_mode(server_args)
    pipeline_config = server_args.pipeline_config
    pipeline_config.srt_thinking_dynamic_batching = bool(server_args.srt_encoder_url)
    if server_args.srt_encoder_url:
        # A new main server must not inherit a failure left by an earlier run
        # against the same external endpoint.
        ThinkingBackendStatus.for_url(server_args.srt_encoder_url).mark_starting()
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
        args=(srt_args, status.log_file),
        name="sensenova-srt-thinking",
    )
    server_args.srt_encoder_url = url
    pipeline_config.srt_thinking_dynamic_batching = True
    return ManagedSRTThinkingServer(process=process, url=url, status=status)
