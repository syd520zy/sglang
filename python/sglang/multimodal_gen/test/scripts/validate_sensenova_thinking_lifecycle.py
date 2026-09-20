"""Validate the SenseNova thinking backend lifecycle and its reported state.

The shell runner starts the server, breaks the internal SRT service and stops the
server again; this script owns the per-phase assertions and writes the evidence:

  info        record /server_info so the runner can find the SRT port.
  startup     the server reports `ready`, one request is served by SRT and the
              SRT log is a file of its own.
  after-kill  the first request either falls back or fails loudly (strict), one
              clear error is logged, later requests do not wait on SRT again.
"""

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "::1")

# A stopped SRT service is only detected by the read timeout, so the first
# request carries that timeout; the second must not.
REPEAT_WAIT_FRACTION = 0.5

BACKEND_ERROR_MARKERS = (
    "SenseNova thinking backend unavailable",
    "SenseNova thinking backend could not be started",
)


def resolve_base_url(raw_url, allowed_hosts):
    """Return the request origin, restricted to the declared hosts.

    The lifecycle scripts always point this at the local server they started; a
    remote host has to be named through --allowed-host.
    """
    parts = urllib.parse.urlsplit(raw_url)
    if parts.scheme not in {"http", "https"}:
        raise ValueError(
            f"--base-url scheme must be http or https, got {parts.scheme!r}"
        )
    if not parts.hostname:
        raise ValueError(f"--base-url has no host: {raw_url!r}")
    if parts.hostname not in allowed_hosts:
        raise ValueError(
            f"--base-url host {parts.hostname!r} is not allowed; "
            "add it with --allowed-host"
        )
    if parts.query or parts.fragment:
        raise ValueError(f"--base-url must not carry a query or fragment: {raw_url!r}")
    return f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"


def get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def thinking_backend_of(base_url, timeout):
    info = get_json(f"{base_url}/server_info", timeout)
    backend = info.get("thinking_backend")
    if backend is None:
        raise RuntimeError(
            f"{base_url}/server_info has no thinking_backend field; "
            "this server does not use the SenseNova thinking pipeline"
        )
    return info, backend


def image_request(args):
    """One thinking image request; returns its usage and client-side latency."""
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "n": 1,
        "response_format": "b64_json",
        "output_format": "png",
        "seed": args.seed,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "think_mode": True,
        "max_think_tokens": args.max_think_tokens,
    }
    request = urllib.request.Request(
        f"{args.base_url}/v1/images/generations",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail[:2000]}") from error
    elapsed_ms = (time.perf_counter() - started) * 1000

    usage = result.get("usage") or {}
    return {
        "elapsed_ms": round(elapsed_ms, 3),
        "thinking_backend": usage.get("thinking_backend"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
    }


def run_probe(args, label):
    """Run one request and keep its outcome; a failure is a result here."""
    try:
        result = image_request(args)
    except Exception as exc:  # noqa: BLE001 - both HTTP and transport errors are outcomes
        return {"label": label, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"label": label, "ok": True, **result}


def backend_error_lines(server_log):
    if server_log is None or not Path(server_log).exists():
        return None, []
    lines = []
    for line in (
        Path(server_log).read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        if any(marker in line for marker in BACKEND_ERROR_MARKERS):
            lines.append(line.strip())
    return str(server_log), lines


def check(checks, name, passed, evidence):
    checks[name] = {"passed": bool(passed), "evidence": evidence}


def log_evidence(args, backend):
    log_file = Path(backend["log_file"]) if backend.get("log_file") else None
    if log_file is None or not log_file.exists():
        return {"log_file": backend.get("log_file"), "exists": False, "lines": []}
    text = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    return {
        "log_file": str(log_file),
        "exists": True,
        "size_bytes": log_file.stat().st_size,
        "first_lines": [line.strip() for line in text[:3]],
    }


def phase_info(args, output_dir):
    info, backend = thinking_backend_of(args.base_url, args.timeout)
    (output_dir / "server-info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    print(json.dumps({"thinking_backend": backend}, indent=2))
    return 0


def phase_startup(args, output_dir):
    checks = {}
    _, backend = thinking_backend_of(args.base_url, args.timeout)
    probe = run_probe(args, "startup")
    _, end_backend = thinking_backend_of(args.base_url, args.timeout)
    log = log_evidence(args, end_backend)

    check(checks, "server_info_exposes_thinking_backend", True, backend)
    check(
        checks,
        "server_reports_ready",
        backend["state"] == "ready",
        backend["state"],
    )
    check(
        checks,
        "backend_is_srt",
        backend["backend"] == "srt",
        backend["backend"],
    )
    check(checks, "request_used_srt", probe.get("thinking_backend") == "srt", probe)
    check(
        checks,
        "request_still_reports_srt",
        end_backend["state"] == "ready",
        end_backend["state"],
    )
    check(checks, "srt_log_is_separate", log["exists"], log)
    check(
        checks,
        "srt_log_not_the_server_log",
        bool(log.get("log_file"))
        and Path(log["log_file"]).name
        not in {Path(args.server_log).name if args.server_log else "", "server.log"},
        log.get("log_file"),
    )
    evidence = {"probe": probe, "backend": backend, "end_backend": end_backend}
    return write_report(args, output_dir, "startup", checks, evidence)


def phase_after_kill(args, output_dir):
    checks = {}
    strict = bool(args.strict)
    _, before_backend = thinking_backend_of(args.base_url, args.timeout)
    first = run_probe(args, "first-after-failure")
    _, middle_backend = thinking_backend_of(args.base_url, args.timeout)
    second = run_probe(args, "second-after-failure")
    _, end_backend = thinking_backend_of(args.base_url, args.timeout)
    log_path, error_lines = backend_error_lines(args.server_log)

    if strict:
        check(
            checks,
            "first_request_fails_instead_of_measuring_native",
            not first["ok"],
            first.get("error"),
        )
        second_is_correct = not second["ok"]
    else:
        check(
            checks,
            "first_request_falls_back_to_native",
            first.get("ok") and first.get("thinking_backend") == "native",
            first,
        )
        second_is_correct = second.get("thinking_backend") == "native"
    check(checks, "second_request_never_reaches_srt", second_is_correct, second)
    check(
        checks,
        "state_reports_native_after_the_failure",
        end_backend["backend"] == "native",
        end_backend,
    )
    check(
        checks,
        "state_is_failed_or_fallback",
        end_backend["state"] == ("failed" if strict else "fallback"),
        end_backend["state"],
    )
    check(
        checks,
        "failure_is_reported_once",
        len(error_lines) == 1,
        {"server_log": log_path, "count": len(error_lines), "lines": error_lines[:3]},
    )
    if args.failure_mode == "stop":
        saved_ms = first.get("elapsed_ms", 0) - second.get("elapsed_ms", 0)
        check(
            checks,
            "later_requests_do_not_wait_again",
            saved_ms >= REPEAT_WAIT_FRACTION * args.srt_timeout_ms,
            {
                "first_ms": first.get("elapsed_ms"),
                "second_ms": second.get("elapsed_ms"),
                "saved_ms": round(saved_ms, 3),
                "expected_saved_ms": round(
                    REPEAT_WAIT_FRACTION * args.srt_timeout_ms, 3
                ),
            },
        )
    evidence = {
        "probes": [first, second],
        "states": {
            "before": before_backend,
            "after_first_request": middle_backend,
            "after_second_request": end_backend,
        },
        "server_log_errors": error_lines,
    }
    return write_report(args, output_dir, "after-kill", checks, evidence)


def write_report(args, output_dir, name, checks, evidence):
    passed = all(entry["passed"] for entry in checks.values())
    report = {
        "phase": name,
        "strict": bool(args.strict),
        "failure_mode": args.failure_mode,
        "base_url": args.base_url,
        "passed": passed,
        "checks": checks,
        "evidence": evidence,
    }
    (output_dir / f"lifecycle-{name}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for key, entry in checks.items():
        print(f"  [{'PASS' if entry['passed'] else 'FAIL'}] {key}: {entry['evidence']}")
    print(f"{name}: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("info", "startup", "after-kill"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--allowed-host", action="append", default=[])
    parser.add_argument("--server-log", type=Path, default=None)
    parser.add_argument("--model", default="sensenova/SenseNova-U1.5-8B-MoT")
    parser.add_argument(
        "--prompt",
        default="A realistic photo of three red apples arranged to the left of a blue bowl.",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--max-think-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--strict", type=int, choices=(0, 1), default=1)
    parser.add_argument("--failure-mode", choices=("stop", "kill"), default="stop")
    parser.add_argument(
        "--srt-timeout-ms",
        type=float,
        default=100_000,
        help="read timeout of one SRT request, in milliseconds (srt_encoder_timeout)",
    )
    args = parser.parse_args()

    allowed_hosts = tuple(DEFAULT_ALLOWED_HOSTS) + tuple(args.allowed_host)
    args.base_url = resolve_base_url(args.base_url, allowed_hosts)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "info":
        return phase_info(args, args.output_dir)
    if args.phase == "startup":
        return phase_startup(args, args.output_dir)
    return phase_after_kill(args, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
