#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SENSENOVA_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${SENSENOVA_REPO_ROOT}"
export PYTHONPATH="${SENSENOVA_REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-sensenova/SenseNova-U1.5-8B-MoT}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-30000}"
GPU_ID="${GPU_ID:-0}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
MAX_THINK_TOKENS="${MAX_THINK_TOKENS:-64}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-1800}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-3600}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-sensenova-thinking-results/$(date +%Y%m%d-%H%M%S)}"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
export OUTPUT_DIR WIDTH HEIGHT MAX_THINK_TOKENS REQUEST_TIMEOUT_SECONDS
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SERVER_PID=""
MONITOR_PID=""

cleanup() {
  if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

{
  echo "git_commit=$(git rev-parse HEAD)"
  echo "model_path=${MODEL_PATH}"
  echo "gpu_id=${GPU_ID}"
  echo "resolution=${WIDTH}x${HEIGHT}"
  echo "max_think_tokens=${MAX_THINK_TOKENS}"
  python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'}")
PY
  nvidia-smi
} >"${OUTPUT_DIR}/environment.txt" 2>&1

if [[ "${RUN_UNIT_TESTS}" == "1" ]]; then
  python -m pytest \
    python/sglang/multimodal_gen/test/unit/test_sensenova_u1.py \
    -q | tee "${OUTPUT_DIR}/unit-tests.log"
fi

nvidia-smi \
  --query-gpu=timestamp,memory.used,utilization.gpu \
  --format=csv \
  -l 1 >"${OUTPUT_DIR}/gpu.csv" 2>&1 &
MONITOR_PID=$!

SERVER_COMMAND=(
  sglang serve
  --model-path "${MODEL_PATH}"
  --host "${SERVER_HOST}"
  --port "${SERVER_PORT}"
)
SERVER_COMMAND+=("$@")
printf '%q ' "${SERVER_COMMAND[@]}" >"${OUTPUT_DIR}/server-command.txt"
printf '\n' >>"${OUTPUT_DIR}/server-command.txt"

"${SERVER_COMMAND[@]}" >"${OUTPUT_DIR}/server.log" 2>&1 &
SERVER_PID=$!

BASE_URL="http://${SERVER_HOST}:${SERVER_PORT}"
deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
until curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "SenseNova server exited before becoming healthy." >&2
    tail -n 200 "${OUTPUT_DIR}/server.log" >&2
    exit 1
  fi
  if ((SECONDS >= deadline)); then
    echo "SenseNova server did not become healthy within ${STARTUP_TIMEOUT_SECONDS}s." >&2
    tail -n 200 "${OUTPUT_DIR}/server.log" >&2
    exit 1
  fi
  sleep 5
done

export BASE_URL
python - <<'PY' | tee "${OUTPUT_DIR}/validation.log"
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

base_url = os.environ["BASE_URL"]
output_dir = Path(os.environ["OUTPUT_DIR"])
width = int(os.environ["WIDTH"])
height = int(os.environ["HEIGHT"])
max_think_tokens = int(os.environ["MAX_THINK_TOKENS"])
timeout = int(os.environ["REQUEST_TIMEOUT_SECONDS"])


def generate(name, **thinking_options):
    payload = {
        "prompt": (
            "A clean technology poster about renewable energy, structured layout, "
            "blue and green palette, clear title Green Future."
        ),
        "width": width,
        "height": height,
        "n": 1,
        "response_format": "b64_json",
        "seed": 42,
        **thinking_options,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/images/generations",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(f"{name} request failed: HTTP {error.code}: {body}") from error
    elapsed = time.perf_counter() - start

    image_b64 = result["data"][0].pop("b64_json")
    image_path = output_dir / f"{name}.png"
    image_path.write_bytes(base64.b64decode(image_b64))
    result["data"][0]["saved_image"] = str(image_path)
    result["elapsed_seconds"] = elapsed
    (output_dir / f"{name}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result, elapsed


baseline, baseline_seconds = generate("baseline")
thinking, thinking_seconds = generate(
    "thinking",
    think_mode=True,
    max_think_tokens=max_think_tokens,
)

baseline_usage = baseline.get("usage") or {}
assert not baseline_usage.get("think_text"), baseline_usage

thinking_usage = thinking.get("usage") or {}
think_text = thinking_usage.get("think_text")
reasoning_tokens = thinking_usage.get("reasoning_tokens")
assert isinstance(think_text, str) and think_text.endswith("</think>"), thinking_usage
assert isinstance(reasoning_tokens, int), thinking_usage
assert 1 <= reasoning_tokens <= max_think_tokens, thinking_usage

summary = {
    "resolution": f"{width}x{height}",
    "max_think_tokens": max_think_tokens,
    "reasoning_tokens": reasoning_tokens,
    "baseline_seconds": baseline_seconds,
    "thinking_seconds": thinking_seconds,
    "thinking_overhead_seconds": thinking_seconds - baseline_seconds,
    "thinking_overhead_ratio": thinking_seconds / baseline_seconds,
    "think_text": think_text,
}
(output_dir / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"Validation passed. Results: {output_dir}")
PY
