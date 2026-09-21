#!/usr/bin/env bash
# Compare layer-0 replay KV from the native model and managed SRT on one request.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-/model/ModelScope/SenseNova/SenseNova-U1.5-8B-MoT}"
GPU_ID="${GPU_ID:-0}"
SERVER_PORT="${SERVER_PORT:-30000}"
RESULT_ROOT="${OUTPUT_DIR:-/workspace/sensenova-kv-compatibility/$(date +%Y%m%d-%H%M%S)}"
RESULT_ROOT="$(mkdir -p "${RESULT_ROOT}" && cd "${RESULT_ROOT}" && pwd)"
KV_DIR="${RESULT_ROOT}/kv"
mkdir -p "${KV_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export SGLANG_SENSENOVA_THINKING_BACKEND=srt
export SGLANG_SENSENOVA_THINKING_STRICT=1
export SGLANG_SENSENOVA_THINKING_MEM_FRACTION="${SGLANG_SENSENOVA_THINKING_MEM_FRACTION:-0.45}"
export SGLANG_SENSENOVA_KV_DIAGNOSTIC_DIR="${KV_DIR}"
export SGLANG_SENSENOVA_THINKING_RUNTIME_DIR="${RESULT_ROOT}/thinking-runtime"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    python - "${SERVER_PID}" <<'PY' || kill "${SERVER_PID}" 2>/dev/null || true
import sys

from sglang.srt.utils import kill_process_tree

kill_process_tree(int(sys.argv[1]), wait_timeout=60)
PY
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
exec > >(tee -a "${RESULT_ROOT}/run.log") 2>&1
trap 'status=$?; echo "FAILED (${status}) at line ${LINENO}: ${BASH_COMMAND}"' ERR

echo "Repository: ${REPO_ROOT}"
echo "Commit: $(git rev-parse HEAD)"
echo "Model: ${MODEL_PATH}"
echo "Results: ${RESULT_ROOT}"

sglang serve --model-path "${MODEL_PATH}" --host 127.0.0.1 --port "${SERVER_PORT}" \
  >"${RESULT_ROOT}/server.log" 2>&1 &
SERVER_PID=$!
deadline=$((SECONDS + 3600))
until curl -fsS "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null || ((SECONDS >= deadline)); then
    tail -n 200 "${RESULT_ROOT}/server.log" >&2
    exit 1
  fi
  sleep 5
done

python "${SCRIPT_DIR}/profile_sensenova_thinking.py" \
  --output-dir "${RESULT_ROOT}/profile" \
  --base-url "http://127.0.0.1:${SERVER_PORT}" \
  --model "${MODEL_PATH}" \
  --width 1024 \
  --height 1024 \
  --steps 1 \
  --budgets 64 \
  --repeats 1 \
  --skip-warmup \
  | tee "${RESULT_ROOT}/profile.log"

python "${SCRIPT_DIR}/compare_sensenova_kv_diagnostic.py" \
  --directory "${KV_DIR}" \
  --output "${RESULT_ROOT}/comparison.json" \
  | tee "${RESULT_ROOT}/comparison.log"

echo "Results: ${RESULT_ROOT}"
