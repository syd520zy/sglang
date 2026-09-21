#!/usr/bin/env bash
# Benchmark shared-memory KV handoff from the live managed SenseNova SRT pool.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-/model/ModelScope/SenseNova/SenseNova-U1.5-8B-MoT}"
GPU_ID="${GPU_ID:-0}"
SERVER_PORT="${SERVER_PORT:-30000}"
RESULT_ROOT="${OUTPUT_DIR:-/workspace/sensenova-kv-transfer-real/$(date +%Y%m%d-%H%M%S)}"
RESULT_ROOT="$(mkdir -p "${RESULT_ROOT}" && cd "${RESULT_ROOT}" && pwd)"
TRANSFER_DIR="/dev/shm/sensenova-kv-transfer-$$_$(date +%s)"
mkdir -p "${TRANSFER_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export SGLANG_SENSENOVA_THINKING_BACKEND=srt
export SGLANG_SENSENOVA_THINKING_STRICT=1
export SGLANG_SENSENOVA_THINKING_MEM_FRACTION="${SGLANG_SENSENOVA_THINKING_MEM_FRACTION:-0.45}"
export SGLANG_SENSENOVA_THINKING_RUNTIME_DIR="${RESULT_ROOT}/thinking-runtime"
export SGLANG_SENSENOVA_KV_TRANSFER_DIR="${TRANSFER_DIR}"

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
  rm -rf -- "${TRANSFER_DIR}"
}
trap cleanup EXIT INT TERM
exec > >(tee -a "${RESULT_ROOT}/run.log") 2>&1
trap 'status=$?; echo "FAILED (${status}) at line ${LINENO}: ${BASH_COMMAND}"' ERR

echo "Repository: ${REPO_ROOT}"
echo "Commit: $(git rev-parse HEAD)"
echo "Model: ${MODEL_PATH}"
echo "Results: ${RESULT_ROOT}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
df -h /dev/shm

python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_sensenova_u1.py::test_sensenova_srt_worker_streams_all_kv_layers \
  -q | tee "${RESULT_ROOT}/unit-test.log"

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

python "${SCRIPT_DIR}/benchmark_sensenova_kv_transfer_real.py" \
  --base-url "http://127.0.0.1:${SERVER_PORT}" \
  --transfer-dir "${TRANSFER_DIR}" \
  --tokens "${TOKENS:-256,512,1024}" \
  --repeats "${REPEATS:-2}" \
  --replay-reference-ms "${REPLAY_REFERENCE_MS:-164}" \
  --output "${RESULT_ROOT}/results.json"

{
  echo "commit=$(git rev-parse HEAD)"
  echo "model_path=${MODEL_PATH}"
  echo "tokens=${TOKENS:-256,512,1024}"
  echo "repeats=${REPEATS:-2}"
  python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")'
  nvidia-smi
} >"${RESULT_ROOT}/environment.txt" 2>&1

echo "Results: ${RESULT_ROOT}"
