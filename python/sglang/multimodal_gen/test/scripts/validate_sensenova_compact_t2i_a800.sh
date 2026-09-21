#!/usr/bin/env bash
# Validate the first compact T2I stage and capture the total GPU-memory peak.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-/model/ModelScope/SenseNova/SenseNova-U1.5-8B-MoT}"
GPU_ID="${GPU_ID:-0}"
SERVER_PORT="${SERVER_PORT:-30300}"
BUDGETS="${BUDGETS:-64}"
RESULT_ROOT="${OUTPUT_DIR:-/workspace/sensenova-compact-t2i/$(date +%Y%m%d-%H%M%S)}"
RESULT_ROOT="$(mkdir -p "${RESULT_ROOT}" && cd "${RESULT_ROOT}" && pwd)"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export SGLANG_SENSENOVA_COMPACT_MODE=1
export SGLANG_SENSENOVA_THINKING_BACKEND=srt
export SGLANG_SENSENOVA_THINKING_MEM_FRACTION="${SGLANG_SENSENOVA_THINKING_MEM_FRACTION:-0.45}"
export SGLANG_SENSENOVA_KV_TRANSFER_REUSE_BUFFER=1
export SGLANG_SENSENOVA_KV_TRANSFER_MAX_TOKENS=4096
export SGLANG_SENSENOVA_THINKING_RUNTIME_DIR="${RESULT_ROOT}/thinking-runtime"

SERVER_PID=""
SAMPLER_PID=""
stop_processes() {
  if [[ -n "${SAMPLER_PID}" ]] && kill -0 "${SAMPLER_PID}" 2>/dev/null; then
    kill -INT "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    python - "${SERVER_PID}" <<'PY' || kill "${SERVER_PID}" 2>/dev/null || true
import sys

from sglang.srt.utils import kill_process_tree

kill_process_tree(int(sys.argv[1]), wait_timeout=60)
PY
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap stop_processes EXIT INT TERM
exec > >(tee -a "${RESULT_ROOT}/run.log") 2>&1
trap 'status=$?; echo "FAILED (${status}) at line ${LINENO}: ${BASH_COMMAND}"' ERR

echo "Repository: ${REPO_ROOT}"
echo "Commit: $(git rev-parse HEAD)"
echo "Model: ${MODEL_PATH}"
echo "Results: ${RESULT_ROOT}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
df -h /dev/shm

TEST_FILE=python/sglang/multimodal_gen/test/unit/test_sensenova_u1.py
python -m pytest \
  "${TEST_FILE}::test_sensenova_compact_mode_configures_strict_local_kv_transfer" \
  "${TEST_FILE}::test_sensenova_compact_mode_prunes_only_dense_language_weights" \
  "${TEST_FILE}::test_sensenova_compact_mode_rejects_it2i" \
  "${TEST_FILE}::test_sensenova_srt_replay_builds_cache_from_transfer" \
  "${TEST_FILE}::test_sensenova_srt_kv_transfer_failure_replays_prefix" \
  -q | tee "${RESULT_ROOT}/unit-tests.log"

python "${SCRIPT_DIR}/sample_gpu_process_memory.py" \
  --gpu-id "${GPU_ID}" \
  --csv "${RESULT_ROOT}/gpu-memory.csv" \
  --summary "${RESULT_ROOT}/gpu-memory-summary.json" &
SAMPLER_PID=$!

sglang serve --model-path "${MODEL_PATH}" --host 127.0.0.1 --port "${SERVER_PORT}" \
  >"${RESULT_ROOT}/server.log" 2>&1 &
SERVER_PID=$!
deadline=$((SECONDS + 3600))
until curl -fsS "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null || ((SECONDS >= deadline)); then
    tail -n 300 "${RESULT_ROOT}/server.log" >&2
    exit 1
  fi
  sleep 5
done

read -r -a budget_args <<<"${BUDGETS//,/ }"
python "${SCRIPT_DIR}/profile_sensenova_thinking.py" \
  --output-dir "${RESULT_ROOT}/profile" \
  --base-url "http://127.0.0.1:${SERVER_PORT}" \
  --model "${MODEL_PATH}" \
  --width "${WIDTH:-1024}" \
  --height "${HEIGHT:-1024}" \
  --steps "${STEPS:-1}" \
  --budgets "${budget_args[@]}" \
  --repeats "${REPEATS:-1}" \
  | tee "${RESULT_ROOT}/profile.log"

stop_processes
SERVER_PID=""
SAMPLER_PID=""

python "${SCRIPT_DIR}/inspect_sensenova_srt_runtime.py" \
  --runtime-dir "${RESULT_ROOT}/thinking-runtime" \
  --context-length 4096 \
  --max-concurrency 2 \
  --cuda-graph-max-bs 2 \
  --output "${RESULT_ROOT}/srt-runtime-summary.json" \
  | tee "${RESULT_ROOT}/srt-runtime-summary.log"
grep -n "compact T2I removed" "${RESULT_ROOT}/server.log" \
  | tee "${RESULT_ROOT}/compact-weight-release.log"
python -m json.tool "${RESULT_ROOT}/gpu-memory-summary.json"

{
  echo "commit=$(git rev-parse HEAD)"
  echo "model_path=${MODEL_PATH}"
  echo "resolution=${WIDTH:-1024}x${HEIGHT:-1024}"
  echo "steps=${STEPS:-1}"
  echo "budgets=${BUDGETS}"
  echo "repeats=${REPEATS:-1}"
  python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")'
} >"${RESULT_ROOT}/environment.txt" 2>&1

echo "Results: ${RESULT_ROOT}"
