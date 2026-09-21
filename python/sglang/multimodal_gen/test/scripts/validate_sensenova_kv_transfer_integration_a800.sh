#!/usr/bin/env bash
# Compare native T2I prefix prefill with SRT condition/uncondition KV transfer.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-/model/ModelScope/SenseNova/SenseNova-U1.5-8B-MoT}"
DEFAULT_PROMPT="A realistic photo of three red apples arranged to the left "
DEFAULT_PROMPT+="of a blue bowl."
PROMPT="${PROMPT:-${DEFAULT_PROMPT}}"
GPU_ID="${GPU_ID:-0}"
SERVER_PORT="${SERVER_PORT:-30000}"
RESULT_ROOT="${OUTPUT_DIR:-/workspace/sensenova-kv-transfer-integration/$(date +%Y%m%d-%H%M%S)}"
RESULT_ROOT="$(mkdir -p "${RESULT_ROOT}" && cd "${RESULT_ROOT}" && pwd)"
TRANSFER_ROOT="/dev/shm/sensenova-kv-integration-$$_$(date +%s)"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export SGLANG_SENSENOVA_THINKING_BACKEND=srt
export SGLANG_SENSENOVA_THINKING_STRICT=1
export SGLANG_SENSENOVA_THINKING_MEM_FRACTION="${SGLANG_SENSENOVA_THINKING_MEM_FRACTION:-0.45}"
export SGLANG_SENSENOVA_KV_TRANSFER_REUSE_BUFFER=1
export SGLANG_SENSENOVA_KV_TRANSFER_MAX_TOKENS=4096
BUDGETS="${BUDGETS:-${BUDGET:-64}}"

SERVER_PID=""
stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    python - "${SERVER_PID}" <<'PY' || kill "${SERVER_PID}" 2>/dev/null || true
import sys

from sglang.srt.utils import kill_process_tree

kill_process_tree(int(sys.argv[1]), wait_timeout=60)
PY
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
cleanup() {
  stop_server
  rm -rf -- "${TRANSFER_ROOT}"
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

TEST_FILE=python/sglang/multimodal_gen/test/unit/test_sensenova_u1.py
python -m pytest \
  "${TEST_FILE}::test_sensenova_srt_worker_streams_all_kv_layers" \
  "${TEST_FILE}::test_sensenova_srt_thinking_client_reuses_session_for_kv_transfer" \
  "${TEST_FILE}::test_sensenova_srt_replay_builds_cache_from_transfer" \
  "${TEST_FILE}::test_sensenova_srt_imports_finalized_text_prefix_without_session" \
  "${TEST_FILE}::test_sensenova_srt_text_prefix_transfer_uses_valid_tokens_only" \
  "${TEST_FILE}::test_sensenova_srt_text_prefix_transfer_failure_allows_native_fallback" \
  "${TEST_FILE}::test_sensenova_srt_kv_transfer_failure_replays_prefix" \
  "${TEST_FILE}::test_sensenova_u1_generation_stage_reports_all_transferred_prefixes" \
  -q | tee "${RESULT_ROOT}/unit-tests.log"

run_mode() {
  local mode="$1"
  local enabled="$2"
  local mode_dir="${RESULT_ROOT}/${mode}"
  mkdir -p "${mode_dir}" "${TRANSFER_ROOT}/${mode}"
  export SGLANG_SENSENOVA_USE_SRT_KV_TRANSFER="${enabled}"
  export SGLANG_SENSENOVA_KV_TRANSFER_DIR="${TRANSFER_ROOT}/${mode}"
  export SGLANG_SENSENOVA_THINKING_RUNTIME_DIR="${mode_dir}/thinking-runtime"

  sglang serve --model-path "${MODEL_PATH}" --host 127.0.0.1 --port "${SERVER_PORT}" \
    >"${mode_dir}/server.log" 2>&1 &
  SERVER_PID=$!
  local deadline=$((SECONDS + 3600))
  until curl -fsS "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null || ((SECONDS >= deadline)); then
      tail -n 200 "${mode_dir}/server.log" >&2
      exit 1
    fi
    sleep 5
  done

  read -r -a budget_args <<<"${BUDGETS//,/ }"
  python "${SCRIPT_DIR}/profile_sensenova_thinking.py" \
    --output-dir "${mode_dir}/profile" \
    --base-url "http://127.0.0.1:${SERVER_PORT}" \
    --model "${MODEL_PATH}" \
    --prompt "${PROMPT}" \
    --width "${WIDTH:-1024}" \
    --height "${HEIGHT:-1024}" \
    --steps "${STEPS:-10}" \
    --budgets "${budget_args[@]}" \
    --repeats "${REPEATS:-2}" \
    | tee "${mode_dir}/profile.log"
  stop_server
}

run_mode replay 0
run_mode transfer 1

python "${SCRIPT_DIR}/compare_sensenova_kv_transfer_integration.py" \
  --replay-dir "${RESULT_ROOT}/replay" \
  --transfer-dir "${RESULT_ROOT}/transfer" \
  --output "${RESULT_ROOT}/comparison.json" \
  | tee "${RESULT_ROOT}/comparison.log"

{
  echo "commit=$(git rev-parse HEAD)"
  echo "model_path=${MODEL_PATH}"
  echo "prompt=${PROMPT}"
  echo "resolution=${WIDTH:-1024}x${HEIGHT:-1024}"
  echo "steps=${STEPS:-10}"
  echo "budgets=${BUDGETS}"
  echo "repeats=${REPEATS:-2}"
  python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")'
  nvidia-smi
} >"${RESULT_ROOT}/environment.txt" 2>&1

echo "Results: ${RESULT_ROOT}"
