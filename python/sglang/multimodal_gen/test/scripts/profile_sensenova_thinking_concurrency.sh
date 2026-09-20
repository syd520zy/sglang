#!/usr/bin/env bash
# One thinking backend: concurrency wave benchmark. Knobs: STEPS, CONCURRENCY,
# BUDGETS, REPEATS, BATCHING_MAX_SIZE, BATCHING_DELAY_MS, BATCHING_METRICS.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-sensenova/SenseNova-U1.5-8B-MoT}"
GPU_ID="${GPU_ID:-0}"
SERVER_PORT="${SERVER_PORT:-30000}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
STEPS="${STEPS:-10}"
REPEATS="${REPEATS:-2}"
BUDGETS="${BUDGETS:-64,128,256}"
CONCURRENCY="${CONCURRENCY:-1 2}"
PROMPT="${PROMPT:-A realistic photo of three red apples arranged to the left of a blue bowl.}"
OUTPUT_DIR="${OUTPUT_DIR:-sensenova-thinking-concurrency-results/$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"
PROFILER="${SCRIPT_DIR}/profile_sensenova_thinking_concurrency.py"

THINKING_BACKEND="${SGLANG_SENSENOVA_THINKING_BACKEND:-srt}"
BATCHING_MAX_SIZE="${BATCHING_MAX_SIZE:-2}"
BATCHING_DELAY_MS="${BATCHING_DELAY_MS:-50}"
BATCHING_METRICS="${BATCHING_METRICS:-1}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SGLANG_SENSENOVA_THINKING_RUNTIME_DIR="${OUTPUT_DIR}/thinking-runtime"

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
  SERVER_PID=""
}
trap cleanup EXIT INT TERM

{
  echo "sglang_commit=$(git rev-parse HEAD)"
  echo "model_path=${MODEL_PATH}"
  echo "resolution=${WIDTH}x${HEIGHT}"
  echo "steps=${STEPS}"
  echo "repeats=${REPEATS}"
  echo "budgets=${BUDGETS}"
  echo "concurrency=${CONCURRENCY}"
  echo "thinking_backend=${THINKING_BACKEND}"
  echo "thinking_strict=${SGLANG_SENSENOVA_THINKING_STRICT:-0}"
  echo "batching_max_size=${BATCHING_MAX_SIZE}"
  echo "batching_delay_ms=${BATCHING_DELAY_MS}"
  echo "batching_metrics=${BATCHING_METRICS}"
  echo "thinking_mem_fraction=${SGLANG_SENSENOVA_THINKING_MEM_FRACTION:-0.45}"
  echo "thinking_max_running_requests=${SGLANG_SENSENOVA_THINKING_MAX_RUNNING_REQUESTS:-8}"
  echo "thinking_cuda_graph_max_bs=${SGLANG_SENSENOVA_THINKING_CUDA_GRAPH_MAX_BS:-2}"
  echo "thinking_context_length=4096"
  echo "thinking_runtime_dir=${SGLANG_SENSENOVA_THINKING_RUNTIME_DIR}"
  python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")'
  nvidia-smi
} >"${OUTPUT_DIR}/environment.txt" 2>&1

if [[ "${RUN_UNIT_TESTS}" == "1" ]]; then
  python -m pytest python/sglang/multimodal_gen/test/unit/test_sensenova_u1.py -q \
    | tee "${OUTPUT_DIR}/unit-tests.log"
fi

SERVER_ARGS=(
  --batching-max-size "${BATCHING_MAX_SIZE}"
  --batching-delay-ms "${BATCHING_DELAY_MS}"
)
if [[ "${BATCHING_METRICS}" == "1" ]]; then
  SERVER_ARGS+=(--enable-batching-metrics)
fi

sglang serve --model-path "${MODEL_PATH}" --host 127.0.0.1 --port "${SERVER_PORT}" \
  "${SERVER_ARGS[@]}" "$@" \
  >"${OUTPUT_DIR}/server.log" 2>&1 &
SERVER_PID=$!
deadline=$((SECONDS + 3600))
until curl -fsS "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null || ((SECONDS >= deadline)); then
    tail -n 150 "${OUTPUT_DIR}/server.log" >&2
    exit 1
  fi
  sleep 5
done

read -r -a BUDGET_ARGS <<<"${BUDGETS//,/ }"
read -r -a CONCURRENCY_ARGS <<<"${CONCURRENCY}"
python "${PROFILER}" \
  --output-dir "${OUTPUT_DIR}" \
  --base-url "http://127.0.0.1:${SERVER_PORT}" \
  --model "${MODEL_PATH}" \
  --prompt "${PROMPT}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --steps "${STEPS}" \
  --repeats "${REPEATS}" \
  --budgets "${BUDGET_ARGS[@]}" \
  --concurrency "${CONCURRENCY_ARGS[@]}" \
  --expected-backend "${THINKING_BACKEND}" \
  | tee "${OUTPUT_DIR}/profile.log"

# Dispatch-level evidence for whether the diffusion pipeline merged a wave.
grep -nE "Dynamic batch stats|Batch admission enabled" "${OUTPUT_DIR}/server.log" \
  >"${OUTPUT_DIR}/batch-metrics.log" || true

if [[ "${THINKING_BACKEND}" == "srt" ]]; then
  MAX_CONCURRENCY="$(printf '%s\n' ${CONCURRENCY} | sort -n | tail -n 1)"
  python "${SCRIPT_DIR}/inspect_sensenova_srt_runtime.py" \
    --runtime-dir "${SGLANG_SENSENOVA_THINKING_RUNTIME_DIR}" \
    --context-length 4096 \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --cuda-graph-max-bs "${SGLANG_SENSENOVA_THINKING_CUDA_GRAPH_MAX_BS:-2}" \
    --output "${OUTPUT_DIR}/srt-runtime-summary.json" \
    | tee "${OUTPUT_DIR}/srt-runtime-summary.log"
fi

echo "Results: ${OUTPUT_DIR}"
