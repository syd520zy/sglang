#!/usr/bin/env bash
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
STEPS="${STEPS:-50}"
REPEATS="${REPEATS:-2}"
BUDGETS="${BUDGETS:-64,128,256}"
PROMPT="${PROMPT:-A realistic photo of three red apples arranged to the left of a blue bowl.}"
OUTPUT_DIR="${OUTPUT_DIR:-sensenova-thinking-profile-results/$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"
PROFILER="${SCRIPT_DIR}/profile_sensenova_thinking.py"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

{
  echo "sglang_commit=$(git rev-parse HEAD)"
  echo "model_path=${MODEL_PATH}"
  echo "resolution=${WIDTH}x${HEIGHT}"
  echo "steps=${STEPS}"
  echo "repeats=${REPEATS}"
  echo "budgets=${BUDGETS}"
  python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")'
  nvidia-smi
} >"${OUTPUT_DIR}/environment.txt" 2>&1

python -m pytest python/sglang/multimodal_gen/test/unit/test_sensenova_u1.py -q \
  | tee "${OUTPUT_DIR}/unit-tests.log"

sglang serve --model-path "${MODEL_PATH}" --host 127.0.0.1 --port "${SERVER_PORT}" "$@" \
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
python "${PROFILER}" \
  --output-dir "${OUTPUT_DIR}/profile" \
  --base-url "http://127.0.0.1:${SERVER_PORT}" \
  --model "${MODEL_PATH}" \
  --prompt "${PROMPT}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --steps "${STEPS}" \
  --repeats "${REPEATS}" \
  --budgets "${BUDGET_ARGS[@]}" \
  | tee "${OUTPUT_DIR}/profile.log"

echo "Results: ${OUTPUT_DIR}"
