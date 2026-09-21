#!/usr/bin/env bash
# Minimal native/SRT memory decomposition. It reads checkpoint headers first,
# then samples per-process GPU memory during one request for each backend.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-sensenova/SenseNova-U1.5-8B-MoT}"
GPU_ID="${GPU_ID:-0}"
STEPS="${STEPS:-1}"
RESULT_ROOT="${OUTPUT_DIR:-/workspace/sensenova-thinking-memory/$(date +%Y%m%d-%H%M%S)}"
RESULT_ROOT="$(mkdir -p "${RESULT_ROOT}" && cd "${RESULT_ROOT}" && pwd)"

python "${SCRIPT_DIR}/analyze_sensenova_checkpoint_memory.py" \
  --model "${MODEL_PATH}" \
  --output "${RESULT_ROOT}/checkpoint-memory.json" \
  | tee "${RESULT_ROOT}/checkpoint-memory.log"

TRACE_GPU_MEMORY=1 \
OUTPUT_DIR="${RESULT_ROOT}/runtime" \
STEPS_LIST="${STEPS}" \
BUDGETS=64 \
REPEATS=1 \
CONCURRENCY="1" \
RUN_UNIT_TESTS=0 \
GPU_ID="${GPU_ID}" \
  bash "${SCRIPT_DIR}/compare_sensenova_thinking_concurrency.sh"

python "${SCRIPT_DIR}/compare_sensenova_memory_footprint.py" \
  --checkpoint "${RESULT_ROOT}/checkpoint-memory.json" \
  --native "${RESULT_ROOT}/runtime/steps-${STEPS}/native/gpu-memory-summary.json" \
  --srt "${RESULT_ROOT}/runtime/steps-${STEPS}/srt/gpu-memory-summary.json" \
  --output "${RESULT_ROOT}/memory-comparison.json" \
  | tee "${RESULT_ROOT}/memory-comparison.log"

echo "Results: ${RESULT_ROOT}"
