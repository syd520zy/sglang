#!/usr/bin/env bash
# Native vs SRT concurrency comparison. STEPS_LIST holds the two rounds: a short
# one that keeps diffusion out of the way, then the real end-to-end step count.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

RUNNER="${SCRIPT_DIR}/profile_sensenova_thinking_concurrency.sh"
COMPARATOR="${SCRIPT_DIR}/compare_sensenova_thinking_concurrency.py"
RESULT_ROOT="${OUTPUT_DIR:-sensenova-thinking-concurrency-compare/$(date +%Y%m%d-%H%M%S)}"
RESULT_ROOT="$(mkdir -p "${RESULT_ROOT}" && cd "${RESULT_ROOT}" && pwd)"
STEPS_LIST="${STEPS_LIST:-10 50}"
GPU_ID="${GPU_ID:-0}"
MAX_IDLE_GPU_MEMORY_MB="${MAX_IDLE_GPU_MEMORY_MB:-1024}"

gpu_memory_used_mb() {
  nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used \
    --format=csv,noheader,nounits | awk 'NR == 1 {print $1}'
}

INITIAL_GPU_MEMORY_MB="$(gpu_memory_used_mb)"
if ((INITIAL_GPU_MEMORY_MB > MAX_IDLE_GPU_MEMORY_MB)); then
  echo "GPU ${GPU_ID} already uses ${INITIAL_GPU_MEMORY_MB} MiB; " \
    "the paired benchmark requires an idle GPU (limit ${MAX_IDLE_GPU_MEMORY_MB} MiB)." >&2
  nvidia-smi --id="${GPU_ID}" >&2
  exit 1
fi

wait_for_gpu_memory_release() {
  local limit_mb=$((INITIAL_GPU_MEMORY_MB + 512))
  local deadline=$((SECONDS + 180))
  local used_mb
  while ((SECONDS < deadline)); do
    used_mb="$(gpu_memory_used_mb)"
    if ((used_mb <= limit_mb)); then
      return
    fi
    sleep 2
  done
  echo "GPU ${GPU_ID} still uses ${used_mb} MiB after the previous stage; " \
    "refusing to start a contaminated run." >&2
  nvidia-smi --id="${GPU_ID}" >&2
  exit 1
}

for steps in ${STEPS_LIST}; do
  echo "Round steps=${steps}: native..."
  SGLANG_SENSENOVA_THINKING_BACKEND=native \
  SGLANG_SENSENOVA_THINKING_STRICT=0 \
  STEPS="${steps}" \
  OUTPUT_DIR="${RESULT_ROOT}/steps-${steps}/native" \
    bash "${RUNNER}" "$@"

  wait_for_gpu_memory_release

  echo "Round steps=${steps}: SRT..."
  SGLANG_SENSENOVA_THINKING_BACKEND=srt \
  SGLANG_SENSENOVA_THINKING_STRICT=1 \
  SGLANG_SENSENOVA_THINKING_MEM_FRACTION="${SGLANG_SENSENOVA_THINKING_MEM_FRACTION:-0.45}" \
  STEPS="${steps}" \
  OUTPUT_DIR="${RESULT_ROOT}/steps-${steps}/srt" \
    bash "${RUNNER}" "$@"

  wait_for_gpu_memory_release

  python "${COMPARATOR}" \
    --native-dir "${RESULT_ROOT}/steps-${steps}/native" \
    --srt-dir "${RESULT_ROOT}/steps-${steps}/srt" \
    --output "${RESULT_ROOT}/steps-${steps}/comparison.json" \
    | tee "${RESULT_ROOT}/steps-${steps}/comparison.log"
done

echo "Results: ${RESULT_ROOT}"
