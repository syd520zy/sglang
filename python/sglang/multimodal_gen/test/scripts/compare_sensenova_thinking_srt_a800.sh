#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

RUNNER="${SCRIPT_DIR}/profile_sensenova_thinking_gpu.sh"
COMPARATOR="${SCRIPT_DIR}/compare_sensenova_thinking_profiles.py"
RESULT_ROOT="${OUTPUT_DIR:-sensenova-thinking-srt-a800-results/$(date +%Y%m%d-%H%M%S)}"
RESULT_ROOT="$(mkdir -p "${RESULT_ROOT}" && cd "${RESULT_ROOT}" && pwd)"
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
  echo "GPU ${GPU_ID} still uses ${used_mb} MiB after the native server stopped; " \
    "refusing to start a contaminated SRT run." >&2
  nvidia-smi --id="${GPU_ID}" >&2
  exit 1
}

echo "Running native baseline..."
SGLANG_SENSENOVA_THINKING_BACKEND=native \
OUTPUT_DIR="${RESULT_ROOT}/native" \
  bash "${RUNNER}" "$@"

wait_for_gpu_memory_release

echo "Running SRT optimized path..."
SGLANG_SENSENOVA_THINKING_BACKEND=srt \
SGLANG_SENSENOVA_THINKING_MEM_FRACTION="${SGLANG_SENSENOVA_THINKING_MEM_FRACTION:-0.45}" \
OUTPUT_DIR="${RESULT_ROOT}/srt" \
  bash "${RUNNER}" "$@"

python "${COMPARATOR}" \
  --native-dir "${RESULT_ROOT}/native" \
  --srt-dir "${RESULT_ROOT}/srt" \
  --output "${RESULT_ROOT}/comparison.json" \
  | tee "${RESULT_ROOT}/comparison.log"

echo "Results: ${RESULT_ROOT}"
