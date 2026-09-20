#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

RUNNER="${SCRIPT_DIR}/profile_sensenova_thinking_gpu.sh"
COMPARATOR="${SCRIPT_DIR}/compare_sensenova_thinking_profiles.py"
RESULT_ROOT="${OUTPUT_DIR:-sensenova-thinking-srt-a800-results/$(date +%Y%m%d-%H%M%S)}"
RESULT_ROOT="$(mkdir -p "${RESULT_ROOT}" && cd "${RESULT_ROOT}" && pwd)"

echo "Running native baseline..."
SGLANG_SENSENOVA_THINKING_BACKEND=native \
OUTPUT_DIR="${RESULT_ROOT}/native" \
  bash "${RUNNER}" "$@"

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
