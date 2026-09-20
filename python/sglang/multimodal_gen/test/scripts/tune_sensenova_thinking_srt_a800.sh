#!/usr/bin/env bash
# Compare the small P1-2 tuning matrix for the target concurrency 1/2 workload.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

RUNNER="${SCRIPT_DIR}/profile_sensenova_thinking_concurrency.sh"
COMPARATOR="${SCRIPT_DIR}/compare_sensenova_srt_tuning.py"
RESULT_ROOT="${OUTPUT_DIR:-sensenova-thinking-srt-tuning/$(date +%Y%m%d-%H%M%S)}"
RESULT_ROOT="$(mkdir -p "${RESULT_ROOT}" && cd "${RESULT_ROOT}" && pwd)"
GRAPH_BATCH_SIZES="${GRAPH_BATCH_SIZES:-1 2}"
MEM_FRACTIONS="${MEM_FRACTIONS:-0.45 0.50}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"
GPU_ID="${GPU_ID:-0}"
MAX_IDLE_GPU_MEMORY_MB="${MAX_IDLE_GPU_MEMORY_MB:-1024}"

initial_memory_mb="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR == 1 {print $1}')"
if ((initial_memory_mb > MAX_IDLE_GPU_MEMORY_MB)); then
  echo "GPU ${GPU_ID} already uses ${initial_memory_mb} MiB; an idle GPU is required." >&2
  exit 1
fi
wait_for_gpu_memory_release() {
  local deadline=$((SECONDS + 180))
  local used_mb
  while ((SECONDS < deadline)); do
    used_mb="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR == 1 {print $1}')"
    if ((used_mb <= initial_memory_mb + 512)); then
      return
    fi
    sleep 2
  done
  echo "GPU ${GPU_ID} still uses ${used_mb} MiB after the previous run." >&2
  exit 1
}

for graph_bs in ${GRAPH_BATCH_SIZES}; do
  for mem_fraction in ${MEM_FRACTIONS}; do
    run_dir="${RESULT_ROOT}/graph-${graph_bs}-mem-${mem_fraction}-running-${MAX_RUNNING_REQUESTS}"
    mkdir -p "${run_dir}"
    echo "SRT tuning: graph_bs=${graph_bs}, mem=${mem_fraction}, max_running=${MAX_RUNNING_REQUESTS}"
    if SGLANG_SENSENOVA_THINKING_BACKEND=srt \
      SGLANG_SENSENOVA_THINKING_STRICT=1 \
      SGLANG_SENSENOVA_THINKING_CUDA_GRAPH_MAX_BS="${graph_bs}" \
      SGLANG_SENSENOVA_THINKING_MEM_FRACTION="${mem_fraction}" \
      SGLANG_SENSENOVA_THINKING_MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}" \
      STEPS="${STEPS:-10}" \
      BUDGETS="${BUDGETS:-64}" \
      REPEATS="${REPEATS:-2}" \
      CONCURRENCY="1 2" \
      OUTPUT_DIR="${run_dir}" \
        bash "${RUNNER}" "$@"; then
      echo passed >"${run_dir}/tuning-status.txt"
    else
      echo failed >"${run_dir}/tuning-status.txt"
    fi
    wait_for_gpu_memory_release
  done
done

python "${COMPARATOR}" \
  --root "${RESULT_ROOT}" \
  --output "${RESULT_ROOT}/tuning-comparison.json" \
  | tee "${RESULT_ROOT}/tuning-comparison.log"

if [[ "${RUN_MAX_RESOLUTION:-1}" == "1" ]]; then
  read -r best_graph_bs best_mem_fraction best_max_running < <(
    python - "${RESULT_ROOT}/tuning-comparison.json" <<'PY'
import json
import sys

best = json.load(open(sys.argv[1], encoding="utf-8"))["best"]
print(best["cuda_graph_max_bs"], best["mem_fraction"], best["max_running_requests"])
PY
  )
  SGLANG_SENSENOVA_THINKING_BACKEND=srt \
  SGLANG_SENSENOVA_THINKING_STRICT=1 \
  SGLANG_SENSENOVA_THINKING_CUDA_GRAPH_MAX_BS="${best_graph_bs}" \
  SGLANG_SENSENOVA_THINKING_MEM_FRACTION="${best_mem_fraction}" \
  SGLANG_SENSENOVA_THINKING_MAX_RUNNING_REQUESTS="${best_max_running}" \
  WIDTH="${MAX_RESOLUTION_WIDTH:-4096}" \
  HEIGHT="${MAX_RESOLUTION_HEIGHT:-4096}" \
  STEPS="${MAX_RESOLUTION_STEPS:-10}" \
  BUDGETS="${BUDGETS:-64}" \
  REPEATS=1 \
  CONCURRENCY=1 \
  OUTPUT_DIR="${RESULT_ROOT}/max-resolution" \
    bash "${RUNNER}" "$@"
  wait_for_gpu_memory_release
fi

echo "Results: ${RESULT_ROOT}"
