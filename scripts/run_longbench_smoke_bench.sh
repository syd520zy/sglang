#!/usr/bin/env bash
set -euo pipefail

# LongBench smoke benchmark for quick validation on smaller GPUs (e.g. RTX4090).
# This script uses bench_serving custom dataset mode.
#
# Default strategy:
# - keep only shorter samples by context cap (--sharegpt-context-len)
# - use small output length (--sharegpt-output-len)
# - low concurrency and small prompt count
#
# Required:
#   LONG_DATASET_PATH: path to a custom JSONL generated from LongBench.
#
# Optional:
#   BASE_URL=http://127.0.0.1:30000
#   MODEL_PATH=/model/ModelScope/Qwen/Qwen3-8B
#   BENCH_BACKEND=sglang
#   OUT_DIR=/workspace/bench_results/longbench_smoke_$(date +%Y%m%d_%H%M%S)
#   ROUNDS="1"
#   CONCURRENCY_LIST="1 2 4"
#   PROMPTS_PER_CONCURRENCY=4
#   SMOKE_CONTEXT_LEN=4096
#   SMOKE_OUTPUT_LEN=32
#   SEED_BASE=20260413

BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
MODEL_PATH="${MODEL_PATH:-/model/ModelScope/Qwen/Qwen3-8B}"
BENCH_BACKEND="${BENCH_BACKEND:-sglang}"
OUT_DIR="${OUT_DIR:-/workspace/bench_results/longbench_smoke_$(date +%Y%m%d_%H%M%S)}"

LONG_DATASET_PATH="${LONG_DATASET_PATH:-}"
ROUNDS=(${ROUNDS:-1})
CONCURRENCY_LIST=(${CONCURRENCY_LIST:-1 2 4})
PROMPTS_PER_CONCURRENCY="${PROMPTS_PER_CONCURRENCY:-4}"
SMOKE_CONTEXT_LEN="${SMOKE_CONTEXT_LEN:-4096}"
SMOKE_OUTPUT_LEN="${SMOKE_OUTPUT_LEN:-32}"
SEED_BASE="${SEED_BASE:-20260413}"

if [[ -z "${LONG_DATASET_PATH}" ]]; then
  echo "[ERROR] LONG_DATASET_PATH is required."
  echo "Example:"
  echo "  LONG_DATASET_PATH=/path/to/longbench_custom.jsonl bash scripts/run_longbench_smoke_bench.sh"
  exit 1
fi

if [[ ! -f "${LONG_DATASET_PATH}" ]]; then
  echo "[ERROR] LONG_DATASET_PATH not found: ${LONG_DATASET_PATH}"
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "BASE_URL=${BASE_URL}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "BENCH_BACKEND=${BENCH_BACKEND}"
echo "LONG_DATASET_PATH=${LONG_DATASET_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "ROUNDS=${ROUNDS[*]}"
echo "CONCURRENCY_LIST=${CONCURRENCY_LIST[*]}"
echo "PROMPTS_PER_CONCURRENCY=${PROMPTS_PER_CONCURRENCY}"
echo "SMOKE_CONTEXT_LEN=${SMOKE_CONTEXT_LEN}"
echo "SMOKE_OUTPUT_LEN=${SMOKE_OUTPUT_LEN}"

run_one_case() {
  local round="$1"
  local conc="$2"
  local num_prompts=$((conc * PROMPTS_PER_CONCURRENCY))
  local seed=$((SEED_BASE + round * 1000 + conc))
  local out_file="${OUT_DIR}/round${round}_long_c${conc}.jsonl"
  local log_file="${OUT_DIR}/round${round}_long_c${conc}.log"

  echo "=== round=${round} longbench_smoke c=${conc} num_prompts=${num_prompts} ==="

  python3 -m sglang.bench_serving \
    --backend "${BENCH_BACKEND}" \
    --base-url "${BASE_URL}" \
    --model "${MODEL_PATH}" \
    --tokenizer "${MODEL_PATH}" \
    --dataset-name custom \
    --dataset-path "${LONG_DATASET_PATH}" \
    --sharegpt-context-len "${SMOKE_CONTEXT_LEN}" \
    --sharegpt-output-len "${SMOKE_OUTPUT_LEN}" \
    --num-prompts "${num_prompts}" \
    --request-rate inf \
    --max-concurrency "${conc}" \
    --seed "${seed}" \
    --warmup-requests 1 \
    --output-file "${out_file}" \
    --disable-tqdm \
    | tee "${log_file}"
}

for round in "${ROUNDS[@]}"; do
  for conc in "${CONCURRENCY_LIST[@]}"; do
    run_one_case "${round}" "${conc}"
  done
done

echo "Smoke benchmark finished."
echo "Raw results: ${OUT_DIR}"
echo "Summary:"
python3 scripts/summarize_bench_serving_results.py \
  --input-dir "${OUT_DIR}" \
  --output-csv "${OUT_DIR}/summary.csv"

