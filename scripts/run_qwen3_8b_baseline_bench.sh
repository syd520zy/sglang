#!/usr/bin/env bash
set -euo pipefail

# Qwen3-8B baseline benchmark matrix:
# - 3 rounds
# - short context: random dataset, 1024 in / 100 out
# - long context: random dataset, 16000 in / 512 out (24GB safe profile)
# - concurrency: 1, 4, 8, 16
# - num_prompts:
#   - concurrency=1 -> 30
#   - concurrency in {4,8,16} -> 120
#
# Prereq:
# 1) sglang server is already running at BASE_URL
# 2) model path exists and is the served model
#
# Usage:
#   bash scripts/run_qwen3_8b_baseline_bench.sh
# Optional env overrides:
#   BASE_URL=http://127.0.0.1:30000
#   MODEL_PATH=/model/ModelScope/Qwen/Qwen3-8B
#   OUT_DIR=/workspace/bench_results/qwen3_8b_baseline
#   BENCH_BACKEND=sglang

BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
MODEL_PATH="${MODEL_PATH:-/model/ModelScope/Qwen/Qwen3-8B}"
OUT_DIR="${OUT_DIR:-/workspace/bench_results/qwen3_8b_baseline}"
BENCH_BACKEND="${BENCH_BACKEND:-sglang}"

CONCURRENCY_LIST=(1 4 8 16)
ROUNDS=(1 2 3)

SHORT_INPUT_LEN=1024
SHORT_OUTPUT_LEN=100

# Long-case 24GB-safe profile on random dataset.
LONG_INPUT_LEN=16000
LONG_OUTPUT_LEN=512

SEED_BASE=20260412

mkdir -p "${OUT_DIR}"

echo "BASE_URL=${BASE_URL}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "BENCH_BACKEND=${BENCH_BACKEND}"

run_short_case() {
  local round="$1"
  local conc="$2"
  local num_prompts
  if [[ "${conc}" -eq 1 ]]; then
    num_prompts=30
  else
    num_prompts=120
  fi
  local seed=$((SEED_BASE + round * 100 + conc))
  local out_file="${OUT_DIR}/round${round}_short_c${conc}.jsonl"
  local log_file="${OUT_DIR}/round${round}_short_c${conc}.log"

  echo "=== round=${round} short c=${conc} num_prompts=${num_prompts} ==="
  python3 -m sglang.bench_serving \
    --backend "${BENCH_BACKEND}" \
    --base-url "${BASE_URL}" \
    --model "${MODEL_PATH}" \
    --tokenizer "${MODEL_PATH}" \
    --dataset-name random \
    --random-input-len "${SHORT_INPUT_LEN}" \
    --random-output-len "${SHORT_OUTPUT_LEN}" \
    --num-prompts "${num_prompts}" \
    --request-rate inf \
    --max-concurrency "${conc}" \
    --seed "${seed}" \
    --warmup-requests 1 \
    --output-file "${out_file}" \
    --disable-tqdm \
    | tee "${log_file}"
}

run_long_case() {
  local round="$1"
  local conc="$2"
  local num_prompts
  if [[ "${conc}" -eq 1 ]]; then
    num_prompts=30
  else
    num_prompts=120
  fi
  local seed=$((SEED_BASE + 5000 + round * 100 + conc))
  local out_file="${OUT_DIR}/round${round}_long_c${conc}.jsonl"
  local log_file="${OUT_DIR}/round${round}_long_c${conc}.log"

  echo "=== round=${round} long c=${conc} num_prompts=${num_prompts} ==="
  python3 -m sglang.bench_serving \
    --backend "${BENCH_BACKEND}" \
    --base-url "${BASE_URL}" \
    --model "${MODEL_PATH}" \
    --tokenizer "${MODEL_PATH}" \
    --dataset-name random \
    --random-input-len "${LONG_INPUT_LEN}" \
    --random-output-len "${LONG_OUTPUT_LEN}" \
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
    run_short_case "${round}" "${conc}"
  done
done

for round in "${ROUNDS[@]}"; do
  for conc in "${CONCURRENCY_LIST[@]}"; do
    run_long_case "${round}" "${conc}"
  done
done

echo "All benchmark jobs finished."
echo "Raw results: ${OUT_DIR}"
echo "Next: python3 scripts/summarize_bench_serving_results.py --input-dir ${OUT_DIR} --output-csv ${OUT_DIR}/summary.csv"
