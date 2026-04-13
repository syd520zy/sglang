#!/usr/bin/env bash
set -euo pipefail

# Benchmark matrix:
# - rounds: default 1 (override by env ROUNDS)
# - short case: 1024 input / 100 output, random dataset, no range variance
# - long case: 30000 input / 100 output, random dataset by default
# - concurrency: 1, 10, 20, 40
# - num_prompts per run: concurrency * 10
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
#   ROUNDS="1 2 3"
#   CONCURRENCY_LIST="1 10 20 40"
#   PROMPTS_PER_CONCURRENCY=10
#   LONG_DATASET_SOURCE=random
#   LONG_DATASET_PATH=/path/to/longbench_custom.json

BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
MODEL_PATH="${MODEL_PATH:-/model/ModelScope/Qwen/Qwen3-8B}"
OUT_DIR="${OUT_DIR:-/workspace/bench_results/qwen3_8b_baseline}"
BENCH_BACKEND="${BENCH_BACKEND:-sglang}"
SEED_BASE="${SEED_BASE:-20260413}"

CONCURRENCY_LIST=(${CONCURRENCY_LIST:-1 10 20 40})
ROUNDS=(${ROUNDS:-1})
PROMPTS_PER_CONCURRENCY="${PROMPTS_PER_CONCURRENCY:-10}"

SHORT_INPUT_LEN=1024
SHORT_OUTPUT_LEN=100
SHORT_RANDOM_RANGE_RATIO=0.0

LONG_INPUT_LEN=30000
LONG_OUTPUT_LEN=100
LONG_RANDOM_RANGE_RATIO=0.0

LONG_DATASET_SOURCE="${LONG_DATASET_SOURCE:-random}"
LONG_DATASET_PATH="${LONG_DATASET_PATH:-}"

mkdir -p "${OUT_DIR}"

echo "BASE_URL=${BASE_URL}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "BENCH_BACKEND=${BENCH_BACKEND}"
echo "ROUNDS=${ROUNDS[*]}"
echo "CONCURRENCY_LIST=${CONCURRENCY_LIST[*]}"
echo "PROMPTS_PER_CONCURRENCY=${PROMPTS_PER_CONCURRENCY}"
echo "SHORT=random (in=${SHORT_INPUT_LEN}, out=${SHORT_OUTPUT_LEN}, range=${SHORT_RANDOM_RANGE_RATIO})"

resolve_long_dataset_args() {
  LONG_DATASET_ARGS=()
  case "${LONG_DATASET_SOURCE}" in
    random)
      LONG_DATASET_ARGS=(
        --dataset-name random
        --random-input-len "${LONG_INPUT_LEN}"
        --random-output-len "${LONG_OUTPUT_LEN}"
        --random-range-ratio "${LONG_RANDOM_RANGE_RATIO}"
      )
      ;;
    custom|longbench)
      if [[ -n "${LONG_DATASET_PATH}" && -f "${LONG_DATASET_PATH}" ]]; then
        LONG_DATASET_ARGS=(
          --dataset-name custom
          --dataset-path "${LONG_DATASET_PATH}"
          --sharegpt-output-len "${LONG_OUTPUT_LEN}"
        )
        echo "LONG dataset source: custom file (${LONG_DATASET_PATH})"
      else
        echo "[WARN] LONG_DATASET_SOURCE=${LONG_DATASET_SOURCE} but LONG_DATASET_PATH is missing or not found."
        echo "[WARN] Falling back to random long-case dataset."
        LONG_DATASET_ARGS=(
          --dataset-name random
          --random-input-len "${LONG_INPUT_LEN}"
          --random-output-len "${LONG_OUTPUT_LEN}"
          --random-range-ratio "${LONG_RANDOM_RANGE_RATIO}"
        )
      fi
      ;;
    *)
      echo "[WARN] Unknown LONG_DATASET_SOURCE=${LONG_DATASET_SOURCE}. Falling back to random."
      LONG_DATASET_ARGS=(
        --dataset-name random
        --random-input-len "${LONG_INPUT_LEN}"
        --random-output-len "${LONG_OUTPUT_LEN}"
        --random-range-ratio "${LONG_RANDOM_RANGE_RATIO}"
      )
      ;;
  esac
}

run_short_case() {
  local round="$1"
  local conc="$2"
  local num_prompts=$((conc * PROMPTS_PER_CONCURRENCY))
  local seed=$((SEED_BASE + round * 1000 + conc))
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
    --random-range-ratio "${SHORT_RANDOM_RANGE_RATIO}" \
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
  local num_prompts=$((conc * PROMPTS_PER_CONCURRENCY))
  local seed=$((SEED_BASE + 500000 + round * 1000 + conc))
  local out_file="${OUT_DIR}/round${round}_long_c${conc}.jsonl"
  local log_file="${OUT_DIR}/round${round}_long_c${conc}.log"

  echo "=== round=${round} long c=${conc} num_prompts=${num_prompts} ==="
  echo "    long dataset source=${LONG_DATASET_SOURCE}"
  python3 -m sglang.bench_serving \
    --backend "${BENCH_BACKEND}" \
    --base-url "${BASE_URL}" \
    --model "${MODEL_PATH}" \
    --tokenizer "${MODEL_PATH}" \
    "${LONG_DATASET_ARGS[@]}" \
    --num-prompts "${num_prompts}" \
    --request-rate inf \
    --max-concurrency "${conc}" \
    --seed "${seed}" \
    --warmup-requests 1 \
    --output-file "${out_file}" \
    --disable-tqdm \
    | tee "${log_file}"
}

resolve_long_dataset_args

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
