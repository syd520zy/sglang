#!/usr/bin/env bash
set -euo pipefail

# Benchmark + compare only (no server start/stop).
#
# Flow:
# 1) Verify current running server is reachable; run baseline benchmark.
# 2) Pause and ask user to manually restart server with CSAttention enabled.
# 3) Verify server is reachable again; run csattention benchmark.
# 4) Generate A/B comparison files.
#
# Usage:
#   bash scripts/playground/run_qwen3_8b_csattention_ab.sh
#
# Manual server startup examples:
#   # baseline
#   SGLANG_NSA_USE_CSATTENTION=0 python -m sglang.launch_server \
#     --model-path /model/ModelScope/Qwen/Qwen3-8B --host 127.0.0.1 --port 30000
#
#   # csattention
#   SGLANG_NSA_USE_CSATTENTION=1 python -m sglang.launch_server \
#     --model-path /model/ModelScope/Qwen/Qwen3-8B --host 127.0.0.1 --port 30000
#
# Optional env overrides:
#   MODEL_PATH=/model/ModelScope/Qwen/Qwen3-8B
#   HOST=127.0.0.1
#   PORT=30000
#   PYTHON_BIN=python
#   OUTPUT_ROOT=benchmark/context_compare
#   CONCURRENCIES=1,10,20
#   REQUEST_MULTIPLIER=10
#   LIVE_BENCH_LOG=1
#   COLLECT_OUTPUT_DETAILS=0
#   BENCH_EXTRA_ARGS="--warmup-requests 2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-/model/ModelScope/Qwen/Qwen3-8B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark/context_compare}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"
CONCURRENCIES="${CONCURRENCIES:-1,10,20}"
REQUEST_MULTIPLIER="${REQUEST_MULTIPLIER:-10}"
LIVE_BENCH_LOG="${LIVE_BENCH_LOG:-1}"
COLLECT_OUTPUT_DETAILS="${COLLECT_OUTPUT_DETAILS:-0}"

# Enforce concurrency-driven test mode.
if [[ "${BENCH_EXTRA_ARGS}" == *"--request-rate"* ]]; then
  echo "[ERROR] BENCH_EXTRA_ARGS should not contain --request-rate."
  echo "[ERROR] This script runs in concurrency-driven mode with --request-rate inf fixed."
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${OUTPUT_ROOT}/qwen3_8b_csattention_ab_${TIMESTAMP}"
BASELINE_DIR="${RUN_ROOT}/baseline_gpu"
CSATTN_DIR="${RUN_ROOT}/csattention_gpu"
AB_DIR="${RUN_ROOT}/ab_compare"
mkdir -p "${BASELINE_DIR}" "${CSATTN_DIR}" "${AB_DIR}"

wait_for_server_ready() {
  local host="$1"
  local port="$2"
  local timeout_sec="${3:-300}"
  local start_ts now elapsed
  start_ts="$(date +%s)"
  while true; do
    if curl -fsS "http://${host}:${port}/health_generate" >/dev/null 2>&1; then
      return 0
    fi
    if curl -fsS "http://${host}:${port}/server_info" >/dev/null 2>&1; then
      return 0
    fi
    now="$(date +%s)"
    elapsed="$((now - start_ts))"
    if [[ "${elapsed}" -ge "${timeout_sec}" ]]; then
      return 1
    fi
    sleep 2
  done
}

run_benchmark_matrix() {
  local out_dir="$1"
  local run_label="$2"
  local live_log_args=""
  local output_details_args=""
  if [[ "${LIVE_BENCH_LOG}" == "1" ]]; then
    live_log_args="--live-log"
  fi
  if [[ "${COLLECT_OUTPUT_DETAILS}" == "1" ]]; then
    output_details_args="--output-details"
  fi
  echo "[INFO] Running benchmark matrix for ${run_label} ..."
  "${PYTHON_BIN}" scripts/playground/auto_context_compare.py \
    --mode run \
    --backend sglang \
    --host "${HOST}" \
    --port "${PORT}" \
    --model "${MODEL_PATH}" \
    --concurrencies "${CONCURRENCIES}" \
    --request-multiplier "${REQUEST_MULTIPLIER}" \
    --request-rate inf \
    --output-dir "${out_dir}" \
    ${live_log_args} \
    ${output_details_args} \
    ${BENCH_EXTRA_ARGS}
}

run_ab_compare() {
  echo "[INFO] Running A/B comparison ..."
  "${PYTHON_BIN}" scripts/playground/auto_context_compare.py \
    --mode compare-ab \
    --baseline-dir "${BASELINE_DIR}" \
    --optimized-dir "${CSATTN_DIR}" \
    --optimized-label csattention \
    --ab-output-dir "${AB_DIR}"
}

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] Model path: ${MODEL_PATH}"
echo "[INFO] Host/Port: ${HOST}:${PORT}"
echo "[INFO] Output root: ${RUN_ROOT}"
echo
echo "[STEP 1/4] Ensure baseline server is already running with:"
echo "  SGLANG_NSA_USE_CSATTENTION=0"
if ! wait_for_server_ready "${HOST}" "${PORT}" 300; then
  echo "[ERROR] Baseline server not reachable at http://${HOST}:${PORT}."
  exit 1
fi
run_benchmark_matrix "${BASELINE_DIR}" "baseline"

echo
echo "[STEP 2/4] Please stop baseline server and start csattention server with:"
echo "  SGLANG_NSA_USE_CSATTENTION=1"
echo "Press Enter to continue after server is ready..."
read -r _

echo "[STEP 3/4] Verifying csattention server ..."
if ! wait_for_server_ready "${HOST}" "${PORT}" 300; then
  echo "[ERROR] CSAttention server not reachable at http://${HOST}:${PORT}."
  exit 1
fi
run_benchmark_matrix "${CSATTN_DIR}" "csattention"

echo
echo "[STEP 4/4] Building A/B report ..."
run_ab_compare

echo
echo "[DONE] All benchmarks finished."
echo "[DONE] Baseline results:   ${BASELINE_DIR}/runs_summary.csv"
echo "[DONE] CSAttention results:${CSATTN_DIR}/runs_summary.csv"
echo "[DONE] A/B CSV:            ${AB_DIR}/ab_comparison.csv"
echo "[DONE] A/B Summary:        ${AB_DIR}/ab_summary.md"
