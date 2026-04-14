#!/usr/bin/env bash
set -euo pipefail

# One-click A/B benchmark for Qwen3-8B:
# 1) baseline server (CSAttention=0) + benchmark
# 2) stop server with `pkill -9 python`
# 3) csattention server (CSAttention=1) + benchmark
# 4) compare-ab report
#
# Usage:
#   bash scripts/playground/run_qwen3_8b_csattention_ab.sh
#
# Optional env overrides:
#   MODEL_PATH=/model/ModelScope/Qwen/Qwen3-8B
#   HOST=127.0.0.1
#   PORT=30000
#   PYTHON_BIN=python
#   OUTPUT_ROOT=benchmark/context_compare
#   CONCURRENCIES=1,10,20
#   REQUEST_MULTIPLIER=10
#   SERVER_EXTRA_ARGS="--tp 1"
#   BENCH_EXTRA_ARGS="--warmup-requests 2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-/model/ModelScope/Qwen/Qwen3-8B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark/context_compare}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"
CONCURRENCIES="${CONCURRENCIES:-1,10,20}"
REQUEST_MULTIPLIER="${REQUEST_MULTIPLIER:-10}"

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
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${BASELINE_DIR}" "${CSATTN_DIR}" "${AB_DIR}" "${LOG_DIR}"

BASELINE_SERVER_LOG="${LOG_DIR}/server_baseline.log"
CSATTN_SERVER_LOG="${LOG_DIR}/server_csattention.log"

wait_for_server_ready() {
  local host="$1"
  local port="$2"
  local timeout_sec="${3:-900}"
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

stop_server() {
  set +e
  pkill -9 python >/dev/null 2>&1
  set -e
  sleep 3
}

start_server() {
  local csattn_flag="$1"
  local log_file="$2"

  stop_server
  echo "[INFO] Launching server with SGLANG_NSA_USE_CSATTENTION=${csattn_flag} ..."
  SGLANG_NSA_USE_CSATTENTION="${csattn_flag}" \
    "${PYTHON_BIN}" -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --host "${HOST}" \
      --port "${PORT}" \
      ${SERVER_EXTRA_ARGS} \
      > "${log_file}" 2>&1 &

  if ! wait_for_server_ready "${HOST}" "${PORT}" 900; then
    echo "[ERROR] Server did not become ready in time. Log tail:"
    tail -n 80 "${log_file}" || true
    exit 1
  fi
  echo "[INFO] Server is ready."
}

run_benchmark_matrix() {
  local out_dir="$1"
  local run_label="$2"
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

cleanup() {
  echo "[INFO] Cleanup: stopping python server(s) with pkill -9 python"
  stop_server
}
trap cleanup EXIT

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] Model path: ${MODEL_PATH}"
echo "[INFO] Output root: ${RUN_ROOT}"

start_server "0" "${BASELINE_SERVER_LOG}"
run_benchmark_matrix "${BASELINE_DIR}" "baseline"

start_server "1" "${CSATTN_SERVER_LOG}"
run_benchmark_matrix "${CSATTN_DIR}" "csattention"

run_ab_compare

echo
echo "[DONE] All benchmarks finished."
echo "[DONE] Baseline results:   ${BASELINE_DIR}/runs_summary.csv"
echo "[DONE] CSAttention results:${CSATTN_DIR}/runs_summary.csv"
echo "[DONE] A/B CSV:            ${AB_DIR}/ab_comparison.csv"
echo "[DONE] A/B Summary:        ${AB_DIR}/ab_summary.md"
