#!/usr/bin/env bash
# Runs the P1-3 lifecycle acceptance for both strict and fallback modes.
# Each mode starts its own server, breaks the internal SRT service and checks
# what the server reports and logs before and after that.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-sensenova/SenseNova-U1.5-8B-MoT}"
GPU_ID="${GPU_ID:-0}"
SERVER_PORT="${SERVER_PORT:-30000}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
STEPS="${STEPS:-10}"
MAX_THINK_TOKENS="${MAX_THINK_TOKENS:-64}"
PROMPT="${PROMPT:-A realistic photo of three red apples arranged to the left of a blue bowl.}"
MODES="${MODES:-fallback strict}"
# stop: the service hangs and is only found by the read timeout.
# kill: the service is gone and refuses connections.
SRT_FAILURE="${SRT_FAILURE:-stop}"
SRT_TIMEOUT_MS="${SRT_TIMEOUT_MS:-100000}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}"
MAX_IDLE_GPU_MEMORY_MB="${MAX_IDLE_GPU_MEMORY_MB:-1024}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-sensenova-thinking-lifecycle-results/$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"
VALIDATOR="${SCRIPT_DIR}/validate_sensenova_thinking_lifecycle.py"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SERVER_PID=""
SRT_PID=""
cleanup() {
  if [[ -n "${SRT_PID}" ]] && kill -0 "${SRT_PID}" 2>/dev/null; then
    kill -CONT "${SRT_PID}" 2>/dev/null || true
  fi
  local pid
  for pid in "${SRT_PID}" "${SERVER_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      python - "${pid}" <<'PY' || kill "${pid}" 2>/dev/null || true
import sys

from sglang.srt.utils import kill_process_tree

kill_process_tree(int(sys.argv[1]), wait_timeout=60)
PY
    fi
  done
  SERVER_PID=""
  SRT_PID=""
}
trap cleanup EXIT INT TERM

gpu_memory_used_mb() {
  nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used \
    --format=csv,noheader,nounits | awk 'NR == 1 {print $1}'
}

gpu_compute_pids() {
  nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -u | paste -sd, -
}

wait_for_gpu_memory_release() {
  local limit_mb=$((INITIAL_GPU_MEMORY_MB + 512))
  local deadline=$((SECONDS + 300))
  local used_mb
  while ((SECONDS < deadline)); do
    used_mb="$(gpu_memory_used_mb)"
    if ((used_mb <= limit_mb)); then
      return 0
    fi
    sleep 3
  done
  echo "GPU ${GPU_ID} still uses ${used_mb} MiB; refusing to start the next mode." >&2
  nvidia-smi --id="${GPU_ID}" >&2
  return 1
}

# Returns the pid listening on a port, or an empty string. The lookup reads
# /proc inside the validator, so it does not depend on ss or lsof being present.
listening_pid_on_port() {
  python "${VALIDATOR}" --phase srt-pid --port "$1" \
    --output-dir "${OUTPUT_DIR}" 2>/dev/null || true
}

run_mode() {
  local mode="$1"
  local strict
  local dir="${OUTPUT_DIR}/${mode}"
  local verdict=0
  if [[ "${mode}" == "strict" ]]; then strict=1; else strict=0; fi
  mkdir -p "${dir}"

  wait_for_gpu_memory_release
  {
    echo "sglang_commit=$(git rev-parse HEAD)"
    echo "mode=${mode}"
    echo "strict=${strict}"
    echo "model_path=${MODEL_PATH}"
    echo "thinking_backend=srt"
    echo "srt_failure=${SRT_FAILURE}"
    echo "srt_timeout_ms=${SRT_TIMEOUT_MS}"
    echo "steps=${STEPS}"
    echo "max_think_tokens=${MAX_THINK_TOKENS}"
    echo "baseline_gpu_compute_pids=$(gpu_compute_pids)"
    python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")'
  } >"${dir}/environment.txt" 2>&1

  echo "=== ${mode}: starting the server (strict=${strict})"
  SGLANG_SENSENOVA_THINKING_BACKEND=srt \
  SGLANG_SENSENOVA_THINKING_STRICT="${strict}" \
  SGLANG_SENSENOVA_THINKING_MEM_FRACTION="${SGLANG_SENSENOVA_THINKING_MEM_FRACTION:-0.45}" \
  sglang serve --model-path "${MODEL_PATH}" --host 127.0.0.1 --port "${SERVER_PORT}" \
    >"${dir}/server.log" 2>&1 &
  SERVER_PID=$!
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  until curl -fsS "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null || ((SECONDS >= deadline)); then
      tail -n 150 "${dir}/server.log" >&2
      return 1
    fi
    sleep 5
  done

  echo "=== ${mode}: checking the ready state"
  if ! python "${VALIDATOR}" --phase info \
    --output-dir "${dir}" --base-url "http://127.0.0.1:${SERVER_PORT}" \
    >"${dir}/server-info.log" 2>&1; then
    cat "${dir}/server-info.log" >&2
    echo "could not read ${SERVER_PORT}/server_info; see ${dir}/server-info.log" >&2
    return 1
  fi
  cat "${dir}/server-info.json"

  echo "=== ${mode}: the ready state must serve thinking with SRT"
  local startup=0
  python "${VALIDATOR}" --phase startup \
    --output-dir "${dir}" --base-url "http://127.0.0.1:${SERVER_PORT}" \
    --server-log "${dir}/server.log" \
    --model "${MODEL_PATH}" --prompt "${PROMPT}" \
    --width "${WIDTH}" --height "${HEIGHT}" --steps "${STEPS}" \
    --max-think-tokens "${MAX_THINK_TOKENS}" \
    >"${dir}/lifecycle-startup.log" 2>&1 || startup=$?
  cat "${dir}/lifecycle-startup.log"

  local srt_port
  srt_port="$(python - "${dir}/server-info.json" <<'PY'
import json
import sys
import urllib.parse

info = json.load(open(sys.argv[1], encoding="utf-8"))
print(urllib.parse.urlparse(info["thinking_backend"]["url"]).port)
PY
)"
  SRT_PID="$(listening_pid_on_port "${srt_port}")"
  if [[ -z "${SRT_PID}" ]]; then
    echo "no process listens on the SRT port ${srt_port}; cannot break the backend" >&2
    echo "server pid ${SERVER_PID} children: $(pgrep -P "${SERVER_PID}" | paste -sd, -)" >&2
    return 1
  fi
  echo "=== ${mode}: internal SRT pid ${SRT_PID} on port ${srt_port}; mode ${SRT_FAILURE}"

  if [[ "${SRT_FAILURE}" == "stop" ]]; then
    # SIGSTOP keeps the socket open but never answers, so only the client read
    # timeout finds out: the strongest form of "the service stopped working".
    kill -STOP "${SRT_PID}"
  else
    python - "${SRT_PID}" <<'PY'
import sys

from sglang.srt.utils import kill_process_tree

kill_process_tree(int(sys.argv[1]), wait_timeout=30)
PY
  fi

  echo "=== ${mode}: one request must discover the failure, later ones must not wait"
  local after_kill=0
  python "${VALIDATOR}" --phase after-kill \
    --output-dir "${dir}" --base-url "http://127.0.0.1:${SERVER_PORT}" \
    --server-log "${dir}/server.log" \
    --model "${MODEL_PATH}" --prompt "${PROMPT}" \
    --width "${WIDTH}" --height "${HEIGHT}" --steps "${STEPS}" \
    --max-think-tokens "${MAX_THINK_TOKENS}" \
    --strict "${strict}" --failure-mode "${SRT_FAILURE}" \
    --srt-timeout-ms "${SRT_TIMEOUT_MS}" \
    >"${dir}/lifecycle-after-kill.log" 2>&1 || after_kill=$?
  cat "${dir}/lifecycle-after-kill.log"

  echo "=== ${mode}: stopping the server and checking for leftovers"
  cleanup
  sleep 5
  local leftover_port
  leftover_port="$(listening_pid_on_port "${srt_port}")"
  local leftover_gpu
  leftover_gpu="$(gpu_compute_pids)"
  local startup_note="PASS"
  if ((startup != 0)); then startup_note="FAIL"; verdict=1; fi
  local after_kill_note="PASS"
  if ((after_kill != 0)); then after_kill_note="FAIL"; verdict=1; fi
  {
    echo "srt_port=${srt_port}"
    echo "srt_port_still_listening=${leftover_port:-none}"
    echo "gpu_compute_pids_after_shutdown=${leftover_gpu:-none}"
    echo "baseline_gpu_compute_pids=$(sed -n 's/^baseline_gpu_compute_pids=//p' "${dir}/environment.txt")"
    echo "startup_verdict=${startup_note}"
    echo "after_kill_verdict=${after_kill_note}"
  } >"${dir}/residue.txt"
  cat "${dir}/residue.txt"

  if [[ -n "${leftover_port}" ]]; then
    echo "${mode}: the SRT port is still listening after shutdown" >&2
    verdict=1
  fi
  return "${verdict}"
}

INITIAL_GPU_MEMORY_MB="$(gpu_memory_used_mb)"
if ((INITIAL_GPU_MEMORY_MB > MAX_IDLE_GPU_MEMORY_MB)); then
  echo "GPU ${GPU_ID} already uses ${INITIAL_GPU_MEMORY_MB} MiB; " \
    "the lifecycle check requires an idle GPU (limit ${MAX_IDLE_GPU_MEMORY_MB} MiB)." >&2
  nvidia-smi --id="${GPU_ID}" >&2
  exit 1
fi

if ((RUN_UNIT_TESTS)); then
  python -m pytest python/sglang/multimodal_gen/test/unit/test_sensenova_u1.py -q \
    | tee "${OUTPUT_DIR}/unit-tests.log"
fi

status=0
for mode in ${MODES}; do
  if run_mode "${mode}"; then
    echo "${mode}: PASSED"
  else
    echo "${mode}: FAILED"
    status=1
  fi
  # A mode that bailed out early must not leave its server behind, or the next
  # mode waits for GPU memory that this one still holds.
  cleanup
done

echo "Results: ${OUTPUT_DIR}"
exit "${status}"
