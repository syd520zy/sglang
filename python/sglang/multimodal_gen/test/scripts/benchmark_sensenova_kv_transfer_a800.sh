#!/usr/bin/env bash
# Benchmark the proposed layer-streamed KV handoff without loading the model.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

RESULT_ROOT="${OUTPUT_DIR:-/workspace/sensenova-kv-transfer/$(date +%Y%m%d-%H%M%S)}"
RESULT_ROOT="$(mkdir -p "${RESULT_ROOT}" && cd "${RESULT_ROOT}" && pwd)"
exec > >(tee -a "${RESULT_ROOT}/run.log") 2>&1
trap 'status=$?; echo "FAILED (${status}) at line ${LINENO}: ${BASH_COMMAND}"' ERR

echo "Repository: ${REPO_ROOT}"
echo "Commit: $(git rev-parse HEAD)"
echo "Results: ${RESULT_ROOT}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
df -h /dev/shm || true

python "${SCRIPT_DIR}/benchmark_sensenova_kv_transfer.py" \
  --tokens "${TOKENS:-256,512,1024,4096}" \
  --batch-sizes "${BATCH_SIZES:-1,2}" \
  --repeats "${REPEATS:-3}" \
  --replay-reference-ms "${REPLAY_REFERENCE_MS:-164}" \
  --output "${RESULT_ROOT}/results.json"

echo "Results: ${RESULT_ROOT}"
