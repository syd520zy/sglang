#!/usr/bin/env bash
set -euo pipefail

# Validate whether bench_serving metrics are valid (non-zero / non-empty)
# for both short and long context profiles.
#
# Usage:
#   bash scripts/validate_bench_metrics.sh
#
# Optional env:
#   BASE_URL=http://127.0.0.1:30000
#   MODEL_PATH=/model/ModelScope/Qwen/Qwen3-8B
#   BENCH_BACKEND=sglang
#   OUT_DIR=/workspace/bench_results/validate_metrics

BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
MODEL_PATH="${MODEL_PATH:-/model/ModelScope/Qwen/Qwen3-8B}"
BENCH_BACKEND="${BENCH_BACKEND:-sglang}"
OUT_DIR="${OUT_DIR:-/workspace/bench_results/validate_metrics}"

mkdir -p "${OUT_DIR}"

SHORT_OUT="${OUT_DIR}/short_validate.jsonl"
LONG_OUT="${OUT_DIR}/long_validate.jsonl"

run_short() {
  python3 -m sglang.bench_serving \
    --backend "${BENCH_BACKEND}" \
    --base-url "${BASE_URL}" \
    --model "${MODEL_PATH}" \
    --tokenizer "${MODEL_PATH}" \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 100 \
    --num-prompts 30 \
    --request-rate inf \
    --max-concurrency 1 \
    --seed 20260413 \
    --warmup-requests 1 \
    --disable-tqdm \
    --output-file "${SHORT_OUT}"
}

run_long() {
  python3 -m sglang.bench_serving \
    --backend "${BENCH_BACKEND}" \
    --base-url "${BASE_URL}" \
    --model "${MODEL_PATH}" \
    --tokenizer "${MODEL_PATH}" \
    --dataset-name random \
    --random-input-len 16000 \
    --random-output-len 512 \
    --num-prompts 20 \
    --request-rate inf \
    --max-concurrency 1 \
    --seed 20260414 \
    --warmup-requests 1 \
    --disable-tqdm \
    --output-file "${LONG_OUT}"
}

validate_one() {
  local jsonl_file="$1"
  local label="$2"
  python3 - "$jsonl_file" "$label" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
label = sys.argv[2]

if not path.exists():
    print(f"[{label}] FAIL: result file not found: {path}")
    sys.exit(2)

lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
if not lines:
    print(f"[{label}] FAIL: result file is empty: {path}")
    sys.exit(2)

obj = json.loads(lines[-1])

checks = {
    "mean_e2e_latency_ms": obj.get("mean_e2e_latency_ms"),
    "output_throughput": obj.get("output_throughput"),
    "mean_ttft_ms": obj.get("mean_ttft_ms"),
    "mean_tpot_ms": obj.get("mean_tpot_ms"),
}

failed = []
for k, v in checks.items():
    if v is None:
        failed.append(f"{k}=None")
    elif not isinstance(v, (int, float)):
        failed.append(f"{k} not numeric: {type(v).__name__}")
    elif v <= 0:
        failed.append(f"{k}<=0 ({v})")

completed = obj.get("completed", 0)
if not isinstance(completed, (int, float)) or completed <= 0:
    failed.append(f"completed<=0 ({completed})")

if failed:
    print(f"[{label}] FAIL")
    for x in failed:
        print(f"  - {x}")
    print(f"  file: {path}")
    sys.exit(1)

print(f"[{label}] PASS")
print(
    f"  TTFT={checks['mean_ttft_ms']:.2f} ms, "
    f"TPOT={checks['mean_tpot_ms']:.2f} ms, "
    f"E2E={checks['mean_e2e_latency_ms']:.2f} ms, "
    f"OUT_TPS={checks['output_throughput']:.2f} tok/s"
)
PY
}

echo "Running short-case validation..."
run_short
validate_one "${SHORT_OUT}" "short"

echo "Running long-case validation..."
run_long
validate_one "${LONG_OUT}" "long"

echo "All metric validations passed."
echo "Results:"
echo "  ${SHORT_OUT}"
echo "  ${LONG_OUT}"
