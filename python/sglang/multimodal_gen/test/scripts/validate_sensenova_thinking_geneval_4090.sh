#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

GENEVAL_DIR="${GENEVAL_DIR:?Set GENEVAL_DIR to the official GenEval checkout}"
MODEL_PATH="${MODEL_PATH:-sensenova/SenseNova-U1.5-8B-MoT}"
GPU_ID="${GPU_ID:-0}"
SERVER_PORT="${SERVER_PORT:-30000}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
MAX_THINK_TOKENS="${MAX_THINK_TOKENS:-256}"
MAX_PER_TAG="${MAX_PER_TAG:-30}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
SCORE="${SCORE:-1}"
GENEVAL_PYTHON="${GENEVAL_PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-sensenova-thinking-geneval-results/$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"
METADATA="${GENEVAL_DIR}/prompts/evaluation_metadata.jsonl"
EVALUATOR="${GENEVAL_DIR}/evaluation/evaluate_images.py"
GENERATOR="${SCRIPT_DIR}/eval_sensenova_thinking_geneval.py"

[[ -f "${METADATA}" && -f "${EVALUATOR}" ]] || {
  echo "Official GenEval checkout is incomplete: ${GENEVAL_DIR}" >&2
  exit 1
}
if [[ "${SCORE}" == "1" ]]; then
  GENEVAL_MODEL_DIR="${GENEVAL_MODEL_DIR:?Set GENEVAL_MODEL_DIR to the downloaded Mask2Former weights directory}"
  [[ -s "${GENEVAL_MODEL_DIR}/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth" ]] || {
    echo "GenEval Mask2Former weights are missing from ${GENEVAL_MODEL_DIR}" >&2
    exit 1
  }
  "${GENEVAL_PYTHON}" -c 'import mmdet, open_clip, clip_benchmark' || {
    echo "GenEval scorer dependencies are missing from GENEVAL_PYTHON. Use SCORE=0 to generate images only." >&2
    exit 1
  }
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

{
  echo "sglang_commit=$(git rev-parse HEAD)"
  echo "geneval_commit=$(git -C "${GENEVAL_DIR}" rev-parse HEAD)"
  echo "model_path=${MODEL_PATH}"
  echo "resolution=${WIDTH}x${HEIGHT}"
  echo "max_think_tokens=${MAX_THINK_TOKENS}"
  echo "max_per_tag=${MAX_PER_TAG}"
  echo "samples_per_prompt=${SAMPLES_PER_PROMPT}"
  python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")'
  nvidia-smi
} >"${OUTPUT_DIR}/environment.txt" 2>&1

python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_sensenova_u1.py \
  test/manual/test_sensenova_thinking_geneval.py -q \
  | tee "${OUTPUT_DIR}/unit-tests.log"

sglang serve --model-path "${MODEL_PATH}" --host 127.0.0.1 --port "${SERVER_PORT}" "$@" \
  >"${OUTPUT_DIR}/server.log" 2>&1 &
SERVER_PID=$!
deadline=$((SECONDS + 3600))
until curl -fsS "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null || ((SECONDS >= deadline)); then
    tail -n 150 "${OUTPUT_DIR}/server.log" >&2
    exit 1
  fi
  sleep 5
done

COMMON=(
  --metadata-file "${METADATA}"
  --base-url "http://127.0.0.1:${SERVER_PORT}"
  --model "${MODEL_PATH}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --max-think-tokens "${MAX_THINK_TOKENS}"
  --max-per-tag "${MAX_PER_TAG}"
  --samples-per-prompt "${SAMPLES_PER_PROMPT}"
)
python "${GENERATOR}" generate "${COMMON[@]}" --smoke \
  --output-dir "${OUTPUT_DIR}/smoke" | tee "${OUTPUT_DIR}/smoke.log"
python "${GENERATOR}" generate "${COMMON[@]}" \
  --output-dir "${OUTPUT_DIR}/benchmark" | tee "${OUTPUT_DIR}/generation.log"

cleanup
SERVER_PID=""
if [[ "${SCORE}" == "1" ]]; then
  for mode in off on; do
    "${GENEVAL_PYTHON}" "${EVALUATOR}" "${OUTPUT_DIR}/benchmark/${mode}" \
      --outfile "${OUTPUT_DIR}/benchmark/${mode}-results.jsonl" \
      --model-path "${GENEVAL_MODEL_DIR}" \
      >"${OUTPUT_DIR}/benchmark/${mode}-evaluation.log" 2>&1
    "${GENEVAL_PYTHON}" "${GENEVAL_DIR}/evaluation/summary_scores.py" \
      "${OUTPUT_DIR}/benchmark/${mode}-results.jsonl" \
      | tee "${OUTPUT_DIR}/benchmark/${mode}-summary.txt"
  done
  python "${GENERATOR}" compare \
    --output-dir "${OUTPUT_DIR}/benchmark" \
    --off-results "${OUTPUT_DIR}/benchmark/off-results.jsonl" \
    --on-results "${OUTPUT_DIR}/benchmark/on-results.jsonl" \
    | tee "${OUTPUT_DIR}/comparison.log"
fi
echo "Results: ${OUTPUT_DIR}"
