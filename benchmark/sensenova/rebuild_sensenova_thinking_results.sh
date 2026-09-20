#!/usr/bin/env bash
# Rebuild the P1 result directories: archive what earlier runs left inside the
# checkout, then re-run the lifecycle acceptance and the concurrency
# comparison into a fresh timestamped root next to it.
#
# Usage: bash benchmark/sensenova/rebuild_sensenova_thinking_results.sh [RESULT_ROOT]
#
# Nothing is deleted: earlier directories are moved under <RESULT_ROOT>/_stale.
# Set SKIP_LIFECYCLE=1 or SKIP_CONCURRENCY=1 to rebuild only one of them, and
# set DRY_RUN=1 to see what would happen without running any benchmark.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCRIPTS="python/sglang/multimodal_gen/test/scripts"
LIFECYCLE="${SCRIPTS}/validate_sensenova_thinking_lifecycle.sh"
CONCURRENCY="${SCRIPTS}/compare_sensenova_thinking_concurrency.sh"
VIEWER="${SCRIPTS}/view_sensenova_thinking_results.py"

# Old runs used the default output paths, which place results in the checkout.
LEGACY_DIRS=(
  "sensenova-thinking-concurrency-compare"
  "sensenova-thinking-lifecycle-results"
)

RESULT_ROOT="${1:-$(dirname "${REPO_ROOT}")/sensenova-thinking-results}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUT="${RESULT_ROOT}/${RUN_ID}"

# The lifecycle covers both the fallback and the strict mode.
MODES="${MODES:-fallback strict}"
SRT_FAILURE="${SRT_FAILURE:-stop}"
STEPS="${STEPS:-10}"
MAX_THINK_TOKENS="${MAX_THINK_TOKENS:-64}"
STEPS_LIST="${STEPS_LIST:-10}"
CONCURRENCY="${CONCURRENCY:-1 2}"
BUDGETS="${BUDGETS:-64,128,256}"
REPEATS="${REPEATS:-2}"

echo "repository:  ${REPO_ROOT}"
echo "result root: ${RESULT_ROOT}"
echo "this run:    ${OUT}"
echo

for name in "${LEGACY_DIRS[@]}"; do
  path="${REPO_ROOT}/${name}"
  if [[ -d "${path}" ]] && [[ -n "$(ls -A "${path}")" ]]; then
    target="${RESULT_ROOT}/_stale/${name}-${RUN_ID}"
    if [[ -n "${DRY_RUN:-}" ]]; then
      echo "would archive ${path} -> ${target}"
      continue
    fi
    mkdir -p "$(dirname "${target}")"
    mv "${path}" "${target}"
    echo "archived ${path} -> ${target}"
  fi
done

if [[ -n "${DRY_RUN:-}" ]]; then
  echo
  echo "dry run: nothing was archived or run"
  exit 0
fi

mkdir -p "${OUT}/lifecycle" "${OUT}/concurrency"

status=0
if [[ -z "${SKIP_LIFECYCLE:-}" ]]; then
  echo "=== lifecycle acceptance (${MODES}) ==="
  MODES="${MODES}" SRT_FAILURE="${SRT_FAILURE}" STEPS="${STEPS}" \
  MAX_THINK_TOKENS="${MAX_THINK_TOKENS}" OUTPUT_DIR="${OUT}/lifecycle" \
    bash "${LIFECYCLE}" || status=1
else
  echo "=== lifecycle acceptance skipped (SKIP_LIFECYCLE=1) ==="
fi

if [[ -z "${SKIP_CONCURRENCY:-}" ]] && (( status == 0 )); then
  echo
  echo "=== concurrency comparison (steps=${STEPS_LIST}) ==="
  STEPS_LIST="${STEPS_LIST}" CONCURRENCY="${CONCURRENCY}" BUDGETS="${BUDGETS}" \
  REPEATS="${REPEATS}" OUTPUT_DIR="${OUT}/concurrency" \
    bash "${CONCURRENCY}"
elif [[ -n "${SKIP_CONCURRENCY:-}" ]]; then
  echo "=== concurrency comparison skipped (SKIP_CONCURRENCY=1) ==="
fi

echo
echo "results: ${OUT}"
echo
echo "read them with:"
echo "  cd ${REPO_ROOT}"
echo "  python ${VIEWER} \\"
echo "    ${OUT}/lifecycle \\"
echo "    ${OUT}/concurrency \\"
echo "    --output ${OUT}/view.txt"
