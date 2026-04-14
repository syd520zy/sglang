#!/usr/bin/env bash
set -euo pipefail

# One-shot runtime environment checker for SGLang serving.
#
# What it checks:
# 1) `python -m sglang.check_env` full environment snapshot
# 2) `python -m pip check` dependency conflicts
# 3) Hard version gates and package consistency:
#    - sglang-kernel >= MIN_SGL_KERNEL_VERSION
#    - flashinfer-python >= MIN_FLASHINFER_VERSION
#    - flashinfer-python version == flashinfer-cubin version (if cubin installed)
#
# Usage:
#   bash scripts/playground/check_runtime_env.sh
#
# Optional environment variables:
#   PYTHON_BIN=python
#   REPORT_ROOT=/tmp/sglang_env_check_reports
#   MIN_SGL_KERNEL_VERSION=0.4.1
#   MIN_FLASHINFER_VERSION=0.6.7.post3

PYTHON_BIN="${PYTHON_BIN:-python}"
REPORT_ROOT="${REPORT_ROOT:-/tmp/sglang_env_check_reports}"
MIN_SGL_KERNEL_VERSION="${MIN_SGL_KERNEL_VERSION:-0.4.1}"
MIN_FLASHINFER_VERSION="${MIN_FLASHINFER_VERSION:-0.6.7.post3}"
export MIN_SGL_KERNEL_VERSION
export MIN_FLASHINFER_VERSION

mkdir -p "${REPORT_ROOT}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${REPORT_ROOT}/check_${TIMESTAMP}"
mkdir -p "${REPORT_DIR}"

ENV_LOG="${REPORT_DIR}/01_check_env.txt"
PIP_CHECK_LOG="${REPORT_DIR}/02_pip_check.txt"
VERSION_CHECK_LOG="${REPORT_DIR}/03_version_check.txt"
SUMMARY_LOG="${REPORT_DIR}/SUMMARY.txt"

echo "[INFO] Python executable: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "[INFO] Report directory: ${REPORT_DIR}"
echo

echo "[STEP 1/3] Collecting environment snapshot ..."
set +e
${PYTHON_BIN} -m sglang.check_env >"${ENV_LOG}" 2>&1
CHECK_ENV_RC=$?
if [[ ${CHECK_ENV_RC} -ne 0 ]]; then
  # Fallback when sglang is not importable as a module but source tree is present.
  ${PYTHON_BIN} python/sglang/check_env.py >"${ENV_LOG}" 2>&1
  CHECK_ENV_RC=$?
fi
set -e
if [[ ${CHECK_ENV_RC} -ne 0 ]]; then
  echo "[WARN] python -m sglang.check_env failed. Continue with other checks."
  echo "[WARN] See ${ENV_LOG}"
else
  echo "[PASS] Environment snapshot collected: ${ENV_LOG}"
fi

echo
echo "[STEP 2/3] Running pip dependency conflict check ..."
set +e
${PYTHON_BIN} -m pip check >"${PIP_CHECK_LOG}" 2>&1
PIP_CHECK_RC=$?
set -e
if [[ ${PIP_CHECK_RC} -ne 0 ]]; then
  echo "[FAIL] pip dependency conflicts found."
  echo "[FAIL] See ${PIP_CHECK_LOG}"
else
  echo "[PASS] pip dependency check passed: ${PIP_CHECK_LOG}"
fi

echo
echo "[STEP 3/3] Running hard version and consistency checks ..."
set +e
${PYTHON_BIN} - <<PY >"${VERSION_CHECK_LOG}" 2>&1
import importlib.metadata as md
import os
import sys
from packaging.version import Version

MIN_SGL_KERNEL_VERSION = Version(os.environ["MIN_SGL_KERNEL_VERSION"])
MIN_FLASHINFER_VERSION = Version(os.environ["MIN_FLASHINFER_VERSION"])


def get_ver(candidates):
    for name in candidates:
        try:
            return name, md.version(name)
        except Exception:
            pass
    return None, None


targets = {
    "sglang-kernel": ["sglang-kernel"],
    "flashinfer-python": ["flashinfer-python", "flashinfer_python"],
    "flashinfer-cubin": ["flashinfer-cubin", "flashinfer_cubin", "flashinfer-cubin-cu12"],
    "flashinfer-jit-cache": ["flashinfer-jit-cache", "flashinfer_jit_cache"],
    "sglang": ["sglang"],
    "torch": ["torch"],
    "triton": ["triton"],
}

resolved = {}
for key, candidates in targets.items():
    pkg_name, version = get_ver(candidates)
    resolved[key] = (pkg_name, version)
    print(f"{key:20s}: {version or 'NOT INSTALLED'} ({pkg_name or '-'})")

ok = True

def fail(msg):
    global ok
    ok = False
    print(f"[FAIL] {msg}")


def check_min(key, min_version):
    _, ver = resolved[key]
    if ver is None:
        fail(f"{key} is not installed, require >= {min_version}")
        return
    if Version(ver) < min_version:
        fail(f"{key}={ver} < required {min_version}")
    else:
        print(f"[PASS] {key}={ver} >= {min_version}")


check_min("sglang-kernel", MIN_SGL_KERNEL_VERSION)
check_min("flashinfer-python", MIN_FLASHINFER_VERSION)

_, fi_py = resolved["flashinfer-python"]
_, fi_cubin = resolved["flashinfer-cubin"]
if fi_py and fi_cubin:
    if fi_py != fi_cubin:
        fail(f"flashinfer-python ({fi_py}) != flashinfer-cubin ({fi_cubin})")
    else:
        print(f"[PASS] flashinfer-python and flashinfer-cubin match: {fi_py}")
else:
    print("[INFO] flashinfer-cubin not installed or not detectable; skip strict match check")

if ok:
    print("[RESULT] PASS")
    sys.exit(0)
print("[RESULT] FAIL")
sys.exit(2)
PY
VERSION_CHECK_RC=$?
set -e

if [[ ${VERSION_CHECK_RC} -ne 0 ]]; then
  echo "[FAIL] hard version/consistency checks failed."
  echo "[FAIL] See ${VERSION_CHECK_LOG}"
else
  echo "[PASS] hard version/consistency checks passed: ${VERSION_CHECK_LOG}"
fi

FINAL_RC=0
if [[ ${PIP_CHECK_RC} -ne 0 ]]; then
  FINAL_RC=1
fi
if [[ ${VERSION_CHECK_RC} -ne 0 ]]; then
  FINAL_RC=1
fi

{
  echo "SGLang Runtime Env Check Summary"
  echo "timestamp: ${TIMESTAMP}"
  echo "python_bin: ${PYTHON_BIN}"
  echo "report_dir: ${REPORT_DIR}"
  echo "check_env_rc: ${CHECK_ENV_RC}"
  echo "pip_check_rc: ${PIP_CHECK_RC}"
  echo "version_check_rc: ${VERSION_CHECK_RC}"
  if [[ ${FINAL_RC} -eq 0 ]]; then
    echo "final_result: PASS"
  else
    echo "final_result: FAIL"
  fi
  echo "env_log: ${ENV_LOG}"
  echo "pip_check_log: ${PIP_CHECK_LOG}"
  echo "version_check_log: ${VERSION_CHECK_LOG}"
} | tee "${SUMMARY_LOG}"

echo
if [[ ${FINAL_RC} -eq 0 ]]; then
  echo "[DONE] Environment check PASSED."
else
  echo "[DONE] Environment check FAILED. Please review logs in ${REPORT_DIR}."
fi

exit ${FINAL_RC}
