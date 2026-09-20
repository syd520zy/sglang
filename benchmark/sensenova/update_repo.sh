#!/usr/bin/env bash
# Updates an existing checkout to the latest SenseNova thinking branch commit.
# It never touches a dirty worktree and it never rewrites history.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-feat/sensenova-thinking-mode}"
EXPECTED_REMOTE_SUFFIX="${EXPECTED_REMOTE_SUFFIX:-syd520zy/sglang.git}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "${REPO_ROOT} is not a git worktree" >&2
  exit 1
fi

REMOTE_URL="$(git remote get-url "${REMOTE}" 2>/dev/null || echo "")"
if [[ -z "${REMOTE_URL}" ]]; then
  echo "remote ${REMOTE} is not configured; add it with:" >&2
  echo "  git remote add ${REMOTE} https://github.com/syd520zy/sglang.git" >&2
  exit 1
fi
if [[ "${REMOTE_URL}" != *"${EXPECTED_REMOTE_SUFFIX}" ]]; then
  echo "warning: ${REMOTE} is ${REMOTE_URL}, expected it to end with ${EXPECTED_REMOTE_SUFFIX}" >&2
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "the worktree has local changes; commit or discard them before updating:" >&2
  git status --short >&2
  exit 1
fi

echo "repository: ${REPO_ROOT}"
echo "remote:     ${REMOTE} (${REMOTE_URL})"
echo "branch:     ${BRANCH}"

git fetch "${REMOTE}" "${BRANCH}"
if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
  git switch "${BRANCH}"
else
  git switch --track "${REMOTE}/${BRANCH}"
fi
git pull --ff-only "${REMOTE}" "${BRANCH}"

echo
echo "checked out:"
git --no-pager log -1 --format='  commit:  %H%n  subject: %s%n  date:    %cd' --date=iso
echo "  branch:  $(git rev-parse --abbrev-ref HEAD)"
echo "  ahead of ${REMOTE}/${BRANCH}: $(git rev-list --count "${REMOTE}/${BRANCH}..HEAD")"

EXPECTED_KERNEL="$(sed -n 's/.*"sglang-kernel==\([0-9.]*\)".*/\1/p' python/pyproject.toml | head -n 1)"
echo
echo "python environment:"
python - "${REPO_ROOT}" "${EXPECTED_KERNEL}" <<'PY'
import sys

repo_root = sys.argv[1]
expected_kernel = sys.argv[2]

try:
    import torch
except ImportError as exc:
    print(f"  torch: missing ({exc})")
else:
    print(f"  torch: {torch.__version__} cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"  gpu:   {torch.cuda.get_device_name(0)}")
    else:
        print("  gpu:   no CUDA device visible")

try:
    import sglang
except ImportError as exc:
    print(f"  sglang: missing ({exc}); install it with `python -m pip install -e python`")
else:
    print(f"  sglang: {sglang.__file__}")
    if repo_root not in str(sglang.__file__):
        print("  warning: the imported sglang is not this checkout")

from importlib.metadata import PackageNotFoundError, version

try:
    installed = version("sglang-kernel")
except PackageNotFoundError:
    installed = None
print(f"  sglang-kernel: {installed or 'missing'} (source requires {expected_kernel})")
if installed and expected_kernel and installed != expected_kernel:
    print(
        "  fix: python -m pip install --force-reinstall --no-deps "
        f"'sglang-kernel=={expected_kernel}'"
    )
PY

echo
echo "next: run the profiling or lifecycle script from this directory."
