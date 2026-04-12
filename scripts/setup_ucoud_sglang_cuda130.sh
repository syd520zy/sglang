#!/usr/bin/env bash
set -euo pipefail

# One-click setup for UCloud base image:
# CUDA 13.0 + Python 3.12 + Ubuntu 22.04
# Target: run current SGLang project in this repo.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "[1/8] Check OS / Python / CUDA"
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Please install Python 3.12 first."
  exit 1
fi
python3 -V
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "WARNING: nvidia-smi not found. Continue, but GPU check is skipped."
fi

echo "[2/8] Install apt dependencies"
SUDO=""
if [[ "${EUID}" -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    echo "Not running as root and sudo not found. Please run as root or install sudo."
    exit 1
  fi
fi

${SUDO} apt-get update
${SUDO} apt-get install -y --no-install-recommends \
  git git-lfs curl wget ca-certificates \
  build-essential cmake ninja-build pkg-config ccache \
  python3-dev python3-pip \
  numactl libnuma1 libnuma-dev \
  ffmpeg \
  libopenmpi-dev
${SUDO} rm -rf /var/lib/apt/lists/*

echo "[3/8] Upgrade pip build tools"
python3 -m pip install -U pip setuptools wheel packaging

echo "[4/8] Install sglang-kernel (CUDA 13.0 wheel)"
python3 -m pip install \
  "https://github.com/sgl-project/whl/releases/download/v0.4.0/sglang_kernel-0.4.0+cu130-cp310-abi3-manylinux2014_x86_64.whl" \
  --force-reinstall --no-deps

echo "[5/8] Install SGLang core deps (editable)"
python3 -m pip install -e "python" --extra-index-url https://download.pytorch.org/whl/cu130

echo "[6/8] Download FlashInfer cubin cache"
FLASHINFER_CUBIN_DOWNLOAD_THREADS="${FLASHINFER_CUBIN_DOWNLOAD_THREADS:-8}" \
FLASHINFER_LOGGING_LEVEL=warning \
python3 -m flashinfer --download-cubin

echo "[7/8] Write env helper"
ENV_HELPER="${REPO_ROOT}/scripts/env_cuda130_4090.sh"
cat > "${ENV_HELPER}" <<'EOF'
#!/usr/bin/env bash
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_HOME=/usr/local/cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Environment set for RTX 4090 + CUDA 13.0"
EOF
chmod +x "${ENV_HELPER}"
echo "Created: ${ENV_HELPER}"

echo "[8/8] Quick runtime check"
python3 - <<'PY'
import torch
import sglang
import flashinfer
print("torch:", torch.__version__)
print("torch.cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("sglang:", getattr(sglang, "__version__", "ok"))
print("flashinfer:", getattr(flashinfer, "__version__", "ok"))
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

cat <<EOF

Setup completed.

Next:
1) source "${ENV_HELPER}"
2) Start server example:
   python3 -m sglang.launch_server --model-path /model/ModelScope/Qwen/Qwen3-8B --host 0.0.0.0 --port 30000

EOF
