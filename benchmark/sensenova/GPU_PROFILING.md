# SenseNova CUDA batching profiling

This captures one bounded denoise profile from the serving path for B1 and B2.
It is intended to determine whether B2 falls back from FlashAttention to SDPA,
and to measure attention-mask, GQA expansion, compute, and device-idle time. It
does not replace the 2048, 50-step end-to-end benchmark.

## 1. Update and prepare

Run on the CUDA server in the existing SGLang environment:

```bash
cd /workspace/sglang-cuda-batching
git fetch origin
git checkout test/sensenova-cuda-batching
git pull --ff-only origin test/sensenova-cuda-batching

export REPO=/workspace/sglang-cuda-batching
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PATH=/model/ModelScope/SenseNova/SenseNova-U1.5-8B-MoT
export PROFILE_ROOT=/workspace/sensenova-gpu-profile-baseline
rm -rf "$PROFILE_ROOT"
mkdir -p "$PROFILE_ROOT"
cd "$REPO"

git rev-parse HEAD | tee "$PROFILE_ROOT/commit.txt"
nvidia-smi | tee "$PROFILE_ROOT/nvidia-smi.txt"
python3 - <<'PY' | tee "$PROFILE_ROOT/environment.txt"
import importlib.util
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name())
print("capability:", torch.cuda.get_device_capability())
print("flash_attn installed:", importlib.util.find_spec("flash_attn") is not None)
print("flash SDPA enabled:", torch.backends.cuda.flash_sdp_enabled())
PY
```

Do not reuse the earlier performance result directory. The profiling request
uses 2048, CFG 4, three denoise steps and records only a bounded denoise window.

## 2. Capture B1

In terminal A:

```bash
cd "$REPO"
export SGLANG_DIFFUSION_TORCH_PROFILER_DIR="$PROFILE_ROOT/b1"
mkdir -p "$SGLANG_DIFFUSION_TORCH_PROFILER_DIR"
bash benchmark/sensenova/gpu_test.sh serve 1
```

After the health check succeeds, run in terminal B:

```bash
cd "$REPO"
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
python3 benchmark/sensenova/profile_gpu_batching.py \
  --model "$MODEL_PATH" --batch-size 1 \
  --output "$PROFILE_ROOT/b1"
```

Stop the B1 server with Ctrl+C. B1 uses the longer prompt that is also present
in B2, so both cases have the same maximum prefix length.

## 3. Capture B2

In terminal A:

```bash
export SGLANG_DIFFUSION_TORCH_PROFILER_DIR="$PROFILE_ROOT/b2"
mkdir -p "$SGLANG_DIFFUSION_TORCH_PROFILER_DIR"
bash benchmark/sensenova/gpu_test.sh serve 2
```

In terminal B:

```bash
python3 benchmark/sensenova/profile_gpu_batching.py \
  --model "$MODEL_PATH" --batch-size 2 \
  --output "$PROFILE_ROOT/b2"
grep 'Processed dynamic batch of 2/2' \
  /workspace/sensenova-gpu-results/server-b2.log | tail
```

The command must report exactly one new trace and the server log must contain a
real 2/2 batch during this request. B2 uses one short and one long prompt, which
activates padding and the attention-mask path.

## 4. Check and package

```bash
python3 - <<'PY'
from pathlib import Path
root = Path('/workspace/sensenova-gpu-profile-baseline')
for batch in ('b1', 'b2'):
    traces = list((root / batch).glob('*.trace.json.gz'))
    assert len(traces) == 1, (batch, traces)
    assert (root / batch / 'metadata.json').is_file()
total = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
assert total < 30_000_000, total
print('total bytes:', total)
PY

cd /workspace
python3 - <<'PY'
import shutil
shutil.make_archive(
    '/workspace/sensenova-gpu-profile-baseline',
    'zip',
    '/workspace',
    'sensenova-gpu-profile-baseline',
)
PY
ls -lh /workspace/sensenova-gpu-profile-baseline.zip
```

Return `sensenova-gpu-profile-baseline.zip`. If either run exceeds its 14 MB
limit, remove both B1 and B2 directories and repeat both commands with
`--size 1024`; do not compare different resolutions.

Inspect the traces for FlashAttention kernels, PyTorch SDPA kernels,
`repeat_interleave`/copy kernels, mask expansion/materialization, GEMM duration,
and CPU or CUDA idle gaps. Compare operator counts and total CUDA time rather
than request latency from this three-step diagnostic run.
