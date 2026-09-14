# Experimental CUDA request batching

Branch: `test/sensenova-cuda-batching`. This branch enables CUDA batching for
validation; GPU correctness and performance results are still pending. Native
NPU FIA, RMSNorm and MLP optimizations remain NPU-only. No patch or generated
checkout is required. Use an existing Linux CUDA SGLang environment and a complete
local checkpoint. The model and B2 must fit in device memory; a 24 GB GPU may not
fit the tested 2048 workload. Do not install GPU dependencies on Windows.

## Get the code

```bash
cd /workspace
git clone --branch test/sensenova-cuda-batching --single-branch \
  https://github.com/syd520zy/sglang.git sglang-cuda-batching
cd /workspace/sglang-cuda-batching
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PATH=/model/ModelScope/SenseNova/SenseNova-U1.5-8B-MoT
export RESULTS=/workspace/sensenova-gpu-results
mkdir -p "$RESULTS"
git rev-parse HEAD > "$RESULTS/commit.txt"
nvidia-smi > "$RESULTS/nvidia-smi.txt"
set -o pipefail
python3 benchmark/sensenova/check_gpu_attention.py | tee "$RESULTS/attention-check.txt"
```

All four checks must pass before continuing. This BF16 test compares prefix KV
and two denoise attention calls for unequal prefix lengths against compact
singletons, with CFG on/off and broadcast/expanded masks (`atol=rtol=0.02`).
It forces SDPA for this numerical check; serving retains automatic backend
selection. These checks do not establish full-model image equivalence.

## B1 smoke and performance

In terminal A, activate the existing environment, enter the repository, and run:

```bash
cd /workspace/sglang-cuda-batching
export MODEL_PATH=/model/ModelScope/SenseNova/SenseNova-U1.5-8B-MoT
bash benchmark/sensenova/gpu_test.sh serve 1
```

In terminal B, activate the same environment and run:

```bash
cd /workspace/sglang-cuda-batching
export MODEL_PATH=/model/ModelScope/SenseNova/SenseNova-U1.5-8B-MoT
bash benchmark/sensenova/gpu_test.sh smoke 1
bash benchmark/sensenova/gpu_test.sh perf 1
```

The script sets PYTHONPATH to this checkout and waits for server readiness.
Smoke: four concurrent requests, 1024, five steps, CFG 4 and CFG 1, saved PNGs.
Performance: four concurrent requests, 2048, 50 steps, CFG 4, one measured round
after four identical-configuration warmup requests. Warmup is excluded from timing.
Each run must report four successes and zero failures.

## B2 smoke and performance

Stop B1 with Ctrl+C in terminal A. Wait for the server to exit and release memory.

Terminal A:

```bash
bash benchmark/sensenova/gpu_test.sh serve 2
```

Terminal B:

```bash
bash benchmark/sensenova/gpu_test.sh smoke 2
bash benchmark/sensenova/gpu_test.sh perf 2
grep 'Processed dynamic batch of 2/2' /workspace/sensenova-gpu-results/server-b2.log
python3 benchmark/sensenova/compare_gpu_results.py
```

Verify real 2/2 batches in both the smoke and measured performance intervals,
not just during warmup. The comparison script checks matching prompts/seeds,
reports PNG pixel differences and B2 throughput gain, and writes comparison.json.
Inspect matching images for semantic or visible quality changes: pixel errors
are diagnostic, not an established perceptual quality threshold.

Use identical device, model, resolution, precision and server settings for B1/B2.
If 2048 runs out of memory, record it and change the performance resolution in
gpu_test.sh to 1024 for BOTH groups, using a fresh RESULTS directory in both
terminals. Results directories cannot be reused. More throughput does not imply
lower request latency; four-request P95 is essentially the slowest request and
does not establish production tail latency or statistical confidence.

## Return results

```bash
cd /workspace
tar -czf sensenova-gpu-results.tar.gz sensenova-gpu-results
```

Include the server logs, attention check output, image pairs, JSON results and
device/commit metadata. This branch is independent of the NPU PR branch.
