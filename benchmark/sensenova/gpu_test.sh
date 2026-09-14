#!/usr/bin/env bash
set -euo pipefail
MODE=${1:?serve, smoke, or perf}
BATCH=${2:?1 or 2}
[[ "$BATCH" == 1 || "$BATCH" == 2 ]] || exit 2
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PATH=${MODEL_PATH:-/model/ModelScope/SenseNova/SenseNova-U1.5-8B-MoT}
export RESULTS=${RESULTS:-/workspace/sensenova-gpu-results}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SGLANG_DIFFUSION_TARGET_DEVICE=cuda
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$REPO"
mkdir -p "$RESULTS"
case "$MODE" in
  serve)
    python3 -c 'import torch, sglang; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name()); print(sglang.__file__); import os; assert os.path.realpath(sglang.__file__).startswith(os.path.realpath(os.environ["REPO"] + "/python") + "/")'
    sglang serve --model-path "$MODEL_PATH" --num-gpus 1 --port 30000 \
      --batching-max-size "$BATCH" --batching-delay-ms 100 \
      --enable-batching-metrics 2>&1 | tee "$RESULTS/server-b${BATCH}.log"
    ;;
  smoke|perf)
    python3 - <<'PY'
import time, urllib.request
for _ in range(600):
    try:
        with urllib.request.urlopen('http://127.0.0.1:30000/health', timeout=5) as r:
            if r.status == 200:
                break
    except Exception:
        pass
    time.sleep(2)
else:
    raise SystemExit('Server did not become ready')
PY
    if [[ "$MODE" == smoke ]]; then
      python3 benchmark/sensenova/validate_npu_batching.py \
        --model "$MODEL_PATH" --device cuda --requests 4 --concurrency 4 \
        --size 1024 --steps 5 --cfg 4 --warmup 0 --save-images \
        --output "$RESULTS/smoke-b${BATCH}-cfg4"
      python3 benchmark/sensenova/validate_npu_batching.py \
        --model "$MODEL_PATH" --device cuda --requests 4 --concurrency 4 \
        --size 1024 --steps 5 --cfg 1 --warmup 0 --save-images \
        --output "$RESULTS/smoke-b${BATCH}-cfg1"
    else
      python3 benchmark/sensenova/validate_npu_batching.py \
        --model "$MODEL_PATH" --device cuda --requests 4 --concurrency 4 \
        --size 2048 --steps 50 --cfg 4 --warmup 4 \
        --output "$RESULTS/perf-b${BATCH}"
    fi
    ;;
  *) echo "Unknown mode: $MODE"; exit 2 ;;
esac
