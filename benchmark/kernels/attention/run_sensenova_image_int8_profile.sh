#!/usr/bin/env bash
# Run from an activated SGLang environment; no FlashAttention dependency.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
OUT="${1:-/workspace/sensenova-image-kv-profile/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
{
    git log -1 --oneline
    git status --short
    nvidia-smi
    python - <<'PY'
import torch
import triton
print("GPU:", torch.cuda.get_device_name())
print("GPU memory bytes:", torch.cuda.get_device_properties(0).total_memory)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Triton:", triton.__version__)
PY
} > "$OUT/environment.log" 2>&1
python -m pytest \
    test/registered/unit/mem_cache/test_sensenova_int8_cache.py \
    test/registered/kernel/attention/test_sensenova_int8_attention.py \
    python/sglang/multimodal_gen/test/unit/test_sensenova_u1.py \
    -q 2>&1 | tee "$OUT/tests.log"
printf 'dtype,batch_size,resolution,exit_code\n' > "$OUT/status.csv"
failed=0
for dtype in bf16 fp16; do
    for batch in 1 2; do
        for resolution in 1024 2048; do
            name="${dtype}-b${batch}-r${resolution}"
            echo "Running $name"
            if python benchmark/kernels/attention/bench_sensenova_int8.py \
                --baseline sdpa --dtype "$dtype" --batch-size "$batch" \
                --prefix-length 256 --resolution "$resolution" --quantize-image \
                --rounds 7 --profile-dir "$OUT/$name" --profile-iterations 10 \
                > "$OUT/$name.json" 2> "$OUT/$name.stderr.log"; then
                code=0
            else
                code=$?
                failed=1
                echo "Failed $name (exit $code); see $OUT/$name.stderr.log" >&2
            fi
            printf '%s,%s,%s,%s\n' "$dtype" "$batch" "$resolution" "$code" >> "$OUT/status.csv"
        done
    done
done
echo "Results: $OUT"
exit "$failed"
