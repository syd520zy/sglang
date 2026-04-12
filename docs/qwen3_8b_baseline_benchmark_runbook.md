# Qwen3-8B Baseline Benchmark Runbook

This runbook targets a single machine and produces 3 rounds of baseline data.

## 1) Start SGLang server (24GB-safe profile)

Use a conservative setup first for 24GB GPUs.

```bash
nohup python3 -m sglang.launch_server \
  --model-path /model/ModelScope/Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.72 \
  --chunked-prefill-size 2048 \
  --context-length 32768 \
  --cuda-graph-bs 1 2 4 8 12 16 \
  --max-running-requests 16 \
  > sglang.log 2>&1 &
  
tail -f sglang.log
```

Health check:

```bash
curl http://127.0.0.1:30000/v1/models
```

CURL: 

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/model/ModelScope/Qwen/Qwen3-8B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "请回复：服务已正常启动"}
    ],
    "temperature": 0,
    "max_tokens": 32
  }'
```

## 2) Run benchmark matrix

Matrix:
- rounds: 1, 2, 3
- cases:
  - short: random, 1024 input / 100 output
  - long: random, 16000 input / 512 output (24GB-safe profile)
- concurrency: 1, 4, 8, 16
- num_prompts:
  - concurrency=1: 30
  - concurrency=4/8/16: 120

Run:

```bash
cd /workspace/SGLANG-FORK
bash scripts/run_qwen3_8b_baseline_bench.sh
```

Optional output path override:

```bash
OUT_DIR=/workspace/bench_results/qwen3_8b_baseline_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_qwen3_8b_baseline_bench.sh
```

Optional backend override:

```bash
BENCH_BACKEND=sglang bash scripts/run_qwen3_8b_baseline_bench.sh
```

## 3) Summarize metrics

```bash
python3 scripts/summarize_bench_serving_results.py \
  --input-dir /workspace/bench_results/qwen3_8b_baseline \
  --output-csv /workspace/bench_results/qwen3_8b_baseline/summary.csv
```

This generates:
- detailed table: `summary.csv` (one row per round/case/concurrency)
- aggregated table: `summary_agg.csv` (average over 3 rounds)

## 4) Record template

Fill this table from `summary_agg.csv`.

| Case | Concurrency | Avg TTFT (ms) | Avg TPOT (ms) | Avg E2E (ms) | Avg Output Throughput (tok/s) |
|---|---:|---:|---:|---:|---:|
| short | 1 |  |  |  |  |
| short | 4 |  |  |  |  |
| short | 8 |  |  |  |  |
| short | 16 |  |  |  |  |
| long | 1 |  |  |  |  |
| long | 4 |  |  |  |  |
| long | 8 |  |  |  |  |
| long | 16 |  |  |  |  |

## 5) Notes for single-machine tests

- Keep all non-benchmark workloads off the GPU during runs.
- Use the same server flags for all rounds to keep comparison fair.
- If long-case at high concurrency fails due OOM, record the failure with concurrency level and error logs, then continue remaining cases.
- Do not set `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` in baseline runs unless you explicitly want out-of-spec experiments.
- `sglang` backend is recommended first for stable TPOT/ITL statistics on long-text pressure tests.
