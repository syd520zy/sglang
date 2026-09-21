"""Extract SenseNova SRT CUDA Graph and KV-pool evidence from its log."""

import argparse
import ast
import json
import re
from pathlib import Path

GRAPH_BEGIN = re.compile(r"Capture target decode CUDA graph begin\..*?bs=(\[[^]]*])")
KV_TOKENS = re.compile(
    r"KV cache (?:is )?allocated\..*?#tokens:\s*(\d+)", re.IGNORECASE
)


def inspect_runtime_log(
    text: str, *, context_length: int, max_concurrency: int, cuda_graph_max_bs: int
):
    capture_batches = []
    for match in GRAPH_BEGIN.finditer(text):
        value = ast.literal_eval(match.group(1))
        if isinstance(value, list):
            capture_batches.extend(int(item) for item in value)
    capture_batches = sorted(set(capture_batches))
    kv_matches = [int(match.group(1)) for match in KV_TOKENS.finditer(text)]
    kv_pool_tokens = kv_matches[-1] if kv_matches else None
    required_batches = list(range(1, min(max_concurrency, cuda_graph_max_bs) + 1))
    required_kv_tokens = context_length * max_concurrency
    failures = [
        line.strip()
        for line in text.splitlines()
        if "Capture cuda graph failed" in line
        or "Capture target decode CUDA graph failed" in line
    ]
    graph_ok = (
        not failures
        and all(batch_size in capture_batches for batch_size in required_batches)
        and "Capture target decode CUDA graph end." in text
    )
    kv_ok = kv_pool_tokens is not None and kv_pool_tokens >= required_kv_tokens
    return {
        "cuda_graph": {
            "configured_max_batch_size": cuda_graph_max_bs,
            "required_batch_sizes": required_batches,
            "captured_batch_sizes": capture_batches,
            "capture_completed": graph_ok,
            "failures": failures,
        },
        "kv_pool": {
            "tokens": kv_pool_tokens,
            "required_tokens": required_kv_tokens,
            "context_length": context_length,
            "max_concurrency": max_concurrency,
            "sufficient": kv_ok,
        },
        "passed": graph_ok and kv_ok,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--max-concurrency", type=int, required=True)
    parser.add_argument("--cuda-graph-max-bs", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    log_files = sorted(args.runtime_dir.glob("*.log"))
    if not log_files:
        parser.error(f"no SRT log found in {args.runtime_dir}")
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in log_files
    )
    report = inspect_runtime_log(
        text,
        context_length=args.context_length,
        max_concurrency=args.max_concurrency,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
    )
    report["log_files"] = [str(path) for path in log_files]
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
