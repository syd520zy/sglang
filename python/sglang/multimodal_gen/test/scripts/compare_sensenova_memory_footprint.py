"""Compare native and managed-SRT GPU peaks with checkpoint weight bytes."""

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--srt", type=Path, required=True)
    parser.add_argument("--target-gib", type=float, default=48.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = load(args.checkpoint)
    native_peak = load(args.native)["peak_total_used_memory_mb"]
    srt_peak = load(args.srt)["peak_total_used_memory_mb"]
    duplicate_mib = checkpoint["current_managed_srt"]["bytes"] / (1 << 20)
    target_mib = args.target_gib * 1024
    split_estimate = srt_peak - duplicate_mib
    report = {
        "native_peak_used_memory_mb": native_peak,
        "managed_srt_peak_used_memory_mb": srt_peak,
        "observed_managed_srt_increment_mb": round(srt_peak - native_peak, 2),
        "duplicated_understanding_weight_mb": round(duplicate_mib, 2),
        "split_kv_handoff_peak_estimate_mb": round(split_estimate, 2),
        "target_memory_mb": round(target_mib, 2),
        "split_estimate_headroom_mb": round(target_mib - split_estimate, 2),
        "split_estimate_fits_target": split_estimate <= target_mib,
        "limitations": [
            "The split estimate subtracts duplicated checkpoint weights from the observed managed-SRT peak.",
            "It does not yet include KV export buffers or allocator changes required by the compact implementation.",
            "nvidia-smi sampling can miss short activation peaks; validate the final design with allocator-level snapshots.",
        ],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
