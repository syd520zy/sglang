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
    selected_mib = checkpoint["current_managed_srt"]["bytes"] / (1 << 20)
    understanding_mib = checkpoint["groups"]["language_understanding"]["bytes"] / (
        1 << 20
    )
    language_io_mib = checkpoint["groups"]["language_io"]["bytes"] / (1 << 20)
    target_mib = args.target_gib * 1024
    split_estimate = srt_peak - selected_mib
    keep_main_io_estimate = srt_peak - understanding_mib
    report = {
        "native_peak_used_memory_mb": native_peak,
        "managed_srt_peak_used_memory_mb": srt_peak,
        "observed_managed_srt_increment_mb": round(srt_peak - native_peak, 2),
        "managed_srt_selected_weight_mb": round(selected_mib, 2),
        "language_understanding_weight_mb": round(understanding_mib, 2),
        "language_io_weight_mb": round(language_io_mib, 2),
        "split_kv_handoff_peak_estimate_mb": round(split_estimate, 2),
        "target_memory_mb": round(target_mib, 2),
        "split_estimate_headroom_mb": round(target_mib - split_estimate, 2),
        "split_estimate_fits_target": split_estimate <= target_mib,
        "split_keep_main_io_peak_estimate_mb": round(keep_main_io_estimate, 2),
        "split_keep_main_io_headroom_mb": round(target_mib - keep_main_io_estimate, 2),
        "split_keep_main_io_fits_target": keep_main_io_estimate <= target_mib,
        "limitations": [
            "The optimistic split estimate removes all SRT-selected weights from the main process.",
            "The keep-main-IO estimate removes dense understanding weights but retains embeddings and lm_head in both processes.",
            "It does not yet include KV export buffers or allocator changes required by the compact implementation.",
            "nvidia-smi sampling can miss short activation peaks; validate the final design with allocator-level snapshots.",
        ],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
