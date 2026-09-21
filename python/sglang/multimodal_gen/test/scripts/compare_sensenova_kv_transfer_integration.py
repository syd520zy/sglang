"""Compare SenseNova SRT thinking with native replay and transferred prefix KV."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def load_records(directory: Path) -> list[dict]:
    return json.loads((directory / "profile" / "records.json").read_text())


def indexed_rows(records: list[dict]) -> dict[tuple[str, int], dict]:
    return {(row["case"], row["seed"]): row for row in records}


def compare_images(left: Path, right: Path) -> dict:
    with Image.open(left) as left_image, Image.open(right) as right_image:
        left_array = np.asarray(left_image.convert("RGB"), dtype=np.int16)
        right_array = np.asarray(right_image.convert("RGB"), dtype=np.int16)
    if left_array.shape != right_array.shape:
        raise RuntimeError(
            f"image shapes differ: {left_array.shape} != {right_array.shape}"
        )

    abs_diff = np.abs(left_array - right_array)
    squared_diff = np.square(left_array - right_array, dtype=np.float64)
    rmse = math.sqrt(float(squared_diff.mean()))
    return {
        "shape": list(left_array.shape),
        "mae": round(float(abs_diff.mean()), 6),
        "rmse": round(rmse, 6),
        "max_abs_error": int(abs_diff.max()),
        "different_pixel_pct": round(
            float(np.any(abs_diff != 0, axis=-1).mean() * 100), 6
        ),
        "psnr_db": None if rmse == 0 else round(20 * math.log10(255 / rmse), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--transfer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay = indexed_rows(load_records(args.replay_dir))
    transfer = indexed_rows(load_records(args.transfer_dir))
    if set(replay) != set(transfer):
        raise RuntimeError("replay and transfer runs contain different cases")

    comparisons = {}
    for key in sorted(replay):
        replay_row = replay[key]
        transfer_row = transfer[key]
        thinking_case = key[0] != "off"
        if thinking_case and replay_row.get("thinking_backend") != "srt":
            raise RuntimeError(f"replay run did not use SRT: {replay_row}")
        if thinking_case and transfer_row.get("thinking_backend") != "srt":
            raise RuntimeError(f"transfer run did not use SRT: {transfer_row}")
        prefix_stage = "think_replay_prefill" if thinking_case else "condition_prefill"
        replay_ms = replay_row["stage_timings_ms"][prefix_stage]
        transfer_ms = transfer_row["stage_timings_ms"][prefix_stage]
        replay_image = args.replay_dir / "profile" / replay_row["image_file"]
        transfer_image = args.transfer_dir / "profile" / transfer_row["image_file"]
        name = f"{key[0]}/seed-{key[1]}"
        comparisons[name] = {
            "case": key[0],
            "prefix_stage": prefix_stage,
            "reasoning_tokens_match": replay_row["reasoning_tokens"]
            == transfer_row["reasoning_tokens"],
            "think_text_hash_match": replay_row["think_text_sha256"]
            == transfer_row["think_text_sha256"],
            "image_hash_match": replay_row["image_sha256"]
            == transfer_row["image_sha256"],
            "image_metrics": compare_images(replay_image, transfer_image),
            "transfer_used": transfer_row.get("srt_kv_transfer_used") is True,
            "transferred_prefixes": transfer_row.get("srt_kv_transferred_prefixes"),
            "replay_used_transfer": replay_row.get("srt_kv_transfer_used") is True,
            "session_reused_tokens": transfer_row.get("srt_kv_session_reused_tokens"),
            "srt_cached_tokens": transfer_row.get("srt_kv_cached_tokens"),
            "replay_prefill_ms": replay_ms,
            "transfer_prefill_ms": transfer_ms,
            "replay_cfg_prefill_ms": replay_row["stage_timings_ms"]["cfg_prefill"],
            "transfer_cfg_prefill_ms": transfer_row["stage_timings_ms"]["cfg_prefill"],
            "speedup": round(replay_ms / transfer_ms, 3),
        }

    automated_passed = all(
        row["reasoning_tokens_match"]
        and row["think_text_hash_match"]
        and row["transfer_used"]
        and set(row["transferred_prefixes"] or ()) == {"condition", "uncondition"}
        and not row["replay_used_transfer"]
        and (
            row["case"] == "off"
            or (
                isinstance(row["session_reused_tokens"], int)
                and row["session_reused_tokens"] > 0
                and isinstance(row["srt_cached_tokens"], int)
                and row["srt_cached_tokens"] > 0
            )
        )
        for row in comparisons.values()
    )
    report = {
        "passed": automated_passed,
        "content_review_required": any(
            not row["image_hash_match"] for row in comparisons.values()
        ),
        "acceptance": (
            "Image hash is diagnostic only. Review generated images for subject, "
            "count, spatial relationships, prompt semantics, composition, and "
            "artifacts."
        ),
        "comparisons": comparisons,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not automated_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
