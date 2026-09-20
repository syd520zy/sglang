"""Compare native and SRT SenseNova thinking profiling results."""

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def thinking_backends(records):
    return {
        record.get("thinking_backend") for record in records if record["case"] != "off"
    }


def common_prefix_length(left: str, right: str) -> int:
    for index, (left_char, right_char) in enumerate(zip(left, right)):
        if left_char != right_char:
            return index
    return min(len(left), len(right))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--srt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    native_records = load_json(args.native_dir / "profile" / "records.json")
    srt_records = load_json(args.srt_dir / "profile" / "records.json")
    native_backends = thinking_backends(native_records)
    srt_backends = thinking_backends(srt_records)
    if native_backends != {"native"}:
        raise RuntimeError(f"Native run used unexpected backends: {native_backends}")
    if srt_backends != {"srt"}:
        raise RuntimeError(
            f"SRT run used {srt_backends}; inspect {args.srt_dir / 'server.log'}"
        )

    native_summary = load_json(args.native_dir / "profile" / "summary.json")
    srt_summary = load_json(args.srt_dir / "profile" / "summary.json")
    native_cases = native_summary["cases"]
    srt_cases = srt_summary["cases"]
    comparisons = {}
    for case in sorted(set(native_cases) & set(srt_cases)):
        if case == "off":
            continue
        native = native_cases[case]
        srt = srt_cases[case]
        native_per_token = native["think_decode_ms_per_token"]
        srt_per_token = srt["think_decode_ms_per_token"]
        comparisons[case] = {
            "native_think_decode_ms_per_token": native_per_token,
            "srt_think_decode_ms_per_token": srt_per_token,
            "think_decode_speedup": round(native_per_token / srt_per_token, 3),
            "native_total_ms": native["stage_timings_ms_mean"]["total"],
            "srt_total_ms": srt["stage_timings_ms_mean"]["total"],
            "srt_replay_prefill_ms": srt["stage_timings_ms_mean"][
                "think_replay_prefill"
            ],
        }

    native_hashes = {
        (record["case"], record["seed"]): record["think_text_sha256"]
        for record in native_records
        if record["case"] != "off"
    }
    srt_hashes = {
        (record["case"], record["seed"]): record["think_text_sha256"]
        for record in srt_records
        if record["case"] != "off"
    }
    keys = sorted(set(native_hashes) | set(srt_hashes))
    text_matches = {
        f"{case}/seed-{seed}": native_hashes.get((case, seed))
        == srt_hashes.get((case, seed))
        for case, seed in keys
    }
    native_rows = {
        (record["case"], record["seed"]): record
        for record in native_records
        if record["case"] != "off"
    }
    srt_rows = {
        (record["case"], record["seed"]): record
        for record in srt_records
        if record["case"] != "off"
    }
    token_counts = {
        f"{case}/seed-{seed}": {
            "native": native_rows[(case, seed)]["reasoning_tokens"],
            "srt": srt_rows[(case, seed)]["reasoning_tokens"],
        }
        for case, seed in keys
        if (case, seed) in native_rows and (case, seed) in srt_rows
    }
    text_differences = {}
    for case, seed in keys:
        native_row = native_rows.get((case, seed))
        srt_row = srt_rows.get((case, seed))
        if (
            native_row is None
            or srt_row is None
            or native_row.get("think_text") is None
            or srt_row.get("think_text") is None
            or text_matches[f"{case}/seed-{seed}"]
        ):
            continue
        native_text = native_row["think_text"]
        srt_text = srt_row["think_text"]
        prefix_length = common_prefix_length(native_text, srt_text)
        text_differences[f"{case}/seed-{seed}"] = {
            "common_prefix_chars": prefix_length,
            "native_chars": len(native_text),
            "srt_chars": len(srt_text),
            "native_from_difference": native_text[prefix_length : prefix_length + 160],
            "srt_from_difference": srt_text[prefix_length : prefix_length + 160],
        }
    report = {
        "native_backends": sorted(native_backends),
        "srt_backends": sorted(srt_backends),
        "all_think_text_hashes_match": all(text_matches.values()),
        "think_text_hash_matches": text_matches,
        "reasoning_token_counts": token_counts,
        "think_text_differences": text_differences,
        "comparisons": comparisons,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
