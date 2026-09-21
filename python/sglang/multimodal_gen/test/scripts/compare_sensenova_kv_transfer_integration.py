"""Compare SenseNova SRT thinking with native replay and transferred prefix KV."""

import argparse
import json
from pathlib import Path


def load_records(directory: Path) -> list[dict]:
    return json.loads((directory / "profile" / "records.json").read_text())


def thinking_rows(records: list[dict]) -> dict[tuple[str, int], dict]:
    return {(row["case"], row["seed"]): row for row in records if row["case"] != "off"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--transfer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay = thinking_rows(load_records(args.replay_dir))
    transfer = thinking_rows(load_records(args.transfer_dir))
    if set(replay) != set(transfer):
        raise RuntimeError("replay and transfer runs contain different cases")

    comparisons = {}
    for key in sorted(replay):
        replay_row = replay[key]
        transfer_row = transfer[key]
        if replay_row.get("thinking_backend") != "srt":
            raise RuntimeError(f"replay run did not use SRT: {replay_row}")
        if transfer_row.get("thinking_backend") != "srt":
            raise RuntimeError(f"transfer run did not use SRT: {transfer_row}")
        replay_ms = replay_row["stage_timings_ms"]["think_replay_prefill"]
        transfer_ms = transfer_row["stage_timings_ms"]["think_replay_prefill"]
        name = f"{key[0]}/seed-{key[1]}"
        comparisons[name] = {
            "reasoning_tokens_match": replay_row["reasoning_tokens"]
            == transfer_row["reasoning_tokens"],
            "think_text_hash_match": replay_row["think_text_sha256"]
            == transfer_row["think_text_sha256"],
            "image_hash_match": replay_row["image_sha256"]
            == transfer_row["image_sha256"],
            "transfer_used": transfer_row.get("srt_kv_transfer_used") is True,
            "replay_used_transfer": replay_row.get("srt_kv_transfer_used") is True,
            "replay_prefill_ms": replay_ms,
            "transfer_prefill_ms": transfer_ms,
            "speedup": round(replay_ms / transfer_ms, 3),
        }

    passed = all(
        row["reasoning_tokens_match"]
        and row["think_text_hash_match"]
        and row["image_hash_match"]
        and row["transfer_used"]
        and not row["replay_used_transfer"]
        for row in comparisons.values()
    )
    report = {"passed": passed, "comparisons": comparisons}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
