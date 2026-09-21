"""Compare one native replay KV layer with the matching SRT diagnostic dump."""

import argparse
import json
from pathlib import Path

import torch


def tensor_metrics(left: torch.Tensor, right: torch.Tensor) -> dict:
    shape_matches = left.shape == right.shape
    if not shape_matches:
        return {
            "native_shape": list(left.shape),
            "srt_shape": list(right.shape),
            "native_dtype": str(left.dtype),
            "srt_dtype": str(right.dtype),
            "dtype_matches": left.dtype == right.dtype,
            "shape_matches": False,
            "allclose": False,
        }
    native_dtype = str(left.dtype)
    srt_dtype = str(right.dtype)
    dtype_matches = left.dtype == right.dtype
    left = left.float()
    right = right.float()
    absolute = (left - right).abs()
    denominator = torch.maximum(left.abs(), right.abs()).clamp_min(1e-6)
    return {
        "native_shape": list(left.shape),
        "srt_shape": list(right.shape),
        "native_dtype": native_dtype,
        "srt_dtype": srt_dtype,
        "dtype_matches": dtype_matches,
        "shape_matches": True,
        "max_abs_error": float(absolute.max().item()),
        "mean_abs_error": float(absolute.mean().item()),
        "max_relative_error": float((absolute / denominator).max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.03)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    native_files = sorted(args.directory.glob("*-native.pt"))
    srt_files = sorted(args.directory.glob("*-srt.pt"))
    if len(native_files) != 1 or len(srt_files) != 1:
        raise RuntimeError(
            "expected exactly one native/SRT KV dump pair, got "
            f"{len(native_files)} native and {len(srt_files)} SRT files"
        )
    native = torch.load(native_files[0], map_location="cpu", weights_only=True)
    srt = torch.load(srt_files[0], map_location="cpu", weights_only=True)
    if native_files[0].stem.removesuffix("-native") != srt_files[0].stem.removesuffix(
        "-srt"
    ):
        raise RuntimeError("native and SRT dump ids do not match")

    keys = tensor_metrics(native["keys"], srt["keys"])
    values = tensor_metrics(native["values"], srt["values"])
    if keys["shape_matches"]:
        keys["allclose"] = torch.allclose(
            native["keys"].float(),
            srt["keys"].float(),
            atol=args.atol,
            rtol=args.rtol,
        )
    if values["shape_matches"]:
        values["allclose"] = torch.allclose(
            native["values"].float(),
            srt["values"].float(),
            atol=args.atol,
            rtol=args.rtol,
        )
    token_hash_matches = native["token_sha256"] == srt["token_sha256"]
    report = {
        "native_file": str(native_files[0]),
        "srt_file": str(srt_files[0]),
        "layer_id_matches": native["layer_id"] == srt["layer_id"],
        "token_hash_matches": token_hash_matches,
        "token_count": len(native["token_ids"]),
        "atol": args.atol,
        "rtol": args.rtol,
        "keys": keys,
        "values": values,
    }
    report["passed"] = all(
        (
            report["layer_id_matches"],
            token_hash_matches,
            keys["dtype_matches"],
            keys["allclose"],
            values["dtype_matches"],
            values["allclose"],
        )
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
