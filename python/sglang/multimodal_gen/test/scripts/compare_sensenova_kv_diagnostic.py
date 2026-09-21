"""Compare one native replay KV layer with the matching SRT diagnostic dump."""

import argparse
import json
from pathlib import Path

import torch


def error_metrics(
    left: torch.Tensor, right: torch.Tensor, *, atol: float, rtol: float
) -> dict:
    left = left.float()
    right = right.float()
    absolute = (left - right).abs()
    denominator = torch.maximum(left.abs(), right.abs()).clamp_min(1e-6)
    within_tolerance = absolute <= atol + rtol * right.abs()
    quantiles = torch.quantile(
        absolute.flatten(), torch.tensor([0.5, 0.95, 0.99, 0.999])
    ).tolist()
    return {
        "max_abs_error": float(absolute.max().item()),
        "mean_abs_error": float(absolute.mean().item()),
        "abs_error_percentiles": {
            name: float(value)
            for name, value in zip(("p50", "p95", "p99", "p99.9"), quantiles)
        },
        "max_relative_error": float((absolute / denominator).max().item()),
        "out_of_tolerance_count": int((~within_tolerance).sum().item()),
        "out_of_tolerance_pct": float(
            (~within_tolerance).float().mean().mul(100).item()
        ),
        "allclose": bool(within_tolerance.all().item()),
    }


def tensor_metrics(
    left: torch.Tensor, right: torch.Tensor, *, atol: float, rtol: float
) -> dict:
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
    return {
        "native_shape": list(left.shape),
        "srt_shape": list(right.shape),
        "native_dtype": native_dtype,
        "srt_dtype": srt_dtype,
        "dtype_matches": dtype_matches,
        "shape_matches": True,
        **error_metrics(left, right, atol=atol, rtol=rtol),
    }


def numerically_compatible(
    metrics: dict,
    *,
    max_abs_error: float,
    max_mean_abs_error: float,
    max_out_of_tolerance_pct: float,
) -> bool:
    return metrics["allclose"] or all(
        (
            metrics["max_abs_error"] <= max_abs_error,
            metrics["mean_abs_error"] <= max_mean_abs_error,
            metrics["out_of_tolerance_pct"] <= max_out_of_tolerance_pct,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.03)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--max-abs-error", type=float, default=0.125)
    parser.add_argument("--max-mean-abs-error", type=float, default=0.005)
    parser.add_argument("--max-out-of-tolerance-pct", type=float, default=0.001)
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

    keys = tensor_metrics(native["keys"], srt["keys"], atol=args.atol, rtol=args.rtol)
    values = tensor_metrics(
        native["values"], srt["values"], atol=args.atol, rtol=args.rtol
    )
    if keys["shape_matches"]:
        half = native["keys"].shape[-1] // 2
        keys["temporal_half"] = error_metrics(
            native["keys"][..., :half],
            srt["keys"][..., :half],
            atol=args.atol,
            rtol=args.rtol,
        )
        keys["spatial_half"] = error_metrics(
            native["keys"][..., half:],
            srt["keys"][..., half:],
            atol=args.atol,
            rtol=args.rtol,
        )
    for metrics in (keys, values):
        if metrics["shape_matches"]:
            metrics["compatible"] = numerically_compatible(
                metrics,
                max_abs_error=args.max_abs_error,
                max_mean_abs_error=args.max_mean_abs_error,
                max_out_of_tolerance_pct=args.max_out_of_tolerance_pct,
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
        "compatibility_limits": {
            "max_abs_error": args.max_abs_error,
            "max_mean_abs_error": args.max_mean_abs_error,
            "max_out_of_tolerance_pct": args.max_out_of_tolerance_pct,
        },
        "keys": keys,
        "values": values,
    }
    report["passed"] = all(
        (
            report["layer_id_matches"],
            token_hash_matches,
            keys["dtype_matches"],
            keys.get("compatible", False),
            values["dtype_matches"],
            values.get("compatible", False),
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
