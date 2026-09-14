"""Compare matching B1/B2 smoke images and end-to-end throughput."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", default="/workspace/sensenova-gpu-results")
args = parser.parse_args()
root = Path(args.results)
report = {"images": []}
for cfg in (1, 4):
    left, right = (root / f"smoke-b{b}-cfg{cfg}" for b in (1, 2))
    a, b = (json.loads((p / "result.json").read_text()) for p in (left, right))
    assert a["successful"] == b["successful"] == 4 and a["failed"] == b["failed"] == 0
    for x, y in zip(a["requests"], b["requests"]):
        assert (x["index"], x["seed"], x["prompt"]) == (
            y["index"],
            y["seed"],
            y["prompt"],
        )
        name = f"{x['index']:03d}.png"
        with Image.open(left / name) as im_a, Image.open(right / name) as im_b:
            diff = (
                np.asarray(im_a.convert("RGB"), dtype=np.float32)
                - np.asarray(im_b.convert("RGB"), dtype=np.float32)
            ) / 255
        report["images"].append(
            {
                "cfg": cfg,
                "image": name,
                "max_abs": float(np.abs(diff).max()),
                "mean_abs": float(np.abs(diff).mean()),
                "rmse": float(np.sqrt(np.mean(diff**2))),
            }
        )
if all((root / f"perf-b{b}/result.json").exists() for b in (1, 2)):
    a, b = (json.loads((root / f"perf-b{i}/result.json").read_text()) for i in (1, 2))
    assert a["failed"] == b["failed"] == 0 and a["successful"] == b["successful"] == 4
    report["performance"] = {
        "b1_duration_s": a["duration_s"],
        "b2_duration_s": b["duration_s"],
        "throughput_gain_pct": (b["outputs_per_s"] / a["outputs_per_s"] - 1) * 100,
        "b1_mean_latency_s": a["mean_latency_s"],
        "b2_mean_latency_s": b["mean_latency_s"],
    }
print(json.dumps(report, indent=2))
(root / "comparison.json").write_text(json.dumps(report, indent=2))
print(
    "Pixel differences are diagnostic only. Inspect image pairs; this is not a perceptual quality pass/fail test."
)
