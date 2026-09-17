import base64
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/multimodal_gen/test/scripts/eval_sensenova_thinking_geneval.py"
)
SPEC = importlib.util.spec_from_file_location("sensenova_thinking_geneval", SCRIPT)
geneval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(geneval)


def test_geneval_generation_checks_paired_api_contract(monkeypatch):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    requests = []

    def urlopen(request, timeout):
        payload = json.loads(request.data)
        requests.append(payload)
        usage = (
            {"think_text": "plan</think>", "reasoning_tokens": 3}
            if payload["think_mode"]
            else {}
        )
        return io.BytesIO(
            json.dumps({"data": [{"b64_json": encoded}], "usage": usage}).encode()
        )

    monkeypatch.setattr(geneval.urllib.request, "urlopen", urlopen)
    args = SimpleNamespace(
        model="sensenova/test",
        width=8,
        height=8,
        steps=50,
        guidance_scale=4.0,
        max_think_tokens=256,
        timeout=30,
    )
    off_image, off = geneval.generate_one(
        "http://127.0.0.1:30000", "a photo of a cat", 42, "off", args
    )
    on_image, on = geneval.generate_one(
        "http://127.0.0.1:30000", "a photo of a cat", 42, "on", args
    )

    assert off_image == on_image == buffer.getvalue()
    assert off["reasoning_tokens"] == 0
    assert on["reasoning_tokens"] == 3
    assert requests[0]["seed"] == requests[1]["seed"] == 42
    assert requests[0]["think_mode"] is False
    assert requests[1]["think_mode"] is True
    assert requests[1]["max_think_tokens"] == 256
    assert all(request["output_format"] == "png" for request in requests)


def test_geneval_compare_requires_complete_paired_scores(tmp_path):
    rows = [
        {"tag": "counting", "prompt": "two cats"},
        {"tag": "position", "prompt": "cat left of dog"},
    ]
    (tmp_path / "evaluation_metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "tags": ["counting", "position"],
                "max_per_tag": 1,
                "selection_seed": 42,
                "smoke": False,
                "samples_per_prompt": 1,
                "max_think_tokens": 256,
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )
    scores = {"off": [False, True], "on": [True, True]}
    for mode in ("off", "on"):
        results = []
        for index, row in enumerate(rows):
            sample = tmp_path / mode / f"{index:05d}" / "samples" / "0000.png"
            sample.parent.mkdir(parents=True)
            sample.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "elapsed_seconds": 10 if mode == "off" else 15,
                        "reasoning_tokens": 0 if mode == "off" else 25,
                        "seed": 42,
                    }
                ),
                encoding="utf-8",
            )
            results.append(
                {
                    "filename": str(sample),
                    "prompt": row["prompt"],
                    "tag": row["tag"],
                    "correct": scores[mode][index],
                }
            )
        (tmp_path / f"{mode}-results.jsonl").write_text(
            "\n".join(json.dumps(row) for row in results), encoding="utf-8"
        )
    args = SimpleNamespace(
        output_dir=tmp_path,
        off_results=tmp_path / "off-results.jsonl",
        on_results=tmp_path / "on-results.jsonl",
    )
    geneval.compare(args)
    report = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert report["pairs"] == 2
    assert report["macro_off"] == 0.5
    assert report["macro_on"] == 1.0
    assert report["macro_delta"] == 0.5
    assert report["paired_on_wins"] == 1
    assert report["paired_on_losses"] == 0

    args.on_results.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete evaluation"):
        geneval.compare(args)


def test_geneval_generation_resumes_and_repairs_corrupt_records(tmp_path, monkeypatch):
    metadata = tmp_path / "source.jsonl"
    metadata.write_text(
        json.dumps({"tag": "counting", "prompt": "two cats"}), encoding="utf-8"
    )
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="PNG")
    image = buffer.getvalue()
    calls = []

    def fake_generate_one(base_url, prompt, seed, mode, args):
        calls.append(mode)
        return image, {
            "seed": seed,
            "elapsed_seconds": 1,
            "reasoning_tokens": 1 if mode == "on" else 0,
            "image_sha256": geneval.hashlib.sha256(image).hexdigest(),
        }

    monkeypatch.setattr(geneval, "generate_one", fake_generate_one)
    args = SimpleNamespace(
        metadata_file=metadata,
        output_dir=tmp_path / "results",
        base_url="http://127.0.0.1:30000",
        model="sensenova/test",
        width=8,
        height=8,
        steps=50,
        guidance_scale=4.0,
        max_think_tokens=256,
        samples_per_prompt=1,
        seed=42,
        tags=["counting"],
        max_per_tag=1,
        selection_seed=42,
        smoke=False,
    )
    geneval.generate(args)
    geneval.generate(args)
    assert calls == ["off", "on"]
    on_record = args.output_dir / "on" / "00000" / "samples" / "0000.json"
    on_record.write_text("broken", encoding="utf-8")
    geneval.generate(args)
    assert calls == ["off", "on", "on"]
