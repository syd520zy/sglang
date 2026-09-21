"""Report SenseNova checkpoint bytes relevant to shared SRT thinking.

The script reads safetensors headers only. It does not materialize parameters and
can therefore run while the model is stopped or while the GPU is in use.
"""

import argparse
import json
import math
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    size_bytes: int

    @property
    def parameters(self) -> int:
        return math.prod(self.shape)


def read_safetensors_header(
    path: Path, allowed_names: set[str] | None = None
) -> list[TensorInfo]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"invalid safetensors header in {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header = json.loads(handle.read(header_length))

    tensors = []
    for name, metadata in header.items():
        if name == "__metadata__" or (
            allowed_names is not None and name not in allowed_names
        ):
            continue
        start, end = metadata["data_offsets"]
        tensors.append(
            TensorInfo(
                name=name,
                shape=tuple(metadata["shape"]),
                dtype=metadata["dtype"],
                size_bytes=end - start,
            )
        )
    return tensors


def resolve_model_path(model: str, revision: str | None, allow_download: bool) -> Path:
    local_path = Path(model).expanduser()
    if local_path.exists():
        return local_path.resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model,
            revision=revision,
            local_files_only=not allow_download,
        )
    )


def checkpoint_tensors(model_path: Path) -> list[TensorInfo]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        names_by_file = defaultdict(set)
        for name, relative_path in weight_map.items():
            names_by_file[relative_path].add(name)
        tensors = []
        for relative_path, names in sorted(names_by_file.items()):
            tensors.extend(read_safetensors_header(model_path / relative_path, names))
        return tensors

    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors checkpoint found in {model_path}")
    return [tensor for path in files for tensor in read_safetensors_header(path)]


def weight_group(name: str) -> str:
    if name.startswith("language_model."):
        if "_mot_gen" in name:
            return "language_generation"
        if ".embed_tokens." in name or name.startswith("language_model.lm_head."):
            return "language_io"
        return "language_understanding"
    if name.startswith("vision_model."):
        return "vision_understanding"
    if name.startswith("fm_modules.vision_model_mot_gen."):
        return "vision_generation"
    if name.startswith("fm_modules."):
        return "flow_matching"
    return "other"


def srt_selects(name: str) -> bool:
    return name.startswith("language_model.") and "_mot_gen" not in name


def statistics(tensors: list[TensorInfo]) -> dict:
    size_bytes = sum(tensor.size_bytes for tensor in tensors)
    return {
        "tensor_count": len(tensors),
        "parameter_count": sum(tensor.parameters for tensor in tensors),
        "bytes": size_bytes,
        "gib": round(size_bytes / (1 << 30), 3),
    }


def fusion_summary(tensors: list[TensorInfo]) -> dict:
    by_name = {tensor.name: tensor for tensor in tensors}
    qkv_sets = []
    gate_up_sets = []
    for name in by_name:
        if name.endswith(".self_attn.q_proj.weight"):
            peers = [
                name,
                name.replace(".q_proj.", ".k_proj."),
                name.replace(".q_proj.", ".v_proj."),
            ]
            if all(peer in by_name for peer in peers):
                qkv_sets.append(peers)
        elif name.endswith(".mlp.gate_proj.weight"):
            peers = [name, name.replace(".gate_proj.", ".up_proj.")]
            if all(peer in by_name for peer in peers):
                gate_up_sets.append(peers)

    def summarize_sets(items):
        return {
            "set_count": len(items),
            "bytes": sum(by_name[name].size_bytes for item in items for name in item),
            "gib": round(
                sum(by_name[name].size_bytes for item in items for name in item)
                / (1 << 30),
                3,
            ),
        }

    return {
        "q_k_v_to_qkv_proj": summarize_sets(qkv_sets),
        "gate_up_to_gate_up_proj": summarize_sets(gate_up_sets),
        "requires_contiguous_relayout": bool(qkv_sets or gate_up_sets),
    }


def kv_handoff_summary(model_path: Path) -> dict | None:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    llm_config = config.get("llm_config") or config.get("text_config") or config
    layer_count = llm_config.get("num_hidden_layers")
    kv_head_count = llm_config.get("num_key_value_heads")
    head_dim = llm_config.get("head_dim")
    if (
        head_dim is None
        and llm_config.get("hidden_size")
        and llm_config.get("num_attention_heads")
    ):
        head_dim = llm_config["hidden_size"] // llm_config["num_attention_heads"]
    if not layer_count or not kv_head_count or not head_dim:
        return None

    bytes_per_token = int(layer_count) * 2 * int(kv_head_count) * int(head_dim) * 2
    return {
        "dtype": "bfloat16",
        "layer_count": int(layer_count),
        "kv_head_count": int(kv_head_count),
        "head_dim": int(head_dim),
        "bytes_per_token_per_request": bytes_per_token,
        "mib_per_token_per_request": round(bytes_per_token / (1 << 20), 4),
        "sequence_estimates": {
            str(sequence_length): {
                "batch_1_mib": round(bytes_per_token * sequence_length / (1 << 20), 2),
                "batch_2_mib": round(
                    bytes_per_token * sequence_length * 2 / (1 << 20), 2
                ),
            }
            for sequence_length in (256, 512, 1024, 4096)
        },
    }


def build_report(model_path: Path, tensors: list[TensorInfo]) -> dict:
    groups = defaultdict(list)
    for tensor in tensors:
        groups[weight_group(tensor.name)].append(tensor)
    srt_tensors = [tensor for tensor in tensors if srt_selects(tensor.name)]
    total = statistics(tensors)
    srt = statistics(srt_tensors)
    return {
        "model_path": str(model_path),
        "checkpoint": total,
        "groups": {name: statistics(group) for name, group in sorted(groups.items())},
        "current_managed_srt": {
            "selection": "language_model.* excluding names containing _mot_gen",
            **srt,
            "two_process_weight_bytes": total["bytes"] + srt["bytes"],
            "two_process_weight_gib": round(
                (total["bytes"] + srt["bytes"]) / (1 << 30), 3
            ),
        },
        "split_with_kv_handoff": {
            "weight_bytes": total["bytes"],
            "weight_gib": total["gib"],
            "theoretical_savings_bytes": srt["bytes"],
            "theoretical_savings_gib": srt["gib"],
            "requires_main_dense_weights": False,
        },
        "srt_layout": fusion_summary(srt_tensors),
        "kv_handoff": kv_handoff_summary(model_path),
        "interpretation": [
            "Checkpoint bytes estimate parameter storage only; KV, CUDA graphs, allocator reserve and activations are measured separately.",
            "The SRT selection is measured before loader-side tied-weight deduplication.",
            "The split estimate assumes SRT exports text-prefix KV in a layout accepted by the image-generation process.",
            "Fused QKV and gate/up tensors have the same byte count but cannot directly alias the current separate main-model tensors.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sensenova/SenseNova-U1.5-8B-MoT")
    parser.add_argument("--revision")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model_path = resolve_model_path(args.model, args.revision, args.allow_download)
    report = build_report(model_path, checkpoint_tensors(model_path))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
