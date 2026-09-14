"""CUDA BF16 prefix/KV/two-step denoise equivalence, including unequal prefixes."""

from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache

from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.configuration_neo_chat import (
    NEOLLMConfig,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_neo_chat import (
    NEOChatModel,
    prepare_flash_kv_cache,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3 import (
    Qwen3Attention,
    create_block_causal_mask,
    set_attn_backend,
)


@torch.no_grad()
def check_attention(cfg_scale, expand_query_mask):
    config = NEOLLMConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng():
        torch.manual_seed(23)
        attention = Qwen3Attention(config, layer_idx=0).eval()
    generator = torch.Generator(device="cuda").manual_seed(31)
    text = torch.randn(2, 5, 64, generator=generator)
    image = torch.randn(2, 3, 64, generator=generator)
    unconditional = torch.randn(1, 2, 64, generator=generator)
    helper = SimpleNamespace(device=torch.device("cuda"))

    def run(prefix, lengths, image_states):
        batch_size, width, _ = prefix.shape
        positions = torch.arange(width).expand(batch_size, -1)
        indexes = torch.stack(
            [positions, torch.zeros_like(positions), torch.zeros_like(positions)], dim=1
        )
        valid = positions < torch.tensor(lengths)[:, None]
        cache = DynamicCache(config=config)
        attention.forward_und(
            prefix, indexes, create_block_causal_mask(positions, valid), cache
        )
        prefix_keys = cache.layers[0].keys.clone()
        prepare_flash_kv_cache(cache, current_len=3, batch_size=batch_size)
        image_indexes = NEOChatModel._build_t2i_image_indexes(
            helper, 1, 3, torch.tensor(lengths), torch.device("cuda")
        )
        mask = torch.cat([valid, torch.ones(batch_size, 3, dtype=torch.bool)], dim=1)
        mask = mask[:, None, None, :]
        if expand_query_mask:
            mask = mask.expand(-1, -1, 3, -1).contiguous()
        outputs = []
        for _ in range(2):
            image_states, _ = attention.forward_gen(
                image_states,
                image_indexes,
                mask,
                cache,
                update_cache=False,
            )
            outputs.append(image_states)
        torch.testing.assert_close(cache.layers[0].keys, prefix_keys)
        return prefix_keys, outputs

    keys, batched = run(text, [2, 5], image)
    if cfg_scale > 1:
        _, uncond_batch = run(unconditional.expand(2, -1, -1), [2, 2], image)
    for i, length in enumerate([2, 5]):
        single_keys, single = run(text[i : i + 1, :length], [length], image[i : i + 1])
        torch.testing.assert_close(
            keys[i : i + 1, :, :length], single_keys, atol=0.02, rtol=0.02
        )
        if cfg_scale > 1:
            _, uncond_single = run(unconditional, [2], image[i : i + 1])
        for step in range(2):
            actual, expected = batched[step][i : i + 1], single[step]
            if cfg_scale > 1:
                actual = uncond_batch[step][i : i + 1] + cfg_scale * (
                    actual - uncond_batch[step][i : i + 1]
                )
                expected = uncond_single[step] + cfg_scale * (
                    expected - uncond_single[step]
                )
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    set_attn_backend("sdpa")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        for cfg in (1.0, 4.0):
            for expanded in (False, True):
                check_attention(cfg, expanded)
                print(f"PASS: cfg={cfg}, expanded_mask={expanded}")
    print("All four CUDA BF16 attention checks passed (atol=rtol=0.02).")
