# SPDX-License-Identifier: Apache-2.0
import unittest
from copy import deepcopy

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.sensenova_int8 import int8_prefix_attention
from sglang.multimodal_gen.runtime.layers.kvcache.sensenova import (
    Int8DenoisingCache,
    quantize_prefix,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")


class TestSenseNovaInt8Attention(CustomTestCase):
    @torch.inference_mode()
    def test_dense_and_moe_denoising_use_read_only_cache(self):
        from transformers.cache_utils import DynamicCache

        from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.configuration_neo_chat import (
            NEOLLMConfig,
            NEOMoELLMConfig,
        )
        from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3 import (
            Qwen3Model,
        )
        from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3_moe import (
            Qwen3MoeModel,
        )

        for config_cls, model_cls in (
            (NEOLLMConfig, Qwen3Model),
            (NEOMoELLMConfig, Qwen3MoeModel),
        ):
            with self.subTest(backbone=model_cls.__name__):
                torch.manual_seed(42)
                config = config_cls(
                    vocab_size=32,
                    hidden_size=128,
                    intermediate_size=128,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=64,
                    max_position_embeddings=128,
                    num_experts=2,
                    num_experts_per_tok=1,
                    moe_intermediate_size=64,
                    attn_implementation="eager",
                )
                model = model_cls(config).cuda().half().eval()
                source = DynamicCache(config=config)
                for layer in range(2):
                    k = torch.randn(1, 1, 7, 64, device="cuda", dtype=torch.float16)
                    source.update(k, torch.randn_like(k), layer)
                compressed = Int8DenoisingCache.from_cache(deepcopy(source))
                indexes = (
                    torch.stack((torch.ones(9), torch.arange(9), torch.zeros(9)))
                    .long()
                    .cuda()
                )
                for _ in range(2):
                    inputs = torch.randn(1, 9, 128, device="cuda", dtype=torch.float16)
                    kwargs = {
                        "inputs_embeds": inputs,
                        "image_gen_indicators": torch.ones(
                            1, 9, device="cuda", dtype=torch.bool
                        ),
                        "indexes": indexes,
                        "attention_mask": {"full_attention": None},
                        "update_cache": False,
                        "use_cache": True,
                    }
                    ref = model(past_key_values=source, **kwargs).last_hidden_state
                    out = model(past_key_values=compressed, **kwargs).last_hidden_state
                    torch.testing.assert_close(out, ref, atol=0.015, rtol=0.015)
                    self.assertEqual(compressed.get_seq_length(), 7)
                    self.assertTrue(
                        all(
                            layer.keys.dtype == torch.int8
                            for layer in compressed.layers
                        )
                    )

    def test_matches_dequantized_and_full_precision_attention(self):
        torch.manual_seed(123)
        for dtype in (torch.float16, torch.bfloat16):
            for p, s, d, shared in (
                (0, 17, 64, True),
                (1, 33, 80, False),
                (65, 71, 128, True),
                (3, 5, 256, False),
            ):
                with self.subTest(dtype=dtype, prefix=p, image=s, dim=d, shared=shared):
                    # Transposed views match the noncontiguous model projections.
                    q = torch.randn(2, s, 4, d, device="cuda", dtype=dtype).transpose(
                        1, 2
                    )
                    k = torch.randn(2, s, 2, d, device="cuda", dtype=dtype).transpose(
                        1, 2
                    )
                    v = torch.randn_like(k)
                    pk = torch.randn(
                        1 if shared else 2, 2, p, d, device="cuda", dtype=dtype
                    )
                    pv = torch.randn_like(pk)
                    ik, ks = quantize_prefix(pk)
                    iv, vs = quantize_prefix(pv)
                    saved_k, saved_v = ik.clone(), iv.clone()
                    for step in range(2):
                        current_k = k + step * 0.1
                        out = int8_prefix_attention(
                            q, current_k, v, ik, iv, ks, vs, 0.17
                        )
                        results = []
                        for ref_k, ref_v in (
                            (
                                (ik.float() * ks[..., None]).to(dtype),
                                (iv.float() * vs[..., None]).to(dtype),
                            ),
                            (pk, pv),
                        ):
                            all_k = torch.cat(
                                (ref_k.expand(2, -1, -1, -1), current_k), dim=2
                            )
                            all_v = torch.cat((ref_v.expand(2, -1, -1, -1), v), dim=2)
                            results.append(
                                F.scaled_dot_product_attention(
                                    q.float(),
                                    all_k.float().repeat_interleave(2, 1),
                                    all_v.float().repeat_interleave(2, 1),
                                    scale=0.17,
                                ).transpose(1, 2)
                            )
                        torch.testing.assert_close(
                            out.float(), results[0], atol=0.015, rtol=0.015
                        )
                        relative_error = (out.float() - results[1]).norm() / results[
                            1
                        ].norm()
                        self.assertLess(relative_error.item(), 0.025)
                    torch.testing.assert_close(ik, saved_k, rtol=0, atol=0)
                    torch.testing.assert_close(iv, saved_v, rtol=0, atol=0)

    def test_prefix_and_image_share_softmax(self):
        q = torch.ones(1, 1, 5, 64, device="cuda", dtype=torch.float16)
        k, v = -q, torch.zeros_like(q)
        pk = torch.ones(1, 1, 3, 64, device="cuda", dtype=q.dtype)
        pv = torch.full_like(pk, 7)
        ik, ks = quantize_prefix(pk)
        iv, vs = quantize_prefix(pv)
        out = int8_prefix_attention(q, k, v, ik, iv, ks, vs, 0.125)
        torch.testing.assert_close(out, torch.full_like(out, 7), atol=0.002, rtol=0)

    def test_rejects_bad_scale_and_head_shapes(self):
        q = torch.zeros(1, 4, 8, 64, device="cuda", dtype=torch.float16)
        k = q[:, :2]
        pk, scale = quantize_prefix(k)
        with self.assertRaisesRegex(ValueError, "scales"):
            int8_prefix_attention(q, k, k, pk, pk, scale.half(), scale, 0.125)
        with self.assertRaisesRegex(ValueError, "GQA"):
            int8_prefix_attention(q[:, :3], k, k, pk, pk, scale, scale, 0.125)


if __name__ == "__main__":
    unittest.main()
