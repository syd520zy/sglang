# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.layers.kvcache.sensenova import (
    Int8DenoisingCache,
    quantize_prefix,
    validate_int8_kv_device,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSenseNovaInt8Cache(CustomTestCase):
    def test_rejects_non_cuda_before_generation(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            validate_int8_kv_device(torch.device("cpu"), torch.bfloat16)

    def test_quantization_error_and_storage(self):
        torch.manual_seed(42)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.randn(2, 4, 17, 128).to(dtype)
                x[:, :, 0] = 0
                x[:, :, 1, 3] = 40
                q, scale = quantize_prefix(x)
                self.assertEqual(q.dtype, torch.int8)
                self.assertEqual(scale.shape, (2, 4, 17))
                self.assertTrue(torch.isfinite(scale).all())
                error = (q.float() * scale[..., None] - x.float()).abs()
                self.assertTrue((error <= scale[..., None] * 0.501 + 1e-6).all())
                self.assertEqual(q[:, :, 0].count_nonzero(), 0)
                self.assertLess(q.nbytes + scale.nbytes, x.nbytes * 0.55)

    def test_consume_shared_prefix(self):
        key = torch.randn(1, 2, 13, 64, dtype=torch.float16).expand(3, -1, -1, -1)
        value = torch.randn_like(key)
        source = SimpleNamespace(layers=[SimpleNamespace(keys=key, values=value)])
        cache = Int8DenoisingCache.from_cache(source)
        self.assertIsNone(source.layers[0].keys)
        self.assertIsNone(source.layers[0].values)
        self.assertEqual(cache.get_seq_length(), 13)
        self.assertEqual(cache.layers[0].keys.shape[0], 1)
        self.assertEqual(cache.layers[0].values.shape[0], 3)
        with self.assertRaisesRegex(RuntimeError, "read-only"):
            cache.update(key, value, 0)

    def test_invalid_prefix_does_not_consume_cache(self):
        key = torch.zeros(1, 2, 3, 64, dtype=torch.float16)
        good = SimpleNamespace(keys=key, values=key)
        bad = SimpleNamespace(keys=key, values=key[:, :, :2])
        source = SimpleNamespace(layers=[good, bad])
        with self.assertRaisesRegex(ValueError, "same shape"):
            Int8DenoisingCache.from_cache(source)
        self.assertIs(source.layers[0].keys, key)

    def test_empty_and_unsupported_prefix(self):
        q, scale = quantize_prefix(torch.empty(1, 2, 0, 64, dtype=torch.float16))
        self.assertEqual(q.shape, (1, 2, 0, 64))
        self.assertEqual(scale.shape, (1, 2, 0))
        for x in (torch.ones(2, 3), torch.ones(1, 2, 3, 64, dtype=torch.int8)):
            with self.assertRaises(ValueError):
                quantize_prefix(x)


if __name__ == "__main__":
    unittest.main()
