# SenseNova 思考模式分阶段 Profiling

该脚本在 RTX 4090 上启动 SenseNova 服务，并以相同提示词和种子依次测试关闭思考、64、128、256 token 四种情况。Profiling 只在请求显式传入 `profile_stages=true` 时启用，普通请求不会增加 CUDA 同步。

运行：

```bash
cd /workspace/sglang
git switch feat/sensenova-thinking-mode
git pull --ff-only origin feat/sensenova-thinking-mode

REPEATS=2 BUDGETS=64,128,256 \
  bash python/sglang/multimodal_gen/test/scripts/profile_sensenova_thinking_4090.sh
```

常用覆盖项：

```bash
GPU_ID=0 SERVER_PORT=30000 WIDTH=1024 HEIGHT=1024 STEPS=50 \
REPEATS=3 BUDGETS=64,128,256 OUTPUT_DIR=/workspace/sensenova-thinking-profile \
  bash python/sglang/multimodal_gen/test/scripts/profile_sensenova_thinking_4090.sh
```

结果目录包含：

- `environment.txt`：代码版本、PyTorch、CUDA、GPU 信息。
- `server.log`：服务日志。
- `unit-tests.log`：相关单测结果。
- `profile/records.json`：每次请求的原始分阶段数据。
- `profile/summary.json`：各组均值以及相对关闭思考的增量。

`stage_timings_ms` 字段含义：

- `input_prepare`：提示词编码、索引和 attention mask 构造。
- `condition_prefill`：条件提示词 prefill。
- `think_decode`：逐 token 思考解码及结束标记写入。
- `cfg_prefill`：CFG 无条件分支 prefill。
- `denoise_prepare`：KV 布局转换、缓存预分配和噪声初始化。
- `denoise_loop`：全部图像去噪步骤。
- `total`：模型 `t2i_generate` 总耗时。

重点看 `think_decode_ms_per_token` 和 `denoise_delta_vs_off_ms`。前者随 token 上限明显上升，说明 DynamicCache 扩容或长上下文 attention 的影响较大；后者明显上升，说明思考 KV 加长了后续图像去噪。
