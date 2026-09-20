# SenseNova 思考模式分阶段 Profiling

单组 profiling 脚本会启动 SenseNova 服务，并以相同提示词和种子依次测试关闭思考、64、128、256 token 四种情况。Profiling 只在请求显式传入 `profile_stages=true` 时启用，普通请求不会增加 CUDA 同步。

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
- `think_decode`：SRT 或原生逐 token 思考解码。
- `think_replay_prefill`：SRT 返回 token 后，主模型一次性重建条件 KV；原生路径为 0。
- `cfg_prefill`：CFG 无条件分支 prefill。
- `denoise_prepare`：KV 布局转换、缓存预分配和噪声初始化。
- `denoise_loop`：全部图像去噪步骤。
- `total`：模型 `t2i_generate` 总耗时。

重点看 `think_decode_ms_per_token` 和 `denoise_delta_vs_off_ms`。首次 4090 测试约为 92–93 ms/token，而去噪耗时没有明显增长；优化目标是降低文本逐 token 解码耗时。

对比原生路径和 SRT 路径，分别运行：

```bash
SGLANG_SENSENOVA_THINKING_BACKEND=native \
OUTPUT_DIR=/workspace/sensenova-thinking-profile-baseline \
  bash python/sglang/multimodal_gen/test/scripts/profile_sensenova_thinking_4090.sh

SGLANG_SENSENOVA_THINKING_BACKEND=srt \
OUTPUT_DIR=/workspace/sensenova-thinking-profile-optimized \
  bash python/sglang/multimodal_gen/test/scripts/profile_sensenova_thinking_4090.sh
```

两次运行使用相同提示词、种子和 token 上限。先检查 `profile/records.json` 的 `thinking_backend`：优化组应为 `srt`，如果是 `native`，说明内部服务启动或调用失败，实际测试的是回退路径。确认后再对比 `profile/summary.json` 中的 `think_decode_ms_per_token`，以及相同 case/seed 的 `think_text_sha256`。SRT 子服务由主服务自动启动并固定使用 Triton attention，不依赖 FlashInfer attention 的版本；调用方仍只设置 `think_mode`，`--srt-encoder-url` 仅保留为外部部署覆盖项。官方 checkpoint 总大小约 35.1 GB，内部 SRT 还需约 18 GB 的稠密文本权重，因此 24/48 GB 4090 的权重容量本身就不足；启动失败时会自动回退原生路径，验证加速应换用 80 GB GPU。

A800 上可用对照脚本一次完成原生路径、SRT 路径及结果比较。比较阶段会校验两组实际使用的后端；SRT 发生回退时脚本会失败并提示检查 `srt/server.log`。

```bash
REPEATS=2 BUDGETS=64,128,256 WIDTH=1024 HEIGHT=1024 \
  bash python/sglang/multimodal_gen/test/scripts/compare_sensenova_thinking_srt_a800.sh
```

压缩完整结果目录：

```bash
bash python/sglang/multimodal_gen/test/scripts/pack_sensenova_thinking_results.sh \
  /workspace/sglang/sensenova-thinking-srt-a800-results/<时间戳>
```
