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

重点看 `think_decode_ms_per_token` 和 `denoise_delta_vs_off_ms`。首次 4090 测试约为 92–93 ms/token，而去噪耗时没有明显增长；优化目标是降低文本逐 token 解码耗时。

对比优化前后的文本解码，分别运行：

```bash
SENSENOVA_TEXT_ATTN_BACKEND=eager SENSENOVA_THINK_KV_CACHE=dynamic \
OUTPUT_DIR=/workspace/sensenova-thinking-profile-baseline \
  bash python/sglang/multimodal_gen/test/scripts/profile_sensenova_thinking_4090.sh

SENSENOVA_TEXT_ATTN_BACKEND=eager SENSENOVA_THINK_KV_CACHE=preallocated \
OUTPUT_DIR=/workspace/sensenova-thinking-profile-optimized \
  bash python/sglang/multimodal_gen/test/scripts/profile_sensenova_thinking_4090.sh
```

两次运行使用相同提示词、种子和 token 上限。对比 `profile/summary.json` 中的 `think_decode_ms_per_token` 和 `profile/records.json` 中相同 case/seed 的 `think_text_sha256`；散列一致表示思考文本逐字一致。4090 交叉测试中，单独开启预分配 KV 缓存保持了 64-token 思考文本一致，单独开启 SDPA 则改变了文本且未带来速度收益。因此 CUDA 思考注意力默认使用 eager，SDPA 仅在显式设置 `SENSENOVA_TEXT_ATTN_BACKEND=sdpa` 时启用；预分配缓存仍为默认。此优化只作用于 CUDA 的思考文本解码，普通前缀和图像去噪 attention 路径保持原样。

下一阶段测试 MoE 单 token 分发。原路径逐个检查所有专家；实验路径只处理路由选中的 top-k 专家。实验路径默认关闭，只在 CUDA 推理且输入为单 token 时生效：

```bash
BUDGETS=64,128,256 REPEATS=2 SENSENOVA_MOE_SINGLE_TOKEN_DISPATCH=all \
OUTPUT_DIR=/workspace/sensenova-thinking-moe-baseline \
  bash python/sglang/multimodal_gen/test/scripts/profile_sensenova_thinking_4090.sh

BUDGETS=64,128,256 REPEATS=2 SENSENOVA_MOE_SINGLE_TOKEN_DISPATCH=topk \
OUTPUT_DIR=/workspace/sensenova-thinking-moe-topk \
  bash python/sglang/multimodal_gen/test/scripts/profile_sensenova_thinking_4090.sh
```

先核对两组的单测和每个 case/seed 的 `reasoning_tokens`、`think_text_sha256`，再比较 `think_decode_ms_per_token`。如文本不一致，不应只凭耗时启用实验路径。
