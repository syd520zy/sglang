# SenseNova-U1 GPU KV 量化：设计与验证

## 目标与开发阶段

优先验证 RTX 4090（SM89），其次是 A800（SM80）。采用 INT8 存储与 FP16/BF16 attention 计算，不依赖 Hopper 或 FP8 attention。功能默认关闭，真实 GPU 正确性、画质和吞吐通过验证后，才能判断部署收益。

1. **缓存与精度基线**：固定的、RoPE 之后的前缀 KV 只量化一次。K、V 分别使用对称量化，每个 token、每个 KV head 各有一个 FP32 scale。当前图像 KV 保持模型精度。进入去噪时释放原始前缀的高精度引用，检查误差、全零输入、batch 共享和存储大小。
2. **GPU 算子与生成接入**：同一个 attention 算子分块读取 INT8 前缀与高精度图像 KV，共用一次在线 softmax。反量化仅发生在算子内部，不向显存写入完整的反量化前缀，也不为每层保留完整的“前缀 + 当前图像”缓存。保留 GQA、模型缩放系数和图像双向 attention。首阶段接入 `t2i_generate`，覆盖 CFG 和 Think。
3. **验证与基准**：对照高精度及显式反量化参考，覆盖多步、非整块长度、共享前缀、GQA、FP16/BF16。分别测量转换、attention、摊销耗时和缓存大小；目标机器另测完整模型的 images/s、峰值显存、P95 延迟和生成质量。

## 后续计划与验收标准

变长请求 batching 和 NPU FIA 在其他尚未合入的 PR 中，本分支不引入这些改动。batching 合入后，补充每个样本的前缀长度和 GPU varlen 批处理，确保 padding 不会作为有效上下文参与量化或 attention。当前相同 prompt 的多图 batch 不能替代这项工作。

图生图、交错生成、NPU 量化、INT4、token 裁剪和合并属于后续阶段，本实现尚未启用。

INT8 将前缀数据从每元素 2 字节降到 1 字节，额外需要每 token/head 一个 FP32 scale。短前缀时，取消各层常驻的图像 KV 工作区可能比前缀量化本身省更多显存。但自定义算子也可能慢于 FlashAttention，必须分别报告显存与吞吐变化。

最终验收需要 RTX 4090 实机正确性、代表性画质检查和端到端吞吐收益。CPU 测试和 kernel 微基准均不能替代这些结果。

## 在 4090 机器获取代码

以下命令按 Linux / Bash 编写，直接克隆包含量化实现的开发分支：

```bash
git clone --branch feat/sensenova-gpu-kv-quant --single-branch https://github.com/syd520zy/sglang.git
cd sglang
git log -1 --oneline
```

本分支基于 main 的 `faaff1eca8876b8b77d8704f080c3ee2064b8b65` 开发。后续更新时，在工作区没有未提交修改的情况下运行 `git pull --ff-only`。

上述命令仅获取源码。运行前仍需按仓库安装说明配置 SGLang、PyTorch、Triton、FlashAttention 和测试依赖。完整模型验证另需模型权重；算子测试和微基准不需要权重。

## 启用方法

原生 T2I pipeline 使用 `--pipeline-config-path sensenova-int8.json`，JSON 内容为：

```json
{"sensenova_kv_cache_dtype": "int8"}
```

默认 `"auto"` 保留原模型精度缓存和 FlashAttention/SDPA 路径。Python pipeline 配置使用同名字段。直接调用模型时：

```python
images = model.t2i_generate(tokenizer, prompt, kv_cache_dtype="int8")
```

要求 NVIDIA SM80 或更新 GPU、Triton、FP16/BF16 模型精度。Prefill 和 Think 使用原缓存，结束后才转换；CFG 条件与无条件前缀分别量化。转换后缓存只用于去噪，不能再用于文本生成或交错追加 token。

## 验证步骤

在已安装依赖的环境中，从仓库根目录运行。先跑正确性测试；如果失败，保留完整报错，先定位问题再测性能。

```bash
python -m pytest test/registered/unit/mem_cache/test_sensenova_int8_cache.py -q
python -m pytest test/registered/kernel/attention/test_sensenova_int8_attention.py -q
python -m pytest python/sglang/multimodal_gen/test/unit/test_sensenova_u1.py -q
```

然后运行微基准，分别保存两种精度的结果：

```bash
python benchmark/kernels/attention/bench_sensenova_int8.py --batch-size 1 --prefix-length 256 --image-tokens 1024 --dtype bf16 > sensenova-int8-bf16.json
python benchmark/kernels/attention/bench_sensenova_int8.py --batch-size 1 --prefix-length 256 --image-tokens 1024 --dtype fp16 > sensenova-int8-fp16.json
```

后续扩展到 batch size 1/2/4、前缀长度 256/1024/4096、实际分辨率对应的图像 token 数。用 `--query-heads`、`--kv-heads`、`--head-dim` 指定真实模型参数，默认值只是合成配置。默认 `--baseline flash` 包含原路径的布局转换和拷贝；只有实际部署使用 SDPA 时才改为 `--baseline sdpa`。

| JSON 字段 | 含义 |
| --- | --- |
| `baseline_cache_bytes_per_layer` / `int8_cache_bytes_per_layer` | 每层缓存数据大小，不是整模型显存峰值 |
| `baseline_prepare_ms` / `int8_prepare_ms` | 缓存准备与量化转换耗时 |
| `baseline_attention_ms` / `int8_attention_ms` | attention 路径耗时 |
| `amortized_attention_speedup` | 按 `--steps` 摊销准备开销后的加速比，大于 1 才表示此微基准加速 |
| `relative_l2_error` | 相对高精度基线的输出 L2 误差 |

微基准同时保留两种缓存表示用于对照，不能用该进程显存峰值推算部署节省的显存。

完整模型验证时，固定权重、prompt、seed、分辨率、步数、CFG 和 dtype，对照 `auto`/`int8`。覆盖长 prompt 和 Think，记录 images/s、allocated/reserved 显存峰值及 P95 延迟，检查文字、细节和提示词遵循程度。

当前 main 服务端仍逐个执行请求，较大的直接模型 batch 不能证明服务端请求 batching 的收益。相关 PR 合入后需要重测。

请提供正确性日志、微基准 JSON、GPU 型号及驱动、PyTorch/CUDA、Triton、FlashAttention 版本。完整模型对照另附模型名称、运行参数、耗时和图片。

## 当前验证状态

- 5 个缓存测试断言通过独立 CPU 入口，覆盖 FP16/BF16 量化误差、原缓存引用释放、共享 batch 和无效输入。标准 pytest 在开发用 Windows 机器上受 SGLang 的 Linux `resource` 依赖阻断，不能声称完整测试套件通过。
- FP16 Triton 解释器通过空前缀、非整块长度、GQA 和共享 batch；小型 Dense、MoE 各两步去噪输出在容差内。这是 CPU 解释器验证，不是真实 GPU 执行。当前版本的 BF16 解释器 dot 不能替代硬件数值验证。
- SM80/SM89 的 FP16/BF16 离线 CUDA 编译通过，包括 head dimension 256。
- 8 组独立 CPU 流程测试覆盖 `auto`/`int8`、CFG 开关、Think 开关；真实 T2I 方法配合模拟模型组件，检查 Think 后转换、CFG 前缀独立、多步复用和原路径清理。
- RTX 4090 实机执行、BF16 GPU 数值一致性、完整模型画质、端到端吞吐仍未验证。功能默认关闭，暂不宣称性能收益。
