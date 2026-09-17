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

上述命令仅获取源码。运行前仍需按仓库安装说明配置 SGLang、PyTorch、Triton 和测试依赖（SDPA 微基准不需要 FA3）。完整模型验证另需模型权重；算子测试和微基准不需要权重。

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

后续扩展到 batch size 1/2/4、前缀长度 256/1024/4096、实际分辨率对应的图像 token 数。用 `--query-heads`、`--kv-heads`、`--head-dim` 指定真实模型参数，默认值只是合成配置。默认 `--baseline sdpa` 复现模型显式重复 GQA heads 的计算方式，包含布局转换、拷贝和输出 contiguous。每次同时报告原生 GQA SDPA 和未量化分离 KV 对照。可用 `--baseline sdpa-gqa` 选择原生 GQA 为主对照；可选 flash 需要 flash_attn_func，不自动切换 FA3。新版原生 GQA 对照包含输出 contiguous，与旧 JSON 并非完全同口径。

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
- 原版 RTX 4090 测试已通过：缓存 5 passed、算子 4 passed、模型 63 passed，包括 BF16 数值验证。本轮新增对照仍需实机验证，完整模型画质、端到端吞吐尚未验证。


## 第二阶段：对照与分块扫描

```bash
python benchmark/kernels/attention/bench_sensenova_int8.py --dtype bf16 --sweep-kernel > sensenova-int8-bf16-v2.json
python benchmark/kernels/attention/bench_sensenova_int8.py --dtype fp16 --sweep-kernel > sensenova-int8-fp16-v2.json
```

- `model_sdpa_attention_ms`：模型显式重复 KV heads 的 SDPA 路径。
- `native_gqa_attention_ms`：原生 GQA SDPA 路径。
- `unquantized_separated_attention_ms`：同一个 Triton 算子读取 FP16/BF16 前缀，不拼接 KV、不读 scale；仅用于 benchmark，未接入模型。
- `unquantized_separated_cache_bytes_per_layer`：原始共享前缀字节数，无额外准备分配。
- `kernel_sweep`：四组分块分别测未量化和 INT8，先检查相对 L2 误差不超过 0.025 再计时。资源不足记录 unsupported，其他异常直接报错。扫描不修改生产默认分块。

缓存字节数不含 attention 临时张量、输出和 allocator 开销。分离 KV 对照与 SDPA 同时改变了算子，时间差不能全部归因于取消拷贝；分离 KV 与 INT8 使用相同分块，适合分析量化代价。

首次 RTX 4090 结果（旧版原生 GQA SDPA）：BF16 基线/INT8 为 0.1766/0.2184 ms，FP16 为 0.1770/0.2155 ms，尚无吞吐收益证据。先复测默认尺寸，再扩展 batch 1/2/4、prefix 256/1024/4096 和实际图像 token 数。微基准不替代端到端图片/秒。


## 第三阶段：稳定计时和配置矩阵

新版 JSON 的 `benchmark_version` 为 3。所有 attention 路径先编译、检查精度，并使用相同的预热过程；默认 `--rounds 7`，每轮用固定随机种子打乱测量顺序，每次计时使用 Triton `do_bench(warmup=100, rep=300, return_mode="median")`，时间参数单位为毫秒。缓存准备单独采用相同的多轮流程。此流程不能消除 GPU 频率、温度或其他任务的干扰。

`timings` 与 `preparation_timings` 包含每轮中位耗时 `round_medians_ms`，以及这些值的中位数、最小值和最大值。范围描述轮间波动，不是置信区间或请求 P95。顶层耗时字段复用对应的中位数，不再额外计时；摊销加速比是这些中位数的组合，不是端到端测量。所有候选路径均在计时前检查有限输出与相对 L2 误差（上限 0.025），误差统一参照模型 SDPA。

激活现有环境并更新代码后执行：

```bash
cd /workspace/sglang
bash benchmark/kernels/attention/run_sensenova_int8_sweep.sh
```

可传入结果目录作为第一个参数。脚本先跑三个测试文件，再扫描两种精度 × batch 1/2/4 × prefix 256/1024/4096，共 18 组；每组扫描四个分块，图像 tokens 固定为 1024。默认 head 参数仍是合成配置，不代表所有模型。脚本不更新仓库、不安装依赖。

每组输出 JSON 和 stderr，`status.csv` 记录退出码。测试失败立即停止；某组 benchmark 失败则保留日志并继续其他组，最终以非零状态退出。运行时尽量保持 GPU 无其他任务；完成后提供整个结果目录，重点比较原生 GQA、未量化分离 KV、INT8 在相同配置下的耗时和波动，再选择端到端验证路径。


## 图像 KV INT8 实验（独立算子，尚未接入模型）

本实验量化每层、每步重新生成的图像 K/V，前缀保持 FP16/BF16。K 的输入位置必须在归一化和 RoPE 后，V 在投影后。一个 Triton kernel 同时量化 K/V，每 token/head 分别保存一个 FP32 scale；attention 内逐块反量化，并对前缀和图像使用同一个 softmax。Q、输出和矩阵乘法输入仍为 FP16/BF16，不减少 attention 的乘法量。

图像量化不使用 `sensenova_kv_cache_dtype=int8` 开关；该开关仍只控制之前的前缀缓存。本阶段用于先判断逐步量化成本能否得到回报，尚不能宣称完整模型画质或吞吐通过验证。

```bash
cd /workspace/sglang
bash benchmark/kernels/attention/run_sensenova_image_int8_sweep.sh
```

脚本覆盖 FP16/BF16 × batch 1/2 × 分辨率 512/1024/2048/4096，共 16 组，前缀固定 256。按有效 patch=32，对应图像 tokens 256/1024/4096/16384；这是默认配置假设，实际权重需要核对 `vision_config.patch_size * int(1 / downsample_ratio)`。单次运行可指定不同的有效 patch：

```bash
python benchmark/kernels/attention/bench_sensenova_int8.py --quantize-image --resolution 2048 --effective-patch-size 32 --dtype bf16 --batch-size 1 --rounds 7
```

- `image_quantization`：每步融合量化的耗时，含输出分配。
- `image_int8_attention`：已有 INT8 图像 KV 时的 attention 耗时。
- `image_int8_total`：实际连续执行量化和 attention 的总耗时（非两项中位数相加）。加速比使用这一项，与原生 GQA 和模型 SDPA 分别比较。
- `unquantized_separated`：相同 64×32、4 warps 分块，图像 KV 保持高精度。
- `image_fp_bytes_per_layer` / `image_int8_bytes_per_layer`：当前 batch 的图像 KV 表示大小，后者包含 scale；不是整模型或 allocator 峰值。
- `extra_peak_allocated_bytes`：比较缓存全部驻留时，每次调用额外产生的 allocated 峰值；不含驻留的原始输入，不能据此推算部署总显存。当前实现量化时原始图像 K/V 仍存活，会增加 INT8 张量和 scale；需后续模型接入和生命周期优化才能评估净显存收益。

所有 attention 对照先检查有限输出与相对 L2 误差 <= 0.025，再进行七轮计时。GPU 单测另比较显式反量化参考，覆盖空前缀、非连续输入、非整块长度、GQA、共享/独立前缀和两次图像 KV 更新。当前误差界限只用于合成数据筛查，不代表图像质量验收。

每组失败保留 stderr 和退出码，包括 OOM；其余配置继续执行。高分辨率可能耗时较长，失败的 JSON 可能为空，应先查看 status.csv。


图像实验本地验证：SM80/SM89、FP16/BF16、head dimension 128/256 的融合量化与 attention 离线编译通过；旧前缀分支离线编译及 FP16 解释器回归通过。图像 attention 的 FP16 解释器验证使用独立的高精度量化参考提供 INT8 输入，空前缀和非整块长度在容差内。本地解释器不能执行 libdevice 舍入函数，融合量化数值与 BF16 执行仍需新增 GPU 单测确认。以上均不是 4090 性能或端到端验证。
