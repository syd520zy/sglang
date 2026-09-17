# SenseNova 思考模式 GenEval 轻量对照验证

SenseNova-U1.5 官方技术报告使用了 GenEval、GenEval2、DPG-Bench、OneIG-Bench 和 Qwen-Image-Bench 等图像生成基准。本实验选 GenEval 官方提示词中的 **计数、相对位置、颜色属性绑定** 三类，每类用固定随机种子抽 30 条，共 90 条提示词。每条分别用相同图像种子生成关闭/开启思考两张图，共 180 张。先额外运行每类 1 条的功能检查，验证接口、PNG 格式和尺寸、思考文本闭合及 token 用量。抽样在生成前确定，失败后可原目录续跑。

这是一项轻量、成对的 A/B 实验。官方 GenEval 完整集有 553 条提示词，参考流程每条生成 4 张图；本实验仅用每条 1 张，而且只抽三类，因此分数**不能**与官方论文或排行榜分数直接比较。报告重点看同一提示词、同一图像种子下的开关差值和 95% bootstrap 区间。区间跨越 0 时，应报告“未观察到明确提升”，不能据此断言两种模式等效。GenEval 不评估中文排版、文字准确率或主观美感。

## 准备

在 4090 服务器上拉取当前分支和官方 [GenEval](https://github.com/djghosh13/geneval) 仓库。图像生成沿用已经验证通过的 SGLang 环境；官方 GenEval 评分器需单独环境，其 [安装说明](https://github.com/djghosh13/geneval#setup) 涉及 mmdet 2.x、mmcv 和 Mask2Former 权重。官方 `environment.yml` 锁定旧版 PyTorch/CUDA；如果评分器无法在 RTX 4090 运行，可以先在 4090 生成图像，再在 A800 上仅执行评分步骤。不要把评分器依赖装进正在运行的 SGLang 环境。

```bash
git clone https://github.com/djghosh13/geneval.git /workspace/geneval
cd /workspace/geneval
./evaluation/download_models.sh /workspace/geneval-models
```

按 GenEval 官方 README 建立评分器环境，设定 `GENEVAL_PYTHON` 指向其中的 Python。先用 `SCORE=0` 可以只验证生图，后续再评分。

## 4090 运行

在 SGLang 仓库根目录执行，按实际路径调整变量：

```bash
export GENEVAL_DIR=/workspace/geneval
export GENEVAL_MODEL_DIR=/workspace/geneval-models
export GENEVAL_PYTHON=/path/to/geneval-env/bin/python
export MODEL_PATH=sensenova/SenseNova-U1.5-8B-MoT
export OUTPUT_DIR=/workspace/sensenova-thinking-geneval-results/$(date +%Y%m%d-%H%M%S)
export GPU_ID=0
bash python/sglang/multimodal_gen/test/scripts/validate_sensenova_thinking_geneval_4090.sh
```

脚本先运行 SenseNova 单测与评测工具单测，再启动服务；功能检查通过后才生成 90 组成对图像。生成结束后停止服务，释放显存，再用官方评分器评分并写出 `benchmark/comparison.json`。默认 `1024×1024`、50 步、guidance scale 4.0、图像种子 42、思考预算 256。用 `MAX_PER_TAG=0` 可跑所选三类全集，用 `SAMPLES_PER_PROMPT=4` 可每条生成 4 张。参数改变后使用新的输出目录；同参数重跑会跳过已完成且哈希匹配的图像。

仅按前次 1024 分辨率单样本的关闭/开启思考耗时约 14.4/20.3 秒粗算，90 对生图约需 52 分钟；本实验的 256-token 思考预算和提示词内容会改变实际耗时，评分时间另计。

如果评分器尚未装好，可设置 `SCORE=0` 先生成图像。`OUTPUT_DIR/smoke` 是功能检查，`OUTPUT_DIR/benchmark` 是评分集；两者分开，功能检查不计入精度分数。

## 生成和评分分机执行

将 `SCORE=0` 生成的整个结果目录复制到评分机器。确保该机器也有相同版本的 GenEval 仓库和 Mask2Former 权重，然后执行：

```bash
RESULTS=/workspace/sensenova-thinking-geneval-results/<本次目录名>
GENEVAL_DIR=/workspace/geneval
GENEVAL_MODEL_DIR=/workspace/geneval-models
GENEVAL_PYTHON=/path/to/geneval-env/bin/python

for mode in off on; do
  "$GENEVAL_PYTHON" "$GENEVAL_DIR/evaluation/evaluate_images.py" "$RESULTS/benchmark/$mode" \
    --outfile "$RESULTS/benchmark/$mode-results.jsonl" \
    --model-path "$GENEVAL_MODEL_DIR"
  "$GENEVAL_PYTHON" "$GENEVAL_DIR/evaluation/summary_scores.py" \
    "$RESULTS/benchmark/$mode-results.jsonl" > "$RESULTS/benchmark/$mode-summary.txt"
done
python python/sglang/multimodal_gen/test/scripts/eval_sensenova_thinking_geneval.py compare \
  --output-dir "$RESULTS/benchmark" \
  --off-results "$RESULTS/benchmark/off-results.jsonl" \
  --on-results "$RESULTS/benchmark/on-results.jsonl" \
  > "$RESULTS/comparison.log"
```

`comparison.json` 包含三类各自的正确率、宏平均、成对赢/输数量、差值 bootstrap 区间、平均请求耗时和思考 token 用量。`thinking_budget_hit_rate_proxy` 是思考 token 恰好达到上限的比例；若很高，先提高 `MAX_THINK_TOKENS` 再重复整组实验，否则不能把分数差归因于完整思考能力。脚本不会自动判断“显著提升”；应结合差值区间、每类得分、样本量与人工复核失败样本下结论。
