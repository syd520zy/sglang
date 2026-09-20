# SenseNova 思考模式 P1 验证脚本

本文档覆盖 P1-1（并发吞吐与 continuous batching 验证）和 P1-3（内部 SRT 生命周期与可观测性）
两部分的脚本、执行命令和结果查看方式。P0 的单请求对照 profiling 见
`README_sensenova_thinking_profiling_zh.md`。

## 1. 更新代码

A800 机器上先更新到最新提交，并确认依赖版本：

```bash
cd /workspace/sglang
bash benchmark/sensenova/update_repo.sh
```

脚本会检查工作区是否干净、拉取 `feat/sensenova-thinking-mode`、打印当前 commit，并核对
`torch`/CUDA/GPU 以及源码要求的 `sglang-kernel==0.4.7`。它只做 `--ff-only` 更新，不会覆盖本地修改。

## 2. P1-1 并发吞吐 A/B（native 对照 SRT）

一次运行完成 native 基线、SRT 优化路径和比较，输出 `comparison.json`。`STEPS_LIST` 是要逐一的
去噪步数列表，每一项都会重启一对服务：

```bash
cd /workspace/sglang

STEPS_LIST=10 CONCURRENCY="1 2" BUDGETS=64,128,256 REPEATS=2 \
BATCHING_MAX_SIZE=2 BATCHING_DELAY_MS=50 \
  bash python/sglang/multimodal_gen/test/scripts/compare_sensenova_thinking_concurrency.sh
```

常用覆盖项：

```bash
GPU_ID=0 SERVER_PORT=30000 WIDTH=1024 HEIGHT=1024 STEPS_LIST="10 50" \
BUDGETS=64,128 CONCURRENCY="1 2 4" REPEATS=3 \
OUTPUT_DIR=/workspace/sensenova-thinking-concurrency \
  bash python/sglang/multimodal_gen/test/scripts/compare_sensenova_thinking_concurrency.sh
```

单侧运行（只测一种 backend）：

```bash
SGLANG_SENSENOVA_THINKING_BACKEND=srt \
OUTPUT_DIR=/workspace/sensenova-thinking-concurrency-srt \
  bash python/sglang/multimodal_gen/test/scripts/profile_sensenova_thinking_concurrency.sh
```

## 3. P1-3 生命周期验收

一次运行覆盖 `fallback` 和 `strict` 两种模式：每个模式各自启动服务，验证 `ready` 状态由 SRT
服务思考请求，然后让内部 SRT 服务失去响应（默认 `SIGSTOP`，即挂起而不是退出，只有客户端读超时能
发现），再验证首个请求如何失败/回退、后续请求是否还会等待、`/server_info` 的 backend 状态、错误
日志条数，以及主服务退出后端口和 GPU 进程是否回收。

```bash
cd /workspace/sglang

MODES="fallback strict" SRT_FAILURE=stop STEPS=10 MAX_THINK_TOKENS=64 \
  bash python/sglang/multimodal_gen/test/scripts/validate_sensenova_thinking_lifecycle.sh
```

也可以只跑一种模式：

```bash
MODES=strict SRT_FAILURE=kill \
  bash python/sglang/multimodal_gen/test/scripts/validate_sensenova_thinking_lifecycle.sh
```

`SRT_FAILURE=kill` 用 `kill_process_tree` 结束内部 SRT，模拟服务已经退出；`stop` 用 `SIGSTOP`
模拟服务卡住，用于验证“后续请求不重复长时间等待不可用服务”。

脚本自己找出内部 SRT 的 pid：它读 `/proc/net/tcp` 的 LISTEN 记录拿到 socket inode，再扫描
`/proc/*/fd` 找到持有者，因此不依赖 `ss` 或 `lsof` 是否安装。手动确认端口归属可以单独调用：

```bash
cd /workspace/sglang

python python/sglang/multimodal_gen/test/scripts/validate_sensenova_thinking_lifecycle.py \
  --phase srt-pid --port <SRT 端口> --output-dir /tmp
```

它以退出码 0 表示找到了 pid（pid 打印在标准输出），1 表示没有进程监听该端口。每个模式结束后脚本
都会停掉本次启动的主服务，即使中途失败也不会把服务留在 GPU 上，所以单次运行可以直接跑到结束。

## 4. 结果查看（不需要压缩）

结果目录里都是小的 JSON 和文本，直接打印成可复制的摘要：

```bash
cd /workspace/sglang

python python/sglang/multimodal_gen/test/scripts/view_sensenova_thinking_results.py \
  /workspace/sensenova-thinking-concurrency/<时间戳> \
  /workspace/sensenova-thinking-lifecycle/<时间戳> \
  --output /workspace/sensenova-thinking-view.txt
```

把打印出来的内容（或 `sensenova-thinking-view.txt`）直接发回即可，不需要打包整个目录。确实需要
搬走完整日志时再用现成的打包脚本：

```bash
bash python/sglang/multimodal_gen/test/scripts/pack_sensenova_thinking_results.sh \
  /workspace/sensenova-thinking-lifecycle/<时间戳>
```

## 5. 结果目录内容

P1-1（`compare_sensenova_thinking_concurrency.sh`）：

- `native/`、`srt/`：两次运行的 `environment.txt`、`server.log`、`batch-metrics.log`、
  `profile/records.json`、`profile/summary.json`。
- `comparison.json`、`comparison.log`：两组对照和判定项。

P1-3（`validate_sensenova_thinking_lifecycle.sh`）：

- `<mode>/environment.txt`：commit、模式、strict、失败注入方式、基线 GPU 进程。
- `<mode>/server.log`：主服务日志；内部 SRT 的日志在单独文件里，路径见
  `/server_info` 的 `thinking_backend.log_file`。
- `<mode>/server-info.json`：启动后的 `/server_info`，含 `thinking_backend` 字段。
- `<mode>/lifecycle-startup.json`、`<mode>/lifecycle-after-kill.json`：每个检查项及其证据。
- `<mode>/residue.txt`：退出后 SRT 端口是否仍在监听、GPU 计算进程是否清空。

## 6. backend 状态与 strict 模式

`/server_info` 的 `thinking_backend` 字段：

| 字段 | 含义 |
| --- | --- |
| `state` | `disabled` / `starting` / `ready` / `failed` / `fallback` / `stopped` |
| `backend` | 当前实际服务思考请求的后端：`srt` 或 `native` |
| `url` | 内部 SRT 地址；未启用时为 `null` |
| `strict` | 是否启用 strict 模式，按当前进程的环境变量实时读取 |
| `reason` | 最近一次失败原因 |
| `log_file` | 内部 SRT 的子服务日志路径 |
| `pid` | 最后一次写入状态文件的进程 pid，用于判断这条记录是否已过期 |

`state`、`reason` 和 `pid` 来自跨进程共享的状态文件，`strict` 和 `url` 始终取当前进程的配置：
同一个 SRT 端口被不同运行复用时，旧文件不会让新服务报告错误的 strict 或地址。

相关环境变量：

- `SGLANG_SENSENOVA_THINKING_BACKEND`：`srt`（默认）或 `native`。
- `SGLANG_SENSENOVA_THINKING_STRICT`：`1` 时要求 SRT，启动失败或运行中失败都会让服务退出/请求
  报错，不会静默回退；benchmark 和 CI 用这个开关，生产默认关闭（允许回退）。
- `SGLANG_SENSENOVA_THINKING_RUNTIME_DIR`：状态文件和 SRT 日志目录，默认在系统临时目录下的
  `sglang-sensenova-thinking/`。
- `SGLANG_SENSENOVA_THINKING_LOG_FILE`：覆盖内部 SRT 日志文件路径。

## 7. 已知前提

- `concurrency>1` 的思考请求目前仍会被调度器串行发放：`SenseNovaU1PipelineConfig`
  的 `supports_dynamic_batching_for_request` 对 `think_mode` 返回 `False`，且
  `--batching-max-size` 默认为 1。因此这一轮 A/B 主要用于记录当前串行行为
  （`parallelism_ratio` 接近 1.0），并在放开思考请求 batching 后作为对照复测。
- 主服务只有一个 scheduler worker 时，`failure_is_reported_once` 才严格等于 1 条错误日志；
  多 worker 场景下每个 worker 各记录一次。
