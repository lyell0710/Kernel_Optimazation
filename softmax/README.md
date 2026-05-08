# cuda-softmax

按 `cuda-reduce` 的版本演进风格搭建的 softmax 练习工程。

## 目标
- 先有一个稳定可对照的 `baseline`
- 再按 `v0 -> v4` 小步迭代优化
- 每版只改一个核心点，便于定位收益和问题

## 版本说明
- `baseline`: 串行行内 softmax（单线程），仅用于正确性对照
- `v0`: 第一版并行化骨架（block 内两次归约：max + sum）
- `v1`: 小优化（分支/地址计算等低风险改动）
- `v2`: 规整 shared-memory 归约访问模式（降低冲突风险）
- `v3`: 每线程多元素 + 向量化 load/store
- `v4`: 归约尾部 warp 化，减少不必要同步

## 构建运行
```bash
cmake -S . -B build
cmake --build build -j
./build/softmax_bench
```

## Benchmark 输出
- 程序会自动输出每个版本的：
  - 正确性（PASS/FAIL）
  - 平均时延（默认 100 次迭代取均值）
  - 最大误差（对比 CPU softmax）
- 同时覆盖刷新 `project-proof/data/benchmark_results.csv`

## 生成图表
```bash
python project-proof/scripts/plot_latency.py
python project-proof/scripts/plot_latency_log.py
python project-proof/scripts/plot_speedup.py
python project-proof/scripts/plot_correctness.py
```

生成文件位于 `project-proof/docs/figures/01-benchmark/`：
- `01-latency.png`
- `02-latency-log.png`
- `03-speedup-vs-baseline.png`
- `04-correctness.png`

## NCU（按版本生成 `.ncu-rep`）
```bash
bash project-proof/scripts/profile_ncu.sh
```

会在 `project-proof/profiling/ncu/` 下生成各版本独立报告：`softmax_<tag>_profile.ncu-rep`。模板 kernel（v0–v4）使用 `regex:` 匹配。采集时 **`SOFTMAX_PROFILE_ONLY`** 由 `profile_ncu.sh` 自动传入。

在 **Mac 终端**拉取到本机示例（先建好目录；把 `ubuntu22`、`Tailscale IP` 换成你的）：

```bash
mkdir -p "/Users/yuzhang_li/Desktop/CUDA/NCU-report/Softmax_NCU"
scp ubuntu22@100.69.98.113:'~/CudaLearing/Kernel_Optimazation/softmax/project-proof/profiling/ncu/softmax_*_profile.ncu-rep' \
  "/Users/yuzhang_li/Desktop/CUDA/NCU-report/Softmax_NCU/"
```

- 默认 `BENCH_ITERS=1`（采集较快）；需要可调：`BENCH_ITERS=10 bash project-proof/scripts/profile_ncu.sh`
- **`plot_ncu_summary.py` 用的扩展 CSV**：`RUN_NCU_CSV=1 bash project-proof/scripts/profile_ncu.sh`，再执行 `python project-proof/scripts/plot_ncu_summary.py`；根目录 `bash scripts/run_ncu_all.sh` 默认会开启 `RUN_NCU_CSV=1`

图表目录：`project-proof/docs/figures/02-profiling/`

> 若提示 `ncu: command not found`，先安装 Nsight Compute（例如：`sudo apt install nsight-compute`）。

## 相关文档
- 优化说明：`project-proof/docs/why-each-version-changed.md`
- 项目介绍：`project-intro.md`
