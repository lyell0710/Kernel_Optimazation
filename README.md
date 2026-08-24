# Kernel_Optimazation

Kernel optimization playground for handwritten CUDA kernels and benchmark-driven iteration.

## Repository Layout
- `PORTFOLIO.md`: 六项目统一入口(方法论 + 跨项目 pattern,现行数字带 EXP 指针)。
- `cuda-reduce/`: reduction optimization project (`baseline` -> `v7`) with proof artifacts.
- `gemv/`: GEMV kernel optimization project (`baseline` -> `v4`).
- `softmax/`: Softmax kernel optimization project (`baseline` -> `v4`).
- `int8-quantize/`: INT8 per-channel quantize optimization project (`baseline` -> `v4`).
- `gemm/`: CUDA Tensor Core GEMM ladder (`v0` naive -> `v4` wmma+cp.async big-tile, vs real cuBLAS).
- `flash-attn/`: CUDA FA2 forward ladder (`v0` warp-per-row -> `v4` wmma+overlap, vs own Triton FA2).

## EXP 索引

| 编号 | slug | 日期 | 状态 | 关键数字(指针) |
|---|---|---|---|---|
| [EXP-K01](records/EXP-K01_4090_rebench.md) | 4090_rebench | 2026-08-23 | 完成(带 8/24 勘误) | 4090 reduce v7 反超 cuBLAS 24.5%(3轮);softmax 对比句作废(对照系自写 kernel,勘误见记录 §5);gemv 84%(对照物限定,**单轮**;3 轮补测=记录 §7 backlog,待 GPU 空闲)→ 各 project-proof/data/ |
| [EXP-K02](records/EXP-K02_cuda_gemm_tc_ladder.md) | cuda_gemm_tc_ladder | 2026-08-24 | 完成 | Tensor Core GEMM v0→v4:133.1±0.97 TFLOPS = 真 cuBLAS 85.6%(4096³,3轮)→ gemm/project-proof/data/ |
| [EXP-K03](records/EXP-K03_cuda_fa2_ladder.md) | cuda_fa2_ladder | 2026-08-24 | 完成 | CUDA FA2 v0→v4:34.8±0.12 TFLOPS = 自家 Triton 28%(跨harness),wmma 架构税量化 → flash-attn/project-proof/data/ |

## Suggested Workflow(对齐 /root/standards CORE 七条铁律)
1. bench 只写 UTC 前缀新文件到各 `project-proof/data/`(`BENCH_OUT` 控制,首行 provenance),永不覆盖已有文件。
2. 数字进 README/简历前 ≥3 轮 mean±std,落 stability/derived 文件。
3. 每个实验一份 `records/` 八节记录(EXP-KNN),并同步上方 EXP 索引表。
4. 对外措辞先过各子项目 README 的措辞红线表(gemm/flash-attn 已建),凡 "vs X" 声明先验 X 的调用点。
5. 收尾跑 `bash /root/standards/check.sh` 六项自检。

## One-Click Scripts

Run all benchmarks and generate all plots:

```bash
bash scripts/run_bench_and_plot_all.sh
```

Run NCU profiling for all projects:

```bash
bash scripts/run_ncu_all.sh
```

打包所有已生成的 `*_profile.ncu-rep` 到 `artifacts/ncu_for_mac/*.tar.gz`（便于一次 `scp` 到 Mac）：

```bash
bash scripts/pack_ncu_reps_for_mac.sh
```

默认会导出各子项目扩展 metrics CSV（`RUN_NCU_CSV=1`）并运行 `plot_ncu_summary.py`。若只需完整 Section 的 `.ncu-rep`、跳过第二次 CSV 采集：

```bash
RUN_NCU_CSV=0 bash scripts/run_ncu_all.sh
```

