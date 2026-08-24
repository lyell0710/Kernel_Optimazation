# Kernel_Optimazation

Kernel optimization playground for handwritten CUDA kernels and benchmark-driven iteration.

## Repository Layout
- `cuda-reduce/`: reduction optimization project (`baseline` -> `v6`) with proof artifacts.
- `gemv/`: GEMV kernel optimization project (`baseline` -> `v4`).
- `softmax/`: Softmax kernel optimization project (`baseline` -> `v4`).
- `int8-quantize/`: INT8 per-channel quantize optimization project (`baseline` -> `v4`).
- `layernorm/`: reserved for LayerNorm kernel optimization experiments.
- `notes/`: experiment notes, interview scripts, and retrospective writeups.

## EXP 索引

| 编号 | slug | 日期 | 状态 | 关键数字(指针) |
|---|---|---|---|---|
| [EXP-K01](records/EXP-K01_4090_rebench.md) | 4090_rebench | 2026-08-23 | 完成(带 8/24 勘误) | 4090 reduce v7 反超 cuBLAS 24.5%(3轮);softmax 对比句作废(对照系自写 kernel,勘误见记录 §5);gemv 84%(对照物限定)→ 各 project-proof/data/ |

## Suggested Workflow
1. Build a baseline kernel and iterate versions (`v0`, `v1`, ...).
2. Keep benchmark settings fixed (input size, warmup, iteration count).
3. Record results in CSV/figures under each project's `project-proof/`.
4. Summarize conclusions and limitations in `notes/`.

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

