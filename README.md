# Kernel_Optimazation

Kernel optimization playground for handwritten CUDA kernels and benchmark-driven iteration.

## Repository Layout
- `cuda-reduce/`: reduction optimization project (`baseline` -> `v6`) with proof artifacts.
- `gemv/`: GEMV kernel optimization project (`baseline` -> `v4`).
- `softmax/`: Softmax kernel optimization project (`baseline` -> `v4`).
- `int8-quantize/`: INT8 per-channel quantize optimization project (`baseline` -> `v4`).
- `layernorm/`: reserved for LayerNorm kernel optimization experiments.
- `notes/`: experiment notes, interview scripts, and retrospective writeups.

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

