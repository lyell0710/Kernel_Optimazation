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

