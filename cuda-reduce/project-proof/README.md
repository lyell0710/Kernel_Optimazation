# CUDA Reduction Project Proof Pack

## Overview
This proof pack records the real benchmark results of a handwritten CUDA reduction optimization experiment.

## Included Materials
- `docs/experiment-summary.md`: detailed experiment summary for documentation and interview use
- `data/benchmark_results.csv`: structured benchmark record
- `data/experiment_env.csv`: experiment environment and configuration metadata
- `scripts/plot_latency.py`: plotting script for latency comparison
- `scripts/plot_correctness.py`: plotting script for correctness comparison
- `src_refs/key-results.txt`: raw key result snapshot

## Experiment Setup
- GPU: `NVIDIA GeForce RTX 4070 Laptop GPU`
- CUDA Runtime: `13.2` (driver reported)
- CUDA Toolkit / nvcc: `11.5`
- OS: `Ubuntu 24.04.3 LTS (WSL2)`
- Input size: `1 << 24`
- baseline: single-thread GPU reduction (`<<<1,1>>>`)
- v0: block-level shared-memory tree reduction (`block size = 256`)
- v6: grid-stride + two-pass reduction (`block size = 256`)

## Core Results
- CPU: `1.67772e+07`
- baseline GPU: `1.67772e+07`
- baseline Diff: `0`
- baseline latency: `349.310730 ms`
- v5 latency: `0.373608 ms`
- v6 latency: `0.312438 ms`
- correctness: all versions (`baseline`~`v6`) pass with `Diff = 0`

## Speedup
- v6 vs baseline: `1118.02x`
- v6 vs v5: `+16.37%`

## Limitation
- `Nsight Compute` kernel profiling is not supported on this WSL environment; all `ncu` records in `profiling/ncu/` are troubleshooting traces rather than valid profiling metrics.
