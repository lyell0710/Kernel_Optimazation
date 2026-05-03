# CUDA Reduction 实验总结（baseline ~ v6）

## 1) 实验目标
验证 CUDA reduction 从 `baseline` 到 `v6` 的优化收益，比较各版本在同一输入规模下的正确性与延迟表现。

## 2) 核心结论
- 所有版本（`baseline`、`v0`、`v1`、`v2`、`v3`、`v4`、`v5`、`v6`）均与 CPU 结果一致，`Diff = 0`。
- 当前最优版本为 `v6`：`0.312438 ms`，相对 `baseline`（`349.310730 ms`）达到 **`1118.02x`** 加速。
- `v6` 相比 `v5`（`0.373608 ms`）进一步提升约 **`16.37%`**。

## 3) 结果总表
| version | cpu_result | gpu_result | diff | latency_ms | speedup_vs_baseline | correctness_pass |
|---|---:|---:|---:|---:|---:|---|
| baseline | 1.67772e+07 | 1.67772e+07 | 0 | 349.310730 | 1.00x | true |
| v0 | 1.67772e+07 | 1.67772e+07 | 0 | 0.511450 | 682.98x | true |
| v1 | 1.67772e+07 | 1.67772e+07 | 0 | 0.614215 | 568.71x | true |
| v2 | 1.67772e+07 | 1.67772e+07 | 0 | 0.587037 | 595.04x | true |
| v3 | 1.67772e+07 | 1.67772e+07 | 0 | 0.379432 | 920.61x | true |
| v4 | 1.67772e+07 | 1.67772e+07 | 0 | 0.372975 | 936.55x | true |
| v5 | 1.67772e+07 | 1.67772e+07 | 0 | 0.373608 | 934.97x | true |
| v6 | 1.67772e+07 | 1.67772e+07 | 0 | 0.312438 | 1118.02x | true |

## 4) 环境与配置
- GPU: `NVIDIA GeForce RTX 4070 Laptop GPU`（`8188 MiB`）
- Driver: `596.21`
- CUDA Runtime: `13.2`
- CUDA Toolkit / nvcc: `11.5.119`
- OS: `Ubuntu 24.04.3 LTS (WSL2, Linux 6.6.87.2-microsoft-standard-WSL2)`
- CPU: `AMD Ryzen 9 7945HX with Radeon Graphics`
- CMake: `4.3.0`
- Compiler: `g++ 11.4.0`
- 输入规模: `N = 1 << 24`
- baseline 启动配置: `<<<1,1>>>`
- 计时方式: `CUDA events`，含 warmup

## 5) 图表文件
- `project-proof/docs/figures/01-benchmark/01-latency-overview.png`
- `project-proof/docs/figures/01-benchmark/02-latency-log.png`
- `project-proof/docs/figures/01-benchmark/03-latency-zoom.png`
- `project-proof/docs/figures/01-benchmark/06-correctness.png`

## 6) 简要分析
- `baseline` 使用近似串行的 GPU 归约方式，延迟远高于优化版本。
- `v0`~`v2` 带来稳定提升，主要来自 block 内共享内存归约与访问模式改进。
- `v3`~`v5` 持续压低延迟，`v6` 在 `v5` 基础上采用 grid-stride + two-pass，进一步获得稳定收益。

## 7) 限制说明
- 当前环境为 WSL2。`Nsight Compute` 在该环境对 GPU kernel profiling 不支持（运行时会提示 `No kernels were profiled`），因此 occupancy / throughput / warp 指标未纳入本次定量结论。
- 本文档结论基于可复现 benchmark（warmup + 100 次均值）与 correctness 对照；后续可在原生 Linux/Windows CUDA 环境补齐 profiler 指标。
