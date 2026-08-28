# NCU 报告导出包(供 Mac 端 Nsight Compute GUI 打开)

共 50 份 `.ncu-rep`，按算子分类于 `reports/<算子>/`。逐份摘要见 `manifest.csv`。本文件由 `scripts/export_ncu_for_mac.py` 生成。

## 采集批次

逐份出处见 `manifest.csv` 的 gpu / sm_count / host / ncu_ver / created 列。

| GPU | SM 数 | 主机 | Nsight Compute | 采集日期 | 报告数 |
|---|---:|---|---|---|---:|
| NVIDIA GeForce RTX 4090 | 128 | 10-60-214-39 | 2025.1.1.0 (build 35528883) (public-release) | 2026-Aug-29 | 44 |
| NVIDIA GeForce RTX 4070 Laptop GPU | 36 | ubuntu22 | 2026.1.1.0 (build 37634170) (public-release) | 2026-May-03 | 5 |
| NVIDIA GeForce RTX 4070 Laptop GPU | 36 | ubuntu22 | 2022.4.1.0 (build 32308335) (public-release) | 2026-May-03 | 1 |

> **GPU 一栏以报告自述为准，不要从主机型号推断。** 本包 2026-05 那批的主机是 Ryzen 9 7945HX 笔记本，很容易被顺手写成「笔记本 4090」，实际是 RTX 4070 Laptop GPU（36 SM / 8 GB），与桌面 RTX 4090（128 SM / 24 GB）差 3.6 倍 SM。占用率、波量化、L2 容量相关的结论在两者之间不可迁移。

> 上表每一行是一个采集批次。**跨批次的数字不可混排为同一行**——不同 GPU 不必说，同一台机器上不同 Nsight Compute 版本的 metric 定义也可能有出入。

## Mac 端打开方式

1. 装 **Nsight Compute 2025.3 或更新**（原生支持 macOS arm64，最低 macOS 13.0）。新版 GUI 可读旧版报告，反之不成立。

2. `File > Open` 打开 `.ncu-rep`，或把 `reports/` 整个拖进去。

3. 版本梯对比用 **Add Baseline**：先开 v0 设为 baseline，再开 v1..vN，Details 页每个 metric 会显示相对增减——这是看"这一版到底改善了什么"最快的读法。

## 覆盖情况

| 算子 | 报告数 | 版本 |
|---|---:|---|
| `cuda-reduce` | 14 | baseline, bench, v0, v1, v2, v3, v4, v5, v6, v7 |
| `flash-attn` | 6 | ref_naive, v0_warp_row, v1_smem_tile, v2_wmma, v3_8warp, v4_overlap |
| `gemm` | 6 | cublas, v0, v1, v2_wmma, v3_dbuf, v4_bigtile |
| `gemv` | 7 | baseline, cublas, v0, v1, v2, v3, v4 |
| `int8-quantize` | 6 | baseline, v0, v1, v2, v3, v4 |
| `softmax` | 11 | baseline, cublas, smoke, v0, v1, v2, v3, v4, v4.2, v4.3, v4.4 |

**未覆盖（零 NCU 数据）**：`activation`, `fused-norm`, `rope`, `w8a8`。

## 含多个 grid 的报告（读图时先认清自己在看哪一次 launch）

下列报告里同一个 kernel 被 launch 了多次且 grid 不同。两种成因：

- **多级算法的各级**——如归约树 `65536 → 256 → 1`，三级都是同一个 kernel。此时只有第一级（grid 最大那次）反映真实的 HBM 行为，后面几级数据量已极小，它们的 SOL 低是必然的，不是优化空间。
- **多 regime 混样**——bench 扫了 decode / l2 / prefill / hbm 几个尺寸，都落进了同一份报告。这种情况下**不能跨实例比较**：L2 区间与 HBM 区间的结论会翻转（等效带宽超过硬件峰值就是落在 L2 的信号，4090 的 L2 是 72 MB）。

在 GUI 里按 launch 实例逐个看，先看 grid 认出自己在读哪一次。

| 报告 | 实例数 | grid 种类 |
|---|---:|---:|
| `reports/cuda-reduce/reduce_v0_profile.ncu-rep` | 6 | 3 |
| `reports/cuda-reduce/reduce_v1_profile.ncu-rep` | 6 | 3 |
| `reports/cuda-reduce/reduce_v2_profile.ncu-rep` | 6 | 3 |
| `reports/cuda-reduce/reduce_v3_profile.ncu-rep` | 6 | 3 |
| `reports/cuda-reduce/reduce_v4_profile.ncu-rep` | 6 | 3 |
| `reports/cuda-reduce/reduce_v5_profile.ncu-rep` | 6 | 3 |
| `reports/cuda-reduce/reduce_v6_profile.ncu-rep` | 4 | 2 |
| `reports/cuda-reduce/reduce_v7_profile.ncu-rep` | 4 | 2 |
| `reports/flash-attn/fa2_ref_naive_profile.ncu-rep` | 8 | 6 |
| `reports/flash-attn/fa2_v0_warp_row_profile.ncu-rep` | 92 | 6 |
| `reports/flash-attn/fa2_v1_smem_tile_profile.ncu-rep` | 92 | 6 |
| `reports/flash-attn/fa2_v2_wmma_profile.ncu-rep` | 92 | 6 |
| `reports/flash-attn/fa2_v3_8warp_profile.ncu-rep` | 92 | 6 |
| `reports/flash-attn/fa2_v4_overlap_profile.ncu-rep` | 92 | 6 |

## 对照臂口径陷阱

- `gemv_cublas_profile` 内是 `gemv2T_kernel_val<...cublasGemvParams...>`，**是真 cuBLAS**，可作标准库对照。
- `softmax_cublas_profile` 内是 `softmax_cublas_kernel`，**是自写 kernel，不是 cuBLAS**（BLAS 规范里没有 softmax）。文件名有误导性，该臂不得用于"对比 cuBLAS"的表述——此为已存档红线项。
- `cuda-reduce` 的标准库对照应为 CUB `DeviceReduce`，本包内无 CUB 臂。
- `*_baseline` 为朴素实现，SOL 与 Occupancy 极低属预期，不代表硬件上限。

## 逐份摘要

取报告内第一个 kernel 实例；`k` 为实例总数，`g` 为不同 grid 数。

| 算子 | 版本 | k | g | 代表 grid | Duration | SM % | Memory % | DRAM % | Occ % | Regs |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `cuda-reduce` | baseline | 2 | 1 | `(1, 1, 1)` | 343.55 ms | 0.03 | 0.03 | 0.02 | 2.08 | 37 |
| `cuda-reduce` | bench | 30 | 1 | `(1, 1, 1)` | 424.15 ms | 0.14 | 0.14 | 0.07 | 2.08 | 37 |
| `cuda-reduce` | v0 | 6 | 3 | `(65536, 1, 1)` | 115.33 us | 87.86 | 87.86 | 59.46 | 86.36 | 16 |
| `cuda-reduce` | v1 | 6 | 3 | `(65536, 1, 1)` | 115.17 us | 87.84 | 87.84 | 59.55 | 86.37 | 16 |
| `cuda-reduce` | v2 | 6 | 3 | `(65536, 1, 1)` | 113.41 us | 89.29 | 89.29 | 60.54 | 86.13 | 16 |
| `cuda-reduce` | v3 | 6 | 3 | `(32768, 1, 1)` | 73.54 us | 77.53 | 93.27 | 93.27 | 89.60 | 16 |
| `cuda-reduce` | v4 | 6 | 3 | `(32768, 1, 1)` | 73.25 us | 35.57 | 93.85 | 93.85 | 81.88 | 16 |
| `cuda-reduce` | v5 | 1 | 1 | `(32768, 1, 1)` | 277.92 us | 48.24 | 96.67 | 96.67 | 82.14 | 16 |
| `cuda-reduce` | v5 | 1 | 1 | `(32768, 1, 1)` | 276.74 us | 48.47 | 96.66 | 96.66 | 81.93 | 16 |
| `cuda-reduce` | v5 | 6 | 3 | `(32768, 1, 1)` | 72.90 us | 37.04 | 94.12 | 94.12 | 84.10 | 16 |
| `cuda-reduce` | v6 | 1 | 1 | `(288, 1, 1)` | 279.17 us | 6.93 | 96.09 | 96.09 | 87.02 | 17 |
| `cuda-reduce` | v6 | 4 | 2 | `(1024, 1, 1)` | 73.70 us | 5.88 | 92.95 | 92.95 | 87.23 | 17 |
| `cuda-reduce` | v7 | 1 | 1 | `(288, 1, 1)` | 319.26 us | 6.10 | 95.79 | 95.79 | 84.05 | 17 |
| `cuda-reduce` | v7 | 4 | 2 | `(1024, 1, 1)` | 73.98 us | 5.88 | 92.71 | 92.71 | 87.86 | 17 |
| `flash-attn` | ref_naive | 8 | 6 | `(4096, 32, 1)` | 42.91 ms | 96.27 | 96.27 | 0.14 | 49.19 | 40 |
| `flash-attn` | v0_warp_row | 92 | 6 | `(4096, 32, 1)` | 26.29 ms | 94.01 | 94.01 | 0.23 | 49.22 | 40 |
| `flash-attn` | v1_smem_tile | 92 | 6 | `(1024, 32, 1)` | 30.32 ms | 46.38 | 46.38 | 0.20 | 24.84 | 40 |
| `flash-attn` | v2_wmma | 92 | 6 | `(64, 32, 1)` | 7.09 ms | 12.80 | 44.57 | 0.83 | 8.33 | 64 |
| `flash-attn` | v3_8warp | 92 | 6 | `(64, 32, 1)` | 5.33 ms | 18.22 | 61.47 | 1.10 | 16.67 | 95 |
| `flash-attn` | v4_overlap | 92 | 6 | `(64, 32, 1)` | 4.99 ms | 20.81 | 62.96 | 1.18 | 16.67 | 80 |
| `gemm` | cublas | 6 | 1 | `(32, 32, 3)` | 977.31 us | 48.77 | 53.57 | 9.93 | 16.36 | 234 |
| `gemm` | v0 | 7 | 1 | `(256, 256, 1)` | 30.46 ms | 98.76 | 98.76 | 0.28 | 99.38 | 26 |
| `gemm` | v1 | 7 | 1 | `(128, 128, 1)` | 26.14 ms | 79.35 | 79.35 | 0.33 | 66.66 | 33 |
| `gemm` | v2_wmma | 5 | 1 | `(64, 64, 1)` | 1.84 ms | 25.65 | 95.42 | 4.59 | 70.20 | 54 |
| `gemm` | v3_dbuf | 5 | 1 | `(64, 64, 1)` | 1.77 ms | 26.83 | 87.73 | 4.83 | 40.25 | 61 |
| `gemm` | v4_bigtile | 5 | 1 | `(32, 32, 1)` | 1.26 ms | 37.77 | 80.60 | 6.71 | 32.38 | 92 |
| `gemv` | baseline | 2 | 1 | `(4096, 1, 1)` | 145.25 us | 82.47 | 82.47 | 23.52 | 38.09 | 37 |
| `gemv` | cublas | 2 | 1 | `(512, 1, 1)` | 38.30 us | 12.95 | 89.26 | 89.26 | 32.62 | 59 |
| `gemv` | v0 | 2 | 1 | `(4096, 1, 1)` | 38.34 us | 25.91 | 89.20 | 89.20 | 93.86 | 38 |
| `gemv` | v1 | 2 | 1 | `(4096, 1, 1)` | 38.37 us | 25.86 | 89.12 | 89.12 | 93.99 | 27 |
| `gemv` | v2 | 2 | 1 | `(4096, 1, 1)` | 37.79 us | 18.66 | 90.49 | 90.49 | 62.69 | 50 |
| `gemv` | v3 | 2 | 1 | `(1024, 1, 1)` | 37.79 us | 10.42 | 90.48 | 90.48 | 62.01 | 38 |
| `gemv` | v4 | 2 | 1 | `(1024, 1, 1)` | 38.46 us | 25.72 | 88.85 | 88.85 | 91.97 | 24 |
| `int8-quantize` | baseline | 2 | 1 | `(1, 1, 1)` | 149.30 ms | 0.03 | 0.02 | 0.00 | 2.08 | 21 |
| `int8-quantize` | v0 | 2 | 1 | `(4096, 1, 1)` | 7.71 us | 37.93 | 55.70 | 55.70 | 75.75 | 22 |
| `int8-quantize` | v1 | 2 | 1 | `(2048, 1, 1)` | 7.30 us | 31.71 | 58.82 | 58.82 | 79.66 | 21 |
| `int8-quantize` | v2 | 2 | 1 | `(1024, 1, 1)` | 7.30 us | 23.75 | 58.88 | 58.88 | 74.61 | 27 |
| `int8-quantize` | v3 | 2 | 1 | `(1024, 1, 1)` | 7.10 us | 13.64 | 60.65 | 60.65 | 77.15 | 20 |
| `int8-quantize` | v4 | 2 | 1 | `(1024, 1, 1)` | 6.82 us | 13.87 | 63.41 | 63.41 | 67.50 | 32 |
| `softmax` | baseline | 2 | 1 | `(1024, 1, 1)` | 109.47 us | 31.57 | 29.39 | 3.91 | 16.51 | 39 |
| `softmax` | cublas | 2 | 1 | `(1024, 1, 1)` | 10.34 us | 26.99 | 42.25 | 41.58 | 76.11 | 26 |
| `softmax` | smoke | 8 | 1 | `(1024, 1, 1)` | 405.38 us | 51.13 | 42.82 | 4.50 | 36.07 | 39 |
| `softmax` | v0 | 2 | 1 | `(1024, 1, 1)` | 8.86 us | 45.49 | 48.54 | 48.54 | 83.39 | 40 |
| `softmax` | v1 | 2 | 1 | `(1024, 1, 1)` | 8.83 us | 46.05 | 48.86 | 48.86 | 83.06 | 40 |
| `softmax` | v2 | 2 | 1 | `(1024, 1, 1)` | 8.96 us | 46.79 | 48.03 | 48.03 | 82.77 | 40 |
| `softmax` | v3 | 2 | 1 | `(1024, 1, 1)` | 8.45 us | 39.45 | 51.00 | 51.00 | 82.91 | 36 |
| `softmax` | v4 | 2 | 1 | `(1024, 1, 1)` | 7.62 us | 24.84 | 56.49 | 56.49 | 82.00 | 36 |
| `softmax` | v4.2 | 2 | 1 | `(1024, 1, 1)` | 10.59 us | 38.52 | 43.10 | 40.63 | 80.71 | 23 |
| `softmax` | v4.3 | 2 | 1 | `(1024, 1, 1)` | 7.68 us | 26.43 | 56.20 | 56.20 | 78.65 | 40 |
| `softmax` | v4.4 | 2 | 1 | `(1024, 1, 1)` | 8.32 us | 40.52 | 51.84 | 51.84 | 80.06 | 40 |
