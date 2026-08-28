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
| `cuda-reduce` | baseline | 2 | 1 | `(1, 1, 1)` | 343.54 ms | 0.03 | 0.03 | 0.02 | 2.08 | 37 |
| `cuda-reduce` | bench | 30 | 1 | `(1, 1, 1)` | 424.15 ms | 0.14 | 0.14 | 0.07 | 2.08 | 37 |
| `cuda-reduce` | v0 | 6 | 3 | `(65536, 1, 1)` | 114.94 us | 87.87 | 87.87 | 59.82 | 86.37 | 16 |
| `cuda-reduce` | v1 | 6 | 3 | `(65536, 1, 1)` | 114.94 us | 87.86 | 87.86 | 59.73 | 86.36 | 16 |
| `cuda-reduce` | v2 | 6 | 3 | `(65536, 1, 1)` | 112.74 us | 89.30 | 89.30 | 60.90 | 86.09 | 16 |
| `cuda-reduce` | v3 | 6 | 3 | `(32768, 1, 1)` | 73.31 us | 77.03 | 93.67 | 93.67 | 89.76 | 16 |
| `cuda-reduce` | v4 | 6 | 3 | `(32768, 1, 1)` | 73.09 us | 35.38 | 94.00 | 94.00 | 82.11 | 16 |
| `cuda-reduce` | v5 | 1 | 1 | `(32768, 1, 1)` | 277.92 us | 48.24 | 96.67 | 96.67 | 82.14 | 16 |
| `cuda-reduce` | v5 | 1 | 1 | `(32768, 1, 1)` | 276.74 us | 48.47 | 96.66 | 96.66 | 81.93 | 16 |
| `cuda-reduce` | v5 | 6 | 3 | `(32768, 1, 1)` | 73.06 us | 37.22 | 94.01 | 94.01 | 84.30 | 16 |
| `cuda-reduce` | v6 | 1 | 1 | `(288, 1, 1)` | 279.17 us | 6.93 | 96.09 | 96.09 | 87.02 | 17 |
| `cuda-reduce` | v6 | 4 | 2 | `(1024, 1, 1)` | 73.86 us | 5.88 | 92.86 | 92.86 | 86.76 | 17 |
| `cuda-reduce` | v7 | 1 | 1 | `(288, 1, 1)` | 319.26 us | 6.10 | 95.79 | 95.79 | 84.05 | 17 |
| `cuda-reduce` | v7 | 4 | 2 | `(1024, 1, 1)` | 73.73 us | 5.91 | 92.90 | 92.90 | 87.27 | 17 |
| `flash-attn` | ref_naive | 8 | 6 | `(4096, 32, 1)` | 43.10 ms | 96.21 | 96.21 | 0.14 | 49.19 | 40 |
| `flash-attn` | v0_warp_row | 92 | 6 | `(4096, 32, 1)` | 26.28 ms | 93.95 | 93.95 | 0.23 | 49.22 | 40 |
| `flash-attn` | v1_smem_tile | 92 | 6 | `(1024, 32, 1)` | 30.31 ms | 46.39 | 46.39 | 0.20 | 24.84 | 40 |
| `flash-attn` | v2_wmma | 92 | 6 | `(64, 32, 1)` | 7.09 ms | 12.80 | 44.58 | 0.83 | 8.33 | 64 |
| `flash-attn` | v3_8warp | 92 | 6 | `(64, 32, 1)` | 5.33 ms | 18.22 | 61.45 | 1.11 | 16.67 | 95 |
| `flash-attn` | v4_overlap | 92 | 6 | `(64, 32, 1)` | 5.01 ms | 20.81 | 62.97 | 1.17 | 16.67 | 80 |
| `gemm` | cublas | 6 | 1 | `(32, 32, 3)` | 971.01 us | 48.74 | 53.77 | 10.00 | 16.36 | 234 |
| `gemm` | v0 | 7 | 1 | `(256, 256, 1)` | 30.41 ms | 98.81 | 98.81 | 0.28 | 99.50 | 26 |
| `gemm` | v1 | 7 | 1 | `(128, 128, 1)` | 26.08 ms | 79.39 | 79.39 | 0.33 | 66.66 | 33 |
| `gemm` | v2_wmma | 5 | 1 | `(64, 64, 1)` | 1.85 ms | 25.64 | 95.39 | 4.55 | 70.23 | 54 |
| `gemm` | v3_dbuf | 5 | 1 | `(64, 64, 1)` | 1.76 ms | 26.89 | 87.92 | 4.86 | 40.25 | 61 |
| `gemm` | v4_bigtile | 5 | 1 | `(32, 32, 1)` | 1.25 ms | 37.71 | 80.48 | 6.76 | 32.36 | 92 |
| `gemv` | baseline | 2 | 1 | `(4096, 1, 1)` | 143.49 us | 82.08 | 82.08 | 23.80 | 38.17 | 37 |
| `gemv` | cublas | 2 | 1 | `(512, 1, 1)` | 38.34 us | 12.85 | 89.24 | 89.24 | 32.61 | 59 |
| `gemv` | v0 | 2 | 1 | `(4096, 1, 1)` | 38.18 us | 25.80 | 89.55 | 89.55 | 93.68 | 38 |
| `gemv` | v1 | 2 | 1 | `(4096, 1, 1)` | 38.34 us | 25.83 | 89.14 | 89.14 | 93.91 | 27 |
| `gemv` | v2 | 2 | 1 | `(4096, 1, 1)` | 37.70 us | 18.64 | 90.75 | 90.75 | 62.62 | 50 |
| `gemv` | v3 | 2 | 1 | `(1024, 1, 1)` | 37.86 us | 10.46 | 90.24 | 90.24 | 62.13 | 38 |
| `gemv` | v4 | 2 | 1 | `(1024, 1, 1)` | 38.43 us | 26.70 | 88.92 | 88.92 | 92.16 | 24 |
| `int8-quantize` | baseline | 2 | 1 | `(1, 1, 1)` | 149.33 ms | 0.03 | 0.02 | 0.00 | 2.08 | 21 |
| `int8-quantize` | v0 | 2 | 1 | `(4096, 1, 1)` | 7.74 us | 38.62 | 55.52 | 55.52 | 75.40 | 22 |
| `int8-quantize` | v1 | 2 | 1 | `(2048, 1, 1)` | 6.98 us | 30.52 | 61.87 | 61.87 | 78.47 | 21 |
| `int8-quantize` | v2 | 2 | 1 | `(1024, 1, 1)` | 7.17 us | 23.52 | 59.91 | 59.91 | 75.34 | 27 |
| `int8-quantize` | v3 | 2 | 1 | `(1024, 1, 1)` | 6.88 us | 13.52 | 62.57 | 62.57 | 77.32 | 20 |
| `int8-quantize` | v4 | 2 | 1 | `(1024, 1, 1)` | 6.85 us | 13.53 | 62.94 | 62.94 | 67.60 | 32 |
| `softmax` | baseline | 2 | 1 | `(1024, 1, 1)` | 110.08 us | 31.98 | 29.20 | 3.89 | 16.51 | 39 |
| `softmax` | cublas | 2 | 1 | `(1024, 1, 1)` | 10.46 us | 26.53 | 41.78 | 41.05 | 76.23 | 26 |
| `softmax` | smoke | 8 | 1 | `(1024, 1, 1)` | 405.38 us | 51.13 | 42.82 | 4.50 | 36.07 | 39 |
| `softmax` | v0 | 2 | 1 | `(1024, 1, 1)` | 9.06 us | 45.81 | 47.59 | 47.59 | 83.43 | 40 |
| `softmax` | v1 | 2 | 1 | `(1024, 1, 1)` | 9.22 us | 44.91 | 46.81 | 46.81 | 82.97 | 40 |
| `softmax` | v2 | 2 | 1 | `(1024, 1, 1)` | 8.86 us | 46.71 | 48.62 | 48.62 | 82.68 | 40 |
| `softmax` | v3 | 2 | 1 | `(1024, 1, 1)` | 8.51 us | 40.07 | 50.49 | 50.49 | 83.09 | 36 |
| `softmax` | v4 | 2 | 1 | `(1024, 1, 1)` | 7.62 us | 24.55 | 56.74 | 56.74 | 81.10 | 36 |
| `softmax` | v4.2 | 2 | 1 | `(1024, 1, 1)` | 10.72 us | 39.09 | 41.84 | 40.03 | 81.69 | 23 |
| `softmax` | v4.3 | 2 | 1 | `(1024, 1, 1)` | 7.58 us | 25.90 | 56.94 | 56.94 | 77.60 | 40 |
| `softmax` | v4.4 | 2 | 1 | `(1024, 1, 1)` | 8.42 us | 39.84 | 51.23 | 51.23 | 80.43 | 40 |
