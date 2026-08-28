# NCU 报告导出包(供 Mac 端 Nsight Compute GUI 打开)

共 38 份 `.ncu-rep`，按算子分类于 `reports/<算子>/`。逐份摘要见 `manifest.csv`。本文件由 `scripts/export_ncu_for_mac.py` 生成。

## 采集批次

逐份出处见 `manifest.csv` 的 gpu / sm_count / host / ncu_ver / created 列。

| GPU | SM 数 | 主机 | Nsight Compute | 采集日期 | 报告数 |
|---|---:|---|---|---|---:|
| NVIDIA GeForce RTX 4070 Laptop GPU | 36 | ubuntu22 | 2026.1.1.0 (build 37634170) (public-release) | 2026-May-03 | 5 |
| NVIDIA GeForce RTX 4070 Laptop GPU | 36 | ubuntu22 | 2022.4.1.0 (build 32308335) (public-release) | 2026-May-03 | 1 |
| NVIDIA GeForce RTX 4070 Laptop GPU | 36 | ubuntu22 | 2022.4.1.0 (build 32308335) (public-release) | 2026-May-23 | 32 |

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
| `gemv` | 7 | baseline, cublas, v0, v1, v2, v3, v4 |
| `int8-quantize` | 6 | baseline, v0, v1, v2, v3, v4 |
| `softmax` | 11 | baseline, cublas, smoke, v0, v1, v2, v3, v4, v4.2, v4.3, v4.4 |

**未覆盖（零 NCU 数据）**：`activation`, `flash-attn`, `fused-norm`, `gemm`, `rope`, `w8a8`。

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

## 对照臂口径陷阱

- `gemv_cublas_profile` 内是 `gemv2T_kernel_val<...cublasGemvParams...>`，**是真 cuBLAS**，可作标准库对照。
- `softmax_cublas_profile` 内是 `softmax_cublas_kernel`，**是自写 kernel，不是 cuBLAS**（BLAS 规范里没有 softmax）。文件名有误导性，该臂不得用于"对比 cuBLAS"的表述——此为已存档红线项。
- `cuda-reduce` 的标准库对照应为 CUB `DeviceReduce`，本包内无 CUB 臂。
- `*_baseline` 为朴素实现，SOL 与 Occupancy 极低属预期，不代表硬件上限。

## 逐份摘要

取报告内第一个 kernel 实例；`k` 为实例总数，`g` 为不同 grid 数。

| 算子 | 版本 | k | g | Duration | SM % | Memory % | DRAM % | Occ % | Regs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cuda-reduce` | baseline | 2 | 1 | 487.89 ms | 0.15 | 0.15 | 0.06 | 2.08 | 37 |
| `cuda-reduce` | bench | 30 | 1 | 424.15 ms | 0.14 | 0.14 | 0.07 | 2.08 | 37 |
| `cuda-reduce` | v0 | 6 | 3 | 681.63 us | 90.13 | 90.13 | 39.00 | 88.67 | 16 |
| `cuda-reduce` | v1 | 6 | 3 | 681.79 us | 90.15 | 90.15 | 39.03 | 88.68 | 16 |
| `cuda-reduce` | v2 | 6 | 3 | 664.13 us | 91.78 | 91.78 | 39.75 | 88.66 | 16 |
| `cuda-reduce` | v3 | 6 | 3 | 370.02 us | 91.88 | 91.88 | 71.61 | 89.40 | 16 |
| `cuda-reduce` | v4 | 6 | 3 | 317.15 us | 56.99 | 96.68 | 96.68 | 78.39 | 16 |
| `cuda-reduce` | v5 | 1 | 1 | 277.92 us | 48.24 | 96.67 | 96.67 | 82.14 | 16 |
| `cuda-reduce` | v5 | 1 | 1 | 276.74 us | 48.47 | 96.66 | 96.66 | 81.93 | 16 |
| `cuda-reduce` | v5 | 6 | 3 | 316.70 us | 51.93 | 96.55 | 96.55 | 81.32 | 16 |
| `cuda-reduce` | v6 | 1 | 1 | 279.17 us | 6.93 | 96.09 | 96.09 | 87.02 | 17 |
| `cuda-reduce` | v6 | 4 | 2 | 318.50 us | 8.56 | 96.23 | 96.23 | 86.67 | 17 |
| `cuda-reduce` | v7 | 1 | 1 | 319.26 us | 6.10 | 95.79 | 95.79 | 84.05 | 17 |
| `cuda-reduce` | v7 | 4 | 2 | 318.72 us | 7.47 | 96.15 | 96.15 | 84.36 | 17 |
| `gemv` | baseline | 2 | 1 | 754.62 us | 95.92 | 95.92 | 17.66 | 46.21 | 37 |
| `gemv` | cublas | 2 | 1 | 143.65 us | 19.40 | 94.65 | 94.65 | 63.56 | 56 |
| `gemv` | v0 | 2 | 1 | 163.17 us | 42.79 | 95.10 | 95.10 | 95.65 | 38 |
| `gemv` | v1 | 2 | 1 | 162.69 us | 37.34 | 95.41 | 95.41 | 95.52 | 27 |
| `gemv` | v2 | 2 | 1 | 162.34 us | 26.77 | 95.40 | 95.40 | 63.53 | 54 |
| `gemv` | v3 | 2 | 1 | 162.05 us | 15.08 | 95.42 | 95.42 | 88.61 | 38 |
| `gemv` | v4 | 2 | 1 | 162.82 us | 39.04 | 95.08 | 95.08 | 96.38 | 25 |
| `int8-quantize` | baseline | 2 | 1 | 252.36 ms | 0.12 | 0.07 | 0.01 | 2.08 | 21 |
| `int8-quantize` | v0 | 2 | 1 | 27.55 us | 61.03 | 69.07 | 69.07 | 82.10 | 22 |
| `int8-quantize` | v1 | 2 | 1 | 22.66 us | 55.84 | 83.61 | 83.61 | 84.96 | 22 |
| `int8-quantize` | v2 | 2 | 1 | 23.07 us | 43.64 | 81.48 | 81.48 | 85.81 | 25 |
| `int8-quantize` | v3 | 2 | 1 | 22.40 us | 25.26 | 84.85 | 84.85 | 88.90 | 20 |
| `int8-quantize` | v4 | 2 | 1 | 22.40 us | 25.26 | 84.57 | 84.57 | 85.91 | 29 |
| `softmax` | baseline | 2 | 1 | 410.94 us | 52.62 | 44.06 | 4.07 | 36.12 | 39 |
| `softmax` | cublas | 2 | 1 | 31.14 us | 53.08 | 60.31 | 60.31 | 91.02 | 26 |
| `softmax` | smoke | 8 | 1 | 405.38 us | 51.13 | 42.82 | 4.50 | 36.07 | 39 |
| `softmax` | v0 | 2 | 1 | 32.13 us | 75.52 | 75.52 | 58.72 | 91.23 | 38 |
| `softmax` | v1 | 2 | 1 | 32.19 us | 75.52 | 75.52 | 58.38 | 91.24 | 38 |
| `softmax` | v2 | 2 | 1 | 31.78 us | 77.03 | 77.03 | 59.61 | 91.27 | 38 |
| `softmax` | v3 | 2 | 1 | 28.67 us | 70.27 | 70.27 | 66.24 | 90.92 | 36 |
| `softmax` | v4 | 2 | 1 | 23.90 us | 45.61 | 79.14 | 79.14 | 89.42 | 36 |
| `softmax` | v4.2 | 2 | 1 | 32.99 us | 74.07 | 74.07 | 57.59 | 91.42 | 23 |
| `softmax` | v4.3 | 2 | 1 | 23.94 us | 46.56 | 79.51 | 79.51 | 89.55 | 40 |
| `softmax` | v4.4 | 2 | 1 | 29.28 us | 69.62 | 69.62 | 65.64 | 91.03 | 40 |
