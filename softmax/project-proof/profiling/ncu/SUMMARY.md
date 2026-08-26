# Softmax NCU Summary (RTX 4070 Laptop, 1024×1024 fp32, fp32 in/out)

> ⛔ **勘误（2026-08-24，红线级）**：下表原「cublas」行**并非 cuBLAS**—— `softmax/src/softmax_cublas.cu` 系自写 warp 原语 kernel（cuBLAS 无 softmax API）， 行名已改为 handwritten_ref；本文一切「vs cuBLAS」解读随之作废， 详见 records/EXP-K01 §5。数据本身（自写参照 kernel 的 NCU 计数）仍有效。

**Latencies**: from `BENCH_ITERS=100 ./build/softmax_bench` (no profiler).

**NCU metrics**： from `RUN_NCU_CSV=1 bash project-proof/scripts/profile_ncu.sh`， single kernel launch per version（softmax 是一个 kernel 处理一整行，所以一次 launch = 一个完整 max+sum+writeback）.

## cols=1024（完美对齐）

| ver | lat(ms) | SM% | DRAM% | Occ% | Sec/Ld | BankLd | BankSt | L2Hit% | StallLSB% | StallBar% |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 0.0263 | 74.78 | 57.92 | 90.94 | 4.00 | 2091.00 | 2633.00 | 8.69 | 17.92 | 25.96 |
| v1 | 0.0262 | 75.20 | 58.48 | 90.98 | 4.00 | 2190.00 | 2505.00 | 8.89 | 17.49 | 26.14 |
| v2 | 0.0257 | 75.83 | 58.79 | 91.37 | 4.00 | 647.00 | 1235.00 | 9.08 | 18.00 | 24.57 |
| v3 | 0.0231 | 70.93 | 66.58 | 91.31 | 16.00 | 1415.00 | 1235.00 | 9.02 | 19.43 | 24.90 |
| v4 | 0.0164 | 40.30 | 75.91 | 88.21 | 16.12 | 1175.00 | 1359.00 | 25.78 | 43.27 | 28.46 |
| v4.2 | 0.0262 | 73.38 | 57.02 | 91.59 | 8.00 | 1422.00 | 804.00 | 34.37 | 24.77 | 21.29 |
| v4.4 | 0.0236 | 53.10 | 69.71 | 86.88 | 15.30 | 4504.00 | 3950.00 | 27.18 | 26.09 | 29.50 |
| handwritten_ref（原误标 cublas） | 0.0220 | 44.76 | 53.62 | 86.59 | 4.49 | 2224.00 | 206.00 | 61.12 | 52.61 | 8.33 |

## cols=1500（非对齐对照）

| ver | lat(ms) | SM% | DRAM% | Occ% | Sec/Ld | BankLd | BankSt | L2Hit% | StallLSB% | StallBar% |
|---|---|---|---|---|---|---|---|---|---|---|
| v4.3 | 0.0227 | 41.80 | 76.22 | 85.72 | 15.30 | 1080.00 | 1242.00 | 25.63 | 40.95 | 28.92 |

## 指标含义

- **SM%**： SM 吞吐占峰值百分比。低=SM 在等内存或同步。
- **DRAM%**： HBM 带宽利用率。Softmax 不像 reduce 能打满 96%——因为每行要做两次归约+一次写回，中间有同步穿插。
- **Sec/Ld**： 每次 global load 的 sector 数。标量 load 完美 coalescing=4。v3/v4/v4.3/v4.4 用 float4 instruction，一次 load 16 bytes×32 lane=512 bytes，等效 sector ~16。注意 NCU 这个指标对 vectorized load 的归一化方式跟标量不同——**这里 16 不代表 coalescing 坏了，反而代表 float4 在正确工作**。
- **BankLd/BankSt**： shared memory bank conflict 次数。**v4.4 的 BankLd 是 v4 的 4 倍（4504 vs 1175），BankSt 是 3 倍（3950 vs 1359），这就是"故意制造 bank conflict"在硬件层面的硬证据**。
- **L2Hit%**： L2 命中率。这是 softmax 项目最反直觉的指标之一——**handwritten_ref（原误标 cublas，非 cuBLAS）的 61% 远高于 v4 的 26%**；原正文的 cuBLAS 归因已作废（见顶部勘误）。
- **StallLSB%**： warp 等 HBM 的占比。越高=越 memory-bound。v4 的 43% 跟 handwritten_ref 的 53% 比，handwritten_ref 反而更 memory-bound——v4 用 float4 把 in-flight 请求压得更紧（对照物系自写 kernel，见顶部勘误）。
