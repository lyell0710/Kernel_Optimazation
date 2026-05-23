# GEMV NCU Summary (RTX 4070 Laptop, 4096×2048 fp32, row-major)

**Latencies**: from `BENCH_ITERS=100 ./build/gemv_bench` (no profiler).

**NCU metrics**: from `RUN_NCU_CSV=1 bash project-proof/scripts/profile_ncu.sh`,
first kernel launch per version (single-kernel algorithm: each block/warp produces one row of `y`).

| ver | lat(ms) | SM% | DRAM% | Occ% | Sec/Ld | BankLd | BankSt | L2Hit% | StallLSB% | StallBar% | StallSSB% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.6181 | 95.81 | 17.64 | 46.56 | 1.00 | 0.00 | 0.00 | 75.68 | 26.21 | 0.00 | 0.03 |
| v0 | 0.0548 | 42.65 | 95.23 | 95.09 | 4.00 | 5.00 | 2972.00 | 2.86 | 79.95 | 5.24 | 4.54 |
| v1 | 0.0551 | 42.59 | 95.20 | 95.08 | 8.00 | 5.00 | 3438.00 | 3.07 | 79.37 | 6.65 | 4.63 |
| v2 | 0.0505 | 30.65 | 95.39 | 63.03 | 16.00 | 2.00 | 4746.00 | 3.08 | 76.11 | 7.03 | 6.32 |
| v3 | 0.0325 | 17.17 | 95.23 | 88.54 | 4.00 | 0.00 | 0.00 | 2.70 | 94.83 | 0.00 | 0.23 |
| v4 | 0.0628 | 39.10 | 95.06 | 96.30 | 4.00 | 29.00 | 13820.00 | 1.65 | 67.08 | 16.35 | 1.91 |
| cublas | 0.0402 | 19.39 | 94.71 | 63.75 | 4.00 | 0.00 | 0.00 | 20.99 | 80.76 | 5.31 | 0.68 |

## 指标含义

- **SM%**: SM 吞吐占峰值。GEMV 是 memory-bound 算子，SM% 通常不高。
- **DRAM%**: HBM 带宽利用率。GEMV 数据量是 mat (4096×2048×4B=32MB) + vec (8KB)，主要靠 mat 的连续读取打带宽。
- **Sec/Ld**: 完美 coalescing=4。v3 用纯 lane-stride，应该是 4；v4 用 shared memory cache vec，索引模式不同。
- **BankLd/BankSt**: shared memory bank conflict。v4 用了 shared memory cache vec，可能在这里浪费时间。
- **L2Hit%**: L2 命中率。vec[c] 被 4 行共用，所以 L2 命中率应该不错。
- **StallLSB%**: warp 等 HBM。memory-bound 算子这个值会很高。
