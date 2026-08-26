# Reduce NCU Metric Summary (RTX 4070 Laptop, N=16.7M floats)

**Latencies**: from `BENCH_ITERS=100 ./build/reduce_bench` (no profiler attached). **NCU metrics**: from `RUN_NCU_CSV=1 bash project-proof/scripts/profile_ncu.sh` (single launch per version, first kernel stage shown).

| ver | lat(ms) | SM% | DRAM% | Occ% | Sec/Ld | BankLd | BankSt | L2Hit% | StallLSB% | StallBar% |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 347.5929 | 0.15 | 0.06 | 8.33 | 1.00 | 0.00 | 0.00 | 76.71 | 76.43 | 0.00 |
| v0 | 0.3950 | 90.15 | 39.04 | 87.99 | 4.00 | 437.00 | 47855.00 | 11.68 | 17.83 | 34.97 |
| v1 | 0.4723 | 90.15 | 39.08 | 87.99 | 4.00 | 404.00 | 46754.00 | 11.63 | 18.84 | 34.92 |
| v2 | 0.4611 | 91.66 | 45.37 | 88.00 | 4.00 | 398.00 | 19984.00 | 11.50 | 19.26 | 30.75 |
| v3 | 0.2926 | 91.87 | 71.57 | 88.69 | 4.00 | 235.00 | 32946.00 | 6.40 | 15.26 | 32.77 |
| v4 | 0.2917 | 49.72 | 96.57 | 77.79 | 4.00 | 98.00 | 73537.00 | 5.95 | 71.56 | 7.67 |
| v5 | 0.2914 | 52.08 | 96.62 | 80.53 | 4.00 | 193.00 | 68516.00 | 5.95 | 75.15 | 7.30 |
| v6 | 1.6623 | 7.48 | 96.24 | 85.11 | 4.00 | 1.00 | 184.00 | 0.41 | 96.60 | 0.76 |
| v7 | 1.6651 | 8.55 | 96.17 | 84.85 | 4.00 | 2.00 | 220.00 | 0.42 | 96.86 | 0.69 |

## Column meanings

- **SM%**: SM throughput vs peak. Low = SM idle (waiting on memory or stalls).
- **DRAM%**: HBM bandwidth utilization. ~96% = memory-bound saturation.
- **Occ%**: active warps / max possible. Below 50% = launch config or register pressure.
- **Sec/Ld**: avg sectors per global load. Perfect coalescing = 4 (one 128B line per warp). >4 means scatter.
- **BankLd / BankSt**: shared-memory bank conflict count.
- **L2Hit%**: L2 cache hit rate for global loads.
- **StallLSB%**: warp issue stalled on Long Scoreboard = waiting on HBM. >50% = memory-bound.
- **StallBar%**: warp issue stalled on __syncthreads().
