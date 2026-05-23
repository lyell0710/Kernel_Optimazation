# INT8 Quantize NCU Summary (RTX 4070 Laptop, 1024×1024 fp32 → int8, per-channel)

**Latencies**: from `BENCH_ITERS=100 ./build/int8_quantize_bench` (no profiler).

**NCU metrics**: from `RUN_NCU_CSV=1 bash project-proof/scripts/profile_ncu.sh`,
single kernel launch per version (quantize 是 element-wise 算子，一次 launch 处理整个 tensor).

## C++ versions (cols=1024)

| ver | lat(ms) | SM% | DRAM% | Occ% | Sec/Ld | BankLd | BankSt | L2Hit% | StallLSB% | StallBar% | StallSSB% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 121.34700 | 0.12 | 0.01 | 8.33 | 1.00 | 0.00 | 0.00 | 96.54 | 29.30 | 0.00 | 17.60 |
| v0 | 0.01479 | 62.12 | 60.94 | 80.78 | 2.50 | 0.00 | 0.00 | 34.52 | 32.08 | 0.00 | 9.68 |
| v1 | 0.01191 | 57.03 | 84.43 | 83.89 | 4.50 | 0.00 | 0.00 | 38.26 | 37.79 | 0.00 | 9.99 |
| v2 | 0.01049 | 43.86 | 81.22 | 86.45 | 8.50 | 0.00 | 0.00 | 40.36 | 59.61 | 0.00 | 7.50 |
| v3 | 0.00749 | 25.55 | 84.43 | 89.45 | 3.40 | 0.00 | 0.00 | 31.64 | 79.22 | 0.00 | 5.38 |
| v4 | 0.00663 | 25.39 | 84.85 | 85.89 | 8.50 | 0.00 | 0.00 | 2.15 | 82.12 | 0.00 | 3.10 |

## PyTorch reference latencies（同 shape，CUDA RTX 4070）

| version | lat(ms) | speedup of C++ v4 |
|---|---|---|
| PyTorch eager (CUDA) | 0.0437 | **6.6×** |
| PyTorch quantize_per_channel (CPU only) | 2.9970 | **452.0×** |

## 指标含义

- **SM%**: SM 吞吐占峰值。量化是 memory-bound + 简单 ALU，SM% 通常不会太高。
- **DRAM%**: HBM 带宽利用率。这个 kernel 读 4MB float + 写 1MB int8 = 5MB 总流量。
- **Sec/Ld**: 完美 coalescing=4（float 标量）或 16（float4 vectorized）。
- **BankLd/BankSt**: shared memory bank conflict。Quantize 几乎不用 shared memory。
- **L2Hit%**: scales[c] 数组只有 4KB，应该 100% 命中。input 是流式访问，L2 命中率会低。
- **StallLSB%**: warp 等 HBM 的占比。memory-bound 算子这个值很高。
