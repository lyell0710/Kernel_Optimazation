# CUDA Reduction 性能分析报告（面试版，含 v7）

## 1. 项目目标与结论一句话
本项目通过手写 CUDA reduction，从 `baseline` 逐步优化到 `v7`。  
**最终结论：`v7` 在保证正确性的前提下达到当前最优延迟 `0.277073 ms`，相对 `baseline` 提升 `1256.26x`，并且解决了 `v6` 的性能回退问题。**

---

## 2. 实验环境（关键可复现信息）
- GPU: `NVIDIA GeForce RTX 4070 Laptop GPU`
- Driver: `595.58.03`
- CUDA Toolkit / nvcc: `11.5.119`
- Nsight Systems: `2025.6.3`
- Nsight Compute: `2026.1.1`
- OS: Ubuntu Linux
- 输入规模: `N = 1 << 24`
- 计时方法: CUDA Events，`100` 次均值（含 warmup）

---

## 3. 主结果（benchmark）
数据来源：`project-proof/data/benchmark_results.csv`

| version | latency_ms | speedup_vs_baseline | correctness |
|---|---:|---:|---|
| baseline | 348.076416 | 1.00x | true |
| v0 | 0.413278 | 842.23x | true |
| v1 | 0.460393 | 756.04x | true |
| v2 | 0.454673 | 765.55x | true |
| v3 | 0.309659 | 1124.06x | true |
| v4 | 0.309069 | 1126.21x | true |
| v5 | 0.308650 | 1127.74x | true |
| v6 | 1.740592 | 199.98x | true |
| v7 | **0.277073** | **1256.26x** | true |

关键观察：
- `v6` 出现明显回退（`1.740592 ms`）。
- `v7` 把性能拉回并超过 `v5`（`0.277073 ms` vs `0.308650 ms`，约 **10.23%** 改善）。

---

## 4. 稳定性结果（重复 3 次）
数据来源：`project-proof/data/benchmark_stability.csv`

| version | mean_latency_ms | std_latency_ms |
|---|---:|---:|
| v5 | 0.290299 | 0.000082 |
| v6 | 1.654690 | 0.044254 |
| v7 | **0.273412** | **0.000012** |

解读：
- `v7` 不仅更快，而且波动最小（标准差最低），表现更稳定。
- `v6` 波动和均值都显著更差，符合“存在非 kernel 计算瓶颈”的特征。

---

## 5. 为什么 v6 会慢，v7 为什么恢复

### 5.1 发现问题（Nsight 证据）
`nsys` 的 CUDA API 汇总显示：
- `cudaGetDeviceProperties` 调用 `102` 次，平均约 `1.45 ms/次`
- `cudaMalloc` / `cudaFree` 调用次数也很高（约 `1920+`）

这说明 `v6` 中每次调用函数都做设备属性查询与临时显存管理，带来可见 host 侧开销。

### 5.2 v7 的改动
`v7` 核心做了两件事：
1. **缓存** `cudaGetDeviceProperties` 派生出的 `max_grid`（避免重复查询）  
2. **复用** `d_partial` 缓冲区（避免每次 `cudaMalloc/cudaFree`）

### 5.3 NCU 指标怎么解释
在 `ncu` 单 kernel 对比中，`v7` 的单次 kernel 时间不一定更短；但端到端 benchmark 明显更快。  
这说明优化收益主要来自 **host/API 路径削减**，而不是纯 kernel 算力提升。

---

## 6. 图表索引（面试展示建议顺序）
推荐按“全局 -> 局部 -> 原因”展示：

1. `project-proof/docs/figures/latency_comparison.png`（总体延迟）
2. `project-proof/docs/figures/speedup_comparison.png`（加速比）
3. `project-proof/docs/figures/benchmark_stability.png`（稳定性）
4. `project-proof/docs/figures/ncu_v5_v6_v7_compare.png`（kernel 指标对比）
5. `project-proof/docs/figures/nsys_cuda_api_breakdown.png`（API 开销证据）
6. `project-proof/docs/figures/correctness_check.png`（正确性闭环）

---

## 7. 面试可直接讲的技术亮点
- **性能诊断方法完整**：benchmark -> nsys -> ncu -> 回改代码 -> 再验证。
- **能区分 kernel 瓶颈与 host 瓶颈**：不是只盯 SM/DRAM 指标。
- **优化目标正确**：追求端到端 latency，而非单一 kernel 峰值指标。
- **结果可复现**：数据、图、脚本、版本都在仓库中可追踪。

---

## 8. 快速复现实验（给评审/面试官）
```bash
cd cuda-reduce
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
cmake --build build -j
./build/reduce_bench

# 生成图
python project-proof/scripts/plot_latency.py
python project-proof/scripts/plot_latency_log.py
python project-proof/scripts/plot_latency_line.py
python project-proof/scripts/plot_correctness.py
python project-proof/scripts/plot_speedup.py
python project-proof/scripts/plot_benchmark_stability.py
python project-proof/scripts/plot_ncu_kernel_compare.py
python project-proof/scripts/plot_nsys_cuda_api.py
```

---

## 9. 后续可继续加分的方向
- 把 `v7` 的缓冲区复用改为显式 workspace 管理接口（避免静态状态）。
- 增加不同输入规模（`2^20`~`2^28`）的 scaling 曲线。
- 增加多 GPU/不同架构设备下的横向验证。
