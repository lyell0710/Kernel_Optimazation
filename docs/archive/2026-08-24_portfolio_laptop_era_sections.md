# 归档：PORTFOLIO Laptop 旧代 reduce/softmax 段落

> **superseded by PORTFOLIO.md @ 2026-08-24**
> 移档原因：① reduce 段基于 Laptop 旧代数字（baseline 348ms / v5 差 1.7% / v6/v7 慢 6×），
> 且"v6/v7 反转"叙事的 Laptop 端数据自相矛盾（旧 results 1.665ms vs 旧 stability 0.273ms），
> 按 records/EXP-K01 §5 降级为"不可确证，不作主张"；② softmax 段的 vs-cuBLAS 对比与归因
> 系对照物误标（softmax_cublas.cu 为自写 kernel，红线级勘误，EXP-K01 §5），全部作废。
> 本文为史料留痕（勘误不删叙事）；**其中一切数字禁止对外引用**。

## [项目 1（reduce）原文]

## 项目 1：CUDA Reduce（基础并行 → 接近 cuBLAS）

**算子**：1600 万个 float 求和

**结果**：

| 版本 | 时延 | 角色 |
|---|---|---|
| baseline (`<<<1,1>>>`) | 348 ms | 参照系 |
| v5 (循环展开) | 0.291 ms | 自写最优 |
| cuBLAS | 0.286 ms | 通用库基准（差 1.7%）|
| v6/v7 (grid-stride two-pass) | 1.66 ms | **教学反例，慢 6×** |

**关键贡献**：grid-stride 反例完美演示"profile > 直觉"。

**NCU 关键数据**：

| 指标 | v5 | v6/v7 | 含义 |
|---|---|---|---|
| Sec/Ld | 4.00 | 4.00 | **coalescing 没坏**（直觉错了）|
| L2Hit% | 5.95% | 0.4% | L2 prefetch + 跨 block 搭便车机会被砍 |
| StallLSB% | 75% | **97%** | SM 几乎所有时间在等 HBM |

**核心叙述**：v6/v7 慢 6× 不是因为破坏了合并访存，而是把"多轮 kernel launch"背后隐藏的两个收益——**L2 prefetch 复用、HBM 调度器并发度**——一起砍掉了。reduce 跟 GEMM 的瓶颈结构完全不同：GEMM 瓶颈在 launch 和 occupancy，所以 persistent kernel 纯赚；reduce 瓶颈在 HBM 调度并发度，多轮 launch 看似浪费实际在喂 HBM 调度器。

**行业延伸**：这个 pattern 解释了为什么 NVIDIA 自己的 CUB、Thrust reduce 都是多 kernel 多轮，而不是 persistent two-pass。

完整稿子参见单独文档（reduce 项目主稿 + 附录 A + 附录 B：GPU 执行模型 / SIMT / lockstep）。

---

## [项目 2（softmax）原文]

## 项目 2：CUDA Softmax（控制变量法 + 反 cuBLAS）

**算子**：1024×1024 fp32 矩阵逐行做 softmax

**结果**：

| 版本 | 时延 | 角色 |
|---|---|---|
| v0 | 0.0262 ms | 起点 |
| v4 (float4 + warp shuffle) | **0.0164 ms** | **自写最优，比 cuBLAS 快 26%** |
| **v4.2** (去 warp shuffle) | 0.0262 ms | **反例 #1**：退回 v0，证明 warp shuffle 是向量化的前提 |
| **v4.3** (main+tail 分离, cols=1500) | 0.0227 ms | **反例 #2**：揭示 v4 真正退化点是负载不均 |
| **v4.4** (故意制造 bank conflict) | 0.0236 ms | **反例 #3**：慢于 cuBLAS，证明 Amdahl 定律 |
| cuBLAS | 0.0220 ms | 通用库基准 |

**关键贡献**：v4.2 / v4.3 / v4.4 三个反例做精细归因。

**NCU 关键数据**：

| 指标 | v4 | cuBLAS | 谁更好 |
|---|---|---|---|
| L2Hit% | 25.8% | **61.1%** | cuBLAS 高 2.4 倍（猜测用了 online softmax）|
| DRAM% | **75.9%** | 53.6% | v4 把 HBM 压得更满 |
| StallLSB% | 43.3% | 52.6% | cuBLAS 反而更 memory-bound |

**核心叙述**：我赢 cuBLAS 26% 不是因为算法更聪明——**cuBLAS 算法更聪明**（L2 命中率是我 2.4 倍，强烈怀疑用了 online softmax 算法把三遍扫描压成两遍）。我赢的是用 float4 把 HBM brute force 压满。**cuBLAS 是 algorithm-optimal，v4 是 hardware-optimal，两者在不同假设下都合理**。

**行业延伸**：online softmax 是 Flash Attention 的算法核心。Flash Attention 之所以能把 attention 从 O(N²) memory 降到 O(N)，关键就是用 online softmax 把 softmax 的归约和 attention 的 matmul fused 在一起，N×N 中间矩阵不写 HBM。**我项目里观察到的 cuBLAS L2 行为，直接对接到现代 LLM 推理 attention kernel 的底层逻辑**。

完整稿子参见单独文档（softmax 项目主稿 + 附录 A + 附录 B：Online Softmax & Flash Attention）。

---

## [方法论原则 1 · reduce 旧 bullet]

- **reduce**：v6/v7 grid-stride 慢 6×，我以为是"破坏 coalescing"，NCU 显示 Sec/Ld 还是 4，真正原因是 L2 局部性 + memory-level parallelism 塌了。

## [方法论原则 1 · softmax 旧 bullet]

- **softmax**：v4 比 cuBLAS 快 26%，我以为是"算法更聪明"，NCU 显示 cuBLAS L2 命中率是我的 2.4 倍（它用了 online softmax），我赢的是 brute force HBM。

## [Pattern 2 表 · softmax 旧行]

| **softmax** | v4 比 cuBLAS 快 26% | DRAM% 76% vs 53%（float4 brute force）| L2 命中率 61% vs 我 26%（在线 softmax 算法）|

## [Pattern 2 推论 1 旧句]

1. 三个项目里两个都是 "**我赢硬件、通用库赢算法**"——这是个**有普适意义的工程观察**。

## [面试用法 · 旧数字行]

2. 1 分钟每个项目讲钩子 + 最强一个数字（reduce 6× / softmax 26% / gemv 19% / quantize 6.6×）

## [Pattern 3 表 · reduce v6/v7 旧行]

| **reduce v6/v7** | L2Hit% 6% → 0.4% | 灾难（确实是）| **真的是灾难**——L2 prefetch 局部性被砍 |
| **reduce v6/v7** | Sec/Ld = 4.00 | 没事（确实没事）| **是 coalescing 没坏**，反而误导了我的诊断方向 |
