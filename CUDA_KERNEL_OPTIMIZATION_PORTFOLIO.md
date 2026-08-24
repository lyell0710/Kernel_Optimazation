> ⛔ **勘误(2026-08-24,红线级)**:本文所有 softmax "vs cuBLAS" 的对比与
> 归因(含 L2 命中率 2.4 倍、online softmax 推断)**作废**——softmax_cublas.cu
> 实为自写 kernel,并非 cuBLAS 调用(cuBLAS 无 softmax API)。gemv/reduce 的
> cuBLAS 对照经验真有效。详见 records/EXP-K01 §5 勘误。简历禁用 softmax
> 的 cuBLAS 对比句。

# CUDA Kernel 优化 Portfolio

> 四个项目（reduce / softmax / gemv / int8 quantize）的统一入口。
> 主轴是方法论，每个项目是落地，最后一节是跨项目 pattern 总结。

---

## 顶层方法论：我做 GPU kernel 优化的四条原则

四个项目里反复用到的方法论，按重要性排：

### 1. **Profile > 直觉**

任何 GPU 优化的第一步是 `ncu`，不是猜。我每个项目都有一个"被 profiler 打脸"的瞬间：

- **reduce**：v6/v7 grid-stride 慢 6×，我以为是"破坏 coalescing"，NCU 显示 Sec/Ld 还是 4，真正原因是 L2 局部性 + memory-level parallelism 塌了。
- **softmax**：v4 比 cuBLAS 快 26%，我以为是"算法更聪明"，NCU 显示 cuBLAS L2 命中率是我的 2.4 倍（它用了 online softmax），我赢的是 brute force HBM。
- **gemv**：v4 加 shared memory 缓存 vec 反而慢一倍，NCU 显示 BankSt 飙到 13820（v3 是 0），我把 vec 从 L1 多搬了一次到 shared memory。
- **quantize**：v4 char4 store 让 L2 命中率从 32% 暴跌到 2%，看似灾难但 latency 更快——原因是消除了 read-modify-write 的补偿读。

**核心教训**：Profile 不只是用来确认你已知的事，**更重要的是挑战你认为已知的事**。

### 2. **控制变量法做归因**

只做 v0 → v_n 一路加速是不够的，那只能说"每一步都有用"，**说不清每一步贡献多少**。我在 softmax 项目做了三个"反例版本"：

- **v4.2**：保留向量化、去掉 warp shuffle → 性能完全退回 v0
  → 证明 v3→v4 的 30% 加速主要来自 warp shuffle，不是向量化
- **v4.3**：main+tail 显式分离 → cols=1500 上不比 v4 快
  → 证明 v4 的真正退化点是负载不均，不是 tail handling
- **v4.4**：保留 float4 主循环、归约故意制造 bank conflict → 慢于 cuBLAS 8%
  → 证明后置环节退化能拖垮所有前置优化

**核心教训**：没有反例就没有归因。**主动构造对比是科研方法在工程里的应用**。

### 3. **判断 memory-bound vs compute-bound 是优化的第一性问题**

四个项目共同验证：**在 compute-bound 阶段，访存层面的小优化收益接近 0；在 memory-bound 阶段，指令层面的小优化收益接近 0**。

判断依据是 NCU 三个指标交叉验证：
- **SM% 高 + DRAM% 低** → compute-bound
- **SM% 低 + DRAM% 高 + StallLSB% 高** → memory-bound

每个项目里都能看到瓶颈类型从 compute-bound 迁移到 memory-bound 的"拐点"：
- reduce：v3→v4 的拐点（DRAM% 71→96）
- softmax：v3→v4 的拐点（DRAM% 66→76）
- gemv：v0 一步就到 memory-bound（DRAM% 直接 95%）
- quantize：v0→v1 的拐点（DRAM% 61→84）

**核心教训**：**判断瓶颈类型 → 选对应的优化手段**。错配会让你在错误的方向上浪费时间。

### 4. **优化是全链路的，不是单点的（Amdahl 定律在 GPU 上的体现）**

任何 GPU kernel 的整体性能 = 最慢链路的性能。**前面跑得再快，后面被一个 syncthreads 卡住就崩**。

最戏剧化的证据是 softmax 项目的 v4.4：main 循环保留 float4（带宽利用率 100%），但归约阶段一退化（bank conflict + 全程 syncthreads），整体直接慢于 cuBLAS。这跟 gemv v4 同源——加 shared memory 把 vec 从 L1 多搬一次，整体慢一倍。

**核心教训**：优化要"齐头并进"，不能只猛攻最热的循环。**任何被忽略的环节都可能成为新的瓶颈**。

---

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

## 项目 3：CUDA GEMV（真实工程踩坑 + 二次反 cuBLAS）

**算子**：mat (4096×2048 fp32) × vec (2048 fp32) = y (4096 fp32)

**结果**：

| 版本 | 时延 | 角色 |
|---|---|---|
| baseline | 0.618 ms | 单线程参照 |
| v0-v2 (block-per-row) | 0.05-0.055 ms | 折腾访问模式 |
| **v3 (warp-per-row + warp shuffle)** | **0.0325 ms** | **自写最优，比 cuBLAS 快 19%** |
| **v4 (block + shared cache vec)** | 0.0628 ms | **真实踩坑：慢一倍，比 cuBLAS 慢 56%** |
| cuBLAS (`cublasSgemv`) | 0.0402 ms | 通用库基准 |

**关键贡献**：v4 是**真实工程踩坑**（不是构造反例）——加 shared memory 缓存 vec 听起来很合理，结果 BankSt 飙到 13820，比 v3 多了一万多个 conflict。

**NCU 关键数据**：

| 指标 | v3 | cuBLAS | 谁更好 |
|---|---|---|---|
| L2Hit% | 2.7% | **21.0%** | cuBLAS 高 8 倍 |
| DRAM% | **95.2%** | 94.7% | v3 略胜 |
| StallLSB% | **94.8%** | 80.8% | v3 SM 100% 专心等 HBM |
| BankLd / BankSt | **0 / 0** | 0 / 0 | v3 完全不用 shared memory |

**核心叙述**：**v3 跟 softmax v4 是同一个故事的第二次出现**——hardware-optimal beats algorithm-optimal。cuBLAS 用 column-major tiling 让 L2 复用更好（命中率 21% vs 我 2.7%），但被通用性的索引/转置开销拖累；v3 用 row-major + warp shuffle 的极简结构把 HBM 调度器压到 95%。

v4 的踩坑教训特别真实：**vec 已经在 L1 cache 里**（vec 8KB << L1 几十 KB），我用 shared memory 缓存 vec 等于多做了一次"L1 → 寄存器 → shared memory → 寄存器"的搬运。**这种坑你不是设计踩进去的，是因为对硬件状态判断错了而踩进去的**——这种错误的代价是几小时调试 + 一个慢一倍的 kernel，profiler 是唯一的解药。

**行业延伸**：v3 在 mat << L2 的尺寸（这里 32MB ≈ L2 容量）上稳赢；但 mat >> L2 时 cuBLAS 的 L2 优势会消失，v3 的 19% 可能持平甚至反输。**这就是为什么 LLM 推理框架（TensorRT-LLM、vLLM）的工作量很大一部分是"为每个 shape 维护一套 specialized kernel"——本质上就是在做"比 cuBLAS 更激进但 scope 更窄"的库**。

---

## 项目 4：CUDA INT8 Quantize（LLM 推理 vs PyTorch eager）

**算子**：fp32 → int8 per-channel symmetric quantize（1024 channels × 1024 hw = 4MB → 1MB）

**结果**：

| 版本 | 时延 | 角色 |
|---|---|---|
| baseline (GPU 单线程) | 121.35 ms | 参照系 |
| v0 (grid-stride) | 0.0148 ms | 已经比 PyTorch eager 快 3× |
| v3 (block-per-channel + float4 read) | 0.00749 ms | 把 scale 缓存到寄存器 |
| **v4 (+ char4 vectorized store)** | **0.00663 ms** | **自写最优，比 PyTorch eager 快 6.6×** |
| **PyTorch eager (CUDA)** | **0.0437 ms** | 工业基准（`(x/s).round().clamp().to(int8)`）|
| PyTorch `quantize_per_channel` (CPU only) | 2.997 ms | PyTorch 官方 PTQ API |

**关键贡献**：跨语言对标 PyTorch + L2 命中率暴跌但更快的反直觉现象。

**NCU 关键数据**：

| 指标 | v3 | v4 | 含义 |
|---|---|---|---|
| DRAM% | 84% | 85% | 接近算子上限（dtype 不对称导致 85% 是天花板）|
| **L2Hit%** | **32%** | **2.15%** | **暴跌但 latency 更快** |
| StallLSB% | 79% | 82% | SM 越来越闲，越来越纯等 HBM |
| 时延 | 0.0075ms | **0.0066ms** | 快 12% |

**核心叙述**：

**第一个反直觉点：L2 命中率暴跌但更快**。原来 v3 每次 store 1 字节会触发 read-modify-write（硬件读 4 字节 word、改 1 字节、写回），这些补偿读算到 L2 hit 里。v4 用 char4 直接 4 字节整体覆盖，**不需要补偿读，L2 hit 计数自然下降但实际访存量减少**。**同一个 L2 命中率暴跌，reduce v6/v7 是性能问题，quantize v4 是性能优化——profiler 给现象，原因要靠模型推理**。

**第二个反直觉点：C++ v4 比 PyTorch eager 快 6.6×**。原因不是算法更聪明，是 **PyTorch eager 把 `(x/s).round().clamp().to(int8)` 拆成 4 个独立 kernel**，每个都要 materialize 4MB 中间 tensor 到 HBM。总 HBM 流量是 32MB，我 C++ 一个 kernel 全 fuse 在 SRAM 里只有 5MB——**6.4× 的流量差刚好对应 6.6× 的 latency 差**。

**行业延伸**：**这是 Flash Attention 核心思想的最小复现**——经典 attention 是 softmax(Q@K^T) @ V 三个独立 kernel，每个 materialize N×N 中间矩阵；Flash Attention fuse 在一个 kernel 里 N×N 只活在 SRAM。**避免中间 tensor 往返 HBM 是现代 GPU kernel 优化最重要的单一原则**。

---

## 项目 5：CUDA Tensor Core GEMM（wmma 版本梯，2026-08-24 · 4090）

**算子**：fp16 GEMM 4096³（fp32 累加），对照 = 真 `cublasGemmEx`（调用点验真——项目 2 勘误后的标准动作）。

**结果**（3 轮，[EXP-K02](records/EXP-K02_cuda_gemm_tc_ladder.md)）：v0 naive 5.2 → v1 smem tile 6.5 → v2 wmma **89.5** → v3 cp.async 双缓冲 95.5 → v4 128² 大 tile **133.1 TFLOPS = cuBLAS 85.6%**。

**核心叙述**：① compute-bound 算子的台阶是**指令世代**（v1→v2 ×13.8），访存微调只有 +25%——与项目 1-4 的 memory-bound 世界完全相反，先判 bound 类型再动手（原则 3 的另一半）。② v4 理论 occupancy 33% 全梯最低却最快（92 reg×256thr + 32KB smem，每 SM 仅 2 block）——Tensor Core 吞吐靠 fragment 级 ILP 与 smem 复用，不靠线程数遮蔽延迟。③ 与自家 Triton 版（triton-kernels#EXP-T02，160.5 TFLOPS）对照：Triton 编译器发射 mma+ldmatrix+swizzle，wmma API 不暴露 smem swizzle——**同一硬件上"写 CUDA"不等于"到上限"，API 层级本身是性能变量**（剩余差距归因为推断，NCU 不可用）。

---

## 项目 6：CUDA FA2 forward（wmma 架构税的定量测量，2026-08-24 · 4090）

**算子**：Flash Attention 2 前向（在线 softmax，D=128，causal+GQA），协议对齐自家 Triton 版（B=1,Hq=32,Hkv=8,S=4096）。

**结果**（3 轮，[EXP-K03](records/EXP-K03_cuda_fa2_ladder.md)）：v0 warp-per-row 4.9 → v1 smem tile 5.5（+11%，L2 已扛住广播读）→ v2 wmma **24.4**（×4.5）→ v3 8warp 32.5（+33%）→ v4 cp.async 重叠 **34.8 TFLOPS**（仅 +6.6%）；全 shape 过 2e-2 正确性 gate。

**核心叙述**：项目 5 和项目 6 用同一套 wmma 工具箱得到相反结局——**GEMM 够到 cuBLAS 86%，FA2 只够到自家 Triton 版（123 TF，跨 harness）的 28%**。原因是结构性的：wmma accumulator fragment 的 lane→元素映射未定义，行级 softmax（max/exp/α）被迫走 smem 往返 + 每 tile 5 次 `__syncthreads` 的相位链；v4 把 K/V 访存全部预取重叠后只涨 6.6%，坐实瓶颈不在访存在相位链。**越是靠"融合免搬运"吃饭的算子，越需要 mma 级寄存器控制——这就是 FA2 官方实现用 CUTLASS/mma 而非 wmma 的定量理由**（面试里这条比"我写了个 FA2"值钱得多）。

---

# ═══════════════════════════════════════
# 跨项目 Pattern 总结（Portfolio 的核武级最后一节）
# ═══════════════════════════════════════

这一节是把四个项目里**反复出现的硬件规律**拼成一张表。这种"成体系"的展示比单个 0.29ms 数字强 10 倍。

## Pattern 1：Memory-bound 阶段 shared memory 微优化收益≈0

**普适规律**：在 SM 大部分时间在等 HBM 的阶段，shared memory 上的小问题被 HBM 等待时间完全掩盖，bank conflict 消除、modulo 改位运算之类的"小优化"几乎没有时延收益。

**四个项目的证据**：
- ✅ reduce v0→v2：bank conflict 砍 50%，时延几乎不变
- ✅ softmax v0→v2：bank conflict 砍 70%，时延只省 2%
- ✅ gemv v0→v2：Sec/Ld 变差，时延反而略降（HBM 等待掩盖了 ALU 的变差）
- ❌ **quantize v0→v1：位运算优化把 DRAM% 从 61% 拉到 84%**——这是反例！

**反例的解释**：quantize 是 **memory-bound + 简单 ALU**，指令路径短（没有 syncthreads 这种大头），所以 ALU 端的小优化才能浮上来。reduce / softmax / gemv 的指令路径里夹着 syncthreads，那些才是大头，v1 的几个 cycle 完全被掩盖。

**最终推论**：**优化技巧的有效性，取决于它优化的环节在总时间里的占比**。不存在普适有用或普适无用的优化技巧，只存在"匹配当前瓶颈的优化"。

---

## Pattern 2：Hardware-optimal beats algorithm-optimal in specialized scope

**普适规律**：当手写 kernel 打过通用库（cuBLAS / PyTorch）时，**很少是因为算法更聪明，绝大多数是因为你能假设通用库不能假设的东西（输入对齐、shape 固定、dtype 固定），从而把硬件压得更狠**。

**三种"赢"的形态**：

| 项目 | 我赢 | 我的赢法 | 通用库的优势 |
|---|---|---|---|
| **softmax** | v4 比 cuBLAS 快 26% | DRAM% 76% vs 53%（float4 brute force）| L2 命中率 61% vs 我 26%（在线 softmax 算法）|
| **gemv** | v3 比 cuBLAS 快 19% | DRAM% 95% vs 95%（warp shuffle 极简结构）| L2 命中率 21% vs 我 2.7%（column-major tiling）|
| **quantize** | v4 比 PyTorch eager 快 6.6× | 1 kernel vs 4 kernel（fusion 避免中间 tensor）| 灵活性（eager 模式支持动态图）|

**最终推论**：

1. 三个项目里两个都是 "**我赢硬件、通用库赢算法**"——这是个**有普适意义的工程观察**。
2. 第三个 quantize 项目展示了 **fusion** 是第三种赢法——这跟 Flash Attention 的思路一致。
3. **scope 意识**：v3 的 19% 在 mat ≈ L2 的尺寸上稳赢，mat >> L2 时优势可能消失。**这就是为什么 LLM 推理框架要为每个 shape 维护一套 specialized kernel**。

---

## Pattern 3：NCU 指标必须配 latency 推理，不能孤立看

**普适规律**：单看一个 NCU 指标的变化无法判断好坏，必须配合 latency 趋势反推机制。**同一个指标的相同变化，在不同场景下可能意味着完全相反的事情**。

**四个项目的"同指标反含义"案例**：

| 案例 | 指标变化 | 看似 | 实际 |
|---|---|---|---|
| **reduce v6/v7** | L2Hit% 6% → 0.4% | 灾难（确实是）| **真的是灾难**——L2 prefetch 局部性被砍 |
| **quantize v4** | L2Hit% 32% → 2% | 灾难 | **是优化**——消除了 read-modify-write 的补偿读 |
| **gemv v3** | StallLSB% 76% → 95% | 灾难 | **是优化**——SM 100% 专心等 HBM，没有任何资源浪费 |
| **softmax v4** | SM% 71% → 40% | 灾难 | **是优化**——计算开销被砍，HBM 被压满 |
| **reduce v6/v7** | Sec/Ld = 4.00 | 没事（确实没事）| **是 coalescing 没坏**，反而误导了我的诊断方向 |
| **softmax v3 / gemv v3** | Sec/Ld = 16 | 灾难 | **是 float4 满分**（每 lane 16 字节 × 32 lane = 16 sector）|

**最终推论**：**Profiler 给的是现象，原因是要推的**。

会看 NCU 的真功夫不是背指标含义，而是能把"指标变化 + latency 变化 + 算法路径"三个证据组合起来反推机制。这是从"会跑 profiler"到"能用 profiler 解决问题"的核心区别。

---

## Pattern 4：优化是链路，前置优化要等后置优化激活才能兑现

**普适规律**：单链路的优化投资在被激活前是亏的，**前置优化（如 bank conflict 消除、向量化）需要等后置优化（如 warp shuffle、fusion）激活才能体现价值**。同样，**后置环节的退化能拖垮所有前置优化**（Amdahl 定律）。

**四个项目的证据**：

| 项目 | 前置优化 | 激活前 | 激活后 |
|---|---|---|---|
| **reduce** | float4 | 单看 v3 收益 12% | + warp shuffle 后总收益 60%（v3→v4 30% 提升） |
| **softmax** | float4 | v4.2 反例：拿掉 warp shuffle，向量化收益归零 | + warp shuffle 后 v4 比 v3 快 30% |
| **gemv** | warp shuffle 让 shared memory 干脆不需要 | v0-v2 用 shared memory 慢 | v3 抛弃 shared memory 后比 cuBLAS 快 19% |
| **quantize** | float4 read + char4 store 配套 | 只优化 read 端收益有限 | read + store 都向量化后才 squeeze 出最后 12% |

**后置退化拖垮前置的反例**：
- **softmax v4.4**：main 循环保留 float4（带宽利用率 100%），但归约阶段制造 bank conflict + 全程同步 → **整体慢于 cuBLAS 8%**
- **gemv v4**：v3 已经最优，加 shared memory 缓存 vec → **L1 多搬一次反而慢一倍**

**最终推论**：**优化是链，不是点**。

工程含义两条：
1. 单链路投资需要等下游环节激活才能兑现——所以**短期看不到收益不代表方向错了**。
2. 任何被忽略的环节都可能成为新的瓶颈——**最热的循环跑得再快也救不了链路上的一个 syncthreads**。

---

# ═══════════════════════════════════════
# 跨项目的硬件画像对比
# ═══════════════════════════════════════

四个项目共同验证：**不同算子的 DRAM% 上限取决于算法结构本身**，不是优化好坏。

| 算子 | 最优版本 DRAM% | 上限原因 |
|---|---|---|
| reduce | 96% | 纯流式读 + 极少同步，HBM 几乎完全打满 |
| gemv | 95% | mat 流式读，跟 reduce 类似 |
| softmax | 76% | reduce + 计算 + 写回 复合算子，两次同步穿插 |
| quantize | 85% | 读 fp32 写 int8 dtype 不对称（4:1），write 端带宽天然低 |

**这个表本身是个 killer feature**——它说明 **DRAM% 不是越高越好的孤立指标，要跟算子结构对照看**。面试官如果问"你这个 76% 是不是没优化好"，你直接拿这个表给他：**76% 对应 softmax 的算法上限，96% 对应 reduce 的算法上限，不能横向比较**。

---

# ═══════════════════════════════════════
# 怎么用这份 Portfolio 在面试里
# ═══════════════════════════════════════

**短版（5-7 分钟）**：
1. 30 秒讲顶层方法论四条
2. 1 分钟每个项目讲钩子 + 最强一个数字（reduce 6× / softmax 26% / gemv 19% / quantize 6.6×）
3. 1 分钟 Pattern 表选 1-2 条讲透

**长版（15-20 分钟）**：
1. 顶层方法论 2 分钟
2. 每个项目 3 分钟（钩子 + NCU 数据 + 行业延伸）
3. 跨项目 Pattern 5 分钟（这是核武级，留时间讲透）
4. Q&A 时翻到各项目附录

**被追问时的资源定位**：

| 问题方向 | 翻到 |
|---|---|
| GPU 执行模型 / SIMT / warp divergence | reduce 项目附录 B |
| Online softmax / Flash Attention | softmax 项目附录 B |
| Bank conflict 推导 / shared memory 陷阱 | gemv 项目主稿 v4 段 |
| LLM 量化 / fusion 重要性 | quantize 项目主稿 v4 vs PyTorch 段 |
| 精度 / fp16 / bf16 / 量化精度 | 各项目附录 A1 |

每个项目都有独立的详细稿和 NCU SUMMARY.md，本 Portfolio 是顶层入口。
