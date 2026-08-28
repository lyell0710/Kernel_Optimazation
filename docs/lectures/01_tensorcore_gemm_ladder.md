# 01 · Tensor Core GEMM 版本梯:v0→v4 全程走读

> 对象：`gemm/src/` 五个 kernel（v0 naive → v1 smem tile → v2 wmma → v3 cp.async 双缓冲 → v4 128×128 大 tile），对照真 cuBLAS（`cublasGemmEx`，调用点验真）。数字权威：`gemm/project-proof/data/derived_gemm4096_stability.csv`（3 轮 mean±std），实验记录 = records/EXP-K02_cuda_gemm_tc_ladder.md（下文简称 EXP-K02《CUDA Tensor Core GEMM 版本梯》）。协议：M=N=K=4096，fp16 存储 / fp32 累加，行主序，RTX 4090(sm_89)。引用规矩：凡属论文/官方文档的论断一律给出处（标题 + arXiv/DOI 编号 + 章节或公式编号；文档给 URL 路径 + 小节号），关键句给原文；凡属本讲义补出的推导或折算，行内标注「本讲义折算」或「账面推断」；无法用检索确认的说法标注「未核实」。本仓自己的数字一律带 EXP 锚，原始 CSV 路径写在正文里。

## 目录

- [1 这一篇回答什么问题](#1-这一篇回答什么问题)
  - [1.1 本篇要建立的五条能力](#11-本篇要建立的五条能力)
  - [1.2 硬件常数表(RTX 4090 / sm_89,全篇口径)](#12-硬件常数表rtx-4090--sm_89全篇口径)
  - [1.3 本篇引用的一级文献(详细出处与「读它解决什么疑问」见 §8.3)](#13-本篇引用的一级文献详细出处与读它解决什么疑问见-83)
- [2 直觉与第一性原理](#2-直觉与第一性原理)
  - [2.1 三条贯穿全篇的公理](#21-三条贯穿全篇的公理)
- [3 完整推导与机制](#3-完整推导与机制)
  - [3.1 第一步账:判定 compute-bound——v1 注定只有 +25%](#31-第一步账判定-compute-boundv1-注定只有-25)
  - [3.2 wmma fragment 模型:13.8 倍的台阶从哪来,以及它的契约边界](#32-wmma-fragment-模型138-倍的台阶从哪来以及它的契约边界)
  - [3.3 cp.async 组语义逐行推:commit / wait_prior 的在途计数](#33-cpasync-组语义逐行推commit--wait_prior-的在途计数)
  - [3.4 128×128 大 tile 的复用账:+39% 值钱在哪](#34-128128-大-tile-的复用账39-值钱在哪)
  - [3.5 occupancy 33% 全梯最低却最快:驻留计算、发射规则与 ILP](#35-occupancy-33-全梯最低却最快驻留计算发射规则与-ilp)
  - [3.6 魔法数总账:每个常数由谁决定](#36-魔法数总账每个常数由谁决定)
  - [3.7 论文/文档怎么说 vs 本项目实测](#37-论文文档怎么说-vs-本项目实测)
  - [3.8 存储层级的容量与带宽台阶:本梯的数据各落在哪一层](#38-存储层级的容量与带宽台阶本梯的数据各落在哪一层)
- [4 代码逐段走读(按执行顺序)](#4-代码逐段走读按执行顺序)
  - [4.1 v0:正确性锚与性能分母(gemm/src/gemm_v0.cu:13-25)](#41-v0正确性锚与性能分母gemmsrcgemm_v0cu13-25)
  - [4.2 v1:把复用显式化,并引入 barrier 纪律(gemm/src/gemm_v1.cu:24-35)](#42-v1把复用显式化并引入-barrier-纪律gemmsrcgemm_v1cu24-35)
  - [4.3 v2 装载段:float4 协同搬运(gemm/src/gemm_v2.cu:45-56)](#43-v2-装载段float4-协同搬运gemmsrcgemm_v2cu45-56)
  - [4.4 v2 计算段:2×2 微内核,fragment 复用的起点(gemm/src/gemm_v2.cu:57-75)](#44-v2-计算段22-微内核fragment-复用的起点gemmsrcgemm_v2cu57-75)
  - [4.5 v3:异步装载器与调度骨架(gemm/src/gemm_v3.cu:31-44、59-69)](#45-v3异步装载器与调度骨架gemmsrcgemm_v3cu31-4459-69)
  - [4.6 v4 主循环:全部要素合流(gemm/src/gemm_v4.cu:66-91)](#46-v4-主循环全部要素合流gemmsrcgemm_v4cu66-91)
  - [4.7 对照物:真 cuBLAS 与行主序技巧(gemm/src/gemm_cublas.cu:14-22)](#47-对照物真-cublas-与行主序技巧gemmsrcgemm_cublascu14-22)
  - [4.8 bench harness:口径是怎么被一行行钉死的(gemm/src/main.cu:96-124)](#48-bench-harness口径是怎么被一行行钉死的gemmsrcmaincu96-124)
- [5 实验数据怎么读](#5-实验数据怎么读)
  - [5.1 轴与口径](#51-轴与口径)
  - [5.2 三笔自洽性核对](#52-三笔自洽性核对)
  - [5.3 误差列的读法:7.58e-04 是什么](#53-误差列的读法758e-04-是什么)
  - [5.4 这个实验设计防了哪些坑](#54-这个实验设计防了哪些坑)
  - [5.5 数字背后的机理账](#55-数字背后的机理账)
- [6 误区与边界](#6-误区与边界)
- [7 连环追问](#7-连环追问)
- [8 工业对照与延伸](#8-工业对照与延伸)
  - [8.1 与 CUTLASS 逐层对照:本梯缺的是哪几层](#81-与-cutlass-逐层对照本梯缺的是哪几层)
  - [8.2 其他三条对照线](#82-其他三条对照线)
  - [8.3 延伸阅读(每条一句「读它能解决什么疑问」)](#83-延伸阅读每条一句读它能解决什么疑问)

## 1 这一篇回答什么问题

这一篇把 GEMM 从 5.2 TFLOPS(v0)走到 133.1 TFLOPS（v4，真 cuBLAS 的 85.6%，EXP-K02 §5）的每一级增量拆成可核验的账：为什么 smem tile 化只有 +25%，为什么换 wmma 一步 13.8 倍，cp.async 的 commit/wait_prior 计数如何推，128×128 大 tile 的复用账怎么算，以及理论 occupancy 33% 全梯最低的 v4 为什么反而最快。读完你应当能：手推 compute-bound 判定与 tile 复用的算术强度公式；逐行解释 v3/v4 的异步流水线为什么正确；面对「occupancy 低是不是问题」「你和 cuBLAS 差在哪」这类追问给出有实验锚的回答。

### 1.1 本篇要建立的五条能力

1. **判定能力**：拿到一个算子，先用 roofline 三步定出它落在带宽侧还是算力侧，并说清这个判定本身依赖哪些假设（§3.1）。这一步决定了后面所有优化的方向，判错了后面全白做。
2. **指令账能力**：能把「快多少」翻译成「每产出 1 FLOP 要付几条指令」，并据此预言一条优化路线的天花板（§3.1.4）。这是本篇对「v0→v1 为什么只有 +25%」给出的定量解释，也是「为什么必须换指令世代」的硬理由。
3. **契约能力**：能背出 wmma 的 fragment 契约（哪些性质是官方明文承诺的、哪些是实践依赖的隐含前提）与 cp.async 的组语义（commit / wait 的精确定义），并用它们逐行论证代码的正确性，而不是「跑通了就行」(§3.2、§3.3)。
4. **资源账能力**：给定 ptxas 的 reg/smem 数字与架构手册的每 SM 上限，不跑 profiler 就能算出每 SM 驻留几个 block、理论 occupancy 是多少，并知道这个数字买的是什么（§3.5）。
5. **口径能力**：任何数字出口都带 shape、精度、轮数与对照物；知道 85.6% 只属于 4096³/fp16/4090 这一个点，知道哪些结论是实测、哪些是账面推断、哪些根本没有测量手段（§5、§6）。

### 1.2 硬件常数表(RTX 4090 / sm_89,全篇口径)

下表是本篇所有账目的分母。左列的值全部来自官方文档，出处写在最右列；凡本讲义自己折算出来的行，单独标注。

| 量 | 值 | 出处 |
|---|---|---|
| SM 数 | 128 | NVIDIA Ada GPU Architecture 白皮书，Appendix A(GeForce RTX 4090 Full Specifications),"SMs" 行 |
| 每 SM CUDA core | 128 | 同上，"CUDA Cores / SM" 行；另见 CUDA C++ Programming Guide §20.7.1 |
| 每 SM Tensor Core | 4（第四代） | 白皮书 Appendix A "Tensor Cores / SM";CUDA PG §20.7.1「4 mixed-precision Fourth-Generation Tensor Cores ... for compute capability 8.9」 |
| Boost 时钟 | 2520 MHz | 白皮书 Appendix A,"GPU Boost Clock (MHz)" |
| FP32 峰值（非 Tensor） | 82.6 TFLOPS | 白皮书 Appendix A,"Peak FP32 TFLOPS (non-Tensor)" |
| FP16 Tensor 峰值（FP32 累加） | 165.2 TFLOPS（稠密）/ 330.4（稀疏） | 白皮书 Appendix A，"Peak FP16 Tensor TFLOPS with FP32 Accumulate"；脚注 2 说明第二个数是稀疏特性下的有效值 |
| 显存位宽 / 数据率 / 带宽 | 384-bit / 21 Gbps / 1008 GB/s | 白皮书 Appendix A,"Memory Interface"、"Memory Clock (Data Rate)"、"Memory Bandwidth" |
| L2 容量 | 73728 KB = 72 MiB | 白皮书 Appendix A,"L2 Cache Size" |
| 每 SM L1/shared 统一缓存 | 128 KB | CUDA PG §20.7.1「a unified data cache and shared memory with a total size of ... 128 KB for devices of compute capabilities 8.6 and 8.9」 |
| 每 SM shared memory 上限 | 100 KB | CUDA PG §20.2 Table 27,"Maximum amount of shared memory per SM"，列 8.9 |
| 每 block shared memory 上限 | 99 KB | 同上，"Maximum amount of shared memory per thread block"；静态 `__shared__` 仍限 48 KB(§20.7.3) |
| shared memory bank 数 / 每 bank 位宽 | 32 / 32-bit | CUDA PG §20.2 Table 27 "Number of shared memory banks";CUDA C++ Best Practices Guide §10.2.3.1「Each bank has a bandwidth of 32 bits every clock cycle」 |
| 每 SM 32-bit 寄存器 | 64 K | CUDA PG §20.2 Table 27,"Number of 32-bit registers per SM" |
| 每 block 32-bit 寄存器上限 | 64 K | 同上 |
| 每线程寄存器上限 | 255 | 同上 |
| 每 SM 常驻 warp / block 上限 | 48 / 24 | 同上，"Maximum number of resident warps per SM"、"resident blocks per SM" |
| 每 SM warp 调度器 | 4（每个 1 个发射单元） | CUDA PG §20.7.1「4 warp schedulers」；白皮书 §Ada GPU Architecture In-Depth「the AD10x SM is divided into four processing blocks (or partitions), with each partition containing a 64 KB register file, an L0 instruction cache, one warp scheduler, one dispatch unit」 |
| 机器平衡点 I\* = 165.2 TFLOPS / 1008 GB/s | ≈ 164 FLOP/B | **本讲义折算**（两个端点均取上表） |
| 每 SM 每周期 Tensor MAC | 256 | **本讲义折算**：165.2e12 / 2 / (128 × 2.52e9) = 256 |
| 每 SM shared memory 峰值带宽 | 128 B/cycle | **本讲义折算**：32 bank × 4 B/bank/cycle(Best Practices §10.2.3.1) |

两条值得单独记住的对照：白皮书正文说完整 AD102 带 98304 KB(96 MiB)L2(§Memory Subsystem：「AD102 has been outfitted with 98304 KB of L2 cache， an improvement of 16x over the 6144 KB that shipped in GA102」)，而 RTX 4090 是裁剪版，Appendix A 的表给的是 73728 KB。本仓设备实测 L2 = 72.0 MB（EXP-K04《标准库基准补齐与两区间重测》§2），与 Appendix A 一致——**引用 L2 容量时必须区分「完整芯片」与「这张卡」**，差了三分之一。第二条：1008 GB/s 是 384 bit × 21 Gbps ÷ 8 的直接结果（本讲义折算：384 × 21e9 / 8 = 1.008e12 B/s），本仓从设备属性实测反推为 1008.1 GB/s(EXP-K04 §3)，两者一致，说明这条理论峰值不是一个需要打折的营销数字。

### 1.3 本篇引用的一级文献(详细出处与「读它解决什么疑问」见 §8.3)

- roofline:Williams, Waterman & Patterson, "Roofline: an insightful visual performance model for multicore architectures", Communications of the ACM 52(4):65-76, 2009, DOI:10.1145/1498765.1498785。
- Tensor Core / wmma 契约：CUDA C++ Programming Guide §10.24 "Warp Matrix Functions"。
- Tensor Core / mma 布局：PTX ISA §9.7.15 "Warp Level Matrix Multiply-Accumulate Instructions"，特别是 §9.7.15.5.8 与 §9.7.15.5.15。
- cp.async 组语义：PTX ISA §9.7.9.26.3.1 / §9.7.9.26.3.2 / §9.7.9.26.3.3;C 层等价定义见 CUDA PG §10.28.4。
- 架构参数：NVIDIA Ada GPU Architecture 白皮书；CUDA PG §20.7(Compute Capability 8.x)。
- 算术强度与 tile/wave quantization:NVIDIA Deep Learning Performance Guide,"Matrix Multiplication Background User's Guide" §2、§3.1、§3.2;"GPU Performance Background User's Guide" §4。
- 工业级分层：CUTLASS 官方文档 "Efficient GEMM in CUDA"(docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)与 `include/cutlass/gemm/threadblock/mma_multistage.h`。

## 2 直觉与第一性原理

先想没有这套东西的世界。GEMM 的原始定义是三重循环：C[i][j] = Σ_k A[i][k]·B[k][j]。4096³ 要做 2·4096³ ≈ 1.374×10¹¹ 次浮点运算（每输出 1 乘 1 加）。若每次乘加都从显存现取两个操作数，访存量是运算量的同数量级——显存带宽立即成为天花板，再多算力也白给。所以 GEMM 优化的全部主题只有两个：**让每个字节被算尽可能多次（复用）**，以及**让做乘加这件事本身尽可能便宜（指令）**。

一个日常类比：工厂流水线。v0 是每拧一颗螺丝跑一趟仓库；v1 是把一箱零件搬到工位旁（smem tile）；v2 是把手动螺丝刀换成气动批（Tensor Core：一条指令做 4096 次乘加）；v3 是安排专人在你干活时提前去搬下一箱（cp.async 预取）；v4 是把工位加大，一箱零件能装配更多产品（大 tile 提高复用）。类比失效点：工厂里搬运工和装配工是不同的人，而 GPU 里 cp.async 的「搬运」占用的是同一批线程发出的异步拷贝引擎，「重叠」不是免费的并行，而是把 DMA 排进指令流后靠计数器等待——这正是 §3.3 要精确推导的部分；另一个失效点是「工位加大」在 GPU 上有硬约束（寄存器与 smem 预算），放大到某一步会把驻留 block 数压成 0，类比给不出这个折断点，§3.5 的驻留计算才给得出。

### 2.1 三条贯穿全篇的公理

- **公理 A（先判定，再优化）**：优化方向由瓶颈类型决定，不由教科书顺序决定。roofline 原论文把这句话写成了一条不等式：可达性能 = min（峰值算力， 峰值带宽 × 算术强度）(Williams et al. 2009，§"The Roofline Model")。落在斜边上就去提复用，落在平顶上就去省指令。本梯的实测把这条抽象命题变成了两个可对照的数字：tiling +25%，换指令世代 13.8 倍。
- **公理 B（指令是一种资源，和字节一样稀缺）**：「算得快」不只是「算力大」。每个 SM 每周期最多发射 4 条指令（4 个调度器各 1 条，CUDA PG §20.7.1），这条上限与 FP32 单元的吞吐是两个独立的天花板。当一条数学指令要陪跑 4 条非数学指令时，数学单元利用率的上界就是 1/5——这是 §3.1.4 的核心。
- **公理 C（每一个魔法数背后必须站着一个约束）**：BM=BN=128、BK=32、LDS=68 这类常数，要么由理论上界决定（复用公式），要么由硬件约束决定（smem 上限、对齐规则、bank 数），要么由实测扫描决定。§3.6 把本梯的每一个常数归到这三类之一；**归不进去的常数就是没想清楚的常数**。

## 3 完整推导与机制

### 3.1 第一步账:判定 compute-bound——v1 注定只有 +25%

#### 3.1.1 roofline 的原始表述与它的两个前提

roofline 原论文给出的模型是(Williams, Waterman & Patterson, CACM 52(4), 2009,DOI:10.1145/1498765.1498785,§"The Roofline Model"):

$$\text{Attainable GFLOPS/s} = \min\big(\text{Peak Floating-Point Performance},\ \text{Peak Memory Bandwidth} \times \text{Operational Intensity}\big)$$

论文对 operational intensity 的定义是「operations per byte of DRAM traffic」，并明确写了这里的 traffic 指的是**缓存与 DRAM 之间**的流量，不是处理器与缓存之间的流量。NVIDIA 的性能指南用的是同一件事的另一套词：算术强度定义为 "the ratio of algorithm implementation operations and the number of bytes accessed"，处理器侧的对应量叫 ops：byte ratio，即 "the ratio of a processor's math and memory bandwidths"(GPU Performance Background User's Guide §4 Understanding Performance)。同一节还把限制因子列成三条："Performance of a function on a given processor is limited by one of the following three factors； memory bandwidth， math bandwidth and latency."

**这个模型有两个前提，不满足就不能用**：

1. **两个端点都要取得到**。峰值算力与峰值带宽都是理想值；若某条路径（比如 fp16 在 CUDA core 上算）根本达不到标称峰值，拿标称峰值画出来的屋顶就是假的。§3.1.4 会看到 v1 的真实天花板远低于 82.6 TFLOPS，这不是 roofline 错了，是**用错了屋顶**。
2. **DRAM 流量要是真的 DRAM 流量**。若数据装得进缓存，测到的「带宽」是缓存带宽，横轴的算术强度也就不是论文定义的那个量。本仓在 reduce 上踩过这个坑并留了实证：67.1 MB 的数组小于 4090 的 72 MB L2，测出来的等效带宽是理论峰值的 1.8 至 3.4 倍（EXP-K04 §4.1）——**等效带宽超过理论峰值就是「数据没落 DRAM」的判据**。第二篇讲义 §8.3.2 的附课把这条讲透。

#### 3.1.2 本算子的 I 与 I\*:每一步为什么合法

按 roofline 三步做判定（口径与 docs/talk/whiteboard_card_roofline.md 一致）：

1. 算术强度 I = FLOPs / 必需字节。4096³ fp16 GEMM：
   - FLOPs = 2·M·N·K = 2·4096³ ≈ 1.374×10¹¹。**为什么是 2 而不是 1**：每个输出元素累加 K 次，每次一乘一加，记 2 FLOP；这是 BLAS 与 NVIDIA 文档的通行口径（Matrix Multiplication Background User's Guide §1），本仓 bench 的公式与之一致（gemm/src/main.cu：92 的 `2.0 * M * N * K`）。换口径（只记乘法）会让所有 TFLOPS 数字腰斩，跨仓比较必须先对齐这一点。
   - 必需字节 = (M·K + K·N + M·N)·2B = 3·4096²·2B ≈ 100.7 MB。**为什么这是「必需」**：A、B 至少各要从 DRAM 读一次，C 至少要写一次；任何实现都不可能比这更少。这是一个**下界**，所以由它算出的 I 是**上界**——用上界去论证「compute-bound」是安全的方向（若上界都低于 I\*，那才是 memory-bound；这里上界远高于 I\*，结论只会更强）。
   - I ≈ 1.374×10¹¹ / 1.007×10⁸ ≈ 1365 FLOP/B。
2. 平衡点 I\* = 峰值算力 / 峰值带宽 = 165.2 TFLOPS / 1.008 TB/s ≈ 164 FLOP/B(§1.2)。作为对照，NVIDIA 的 GEMM 指南给 V100 上 FP16 with FP32 accumulate 的 FLOPS：B 是 138.9(Matrix Multiplication Background User's Guide §2 Math And Memory Bounds)——同一个量，不同世代，数值同一量级，说明 164 这个数不是本仓算错了。
3. I ≈ 1365 ≫ I\* ≈ 164，深度 compute-bound：只要复用做得起码及格，带宽就不是限制，收益必须从指令侧找。

**边界条件**：这三步用的是「整个 GEMM」的算术强度，它默认了实现能把复用做到理论上界。真实实现的算术强度由 tile 尺寸决定，NVIDIA 给的实现侧公式是 `M·N·K / (M·K + N·K + M·N)`（同上 §2），把 M、N 换成 tile 的 BM、BN 就得到 §3.4 的复用公式。**所以 §3.1 判定的是「这个算子有没有希望 compute-bound」，§3.4 判定的是「这个实现有没有做到」——两个问题，不能混。**

#### 3.1.3 v0 的带宽账:三种「字节」必须分开数

v0 复用为零，先把它的访存量算清楚。这里要区分三种字节，混起来算就会得出荒谬结论：

- **指令级请求字节**：每次乘加各取一个 A、B 元素，总请求 = 2·M·N·K·2B ≈ 2.749×10¹¹ B ≈ **275 GB**。
- **L1↔L2 的事务字节**：硬件按 32 B sector 搬运，warp 内合并之后远小于请求字节。v0 的 block 是 16×16，一个 warp 覆盖 2 行 × 16 列：`B[k*N+col]` 的 16 个 half 连续 = 32 B，恰好一个 sector；`A[row*K+k]` 只有 2 个不同地址，分属 2 个 sector。
- **DRAM 字节**：下界是 100.7 MB(§3.1.2)。

**一个不需要 profiler 的上界反推**：实测 26.369±0.472 ms(EXP-K02 §5)，若这段时间显存一直满速跑，DRAM 流量至多 26.369e-3 × 1008e9 ≈ **26.6 GB**；而请求字节是 275 GB。所以缓存层次至少替我们做掉了 275/26.6 ≈ 10 倍的复用（**本讲义折算，上界式论证**）。反过来说，若真按 275 GB 全打 DRAM，需要 273 ms，是实测的 10 倍——**v0 并不是「纯 DRAM 饥饿」**。

那 275 GB 是从哪儿供出来的？按 §1.2 折算，shared/L1 这一层的峰值是 128 B/cycle/SM，全卡 128 × 128 × 2.52e9 ≈ 41 TB/s（**本讲义折算；该带宽数字由 shared memory 的 bank 结构推得，把它当作 L1 的上界是推断，官方未直接给出 L1 带宽指标，标注账面推断**）。v0 的等效请求带宽 = 275 GB / 26.369 ms ≈ 10.4 TB/s，落在这个上界之内，而远高于本仓在 L2 常驻区间实测到的 3.4 TB/s 量级（EXP-K04 §4.1，CUB 在 67 MB 上的等效带宽）。**结论：v0 的复用主要发生在 L1，不是 L2，更不是 DRAM。**

#### 3.1.4 v1 为什么只有 +25%:指令路径账(本篇第一个关键推导)

v1 把复用显式化：32×32 tile 里每个元素装载一次供 32 个线程使用，全局读降到（M/32）(N/32)·2·32·K·2B ≈ 8.6 GB（本讲义折算：16384 个 block，每 block 读 2×32×4096 个 half = 512 KB），摊到实测 21.114 ms 上仅约 408 GB/s，远低于带宽峰值；且 A、B 各 32 MB，合计装得进 4090 的 72 MB L2，DRAM 强制流量只有约 100 MB。**v1 不缺带宽。**

那它缺什么？EXP-K02 §6 的原话是「~6.5 TFLOPS 已近该路线实际上限」。这句话是对的，但太粗——**「算力上限」到底是哪个上限？** 把指令数一条条数出来就清楚了：

内层循环 `acc += __half2float(As[ty][k]) * __half2float(Bs[k][tx])` 每产出 1 次 FMA(2 FLOP)，warp 级要发：

| 指令 | 条数 | 说明 |
|---|---|---|
| LDS（shared 读 As） | 1 | v0 对应 LDG |
| LDS（shared 读 Bs） | 1 | v0 对应 LDG |
| CVT f32.f16（As 元素） | 1 | `__half2float` |
| CVT f32.f16（Bs 元素） | 1 | 同上 |
| FFMA | 1 | 唯一一条数学指令 |
| **合计** | **5** | 其中数学指令占 1/5 |

于是可以给出一个**与实测无关的天花板**（本讲义折算）：

- 每 SM 每周期最多发射 4 条指令（4 个调度器 × 1 个发射单元，CUDA PG §20.7.1:"An SM statically distributes its warps among its schedulers. Then, at every instruction issue time, each scheduler issues one instruction for one of its assigned warps that is ready to execute, if any."）。
- 5 条指令里只有 1 条产出 2 FLOP × 32 lane = 64 FLOP。
- ⇒ 该指令路径的算力上界 = 4/5 × 64 FLOP/cycle/SM × 128 SM × 2.52e9 Hz ≈ **16.5 TFLOPS**，只有 FP32 标称峰值 82.6 TFLOPS 的 **1/5**。

实测 6.5 TFLOPS 是这个 16.5 的 39%（剩下的差距是延迟未被完全遮蔽，本讲义未做进一步分解，标注开放）。**但结论已经足够硬：哪怕把 v1 的延迟遮蔽做到满，这条路线的顶也只有 16.5 TFLOPS，离 Tensor Core 路线的 165.2 差一个数量级。**

这笔账顺带解释了 v0→v1 只有 +25% 的**上限从哪来**：v0 的每 FMA 指令条数同样是 5（只是把 2 条 LDS 换成 2 条 LDG）。**指令路径没变，天花板就没变**；tiling 改善的只是这 2 条访存指令的延迟与事务数，不可能改变「5 条指令产 2 FLOP」这个比例。所以 +25% 不是 v1 写坏了，而是**这个方向本来就只有这么多**。

对照 Tensor Core 路线：一条 `wmma::mma_sync` 的 16×16×16 做 4096 次乘加 = 8192 FLOP。它在 sm_89 上展开为 2 条 `mma.m16n8k16`（**本讲义折算**：m16n8k16 每条覆盖 16×8×16 = 2048 MAC，两条恰好拼出 16×16×16；wmma 的实际 SASS 展开未逐条反汇编核对，标注推断级），加上操作数装载，v4 的每 kk 步是 6 次 `load_matrix_sync` 喂 8 次 `mma_sync` = 约 16 条 mma 加数十条装载指令，产出 8 × 8192 = 65536 FLOP。**每条指令产出的 FLOP 从 0.4 变成 10³ 量级**——这才是 13.8 倍台阶的机制，不是「Tensor Core 更快」这种同义反复。

所以 v0→v1 的 +25% 是本梯最重要的测量之一：它证明访存微调在这个算子上没有出路，台阶只能来自指令世代。

### 3.2 wmma fragment 模型:13.8 倍的台阶从哪来,以及它的契约边界

#### 3.2.1 官方文档给了什么

`wmma::mma_sync` 是 warp 级协作指令：一个 warp 的 32 个线程共同完成一次 16×16×16 的小矩阵乘加 D = A·B + C。CUDA C++ Programming Guide §10.24 开头给的定语值得逐字读："C++ warp matrix operations leverage Tensor Cores to accelerate matrix problems of the form D=A*B+C. ... This requires co-operation from all threads in a warp. In addition, these operations are allowed in conditional code only if the condition evaluates identically across the entire warp, otherwise the code execution is likely to hang."

两条硬约束由此确定：**(a) 全 warp 必须一起调用**；**(b) 放在分支里时，分支条件必须全 warp 一致**。本仓的 v2/v3/v4 把 wmma 调用全部放在无分支的 `#pragma unroll` 循环里，正是为了不去踩（b）；第二篇讲义的 FA2 v2 则要靠 `nlimit` 全 block 一致来保证同一件事。

#### 3.2.2 fragment 的三条性质,以及它们各自的出处

1. **fragment 是 warp 级寄存器容器**。文档定义："An overloaded class containing a section of a matrix distributed across all threads in the warp."（CUDA PG §10.24.1，`fragment` 条）一个 16×16 fp32 accumulator 的 256 个元素分布在 32 个 lane 的寄存器里，每 lane 摊 8 个（256/32）——这也是后面算寄存器账的依据（§3.5）。
2. **lane→元素映射不公开**。文档原文只有一句，但这一句是本仓两篇讲义的全部戏眼："**The mapping of matrix elements into fragment internal storage is unspecified and subject to change in future architectures.**"（CUDA PG §10.24.1，`fragment` 条）注意它说了两件事：一是**现在**不告诉你，二是**将来**还可能变——所以就算你反汇编测出了 sm_89 上的映射，也不能写进代码。
3. **唯一可依赖的对称性，以及它的确切范围**。文档在 `mma_sync` 之后写道："Because the map of matrix elements into each thread's fragment is unspecified, individual matrix elements must be accessed from memory (shared or global) after calling `store_matrix_sync`. In the special case where all threads in the warp will apply an element-wise operation uniformly to all fragment elements, direct element access can be implemented using the following fragment class members." 随后给出的示例是把一个 accumulator 整体乘以一个**全 warp 相同的标量** alpha。

**性质 3 的边界必须说清楚，不能含糊**。文档明文授权的是「全 warp 一致地对所有 fragment 元素施加同一个逐元素运算」。本仓依赖的有两种写法：

- `h.x[e] = __float2half(acc[i][j].x[e]);`(gemm_v2.cu：91)：对同一个 fragment 的每个元素做同一个一元运算，**完全落在文档明文授权的范围内**。
- `pv.x[e] += oacc.x[e]`（第二篇讲义的 fa2_v2.cu：160）：把**两个同 shape、同 Use、同数据类型**的 accumulator fragment 逐元素相加。文档没有正面写这一句，它成立依赖一个未明文的前提——同一编译单元、同一 shape/类型的 fragment 使用同一套（未指定的）映射。这个前提在实践中成立（否则 `mma_sync(d, a, b, c)` 里 c 与 d 的对应关系本身就无法定义），但**它不是文档承诺，本讲义标注「文档未明文，实践依赖」**。本仓正确性 gate 全过（err 4.88e-04，EXP-K03《CUDA FA2 forward 简化版版本梯》§5）是这条依赖成立的间接证据，不是证明。

性质 2 对 GEMM 无害（GEMM 不需要知道元素在哪一行），对 FA2 是致命税负（行级 softmax 必须知道行号）——这是第二篇讲义的主线，伏笔埋在这里。

#### 3.2.3 load/store_matrix_sync 的对齐契约,与本仓的逐条核对

这一条常被当成「玄学」，其实文档写得很死（CUDA PG §10.24.1，`load_matrix_sync` 与 `store_matrix_sync` 条）：

> "mptr must be a 256-bit aligned pointer pointing to the first element of the matrix in memory. ldm describes the stride in elements between consecutive rows (for row major layout) or columns (for column major layout) and must be a multiple of 8 for `__half` element type or multiple of 4 for `float` element type. (i.e., multiple of 16 bytes in both cases). ... The values of mptr, ldm, layout and all template parameters for a must be the same for all threads in the warp."

拆成三条可核对的约束：

- **C1**：指针 256-bit 对齐，即 **32 字节**对齐（注意不是 16 字节）。
- **C2**：`ldm` 对 `__half` 是 8 的倍数，对 `float` 是 4 的倍数（两者都等价于 16 字节）。
- **C3**：同 warp 内 `mptr`、`ldm`、`layout` 与全部模板参数必须一致。

把本梯的每一处调用对着核一遍（**本讲义核对**）：

| 调用点 | 类型 | ldm | C2 | 指针字节偏移 | C1 |
|---|---|---|---|---|---|
| gemm_v2.cu:66 `af[i] ← As[wr*32+i*16][kk]` | half | BK=32 | 32 % 8 = 0，满足 | (r·32+kk)·2 = 64r+2kk,r∈{0,16,32,48},kk∈{0,16} → 最小非零 32 | 均为 32 的倍数，满足 |
| gemm_v2.cu:69 `bf[j] ← Bs[kk][wc*32+j*16]` | half | BN=64 | 64 % 8 = 0，满足 | (kk·64+c)·2 = 128kk+2c,c∈{0,16,32,48} | 均为 32 的倍数，满足 |
| gemm_v4.cu:79 `af[i] ← As[p][wr*64+i*16][kk]` | half | BK=32 | 满足 | 同 v2 形式，r∈{0,16,…,112} | 满足 |
| gemm_v2.cu：92 `store → C[...]` | half | N=4096 | 4096 % 8 = 0，满足 | 行首偏移 =（行号·4096+列号）·2，列号为 16 的倍数 | 满足 |

**为什么要做这一步核对**：C1/C2 不满足时的行为是未定义，而未定义**通常不会崩**——它给你一片错位的数据。这类错误在 4096³ 的随机输入上表现为「结果全错但程序正常退出」，是本梯正确性 gate（相对误差 < 2e-2）存在的第一理由。第二篇讲义的 LDS=68 / LDP=72 两个魔法数，就是 C2 的直接产物：68 是「大于 64 且是 4 的倍数」的最小值，72 是「大于 64 且是 8 的倍数」的最小值——**不是拍脑袋填出来的，是被 C2 逼出来的唯一解**。

#### 3.2.4 13.8 倍台阶的控制变量设计

v2 刻意保留了与 v1 同级的朴素同步装载（gemm_v2.cu：7 的文件头写明「访存策略刻意保持与 v1 同级的朴素（同步装载），使 v1→v2 的差异只剩指令世代」），使这一级的唯一变量就是指令世代。实测 13.8 倍（6.5→89.5 TFLOPS，EXP-K02 §5）。**这个设计是本梯最值钱的地方**：如果 v2 同时换了指令又换了装载方式，13.8 倍就没法归因，只能说「新版本更快」。

顺带核一个数：89.5 TFLOPS 是 165.2 峰值的 54.2%（本讲义折算）。v2 用的是最朴素的同步装载与 2×2 微内核，一步就吃到峰值的一半以上——这正说明前面 v0/v1 的瓶颈完全不在数学单元。

### 3.3 cp.async 组语义逐行推:commit / wait_prior 的在途计数

v2 的问题：每个 K 块「同步装载 → 计算」串行，Tensor Core 在装载期间空等。v3 引入 `__pipeline_memcpy_async`（编译为 cp.async：全局内存→shared memory 的异步拷贝，不经寄存器中转），配 smem 双缓冲 `As/Bs[2][...]`。它的正确性完全建立在一套「组计数」上，而这套计数在两层文档里各有一份定义——**两份都要读，因为它们各自补上了对方没说的那一半**。

#### 3.3.1 PTX 层:三条指令的原文语义

PTX ISA(Release 9.3)的三节：

- **§9.7.9.26.3.1 `cp.async`**:"cp.async is a non-blocking instruction which initiates an asynchronous copy operation of data from the location specified by source address operand src to the location specified by destination address operand dst. Operand src specifies a location in the global state space and dst specifies a location in the shared state space." 尺寸约束写得很死："Operand cp-size is an integer constant which specifies the size of data in bytes to be copied to the destination dst. **cp-size can only be 4, 8 and 16.**" 缓存行为：".cg qualifier indicates caching of data only at global level cache L2 and not at L1 whereas .ca qualifier indicates caching of data at all levels including L1 cache." 同时强调："There is no ordering guarantee between two cp.async operations if they are not explicitly synchronized using cp.async.wait_all or cp.async.wait_group or mbarrier instructions."
- **§9.7.9.26.3.2 `cp.async.commit_group`**:"cp.async.commit_group instruction **creates a new cp.async-group per thread** and batches all prior cp.async instructions initiated by the executing thread but not committed to any cp.async-group into the new cp.async-group. **If there are no uncommitted cp.async instructions then cp.async.commit_group results in an empty cp.async-group.**" 另有一句在多缓冲场景里很要命："There is no memory ordering guarantee provided between any two cp.async operations within the same cp.async-group. So two or more cp.async operations within a cp.async-group copying data to the same location results in undefined behavior."
- **§9.7.9.26.3.3 `cp.async.wait_group` / `cp.async.wait_all`**:"cp.async.wait_group instruction will cause executing thread to wait till only **N or fewer of the most recent cp.async-groups are pending** and all the prior cp.async-groups committed by the executing threads are complete." 可见性条款单独列出："Writes performed by cp.async operations are made visible **to the executing thread** only after: 1. The completion of cp.async.wait_all or 2. The completion of cp.async.wait_group on the cp.async-group in which the cp.async belongs to or 3. mbarrier.test_wait returns True ..."

从这三段里提炼出四条本梯直接用到的事实：

- **F1（每线程语义）**：组是**每线程**的。同一个 block 里不同线程的组计数各自独立，barrier 不会把它们对齐。
- **F2（空组也是组）**：`commit_group` 即使没有待封的拷贝也会造出一个空组。所以「条件性地 commit」会让不同线程的组序列长度不同——这是最隐蔽的一类错误。
- **F3（可见性只到发起线程）**：wait 之后，拷贝结果只对**发出这条 cp.async 的线程**可见。别的线程要看到，还需要一次跨线程同步。
- **F4（16 B 是上限）**：`cp-size` 只能是 4/8/16，所以「一条 cp.async 搬 16 B」不是调优选择，是硬上限；本梯用 `float4`（8 个 half = 16 B）正好顶格。

#### 3.3.2 C 层:`__pipeline_*` 的等价定义,以及它给出的更好用的形式

CUDA C++ Programming Guide §10.28.4 "Pipeline Primitives Interface" 给的定义与 PTX 等价，但写法更适合做归纳证明：

- §10.28.4.1 `__pipeline_memcpy_async(dst_shared, src_global, size_and_align, zfill=0)`:"size_and_align must be 4, 8, or 16." "size_and_align must be the alignment of dst_shared and src_global." 还有一条竞态条款："It is a race condition for any thread to modify the source memory or observe the destination memory prior to waiting for the memcpy_async operation to complete."
- §10.28.4.2 `__pipeline_commit()`:"Commit submitted memcpy_async to the pipeline as the current batch."
- §10.28.4.3 `__pipeline_wait_prior(N)`:"**Let {0, 1, 2, ..., L} be the sequence of indices associated with invocations of `__pipeline_commit()` by a given thread. Wait for completion of batches at least up to and including L-N.**"

**这个 L−N 形式是本节的关键工具**。「等到最新 N 组之外的都完成」是一句需要在脑子里绕一圈的话；「等到第 L−N 批完成」是一个可以直接代数字的式子。下面的归纳证明就用它写。

#### 3.3.3 v3 稳态的归纳证明(用 L−N 形式重写)

记第 t 个 K 块对应的批为 $G(t)$。v3 的 commit 序列（每线程）由两处产生：序幕一次（gemm_v3.cu：60-61），循环体内每轮至多一次（gemm_v3.cu：64-66）。`load_tile_async` 每次调用**恰好** commit 一组（gemm_v3.cu：43），这是整套计数的前提。

- **序幕**：commit $G(0)$。此时 L = 0。
- **轮 t = 0**：先 commit $G(1)$ ⇒ L = 1。`__pipeline_wait_prior(1)` ⇒ 等到第 L−N = 1−1 = **0** 批完成，即 $G(0)$。而 $G(0)$ 装的正是 buf[0] = 当前要消费的那一面。$G(1)$ 仍在途，与本轮计算重叠。计数成立。
- **归纳假设**：进入轮 t 时已 commit 的批数为 t+1（索引 0..t），且 buf[t mod 2] 由 $G(t)$ 填充。
- **轮 t(t+1 < K/BK)**：commit $G(t{+}1)$ ⇒ L = t+1。`wait_prior(1)` ⇒ 等到第（t+1）−1 = **t** 批，即 $G(t)$，正是本轮要读的 buf[t mod 2]。$G(t{+}1)$ 写的是 buf[(t+1) mod 2]，与本轮读的那一面不同，所以「在途写」与「本轮读」不冲突。计数成立。
- **末轮**：不 commit（gemm_v3.cu：64 的 `if (k0 + BK < K)`）⇒ L 仍为 t。`wait_prior(0)` ⇒ 等到第 t−0 = **t** 批，即 $G(t)$。若这里仍写 1，等到的是第 t−1 批，而 $G(t)$ 可能还没搬完——**读到未定义数据，且不崩**。计数成立。

**归纳里用到的三个前提，缺一不可**：(a) 每轮恰好 commit 一组（F2：多一次空 commit 就会让 L 多加 1，所有 wait 集体指错对象）；(b) 全线程执行同一条 commit 路径（F1：`load_tile_async` 内的循环 `for (t = tid; t < …; t += nthr)` 无分支，即使某个线程一条也不搬，它仍然会执行 `__pipeline_commit()` 造一个空组——**空组和满组在计数上等价，所以线程之间仍然对齐**，这正是 F2 在这里从坑变成保障的地方）；(c) 末轮的 wait 参数与「是否 commit」用同一个条件表达式（gemm_v3.cu：64 与：67 都写 `k0 + BK < K`），两处必须一致。

#### 3.3.4 为什么 wait 之后还必须 `__syncthreads`:两条独立的理由

`__pipeline_wait_prior` 之后紧跟一个 `__syncthreads()`(gemm_v3.cu：68-69)。这不是保险起见，而是两条独立条款各自要求的：

1. **可见性**(F3)：PTX §9.7.9.26.3.3 明写拷贝结果只对**发起线程**可见。tile 由全 block 分片搬运（`for (t = tid; …; t += nthr)`），消费按 warp 分块——**你要读的那一段多半不是你搬的**，所以必须有一次跨线程的可见性事件。
2. **完成性**(F1)：组计数是每线程的。你自己的 `wait_prior` 只保证**你的**批完成，不保证别的线程的批完成。哪怕可见性没问题，别人的数据也可能还没到。

CUDA PG §10.28.2 给了这条规则的一个例外："If the compute operation only reads shared memory written to by other threads in the same warp as the current thread， `__syncwarp()` suffices." 本梯不满足这个例外（装载按线性 tid 分片跨越全 block），所以只能用 `__syncthreads()`。

#### 3.3.5 与 CUTLASS 的参数对照:wait 参数的通式

CUTLASS 的 multistage 主循环给出了同一套语义的工业写法（`include/cutlass/gemm/threadblock/mma_multistage.h`）：

- 序幕："GEMM prologue. Bootstrap the global->shared memory pipeline by fetching the global fragments needed by the first kStages-1 threadblock mainloop iterations"(:360-362)，循环 `for (int stage = 0; stage < Base::kStages - 1; ++stage, --gemm_k_iterations)`(:370)。
- 主循环里的等待：注释 "Wait until we have at least one committed global fetch stage. (#uncommitted = Base::kStages - 1 - #committed)"(:488)后接 `cutlass::arch::cp_async_wait<Base::kStages - 2>();`(:489)。
- 收尾：`cutlass::arch::cp_async_wait<0>();`(:665)。

把两边对齐可以得到一个通式（**本讲义折算**）：设预取深度 D =「在 wait 之前允许同时在途的批数」，则 **wait 参数 N = D − 1**，收尾一律 N = 0。CUTLASS 的 D = kStages − 1（序幕先发 kStages−1 批），故 N = kStages − 2；本梯 v3/v4 的 D = 2（序幕 1 批 + 循环内 1 批），故 N = 1。**两边是同一个式子的两个取值**，不是两套机制。

差别在**代价**：CUTLASS 用 kStages 块 smem 缓冲把「正在被计算消费的那一块」也让出来，于是 DMA 有 kStages−2 段计算的窗口；本梯只有 2 块缓冲，DMA 窗口只有一段计算。**这就是 §8 说的「本梯只有 2 级流水」的精确含义**——不是「少写了几行」，是 smem 预算换来的窗口长度。v4 的 32 KB smem 若要升到 kStages=4，需要 64 KB，按 §3.5 的驻留计算会把每 SM block 数从 2 压到 1，本梯没有做这个对照臂（账面推断）。

#### 3.3.6 +6.7% 的上限:重叠能省的就是装载占比

实测 v2→v3 只有 +6.7%(89.5→95.5，EXP-K02 §5)。这个数字有一个先验上界：**重叠最多只能省掉「装载在总时间里的占比」**。v2 已经吃到峰值的 54.2%(§3.2.4)，说明装载段本来就没占多少；若装载占 x，理想重叠后时间变成（1−x），加速 = 1/(1−x)，实测 1.067 反推 x ≈ 6.3%（**本讲义折算，理想化上界模型**）。这个 6.3% 与「v2 每 K 块搬 8 KB、算 16 个 mma」的量级是自洽的。

这一级的教学价值在于对照 v4——**复用（+39%）比重叠（+6.7%）值钱**，顺序不能反。若先做 v3 再做 v4，拿到的是 6.7% 然后 39%；若先做 v4 再做 v3，拿到的是 39% 然后一个更小的重叠收益（因为大 tile 之后装载占比更低）。**两条路径的终点相同，但只有一条路径能让你在中途就知道该往哪走。**

### 3.4 128×128 大 tile 的复用账:+39% 值钱在哪

#### 3.4.1 面积/周长定律的推导

tile 复用的核心公式：一个 K 块内，block 从 smem 装载（BM+BN）·BK 个元素，喂 2·BM·BN·BK 次浮点运算，所以

$$\text{每装载一个元素支撑的 FLOP} = \frac{2\,BM \cdot BN \cdot BK}{(BM+BN)\cdot BK} = \frac{2\,BM \cdot BN}{BM+BN}$$

**逐步说明每一步为什么合法**：

- 分子 2·BM·BN·BK：tile 内的 GEMM 是一个 BM×BN×BK 的小 GEMM，MAC 数 = BM·BN·BK，每 MAC 记 2 FLOP（§3.1.2 的口径）。
- 分母（BM+BN）·BK：A 的子块是 BM×BK，B 的子块是 BK×BN，合计（BM+BN）·BK 个元素。**这里默认了「每个元素只从 smem 读一次」**——实际不成立（fragment 装载会重复读），所以这个式子给的是**复用的理论上界**，不是实测值。§3.4.3 会把 fragment 层的真实读数补上。
- **BK 在分子分母里同时出现并被约掉**，这是 §3.4.4 要单独说的结论。

代入：

- v2/v3(64×64):2·64·64/128 = **64 FLOP/元素**；
- v4(128×128):2·128·128/256 = **128 FLOP/元素**，翻倍。

#### 3.4.2 与 NVIDIA 官方公式的关系

NVIDIA 的 GEMM 指南给整个 GEMM 的算术强度公式是 `M·N·K / (M·K + N·K + M·N)`(Matrix Multiplication Background User's Guide §2 Math And Memory Bounds)。把 M→BM、N→BN、K→BK 代入，得

$$\frac{BM \cdot BN \cdot BK}{BM\cdot BK + BN \cdot BK + BM \cdot BN}$$

当 BK 与 BM、BN 同量级时，分母第三项不可忽略；当 **BK ≪ BM， BN** 时（本梯 BK=32 对 BM=BN=128），第三项 BM·BN = 16384 反而是最大的一项——但那一项对应的是**输出 C 的读写**，而 C 在本梯里全程驻寄存器、只在 kernel 结束时写一次，**不在每个 K 块里反复动**。所以本梯的每 K 块复用式子里没有它，官方式子里有它：**两个式子回答的不是同一个问题**（官方式子问「整个 GEMM 的 DRAM 强度」，本节式子问「每个 K 块的 smem 强度」）。把两者混用是一个常见错误。

#### 3.4.3 三层 tile 的复用账:与 CUTLASS 的分层对齐

CUTLASS 官方文档把 GEMM 拆成三层 tile(Efficient GEMM in CUDA，§"Hierarchical Structure")：threadblock tile 从全局内存取数、warp tile 从 shared memory 取数进寄存器、thread/instruction tile 在寄存器上做乘加，并明确 "to maximize data reuse within the warp， a large warp-level GEMM tile should be chosen"。本梯与它逐层对应：

| 层 | CUTLASS 术语 | 本梯 v2/v3 | 本梯 v4 | 复用度量 |
|---|---|---|---|---|
| 线程块 | ThreadblockShape | 64×64×32 | 128×128×32 | 每 smem 元素支撑 64 / 128 FLOP(§3.4.1) |
| warp | WarpShape | 32×32(4 warp) | 64×32(8 warp) | 每 kk 步 4 load 喂 4 mma / 6 load 喂 8 mma |
| 指令 | InstructionShape | 16×16×16(wmma) | 16×16×16(wmma) | 一条 mma 8192 FLOP |

**warp 层的账**值得单独算：v2 每个 kk 步取 af[2] + bf[2] = 4 个 fragment，做 2×2 = 4 次 mma，**平均每个 fragment 参与 1 次 mma 之外还被复用 1 次**；v4 取 af[4] + bf[2] = 6 个 fragment，做 4×2 = 8 次 mma，af[i] 各参与 2 次、bf[j] 各参与 4 次，**6 次装载喂 8 次 mma**。以「每次 smem 装载支撑的 mma 数」计：v2 = 4/4 = 1.0，v4 = 8/6 ≈ 1.33，提升 33%。这一层的提升与线程块层的翻倍叠加，共同构成 +39%。

全局流量同理：总全局读 = M·N·K·2B·(1/BM + 1/BN)，v2/v3 ≈ 4.3 GB，v4 ≈ 2.15 GB，减半（大部分由 L2 供给——A、B 各 32 MB，合计 64 MB 在 4090 的 72 MB L2 之内，所以 DRAM 强制流量不变，**账面推断**）。

两层复用叠加，实测 v3→v4 +39%(95.5→133.1，EXP-K02 §5)，是版本梯对 cuBLAS 差距的主要收口。

#### 3.4.4 为什么 BK 停在 32:三个约束的交点

- **理论侧**：§3.4.1 已证算术强度与 BK 无关（BK 被约掉）。所以加大 BK **不带来任何复用收益**——这一条排除了「BK 越大越好」。
- **硬件侧**：BK 翻倍到 64，v4 的 smem 从 2×(128×32 + 32×128)×2B = 32 KB 冲到 64 KB。按 §3.5 的驻留计算，64 KB 使每 SM 只能放 1 个 block（100 KB 的 shared memory 上限，CUDA PG §20.2 Table 27），block 数从 2 掉到 1，**纯亏**（gemm_v4.cu：23-24 面试点②）。
- **同步侧**：BK 减半到 16，每个 K 块的计算量减半而 barrier 次数翻倍；K=4096 时 `__syncthreads` 从 128 次涨到 256 次（每轮 2 次，gemm_v3.cu：68 与：86）。
- **另一条硬约束**：wmma 的 k 维是 16，所以 BK 必须是 16 的倍数；BK=32 = 2 步 wmma-k，是「大于 16 且能让内层展开成 2 步」的最小值（gemm_v2.cu：24-25）。

**结论：BK=32 由「理论上无收益 + smem 预算 + 同步频率 + wmma 粒度」四条共同钉死，不是扫出来的。** 本梯没有做 BK 的实测扫描，这一条属于**账面推断**；若要坐实，需要 BK∈{16,32,64} 三个对照臂。

至于为什么不再放大 BM/BN：每 warp 的 accumulator fragment 数量随 tile 面积增长，寄存器先爆（见 §3.5 的账），这是「工位加大」类比折断的地方（推断级，本梯未做 256 级别的对照臂）。

### 3.5 occupancy 33% 全梯最低却最快:驻留计算、发射规则与 ILP

#### 3.5.1 资源画像(ptxas -v 实测)

数据来自 `gemm/project-proof/data/ptxas_resource_usage.txt`(EXP-K02 §5)，该文件逐字记录了三行：

```text
ptxas info    : Function properties for _Z14gemm_v2_kernelPK6__halfS1_PS_iii
ptxas info    : Used 54 registers, used 1 barriers, 8192 bytes smem, 388 bytes cmem[0]
ptxas info    : Function properties for _Z14gemm_v3_kernelPK6__halfS1_PS_iii
ptxas info    : Used 61 registers, used 1 barriers, 16384 bytes smem, 388 bytes cmem[0]
ptxas info    : Function properties for _Z14gemm_v4_kernelPK6__halfS1_PS_iii
ptxas info    : Used 92 registers, used 1 barriers, 32768 bytes smem, 388 bytes cmem[0], 8 bytes cmem[2]
```

| 版本 | regs/thr | smem | 线程/block | 每 SM 驻留 block（限制因子） | 理论 occupancy |
|---|---|---|---|---|---|
| v2 | 54 | 8KB | 128 | 9(regs) | 75% |
| v3 | 61 | 16KB | 128 | 6(smem) | 50% |
| v4 | 92 | 32KB | 256 | 2(regs) | 33% |

#### 3.5.2 驻留计算逐步做一遍(以 v4 为例)

三个上限各算一遍，取最小：

1. **寄存器**：每 block 用寄存器 = 92 × 256 = 23552。每 SM 寄存器文件 65536 个（§1.2）⇒ ⌊65536 / 23552⌋ = **2 block**。
   - 一个更精细的版本：寄存器按 warp 为单位分配，分配粒度为 256 个寄存器/warp（即每线程 8 个的整数倍）。取整后每 warp = ⌈92×32/256⌉×256 = 3072，每 block 8 warp = 24576，⌊65536/24576⌋ 仍是 **2**。**两种算法结论一致，所以本梯的 2 这个数是稳的**；分配粒度这条规则本讲义未在当前版官方文档中找到正面表述，标注**未核实**，仅作为「结论对粒度不敏感」的稳健性检查。
2. **shared memory**:32 KB/block，每 SM 上限 100 KB ⇒ ⌊100/32⌋ = 3 block。
3. **常驻 warp**：每 block 8 warp，每 SM 上限 48 warp ⇒ 6 block；常驻 block 上限 24，不构成约束。

取最小 ⇒ **2 block/SM，限制因子是寄存器**。2 × 8 = 16 warp，除以 48 = **理论 occupancy 33%**，全梯最低。

寄存器的大头有账可查：8 个 fp32 accumulator fragment × 每 lane 8 个元素 = 64 个寄存器/线程，占 92 的七成——**「大 tile」的代价直接写在寄存器文件里**（每 lane 8 个元素的来历见 §3.2.2 性质 1:16×16 = 256 个元素分给 32 lane）。

同样的算法用在 v2 上：54 × 128 = 6912，⌊65536/6912⌋ = 9；smem 8 KB ⇒ ⌊100/8⌋ = 12；warp 4/block ⇒ 12 block。取最小 9，限制因子是寄存器，9 × 4 = 36 warp / 48 = 75%，与表一致。v3:61 × 128 = 7808 ⇒ 8 block；smem 16 KB ⇒ ⌊100/16⌋ = 6 block（**这里 smem 反超成为限制因子**）；6 × 4 = 24 / 48 = 50%，同样与表一致。

#### 3.5.3 发射规则决定了 occupancy 到底买到什么

CUDA PG §20.7.1 把 SM 的发射规则写成一句："An SM statically distributes its warps among its schedulers. Then, at every instruction issue time, each scheduler issues one instruction for one of its assigned warps that is ready to execute, if any." Ada 白皮书把硬件侧对应上：每个 SM 分四个 partition，每个 partition 有 **一个 warp scheduler 与一个 dispatch unit**。

把这两句读严：

- 每 SM 每周期最多发 4 条指令，**而且是从 4 个互不相通的池子里各发一条**。warp 静态分配给调度器，所以「16 个 warp」实际是「每个调度器 4 个 warp」。
- 一个调度器在某周期能不能发指令，取决于它那 4 个 warp 里**有没有一个 ready**。occupancy 高 = 每个调度器手上的 warp 多 = 某一个卡住时更容易找到替补。**这就是 occupancy 买的全部东西：替补的数量。**
- 但「有指令可发」还有另一条来源：**同一个 warp 内部有多条互不依赖的指令**(ILP)。一个 warp 若能背靠背发出 8 条无依赖的 mma，它自己就把调度器喂饱了，不需要替补。

#### 3.5.4 v4 的 ILP 账

v4 的 8 个 accumulator fragment 互不依赖（`acc[i][j]` 两两独立），8 条 `mma_sync` 可以背靠背发射，前一条没算完不妨碍下一条进管线；加上 af/bf 的多次复用摊薄了 smem 读，Tensor Core 的供数与发射两头都被 fragment 级 ILP 喂饱——它不需要靠换 warp 过日子。

给一个数量级的核对（**本讲义折算**）：v4 达 133.1 TFLOPS，除以 128 SM 与 2.52 GHz = 每 SM 每周期 413 FLOP = 206 MAC，而 §1.2 折算的每 SM 每周期 Tensor MAC 上限是 256。**即 Tensor Core 的占用率约 81%**（与 133.1/165.2 = 80.6% 一致，自洽）。以每 SM 16 个 warp 计，平均每个 warp 每 5 个周期要交出一条 mma；考虑到每条 wmma 16×16×16 展开为 2 条 m16n8k16（§3.1.4 的折算），每 SM 每周期需要发出约 0.4 条 mma 指令，而 4 个调度器每周期能发 4 条——**发射带宽远远不是瓶颈，Tensor Core 的执行吞吐才是**。这正是「occupancy 低但不缺」的定量表述。

结论：**occupancy 是手段，不是目标**(EXP-K02 §6)；判断标准是「延迟是否被遮蔽」，不是「线程是否足够多」。反例在第二篇讲义：FA2 被 90.75 KB smem 钉死在 1 block/SM 时，加 warp(v3，+33%)就是唯一的放大招——同一个指标，两种读法，取决于瓶颈在哪。

### 3.6 魔法数总账:每个常数由谁决定

按公理 C，把本梯出现的每一个常数归到「理论上界 / 硬件约束 / 实测扫描」三类之一。**没有一个常数是「试出来的」但没写清理由**；凡本梯未做扫描的，如实标为账面推断。

| 常数 | 值 | 决定它的是 | 依据 |
|---|---|---|---|
| v0 block 形状 | 16×16 = 256 线程 | 硬件约束 | 一线程一输出；256 是常见的调度友好粒度，本梯未扫（**账面推断**） |
| v1 tile T | 32 | 硬件约束 | 32×32 = 1024 = 每 block 线程上限（CUDA PG §20.2 Table 27 "Maximum number of threads per block"）；再大就无法一线程一输出（gemm_v1.cu：17） |
| v2/v3 BM=BN | 64 | 理论上界 | 4 warp × 每 warp 32×32；取 64 是为了与 v1 的 32 拉开一档而不立刻爆寄存器 |
| v4 BM=BN | 128 | 理论上界 + 硬件约束 | 复用公式给收益（§3.4.1），寄存器给上限（§3.5.2：再翻倍 accumulator 就到 128 reg/thr 量级） |
| BK | 32 | 四条约束的交点 | §3.4.4：理论无收益 + smem 预算 + 同步频率 + wmma k=16 的倍数 |
| 装载粒度 | 16 B(float4 / cp.async) | 硬件约束 | PTX §9.7.9.26.3.1「cp-size can only be 4, 8 and 16」;CUDA PG §10.28.4.1 同 |
| wmma instruction shape | 16×16×16 | 硬件约束 | CUDA PG §10.24.6 Element Types and Matrix Sizes 允许的 shape 组合之一；fp16×fp16→fp32 的三种 shape 中最方阵的一个 |
| `wait_prior` 参数 | 1（末轮 0） | 理论 | §3.3.5 通式 N = D − 1，D = 2 |
| accumulator 精度 | fp32 | 理论上界 | §7 Q6 的舍入分析；换 fp16 accumulator 峰值翻倍（330.3 TFLOPS）但 4096 长点积的误差不可接受 |
| bench iters | 50(v0/v1 取 max(3, 5)) | 实测扫描 | v1 的轮间 std ≈ 0(21.114±0.047),5 iters 统计已够（EXP-K02 §7）;main.cu:105-107 |
| 抽样步长 | 997 / 97 | 理论 | 素数，避开 2 的幂结构的周期性采偏（main.cu：63-65） |
| 正确性阈值 | 2e-2 | 理论上界 | fp16 存储的合理界；实测 max_rel_err = 7.58e-04，余量 26 倍（EXP-K02 §5） |

### 3.7 论文/文档怎么说 vs 本项目实测

这一节把「各方声称什么」与「我们在 RTX 4090 上测到什么」并排放，并解释差异来源。**只要差异存在就写出来，不粉饰。**

#### 3.7.1 峰值算力:白皮书 165.2 vs 本梯 133.1

- **文档说**：RTX 4090 的 FP16 Tensor with FP32 Accumulate 峰值为 165.2 TFLOPS（Ada 白皮书 Appendix A）；脚注 1 写明 "Peak rates are based on GPU Boost Clock."
- **本仓实测**：v4 = 133.1±0.97 TFLOPS（80.6% 峰值）,cuBLAS = 155.4±0.62（94.1% 峰值）,3 轮（EXP-K02 §5）。
- **差异来源**：(a) 峰值按 boost 时钟折算，而 4096³ 连续跑 50 iters 会触发功耗/温度降频，实际时钟低于 2520 MHz——本容器无法读取运行时时钟，**未核实**；(b) 峰值不含任何装载、同步与 epilogue；(c) 本梯的 wmma 布局无法做 smem swizzle(§8)，cuBLAS 可以。**cuBLAS 拿到 94.1% 这个事实说明（a） 的降频影响至多几个百分点**，否则谁也到不了 94%——这是一个用对照物给出的、不需要 profiler 的间接判据。

#### 3.7.2 tile 尺寸:NVIDIA 说 256×128 最有效率,本梯用 128×128

- **文档说**：cuBLAS 的典型 tile 尺寸里 "256x128 and 128x256 (most efficient)" 效率最高，64×64 之类的小 tile 提供更多并行度但效率明显更低（Matrix Multiplication Background User's Guide §2.3 Typical Tile Dimensions In cuBLAS）。
- **本梯实测**：64×64(v2/v3)→ 128×128(v4)确实 +39%，方向与文档一致；但本梯**没有**做到 256×128。
- **差异来源**：寄存器预算（§3.5.2）。256×128 在本梯的 warp 划分下会把每 warp 的 accumulator fragment 数从 8 提到 16，寄存器直接翻到 156 以上，驻留 block 掉到 1 甚至 0。cuBLAS 能用 256×128，是因为它走 mma + ldmatrix，accumulator 的寄存器占用与数据布局都由它自己控制，还可以用寄存器双缓冲把装载摊开（CUTLASS Efficient GEMM 的 "Pipelining" 节明写在 "warp-scoped register fragments" 上也做双缓冲："one fragment is passed to CUDA and TensorCores during the current matrix computation， while the other is used to receive shared memory fetch returns"）。**本梯未做 256 级对照臂，这条是账面推断。**

#### 3.7.3 流水级数:CUTLASS 3-5 级 vs 本梯 2 级

- **文档/源码说**：CUTLASS multistage 的序幕预取 kStages−1 批（mma_multistage.h：370），等待参数 kStages−2(：489)，典型 kStages 为 3 或更多。
- **本梯**：D = 2,N = 1(§3.3.5)。
- **差异来源**：smem 预算。本梯 v4 已用 32 KB，升到 kStages=4 需 64 KB，驻留 block 从 2 掉到 1。**这是一个可以做但本梯没做的对照臂**；做了之后是涨是跌不确定——多级流水买到更长的 DMA 窗口，但同时把 TLP 砍半，两者方向相反。**如实记为开放问题。**

#### 3.7.4 roofline 的适用性:论文的前提 vs 本仓在别的算子上的翻车

- **论文说**：operational intensity 的分母是 DRAM traffic(Williams et al. 2009)。
- **本仓实测（另一个算子）**：reduce 在 N = 1<<24(67.1 MB)上测得的等效带宽是 HBM 理论峰值的 1.8 至 3.4 倍（EXP-K04 §4.1）。**等效带宽超过理论峰值，只能说明数据没落 DRAM**——那个尺寸小于 4090 的 72 MB L2，测的是 L2 带宽。
- **对本篇的意义**：GEMM 这边没有踩这个坑，因为 A+B+C = 100.7 MB > 72 MB，而且 I ≫ I\* 让判定对分母的精确值不敏感（1365 与 164 差 8 倍，分母就算错一倍结论也不变）。**但「判定对分母不敏感」这件事本身要说出来，而不是碰巧没错**。第二篇讲义 §8.3 的附课会把 reduce 那次的完整教训讲透。

#### 3.7.5 wmma 契约:文档承诺 vs 本仓依赖

已在 §3.2.2 展开。一句话版本：**本仓有一处依赖（两个同 shape accumulator fragment 逐元素相加）超出了文档明文授权的范围**，虽然实践上成立且 gate 全过，但按证据分级只能算「实践依赖」，不能算「文档保证」。**把这类依赖显式列出来，是代码能被别人接手的前提。**

### 3.8 存储层级的容量与带宽台阶:本梯的数据各落在哪一层

前面几节反复用到「装得进 L2」「由 L1 供给」这类判断，这里把整张层级表摊开。**容量列全部有官方出处；带宽列只有 DRAM 一项是官方数字，其余都是折算或实测下界——这个不对称本身就是重要信息**：NVIDIA 公布容量，但基本不公布片上带宽，所以任何「L2 带宽是多少」的说法都要问清楚是怎么来的。

| 层级 | 每 SM 容量 | 全卡容量 | 带宽（每 SM） | 带宽（全卡） | 出处 / 证据等级 |
|---|---|---|---|---|---|
| 寄存器文件 | 256 KB(64 K × 4 B) | 32768 KB = 32 MB | ≤ 1536 B/cycle | ≤ 约 495 TB/s | 容量：CUDA PG §20.2 Table 27 与白皮书 Appendix A "Register File Size"；带宽：**本讲义折算上界**（4 条指令/周期 × 3 个源操作数 × 32 lane × 4 B） |
| shared memory / L1（统一） | 128 KB（smem 至多 100 KB） | 16384 KB = 16 MB | 128 B/cycle | 约 41 TB/s | 容量：CUDA PG §20.7.1、§20.2 Table 27、白皮书 Appendix A "L1 Data Cache/Shared Memory"；带宽：**本讲义折算**(32 bank × 4 B/cycle,Best Practices §10.2.3.1)，把 smem 的 bank 带宽当作 L1 的上界属**账面推断** |
| L2 |— | 73728 KB = 72 MiB |— | ≥ 3.4 TB/s（实测下界） | 容量：白皮书 Appendix A "L2 Cache Size"；带宽：**本仓实测折算**，CUB 在 67.1 MB 常驻数组上 0.019828 ms 读完一遍 = 3384 GB/s(EXP-K04 §4.1)，这是「某个访问模式下达成的」，不是峰值 |
| DRAM(GDDR6X) |— | 24 GB |— | 1008 GB/s | 白皮书 Appendix A "Memory Bandwidth"；本仓设备属性反推 1008.1 GB/s(EXP-K04 §3) |

**相邻两级之间的台阶大约是一个数量级**：41 TB/s → 3.4 TB/s → 1.008 TB/s。注意 L2 到 DRAM 只有约 3.4 倍，而 L1 到 L2 有约 12 倍——**所以「把数据从 L2 挪到 L1/smem」的收益通常大于「把数据从 DRAM 挪到 L2」**，这也解释了为什么 GEMM 的主战场是 shared memory 而不是 L2 命中率。

把本梯的每一份数据对号入座（4096³，fp16）：

| 数据 | 大小 | 常驻在哪 | 依据 |
|---|---|---|---|
| C 的 accumulator（v4 每 block） | 8 fragment × 8 元素 × 4 B × 256 thr = 64 KB/block | 寄存器 | §3.5.2 的寄存器账 |
| A/B 的当前 K 块（v4，双缓冲） | 32 KB/block | shared memory | ptxas 报告 32768 bytes smem |
| A 全矩阵 | 32 MB | L2（与 B 合计 64 MB < 72 MB） | §3.4.3 |
| B 全矩阵 | 32 MB | 同上 | 同上 |
| C 全矩阵 | 32 MB | DRAM（只写一次，无复用） | 每个输出只被一个 block 写 |

**A + B = 64 MB 落在 72 MB L2 之内，是本梯很多结论成立的隐含前提**。它意味着：v1 的 8.6 GB 全局读里绝大部分是 L2 命中，DRAM 强制流量只有约 100 MB；v4 把全局读从 4.3 GB 砍到 2.15 GB，砍掉的也主要是 L2 流量而非 DRAM 流量。**换一张 L2 小的卡，这些结论要重算**(§7 Q16)。

**这张表也给出了 §5.2 那条自洽性检查的一般形式**：算出实现的等效带宽，看它落在哪一档。落在 1 TB/s 以下 → 可能真的在读 DRAM；落在几 TB/s → 数据在 L2；落在几十 TB/s → 数据在 L1/smem。本仓 reduce 的两区间实验就是这条判据的完整演示（EXP-K04 §4.1）：同一份代码、同一台机器，只把数组从 67.1 MB 换成 1.07 GB，等效带宽从「超理论峰值 2 至 3 倍」掉到「理论峰值的 93.9%」，版本之间的相对排序也跟着变。**性能数字的第一个属性不是大小，是「它测的是哪一层」。**

## 4 代码逐段走读(按执行顺序)

阅读约定：每段先说**角色**（它在版本梯里承担什么），再点**关键行**（哪一行是硬约束的落点），再给**硬件语义**（这一行为什么必须这么写，依据哪条文档），最后给**改错会怎样**（把正确性押在哪里）。

### 4.1 v0:正确性锚与性能分母(gemm/src/gemm_v0.cu:13-25)

```cuda
__global__ void gemm_v0_kernel(const half* A, const half* B, half* C,
                               int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;   // 末块越界线程直接退出:各输出互不相交,无部分和要合并
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        // B[k*N+col]:warp 内 col 连续 → 按行合并;A[row*K+k]:warp 内同 row 广播。
        // 问题不在合并度,在复用为零:同一 A/B 元素被不同 block 反复从
        // DRAM/L2 拉取——这笔带宽账引出 v1 的 smem tile。
        acc += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    C[row * N + col] = __float2half(acc);   // fp32 累加全程保精度,仅写回舍入一次
}
```

**角色**：版本梯的分母，一线程一输出、肯定对。

**关键行**：L17 的提前 return 之所以安全，是因为每个输出互不相交、没有部分和要跨线程合并（对比 FA2 v1 里越界线程必须活着陪跑 barrier）；L23 的访存模式其实已经合并（col 连续），v0 的病不在合并度而在零复用。

**硬件语义**：这个 kernel 的 block 是 `dim3 blk(16, 16)`(gemm_v0.cu：27)，即 256 线程 = 8 个 warp。CUDA 把 threadIdx 按 x 最快的顺序线性化，所以一个 warp 覆盖 `threadIdx.y` 的两个相邻值 × `threadIdx.x` 的 16 个值——**一个 warp 横跨 2 行 16 列**。于是 `B[k*N+col]` 在 warp 内只有 16 个连续 half = 32 B，恰好一个 sector；`A[row*K+k]` 只有 2 个地址。§3.1.3 的三种字节账就是从这里推出来的。这也解释了为什么 v0 的等效请求带宽能到 10 TB/s 量级：同一个 32 B sector 被 warp 里 32 个线程共享，L1 只需服务一次。

**改错会怎样**：去掉 guard，尾块线程越界读写；把 acc 换成 half 累加，4096 长点积的舍入误差会随 K 累积（大数吃小数），正确性 gate 未必再过（定量分析见 §7 Q6）。

### 4.2 v1:把复用显式化,并引入 barrier 纪律(gemm/src/gemm_v1.cu:24-35)

```cuda
    for (int k0 = 0; k0 < K; k0 += T) {
        // 装载:两条读 tx 连续 → 全局访存均按行合并
        As[threadIdx.y][threadIdx.x] = A[row * K + k0 + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * N + col];
        __syncthreads();   // 防 tile 未装满即有线程开读:内积要读整行/列,
                           // 大部分由别的线程装载(跨线程 RAW)
        #pragma unroll
        for (int k = 0; k < T; ++k)
            // As[ty][k]:warp 内同址 → smem 广播;Bs[k][tx]:tx 连续,行内顺序访问
            acc += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]);
        __syncthreads();   // 防快线程进入下一 k0 覆盖 As/Bs 时,慢线程仍在读旧 tile(WAR)
    }
```

**角色**：smem tile 的教科书形态，每元素装载一次供 32 线程复用，全局访存 /32。

**关键行**：两个 `__syncthreads` 各防一条竞态，方向相反：第一个防「读到没装完的」（跨线程 RAW），第二个防「快线程下一轮覆盖时慢线程还在读上一轮」(WAR)。

**硬件语义**：内层两条 smem 读走的是两条不同的 bank 路径。`As[threadIdx.y][k]`：同一个 warp 里 `threadIdx.y` 只取 2 个值（见 4.1 的线性化），`k` 是循环变量对全 warp 相同 ⇒ **一个 warp 只访问 2 个 smem 地址**，命中 Best Practices §10.2.3.1 描述的广播路径（"When multiple threads in a warp address the same shared memory location， resulting in a broadcast ... coalesced into a single multicast"）。`Bs[k][threadIdx.x]`：`threadIdx.x` 连续 16 个 half = 32 B，跨 8 个 bank，无冲突。**所以 v1 的 smem 访问是干净的**——这一点很重要：它排除了「v1 慢是因为 bank conflict」这个假设，把病因唯一地留给了 §3.1.4 的指令路径账。

**改错会怎样**：删第二个 barrier，错误只在个别调度时序下出现，是典型的「偶发错一位」难查 bug；这套 RAW/WAR 注释纪律贯穿后面所有版本。实测只 +25%（§3.1.4 的账），但它把「访存问题」与「算力问题」拆成两个独立变量，是控制变量设计，不是失败。

### 4.3 v2 装载段:float4 协同搬运(gemm/src/gemm_v2.cu:45-56)

```cuda
        for (int t = threadIdx.x; t < BM * BK / 8; t += blockDim.x) {
            int r = (t * 8) / BK, c = (t * 8) % BK;
            *reinterpret_cast<float4*>(&As[r][c]) =
                *reinterpret_cast<const float4*>(&A[(bm + r) * K + k0 + c]);
        }
        for (int t = threadIdx.x; t < BK * BN / 8; t += blockDim.x) {
            int r = (t * 8) / BN, c = (t * 8) % BN;
            *reinterpret_cast<float4*>(&Bs[r][c]) =
                *reinterpret_cast<const float4*>(&B[(k0 + r) * N + bn + c]);
        }
        __syncthreads();   // 防 As/Bs 未写全即被读:装载按线性 tid 分片、
                           // 消费按 warp 分块,线程集不重合(跨线程 RAW)
```

**角色**：把「谁算」与「谁搬」解耦——装载按线性 tid 分片（全 block 128 线程摊 512 条 float4），消费按 warp 分块（每 warp 一个 32×32 子区），中间靠 barrier 交接。

**关键行**：一条 float4 = 8 个 half = 16 B；`c` 恒为 8 的倍数且 `K % 32 == 0` 保证全局地址 16 B 对齐——float4 的硬性要求（gemm_v2.cu：42-44 注释）。

**硬件语义**：16 B 是这条链上反复出现的同一个数，但它在三个地方各有独立出处，不能只记一个：(a) `float4` 是 128-bit 向量类型，CUDA 要求其地址按 16 B 对齐，否则触发 misaligned address；(b) cp.async 的 `cp-size` 只能取 4/8/16(PTX §9.7.9.26.3.1)，v3 换成异步拷贝后 16 B 仍是上限；(c) wmma 的 `ldm` 要求换算成字节也是 16 B 的倍数（CUDA PG §10.24.1）。**三条约束恰好收敛到同一个数，这不是巧合——它们都源自 128-bit 这个访存事务的自然粒度。**

**分片方式为什么用 `t += blockDim.x` 而不是连续切块**：相邻 tid 拿到相邻的 float4，于是一个 warp 的 32 条 16 B 请求覆盖连续 512 B，合并度最好。若改成「每线程负责连续的 4 条」，一个 warp 的请求会散在 2 KB 上，合并度掉一档。

**改错会怎样**：把对齐前提破坏（如 K=4090），`reinterpret_cast` 的 128-bit 访问直接 misaligned address 错误；忘了 barrier，wmma 读到装了一半的 tile，结果错而不崩，是最恶劣的一类错。

### 4.4 v2 计算段:2×2 微内核,fragment 复用的起点(gemm/src/gemm_v2.cu:57-75)

```cuda
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            // [计算段] 2x2x2 微内核:af[i]/bf[j] 各 load 1 次、各参与 2 次
            // mma_sync——fragment 级数据复用的起点(v4 扩到 4x2)。
            // af/bf 均 row_major(A、B 本就按行存),leading dim = 所在 tile 行宽。
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(af[i], &As[wr * 32 + i * 16][kk], BK);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[j], &Bs[kk][wc * 32 + j * 16], BN);
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
```

**角色**：版本梯最大台阶（13.8 倍）的现场。

**关键行**：`load_matrix_sync` 的第三个参数是 leading dimension（源矩阵行宽），af 从 As 读用 BK、bf 从 Bs 读用 BN——写错不报错，只给你一片错位的数据；`mma_sync(acc, af, bf, acc)` 的第四参即累加输入，acc 全程驻寄存器不落 smem。

**硬件语义**：三条契约在这十几行里同时生效。(a) `#pragma unroll` 不只是性能提示——它保证 `load_matrix_sync` / `mma_sync` 不出现在任何全 warp 不一致的分支里，满足 CUDA PG §10.24 开头的条件（"these operations are allowed in conditional code only if the condition evaluates identically across the entire warp"）。(b) `ldm` = BK = 32 与 BN = 64 都是 8 的倍数，满足 §3.2.3 的 C2。(c) `acc[i][j]` 作为 `mma_sync` 的第三与第四个实参（C 与 D 同一个对象）是文档明确支持的写法："The in-place operation， C=A*B+C， is also supported."（CUDA PG §10.24.1，`mma_sync` 条）——这一条让 accumulator 全程不必往返内存，是 GEMM 侧「fragment 不透明也无所谓」的根本原因。

**为什么 `bf` 用 row_major 而 FA2 里用 col_major**：这里 B 本身就按行存，`matrix_b` 的 tile 是 k×n，行主序即 `Bs[kk][...]` 的自然布局。FA2 需要的是 K 的转置，那边靠把同一块内存声明成 `col_major` 拿到转置视图（第二篇讲义 §4.5）——**同一个 API，靠模板参数换语义，零数据搬运**。

**改错会怎样**：leading dim 传错是 wmma 新手第一大坑（结果全错但不崩）；把 acc 声明挪进 kk 循环内，每次清零，等于只算最后一个 K 块。

### 4.5 v3:异步装载器与调度骨架(gemm/src/gemm_v3.cu:31-44、59-69)

```cuda
__device__ __forceinline__ void load_tile_async(
    half (*As)[BK], half (*Bs)[BN],
    const half* A, const half* B, int M, int N, int K,
    int bm, int bn, int k0, int tid, int nthr) {
    for (int t = tid; t < BM * BK / 8; t += nthr) {
        int r = (t * 8) / BK, c = (t * 8) % BK;
        __pipeline_memcpy_async(&As[r][c], &A[(bm + r) * K + k0 + c], 16);
    }
    for (int t = tid; t < BK * BN / 8; t += nthr) {
        int r = (t * 8) / BN, c = (t * 8) % BN;
        __pipeline_memcpy_async(&Bs[r][c], &B[(k0 + r) * N + bn + c], 16);
    }
    __pipeline_commit();   // 本 tile 封组:调用一次 = 恰好一组,组节奏见文件头
}
```

```cuda
    // 序幕:先发第 0 组,循环内的 wait_prior 才有对象;首轮无重叠(冷启动税只付一次)
    load_tile_async(As[0], Bs[0], A, B, M, N, K, bm, bn, 0,
                    threadIdx.x, blockDim.x);
    int p = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K)                       // 末轮不预取:越界读 + 多出一组破坏 wait 计数
            load_tile_async(As[p ^ 1], Bs[p ^ 1], A, B, M, N, K,
                            bm, bn, k0 + BK, threadIdx.x, blockDim.x);
        __pipeline_wait_prior(k0 + BK < K ? 1 : 0);   // 留 1 组(刚发的预取)在途,只等当前块;末轮清空
        __syncthreads();   // cp.async 的完成只对发起线程可见,而 tile 由全 block
                           // 分片搬运:barrier 后任何 warp 才能读到别的线程搬的段(跨线程 RAW)
```

**角色**：§3.3 推导的代码形态。

**关键行**：`load_tile_async` 每次调用恰好 commit 一组——这个不变量是整套计数正确的前提；`wait_prior` 之后还要 `__syncthreads`，因为 cp.async 的完成只对发起线程可见，而 tile 是全 block 分片搬的，别的 warp 要靠 barrier 才能看到你搬的那段。

**硬件语义**：三处细节值得对着文档逐条读。(a) `__pipeline_memcpy_async(..., 16)` 的第三参在文档里叫 `size_and_align`，并规定 "size_and_align must be the alignment of dst_shared and src_global"(CUDA PG §10.28.4.1)——**它同时是尺寸和对齐要求**，传 16 就等于声明两端都 16 B 对齐，不满足是未定义行为。(b) `__pipeline_commit()` 放在函数末尾而不是循环内，保证「一次调用 = 一个批」；即使某个线程因为 `t < BM*BK/8` 的边界一条也没发，它仍然会造一个空批（PTX §9.7.9.26.3.2："If there are no uncommitted cp.async instructions then cp.async.commit_group results in an empty cp.async-group"），**空批与满批在计数上等价，线程之间的批索引因此仍然对齐**。(c) `if (k0 + BK < K)` 是一个全 block 一致的条件（只依赖循环变量），所以它不会让不同线程的批序列分叉——**若这个条件依赖 tid，整套计数立刻失效**。

**改错会怎样**：把 1 写成 0，连刚发的预取组一起等，重叠归零、性能退回 v2（正确但慢，最隐蔽）；末轮照常预取，越界读加计数错乱（错误数据）；在装载器里多调一次 commit，所有 wait_prior 集体指错组。

### 4.6 v4 主循环:全部要素合流(gemm/src/gemm_v4.cu:66-91)

```cuda
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K)                    // 末轮不预取(越界 + 破坏组计数)
            load_tile_async4(As[p ^ 1], Bs[p ^ 1], A, B, N, K,
                             bm, bn, k0 + BK, threadIdx.x, blockDim.x);
        __pipeline_wait_prior(k0 + BK < K ? 1 : 0);   // 留预取组在途只等当前块;末轮清空(推导见 v3)
        __syncthreads();   // cp.async 完成仅发起线程可见 → barrier 后全 warp 才能读全 tile(跨线程 RAW)
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            // 4x2x2 微内核:6 次 load 喂 8 次 mma;af[i] 复用 2 次、bf[j] 复用 4 次
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                wmma::load_matrix_sync(af[i], &As[p][wr * 64 + i * 16][kk], BK);   // wr*64:每 warp 管 64 行
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[j], &Bs[p][kk][wc * 32 + j * 16], BN);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();   // 防下一轮 cp.async 覆盖 buf[p] 时慢 warp 仍在读(WAR,同 v3)
        p ^= 1;
    }
```

**角色**：133.1 TFLOPS 的主体。8 warp 按 2×4 排布，每 warp 输出 64×32 = 4×2 个 fragment 常驻寄存器整个 kernel 生命周期（gemm_v4.cu：10-14 小图）。

**关键行**：第二个 `__syncthreads` 防的竞态很微妙——双缓冲只隔离「计算 vs 在途预取」，不隔离「本轮读 buf[p] vs 下一轮 cp.async 写同一 buf[p]」（翻面之后 p 就成了预取目标），所以计算完还要一个 WAR barrier。

**硬件语义**：这个 WAR barrier 的必要性有一条文档依据可以直接引：CUDA PG §10.28.4.1 明写 "It is a race condition for any thread to modify the source memory or observe the destination memory prior to waiting for the memcpy_async operation to complete. Between submitting a memcpy_async operation and waiting for its completion， any of the following actions introduces a race condition： Loading from dst_shared； Storing to dst_shared or src_global； ..." 下一轮的 `load_tile_async4` 会把 buf[p] 作为 `dst_shared` 提交，而本轮的慢 warp 可能仍在 `Loading from dst_shared`——**逐字命中文档列出的第一条竞态**。所以这个 barrier 不是「保险」，是文档要求。

**ILP 的代码形态**：内层 8 次 `mma_sync` 的目标 `acc[i][j]` 两两不同，没有写后读依赖；`#pragma unroll` 把它们全部展平成 8 条独立指令。§3.5.4 的 ILP 账就写在这八行里——**要看一段 Tensor Core 代码有没有 ILP，就数它连续几条 mma 的 accumulator 是否互不相同。**

**改错会怎样**：删 WAR barrier，大多数时序下结果还对，压力大时偶发错——比一直错更难修；把 `p ^= 1` 忘掉，双缓冲退化为单缓冲且计算读的永远是同一面，结果稳定地错。

### 4.7 对照物:真 cuBLAS 与行主序技巧(gemm/src/gemm_cublas.cu:14-22)

```cuda
static cublasHandle_t handle = nullptr;
void gemm_cublas(const half* A, const half* B, half* C, int M, int N, int K) {
    if (!handle) cublasCreate(&handle);
    const float alpha = 1.f, beta = 0.f;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                 &alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, K,
                 &beta, C, CUDA_R_16F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
```

**角色**：一切「vs cuBLAS」数字的对照物，155.4±0.62 TFLOPS(EXP-K02 §5)。

**关键行**：cuBLAS 只认列主序，行主序的 C = A·B 在列主序视角下等价于 C^T = B^T·A^T，所以以（N， M， K） 调用并交换 A/B 指针，零转置零拷贝；handle 懒建后进程内复用——`cublasCreate` 含上下文与 workspace 初始化（百 ms 级），每次重建会把初始化开销计进被测时延，对照失真。

**为什么这个技巧是恒等而不是近似**：cuBLAS 文档开宗明义 "For maximum compatibility with existing Fortran environments， the cuBLAS library uses column-major storage， and 1-based indexing."(cuBLAS Library §1.1 Data Layout)。一块行主序的 M×N 内存，用列主序的眼睛去看，就是一个 N×M 的矩阵——这是同一块字节的两种读法，没有任何数据移动。于是 `C_row = A_row · B_row` 被读成 `C_col^T = ...`，而（A·B）^T = B^T·A^T，所以传（N， M， K） 并交换指针即可。**这里唯一需要小心的是 leading dimension**：传给 B 的 ld 是 N（行主序下 B 的行宽），传给 A 的 ld 是 K，传给 C 的 ld 是 N——都写的是「行主序的行宽」，因为在列主序视角下它们正是各自的列高。

**为什么强调「真」**：本仓 softmax 项目曾有对照物经源码核查系自写 kernel（cuBLAS 并无 softmax API），相关对比句整体撤销（EXP-K01《四 kernel 4090 重基准》§5）；现行 softmax 对照已换成同算子的官方实现 `cudnnSoftmaxForward`(EXP-K04 §4.2)。此后立的规矩是凡「vs cublas」先验调用点，本文件就是被验对象。

**关于 `CUBLAS_GEMM_DEFAULT_TENSOR_OP`**：cuBLAS 文档现在把这个枚举标为 deprecated，语义是 "Apply Heuristics to select the GEMM algorithm， while allowing use of reduced precision CUBLAS_COMPUTE_32F_FAST_16F kernels (for backward compatibility)"(cuBLAS Library §2.8.11 cublas<t>gemmEx())。**这一点对口径有实际影响**：它允许库在 FP32 计算类型下降到 FP16 累加的快路径。本梯的 compute type 显式写的是 `CUBLAS_COMPUTE_32F`，而正确性核对显示 cuBLAS 输出与各版本的 max_rel_err 同为 7.58e-04 量级（EXP-K02 §5），与「fp32 累加」的预期一致；但**本讲义没有直接证据证明 cuBLAS 在本次运行中选的一定是 fp32 累加 kernel**，标注**未核实**。若它选了 fp16 累加，155.4 这个对照值会偏高，本梯 85.6% 的相对位置会偏低——即这个不确定性的方向对本梯的结论是**保守**的。

**改错会怎样**：不交换 A/B 直接按行主序传参，数值全错；compute type 用 16F，对照口径与版本梯（fp32 累加）不再可比。

### 4.8 bench harness:口径是怎么被一行行钉死的(gemm/src/main.cu:96-124)

```cuda
    for (auto& v : vs) {
        v.fn(A, B, C, M, N, K); cudaDeviceSynchronize();       // 预热+出数
        cudaMemcpy(hout.data(), C, hout.size() * 2, cudaMemcpyDeviceToHost);
        float maxrel = 0;
        for (size_t i = 0; i < hout.size(); i += 97) {         // 素数步长抽样,理由同上
            float d = fabsf(__half2float(hout[i]) - __half2float(href[i]));
            maxrel = fmaxf(maxrel, d / ref_absmax);
        }
        bool pass = maxrel < 2e-2f;
        int it = (strcmp(v.name, "v0") == 0 || strcmp(v.name, "v1") == 0)
                     ? std::max(3, iters / 10) : iters;         // 慢版少跑:>20ms/iter,
                                                                // 5 次统计已够(EXP-K02 §7 v1 std≈0)
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int w = 0; w < 3; ++w) v.fn(A, B, C, M, N, K);     // 预热:驱走冷时钟/懒初始化
        cudaEventRecord(e0);
        for (int i = 0; i < it; ++i) v.fn(A, B, C, M, N, K);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= it;  // 单 event 对包住整段:免逐次 event 开销偏置
        if (strcmp(v.name, "v0") == 0) v0_ms = ms;
        double tf = flops / (ms / 1e3) / 1e12;
        printf("%-10s %9.4f ms  %7.1f TFLOPS  maxrel=%.2e  %s\n",
               v.name, ms, tf, maxrel, pass ? "PASS" : "FAIL");
        csv << v.name << "," << M << "," << N << "," << K << ","
            << std::fixed << std::setprecision(4) << ms << ","
            << std::setprecision(1) << tf << ","
            << std::setprecision(2) << (v0_ms / ms) << ","
            << std::scientific << maxrel << ","
            << (pass ? "true" : "false") << "\n";
    }
```

**角色**：整篇讲义所有数字的出口。**一个 kernel 写得再好，harness 有偏就全白搭**，所以它值得和 kernel 一样逐行读。

**五个设计决策，各防一类偏**：

1. **单 event 对包整段再除 iters**(`cudaEventRecord(e0)`… `e1`… `ms /= it`)：若改成逐次记录再平均，每次 `cudaEventRecord` 自身要往流里插一条命令，对 1 ms 级的 kernel 会引入系统性偏置。代价是拿不到逐次分布（只有均值），所以轮间方差靠「跑 3 个独立进程」来度量，而不是靠 iters 内的方差——**两种方差是两回事，不能互相替代**。
2. **3 次预热**：驱走冷时钟与懒初始化。`cublasCreate` 的百 ms 级初始化在第一次调用时发生，预热把它挡在计时外（gemm_cublas.cu：16 的懒建也是为此）。
3. **慢版少跑**（v0/v1 取 `max(3, iters/10)`）：v0/v1 单次 20 ms 以上，50 iters 要一秒多。少跑的合法性有数据支撑：v1 的轮间 std ≈ 0（21.114±0.047 ms，相对 0.22%），5 次统计已够（EXP-K02 §7）。**这是一个用实测结果反过来为实验设计辩护的例子**，不是拍脑袋。
4. **相对误差的分母用全局 absmax 而非逐元素**（main.cu：63-71 与本段的 `d / ref_absmax`）：近零元素上的逐元素相对误差会无意义地爆炸，一个 1e-8 的输出配 1e-8 的绝对误差就是 100% 相对误差。用全局 absmax 做分母等价于「按输出量级归一的绝对误差」，是这类稠密数值 kernel 的标准做法。
5. **抽样步长取素数 997 / 97**：输出是 4096×4096，任何 2 的幂步长都会周期性地只采到某些列（比如步长 4096 只采第 0 列）。素数步长与 2 的幂互素，遍历上是均匀的。

**还有两条写在别处但同样重要**：固定 `srand(42)`(main.cu：53)使所有版本所有轮吃同一输入，否则「版本差异」里混着「输入差异」；结果只写 `BENCH_OUT` 指定的新文件、首行写 provenance(main.cu：77-90)，历史数据永不覆盖。

**改错会怎样**：去掉预热，第一个被测版本会背上冷时钟与初始化的锅，版本梯的第一级凭空变慢；把 `ms /= it` 忘了，所有时延乘以 iters，TFLOPS 除以 iters——这类错会让整张表一起偏，反而不容易被「某个版本看起来不对」察觉。

## 5 实验数据怎么读

现行数字（`gemm/project-proof/data/derived_gemm4096_stability.csv`，3 轮 mean±std，EXP-K02 §5）：

| 版本 | latency (ms) | TFLOPS | vs cuBLAS | 逐级归因 |
|---|---|---|---|---|
| v0 naive | 26.369±0.472 | 5.2±0.12 | 3.4% | — |
| v1 tile | 21.114±0.047 | 6.5±0.00 | 4.2% | smem tiling 仅 +25% |
| v2 wmma | 1.536±0.008 | 89.5±0.46 | 57.6% | Tensor Core ×13.8 |
| v3 dbuf | 1.439±0.007 | 95.5±0.49 | 61.4% | 双缓冲 +6.7% |
| v4 bigtile | 1.033±0.007 | 133.1±0.97 | 85.6% | 大 tile +39% |
| cublas（真库） | 0.884±0.004 | 155.4±0.62 | 100% |— |

### 5.1 轴与口径

TFLOPS = 2·M·N·K / 时延（gemm/src/main.cu：92，「每输出 1 mul + 1 add」口径）；「3 轮」指三次独立进程运行（raw 各一份，UTC 前缀落盘），±号后是**轮间** std；每轮的时延本身已是 3 次预热后 50 iters 的均值（v0/v1 慢版取 iters/10、下限 3，main.cu：105-107）。README 图 1(figures/01_gemm_tc_ladder.png)就是此表的水平条形图：横轴 TFLOPS，误差条 = 轮间 std，条上百分比 = vs cuBLAS；标题即结论句，脚注给源数据文件——图不携带表之外的信息，存疑时回 CSV。

**「vs cuBLAS」这一列的分母是同一 harness 内的 cuBLAS，不是别处的 cuBLAS。** 这一点在跨仓比较时是关键：自家 Triton 版所在 harness 下 cuBLAS = 159.8 TFLOPS，本 harness 下 155.4，两者差约 3%(EXP-K02 §6)。所以「85.6%」这个比值只在本 harness 内有意义，跨 harness 的比较必须带这个限定。

### 5.2 三笔自洽性核对

拿到一张性能表，先做三个不需要额外实验的核对——**任何一个对不上，先怀疑表而不是怀疑硬件**。

1. **TFLOPS 与时延自洽**：1.374×10¹¹ FLOP / 1.033×10⁻³ s ≈ 1.33×10¹⁴ = 133 TFLOPS，与表一致。同理 cuBLAS:1.374×10¹¹ / 0.884e-3 = 1.554×10¹⁴，亦一致。
2. **不越过物理上界**：v4 的 133.1 是 165.2 峰值的 80.6%，cuBLAS 155.4 是 94.1%，都在 100% 以内，合理。**若某一行超过 100%，不是破纪录，是口径错了**（FLOP 计数、时延单位或对照物三者之一）。这条检查在 memory-bound 算子上更常救命：本仓 reduce 就是靠「等效带宽 > 理论峰值」发现测量落在 L2 而非 HBM 的（EXP-K04 §4.1）。
3. **带宽侧不触顶**：v4 的全局读约 2.15 GB(§3.4.3)，摊到 1.033 ms 上是 2.08 TB/s——**这个数超过 1008 GB/s 的 DRAM 峰值**，再次说明大部分请求由 L2/L1 供给，DRAM 强制流量只有 100.7 MB 量级（0.1 GB / 1.033 ms = 97 GB/s，占峰值不到 10%）。带宽侧确实远未触顶，与 §3.1 的 compute-bound 判定一致。

### 5.3 误差列的读法:7.58e-04 是什么

正确性列有一个值得读的细节：v0-v4 的 `max_rel_err` **全部等于 7.58e-04**，与累加顺序无关（EXP-K02 §5）。

这个数不是随机的。fp16 的尾数是 10 位，单位舍入 u = 2⁻¹¹ ≈ 4.88e-04；输出 C 只在写回时舍入一次，相对误差上界就是 u。实测 7.58e-04 ≈ 1.55u，略高于 u，来源是**输入 A、B 也是 fp16**：每个输入元素带一次相对误差 ≤ u，乘积带 2u，4096 项求和在 fp32 里累加不再显著放大（fp32 的 u 是 2⁻²⁴，小 8192 倍），最后写回再加一次 u。粗算 3u ≈ 1.46e-03 是上界，实测 7.58e-04 落在其内（**本讲义折算**）。

**五个版本同值这件事本身是结论**：它说明误差被输入/输出的 fp16 舍入**地板**主导，而不同版本的累加顺序差异（v0 是纯串行、v2-v4 是 Tensor Core 内部的树形累加）全部埋在这条地板之下。反过来说，这个 gate 的分辨率上限就到 fp16 的 ulp——**比这更细的算法差异，它测不出来**。这是边界不是优点；若要区分累加顺序的影响，需要 fp32 输入输出的另一套对照。

### 5.4 这个实验设计防了哪些坑

预热 3 次驱走冷时钟与懒初始化（main.cu：109）；计时用单 event 对包住整段循环再除 iters(main.cu：110-113)，避免逐次 event 记录本身的开销偏置；固定 srand(42) 使所有版本所有轮吃同一输入（main.cu：53-55），否则「版本差异」里混着「输入差异」；正确性以 cuBLAS 输出为参考，相对误差分母用全局 absmax 而非逐元素（近零元素上逐元素相对误差无意义地爆炸），抽样步长取素数 997/97 避开 2 的幂结构的周期性采偏（main.cu：63-71、100-103）；结果只写 BENCH_OUT 指定的新文件、首行 provenance(main.cu：77-90)，历史数据永不覆盖。慢版少跑的合法性有数据支撑：v1 的 std≈0(21.114±0.047)，5 iters 统计已够（EXP-K02 §7）。

**它没防住什么，也要说**：(a) 没有做 shape 扫描，所有结论只属于 4096³ 这一个点；(b) 没有控制 GPU 时钟（无锁频），轮间 std 里混着时钟波动——v0 的 ±0.472 ms(1.8%)明显大于其余版本（0.2%–0.7%），但 v0 本身慢，绝对波动占比小，不影响排序；(c) NCU 计数器在本容器不可用（EXP-K01 §7），所以所有「bank conflict」「stall reason」层面的解释都只能是账面推断。

### 5.5 数字背后的机理账

核验 133.1:1.374×10¹¹ FLOP / 1.033×10⁻³ s ≈ 1.33×10¹⁴ = 133 TFLOPS，自洽；对 4090 的 165.2 TFLOPS 峰值即 81%（docs/talk/whiteboard_card_roofline.md 口径），cuBLAS 155.4 约为峰值 94%（账面）；带宽侧远未触顶（§3.1），所以逐级增量全部应从指令与复用侧解释，与归因列一致。

把五级增量按「它买到了什么」重新排一遍：

| 台阶 | 倍率 | 买到的东西 | 理论上界（本讲义折算） |
|---|---|---|---|
| v0→v1 | +25% | 访存延迟与事务数 | 指令路径不变，天花板仍是 16.5 TFLOPS(§3.1.4) |
| v1→v2 | ×13.8 | 指令世代：每条指令产出的 FLOP 从 0.4 到 10³ 量级 | 165.2 TFLOPS（Tensor Core 峰值） |
| v2→v3 | +6.7% | 装载与计算的重叠 | 1/（1−装载占比） ≈ 1.07(§3.3.6) |
| v3→v4 | +39% | 两层复用（threadblock 层翻倍 + warp 层 +33%） | 无独立上界，受寄存器预算限（§3.5.2） |
| v4→cuBLAS | +16.7% | 布局控制、多级流水、tile 形状选择 | 推断级，未做对照臂（§3.7） |

**这张表比原表更有用的地方**：它把「快了多少」换成「为什么最多只能这么多」。面试里被追问「你还能再快多少」时，能答的不是「再试试」，而是「v3 那一级已经贴着 1.07 的理论上界，再优化它没有意义；有余量的是 v4→cuBLAS 那 16.7%，候选原因有三个，检验方式是 v5」。

## 6 误区与边界

**误区 1：「occupancy 越高越好」。** v2 理论 occupancy 75% 却只有 89.5 TFLOPS，v4 以 33% 拿到 133.1(EXP-K02 §5)。occupancy 买的是「每个调度器手上的替补 warp 数」(§3.5.3)，而 Tensor Core 吞吐靠 fragment 级 ILP 与复用喂满（§3.5.4）。但注意适用边界：这不是「occupancy 无用论」——FA2 v3 在 1 block/SM 的约束下加 warp 直接 +33%(EXP-K03)，延迟遮蔽缺口真实存在时 occupancy 就是要害。先判断延迟是否已被遮蔽，再谈 occupancy。**可操作的判据**：把实测吞吐除以该路线的理论上界；接近 1 就说明延迟已被遮蔽（v4 是 81%），远小于 1 才需要考虑加线程（v1 是 39%，§3.1.4）。

**误区 2：「smem tiling 是 GEMM 优化的大头」。** 教科书顺序造成的错觉。本梯实测：tiling +25%，换指令世代 13.8 倍（EXP-K02 §5）。§3.1.4 给了这个错觉的机制解释：tiling 不改变「5 条指令产 2 FLOP」的指令路径，所以它的天花板与 v0 一模一样。方向由 bound 类型决定：compute-bound 的 GEMM 里访存微调是坡，指令世代是台阶；memory-bound 的 gemv/reduce 里恰好反过来，指令层面微调收益趋近 0（第二篇讲义附课）。错配的优化在错误的方向上没有回报。

**误区 3：「对照物可以信文件名，单轮数字可以进结论」。** 本仓两个被证伪的实例，是最硬的教学材料：

- **对照物必须验调用点**。softmax 曾有的「对照库」对比，对照物经源码核查系自写 warp 原语 kernel（cuBLAS 并无 softmax API），整条对比链撤销（EXP-K01 §5）；该文件在仓内已改名 `handwritten_ref`，现行对照换成同算子的官方实现 `cudnnSoftmaxForward`，结果是**对齐形状 1024×1024 上 v4 快 cuDNN 6.7%，非对齐形状 1024×1500 上 cuDNN 反过来快 9.9%**（3 轮，EXP-K04 §4.2）。**注意换对照物之后结论的形状不一样了**：从一个单向的「快 N%」变成一对有条件的数字。凡「vs X」先验 X 的调用点，gemm_cublas.cu 的验真就是这条规矩的产物。
- **对照物必须同算子**。reduce 曾用 `cublasSasum` 作对照，而 asum 算的是 Σ|x|，与被测的 Σx 不是同一个算子；现行对照换成同算子的官方实现 CUB `DeviceReduce::Sum`(EXP-K04 §1)。
- **对照物也要跑 3 轮**。gemv 的领先幅度在单轮口径下曾被测得远高于现值，补齐对照侧的 3 轮之后回落——**自家 kernel 完全复现，坏轮全在对照那一边**；现行口径是 v3 比 `cublasSgemv` 快 **34.1%**（4096×2048,3 轮，EXP-K04 §4.3），那个单轮数字已撤销。完整过程见第二篇讲义 §8.3.3。

**误区 4：「异步预取/双缓冲是万能提速键」。** v3 仅 +6.7%：compute-bound 下装载占比本来就小，重叠的上限就是这块占比(§3.3.6 给了 1/(1−x) 的模型与反推的 x ≈ 6.3%)。且 wait_prior 参数写保守（0）不会报错，只会把重叠静默归零——「没错但没用」的优化比报错更浪费时间。复用优先于重叠（+39% vs +6.7%）。

**误区 5：「16 B 这个数是调优扫出来的」。** 它在三处各有独立的硬出处：float4 的 128-bit 对齐、cp.async 的 `cp-size ∈ {4,8,16}`(PTX §9.7.9.26.3.1)、wmma 的 `ldm` 字节倍数（CUDA PG §10.24.1）。**魔法数分两种：有出处的和没想清楚的。** §3.6 的表就是用来把每个常数逼到有出处那一类的。

**误区 6：「理论 occupancy 算出来是多少，实际就是多少」。** 理论 occupancy 只考虑寄存器、smem 与 warp 上限三个静态约束；实际 occupancy 还受 grid 大小（尾波）、block 调度顺序与运行时资源竞争影响。本梯 v4 的 grid 是（4096/128）² = 1024 个 block，128 个 SM 每个驻 2 个 ⇒ 一次可容纳 256 个 block，要跑 4 波，尾波量化（wave quantization，Matrix Multiplication Background User's Guide §3.2）带来的损失是 1024/256 = 4.0 整波，**恰好整除，没有尾波损失**（本讲义折算）。这是运气好，不是设计——换成 M=N=4096+128 就会多出一个只有 8 个 block 的尾波，SM 利用率掉到 3%。**本梯没有做非整除 shape 的对照臂，这条属账面推断。**

**边界声明**：本梯所有结论的实测范围是 4096³ 单一 shape、fp16 存储 / fp32 累加、行主序、RTX 4090；v1/v2/v4 有尺寸整除前置条件（gemm/include/gemm_common.h：8-13），无尾块处理——目标是归因，不是产品化。85.6% 不是「追平 cuBLAS」，措辞以 gemm/README.md 的约束表为准；剩余约 14% 差距的去向（smem swizzle、多级流水、tile 形状选择）为推断级，NCU 计数器在本容器不可用，未做计数器级确认（EXP-K02 §6）。§3.7 逐条列出了本篇与官方文档/工业实现之间的每一处差异及其来源，其中「256×128 tile」「多级流水」「cuBLAS 是否选了 fp32 累加 kernel」三条均为未验证项。

## 7 连环追问

**Q1：mma_sync 一条指令做多少 FLOP？** 16×16×16 tile 的 D = A·B + C：16³ = 4096 次乘加 = 8192 FLOP，由一个 warp 的 32 线程协作完成。对比：v1 内层每条 FFMA 只贡献 2 FLOP，还要配两条 half→float 转换与两条 smem 读——**每条指令 0.4 FLOP 对每条指令 10³ 量级 FLOP**，这才是 13.8 倍的机制（§3.1.4）。

**Q2：v0→v1 只有 +25%，是不是 v1 写坏了？** 不是，而且可以证明它「不可能更好」。v0 与 v1 的每次 FMA 都要 5 条 warp 级指令（2 条访存 + 2 条转换 + 1 条 FFMA），tiling 只把 LDG 换成 LDS，**指令条数一条没少**。按每 SM 每周期 4 条发射的上限折算，这条路线的算力顶是 16.5 TFLOPS(§3.1.4)，而实测 6.5 已是其 39%。带宽账也支持：v1 只用了约 408 GB/s，远未触顶。tiling 本身是对的，只是这个算子的病不在这。

**Q3：BK 为什么取 32，不是 16 或 64？** 四条约束的交点（§3.4.4）：复用公式 2·BM·BN/(BM+BN) 与 BK 无关，所以加大**无收益**；BK=64 使 smem 翻倍、驻留 block 减半；BK=16 使 `__syncthreads` 频率翻倍；wmma 的 k 维是 16，BK 必须是 16 的倍数。32 是同时满足四条的最小值。**本梯未做 BK 扫描，这是账面推断。**

**Q4：`__pipeline_wait_prior(1)` 的「1」精确指什么？写 0 会怎样？** 文档给的定义是索引式的：「设 {0,1，…，L} 是本线程调用 `__pipeline_commit()` 的序号，则 `wait_prior(N)` 等到第 L−N 批完成」(CUDA PG §10.28.4.3)。v3 在轮 t 先 commit 第 t+1 批（L = t+1），`wait_prior(1)` 等到第 t 批——**恰是本轮要读的那一面**。写 0 = 等到第 t+1 批，连刚发的预取一起等，重叠归零，性能退回 v2，而结果完全正确——最难发现的一类性能 bug(§3.3.3)。

**Q5：双缓冲已经隔离了读写，为什么每轮还要两个 `__syncthreads`？** 第一个：cp.async 的完成只对发起线程可见（PTX §9.7.9.26.3.3 的可见性条款），且组计数是每线程的（§9.7.9.26.3.2），而 tile 由全 block 分片搬运——**两条独立理由各自都要求一次跨线程同步**(§3.3.4)。第二个：双缓冲隔离的是「计算 vs 在途预取」，不隔离「本轮读 buf[p] vs 下轮 cp.async 写 buf[p]」——p 翻面后就成了预取目标，这正是 CUDA PG §10.28.4.1 竞态清单里的第一条 "Loading from dst_shared"(§4.6)。

**Q6：accumulator 为什么必须 fp32？** fp16 尾数 10 位，单位舍入 2⁻¹¹ ≈ 4.9e-04。输入取（−1,1） 零均值（main.cu：51），4096 长点积的部分和量级约 √4096 ≈ 64 倍单项量级，即 O(20)；继续累加时，当部分和达到 20 而新增项是 O(0.3) 时，fp16 的相邻可表示数间隔在 20 附近是 2⁻¹¹×16 ≈ 0.0078，尚能表示，但误差按随机游走累积约 √K·u·|部分和| 量级，4096 项下相对误差可到 6e-2 量级（**本讲义折算，量级估计**），已超过 2e-2 的 gate。fp32 累加下实测 max_rel_err = 7.58e-04，且由 I/O 舍入主导（§5.3）。**顺带一提**：fp16 accumulator 的 Tensor Core 峰值是 330.3 TFLOPS，是 fp32 累加的两倍（Ada 白皮书 Appendix A）——**这就是精度与吞吐的交易在硬件表上的直接价格**。

**Q7：写回时对 fragment 逐元素做 fp32→fp16 转换，凭什么合法？** CUDA PG §10.24.1 明写："In the special case where all threads in the warp will apply an element-wise operation uniformly to all fragment elements， direct element access can be implemented using the following fragment class members." 本梯的写回段（gemm_v2.cu：88-91）对同一个 fragment 的所有元素施加同一个一元转换，**正是这句话授权的场景**。但映射本身仍是黑箱——你不知道 x[e] 是第几行第几列，这正是 FA2 做不了行级 softmax 的根源。**注意**：第二篇讲义里「两个同 shape fragment 逐元素相加」比这条授权更进一步，属实践依赖而非文档承诺（§3.2.2）。

**Q8：每 SM 只驻 2 个 block 是谁限的？算一遍。** 92 reg × 256 thr = 23552；⌊65536/23552⌋ = 2（寄存器限）。smem 侧 32 KB 对 100 KB 上限允许 3 个，warp 侧 8 warp/block 对 48 warp 上限允许 6 个，都不是瓶颈。2 block × 8 warp = 16 warp / 48 = 33%(§3.5.2)。其中 64 reg/thr 是 8 个 accumulator fragment 的账（每 fragment 每 lane 8 个元素）。**稳健性**：按 warp 粒度取整重算仍是 2(§3.5.2)，结论对分配粒度不敏感。

**Q9：为什么不把 tile 再放大（如 256×128）？** 复用公式还会涨（2·256·128/384 = 170.7 FLOP/元素，比 128 涨 33%），但 accumulator fragment 数量随面积线性涨，寄存器先爆（92 已用去 64 个在 acc 上；翻倍到 16 个 fragment 就是 128 reg 只在 acc 上，总数逼近 160，驻留 block 掉到 1）；smem 也要涨 50%。NVIDIA 自己的文档说 256×128 是 cuBLAS 最有效率的 tile 之一（Matrix Multiplication Background User's Guide §2.3），它能用而本梯不能用，差别在于它走 mma 路线、能自己控制 accumulator 的寄存器布局并做寄存器双缓冲（§3.7.2）。**此为账面推断——本梯未做 256 级对照臂。**

**Q10：cuBLAS 对照是怎么「验真」的，为什么较真？** 源码核查调用点：gemm_cublas.cu 有 `cublasCreate` + `cublasGemmEx`，确系真库（EXP-K02 §2）。较真的原因是本仓 softmax 的教训：对照物系自写 kernel，整条对比叙事作废（EXP-K01 §5），后来换成 cuDNN 重做才有了可用的结论（EXP-K04 §4.2）。对照物命名诚实是所有「vs X」数字的地基。

**Q11：cuBLAS 那 155.4 会不会本身就没跑在 fp32 累加上？** 好问题，而且诚实回答是**未核实**。本梯传的 compute type 是 `CUBLAS_COMPUTE_32F`，但 algo 用的 `CUBLAS_GEMM_DEFAULT_TENSOR_OP` 在文档里被描述为「允许使用降精度的 `CUBLAS_COMPUTE_32F_FAST_16F` kernel（为向后兼容）」(cuBLAS Library §2.8.11)。间接证据是误差量级与各版本一致（7.58e-04），但这不是证明。**这个不确定性的方向对本梯结论是保守的**：若 cuBLAS 实际走了 fp16 累加的快路径，那 155.4 偏高，本梯的 85.6% 会偏低。

**Q12:85.6% 会不会只是 4096³ 一个点的幸运？** 诚实回答：是单 shape 实测，不外推。非方阵/LLM 实际形状（如小 M 大 N 的 decode 形状）未测，列为后续工作（EXP-K02 §7）；wave quantization（§6 误区 6）、tile quantization 与 tile 形状选择在别的 shape 上会改变双方相对位置。可以说的是：同协议同 harness 下 3 轮稳定（±0.97 TFLOPS，相对 0.73%），这个点本身立得住。

**Q13：你说剩余差距在 swizzle/多级流水，证据呢？** 诚实回答：推断级，不可当实测说（gemm/README.md 约束表）。NCU 计数器在本容器不可用（EXP-K01 §7），没有 bank conflict 计数的直接证据。旁证有二：自家 Triton 版同尺寸 160.5 TFLOPS（triton-kernels 仓，跨 harness，其 harness 下 cuBLAS = 159.8，本 harness 155.4，差约 3%），Triton 编译器发射 mma + ldmatrix + swizzle 而 wmma 不暴露 swizzle，差距方向与假设一致；结构分析上 wmma 固定布局的 smem 访问模式无法错位。检验方式明确：v5 用 mma PTX + ldmatrix + 手工 swizzle 重写，差距收窄即证实（EXP-K02 §7）。

**Q14：为什么 wmma 不能做 swizzle，而 mma 可以？** 因为 swizzle 是在**地址计算**层面把 smem 里的数据打散，而 `wmma::load_matrix_sync` 只接受（指针， ldm） 两个参数，它内部按什么模式去取这块（16×16， ldm） 的数据是不公开的——**你没有一个接口可以告诉它「我的数据按 XOR swizzle 摆过了」**。`ldmatrix` 则相反：它由 32 个线程各自提供一个地址（PTX §9.7.15.5.15：".x4 时 Threads 0–7 提供 addr0–addr7，Threads 8–15 提供 addr8–addr15，依此类推"），地址由你算，所以你可以在算地址时把 swizzle 加进去。**「谁算地址」决定了「谁能 swizzle」**，这是 wmma 与 mma 分界的另一个具体面。

**Q15:16 B 的 cp.async 与 4 B/8 B 有什么实际差别？** 除了条数（16 B 一条顶 4 B 四条），还有缓存行为的差别。CUDA Programming Guide 的异步拷贝一节写明："Copying 4 or 8 bytes always happens in the so called L1 ACCESS mode， in which case data is also cached in the L1， while copying 16-bytes enables the L1 BYPASS mode， in which case the L1 is not polluted."(CUDA Programming Guide §4.11.1 Using LDGSTS)对应到 PTX 就是 `.ca` 与 `.cg` 两个 qualifier(§9.7.9.26.3.1)。**对 GEMM 这种「搬进 smem 之后就不再从 L1 读同一份」的场景，不污染 L1 是纯赚**——16 B 因此不只是「更快」，还「更干净」。

**Q16：如果换一张 L2 更小的卡（比如 6 MB 的 GA102），本梯的结论会变吗？** 会变一部分。§3.1 的 compute-bound 判定不变（它只依赖 I 与 I\*）。但 §3.4.3 里「A、B 合计 64 MB 装得进 72 MB L2 所以 DRAM 强制流量不变」这句话会失效：6 MB 的 L2 装不下，v2/v3 的 4.3 GB 与 v4 的 2.15 GB 会有相当一部分真的打到 DRAM，于是**大 tile 的收益会比 +39% 更大**（它同时省了 DRAM 流量，而不只是省 smem 往返）。本仓在 reduce 上有同一件事的直接演示：同一个算子在 L2 常驻区间与 HBM-bound 区间给出方向相反的版本排序（EXP-K04 §5）。**跨卡引用性能结论前，先检查数据落在哪一级缓存。**

**Q17：v2 只有 4 个 warp、v4 有 8 个，warp 数是怎么定的？** 由 tile 划分反推，不是先定 warp 数。v2 的 block tile 是 64×64，每 warp 拿 32×32 ⇒ 4 warp；v4 的 128×128，每 warp 拿 64×32 ⇒ 8 warp（gemm_v4.cu：10-14 的小图）。**真正的自由度是「每 warp 管多大」**：管太小则 fragment 复用不足（v2 每次装载喂 1 次 mma），管太大则寄存器爆。v4 选 64×32 = 4×2 个 fragment，是「6 次装载喂 8 次 mma」与「64 个 accumulator 寄存器」的折中点。**本梯未扫描 warp tile 形状（比如 32×64 或 128×16），这是账面推断。**

**Q18：`__syncthreads()` 一次要多少代价？每轮两次会不会太贵？** ptxas 报告三个 kernel 都是 `used 1 barriers`(§3.5.1)，说明编译器只用了一个硬件 barrier 资源，不同调用点复用它。代价的主体不是 barrier 指令本身，而是**它把整个 block 拉齐到最慢的 warp**：所有 warp 必须等最后一个到达。v4 每个 K 块两次 barrier，K=4096/BK=32 ⇒ 128 轮 × 2 = 256 次。以 1.033 ms / 128 轮 ≈ 8 μs 每轮计，barrier 若各占 100 ns 量级也只是百分之几（**本讲义量级估计，未实测**）。**对照第二篇讲义**：FA2 每个 tile 要 5 次 barrier，且中间夹着一段纯标量的 softmax，barrier 的代价就完全不是一个量级——同一个原语，贵不贵取决于它把什么东西拉齐。

**Q19：为什么 v2/v3 的 block 是 128 线程（4 warp），而不是 256？** 128 线程 = 4 warp 恰好对应 2×2 的 warp 排布，与 64×64 的 block tile 匹配。若强行用 256 线程算同一个 64×64 tile，每 warp 只能拿 32×16 或 16×32，fragment 数从 4 掉到 2，**复用反而变差**。这条说明「线程越多越好」在 Tensor Core 代码里不成立：线程数是 tile 划分的结果，不是可以独立调的旋钮。v3 因此也保持 128 线程——它与 v2 的唯一变量是装载方式（控制变量设计），连线程数都不许变。

**Q20：如果只允许你改一处代码把 v4 再推高一点，你改哪里？** 按 §5.5 的上界表，v3→v4 之后唯一还有明确余量的是「v4→cuBLAS 的 16.7%」，而那一层的入口是 mma + ldmatrix(§8.1)。**但那不是「改一处」，是重写。** 如果限定「只改一处」，最有性价比的候选是给 `As`/`Bs` 加 padding 试着打散 wmma 装载的 bank 模式——代价只有几 KB smem（32 KB 到 36 KB 仍是 2 block/SM），风险是 padding 会破坏 float4/cp.async 的 16 B 对齐，需要把 padding 取成 8 个 half 的倍数。**这是一个可做的对照臂，本梯没做，如实记为开放项**；它的预期收益也说不准，因为 wmma 内部按什么模式取数是不公开的（§7 Q14），padding 打散的是不是它真正撞的那个模式，只能试出来。

## 8 工业对照与延伸

### 8.1 与 CUTLASS 逐层对照:本梯缺的是哪几层

CUTLASS 是 cuBLAS 内核的开源近亲，它的官方文档把 GEMM 的分层写得最清楚（Efficient GEMM in CUDA，§"Hierarchical Structure"）：threadblock tile 负责从全局内存取数并放进 shared memory，warp tile 负责从 shared memory 取进寄存器，thread/instruction tile 在寄存器上做乘加。文档给的循环嵌套是「threadblock 的 M/N 循环 → K 主循环（一次迭代 = 一个 stage）→ warp 的 N/M 循环 → warp-K 循环（完全展开，一次迭代 = 一个 k Group）→ MMA 指令循环」。**本梯的三层与它一一对应**（§3.4.3 的表），差的是每一层上的工程：

| 层 | CUTLASS 有 | 本梯有 | 差在哪 / 为什么 |
|---|---|---|---|
| threadblock | 多 tile 形状模板 + 按 shape/arch 的启发式选择 | 单一 128×128×32 | 本梯是归因实验，不是库；多形状要配 tuning 基础设施 |
| threadblock | multistage 软件流水（kStages 3-5，cp.async 深度可配） | 双缓冲 2 级 | smem 预算：升到 4 级要 64 KB，驻留 block 从 2 掉到 1(§3.7.3) |
| threadblock | smem swizzle 布局（消 bank conflict） | 只能做 padding | wmma API 不暴露地址计算，swizzle 无处可加（§7 Q14） |
| warp | 寄存器双缓冲（"one fragment is passed to CUDA and TensorCores during the current matrix computation， while the other is used to receive shared memory fetch returns"） | 无 | wmma fragment 的装载时机由编译器决定，手工双缓冲无接口 |
| instruction | mma.sync + ldmatrix，布局公开 | wmma，布局不公开 | §3.2.2 性质 2 |
| epilogue | 融合（bias/activation/split-K 规约） | 直写 | 本梯不需要 |

**一句话结论：差距集中在 instruction/warp 层的布局控制，不在算法。** 本梯的 v0→v4 已经把「算法层」能拿的都拿了（tiling、Tensor Core、异步预取、大 tile），剩下的 14.4% 全在「怎么摆数据」这一层——而这一层的入口是 mma PTX，不是 wmma。

入口文件：`include/cutlass/gemm/threadblock/mma_multistage.h`（流水线，§3.3.5 引用了它的：360-370 与：488-489）、官方文档 "Efficient GEMM in CUDA"（分层讲解）。

### 8.2 其他三条对照线

- **Triton**：`tl.dot` 自动编译到 mma.sync + ldmatrix + swizzle，程序员只写 tile 逻辑。自家 Triton 版 160.5 TFLOPS（跨 harness，推断级）对本梯 v4 的约 17% 领先，即「编译器代管布局」对「wmma 固定布局」的溢价（EXP-K02 §6）。**这条对照的价值在于它把「布局控制」这件事的价格标出来了**：不需要你自己写 PTX，只需要换一个把布局当一等公民的编程模型。
- **cuBLAS/cuBLASLt**：运行时按 shape/arch 启发式选 kernel；155.4 TFLOPS = 峰值 94.1% 是它在本协议点的位置。文档也承认这套启发式会做与数值相关的选择（cuBLAS Library §2.1.10 GEMM Algorithms Numerical Behavior 提到 heuristics 可能沿 K 维切分以提高 occupancy）。手写单 shape 追到 85.6% 的含义：**通用库的领先主要是布局与流水线工程，不是不可知的魔法**；但它的通用性（任意 shape 都不塌）是本梯完全没有的东西，这一点在 §6 误区 3 的 softmax 对照里有直接的实证——同一个手写 kernel 在对齐形状上快 6.7%，换个非对齐形状就被官方库反超 9.9%(EXP-K04 §4.2)。
- **世代边界**：cp.async 是 Ampere/Ada 的机制（PTX §9.7.9.26.3.1 明写 "Requires sm_80 or higher"）；Hopper 换 TMA(Tensor Memory Accelerator)+ warp specialization（生产者/消费者 warp 分工），CUDA PG §10.29 单列一节讲它。DeepGEMM、CUTLASS 3.x 的 Hopper kernel 即此路线——**本讲义的组计数推导在 Hopper 上对应 mbarrier 事务计数，概念同构、原语不同**：组计数问「我发了几批、等到第几批」，mbarrier 事务计数问「我期待多少字节到达、到齐了没有」，后者更精确但也更啰嗦。

### 8.3 延伸阅读(每条一句「读它能解决什么疑问」)

**仓内一手材料**

1. `gemm/src/gemm_v3.cu:15-24` 与 `gemm/src/gemm_v4.cu:20-25` 两个文件头——cp.async 组语义、occupancy/ILP 与 BK 选择的仓内一手推导。读它能解决：「§3.3 的归纳证明与 §3.5 的 ILP 论证在代码里长什么样，以及同一个结论写给自己看和写给别人看有什么差别」。
2. `records/EXP-K02_cuda_gemm_tc_ladder.md` §6-§7——H1/H2 假设的判定过程、Triton 对照的完整口径与开放问题。读它能解决：「跑之前锁定的假设是什么，跑完哪些成立哪些降级，以及为什么单轮数字被整改成 3 轮」。
3. `records/EXP-K04_standard_library_baselines.md`——同算子官方基准的补齐过程。读它能解决：「对照物选错（异算子）、尺寸选错（落 L2 不落 DRAM）分别会把结论带偏多远」。
4. `docs/lectures/02_wmma_tax_fa2.md`——同一套 wmma 工具箱在 FA2 上的反面结局。读它能解决：「§3.2.2 的性质 2（布局不公开）在什么算子上会从『无所谓』变成『致命』」。

**官方文档**

5. CUDA C++ Programming Guide §10.24 "Warp Matrix Functions"——wmma 的完整契约。读它能解决：「哪些 fragment 操作是文档承诺的、哪些是你自己赌的」，以及 `ldm` 与指针对齐的确切要求（§3.2.3 的 C1/C2 就出自这里）。
6. PTX ISA §9.7.9.26.3(cp.async / commit_group / wait_group)——异步拷贝的硬件语义。读它能解决：「wait 参数到底在等什么、空组算不算一组、拷贝结果对谁可见」这三个把 §3.3 的正确性押在上面的问题。
7. PTX ISA §9.7.15.5.8 "Matrix Fragments for mma.m16n8k16"——mma 的 lane→元素映射公式。读它能解决：「wmma 不告诉你的那张表，mma 是怎么告诉你的」；第二篇讲义 §3.8 的 v5 路线图全部建立在这一节上。
8. PTX ISA §9.7.15.5.15 `ldmatrix`——按 mma 布局从 smem 取数的指令。读它能解决：「为什么 mma 路线能做 swizzle 而 wmma 不能」(§7 Q14)，以及「哪些线程提供地址」这个决定一切的细节。
9. CUDA C++ Programming Guide §20.7 "Compute Capability 8.x"——sm_89 的 SM 结构、发射规则与资源上限。读它能解决：「§3.5 的驻留计算用的每一个上限数字从哪来」，特别是那句「each scheduler issues one instruction for one of its assigned warps that is ready to execute」——它是 §3.1.4 指令路径账的地基。
10. NVIDIA Ada GPU Architecture 白皮书，Appendix A + §Memory Subsystem——RTX 4090 的完整规格与 L2 的世代变化。读它能解决：「165.2 / 1008 / 72 MB 这三个分母各自的准确出处，以及为什么完整 AD102 的 96 MB 不能拿来当 4090 的 L2」。
11. NVIDIA "Matrix Multiplication Background User's Guide" §2、§3.1、§3.2——算术强度、tile quantization、wave quantization。读它能解决：「除了算术强度，还有哪两种量化效应会在换 shape 时把性能砍半」（§6 误区 6）。
12. NVIDIA "GPU Performance Background User's Guide" §4——三个限制因子与 ops：byte 的官方定义。读它能解决：「roofline 的 NVIDIA 方言怎么说，以及 latency 为什么被单列为第三个因子（它不在 roofline 的两条边上）」。
13. CUDA C++ Best Practices Guide §10.2.3.1 "Shared Memory and Memory Banks"——bank 结构、冲突与广播。读它能解决：「§4.2 里 As 的广播读与 Bs 的连续读为什么都不冲突」，以及第二篇讲义 LDS=68 的另一半理由。

**论文与工业实现**

14. Williams， Waterman & Patterson， "Roofline： an insightful visual performance model for multicore architectures"， CACM 52(4)：65-76, 2009， DOI：10.1145/1498765.1498785——roofline 原论文。读它能解决：「operational intensity 的分母为什么必须是 DRAM traffic」——这一条正是本仓在 reduce 上踩过的坑（§3.7.4），也是把 roofline 用对的唯一前提。
15. CUTLASS 官方文档 "Efficient GEMM in CUDA" + `include/cutlass/gemm/threadblock/mma_multistage.h`——工业级 GEMM 的分层设计与多级流水实现。读它能解决：「本梯还差哪几层」（§8.1 的表），以及「wait 参数的通式 N = D − 1 在工业代码里长什么样」(§3.3.5)。
