# 02 · wmma 架构税:FA2 forward 五版走读

> 对象：`flash-attn/src/` 五个 kernel（v0 warp-per-row → v1 K/V 进 smem → v2 wmma + smem 往返 softmax → v3 8 warp 并行组织 → v4 双 pipeline 重叠），对照自家 Triton 版（mma + 寄存器驻留，跨 harness，推断级）。数字权威：`flash-attn/project-proof/data/derived_fa2_proto_stability.csv`（3 轮 mean±std），实验记录 = records/EXP-K03_cuda_fa2_ladder.md（下文简称 EXP-K03《CUDA FA2 forward 简化版版本梯》）；对照口径与 gemv/reduce 的访存课来自 records/EXP-K01_4090_rebench.md（简称 EXP-K01《四 kernel 4090 重基准》）。协议：B=1，Hq=32，Hkv=8(GQA 4:1)，D=128，causal，S=512..4096，fp16 存储 / fp32 在线累加，RTX 4090(sm_89)。前置：讲义 01 §3.2（wmma fragment 的三条性质）与 §3.3（cp.async 组语义）——本篇直接在那两处的结论上继续推。引用规矩：凡属论文/官方文档的论断一律给出处（标题 + arXiv/DOI 编号 + 章节或公式编号；文档给 URL 路径 + 小节号），关键句给原文；凡属本讲义补出的推导或折算，行内标注「本讲义折算」或「账面推断」；无法用检索确认的说法标注「未核实」。本仓自己的数字一律带 EXP 锚。

## 目录

- [1 这一篇回答什么问题](#1-这一篇回答什么问题)
  - [1.1 本篇要建立的五条能力](#11-本篇要建立的五条能力)
  - [1.2 符号、协议与硬件常数](#12-符号协议与硬件常数)
  - [1.3 本篇引用的一级文献(详细出处与「读它能解决什么疑问」见 §8.4)](#13-本篇引用的一级文献详细出处与读它能解决什么疑问见-84)
- [2 直觉与第一性原理](#2-直觉与第一性原理)
  - [2.1 论文把这件事写成了什么:IO 复杂度](#21-论文把这件事写成了什么io-复杂度)
  - [2.2 三条贯穿全篇的公理](#22-三条贯穿全篇的公理)
- [3 完整推导与机制](#3-完整推导与机制)
  - [3.1 在线 softmax 三件套:α 修正是恒等式,不是近似](#31-在线-softmax-三件套α-修正是恒等式不是近似)
  - [3.2 架构税的根:fragment 不知道自己属于哪一行](#32-架构税的根fragment-不知道自己属于哪一行)
  - [3.3 smem 分区表逐字段推:90.75KB 与 opt-in 的因果链](#33-smem-分区表逐字段推9075kb-与-opt-in-的因果链)
  - [3.4 相位链五段与每个 barrier 守的竞态对象](#34-相位链五段与每个-barrier-守的竞态对象)
  - [3.5 v3 的并行组织:被钉在 1 block/SM 之后唯一的旋钮](#35-v3-的并行组织被钉在-1-blocksm-之后唯一的旋钮)
  - [3.6 v4 的两组 pipeline:交错等待的正确性论证](#36-v4-的两组-pipeline交错等待的正确性论证)
  - [3.7 28% 逐层拆:哪一层拿到了,哪一层没有](#37-28-逐层拆哪一层拿到了哪一层没有)
  - [3.8 v5 路线图:mma + ldmatrix 到底换来什么](#38-v5-路线图mma--ldmatrix-到底换来什么)
  - [3.9 魔法数总账:每个常数由谁决定](#39-魔法数总账每个常数由谁决定)
  - [3.10 论文/文档怎么说 vs 本项目实测](#310-论文文档怎么说-vs-本项目实测)
- [4 代码逐段走读(按执行顺序)](#4-代码逐段走读按执行顺序)
  - [4.1 在线 softmax 三件套的最小形态(flash-attn/src/fa2_v0.cu:40-48)](#41-在线-softmax-三件套的最小形态flash-attnsrcfa2_v0cu40-48)
  - [4.2 v2 的 smem 分区表(flash-attn/src/fa2_v2.cu:27-47)](#42-v2-的-smem-分区表flash-attnsrcfa2_v2cu27-47)
  - [4.3 opt-in 与 launch(flash-attn/src/fa2_v2.cu:173-181)](#43-opt-in-与-launchflash-attnsrcfa2_v2cu173-181)
  - [4.4 相位 ① 与它两侧的 barrier(flash-attn/src/fa2_v2.cu:86-96)](#44-相位-①-与它两侧的-barrierflash-attnsrcfa2_v2cu86-96)
  - [4.5 相位 ②:QK^T 与「架构税现场」(flash-attn/src/fa2_v2.cu:97-114)](#45-相位-②qkt-与架构税现场flash-attnsrcfa2_v2cu97-114)
  - [4.6 相位 ③④:标量段 softmax 与 O 的按行重缩放(flash-attn/src/fa2_v2.cu:115-139)](#46-相位-③④标量段-softmax-与-o-的按行重缩放flash-attnsrcfa2_v2cu115-139)
  - [4.7 v3:每行两线程与 shfl 合并(flash-attn/src/fa2_v3.cu:91-115)](#47-v3每行两线程与-shfl-合并flash-attnsrcfa2_v3cu91-115)
  - [4.8 v4:两组 pipeline 的三个落点(flash-attn/src/fa2_v4.cu:98-107、130-135、163-167)](#48-v4两组-pipeline-的三个落点flash-attnsrcfa2_v4cu98-107130-135163-167)
- [5 实验数据怎么读](#5-实验数据怎么读)
- [6 误区与边界](#6-误区与边界)
- [7 连环追问](#7-连环追问)
- [8 工业对照与延伸](#8-工业对照与延伸)
  - [8.1 与生产实现逐层定位](#81-与生产实现逐层定位)
  - [8.2 一个跨算子的对照:同一个「税」在别处长什么样](#82-一个跨算子的对照同一个税在别处长什么样)
  - [8.3 附课 · 两区间与对照物:一个数字要满足哪四个条件才能用](#83-附课--两区间与对照物一个数字要满足哪四个条件才能用)
  - [8.4 延伸阅读(每条一句「读它能解决什么疑问」)](#84-延伸阅读每条一句读它能解决什么疑问)

## 1 这一篇回答什么问题

讲义 01 的结论是「同一套 wmma 工具箱，GEMM 够到真 cuBLAS 的 85.6%」。这一篇处理它的反面：同一套工具箱写 FA2 forward，只够到自家 Triton 版的 **28%（跨 harness，推断级）**。读完你应当能：①手推在线 softmax 的三件套（m / l / α）并说清 α 修正为什么是数学恒等而不是近似；②凭空写出 v2 的 smem 分区表并算出 90.75KB，解释为什么这个数字逼出 opt-in 动态 smem 与 1 block/SM；③把每 tile 5 次 `__syncthreads` 各自守的竞态对象逐个说出来，并回答「能删掉哪几个」；④证明 v4 两条 cp.async 流交错等待的正确性——用在途组计数，而不是用「感觉上应该没问题」；⑤把 28% 这个差距逐层拆开，并诚实标出哪一层是实测、哪一层是账面推断、哪一层根本没有测量手段。收尾在 §8.3 附一节访存课：同一个 reduce 算子在 L2 常驻与 HBM-bound 两个区间给出方向相反的结论，以及两次被本仓自己推翻的对照物口径。

### 1.1 本篇要建立的五条能力

1. **恒等式能力**：能把在线 softmax 的三件套从头推一遍，并对每一步回答「它是恒等还是近似、误差量级多大、边界情况怎么处理」(§3.1)。这决定了你能不能理解「FA 是精确算法」这句话到底精确在哪。
2. **契约能力**：能背出 wmma fragment 的能与不能（哪些操作有文档明文授权、哪些是实践依赖），并用它逐条判定 FA2 的每个环节能不能做进 fragment(§3.2)。
3. **预算能力**：能从零推出 v2 的 smem 分区表、算出 90.75 KB，并解释这个数字如何逐级逼出「必须 opt-in」「只能 1 block/SM」「V 不能双缓冲」三个后果（§3.3、§3.6）。
4. **同步能力**：能说清每一次 `__syncthreads` 守的是哪一对线程集的哪一类竞态（RAW / WAR），并判断哪些删得掉、哪些删不掉、为什么（§3.4）。同样地，能用组计数（而不是直觉）证明两条 cp.async 流交错等待的正确性（§3.6）。
5. **归因能力**：能把一个 28% 的差距逐层拆开，分清「实测」「排除」「账面推断」「无测量手段」四种证据等级，并说出每一条的检验方式（§3.7）。以及：能识别一个对照数字什么时候不能用（对照物异算子、尺寸落错缓存层级、跨 harness、单轮），§8.3 附课给了本仓的三个真实案例。

### 1.2 符号、协议与硬件常数

| 符号 | 含义 | 本篇取值 |
|---|---|---|
| B / Hq / Hkv | batch / query 头数 / kv 头数 | 1 / 32 / 8(GQA 4:1) |
| S | 序列长度 | 512、1024、2048、4096 |
| D | head_dim | 128(`FA_D`) |
| BM / BN | query 行块 / key 列块 | 64 / 64 |
| WARPS | 每 block warp 数 | v2 为 4,v3/v4 为 8 |
| m / l / α | 行 max / 行分母 / 重缩放因子 | 见 §3.1 |
| LDS / LDP / LDSP | Ssm / Psm / SP 的行跨距 | 68 / 72 / 72 |

硬件常数与讲义 01 §1.2 同表，本篇直接用到的四条（RTX 4090 / sm_89）：**每 SM shared memory 上限 100 KB、每 block 上限 99 KB、静态 `__shared__` 上限 48 KB**（CUDA C++ Programming Guide §20.2 Table 27 与 §20.7.3）；**每 SM 常驻 warp 上限 48**（同表）；**shared memory 32 个 bank、每 bank 每周期 32 bit**（同表与 CUDA C++ Best Practices Guide §10.2.3.1）；**FP16 Tensor 峰值（FP32 累加）165.2 TFLOPS**（NVIDIA Ada GPU Architecture 白皮书 Appendix A）。

### 1.3 本篇引用的一级文献(详细出处与「读它能解决什么疑问」见 §8.4)

- 在线 softmax 的源头：Milakov & Gimelshein, "Online normalizer calculation for softmax", arXiv:1805.02867。
- 分块注意力与常数显存：Rabe & Staats, "Self-attention Does Not Need $O(n^2)$ Memory", arXiv:2112.05682。
- FlashAttention（IO 复杂度与 tiling）:Dao, Fu, Ermon, Rudra & Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", arXiv:2205.14135。
- FlashAttention-2（工作划分）:Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", arXiv:2307.08691。
- FlashAttention-3（Hopper 异步与 pingpong）:Shah, Bikshandi, Zhang, Thakkar, Ramani & Dao, "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", arXiv:2407.08608。
- GQA:Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón & Sanghai, "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", arXiv:2305.13245。
- wmma 契约：CUDA C++ Programming Guide §10.24。
- mma 布局与 ldmatrix:PTX ISA §9.7.15.5.8、§9.7.15.5.15。
- cp.async 组语义：PTX ISA §9.7.9.26.3；C 层等价定义 CUDA PG §10.28.4。

## 2 直觉与第一性原理

**没有 FlashAttention 的世界**。注意力的定义是 $O = \mathrm{softmax}(QK^\top/\sqrt{D})\,V$。照定义实现要先把 $S = QK^\top$ 整个物化出来：$S$ 是 $S_{\text{len}} \times S_{\text{len}}$ 的矩阵，$S_{\text{len}}=4096$、fp16 时单个 head 就是 32MB，32 个 head 是 1GB，而且这 1GB 要写一遍（GEMM 1 的输出）、读一遍（softmax 的输入）、写一遍（softmax 的输出）、再读一遍（GEMM 2 的输入）。计算量只有 $O(S_{\text{len}}^2 D)$，访存量却是 $O(S_{\text{len}}^2)$ 且系数是 4——算术强度被 $D$ 以下的常数钉死，注意力于是变成一个「明明该 compute-bound 却被中间结果拖成 memory-bound」的算子。

**FA 的想法**：$S$ 不必物化。把 K/V 按列切成 tile，一次只算 $Q$ 的一条行带对一个 K tile 的 $S$ 分块（$64\times64$），当场做完 softmax 的增量更新，当场乘进 $V$，然后丢掉。整个 $S$ 从来不出 shared memory，HBM 上只有 Q、K、V、O 各走一遍。这就是「融合免搬运」：收益不来自任何一条指令变快，只来自中间结果不落地。

**这个想法的代价**：softmax 的分母要看到整行，而 tile 只看到 64 列。所以必须把「先求 max 再 exp 求和」的两遍法改写成能增量合并的形式——这就是在线 softmax，§3.1 逐步推。

**日常类比与失效点**：像流水盘点仓库，每盘完一个货架就把小计并进总数，不等全部盘完再算。类比在两处失效：①盘点的小计可以直接相加，softmax 的小计不能——因为每个小计是相对自己那段的 max 归一化的，合并时要先换基准（这就是 α 因子）；②盘点员随时知道自己手上那张纸对应哪个货架，而 GPU 上 wmma 的 accumulator fragment **不知道自己手上的数属于哪一行**——这正是本篇的主题，类比给不出这个折断点。

**为什么这一篇叫「架构税」**。FA 的全部收益押在「中间结果不落地」上。而 wmma 的 fragment 布局不透明，逼着 $S$ 落一次 shared memory、$P$ 再从 shared memory 读回、$O$ 也只能驻 shared memory——FA 想省的搬运，被 API 从 HBM 层挪到 smem 层重新收了一遍。收多少，就是这一篇要量化的东西。

### 2.1 论文把这件事写成了什么:IO 复杂度

上面那段直觉在 FlashAttention 原论文里有一个精确版本。论文的核心定理（Dao et al.， "FlashAttention： Fast and Memory-Efficient Exact Attention with IO-Awareness"， arXiv:2205.14135，Theorem 2）是：

> "Standard attention requires $\Theta(Nd+N^2)$ HBM accesses, while FlashAttention requires $\Theta(N^2d^2M^{-1})$ HBM accesses."

其中 $M$ 是片上 SRAM 的容量。把本篇的数代进去看它意味着什么（**本讲义折算**）：$N = 4096$、$d = 128$、$M$ 取 shared memory 每 block 的 99 KB ≈ $2.5\times10^4$ 个 fp32 位置。标准实现的 $N^2$ 项 = $1.68\times10^7$；FA 的 $N^2d^2/M$ = $4096^2 \times 128^2 / 2.5\times10^4 \approx 1.1\times10^7$。两者同量级？**是的，在这个 $d$ 和这个 $M$ 下确实同量级**——论文自己也说明这个优势依赖 $d^2 \ll M$。真正决定性的差别是常数与**显存占用**：标准实现要把 $N\times N$ 的 $S$ 物化（32 MB/head，32 head 就是 1 GB），FA 完全不物化。

论文还给了一个配套的下界（Proposition 3）：

> "There does not exist an algorithm to compute exact attention with $o(N^2d^2M^{-1})$ HBM accesses for all $M$ in the range $[d, Nd]$."

**这两条对本篇的意义**：(a) FA 的收益是「不物化中间结果」，不是任何一条指令变快——所以任何把中间结果重新落地的实现（比如本梯被 wmma 逼着把 $S$、$P$、$O$ 落 shared memory）都在往回走，只是往回走到了片上而不是片外；(b) 论文的分析停在 HBM 层，**它没有分析 shared memory 层的往返代价**，而这恰恰是本篇要量化的东西（§3.4）。所以本篇的账不是在验证论文，是在补论文没算的那一层。

算法的另外两个源头也值得点名：分块 softmax 的在线形式来自 Milakov & Gimelshein 的 "Online normalizer calculation for softmax"(arXiv:1805.02867)，该文的目标是把 softmax 的三遍访存降到两遍；而「注意力可以只用常数显存」的构造来自 Rabe & Staats， "Self-attention Does Not Need $O(n^2)$ Memory"(arXiv:2112.05682)。**FlashAttention 的贡献是把这两件事与 GPU 存储层级绑在一起做 IO 感知的分块**，不是发明在线 softmax。

### 2.2 三条贯穿全篇的公理

- **公理 A（融合的收益等于省下的搬运）**：FA 不改变任何一次浮点运算，它只改变数据走过的路径。所以「FA 快多少」的上界永远等于「搬运省了多少」。推论：**当搬运被从 HBM 层挪回 shared memory 层重新收一遍时，收益就被吃掉一部分**——这是「架构税」的定义式。
- **公理 B（能不能做进寄存器，取决于你知不知道数据在哪）**：一个操作只要不需要位置信息（逐元素），就能在不透明的 fragment 上做；只要需要位置信息（沿行/沿列规约），就必须先把数据倒到一个你能寻址的地方。§3.2 的那张表就是这条公理的逐项展开。
- **公理 C（同步的代价等于它拉齐了什么）**：`__syncthreads` 本身很便宜，贵的是它把整个 block 拉齐到最慢的 warp。当 barrier 两侧是两段同质的并行工作时，代价小；当 barrier 中间夹着一段只有一部分线程干活、而且 Tensor Core 完全空转的标量段时，代价就是那一整段（§3.7 第 3 条）。

## 3 完整推导与机制

### 3.1 在线 softmax 三件套:α 修正是恒等式,不是近似

设一行的打分序列为 $s_1,\dots,s_N$（已乘 $1/\sqrt D$）。目标 $o=\sum_j \frac{e^{s_j}}{\sum_i e^{s_i}} v_j$。

定义处理完前 $t$ 个键后的三个量（这就是「三件套」）：

$$m_t=\max_{j\le t}s_j,\qquad \ell_t=\sum_{j\le t}e^{s_j-m_t},\qquad a_t=\sum_{j\le t}e^{s_j-m_t}v_j$$

归纳推进一步。来了新键 $s$，令 $m_{t+1}=\max(m_t,s)$，$\alpha=e^{m_t-m_{t+1}}$：

1. 对任意 $j\le t$：$e^{s_j-m_{t+1}}=e^{(s_j-m_t)+(m_t-m_{t+1})}=e^{s_j-m_t}\cdot\alpha$。——**这一步是指数的可加性，精确成立，不是近似**。所以「把此前的部分和整体乘 α」等价于「把它们全部换算到新基准重算一遍」，一个乘法顶一遍重算。
2. 于是 $\ell_{t+1}=\ell_t\alpha+e^{s-m_{t+1}}$，$a_{t+1}=a_t\alpha+e^{s-m_{t+1}}v$。——两个累加器用同一个 α，所以只需要维护一个标量 α，不需要按元素记账。
3. 终值 $o=a_N/\ell_N$。——除法**只在最后做一次**。中途每步都除会引入 $N$ 次舍入，也白白多出 $N$ 次除法；延迟到最后一除是纯赚（v0 的 `acc[j] / l`）。

**为什么必须减 max**。数学上 softmax 对所有 $s_j$ 同时平移不变，减 max 不改结果。数值上，$e^x$ 在 fp32 里的参数一旦超过约 88 就溢出到 inf，而 $s$ 的量级随 $D$ 与输入分布自由增长。减掉当前 max 之后，指数参数恒 $\le 0$、结果恒落在 $(0,1]$——这不是「把溢出阈值推远一点」，是**把溢出面整个消掉**。仓内把这条记成面试点（fa2_v0.cu：11-13）。

**为什么 α 也不会出问题**：$m$ 单调不减 ⇒ $m_t-m_{t+1}\le 0$ ⇒ $\alpha\le 1$。重缩放只会让历史部分和变小，永远不会放大——上溢在 $\ell$ 和 $a$ 两个累加器上同样不存在。代价是 $\alpha$ 可能极小（某个 tile 出现远大的 max 时），历史贡献被正确地压成接近 0，这是数学上应该的，不是精度损失。

**参考实现故意不用在线法**。正确性 gate 的对照是两遍法：pass0 扫全行求 max，pass1 定基 exp/求和/加权（ref_naive.cu：4-7）。理由写在文件头：两遍法**没有 m/l/α 的 rescale 环节**，数值路径与控制路径都与被测的在线单遍法独立。若参考也写成在线法，一个 α 公式写错会两边同错、gate 全过——对照物的价值在于它错得跟你不一样。

#### 3.1.1 三件套的每一步分别是恒等、近似还是数值选择

把 §3.1 的推导按「证据类型」重新标一遍，这是本节最该带走的东西：

| 步骤 | 类型 | 依据 / 误差量级 |
|---|---|---|
| $e^{s_j-m_{t+1}}=e^{s_j-m_t}\cdot\alpha$ | **数学恒等** | 指数的可加性，在实数上精确；在 fp32 上引入一次乘法的舍入，相对误差 ≤ $2^{-24}$ |
| $\ell_{t+1}=\ell_t\alpha+e^{s-m_{t+1}}$ | **数学恒等** | 同上；累加顺序改变，但和的值在实数上不变 |
| 减 max | **数值选择**（数学上不变） | softmax 对平移不变，是恒等；但它把 $e^x$ 的参数从无界压到 $(-\infty,0]$，消掉上溢面 |
| 分母只在最后除一次 | **数值选择**（数学上不变） | 少 $N-1$ 次除法与 $N-1$ 次舍入 |
| $S$ 存 fp16(v4) | **有损** | 相对误差 ≤ $2^{-11}$，实测未推高最终误差（§5.3） |

**只有最后一行是有损的**，而 FA 的算法本体（前四行）全部无损。所以「FlashAttention 是精确注意力」这句话可以严格成立——论文标题里的 "Exact" 不是宣传语。

#### 3.1.2 一个容易被跳过的边界:第一个 tile

$m$ 的初值是哨兵 $-10^{30}$（fa2_v2.cu：71 的 `m_s[tid] = -1e30f`）。第一个 tile 处理时，$\alpha = e^{m_{\text{init}} - m_1} = e^{-10^{30} - m_1}$——**指数参数是一个绝对值极大的负数，`__expf` 返回 0**，于是 $\ell_1 = 0\cdot\alpha + \text{sum} = \text{sum}$，$O_1 = 0\cdot\alpha + PV = PV$。**哨兵与「乘 α」这条规则恰好自洽，不需要为第一个 tile 写特例分支**。

但这里藏着一个真实的边界：若某一行的所有可见键都被 causal 屏蔽（`jend <= 0`），`rmax` 保持 $-10^{30}$，`mn` 也是 $-10^{30}$，`alpha = __expf(0) = 1`，`sum = 0`，于是 $\ell$ 保持 0，写回时会做 $0/0$。本梯不会触发它：block 的 `q0` 行基址与 tile 的 `n0` 起点都是 64 的倍数，而 causal 的 `nlimit = min(q0+BM, S)` 保证每个 block 至少能看到自己那一段对角块，每行至少可见自己（`jend >= 1`，因为 `row + 1 - n0 >= 1`）。**换成非整除的 S 或者带 window 的 causal 变体，这条前提就要重新检查**——这是「目标是归因不是产品化」这句边界声明的一个具体内容。

#### 3.1.3 论文的第三个改动,以及本仓做到了哪几个

FlashAttention-2(arXiv:2307.08691，§3.1)对在线 softmax 提了两处算法级修改，理由是 GPU 上非矩阵乘法运算比矩阵乘法贵得多——论文的原话是 "each non-matmul FLOP is 16× more expensive than a matmul FLOP"，并举 A100 为例："312 TFLOPs/s of FP16/BF16 matmul， but only 19.5 TFLOPs/s of non-matmul FP32"。两处修改是：

1. **不在每一步除以 $\ell$**，而是维护一个未归一化的 $\tilde O$，只在最后做一次 $\mathrm{diag}(\ell)^{-1}$。
2. **只存 logsumexp $L = m + \log \ell$**，而不是分别存 $m$ 与 $\ell$。

**本仓的对照**：第 1 条**已经做到**——`l_s` 全程只累加，除法在写回时一次完成（fa2_v2.cu：168）。第 2 条**没有做**：v2/v3/v4 分别保留 `m_s`、`l_s`、`a_s` 三个 float[64] 数组，合计 768 B（§3.3 的分区表）。做了能省多少？把 m 与 l 合成 L 可省 256 B，而 v2 的 smem 余量是 99 − 90.75 = 8.25 KB——**256 B 在这个预算里毫无意义**，所以不做是对的。这是一个「论文的优化在本场景下不值得做」的例子，值得原样保留而不是盲从。

不过第 1 条的对照还有一层：FA2 省掉的是**每步的除法**，但它仍然保留了每步对 $O$ 乘 $\alpha$ 的重缩放。本梯的 ④ 段做的正是这个乘 $\alpha$，**它省不掉**——省掉它就要在最后重算所有历史项的基准，那才是真正的重算。所以 ④ 段的存在不是本梯的实现问题，是算法本身要求的；**本梯的问题是它必须在 shared memory 上做，而不是在寄存器上做**(§3.2)。

### 3.2 架构税的根:fragment 不知道自己属于哪一行

讲义 01 §3.2 列过 wmma fragment 的三条性质，这里只用第 2、3 条：

- **性质 2**：lane → 元素的映射是编译器/架构私有的，你不能问「我这个 lane 拿的是第几行第几列」。
- **性质 3**：同 shape 的 accumulator fragment 在同架构上映射一致，所以对两个同 shape fragment 做**同位置**的逐元素运算是合法的。

把这两条对着 FA2 的三个环节看：

| 环节 | 需要什么 | wmma 能不能做 |
|---|---|---|
| $S$ 的行级 max / 求和 | 沿**行**规约，必须知道行号 | 不能（性质 2） |
| $P=\exp(S-m)$ | 逐元素，但要减去**按行**的 $m$ | 不能（仍要行号） |
| $O \leftarrow \alpha O$ | 逐元素乘，但 α **按行** | 不能（仍要行号） |
| $O \leftarrow O + PV$ | 同 shape accumulator 逐元素加 | **能**（性质 3，fa2_v2.cu：159-160） |
| $S$(fp32)→ 存 fp16 | 同 shape 逐元素转换 | **能**（性质 3，fa2_v4.cu：123-126） |

结论很干净：**凡是「沿行」的操作，wmma 一律做不了；凡是「同位置逐元素」的操作，wmma 都能做。** FA2 的 softmax 恰好三样全是沿行的，GEMM 的 epilogue 恰好全是逐元素的——这就是同一套工具箱两种结局的机制解释，不是「FA2 更难写」这种感觉话。

代价具体化为三条往返：$S$ 必须 `store_matrix_sync` 落 smem 交给标量段；$P$ 必须由标量段写进 smem 再 `load_matrix_sync` 读回 fragment；$O$ 因为要按行 ×α，只能整块驻 smem，每 tile 读出改写再写回。三条往返把「标量段」硬插进 Tensor Core 的中间，相位链就是这么长出来的。

#### 3.2.1 「不公开」这件事的文档原文

这条结论不是从行为反推的，官方文档正面写了。CUDA C++ Programming Guide §10.24.1 在 `fragment` 的定义处写：

> "An overloaded class containing a section of a matrix distributed across all threads in the warp. **The mapping of matrix elements into fragment internal storage is unspecified and subject to change in future architectures.**"

紧接着在 `mma_sync` 之后给出「那你能做什么」：

> "Because the map of matrix elements into each thread's fragment is unspecified, individual matrix elements must be accessed from memory (shared or global) after calling `store_matrix_sync`. **In the special case where all threads in the warp will apply an element-wise operation uniformly to all fragment elements**, direct element access can be implemented using the following fragment class members."

两句话合起来就是本篇的全部前提：**「必须从内存访问」是默认规则，「逐元素统一运算」是唯一的例外**。FA2 的行级 max / exp / ×α 三样都不是逐元素统一运算（它们对不同的行用不同的标量），所以三样都落进默认规则，三样都要经内存。

**本仓有一处依赖比文档明文更进一步**：fa2_v2.cu：160 的 `pv.x[e] += oacc.x[e]` 把两个同 shape 的 accumulator fragment 逐元素相加。文档的例外条款说的是「对所有 fragment 元素施加统一的逐元素运算」，示例里的 alpha 是一个全 warp 相同的标量；两个 fragment 之间的对位相加需要额外假设「同 shape、同类型的 fragment 使用同一套（未指定的）映射」。这个假设在实践中成立（否则 `mma_sync(d,a,b,c)` 里 C 与 D 的对应关系本身就无从定义），本仓全 shape gate 通过（err 4.88e-04，EXP-K03 §5）是它成立的间接证据，**但按证据分级只能记为「实践依赖，文档未明文」**——讲义 01 §3.2.2 对同一条依赖有相同的标注。

#### 3.2.2 为什么 GEMM 不疼而 FA2 疼:一句话判据

把两个算子的 epilogue 并排看：

| | GEMM 的 epilogue | FA2 的每 tile 后处理 |
|---|---|---|
| 要做什么 | $C \leftarrow \alpha(A B) + \beta C$，逐元素 | 行级 max、行级求和、行级 ×α |
| 需要位置信息吗 | 不需要（$\alpha,\beta$ 全局标量） | **需要行号** |
| 能否在 fragment 上做 | 能（文档例外条款） | 不能 |
| 后果 | accumulator 全程驻寄存器 | $S$、$P$、$O$ 三次往返 shared memory |

**判据：看这个后处理需不需要「我是第几行」。** 这一句就能在写代码之前预判一个算子会不会被 wmma 卡住，不必等跑出来才发现。按这个判据，layernorm、softmax、top-k、行级 argmax 全部会被卡住；bias 加法、激活函数、类型转换、residual 相加全部不会。

### 3.3 smem 分区表逐字段推:90.75KB 与 opt-in 的因果链

v2 的分区表（fa2_v2.cu：27-47，§4.2 逐字引用）每一行都能独立推：

| 区 | 形状 | 字节 | 为什么是这个类型/形状 |
|---|---|---|---|
| Osm | float[64][128] | 32768 | 跨 tile 的累加器，且要按行 ×α ⇒ 必须驻 smem；fp32 是因为它要吃几十个 tile 的累加，fp16 尾数 10 位扛不住 |
| Ssm | float[64][68] | 17408 | wmma accumulator 原生 fp32，直接 store 不转换；行宽 68 见下 |
| m/l/a | float[64]×3 | 768 | 一行一份；α 落 smem 是因为写它的是 softmax 线程、读它的是重缩放线程，跨线程 |
| Ks | half[64][128] | 16384 | 一个 K tile |
| Vs | half[64][128] | 16384 | 一个 V tile |
| Psm | half[64][72] | 9216 | 它是 ⑤ 的 A 操作数，wmma 的 matrix_a 必须是 half ⇒ 存 half；行宽 72 见下 |

合计 $32768+17408+768+16384+16384+9216 = 92928$ B $= 90.75$ KiB——与源码里 `SMEM_BYTES = 92928` 逐字节相符，而且六个 `OFF_*` 常量就是这张表的前缀和（50176 / 50944 / 67328 / 83712 恰好逐项落位），表和代码互为校验。

**行宽为什么不是 64**。smem 有 32 个 bank，每 bank 宽 4B，一圈 128B。`Ssm` 若按 64 float 一行，行跨距 $64\equiv 0 \pmod{32}$，同一列的不同行落进同一个 bank；而 wmma 的 $16\times16$ store 恰恰是列向散布的，32 个 lane 全撞一个 bank，退化成 32 次串行访问。+4 float 的错位打散了这个模式。**为什么不是 +1(65)**：65 float 的行首偏移是 260B，不再是 16B 的倍数——float4 装载与 wmma 的对齐前提当场破坏。所以 68 =「打散 bank」与「保住 16B 对齐」两个约束下的最小值；`Psm` 是 half 粒度，16B = 8 half，同理取 72。

**90.75KB 逼出的两件事**：

1. **必须 opt-in**。静态 `__shared__` 数组的上限是 48KB，这是编译期硬上限，不是可以调的参数。超过就只能走 `extern __shared__` + `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` 显式放行，Ada 每 block 的上限是 99KB(fa2_v2.cu：27-29)。90.75 < 99，通过——但也只剩 8KB 余量，这个余量在 §3.6 会直接决定「V 能不能双缓冲」。
2. **1 block/SM**。Ada 每 SM 的 smem 是 100KB，90.75KB 一块就占掉九成，第二个 block 放不下。于是 SM 上只剩 block 内部的 warp 可以互相遮蔽延迟——v2 的 4 warp 只有 $4/48 = 8.3\%$ 理论 occupancy。**这一条直接决定了 v3 的形态**：加 block 不是选项，唯一的旋钮是加 warp。

**v4 的重排账**。v4 把 S 与 P 合并成一块 half 缓冲 SP（exp 原位改写）：省掉 float 的 Ssm 17408B 与独立 Psm 9216B，换来 SP 9216B，净省 17408B；这 17KB 拿去给 K 开双缓冲（+16384B）。合计 $32768+768+9216+32768+16384=91904$ B $=89.75$ KiB——**开了双缓冲总量反而少 1KB**。精度代价过 gate（err 4.88e-04，与 v2/v3 同值，EXP-K03 §5），机理上说得通：$P$ 在 v2/v3 里本来就是 half，v4 只是把它前面那一步的 $S$ 也降到 half，而 $S$ 紧接着就要过 exp 再存成 half，新引入的相对误差与既有的 $P$ 存储误差同阶，没有多出一个量级。注意这是**实测通过**，不是理论保证：换分布、换 $D$ 要重验。

#### 3.3.1 LDS=68 / LDP=72 不是启发式,是被文档逼出来的唯一解

原文里「+4 是保住 16B 对齐的最小错位」这句话是对的，但它把一条硬约束说成了经验。**官方文档把这条写死了**（CUDA C++ Programming Guide §10.24.1，`load_matrix_sync` 与 `store_matrix_sync` 条）：

> "mptr must be a **256-bit aligned** pointer pointing to the first element of the matrix in memory. **ldm** describes the stride in elements between consecutive rows (for row major layout) or columns (for column major layout) and **must be a multiple of 8 for `__half` element type or multiple of 4 for `float` element type**. (i.e., multiple of 16 bytes in both cases)."

于是两个魔法数变成一道只有一个解的算术题：

- `Ssm` 是 float，`ldm` 必须是 4 的倍数；要打散 bank 就必须 $\not\equiv 0 \pmod{32}$（32 个 bank，每 bank 4 B，所以行跨距是 32 的倍数时同一列的不同行全撞同一 bank）。**大于 64 且是 4 的倍数、且不是 32 的倍数的最小值 = 68**。65、66、67 全部违反 `ldm` 是 4 的倍数这一条，**不是「对齐更差」，是文档明文禁止**。
- `Psm` / `SP` 是 half，`ldm` 必须是 8 的倍数；同理要求不是 16 的倍数（half 粒度下 32 bank × 4 B = 64 half 一圈，行跨距 64 时同列全撞；取 72 使每行错开 8 half = 32 B = 8 个 bank）。**大于 64 且是 8 的倍数的最小值 = 72**。

**顺带修正一个措辞**：原文说「LDS = 68 是保住 16B 对齐的最小错位」，更准确的说法是「68 是同时满足 `ldm` 合法与 bank 错位的最小值」——16 B 对齐是 `ldm` 合法性的等价表述，不是一个独立的额外条件。

#### 3.3.2 指针对齐核对:C1 那一条真的满足吗

`ldm` 只是两条约束里的一条，另一条是**指针 256-bit（32 字节）对齐**——比常说的 16 B 更严。把 v2 的每一处 wmma 访问逐个核一遍（**本讲义核对**；区基址 = §3.3 表里的 `OFF_*`）：

| 调用点 | 区基址 | 区内字节偏移 | 总偏移 | 32 的倍数？ |
|---|---|---|---|---|
| `store → Ssm[(warp*16)*LDS + n*16]`(fa2_v2.cu:110) | 32768 | $4\times(1088\,\text{warp} + 16n) = 4352\,\text{warp} + 64n$ | 32768 + 上式 | 32768/32=1024,4352/32=136,64/32=2，全是 |
| `load ← Psm[(warp*16)*LDP + kk*16]`(fa2_v2.cu:151) | 83712 | $2\times(1152\,\text{warp} + 16kk) = 2304\,\text{warp} + 32kk$ | 83712 + 上式 | 83712/32=2616,2304/32=72,32/32=1，全是 |
| `load ← Ks[(n*16)*FA_D + kk*16]`(fa2_v2.cu:105) | 50944 | $2\times(2048n + 16kk) = 4096n + 32kk$ | 50944 + 上式 | 50944/32=1592，全是 |
| `load ← Vs[(kk*16)*FA_D + c*16]`(fa2_v2.cu:152) | 67328 | $2\times(2048kk + 16c) = 4096kk + 32c$ | 67328 + 上式 | 67328/32=2104，全是 |
| `load/store Osm[(warp*16)*FA_D + c*16]`(fa2_v2.cu:157-161) | 0 | $4\times(2048\,\text{warp} + 16c) = 8192\,\text{warp} + 64c$ | 同左 | 全是 |
| v4 的 `SP[(wr*16)*LDSP + nc*16]`(fa2_v4.cu:127) | 33536 | $2304\,wr + 32nc$ | 33536 + 上式 | 33536/32=1048，全是 |

**六处全部满足，而且不是碰巧**：六个 `OFF_*` 常量本身都是 32 的倍数（0 / 32768 / 50176 / 50944 / 67328 / 83712，以及 v4 的 0 / 32768 / 33536 / 42752 / 75520），而 wmma 只以 16 行、16 列为基址取块，行跨距 68 float = 272 B、72 half = 144 B 分别乘 16 之后是 4352 与 2304，都被 32 整除。**做这个核对的价值在于：一旦有人改动分区表（比如为了塞进新的统计量把某个 `OFF_*` 加了 4 B），C1 就会静默失效，而失效的表现是数据错位而不是报错。** 这也是为什么 §4.2 强调三个常量必须同步改。

#### 3.3.3 opt-in 的文档条款与它的两个后果

「超过 48 KB 必须动态 smem + 显式放行」这条同样有明文（CUDA C++ Programming Guide §20.7.3 Shared Memory，compute capability 8.6/8.9 段）：

> "Devices of compute capabilities 8.6 and 8.9 allow up to 99 KB of shared memory. **Kernels relying on shared memory allocations over 48 KB per block are architecture-specific, and must use dynamic shared memory rather than statically sized shared memory arrays. These kernels require an explicit opt-in by using `cudaFuncSetAttribute()` to set the `cudaFuncAttributeMaxDynamicSharedMemorySize`**."

同一节还解释了 99 与 100 的那 1 KB 差额去哪了：

> "Note that the maximum amount of shared memory per thread block is smaller than the maximum shared memory partition available per SM. **The 1 KB of shared memory not made available to a thread block is reserved for system use.**"

**这一句解释了为什么余量是 8.25 KB 而不是 9.25 KB**：99 − 90.75 = 8.25 KB。§3.6 里「V 再开一份双缓冲需要 16 KB，放不下」这个判断，用的正是 99 这个上限而不是 100。

第二个后果是 1 block/SM：每 SM shared memory 100 KB，一个 block 占 90.75 KB，第二个放不下。**注意这里 100 与 99 的区别再次生效**：即使 block 只用 49 KB，两个 block 也是 98 KB < 100 KB，能放下；是 90.75 这个数字把它钉死的。于是 SM 上只剩 block 内部的 warp 可以互相遮蔽延迟——v2 的 4 warp 只有 $4/48 = 8.3\%$ 理论 occupancy。**这一条直接决定了 v3 的形态**：加 block 不是选项，唯一的旋钮是加 warp。

### 3.4 相位链五段与每个 barrier 守的竞态对象

v2/v3/v4 每个 tile 都走同一条五段链（fa2_v2.cu：10-13）：

```
      ┌─(loop-top barrier · WAR)
① K/V 装载 ──(barrier · RAW)── ② QK^T(wmma)→ S 落 smem ──(barrier · RAW)──
③ 标量段行级 softmax(S→P,更新 m/l/α)──(barrier · RAW)── ④ O ×α 重缩放
──(barrier · RAW)── ⑤ P·V(wmma)累加进 O ──┘
```

逐个 barrier 说清守的是谁对谁：

1. **loop-top(WAR)**：上一轮 ⑤ 还在读 Vs 与 Psm，本轮 ① 就要覆盖 Ks/Vs。写方是全 block 的装载分片，读方是上一轮的 wmma warp，两者线程集不重合。删掉它：上一轮慢 warp 读到本轮的新 V，结果错而不崩。v4 这一处更险——SP 合一之后，本轮 ② 写 S 会覆盖上一轮 ⑤ 正在读的 P，于是这一个 barrier 同时守两块（fa2_v4.cu：101-102 注为 WAR×2）。
2. **①→②(RAW)**：装载按线性 tid 分片（`for t = tid; t < BN*FA_D/8; t += blockDim.x`），消费按 warp 分块——你搬的那一段多半不是你要读的那一段，必须靠 barrier 让别人的写对你可见。
3. **②→③(RAW)**：第 $r$ 行的 $S$ 由 warp $r/16$ 写（v2）或 warp $(r/16)\times 2 + \text{半区}$ 写（v3/v4），由线程 $tid=r$(v2)或 $tid=2r,2r{+}1$(v3/v4)读——跨 warp 的写读关系。删掉它：softmax 读到半块 $S$，max 与分母全错。
4. **③→④(RAW)**：α 由 softmax 线程写进 `a_s`，由重缩放段的**别的**线程读（重缩放按线性 tid 分片，`a_s[i / FA_D]` 取行号）。同一个 barrier 也保证 $P$ 对 ⑤ 可见。
5. **④→⑤(RAW)**：重缩放按线性 tid 分片写 Osm，⑤ 按 warp 行条带 `load_matrix_sync` 读旧 O——又是线程集不重合。**顺序不能反**：$O_{\text{new}}=\alpha O_{\text{old}}+PV$，若 ⑤ 先加再缩放，历史项会被多乘一次 α。

循环之外还有一次收尾 barrier(fa2_v2.cu：164)：末轮 ⑤ 的 O 写（按 warp）要对写回段（按线性 tid 读全 Osm）可见。

**能删掉哪几个**。诚实答案是一个都删不掉，但理由分两类：第 2、3、4、5 条是「写者线程集 ≠ 读者线程集」，这是分工方式决定的硬约束；第 1 条是缓冲复用决定的，给 K/V 各开一份双缓冲就能去掉——但 §3.3 的账已经说了 smem 只剩 8KB 余量，K 一份双缓冲就要 16KB。所以这条链的长度不是懒，是 smem 预算与 fragment 不透明两个约束的交点。

**这条链值多少**。做一笔每 tile 的 smem 流量账（v2,4 warp；单位 B，**账面推断**，NCU 计数器在本容器不可用）：

| 段 | 内容 | 字节 |
|---|---|---|
| ① | 写 Ks + Vs | 32,768 |
| ② | wmma 读 K（4 warp × 32 个 16×16 half fragment） | 65,536 |
| ② | store S | 16,384 |
| ③ | 读 S 两遍（求 max 一遍、exp 一遍） | 32,768 |
| ③ | 写 P | 8,192 |
| ④ | 读 O + 写 O | 65,536 |
| ⑤ | wmma 读 P | 65,536 |
| ⑤ | wmma 读 V | 65,536 |
| ⑤ | 读旧 O + 写新 O | 65,536 |
| | **合计** | **417,792 ≈ 408 KB** |

同一个 tile 的有效计算是两个 $64\times64\times128$ 的矩阵乘 $=2{,}097{,}152$ FLOP，于是 **5.0 FLOP / smem 字节**。对照讲义 01 的 gemm v4：每个 K 块写 16,384 + 读 49,152 = 65,536 B 喂 1,048,576 FLOP，**16 FLOP / 字节**。差 3.2 倍。

再把账拆一次：上表中**只因为 fragment 不透明才存在**的部分是 store S(16,384)+ 读 S(32,768)+ 写 P(8,192)+ 读 P(65,536)+ O 的 ×α 读写（65,536）+ O 的累加读写（65,536）= 253,952 B，占 61%。mma 路线上这六项全部消失（$S$、$P$、$O$ 都留在寄存器），剩下的 163,840 B 对应 **12.8 FLOP/字节**——回到 GEMM 的同一档。这就是「架构税」在字节上的形状。再次强调这是账面推导，不是计数器实测。

### 3.5 v3 的并行组织:被钉在 1 block/SM 之后唯一的旋钮

v2 的 128 线程在 1 block/SM 的机器上明显吃不满。v3 的唯一变量是并行组织，算法与 smem 布局逐字节不变（fa2_v3.cu：7-10、21）——这是控制变量设计，收益大小才可归因。

8 个 warp 按 `wr = warp/2`（4 个 16 行条带）× `wc = warp%2`（2 个列半区）排布：

- **②**：每 warp 算 $16\times32$（v2 是 $16\times64$），列块号 `nc = wc*2 + n`。
- **③**：每行 2 个线程，`row = tid/2`、`hf = tid%2`，各扫 32 列。**分工方式照顾了原语可用性**：偶/奇 tid 相邻 ⇒ 同 warp 的相邻 lane ⇒ 两半区的 max 与分母增量可以用 `__shfl_xor_sync(..., 1)` 合并，不必再过一次 smem 和 barrier。若改成「前 64 个线程管左半、后 64 个管右半」，搭档就跨 warp 了，shfl 用不了，得多一次 smem 往返加一次 barrier——同样的并行度，更长的链。
- **⑤**：每 warp 算 $16\times64$（v2 是 $16\times128$），输出列块号 `c = wc*4 + c0`。
- **代价**：同一行条带的两个 warp 各持一份**相同的** `af[8]`，这是寄存器冗余（95 reg/thr × 256 thr = 24,320 个寄存器/block，仍远低于 65,536 上限）。用寄存器冗余换掉跨 warp 共享 Q 所需的同步，是划算的（fa2_v3.cu：55-56）。

注意 §3.4 的流量账在 v3 下**逐字节不变**：每 warp 少读一半 fragment，warp 数翻倍，总量相同。所以 v3 的 +33% 是纯粹的「同样的活分给更多 warp 干」，不掺访存变化——这正是控制变量设计想要的干净归因。

理论 occupancy 从 8.3% 到 16.7%，但**不要把 +33% 归给 occupancy 这个数**：真正发生的是 ③ 段从 64 个线程干活变成 128 个、②⑤ 段的 wmma 工作摊到 8 个 warp。对照讲义 01 的 gemm v4（occupancy 33% 全梯最低却最快）：同一个指标两种读法，判据永远是「延迟是否已被遮蔽」，不是「线程够不够多」。

#### 3.5.1 论文怎么划分 warp,本梯怎么划分,以及为什么不一样

FlashAttention-2 用一整节讲 warp 之间怎么分活（arXiv:2307.08691，§3.3 "Work Partitioning Between Warps"），结论是从 split-K 换成 split-Q：

- **split-K（FlashAttention-1 的做法）**：把 $K$、$V$ 切给 4 个 warp，$Q$ 全 warp 共享。论文指出它的代价是 "all warps need to write their intermediate results out to shared memory， synchronize， then add up"——每个 warp 只算出部分的 $QK^\top$，必须经 shared memory 规约。
- **split-Q（FlashAttention-2 的做法）**：把 $Q$ 切给各 warp，$K$、$V$ 全 warp 共享。每个 warp 独立算完自己那几行的 $QK^\top$ 与 $PV$，**不需要跨 warp 规约**。

**本梯的位置**：v2 是纯 split-Q（每 warp 拿 16 行 $Q$、读全部 64 列 $K$，fa2_v2.cu：98 的 `for (int n = 0; n < 4; ++n)` 扫满 4 个 16 列块），与 FA2 一致。v3 把并行度翻倍时，行方向只有 64 行、切成 4 条 16 行带就到头了，**要再翻一倍只能往列方向切**——于是 v3 变成「4 条行带 × 2 个列半区」的**混合划分**，列方向那一刀正是论文警告的 split-K。

那 v3 为什么没被论文说中的代价咬到？**因为它把规约的粒度压到了 warp 以内**：`row = tid / 2, hf = tid % 2`(fa2_v3.cu：93)让同一行的两个搭档成为**同一个 warp 的相邻 lane**，于是「合并两个半区的 max 与分母」用一条 `__shfl_xor_sync(..., 1)` 就完成（fa2_v3.cu：100、110），既不写 shared memory 也不加 barrier。论文批评的是「写 smem + 同步 + 相加」这三件事，v3 一件都没做。

**代价换到了别处**：同一行带的两个 warp 各持一份**相同的** `af[8]`（fa2_v3.cu：55-57 的注释写明这是有意的寄存器冗余）。按每 fragment 每 lane 8 个 half = 4 个 32-bit 寄存器计，8 个 fragment = 32 个寄存器/线程被复制了一份；ptxas 报告 v3 用 95 reg/thr、256 thr = 24320 个寄存器/block，仍远低于 65536 的上限（EXP-K03 §5）。**在寄存器有余量、smem 没余量的场景里，用寄存器冗余换掉一次 smem 往返是划算的**——这正是 FA2 论文那条建议在不同资源约束下的正确变形，而不是违背。

**这一段值得单独记住的方法论**：论文给的是「在它的约束下最优的划分」，不是「任何情况下都要照抄的划分」。照抄 split-Q 到底（只切行）在本梯会卡在 4 warp，而 4 warp 在 1 block/SM 的机器上吃不满——v3 实测 +33% 就是这个约束的价格。**先问自己的约束是什么，再决定论文的哪一条适用。**

#### 3.5.2 +33% 该归给什么:三个候选与排除

理论 occupancy 从 8.3% 到 16.7%，但**不要把 +33% 归给 occupancy 这个数**。三个候选：

1. **occupancy 翻倍**(8.3% → 16.7%)：这是一个结果不是原因，而且 16.7% 在绝对值上仍然很低——如果 occupancy 是主因，16.7% 不该够。
2. **③ 段的并行度翻倍**：v2 的 softmax 是「一行一线程」，128 个线程里只有 64 个在干活（fa2_v2.cu：115 的 `if (tid < BM)`）；v3 是「一行两线程」，256 个线程里有 128 个在干活。**这一段的绝对工作量不变（还是 64×64 个 exp），但干活的线程翻倍**。
3. **②⑤ 段的 wmma 工作摊到 8 个 warp**：每 warp 的 fragment 装载与 mma 次数减半，总量不变。

**§3.4 的流量账在 v3 下逐字节不变**——每 warp 少读一半 fragment，warp 数翻倍，总量相同。所以 +33% 里没有访存变化的成分，是纯粹的「同样的活分给更多 warp 干」。候选 2 与 3 无法用本梯的数据分开（要分开需要一个「只改 ③ 段分工、②⑤ 保持 4 warp」的对照臂，本梯没做，**账面推断**）。可以确定的是候选 1 不是原因：**occupancy 是这次改动的副产品，不是它的机制**。

对照讲义 01 的 gemm v4（occupancy 33% 全梯最低却最快）：同一个指标两种读法，判据永远是「延迟是否已被遮蔽」，不是「线程够不够多」。**两篇讲义放在一起才构成完整的答案**：GEMM 那边 ILP 已经把发射端喂饱，加线程无用；FA2 这边有一整段标量工作只有一半线程在做，加线程直接有用。

### 3.6 v4 的两组 pipeline:交错等待的正确性论证

v4 同时开两条 cp.async 流（K 双缓冲、V 单缓冲），它们**共用同一个组计数器**。讲义 01 §3.3 的组语义在这里要用严：`__pipeline_commit()` 把此前发出的异步拷贝封成一组，组按 commit 顺序排成 FIFO；`__pipeline_wait_prior(N)` 阻塞到「最新 N 个组之外」的所有组完成。

commit 序列（每线程，`async_tile` 每次调用恰好 commit 一组，fa2_v4.cu：54-64）：

$$G_K(0),\ G_V(0),\ G_K(1),\ G_V(1),\ G_K(2),\ \dots$$

**不变量 I**：轮 $t$ 走到第一个 `wait_prior(1)` 时，在途组恰为 $[G_K(t),\ G_V(t)]$（老在前）。

- **基例** $t=0$：序幕 commit 了 $G_K(0)$(fa2_v4.cu：98)，轮首 commit 了 $G_V(0)$，尚无 wait。成立。
- **归纳**：设轮 $t$ 成立。
  1. 第一个 `wait_prior(1)` 留最新 1 组在途 ⇒ 等掉 $G_K(t)$，剩 $[G_V(t)]$。**等掉的恰是 ② 马上要读的 K tile**；$G_V(t)$ 继续在途，与 ②③④ 三段重叠。
  2. ② 之后 commit $G_K(t+1)$ 进另一缓冲 ⇒ $[G_V(t),\ G_K(t+1)]$。
  3. ④ 之后的 `wait_prior(1)` 等掉 $G_V(t)$，剩 $[G_K(t+1)]$。**等掉的恰是 ⑤ 马上要读的 V tile**；$G_K(t+1)$ 继续在途，跨过 ⑤ 与下一轮开头。
  4. 轮 $t+1$ 首 commit $G_V(t+1)$ ⇒ $[G_K(t+1),\ G_V(t+1)]$，I 在 $t+1$ 成立。∎

于是两个 `wait_prior(1)` 各取所需，而参数都是 1——**能这么写完全依赖「K、V 每轮各恰好一组、严格交替」这个节奏**。两条流各自拿到三段的 DMA 时间：$G_V(t)$ 覆盖 ②③④，$G_K(t+1)$ 覆盖 ③④⑤。

**末轮**：不发 $G_K(t+1)$（否则越界读），所以第二个 wait 时在途只有 $[G_V(t)]$，参数必须降为 0 才是「清空」。写成 1 的话，「留最新 1 组在途」留下的正是 $G_V(t)$，⑤ 读到没搬完的 V——**结果错而不崩，且只在特定时序下错**。

**为什么 V 不双缓冲**。两个理由，第二个是硬的：①$V$ 的消费点唯一（⑤），而 $G_V(t)$ 已有 ②③④ 整段重叠窗口，再开一份缓冲换不来新的重叠；②smem 已用 89.75KB，再加 16KB 是 105.75KB，**超过 Ada 每 block 99KB 的上限**，`cudaFuncSetAttribute` 直接失败——这不是「不划算」，是「放不下」。

**危险面比 gemm v3 高一级**。gemm v3 是单流双缓冲，组计数只服务一条流；v4 是两条流共用一个计数器，任何一处多发或漏发一组，两个 `wait_prior` 会**同时**指错对象。特别注意 `__pipeline_*` 是**每线程**语义：若在 ③ 之类的分支里条件性地发 cp.async（比如只让 `tid < 128` 发），不同线程的组计数就此分叉，wait 之后再 barrier 也救不回来。`async_tile` 的循环 `for (t = tid; t < BN*FA_D/8; t += nthr)` 让 1024 条 16B 拷贝均摊到 256 线程各 4 条、无分支，正是为了守住这个不变量。

还有一条容易漏的：`wait_prior` 只保证**调用线程自己**发出的拷贝完成，而 tile 是全 block 分片搬的，所以 wait 之后仍必须 `__syncthreads`(fa2_v4.cu：106-107)——这与讲义 01 §4.5 是同一条道理。

#### 3.6.1 把归纳证明改写成可代数字的形式

上面那套「最新 N 组之外」的说法需要在脑子里绕一圈。官方 C 层文档给了一个更好用的等价定义（CUDA C++ Programming Guide §10.28.4.3 Wait Primitive）：

> "Let {0, 1, 2, ..., L} be the sequence of indices associated with invocations of `__pipeline_commit()` by a given thread. **Wait for completion of batches at least up to and including L−N.**"

用这个形式把 v4 的两处等待重算一遍（每线程的批索引，`async_tile` 每次调用恰好 commit 一批，fa2_v4.cu：63）：

| 时刻 | 已 commit 的批（索引） | L | wait 参数 N | 等到第 L−N 批 | 那一批是谁 | 它是谁要读的 |
|---|---|---|---|---|---|---|
| 序幕（fa2_v4.cu:98） | $G_K(0)$ = 0 | 0 |— |— |— |— |
| 轮 0 首（:103） | +$G_V(0)$ = 1 | 1 | 1(:104) | 0 | $G_K(0)$ | ② 的 K tile |
| 轮 0 ②后（:133） | +$G_K(1)$ = 2 | 2 | 1(:165) | 1 | $G_V(0)$ | ⑤ 的 V tile |
| 轮 1 首 | +$G_V(1)$ = 3 | 3 | 1 | 2 | $G_K(1)$ | ② 的 K tile |
| 轮 1 ②后 | +$G_K(2)$ = 4 | 4 | 1 | 3 | $G_V(1)$ | ⑤ 的 V tile |
| … | … | … | … | … | … | … |
| 末轮首 | +$G_V(t)$ = 2t+1 | 2t+1 | 1 | 2t | $G_K(t)$ | ② 的 K tile |
| 末轮 ②后 | 不发 | 2t+1 | **0** | 2t+1 | $G_V(t)$ | ⑤ 的 V tile |

**这张表就是证明**：每一行的「等到第 L−N 批」都恰好落在「那一批是谁」这一列上，而那一列恰好是下一段马上要读的东西。末轮那一行是唯一的特例——因为没有新批推高 L，所以 N 必须从 1 降到 0，否则等到的是第 2t 批（$G_K(t)$，早就完成了），而 $G_V(t)$ 可能还在途，⑤ 会读到没搬完的 V。

**为什么这个形式更不容易出错**：「最新 N 组之外」要求你先数清楚在途组数；「第 L−N 批」只要求你知道自己 commit 了几次。**L 是一个单调递增的计数器，永远不会数错；在途组数是一个会变的量，很容易数错。** 讲义 01 §3.3.3 对 gemm v3 用的是同一套写法。

#### 3.6.2 交错节奏的三个隐含前提,以及它们各自会怎么破

不变量成立依赖三件事，每一件都有具体的破法：

1. **每轮 K、V 各恰好一批，严格交替**。破法：在 ③ 的 `if (tid < 2 * BM)` 分支里放一次 `async_tile`。PTX 文档明写组是每线程的（§9.7.9.26.3.2："creates a new cp.async-group per thread"），于是 `tid < 128` 的线程 L 会比别人多 1，**两组线程的 L−N 指向不同的批**，barrier 也救不回来。
2. **`async_tile` 内部无分支**。它的循环 `for (t = tid; t < BN*FA_D/8; t += nthr)` 让 1024 条 16 B 拷贝均摊到 256 线程各 4 条，**每个线程都进入循环、都执行末尾的 commit**。即使某个线程一条也没发，PTX 也保证 "If there are no uncommitted cp.async instructions then cp.async.commit_group results in an empty cp.async-group"——**空批与满批在计数上等价，所以线程之间仍然对齐**。
3. **两处条件表达式必须字面一致**。fa2_v4.cu：130 的 `if (n0 + BN < nlimit)` 与：165 的 `n0 + BN < nlimit ? 1 : 0` 是同一个条件；改了一处不改另一处，末轮就会等错批。

**危险面比 gemm v3 高一级**：gemm v3 是单流双缓冲，组计数只服务一条流；v4 是两条流共用一个计数器，任何一处多发或漏发一组，两个 `wait_prior` 会**同时**指错对象。

#### 3.6.3 wait 之后仍要 barrier:两条独立理由

`__pipeline_wait_prior` 只保证**调用线程自己**发出的拷贝完成，而 tile 是全 block 分片搬的，所以 wait 之后仍必须 `__syncthreads`(fa2_v4.cu：106-107)。这条有两个互相独立的出处，任一条单独成立都足以要求这次 barrier：

- **可见性**：PTX §9.7.9.26.3.3 明写 "Writes performed by cp.async operations are made visible **to the executing thread** only after ... The completion of `cp.async.wait_group` on the cp.async-group in which the cp.async belongs to"。
- **完成性**：组计数是每线程的（§9.7.9.26.3.2），你的 wait 不约束别人的批。

这与讲义 01 §3.3.4 是同一条道理，不再展开。

### 3.7 28% 逐层拆:哪一层拿到了,哪一层没有

v4 的 34.8±0.12 TFLOPS 对自家 Triton 版的 123 TFLOPS（S=4096,1.119 ms，triton-kernels#EXP-T01《Triton FA2 forward》，**跨 harness，推断级**）= **28%**；对 sdpa-flash 的约 140 TFLOPS = 25%（同样跨 harness，推断级）。把这个差距按优化层次拆：

| 层 | 本梯拿到的 | 证据等级 |
|---|---|---|
| 指令世代（CUDA core → Tensor Core） | v1 5.5 → v2 24.4 TFLOPS,**×4.5** | 实测，3 轮 |
| 并行组织（4 warp → 8 warp） | v2 24.4 → v3 32.5,**+33%** | 实测，3 轮 |
| 访存重叠（K 双缓冲 + V 重叠 + S/P 合一） | v3 32.5 → v4 34.8，**+7.1%** | 实测，3 轮 |
| 布局控制（fragment 透明 ⇒ 免 smem 往返、免相位链） | **未做**，即差距所在 |— |

第一层说明差距**不在**「用没用 Tensor Core」——那一档收益 FA2 和 GEMM 一样拿满了。第三层是本实验最有信息量的数字：**K/V 访存能藏的都藏了，只值 7.1%**。它的作用是**排除**访存假设，而不是证明相位链假设——这两件事必须分开说。

剩下那 72% 的候选（全部**推断级**，NCU 计数器在本容器不可用，EXP-K01 §7）：

1. **smem 往返流量**：§3.4 的账面 408 KB/tile，其中 61% 是 fragment 不透明逼出来的；mma 路线上这部分消失。
2. **相位链的串行化**：每 tile 5 次 block 级 barrier，S=4096 的 causal 下每个 block 平均走几十个 tile，累计上百次 barrier；每次都把整块拉齐到最慢的 warp。
3. **③ 段 Tensor Core 空转**：softmax 是纯标量段，$64\times64$ 个 exp 全部由 128 个线程串着做，这段时间 Tensor Core 一条指令都发不出去。
4. **1 block/SM**：barrier 期间 SM 上没有别的 block 顶上；gemm v4 至少还有 2 个 block 可以互相错开。

**不能说的是每一层各占多少百分比**。本仓对「smem 往返是差距主因」的定级就是推断，不当实测说（flash-attn/README 的约束表）。检验方式是明确的：v5 走 mma + ldmatrix 重写，差距收窄即证实。

**28% 这个比值本身也带口径风险**：Triton 版是 wall-clock 计时，本仓是 CUDA event；两边都是 100 iters 稳态、ms 级 kernel 差异 <1%(EXP-K03 §7)，所以记为跨 harness 推断级，引用必须带这个限定。真正的解法不是多跑几轮，是同 harness 复测——这与 §8.3.3 里 gemv 的问题是**两类不同的风险**，不要混为一谈。

### 3.8 v5 路线图:mma + ldmatrix 到底换来什么

`mma.sync.aligned.m16n8k16...` 是 PTX 层的 Tensor Core 指令，它与 wmma 的关键差别只有一条：**操作数寄存器的 lane → 元素映射在 PTX ISA 文档里公开且固定**。拿到映射之后：

- $S$ 的 accumulator 布局已知 ⇒ 每个 lane 能算出自己手上元素的行号 ⇒ 行级 max / 求和用 `__shfl_xor_sync` 在寄存器里做，**$S$ 不落 smem**，②→③ 的 barrier 消失。
- $P$ 直接由寄存器里的 exp 结果拼成下一条 mma 的 A 操作数（布局可对齐）⇒ **$P$ 不落 smem**，③→④ 的 barrier 消失。
- $O$ 常驻寄存器，×α 按 lane 自己的行号做 ⇒ **④ 整段消失**，④→⑤ 的 barrier 也消失。
- `ldmatrix.sync.aligned.m8n8.x4.shared.b16` 一条指令按 mma 需要的 lane 布局从 smem 取 8×8 块，替代 `wmma::load_matrix_sync`；配合手工 smem swizzle 消 bank conflict（wmma API 不暴露 swizzle，§3.3 的 68/72 填充是它唯一能用的手段）。

净效果：五段相位链塌成「装载 / 计算」两段，`__syncthreads` 从每 tile 5 次降到 1–2 次，§3.4 里那 61% 的 smem 流量归零。这就是官方 FA2 与 CUTLASS 采用 mma 而非 wmma 的定量理由，也是本仓 v5 技能票的内容——与 gemm 的 v5 是同一张票，共用一次学习成本（EXP-K02《CUDA Tensor Core GEMM 版本梯》§7、EXP-K03 §8），列为后续工作，不阻塞现有结论。

#### 3.8.1 PTX 到底给了什么:m16n8k16 的三张映射表

上面那四条「拿到映射之后就能怎样」的说法，必须落到具体公式上才算数。PTX ISA §9.7.15.5.8 "Matrix Fragments for `mma.m16n8k16` with floating point type" 给的是**可以直接写进代码的算术式**。三个操作数各一张，先记两个共同的辅助量：

```text
groupID           = %laneid >> 2
threadID_in_group = %laneid % 4
```

**累加器 C / D（16×8，fp32，每 lane 4 个元素 c0..c3）**：

```text
row = groupID                                   for ci where i < 2
      groupID + 8                               for ci where i >= 2
col = (threadID_in_group * 2) + (i & 0x1)       for ci where i = {0,..,3}
```

**乘数 A（16×16，fp16/bf16，每 lane 8 个元素 a0..a7，装在 4 个 .f16x2 寄存器里）**：

```text
row = groupID                                   for ai where 0 <= i < 2 || 4 <= i < 6
      groupID + 8                               Otherwise
col = (threadID_in_group * 2) + (i & 0x1)       for ai where i < 4
      (threadID_in_group * 2) + (i & 0x1) + 8   for ai where i >= 4
```

**乘数 B（16×8，fp16/bf16，每 lane 4 个元素 b0..b3）**：

```text
row = (threadID_in_group * 2) + (i & 0x1)       for bi where i < 2
      (threadID_in_group * 2) + (i & 0x1) + 8   for bi where i >= 2
col = groupID
```

**这三张表就是 wmma 拒绝告诉你的东西。** 下面三小节把它们分别用在 FA2 的三个卡点上。

#### 3.8.2 卡点一:行级 max / 求和 变成两条 shfl

看 C 的表：一个 lane 持有的 4 个元素只落在**两行**上——`groupID` 与 `groupID + 8`。反过来问「第 r 行的 8 个列分布在哪些 lane」：需要 `groupID = r`（或 `r − 8`），即 `laneid >> 2 == r`，也就是 **lane $4r$、$4r{+}1$、$4r{+}2$、$4r{+}3$ 这连续四个 lane**；它们的 `threadID_in_group` 分别是 0/1/2/3，`col = 2t + (i&1)` 依次覆盖 {0,1}、{2,3}、{4,5}、{6,7}，**恰好拼满 8 列，不重不漏**。

于是行级规约的算法完全确定（**本讲义推导**）：

1. 每个 lane 先对自己手上属于同一行的元素做局部规约。对一个 16×16 的 $S$ 分块（由两条 `mma.m16n8k16` 拼出，列 0-7 与 8-15），每 lane 每行持有 $2 + 2 = 4$ 个值 ⇒ 3 次标量运算。
2. 跨 lane 合并：`__shfl_xor_sync(mask, v, 1)` 与 `__shfl_xor_sync(mask, v, 2)` **两步**即可把 4 个 lane 合成同值（1 与 2 的 XOR 组合遍历 $4r$..$4r{+}3$）。
3. 结束时，lane $4r$..$4r{+}3$ 每个都持有第 $r$ 行与第 $r{+}8$ 行的完整 max / 分母，**不需要任何 smem，不需要任何 barrier**。

对 FA2 的 64 列 tile（8 条 mma 沿 n 拼出），第 1 步变成 16 个值的局部规约，第 2、3 步不变——**跨 lane 的通信量与 tile 宽度无关，永远是 2 次 shuffle**。对照本梯 v2 的做法：$S$ 整块 `store_matrix_sync` 落 smem(16 KB)、标量段读两遍（32 KB）、一次 barrier。**这就是 §3.4 那笔 61% 流量账里最大的一块凭空消失的原因。**

#### 3.8.3 卡点二:P 不落 smem,因为 C 的布局与 A 的布局逐元素对齐

这是三条里最漂亮的一条，也是最容易被说成「大概能行」的一条。把 C 与 A 的表并排代入，取同一个 lane（同一组 `groupID = g`、`threadID_in_group = t`），把两条 mma 产出的 $S$（列 0-7 与列 8-15）拼成一个 16×16 的块：

| C 的元素 | 它是 $S$ 的哪个位置 | A 的元素 | 它要求的位置 | 一致？ |
|---|---|---|---|---|
| 第 1 条 mma 的 c0 | (g, 2t) | a0 | (g, 2t) | 一致 |
| 第 1 条 mma 的 c1 | (g, 2t+1) | a1 | (g, 2t+1) | 一致 |
| 第 1 条 mma 的 c2 | (g+8, 2t) | a2 | (g+8, 2t) | 一致 |
| 第 1 条 mma 的 c3 | (g+8, 2t+1) | a3 | (g+8, 2t+1) | 一致 |
| 第 2 条 mma 的 c0 | (g, 8+2t) | a4 | (g, 2t+8) | 一致 |
| 第 2 条 mma 的 c1 | (g, 8+2t+1) | a5 | (g, 2t+9) | 一致 |
| 第 2 条 mma 的 c2 | (g+8, 8+2t) | a6 | (g+8, 2t+8) | 一致 |
| 第 2 条 mma 的 c3 | (g+8, 8+2t+1) | a7 | (g+8, 2t+9) | 一致 |

**八个元素，逐位对应，顺序都一样。** 也就是说：把两条 mma 的 fp32 累加器按（c0，c1，c2，c3） → (a0，a1，a2，a3)、(c0'，c1'，c2'，c3') → (a4，a5，a6，a7) 的顺序做 fp32→fp16 转换并打包成 4 个 `.f16x2` 寄存器，得到的**正好**是下一条 `mma.m16n8k16` 需要的 A 操作数，**零跨 lane 数据移动，零 shared memory**。

这就是「$P$ 常驻寄存器」在硬件层面的确切依据（**本讲义推导，依据 PTX §9.7.15.5.8 的三张表**），而不是「官方实现更聪明」这种说法。对照本梯：$P$ 要写 smem(8 KB)、再 `load_matrix_sync` 读回（64 KB，4 warp × 每 warp 多次），外加一次 barrier。

**这条对应关系为什么会存在**：因为 NVIDIA 设计 mma 布局时就考虑了「一条 mma 的输出直接喂下一条 mma 的输入」这种链式用法——注意力恰好是这种链式（QK^T 的输出经逐元素变换后成为 PV 的输入）。**布局公开不只是「让你能看」，它让一整类融合成为可能。**

#### 3.8.4 卡点三:O 的 ×α 与 ldmatrix 的角色

$O$ 是 PV 的累加器，布局同样是 C 的那张表：每个 lane 知道自己的 4 个元素落在 `groupID` 与 `groupID + 8` 两行。而 α 是按行的标量，§3.8.2 已经让每个 lane 手上就有自己那两行的 m/l/α。**于是 `O[e] *= alpha[row(e)]` 是纯 lane 本地运算**——④ 段整个消失，连带 ③→④ 与 ④→⑤ 两次 barrier 一起消失。

最后一块是从 smem 取 K/V 进 fragment。`ldmatrix` 的语义（PTX §9.7.15.5.15）是 "Collectively load one or more matrices across all threads in a warp from ... `.shared` state space"，形状 `.m8n8`、元素 16-bit，`.num` 取 `.x1/.x2/.x4`。关键的一句在于**地址由谁提供**：

> "The eight addresses required for each matrix are provided by eight threads, depending upon the value of `.num` ... Addresses addr0–addr7 correspond to the rows of the first matrix, addresses addr8–addr15 correspond to the rows of the second matrix, and so on."

对应的分配表是：`.x1` 用 Threads 0–7，`.x2` 用 0–7 与 8–15，`.x4` 用满 32 个线程的四组。**「地址由你算」这一点，正是 wmma 与 mma 在 swizzle 上的分界**：`wmma::load_matrix_sync` 只收（指针， ldm），内部按什么模式取数不公开，你没有接口告诉它「我的数据按 XOR swizzle 摆过了」；`ldmatrix` 让 32 个线程各交一个地址，swizzle 就写在算地址那一步里。文档还提醒 "When reading 8x8 matrices， a group of four consecutive threads loads 16 bytes. The matrix addresses must be naturally aligned accordingly."——**16 B 这个数第三次出现**（讲义 01 §4.3 列了另外三处）。

#### 3.8.5 净效果与代价

净效果：五段相位链塌成「装载 / 计算」两段，`__syncthreads` 从每 tile 5 次降到 1–2 次，§3.4 里那 61% 的 smem 流量归零。**这就是官方 FA2 与 CUTLASS 采用 mma 而非 wmma 的定量理由**，也是本仓 v5 技能票的内容——与 gemm 的 v5 是同一张票，共用一次学习成本（EXP-K02 §7、EXP-K03 §8），列为后续工作，不阻塞现有结论。

代价也要说清楚，否则这条路线看起来太便宜：

- **布局是「公开且固定」，不是「随便摆」**。上面三张表意味着你的每一个索引计算都要跟着它走；写错一个 `& 0x1` 不会报错，只会算错。
- **`mma.m16n8k16` 的 n 只有 8**，而 wmma 的 n 是 16。同样覆盖 16 列要发两条指令，循环结构与 fragment 数组的组织都要重写。
- **`.aligned` 与 `.sync` 的要求更严**：PTX §9.7.15.5.15 明写 "The behavior of `ldmatrix` is undefined if all threads do not use the same qualifiers, or if any thread in the warp has exited." 分支里用它比 wmma 更危险。
- **本梯没有做这件事**，所以「v5 能收窄多少」是一个**假设，不是预测**；§3.7 已经说明它的作用是提供一个可证伪的检验路径。

### 3.9 魔法数总账:每个常数由谁决定

与讲义 01 §3.6 同一套分类（理论上界 / 硬件约束 / 实测扫描）。**归不进任何一类的常数，就是还没想清楚的常数。**

| 常数 | 值 | 决定它的是 | 依据 |
|---|---|---|---|
| BM（query 行块） | 64 | 硬件约束 | 4 个 wmma 行块；再大则 Osm(BM×128 fp32)超预算：BM=128 时 Osm 就要 64 KB |
| BN（key 列块） | 64 | 硬件约束 | Ks/Vs 各 BN×128 half；BN=128 时两者共 64 KB，加 Osm 直接爆 99 KB |
| D | 128 | 外部给定 | 协议参数（Qwen 系列 head_dim），不是可调项 |
| LDS | 68 | 硬件约束（唯一解） | `ldm` 须为 4 的倍数（CUDA PG §10.24.1）且行跨距不得 ≡ 0 (mod 32 bank)；最小值即 68(§3.3.1) |
| LDP / LDSP | 72 | 硬件约束（唯一解） | `ldm` 须为 8 的倍数；最小值即 72(§3.3.1) |
| SMEM_BYTES | 92928(v2/v3)、91904(v4) | 理论（前缀和） | 分区表逐字段之和（§3.3）；与 `OFF_*` 互为校验 |
| WARPS | 4(v2)→ 8(v3/v4) | 硬件约束 | 1 block/SM 之下唯一的并行旋钮；8 warp 时 ③ 段每行 2 线程恰好用满 128 个活跃线程（§3.5） |
| `wait_prior` 参数 | 1（末轮 0） | 理论 | §3.6.1 的 L−N 表 |
| K 缓冲深度 | 2 | 硬件约束 | 再加一级需 16 KB，余量只有 99 − 89.75 = 9.25 KB(§3.6) |
| V 缓冲深度 | 1 | 硬件约束 | 双缓冲需 105.75 KB > 99 KB 上限（§3.6） |
| m 的哨兵 | −1e30 | 理论 | 需要一个乘 α 后自动归零的下界；−1e30 使 `__expf` 下溢到 0(§3.1.2) |
| 正确性阈值 | 2e-2 | 理论上界 | fp16 输出的合理界；实测 4.88e-04，余量约 41 倍（EXP-K03 §5） |
| bench iters / warmup | 100 / 20 | 实测扫描 | 3 轮轮间 std 在 1e-2 ms 量级（EXP-K03 §5），统计已够 |
| gate 形状族 | 4 组（见 main.cu：91-94） | 理论 | 每组针对一类独立的错法：非 causal 分支、GQA 1:1/2:1/4:1、S 跨度 |

**唯一一个「没有独立依据」的常数是 v0 的 lane 分工 4 维/lane**：它由 D=128 / 32 lane 直接得出，是被 D 决定的，不是选出来的。

### 3.10 论文/文档怎么说 vs 本项目实测

#### 3.10.1 FA2 论文的 warp 划分 vs 本梯 v3

- **论文说**：split-K 会逼出 "all warps need to write their intermediate results out to shared memory, synchronize, then add up"，所以改用 split-Q(arXiv:2307.08691,§3.3)。
- **本梯**：v2 纯 split-Q；v3 因为行方向只有 64 行、切不出 8 份，被迫在列方向再切一刀（局部 split-K），但把搭档配成同 warp 相邻 lane，用 `__shfl_xor_sync` 替代 smem 规约。实测 +33%。
- **差异来源**：约束不同。论文的场景里 occupancy 有别的来源（可以多开 block），本梯被 90.75 KB smem 钉死在 1 block/SM，warp 数是唯一旋钮。**论文的建议在本梯的约束下需要变形，而不是照抄。**

#### 3.10.2 FA2 论文的两处算法修改 vs 本仓

- **论文说**：(1) 不每步除 $\ell$；(2) 只存 logsumexp。
- **本仓**：(1) 已做（fa2_v2.cu：168 的一次除法）；(2) 未做，因为省下的 256 B 在 8.25 KB 余量里没有意义（§3.1.3）。
- **差异来源**：论文优化的是「非矩阵乘法 FLOP 的条数」（它举 A100 的 312 : 19.5 TFLOPS 之比），本梯的瓶颈不在那里而在 smem 往返与相位链。**同一条优化在不同瓶颈下的价值不同。**

#### 3.10.3 论文的 IO 分析层级 vs 本篇的账

- **论文说**：FlashAttention 的 HBM 访问是 $\Theta(N^2d^2M^{-1})$，并证明这是下界（arXiv:2205.14135，Theorem 2 与 Proposition 3）。
- **本篇**：§3.4 的 408 KB/tile 全部是 **shared memory 层**的流量，HBM 层的流量本梯与官方实现同阶（Q、K、V、O 各走一遍）。
- **差异来源**：**分析层级不同，不是结论冲突**。论文的定理在 HBM 层成立，本梯也满足；架构税发生在论文没有分析的那一层。**这是本篇最想让人记住的一个方法论点：一个模型的最优性只在它建模的那一层成立，换一层就要重新算账。**

#### 3.10.4 官方实现的 wmma 定位 vs 本梯的实测

- **文档说**：CUDA C++ Programming Guide 把 wmma 作为 Tensor Core 的 C++ 接口给出，没有任何一处说它不适合融合算子；fragment 布局不公开这一条也只是平铺直叙地写在 `fragment` 定义里。
- **本梯实测**：同一套 wmma，GEMM 到真 cuBLAS 的 85.6%(EXP-K02 §5)，FA2 到自家 Triton 版的 28%（跨 harness，推断级，EXP-K03 §6）。
- **差异来源**：文档描述的是**能力**，不是**适用性**。「布局不公开」对不需要位置信息的算子是零成本，对需要位置信息的算子是结构性成本——**文档不会替你做这个判断，§3.2.2 的那句判据才会。**

#### 3.10.5 一个方向相反的对照:CUDA PG 的 100 KB vs 实际能用的 99 KB

- **文档说**：每 SM shared memory 100 KB，每 block 99 KB，差额 "The 1 KB of shared memory not made available to a thread block is reserved for system use."(§20.7.3)
- **本梯**：v2 用 90.75 KB，余量按 99 算是 8.25 KB。若误按 100 算成 9.25 KB，可能会误以为「再挤一挤能塞下点什么」。
- **差异来源**：两个上限服务两个不同的问题（「一个 block 最多申请多少」与「一个 SM 最多分配多少」）。§3.6 判定「V 不能双缓冲」用的是前者，§3.3 判定「1 block/SM」用的是后者——**用错哪一个，结论都会错。**

## 4 代码逐段走读(按执行顺序)

阅读约定与讲义 01 §4 一致：每段先说**角色**，再点**关键行**，再补**硬件语义**（这一行为什么必须这么写、依据哪条文档），最后给**改错会怎样**。本篇的硬件语义集中在三处：wmma 的 `ldm` 与指针对齐契约（§3.3.1-§3.3.2）、shared memory 的 bank 结构（§3.3.1）、cp.async 的组语义与可见性（§3.6）——走读时不再重复推导，只指回去。

### 4.1 在线 softmax 三件套的最小形态(flash-attn/src/fa2_v0.cu:40-48)

```cuda
        // 在线更新:新 max → 折算因子 α → 本键权重 p;先乘 α 再累加,
        // 等价于把此前的部分和整体换到新基准
        float mn = fmaxf(m, s), alpha = __expf(m - mn), p = __expf(s - mn);
        l = l * alpha + p;
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            acc[j] = acc[j] * alpha
                     + p * __half2float(v[(size_t)n * FA_D + lane * 4 + j]);
        m = mn;
```

角色：整条版本梯的算法内核，后面四版全部是这九行的重新排布。关键行：`mn/alpha/p` 三个量一次算完，`l` 与 `acc` 用**同一个** α——§3.1 第 2 步的直接落地；`acc[j] = acc[j] * alpha + p * v` 的顺序是「先折算历史、再加新项」，写反了（先加后折算）会把新项也乘一次 α；分母 `l` 全程只累加不做除，除法留到写回（fa2_v0.cu：52）。改错会怎样：把 `alpha = __expf(m - mn)` 写成 `__expf(mn - m)`，α 变成 $\ge 1$ 的放大因子，历史部分和每 tile 放大一次，长序列直接溢出；把 `p = __expf(s - mn)` 里的 `mn` 写成 `m`（旧 max），当 $s>m$ 时指数参数为正，数值上就是本节说的上溢面又被打开。

### 4.2 v2 的 smem 分区表(flash-attn/src/fa2_v2.cu:27-47)

```cuda
// 动态 smem 分区(字节偏移,均 16B 对齐——float4 装载与 cp.async/wmma 的
// 对齐要求;合计 92928B = 90.75KB,超 48KB 静态上限,须动态 smem +
// cudaFuncSetAttribute opt-in,Ada 每 block 上限 99KB):
//   区    类型/形状          字节   用途
//   Osm   float[64][128]    32768   O 累加器:fp32 驻 smem,逐 tile ×α 重缩放
//   Ssm   float[64][68]     17408   QK^T 结果(③ 的读源)
//   m/l/a float[64] x3        768   在线统计:运行 max / 分母 / 本轮 α
//   Ks    half [64][128]    16384   K tile
//   Vs    half [64][128]    16384   V tile
//   Psm   half [64][72]      9216   exp 后的 P(⑤ 的 A 操作数,故存 half)
// LDS=68=64+4:行跨距 64 float ≡ 0 mod 32 bank,wmma 16x16 store 的列向
// 访问会全撞同 bank;+4 是保住 16B 对齐(4 float)的最小错位。
// LDP=72=64+8:同理,half 粒度下 16B 对齐(8 half)的最小错位。
constexpr int OFF_O = 0;                       // float [64][128] 32768
constexpr int OFF_S = 32768;                   // float [64][68]  17408
constexpr int OFF_ML = 50176;                  // float m[64] l[64] a[64] 768
constexpr int OFF_K = 50944;                   // half  [64][128] 16384
constexpr int OFF_V = 67328;                   // half  [64][128] 16384
constexpr int OFF_P = 83712;                   // half  [64][72]   9216
constexpr int SMEM_BYTES = 92928;
constexpr int LDS = 68, LDP = 72;
```

角色：整个 v2/v3 的资源契约，§3.3 那张表就是它的逐字段展开。

**硬件语义（bank 的算术）**：shared memory 有 32 个 bank、每 bank 每周期 32 bit(CUDA C++ Best Practices Guide §10.2.3.1："Each bank has a bandwidth of 32 bits every clock cycle， and successive 32-bit words are assigned to successive banks")。所以对 float 数组，地址 $a$ 落在 bank $(a \bmod 32)$；行跨距 64 float 时，第 $r$ 行第 $c$ 列的 bank 号是 $(64r + c) \bmod 32 = c \bmod 32$—— **与 $r$ 无关**，同一列的所有行全撞同一个 bank。改成 68 之后是 $(68r + c) \bmod 32 = (4r + c) \bmod 32$，相邻行错开 4 个 bank，$r=0..7$ 覆盖 bank $c, c{+}4, \dots, c{+}28$ 共 8 个，$r=8$ 起绕回—— **同一列上 16 行只落在 8 个 bank 上，列向访问的冲突度从 16 路降到 2 路**（**本讲义折算**；`load/store_matrix_sync` 内部的实际访问模式不公开，这里算的是「同列 16 行」这一最坏情形，标注账面推断）。`Psm` 是 half，两个 half 共用一个 32-bit 字，行跨距 72 half = 36 字，$(36r + c/2) \bmod 32 = (4r + c/2) \bmod 32$，同样是每行错 4 个 bank。**两个魔法数的「打散」效果是同一个 4**，这不是巧合：4 float = 8 half = 16 B 正是 `ldm` 合法性允许的最小步长（§3.3.1）。

关键行：六个 `OFF_*` 是前缀和，`SMEM_BYTES = 92928` 是总和——三者任何一处改动都必须同步改另外两处，否则相邻两区直接重叠（区与区之间没有任何运行时检查）；`LDS = 68` 与 `LDP = 72` 是 §3.3 推的两个约束（打散 bank + 保住 16B 对齐）的解。改错会怎样：把 LDS 改回 64，程序**完全正确**但 wmma 的列向 store 全撞同一 bank，慢下来而不报错——这是最难查的一类性能 bug；把某个 `OFF_*` 少算 768（漏掉 m/l/a 区），Ks 会覆盖在线统计上，softmax 的 max 与分母被 K 的比特图案污染，错得毫无规律。

### 4.3 opt-in 与 launch(flash-attn/src/fa2_v2.cu:173-181)

```cuda
    static bool configured = false;   // 一次性 opt-in:>48KB 动态 smem 须显式放行;
    if (!configured) {                // bench 单线程调用,无并发初始化问题
        cudaFuncSetAttribute(fa2_v2_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM_BYTES);
        configured = true;
    }
    fa2_v2_kernel<<<dim3(S / BM, Hq, B), WARPS * 32, SMEM_BYTES>>>(
        Q, K, V, O, Hq, Hkv, S, causal);
```

角色：把 90.75KB 这个数字变成可执行的前提。关键行：`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES)` 是超 48KB 静态上限的唯一放行方式，且**按 kernel 逐个设置**（v2/v3/v4 三个 kernel 各设一次）；第三个 launch 参数 `SMEM_BYTES` 才是真正分配的动态 smem 大小，`extern __shared__ char smem[]` 只是它的入口。`static bool configured` 是省掉每次调用的属性设置开销，注释同时交代了它的前提（bench 单线程调用，无并发初始化竞争）——把前提写出来，别人才知道搬到多线程环境要改。改错会怎样：漏掉 `cudaFuncSetAttribute`，launch 直接返回 `cudaErrorInvalidValue`，kernel 一次都不跑；而 bench 的正确性 gate 恰好为此准备了兜底——launch 失败记 $10^9$ 误差（main.cu：106-108），崩溃版本不得静默跳过。

### 4.4 相位 ① 与它两侧的 barrier(flash-attn/src/fa2_v2.cu:86-96)

```cuda
    for (int n0 = 0; n0 < nlimit; n0 += BN) {
        __syncthreads();   // ①前置:防新 tile 装载覆盖 Ks/Vs 时,上一轮 ⑤
                           // 仍有 warp 在读 Vs/Psm(WAR)
        // [① K/V 装载] float4 协同搬运;无边界 guard,靠 S % 64 == 0 前置条件
        for (int t = tid; t < BN * FA_D / 8; t += blockDim.x) {
            int r = (t * 8) / FA_D, c = (t * 8) % FA_D;
            *(float4*)&Ks[r * FA_D + c] = *(const float4*)&k[(size_t)(n0 + r) * FA_D + c];
            *(float4*)&Vs[r * FA_D + c] = *(const float4*)&v[(size_t)(n0 + r) * FA_D + c];
        }
        __syncthreads();   // ①→②:装载按线性 tid 分片、wmma 按 warp 读——
                           // 线程集不重合,barrier 后写才可见(跨线程 RAW)
```

角色：相位链的入口，也是 §3.4 前两个 barrier 的现场。关键行：循环上界 `nlimit = causal ? min(q0 + BM, S) : S`(fa2_v2.cu：84)必须**全 block 一致**——循环体内有 `__syncthreads`，若不同线程走不同轮数就会有线程停在 barrier 上等一个永远不来的伙伴（发散到 barrier 是未定义行为）；行级的精确因果边界留给 ③ 的 `jend` 去收。装载循环没有边界 guard，靠 `S % 64 == 0` 这个前置条件兜底（通用尾块由 v0/v1 负责）。改错会怎样：把 `nlimit` 改成行级上界（`row + 1` 之类），同 block 内不同行的循环次数不同，barrier 立刻发散；删掉 ①→② 的 barrier，wmma 读到装了一半的 K tile，$S$ 里混进上一轮的残值，错而不崩。

### 4.5 相位 ②:QK^T 与「架构税现场」(flash-attn/src/fa2_v2.cu:97-114)

```cuda
        #pragma unroll
        for (int n = 0; n < 4; ++n) {                  // [② QK^T] 每 warp 16 行 x 全 64 列条带
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sc;
            wmma::fill_fragment(sc, 0.f);
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                               wmma::col_major> bf;    // K^T 的第 j 列 = K 的第 j 行:
                wmma::load_matrix_sync(bf, &Ks[(n * 16) * FA_D + kk * 16], FA_D);   // 声明 col_major 即得转置视图,免物化 K^T
                wmma::mma_sync(sc, af[kk], bf, sc);
            }
            // 架构税现场:sc 内做不了行级 max(lane→元素映射私有),
            // 只能整块 store 落 smem 交给 ③ 的标量段
            wmma::store_matrix_sync(&Ssm[(warp * 16) * LDS + n * 16], sc,
                                    LDS, wmma::mem_row_major);
        }
        __syncthreads();   // ②→③:行 r 的 S 由 warp r/16 写、由线程 tid=r
                           // (属 warp r/32)读——跨 warp RAW
```

角色：Tensor Core 的第一个 GEMM，以及 §3.2 那条结论的落点。关键行：`bf` 声明成 `col_major` 就得到了 $K^\top$ 的视图——$K^\top$ 的第 $j$ 列就是 $K$ 的第 $j$ 行，声明一改即免去物化转置，这是 wmma 少有的白送；`af[kk]` 是 kernel 开头一次性装载、整个生命周期常驻寄存器的 Q fragment(fa2_v2.cu：73-78)，因为 Q 是唯一不随 tile 变化的操作数，零重读。最关键的是最后那句 `store_matrix_sync`：注释直书「架构税现场」——`sc` 里已经有答案了，但没法在里面做行级 max，只能整块倒回 smem 交给标量段。改错会怎样：`bf` 写成 `row_major`，算出来的是 $QK$ 而不是 $QK^\top$，数值全错且不崩；`store_matrix_sync` 的 leading dim 传 64 而不是 `LDS`，行与行之间错位 4 个 float，softmax 读到的每一行都掺着邻行的尾巴。

### 4.6 相位 ③④:标量段 softmax 与 O 的按行重缩放(flash-attn/src/fa2_v2.cu:115-139)

```cuda
        if (tid < BM) {                                // [③ 行级在线 softmax] 一行一线程
                                                       // (128 线程闲一半——v3 的改进点)
            const int row = q0 + tid;
            const int jend = min(BN, (causal ? row + 1 : S) - n0);   // row+1:causal 含对角
            float rmax = -1e30f;
            for (int j = 0; j < jend; ++j)
                rmax = fmaxf(rmax, Ssm[tid * LDS + j] * scale);   // scale 读时乘,免改写 S 一遍
            const float mn = fmaxf(m_s[tid], rmax);    // 新全局 max 单调不减 → exp 参数恒 <=0
            const float alpha = __expf(m_s[tid] - mn); // 历史部分和折算因子
            float sum = 0.f;
            for (int j = 0; j < BN; ++j) {
                // j >= jend 写 0:P 的因果/越界列必须清零——⑤ 的 wmma 把
                // 整 64 列乘进去,mask 即零填充(面试点③)
                float p = j < jend ? __expf(Ssm[tid * LDS + j] * scale - mn) : 0.f;
                Psm[tid * LDP + j] = __float2half(p);
                sum += p;
            }
            l_s[tid] = l_s[tid] * alpha + sum;         // 在线分母:旧 l 折算 + 本 tile 增量
            m_s[tid] = mn; a_s[tid] = alpha;           // α 落 smem:④ 由别的线程读
        }
        __syncthreads();   // ③→④:Psm/a_s 写完,④ 才能读 a_s、⑤ 才能读 Psm(RAW)
        for (int i = tid; i < BM * FA_D; i += blockDim.x)   // [④ O ×α] i/FA_D = 行号;
            Osm[i] *= a_s[i / FA_D];                        // 必须先于 ⑤:O_new = α·O_old + P·V
        __syncthreads();   // ④→⑤:重缩放按线性 tid 分片,⑤ 按 warp 行条带
                           // load 旧 O——线程集不重合(RAW)
```

角色：整条链上唯一的标量段，也是 Tensor Core 空转的那一段。关键行：`jend` 同时收因果边界与尾块边界，`row + 1` 表示 causal 含对角（token 可见自身）；求 max 的循环只扫到 `jend`，而写 P 的循环扫满 `BN`——**越界列必须写 0**，因为 ⑤ 的 wmma 会把整 64 列乘进去，mask 在这里就是零填充（fa2_v2.cu：126-127）；`scale` 在读取时乘而不是先把 $S$ 整体改写一遍，省掉一次 $64\times64$ 的 smem 读写往返；α 写进 `a_s` 是因为读它的是 ④ 的**别的**线程。改错会怎样：写 P 的循环上界也用 `jend`，越界列留着上一轮的残值，causal 被悄悄破坏——而且 gate 未必抓得住（残值可能恰好很小）；④ 与 ⑤ 顺序对调，历史 $O$ 会被多乘一次 α，长序列上误差逐 tile 复利。

⑤ 段收尾的三行值得单独看（fa2_v2.cu：155-161）：

```cuda
            // wmma 无「累加到内存」原语:load 旧 O → 逐元素加 → store 回;
            // 逐元素加合法:同 shape accumulator 映射一致(见 gemm_v2 面试点②)
            float* optr = &Osm[(warp * 16) * FA_D + c * 16];
            wmma::load_matrix_sync(oacc, optr, FA_D, wmma::mem_row_major);
            #pragma unroll
            for (int e = 0; e < pv.num_elements; ++e) pv.x[e] += oacc.x[e];
            wmma::store_matrix_sync(optr, pv, FA_D, wmma::mem_row_major);
```

wmma 没有「累加到内存」这种原语，所以只能 load 旧 $O$ → 逐元素加 → store 回。这个逐元素加**合法**的唯一依据是 §3.2 的性质 3（同 shape accumulator 映射一致），不是「看起来应该行」——换成两个不同 shape 的 fragment 相加，行为未定义。改错会怎样：把 `pv.x[e] += oacc.x[e]` 换成先 store 再在 smem 上加，结果一样但多一次 32KB 往返；忘了这一步直接 store `pv`，每个 tile 都把 $O$ 覆盖掉，只剩最后一个 tile 的贡献。

### 4.7 v3:每行两线程与 shfl 合并(flash-attn/src/fa2_v3.cu:91-115)

```cuda
        if (tid < 2 * BM) {   // [③] 每行 2 线程:tid 偶/奇 = 同行左右半区,
                              // 且为同 warp 相邻 lane → 可用 shfl 而非 smem 交换
            const int row = tid / 2, hf = tid % 2;
            const int grow = q0 + row;
            const int jend = min(BN, (causal ? grow + 1 : S) - n0);
            const int j0 = hf * 32, j1 = min(j0 + 32, jend);   // 本线程负责的 32 列窗口
            float rmax = -1e30f;
            for (int j = j0; j < j1; ++j)
                rmax = fmaxf(rmax, Ssm[row * LDS + j] * scale);
            rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffff, rmax, 1));   // 与同行搭档合并 max
            const float mn = fmaxf(m_s[row], rmax);
            const float alpha = __expf(m_s[row] - mn);
            float sum = 0.f;
            for (int j = j0; j < j0 + 32; ++j) {   // 上界 j0+32 而非 j1:越界列写 0,
                                                   // ⑤ 的 wmma 吃整行(mask 即零填充,见 v2)
                float p = j < jend ? __expf(Ssm[row * LDS + j] * scale - mn) : 0.f;
                Psm[row * LDP + j] = __float2half(p);
                sum += p;
            }
            sum += __shfl_xor_sync(0xffffffff, sum, 1);   // 与搭档合并分母增量
            if (hf == 0) {   // 单写者:两线程算得同值,限一个写免 WAW(纪律性写法)
                l_s[row] = l_s[row] * alpha + sum;
                m_s[row] = mn; a_s[row] = alpha;
            }
        }
```

角色：§3.5 那套分工的核心段落，也是「分工要照顾原语可用性」的样本。关键行：`row = tid / 2, hf = tid % 2` 让搭档成为同 warp 的相邻 lane，于是 max 与分母都能用 `__shfl_xor_sync(..., 1)` 合并，省掉一次 smem 往返和一次 barrier；写 P 的上界仍是 `j0 + 32` 而不是 `j1`，零填充规则与 v2 一致。最后的 `if (hf == 0)` 注释写作「纪律性写法」，**把它读严一点更有价值**：`m_s[row] = mn` 与 `a_s[row] = alpha` 确实是两线程写同值、无害，但 `l_s[row] = l_s[row] * alpha + sum` 是 read-modify-write——Volta 起线程不再保证 lockstep，若一个线程先写、另一个后读，后者会在**已经乘过 α 的** $l$ 上再乘一次，分母被静默污染。所以这不是纯纪律，是实打实的竞态封堵。**文档依据**：CUDA C++ Programming Guide §20.6.2 "Independent Thread Scheduling" 明写 "The NVIDIA Volta GPU Architecture introduces Independent Thread Scheduling among threads in a warp， enabling intra-warp synchronization patterns previously unavailable ... However， this can lead to a rather different set of threads participating in the executed code than intended if the developer made assumptions about warp-synchronicity of previous hardware architectures." **「同 warp 的两个 lane 一定同步推进」正是被这一条废掉的假设**；`__shfl_xor_sync` 带 `_sync` 后缀与显式掩码也是同一场变更的产物（同节第 1 条）。改错会怎样：去掉 `if (hf == 0)`，大多数时序下结果照样对（两 lane 通常同步推进），压力大或换架构时偶发分母偏小——最难复现的一类 bug；把搭档改成跨 warp 配对，shfl 直接失效（不同 warp 之间没有 lane 交换）。

### 4.8 v4:两组 pipeline 的三个落点(flash-attn/src/fa2_v4.cu:98-107、130-135、163-167)

```cuda
    async_tile(Ks, k, 0, S, tid, blockDim.x);      // 序幕:commit G_K(0),循环内 wait 才有对象
    int p = 0;
    for (int n0 = 0; n0 < nlimit; n0 += BN) {
        __syncthreads();                           // WAR x2:上一轮 ⑤ 读完 Vs 才许 V(t) 覆盖;
                                                   // 读完 SP 才许 QK^T(t) 改写
        async_tile(Vs, v, n0, S, tid, blockDim.x); // commit G_V(t):V 到 ⑤ 才用,与 ②③④ 重叠
        __pipeline_wait_prior(1);                  // 等掉更老的 G_K(t)(交错推导见文件头);
                                                   // 刚发的 G_V(t) 留在途
        __syncthreads();                           // RAW:cp.async 完成仅发起线程可见,
                                                   // barrier 后全 warp 才能读全 K tile
```

```cuda
        if (n0 + BN < nlimit)                      // commit G_K(t+1) 入另一缓冲:此处发出,
                                                   // 到下轮 QK^T 前的 wait 有 ③④⑤ 整段给 DMA;
                                                   // 末轮不发(越界 + 破坏交错计数)
            async_tile(Ks + (p ^ 1) * BN * FA_D, k, n0 + BN, S, tid, blockDim.x);
        __syncthreads();                           // RAW:SP 的 wmma 写(按 warp 行条带)
                                                   // 对 ③ 的读者(每行 2 线程,跨 warp)可见
```

```cuda
        for (int i = tid; i < BM * FA_D; i += blockDim.x)   // [④ O xα]
            Osm[i] *= a_s[i / FA_D];
        __pipeline_wait_prior(n0 + BN < nlimit ? 1 : 0);   // 等掉更老的 G_V(t)(留 G_K(t+1) 在途);
                                                           // 末轮在途只剩 G_V(t),0 = 清空
        __syncthreads();                           // RAW x2:④ 的重缩放写对 ⑤ 的 oacc 读可见;
```

角色：§3.6 归纳证明的三个观测点，按执行顺序是「序幕 + 轮首 wait」→「② 之后发 K(t+1)」→「④ 之后 wait V(t)」。关键行：序幕的 `async_tile(Ks, k, 0, ...)` 让循环内第一个 `wait_prior` 有对象，少了它第一轮等的是空队列、②直接读到未初始化的 smem；`if (n0 + BN < nlimit)` 的末轮不发，与最后那个 `n0 + BN < nlimit ? 1 : 0` 是**同一个条件**，两处必须一致——这正是不变量 I 的边界条款；`__pipeline_wait_prior(1)` 之后紧跟的 `__syncthreads` 不能省，因为 cp.async 的完成只对发起线程可见。改错会怎样：末轮把参数写死 1,⑤ 读到未搬完的 V，错而不崩且只在特定时序下现形；在 ③ 的 `if (tid < 2 * BM)` 分支里加一次 `async_tile`，不同线程的组计数分叉，两个 `wait_prior` 同时失去意义；把 `p ^= 1` 忘掉，K 双缓冲退化为「永远读同一面」，②算的一直是第一个 tile 的 $S$。

## 5 实验数据怎么读

现行数字（`flash-attn/project-proof/data/derived_fa2_proto_stability.csv`，3 轮 mean±std，EXP-K03 §5）：

| 版本 | S=4096 latency (ms) | TFLOPS | 逐级归因 |
|---|---|---|---|
| v0 warp-row | 27.795±0.047 | 4.9±0.06 | — |
| v1 smem tile | 25.113±0.071 | 5.5±0.00 | smem 仅 +11% |
| v2 wmma | 5.635±0.017 | 24.4±0.06 | Tensor Core ×4.5 |
| v3 8warp | 4.229±0.012 | 32.5±0.10 | 并行组织 +33% |
| v4 overlap | **3.949±0.012** | **34.8±0.12** | 预取 + half S/P 仅 +7.1% |

跨尺寸（v4,3 轮 mean）:S=512 / 1K / 2K / 4K = 20.0 / 26.6 / 31.7 / 34.8 TFLOPS。

**轴与口径**。TFLOPS $= 4 \cdot B \cdot H_q \cdot S^2 \cdot D / 2\ /$ 时延（main.cu：143-145）：两个 GEMM，每个 MAC 记 2 FLOP，causal 再除以 2。这个式子与 Triton 版**同式**，是跨 harness 对照能成立的前提；但它对 causal 的处理是理想化的——tile 级裁剪只能裁到 64 行粒度，block $b$ 实际走 $b{+}1$ 个 tile，总量比 $S^2/2$ 多 $64/S$，S=4096 时是 1.56%（账面推断）。也就是说表里的 TFLOPS 系统性低估约 1.5%，而两边同式，比值不受影响。「3 轮」指三次独立进程运行（raw 各一份，UTC 前缀落盘），±后是**轮间** std；每轮本身已是 20 次预热 + 100 iters 的均值，单 event 对包整段再除 iters(main.cu：151-157)。

**误差列有个漂亮的读法**。五个版本、全部 shape 的 `max_abs_err` 都是 **4.88e-04**，而 $2^{-11} = 4.8828125\times 10^{-4}$——这正是 fp16 在 $[0.5,1)$ 区间的半个 ulp。输出经 softmax 加权、量级 $O(1)$，所以**误差被最终写回的 fp16 舍入地板锁死了**：在线单遍 vs 两遍法、fp32 的 $S$ vs fp16 的 $S$，五种算法路径的差异全部埋在这条地板之下。这既说明 gate 过得干净（EXP-K03 §5，H1 成立），也划出了 gate 的分辨率上限——比 $2^{-11}$ 更细的算法差异，这个 gate 测不出来，这是边界不是优点。

**这个实验设计防了哪些坑**。①参考实现故意用两遍法（ref_naive.cu：4-7），算法路径与被测的在线法独立——若参考也写在线法，一个 α 公式写错会两边同错、gate 全过；②每次测前 `cudaMemset(b.O, 0, ...)`(main.cu：103)，防上一版本的残留输出让「什么都不写」的错误实现蒙混过关；③launch 失败记 $10^9$ 误差（main.cu：106-108），崩溃版本不得静默跳过；④固定 `srand(42)`(main.cu：82)，版本间与轮间同输入；⑤形状族刻意打散（main.cu：91-94）：非 causal 分支、GQA 1:1/2:1/4:1、S 从 512 到 2048——causal 的 `jend` 边界与 GQA 的 `kvh` 映射各有独立的错法，不打散就查不出；⑥结果只写 `BENCH_OUT` 指定的 UTC 前缀新文件、首行 provenance(main.cu：120-133)，历史数据永不覆盖；⑦profiler 环境测得的时延不进这张表。

**机理账怎么列**。核验 34.8：$4\times 32\times 4096^2\times 128/2 = 1.374\times 10^{11}$ FLOP，除以 $3.949\times 10^{-3}$ s $\approx 3.48\times 10^{13}$，自洽。这个 FLOP 数值得停一下——它与讲义 01 里 $4096^3$ GEMM 的 $2\cdot 4096^3 = 1.374\times 10^{11}$ **完全相同**。于是两篇讲义的头条数字可以直接对读：**同样 $1.374\times 10^{11}$ 次浮点运算，GEMM 用 1.033 ms，FA2 用 3.949 ms，差 3.82 倍**；换成占 4090 峰值（165.2 TFLOPS）的比例，是 81% 对 21%。架构税就是这 2.9 ms。

**小 S 为什么被压低**。S=512 的 v4 是 0.1075 ms，S=4096 是 3.949 ms：计算量 $\propto S^2$ 涨了 64 倍，时间只涨 36.7 倍——每 FLOP 的时间降了 1.74 倍，与 TFLOPS 从 20.0 涨到 34.8（也是 1.74 倍）完全对上。多出来的那份效率就是固定开销被摊薄：Osm 清零 32KB、8 个 Q fragment 装载、kernel 启动，这些与 tile 循环长度无关，短序列上占比大（EXP-K03 §5 记为「wave quantization 与固定开销」）。

**图怎么读**。README 图 2(figures/02_fa2_wmma_ladder.png)是这张表的版本梯条形图，误差条 = 轮间 std；它与图 1（GEMM 版本梯）共用同一套图形语言，对读即见「同一套 wmma 工具箱，两条完全不同的曲线」。图不携带表之外的信息，存疑时回 CSV。

**三笔不需要额外实验的自洽性核对**。拿到一张性能表先做这三步，任何一步对不上都先怀疑表：

1. **TFLOPS 与时延自洽**：上一段已核过 34.8。
2. **不越过物理上界**：34.8 / 165.2 = 21.1%，在 100% 以内；v0 的 4.9 TFLOPS 对 CUDA core 的 FP32 峰值 82.6 TFLOPS 是 5.9%，也合理。**若某一行超过 100%，不是破纪录，是口径错了。** §8.3.2 给了这条检查在 memory-bound 算子上救命的实例。
3. **访存侧不触顶**：S=4096、Hq=32、Hkv=8、D=128、fp16 时，Q 是 $32\times4096\times128\times2$ B = 32 MB，K 与 V 各 $8\times4096\times128\times2$ B = 8 MB，O 32 MB，合计 80 MB。摊到 3.949 ms 上是 20 GB/s，占 1008 GB/s 的 2%（**本讲义折算**）。**HBM 完全不是瓶颈**——这与 §3.7 的结论一致，也是「FA 的 IO 优化已经做完了」的直接体现。

**v0→v1 只有 +11% 的账**（与讲义 01 §3.1.4 同一套方法）。v1 相对 v0 的唯一变化是把 K/V 从全局读改成先进 shared memory。收益小的原因可以不靠 profiler 算出来：单个 kv-head 的 K 是 $S \times D \times 2$ B = $4096\times128\times2$ = **1 MB**，V 同样 1 MB；4090 的 L2 是 72 MB。**同一个 kv-head 的 K/V 被 4 个 q-head(GQA 4:1)反复读，而它只有 1 MB，第一次读完就长期驻留 L2**——v0 的「全局读」其实绝大部分是 L2 命中，shared memory 再省一层，省的是 L2 到 L1 那一跳，量级本来就小。EXP-K03 §6 的原话是「K/V 广播读 L2 早已扛住」，这里给的是它的数量版本。**对照讲义 01 的 gemm v0→v1(+25%)**：两边都是「tiling 只有个位数到二十几个百分点」，但机制不同——GEMM 那边是指令路径没变（每 FMA 仍是 5 条指令），FA2 这边是数据本来就在 L2。**同样的现象，不同的病因；不把病因说清楚，下一次就会用错药。**

**跨尺寸那一行还能读出什么**。S=512 到 4096，TFLOPS 从 20.0 涨到 34.8，而 causal 下 tile 数从平均 4.5 涨到 32.5（block $b$ 走 $b+1$ 个 tile，$S/64$ 个 block 的平均值 = $(S/64+1)/2$）。**固定开销（Osm 清零 32 KB、8 个 Q fragment 装载、kernel 启动）被摊到的 tile 数涨了 7.2 倍，而效率只涨 1.74 倍**——说明固定开销不是唯一的因素，tile 循环本身的效率也随 S 变化（长循环里预取窗口更充分、尾波占比更低）。**本梯没有把这两项分开测，账面推断。**

## 6 误区与边界

**误区 1：「FA2 慢是因为访存没优化好」。** 这是最自然也最贵的错误直觉——注意力看起来就该是访存问题。v4 把 K 双缓冲、V 重叠、S/P 合一全部做完，只值 **+7.1%**（实测，3 轮）。这个数字的作用是**排除**访存假设：能藏的都藏了，时间不在那里。注意它只做了排除，并不直接证明「相位链是主因」——那一步仍是推断（§3.7）。

**误区 2：「occupancy 8.3% 太低了，多开几个 block 就好」。** 没有「多开 block」这个选项：90.75KB smem 对 Ada 每 SM 100KB，第二块放不下（§3.3）。唯一的旋钮是 block 内的 warp 数，v3 拧了这个旋钮拿到 +33%。反过来看讲义 01 的 gemm v4：理论 occupancy 33% 全梯最低却最快。同一个指标两种读法——判据永远是「延迟是否已被遮蔽」，不是「线程够不够多」。

**误区 3：「wmma 和 mma 只是 API 包装层的差别，底下是同一条指令，性能应该一样」。** 对 GEMM 基本成立（wmma 够到真 cuBLAS 的 85.6%），对 FA2 不成立（28%，跨 harness，推断级）。差别不在指令，在**布局是否公开**：GEMM 的 epilogue 只需逐元素运算，FA2 的 softmax 需要行号（§3.2 那张表）。越是靠融合吃饭的算子，越需要 mma 级的寄存器控制。

**误区 4：「shared memory 是优化手段，加了总没坏处」。** 本仓有一个真实踩坑：gemv v4 用 shared memory 缓存 vec，BankSt 从 0 升到 13,820，时延慢一倍（PORTFOLIO「gemv」节，Laptop 口径）。smem 是**一次额外的搬运**，只有当复用次数把这次搬运摊薄了才划算；vec 靠 L1 广播已经够了，再进 smem 就是纯亏。FA2 的 smem 往返在字节上同源，区别只在于：gemv 那次是可以不犯的错，FA2 这次是 API 层强加的税。

**误区 5：「一个对照数字只要是实测就能用」。** 本仓有三个被自己推翻的实例，分别对应三种不同的失效方式：**对照物不是库**（softmax 曾用的对照经源码核查系自写 kernel，cuBLAS 并无 softmax API）、**对照物不是同一个算子**（reduce 曾用 `cublasSasum` 的 Σ|x| 对照 Σx）、**对照物只跑了一轮**（gemv 的单轮领先幅度在补齐对照侧 3 轮后腰斩，而自家 kernel 完全复现）。三者都通过了「是实测」这一关，三者都不能用。完整过程与现行口径见 §8.3。

**误区 6：「测出来的带宽就是硬件的带宽」。** 判据只有一条：等效带宽对理论峰值。本仓 reduce 在 67.1 MB 数组上测得的等效带宽是 4090 HBM 理论峰值的 1.8 至 3.4 倍（EXP-K04《标准库基准补齐与两区间重测》§4.1）——**超过理论峰值只能说明数据没落 DRAM**，那个尺寸小于 72 MB 的 L2。同一份代码换成 1.07 GB 的数组，等效带宽立刻回到理论峰值的 93.9%，而且版本之间的相对排序也变了。**一个性能数字的第一个属性不是大小，是「它测的是哪一层」。**

**边界声明**：本篇全部实测数字的适用范围是 $D=128$、$B{=}1$、$H_q{=}32$、$H_{kv}{=}8$、causal、$S\in\{512,1024,2048,4096\}$、fp16 存储 / fp32 在线累加、RTX 4090(sm_89)、合成随机输入；v2/v3/v4 额外要求 $S \bmod 64 = 0$，通用尾块由 v0/v1 兜底，目标是归因不是产品化。28%/25% 是跨 harness 推断级，引用必须带这个限定。「smem 往返与相位链是差距主因」是推断，不当实测说——NCU 计数器在本容器不可用（EXP-K01 §7），没有 stall reason 的直接测量。§3.4 的 408 KB/tile 与 61% 是账面推导，不是计数器读数。decode 形状（$S_q{=}1$）与非整除 $S$ 均未测。

## 7 连环追问

**Q1：在线 softmax 的 α 修正是近似吗？** 不是，是恒等式。$e^{s_j-m_{\text{new}}} = e^{s_j-m_{\text{old}}}\cdot e^{m_{\text{old}}-m_{\text{new}}}$ 由指数可加性精确成立（§3.1 第 1 步）。所以「乘 α」与「按新基准重算一遍」逐位等价，FA 的在线法在数学上没有付出任何精度代价，付出的是每 tile 一次重缩放的**算力**。

**Q2：为什么必须减 max？不减会怎样？** fp32 的 $e^x$ 在 $x\gtrsim 88$ 溢出到 inf，而 $s$ 的量级随 $D$ 与输入分布自由增长。减掉当前 max 后指数参数恒 $\le 0$、结果恒在 $(0,1]$，溢出面被整个消掉而不是推远。附带好处：$m$ 单调不减 ⇒ $\alpha \le 1$，重缩放只缩不放，两个累加器也不会溢出。

**Q3：为什么 wmma 能做 fp32→fp16 的逐元素转换，却做不了行级 max？** 性质 3 只承诺「同 shape 的 fragment 映射一致」，所以**同位置**的逐元素运算合法（fa2_v4.cu：123-126 的转换、fa2_v2.cu：159-160 的加法都靠这条）。行级 max 是**沿行规约**，需要知道元素的行号，而 lane→元素映射是编译器私有的（性质 2）。一个不需要位置信息，一个需要——这就是全部区别。

**Q4:90.75KB 怎么来的？为什么必须 opt-in？** $32768(O)+17408(S)+768(m/l/a)+16384(K)+16384(V)+9216(P)=92928$ B（§3.3 逐字段）。静态 `__shared__` 的 48KB 是编译期硬上限，超过只能走动态 smem + `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)`，Ada 每 block 上限 99KB。后果是每 SM 只驻 1 个 block。

**Q5：LDS 为什么是 68？65 行不行？** 两个约束的交点。行跨距 64 float $\equiv 0 \pmod{32}$，同列不同行全落一个 bank，而 wmma 的 $16\times16$ store 恰是列向散布——必须错位。65 能错位，但 65 float = 260B 不再 16B 对齐，破坏 float4 装载与 wmma 的对齐前提。4 float = 16B 是保住对齐的最小错位，所以 68；`Psm` 是 half，8 half = 16B，所以 72。

**Q6：每 tile 5 次 `__syncthreads`，能删掉哪几个？** 中间 4 个删不掉：它们守的都是「写者线程集 ≠ 读者线程集」（装载按线性 tid 分片 / 消费按 warp 分块；$S$ 跨 warp 写读；α 与 $P$ 跨线程；$O$ 重缩放跨线程）。loop-top 那个 WAR barrier 理论上可以用 K/V 双缓冲消掉，但 smem 只剩 8KB 余量，K 一份双缓冲就要 16KB——链的长度是 smem 预算与 fragment 不透明的交点，不是懒（§3.4）。

**Q7：v4 的两个 `wait_prior(1)` 分别在等谁？末轮为什么降为 0？** commit 序列严格交替 $G_K(0),G_V(0),G_K(1),\dots$，不变量是「轮 $t$ 首个 wait 时在途恰为 $[G_K(t),G_V(t)]$」。第一个 wait 留最新 1 组 ⇒ 等掉 $G_K(t)$（② 要读的 K）；第二个 wait 时在途是 $[G_V(t),G_K(t{+}1)]$ ⇒ 等掉 $G_V(t)$（⑤ 要读的 V）。末轮不发 $G_K(t{+}1)$，在途只剩 $[G_V(t)]$，参数写 1 就会把它留在途，⑤ 读到没搬完的 V（§3.6 归纳证明）。

**Q8：V 为什么不双缓冲？** 两个理由。软的：$V$ 只在 ⑤ 消费，而 $G_V(t)$ 已经有 ②③④ 三段重叠窗口，再开一份换不来新窗口。硬的：smem 已 89.75KB，再加 16KB 是 105.75KB，超过 Ada 每 block 99KB 上限，属性设置直接失败——是放不下，不是不划算。

**Q9：v3 里 `if (hf == 0)` 是不是多余？两个线程算的明明是同一个值。** `m_s`/`a_s` 确实是同值写，无害；但 `l_s[row] = l_s[row] * alpha + sum` 是 read-modify-write。Volta 起独立线程调度不再保证 lockstep，若一个线程先写、另一个后读，后者会在已乘过 α 的 $l$ 上再乘一次，分母被静默污染。所以它封堵的是真竞态，不只是纪律（§4.7）。

**Q10：五个版本的 `max_abs_err` 都是 4.88e-04，是巧合吗？** 不是。$2^{-11}=4.8828\times 10^{-4}$ 正是 fp16 在 $[0.5,1)$ 的半个 ulp，而输出经 softmax 加权、量级 $O(1)$。误差被最终写回的 fp16 舍入地板锁死，五种算法路径的差异全埋在地板之下。反过来说，这个 gate 的分辨率就到 $2^{-11}$，更细的差异它测不出来。

**Q11（压力）：28% 这个数字站得住吗？** 诚实答：它是**跨 harness 的推断级**结论，不是同 harness 实测。Triton 版用 wall-clock、本仓用 CUDA event，两边都是 100 iters 稳态、ms 级 kernel 上差异 <1%(EXP-K03 §7)，所以方向可信、量级可用，但不该被当成一个精确的比值去做二阶推论。真正的解法不是多跑几轮——3 轮解决的是同一 harness 内的方差，解决不了口径差；需要的是把两版放进同一个 harness 复测，这是列出的后续工作。

**Q12（压力）：你说瓶颈是相位链，证据呢？这套结论能外推到别的形状/别的卡吗？** 诚实答：**没有直接证据**。有的是一条排除（v4 把 K/V 访存全藏了只值 +7.1%，实测）和一笔账面推导（§3.4 的 408 KB/tile、其中 61% 由 fragment 不透明造成）。NCU 计数器在本容器不可用，拿不到 stall reason，所以本仓对这条的定级就是推断，不当实测说。检验方式是明确的：v5 走 mma + ldmatrix 重写，差距收窄即证实、不收窄即证伪。外推方面更保守：全部数字来自单一协议点（$D{=}128$、causal、GQA 4:1、4090），decode 形状（$S_q{=}1$）结构完全不同（生产实现走 flash-decoding 那一路），本仓没测，不外推。

**Q13：PTX 给的 mma 布局公式，怎么用一句话说清它为什么能救 FA2？** 因为它让每个 lane 能算出「我手上这个数是第几行」。具体地，`mma.m16n8k16` 的累加器布局是 `row = groupID (i<2) / groupID+8 (i>=2)`、`col = threadID_in_group*2 + (i&1)`，其中 `groupID = laneid >> 2`(PTX §9.7.15.5.8)。由此可推：**一整行的 8 列恰好分布在 4 个连续 lane 上**，所以行级规约只要两条 `__shfl_xor_sync`（掩码 1 与 2），$S$ 一个字节都不用落 shared memory(§3.8.2)。

**Q14：为什么说 P 可以「零搬运」地喂给下一条 mma？** 把 C 的布局表与 A 的布局表逐元素比对，同一个 lane 的 8 个累加器元素与 A 需要的 a0..a7 **位置一一对应、顺序一致**（§3.8.3 的表）。所以只要做 fp32→fp16 的逐元素转换再打包成 4 个 `.f16x2` 寄存器，就直接是下一条 mma 的 A 操作数。**这不是巧合**：NVIDIA 设计布局时就考虑了「一条 mma 的输出喂下一条 mma 的输入」这种链式用法，而注意力恰好是链式的。

**Q15：v5 路线看起来全是好处，它的代价是什么？** 四条（§3.8.5）：布局公开等于布局固定，每一个索引计算都要跟着公式走，写错不报错；`mma.m16n8k16` 的 n 只有 8 而 wmma 是 16，循环结构与 fragment 组织要重写；`ldmatrix` 的 `.sync`/`.aligned` 要求比 wmma 更严（任一线程退出即未定义）；最重要的是**本梯没做**，所以「能收窄多少」是假设不是预测。

**Q16：附课里那条计时口径不对称，会不会推翻 reduce 的两区间结论？** 不会推翻，但会改变读法。`reduce_v7`/`reduce_v6` 在被计时的函数内部做 `cudaMalloc`/`cudaFree`(reduce_v7.cu：111-112、：125)，CUB 对照的 `temp_storage` 分配在计时外（reduce_cub.cu：23-28），而计时是包住单次调用的 event 对（cuda-reduce/src/main.cu：67-73）。量级上这处不对称与被讨论的差距同量级，所以 HBM 区间的「差 0.7%」更该读成「在测量分辨率内贴平」，L2 区间的 33.3% 里有多少来自算法本讲义无法判定。**方向是保守的**：消除不对称只会让自写侧变好或不变。检验方式明确（把分配提到计时外重测），**列为开放问题**(§8.3.4)。

**Q17：如果只允许改一处代码，你会怎么缩短相位链？** 在不换 API 的前提下，唯一还有结构空间的是**把 ③ 段与 ②⑤ 段错开**——即 FA3 的 pingpong 思路：让一半 warp 做本 tile 的 softmax，另一半 warp 同时做下一 tile 的 $QK^\top$。但它要求两组 warp 在不同相位，而本梯的 5 次 barrier 是全 block 的，**必须先把 barrier 换成 warp 组之间的细粒度同步（`__syncwarp` 或 named barrier）**，这已经不是「改一处」。**如实说：在 wmma 的约束下，本梯的链长度已经接近它的下界**（§3.4 的「一个都删不掉」），真正的空间在换 API。

**Q18：这两篇讲义放在一起，最该带走的一句话是什么？** **同一套工具，在两个算子上给出 85.6% 与 28% 两个结局，差别不在算法难度，在这个算子的后处理需不需要位置信息。** 由此可以推广出一条写代码之前就能用的判据（§3.2.2）：后处理是逐元素的，wmma 够用；后处理是沿行/沿列规约的，直接上 mma，不要先写一版 wmma 再来测它慢多少。

## 8 工业对照与延伸

### 8.1 与生产实现逐层定位

- **官方 FlashAttention-2 / CUTLASS**：算法与本梯**同构**——在线 softmax 三件套、Q 常驻、O 增量重缩放，一样不多一样不少。差距全部在 warp/instruction 层：mma PTX 拿到公开布局 ⇒ $S$/$P$/$O$ 全程寄存器驻留、softmax 用 warp shuffle 在寄存器里做（§3.8.2 到 §3.8.4 给了这三条各自的布局依据）；ldmatrix + smem swizzle 消 bank conflict；cp.async 多级流水（本梯只有 2 级双缓冲）。**不是算法差距，是布局控制差距**——这是本篇最想留下的一句话。
- **Triton**：`tl.dot` 自动编译到 mma.sync + ldmatrix，程序员只写 tile 逻辑。自家 Triton 版 S=4096 为 1.119 ms ≈ 123 TFLOPS（跨 harness，推断级），本梯 v4 为其 28%；PyTorch sdpa-flash 约 140 TFLOPS（同样跨 harness，推断级），本梯为其 25%。**这条对照的价值是把「布局控制」标了价**：不需要自己写 PTX，只需要换一个把布局当一等公民的编程模型。
- **Hopper 与 FA3**：换 TMA + warp specialization（生产者 warp 搬运、消费者 warp 计算）与 FP8。论文的三条技术写在摘要里（Shah et al.， "FlashAttention-3： Fast and Accurate Attention with Asynchrony and Low-precision"， arXiv:2407.08608）："exploiting asynchrony of the Tensor Cores and TMA to (1) overlap overall computation and data movement via warp-specialization and (2) interleave block-wise matmul and softmax operations， and (3) block quantization and incoherent processing that leverages hardware support for FP8 low-precision." 值得注意的是 FA3 对「相位链串行」给出的解法**不是缩短链**，而是第（2） 条的交错调度——让两组 warp 处在不同相位，一组做 softmax 时另一组做 GEMM，Tensor Core 就不再空转。本篇 §3.7 第 3 条（③ 段 Tensor Core 空转）在 Hopper 上正是被这样解决的，但它依赖 mbarrier/TMA，不是 Ada 上能直接搬的方案。**两条路线的对照很有教学意义**：本篇 §3.8 的 v5 路线是「把链缩短」，FA3 是「让链的两段重叠」；前者需要布局公开，后者需要异步原语与足够的 warp 预算。
- **decode 形状**：$S_q{=}1$ 时 FA2 的 tile 结构完全不适用（一行 query 撑不起 $64\times64$ 的 tile），生产实现走 flash-decoding——沿 KV 维切分成多个分片并行，再做一次跨分片的二次规约（合并各分片的 $m$、$\ell$、$O$，合并规则就是 §3.1 的那套三件套）。本仓未测，不外推。

### 8.2 一个跨算子的对照:同一个「税」在别处长什么样

本篇的税是 API 强加的；但同源的 shared memory 往返也可能是自己加上去的。本仓 gemv 有一个真实踩坑：v4 给 vec 加了一层 shared memory 缓存，BankSt 从 0 升到 13,820，时延慢一倍（PORTFOLIO「gemv」节，NCU 指标为 4070 Laptop 采集口径）。**机制同源，处置方式不同**：gemv 那次删掉就行，FA2 这次要换 API（§3.8 的 v5 路线）。

一句话的访存课：**shared memory 不是「加了就好」，它是一次额外的搬运；只有当复用次数把这次搬运摊薄了才划算。** §3.4 那 408 KB/tile 的 smem 往返，本质上就是 gemv v4 那个错误的放大版。

### 8.3 附课 · 两区间与对照物:一个数字要满足哪四个条件才能用

这一节把主线之外的访存类结论接进来，因为它们回答的正是「28% 这类数字要怎么才算数」。本仓在四个 memory-bound 项目上补齐同算子官方基准并分区间重测的完整过程见 EXP-K04；下面只取教学价值最高的三段。

#### 8.3.1 同一个 reduce,两个区间,两个方向相反的结论

被测算子是 float 数组求和 $\sum x_i$，对照物是官方 **CUB `DeviceReduce::Sum`**（随 CUDA toolkit 分发的同算子官方实现）。分两个尺寸各跑 3 轮（RTX 4090，L2 = 72 MB，HBM 理论峰值 1008.1 GB/s；EXP-K04 §4.1，原始数据 `records/data/exp_k04_reduce_hbmbound_3rounds.csv` 与 `records/data/exp_k04_cuda_reduce_3rounds.csv`）：

| 区间 | 版本 | 时延（ms，mean±std） | 等效带宽 | 占理论峰值 |
|---|---|---|---|---|
| **HBM-bound**(N=1<<28,1.07 GB) | CUB | 1.12730±0.00013 | 952.5 GB/s | 94.5% |
| | **v7（自写最优）** | 1.13483±0.00023 | 946.2 GB/s | **93.9%** |
| | cuBLAS Sasum（异算子，仅作参照） | 1.14722±0.00103 | 935.9 GB/s | 92.8% |
| | v6 / v4 / v0 | 1.161 / 1.452 / 1.807 | 924.9 / 739.6 / 594.3 GB/s | 91.7 / 73.4 / 59.0% |
| **L2 常驻**(N=1<<24,67.1 MB) | CUB | 0.019828±0.000098 |（超理论峰值） | 不适用 |
| | v7 | 0.029740±0.000124 |（超理论峰值） | 不适用 |
| | cuBLAS Sasum | 0.037181±0.000079 |（超理论峰值） | 不适用 |

两个方向相反的结论：

- **真正 DRAM-bound 时，手写与官方库贴的是同一堵墙。** v7 与 CUB 分别达理论峰值的 93.9% 与 94.5%，**相差 0.7%**。代码优劣的空间被同一条 DRAM 带宽线压到百分之一量级——这类算子的正确目标是**逼近峰值**，不是超越对手。
- **数据装进 L2 之后，厂商库的分尺寸调参才显出价值：CUB 快 33.3%**（时间比 1.50×）。瓶颈从 DRAM 带宽回到延迟隐藏、展开度、tile 尺寸与两阶段规约策略，CUB 按架构分派的 tuning 正是为此存在。自写 kernel 想追平，要做的是分尺寸调参，而不是再省一次访存。

README 图 3(figures/03_reduce_two_regimes.png)是这两行的并排条形图。

#### 8.3.2 测量效度:67 MB 的经典配置在 72 MB L2 的卡上会静默变成 L2 基准

reduce 的教科书规模是 $N = 2^{24}$ = 1677 万个 float = 67.1 MB。**在 L2 只有几 MB 的世代，这个尺寸确实是 HBM-bound 的；在 L2 有 72 MB 的 4090 上，整个数组常驻 L2。** 判据不需要 profiler：

- 三个版本在该尺寸下的等效带宽是 3384 / 2256 / 1805 GB/s，而 HBM 理论峰值只有 1008.1 GB/s。**等效带宽 1.8 至 3.4 倍于理论峰值，只能说明数据没落 DRAM。**
- 所以该区间**只报时延，不报带宽占比**——报了即错。

这条效度问题会污染两类结论。**第一类是跨机比较**：本仓早期在 4070 Laptop(32 MB L2)与 4090(72 MB L2)之间比较版本排序，同一个 67.1 MB 的尺寸在前者是 HBM-bound、在后者是 L2 常驻，**两台机器根本不在同一个区间**，排序变化的更可能解释是区间不同，而不是「roofline 迁移」。**第二类是「谁更快」的叙事**：同一份代码在两个区间给出 0.7% 与 33.3% 两个完全不同的答案，只报其中一个都是片面的。

**可推广的检查清单**（本讲义总结）：跑任何带宽类 benchmark 之前，先算 `工作集字节 / L2 容量`；比值小于 1 就必须换尺寸或者明确声明测的是 L2。这条对 GEMM 同样适用——讲义 01 §3.8 就是靠「A + B = 64 MB < 72 MB L2」这一步才敢说「DRAM 强制流量不变」。

#### 8.3.3 对照物的四个条件,以及本仓各踩过一次

一个「vs X」的数字要能用，X 必须同时满足四条。本仓在这四条上各踩过一次，四次都留了记录：

1. **X 必须是真的库**。softmax 曾经的对照物经源码核查系自写 warp 原语 kernel（cuBLAS 规范里根本没有 softmax），整条对比链撤销（EXP-K01 §5）；该文件在仓内已改名 `handwritten_ref`。**规矩：凡「vs X」先验 X 的调用点**——讲义 01 §4.7 的 `gemm_cublas.cu` 就是被验对象。
2. **X 必须是同一个算子**。reduce 曾用 `cublasSasum` 作对照，而 asum 算的是 $\sum|x_i|$，与被测的 $\sum x_i$ 语义不同。换成同算子的 CUB 之后才有了 §8.3.1 那张表；顺带一提，asum 在 HBM 区间也慢于 CUB（92.8% 对 94.5%），说明 BLAS 的规约路径本来就不是为纯求和优化的（EXP-K04 §5）。
3. **X 必须跑同样多轮**。gemv 的领先幅度在单轮口径下曾被测得远高于现值；补齐对照侧的 3 轮之后，**自家 v3 完全复现，坏轮全在对照那一边**。现行口径：v3 比 `cublasSgemv` 快 **34.1%**（4096×2048,3 轮，EXP-K04 §4.3；轮间波动约 ±2 个百分点，对外取较保守的一轮）。那个单轮数字已撤销。**如果当时只给自家 kernel 跑 3 轮、对照沿用单轮值，那个虚高的数字会以「3 轮实测」的名义活下来——听起来完全合规，实际上一半是假的。**
4. **X 必须在同一个区间/形状上比**。softmax 换成官方的 `cudnnSoftmaxForward`（`MODE_INSTANCE` + `SOFTMAX_ACCURATE`，与本仓 v0 至 v4 同为「减最大值」的数值稳定口径）之后，结论变成**一对**数字而不是一个：

| 形状 | v4（自写最优） | cuDNN | 判定 |
|---|---|---|---|
| 1024×1024（对齐） | **0.007768±0.000103 ms** | 0.008291±0.000023 ms | v4 快 **6.7%** |
| 1024×1500（非对齐） | 0.009832±0.000045 ms | **0.008947±0.000095 ms** | cuDNN 快 **9.9%** |

（3 轮 mean±std，EXP-K04 §4.2，原始数据 `records/data/exp_k04_softmax_3rounds.csv`。）**手写的优势只在对齐形状成立**；1500 列卡在 v4 主循环步长 `blockSize × packSize = 1024` 的中间，第二轮只有一小部分线程还在走 float4 路径，其余空转——这是行内的尾波量化（**账面推断**，NCU 计数器在本容器不可用）。**厂商库的价值有很大一部分就是「所有形状都不塌」**，单一 shape 特化的收益必须连着它的适用边界一起报。

#### 8.3.4 一处本讲义自己发现的口径不对称(**已实测闭合**)

把 §8.3.1 的结论再压一道：reduce 的 harness 里，`reduce_v7` / `reduce_v6` 在**被计时的函数内部**做了一次 `cudaMalloc` 与 `cudaFree`（为两段式规约的中间数组；修复后见 `cuda-reduce/src/reduce_v7.cu:124-133` 的静态缓存，v6 同款在 `reduce_v6.cu:131-140`），而 CUB 对照把 `temp_storage` 一次性分配在计时外（`cuda-reduce/src/reduce_cub.cu:23-28` 的注释写明了这个口径）。计时用的是包住单次调用的 event 对（`cuda-reduce/src/main.cu:67-73`），所以这次主机侧分配的耗时会落进被测区间。

量级估计（**本讲义分析，未实测，列为开放问题**）：HBM 区间 v7 是 1134.8 μs，0.7% 就是 8 μs；L2 区间 v7 是 29.7 μs，与 CUB 的差是 9.9 μs。**一对 4 KB 的 `cudaMalloc`/`cudaFree` 通常在微秒量级**，因此它在两个区间里都与被讨论的差距同量级。

**这不会推翻 §8.3.1 的两条结论，但会改变它们的读法**：

- HBM 区间的「差 0.7%」更准确的说法是「**在测量分辨率内两者贴平**」，而不是「CUB 略胜」；
- L2 区间的「CUB 快 33.3%」中有多少来自算法、多少来自这处口径不对称，**本讲义无法判定**。

> **后续（EXP-K09 §5.17，已实测）**：口径修正后重跑两区间各 3 轮，本节两条预测**全部命中**：
> HBM 区间 CUB 与 v7 差 **−0.0%**（1.127460 对 1.127220 ms）——「测量分辨率内贴平」得到证实；
> L2 区间「CUB 快 33.3%」降到 **13.5%**（CUB 0.021228 对 v7 0.024545 ms），
> 即 33.3 个百分点里约 **20 个来自口径不对称**，算法差距只剩 13.5%。
> 另外本节当时只点出了 `cudaMalloc`/`cudaFree`，实际还有更大的一项——
> v6/v7 在计时区内调 `cudaGetDeviceProperties`（v1–v5 没有）。
> 该调用在 driver 570 上单次可达毫秒级，使 v6/v7 一度测出 1.6–1.7 ms（慢 50 余倍）。

**方向是明确的**：消除不对称只会让自写 kernel 的数字变好或不变，所以现行口径对自写侧是**保守**的。检验方式也是明确的：把 `d_partial` 的分配提到计时外（与 CUB 的 `temp_storage` 同口径）后重测两个区间。**在做这件事之前，§8.3.1 的两条结论应当带着这条限定引用。**

**这一段本身就是附课的主题**：一个数字的可用性不只取决于对照物选得对不对，还取决于**两边被计时的东西是不是同一件事**。前面三条讲的是「X 选对了吗」，这一条讲的是「计时边界画在同一个地方了吗」。

### 8.4 延伸阅读(每条一句「读它能解决什么疑问」)

**仓内一手材料**

1. `flash-attn/src/fa2_v2.cu:3-23` 文件头——五段相位链的定义与三个面试点的仓内一手陈述。读它能解决：「§3.2 与 §3.4 的结论在作者动手写代码时是怎么想的，以及哪些约束是写之前就知道的」。
2. `flash-attn/src/fa2_v4.cu:27-41` 文件头——两组 pipeline 交错等待的原始推导。读它能解决：「§3.6 的 L−N 表在代码注释里长什么样」，以及「为什么作者认为这里比 gemm v3 危险一级」。
3. `records/EXP-K03_cuda_fa2_ladder.md` §6-§7——H1/H2 假设的判定过程、跨 harness 对照的完整口径与开放问题。读它能解决：「跑之前锁定了什么、跑完哪些成立哪些降级」。
4. `records/EXP-K04_standard_library_baselines.md`——同算子官方基准（CUB / cuDNN）的补齐与两区间重测。读它能解决：「§8.3 的每一个数字从哪来，以及对照物选错会把结论带偏多远」。
5. `docs/lectures/01_tensorcore_gemm_ladder.md` §3.2、§3.3 与 §3.8——wmma fragment 的契约、cp.async 组语义、存储层级的容量与带宽台阶。读它能解决：「本篇全程引用的三个前置各自的完整推导」。

**官方文档**

6. CUDA C++ Programming Guide §10.24 "Warp Matrix Functions"——wmma 的完整契约。读它能解决：「『布局不公开』这句话的原文是什么，以及唯一的例外条款授权到哪一步为止」(§3.2.1)。
7. CUDA C++ Programming Guide §20.7.3 "Shared Memory"(compute capability 8.x)——100/99/48 KB 三个上限与那 1 KB 的去向。读它能解决：「§3.3 的 opt-in 与 §3.6 的『V 放不下』分别该用哪个上限」。
8. CUDA C++ Programming Guide §10.28.4 "Pipeline Primitives Interface"——`__pipeline_*` 三个原语的定义，特别是 wait 的 L−N 形式。读它能解决：「怎么把在途组计数写成一个能代数字的式子」(§3.6.1)。
9. PTX ISA §9.7.15.5.8 "Matrix Fragments for `mma.m16n8k16` with floating point type"——A/B/C 三张 lane→元素映射表。读它能解决：「wmma 拒绝告诉你的那张表长什么样」，以及 §3.8.2 与 §3.8.3 的全部推导依据。**这是本篇最该读原文的一节。**
10. PTX ISA §9.7.15.5.15 `ldmatrix`——按 mma 布局从 smem 取数，地址由 32 个线程各自提供。读它能解决：「为什么 mma 路线能做 smem swizzle 而 wmma 不能」(§3.8.4)。
11. PTX ISA §9.7.9.26.3(cp.async / commit_group / wait_group)——组语义与可见性条款。读它能解决：「空组算不算一组、拷贝对谁可见、为什么 wait 之后还要 barrier」(§3.6.2、§3.6.3)。
12. CUDA C++ Best Practices Guide §10.2.3.1 "Shared Memory and Memory Banks"——bank 结构、冲突与广播。读它能解决：「LDS=68 里『打散 bank』那一半的依据」(§3.3.1)。

**论文**

13. Dao et al.， "FlashAttention： Fast and Memory-Efficient Exact Attention with IO-Awareness"， arXiv:2205.14135(Theorem 2、Proposition 3)——IO 复杂度与它的下界。读它能解决：「FA 的收益在数学上被界定成什么，以及这个界定停在哪一层」(§2.1、§3.10.3)。
14. Dao， "FlashAttention-2： Faster Attention with Better Parallelism and Work Partitioning"， arXiv:2307.08691（§3.1 算法修改、§3.3 warp 划分）——split-K 与 split-Q 的取舍。读它能解决：「本梯 v3 的『行带 × 列半区』混合划分为什么没有被论文说中的代价咬到」(§3.5.1)。
15. Milakov & Gimelshein， "Online normalizer calculation for softmax"， arXiv:1805.02867——在线 softmax 的原始出处。读它能解决：「三件套不是 FlashAttention 发明的，它原本是为了把 softmax 的三遍访存降到两遍」(§2.1)。
