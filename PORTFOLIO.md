# CUDA Kernel 优化 Portfolio

*README 的深读层：方法论、逐项目拆解与跨项目规律*

本文是六个项目（reduce、softmax、gemv、int8 quantize、Tensor Core GEMM、FA2 forward）的统一深读入口。主轴是方法论，每个项目作为落地案例，跨项目规律收尾；现行数字一律附实验记录指针。门面与性能结果表见 [README.md](README.md)。

## 目录

- [方法论](#方法论)
  - [profile 优先于直觉](#profile-优先于直觉)
  - [控制变量归因](#控制变量归因)
  - [先判定 memory-bound 还是 compute-bound](#先判定-memory-bound-还是-compute-bound)
  - [优化是全链路问题](#优化是全链路问题)
- [项目拆解](#项目拆解)
  - [reduce:两个区间,两个结论](#reduce两个区间两个结论)
  - [softmax:与 cuDNN 的形状敏感性,加控制变量法归因](#softmax与-cudnn-的形状敏感性加控制变量法归因)
  - [gemv:极简结构反超,与一次真实工程踩坑](#gemv极简结构反超与一次真实工程踩坑)
  - [int8 quantize:kernel 融合对比 PyTorch eager](#int8-quantizekernel-融合对比-pytorch-eager)
  - [Tensor Core GEMM:wmma 版本梯](#tensor-core-gemmwmma-版本梯)
  - [FA2 forward:wmma 架构税的定量测量](#fa2-forwardwmma-架构税的定量测量)
- [跨项目规律](#跨项目规律)
  - [memory-bound 阶段,shared memory 微优化收益趋近于 0](#memory-bound-阶段shared-memory-微优化收益趋近于-0)
  - [特化 scope 内,hardware-optimal 胜过 algorithm-optimal](#特化-scope-内hardware-optimal-胜过-algorithm-optimal)
  - [NCU 指标必须与时延趋势联读](#ncu-指标必须与时延趋势联读)
  - [前置优化需要后置环节激活才能兑现](#前置优化需要后置环节激活才能兑现)
- [算子带宽画像](#算子带宽画像)
- [文档索引](#文档索引)

## 方法论

贯穿全部项目的四条原则，按重要性排序。

### profile 优先于直觉

任何 GPU 优化的第一步是 `ncu`，不是猜测。每个项目都有一次 profiler 推翻预判的记录：

- reduce：旧一代（Laptop）记录曾主张「v6/v7 grid-stride 慢 6x」——该叙事按 [EXP-K01《四 kernel 4090 重基准》§5](records/EXP-K01_4090_rebench.md) 降级为「旧数据不可确证，不作主张」；4090 现行结论见下文两区间表（[EXP-K04《标准库基准补齐与两区间重测》](records/EXP-K04_standard_library_baselines.md)）。同一算子的测量口径也被推翻过一次：经典 1600 万元素（67.1 MB）配置在 L2 达 72 MB 的 4090 上测到的是 L2 带宽而非 HBM 带宽——等效带宽超过理论峰值即是判据。原始过程稿见 docs/archive/。
- softmax：原「v4 比 cuBLAS 快 26%，NCU 显示 cuBLAS L2 命中率 2.4 倍」整段撤销——对照物经源码核查实为自写 kernel(EXP-K01 §5)，归因对象并非 cuBLAS。
- gemv：v4 用 shared memory 缓存 vec 反而慢一倍，NCU 显示 BankSt 升至 13820（v3 为 0）——vec 被从 L1 多搬了一次。
- quantize：v4 的 char4 store 使 L2 命中率从 32% 跌至 2%，时延反而更快——原因是消除了 read-modify-write 的补偿读。

profile 的价值不在确认已知，而在挑战自以为已知的判断。

### 控制变量归因

只做 v0 至 v_n 的一路加速不足以归因：它只能说明每一步有用，说不清每一步贡献多少。softmax 项目为此构造了三个反例版本：

- v4.2（保留向量化、去掉 warp shuffle）：性能完全退回 v0——证明 v3 至 v4 的 30% 加速主要来自 warp shuffle，不是向量化。
- v4.3（main+tail 显式分离）：cols=1500 上不比 v4 快——证明 v4 的真正退化点是负载不均，不是 tail handling。
- v4.4（保留 float4 主循环、归约阶段故意制造 bank conflict）：慢于自写 warp 参照 8%——证明后置环节退化能拖垮所有前置优化。

没有反例就没有归因；主动构造对比，是把实验方法用进工程。

### 先判定 memory-bound 还是 compute-bound

各项目共同验证：在 compute-bound 阶段，访存层面的小优化收益接近 0；在 memory-bound 阶段，指令层面的小优化收益接近 0。判定依据是 NCU 三个指标的交叉验证：

- SM% 高且 DRAM% 低：compute-bound;
- SM% 低、DRAM% 高且 StallLSB% 高：memory-bound。

每个项目都能观察到瓶颈类型迁移的拐点：reduce 在 v3 至 v4（DRAM% 71 至 96），softmax 在 v3 至 v4（DRAM% 66 至 76），gemv 在 v0 一步到位（DRAM% 直接 95%），quantize 在 v0 至 v1（DRAM% 61 至 84）。先判定瓶颈类型，再选对应的优化手段；错配的优化在错误的方向上没有回报。

### 优化是全链路问题

kernel 的整体性能由最慢链路决定；前段再快，被一个 `__syncthreads` 卡住就整体回落（Amdahl 定律在 GPU 上的体现）。最直接的证据是 softmax v4.4：main 循环保留 float4（带宽利用率 100%），归约阶段一退化（bank conflict 加全程同步），整体反而慢于自写 warp 参照。gemv v4 同源：加 shared memory 把 vec 从 L1 多搬一次，整体慢一倍。优化要覆盖全链路，任何被忽略的环节都可能成为新的瓶颈。

## 项目拆解

### reduce:两个区间,两个结论

算子：float 数组求和（Σx）。对照物是官方 **CUB `DeviceReduce::Sum`**（随 toolkit 分发的同算子官方实现）；此前用的 `cublasSasum` 算的是 Σ|x|，并非同一个算子，该对照口径不再使用。

结果（RTX 4090,3 轮 mean±std，[EXP-K04](records/EXP-K04_standard_library_baselines.md)，原始数据 `records/data/exp_k04_reduce_hbmbound_3rounds.csv` 与 `records/data/exp_k04_cuda_reduce_3rounds.csv`）：

| 区间 | 版本 | 时延（ms，mean±std） | 等效带宽 | 占 HBM 理论峰值（1008.1 GB/s） |
|---|---|---|---|---|
| HBM-bound(1.07 GB) | CUB | 1.12730±0.00013 | 952.5 GB/s | 94.5% |
| | v7（自写最优） | 1.13483±0.00023 | 946.2 GB/s | **93.9%** |
| | cuBLAS Sasum（异算子，仅作参照） | 1.14722±0.00103 | 935.9 GB/s | 92.8% |
| | v6 / v4 / v0 | 1.161 / 1.452 / 1.807 | 924.9 / 739.6 / 594.3 GB/s | 91.7 / 73.4 / 59.0% |
| L2 常驻（67.1 MB） | CUB | 0.019828±0.000098 |（超理论峰值） | 不适用 |
| | v7 | 0.029740±0.000124 |（超理论峰值） | 不适用 |
| | cuBLAS Sasum | 0.037181±0.000079 |（超理论峰值） | 不适用 |

两个区间给出两个方向相反的结论：

- **真正 DRAM-bound 时，手写与官方库同贴一堵物理墙。** 1.07 GB 数组远大于 4090 的 72 MB L2，v7 与 CUB 分别跑到理论峰值的 93.9% 与 94.5%，**相差 0.7%**。代码优劣的空间被同一条 DRAM 带宽线压到百分之一量级——这类算子的正确目标是逼近峰值，不是超越对手。
- **数据装进 L2 后，厂商库的调参优势才显现：CUB 快 33.3%。** 67.1 MB 小于 L2 容量，瓶颈从 DRAM 带宽回到延迟隐藏、展开度、tile 尺寸与两阶段规约策略，CUB 按架构分派的 tuning 正是为此存在。自写 kernel 想追平，要做的是分尺寸调参，而不是再省一次访存。

L2 常驻区间三个版本的等效带宽都是理论峰值的 1.8 至 3.4 倍，这本身就证明数据没有落到 DRAM；该区间只报时延不报带宽占比，报了即错。带宽类结论必须先过「等效带宽对理论峰值」这一步合理性检查，否则会把 L2 带宽当成 HBM 带宽汇报——本文早期跨机比较里出现的版本排序变化，更可能的解释就是两台机器不在同一区间。

旧一代数据的处理：本节旧稿以 4070 Laptop 数字立论（baseline 348 ms、「v6/v7 grid-stride 慢 6x 教学反例」及其 NCU 归因与 CUB/Thrust 行业延伸）。复查发现 Laptop 端 v7 在两份数据文件中自相矛盾（1.665 ms vs 0.273 ms），「v6/v7 回退、4090 反转」的叙事不可确证、不作主张（EXP-K01 §5）；旧稿全文移 docs/archive/ 留痕，其中数字不对外引用。唯一例外：端到端口径「347.6 ms 至 0.291 ms，约 1193x」为 4070 Laptop 测量，引用时必须带 Laptop 定语。Laptop 时代的 NCU 机理参照（Sec/Ld=4 证明 coalescing 未坏等）保留在归档稿与 `artifacts/ncu_for_mac/`。

现行结论：4090 上 v7(grid-stride two-pass)在 HBM-bound 区间达 HBM 理论带宽的 93.9%，与官方 CUB 差 0.7%；L2 常驻区间 CUB 快 33.3%（均 3 轮）。所有标注库名的对照均经调用点验真。

深挖材料：`cuda-reduce/project-proof/docs/interview-analysis-v7.md`。

### softmax:与 cuDNN 的形状敏感性,加控制变量法归因

算子：1024x1024 fp32 矩阵逐行做 softmax。

对照物更正：本节旧稿头条「v4 比 cuBLAS 快 26%」及全部 vs-cuBLAS 归因（L2 命中率 2.4 倍、online softmax 推断、行业延伸）作废——对照物 `softmax_cublas.cu` 经源码核查系自写 warp 原语 kernel，并非 cuBLAS（cuBLAS 无 softmax API；EXP-K01 §5）。作废段落移 docs/archive/ 留痕，其中数字不对外引用；该文件在仓内已改名 `handwritten_ref`。

标准库对照：softmax 不在 BLAS 规范内，同算子的官方实现属 cuDNN。补齐后的对照为 `cudnnSoftmaxForward`（`MODE_INSTANCE` + `SOFTMAX_ACCURATE`，与本仓 v0 至 v4 同为「减最大值」的数值稳定口径），RTX 4090,3 轮 mean±std（[EXP-K04](records/EXP-K04_standard_library_baselines.md)，原始数据 `records/data/exp_k04_softmax_3rounds.csv`）：

| 形状 | v4（自写最优） | cuDNN | 判定 |
|---|---|---|---|
| 1024x1024（对齐） | **0.007768±0.000103 ms** | 0.008291±0.000023 ms | v4 快 **6.7%** |
| 1024x1500（非对齐） | 0.009832±0.000045 ms | **0.008947±0.000095 ms** | cuDNN 快 **9.9%** |

手写的优势只在对齐形状成立；换成非对齐的 1500 列，cuDNN 反过来快 9.9%。厂商库的价值有很大一部分就是「所有形状都不塌」，而单一 shape 特化的收益必须连着它的适用边界一起报。这也从外部印证了下文 v4.3 反例的结论：v4 的真正退化点是非对齐形状上的负载不均。

仍然成立的部分是控制变量法归因——版本间比较不依赖外部对照物（时延为 Laptop 旧代口径，4090 端 3 轮复测排序一致，见 EXP-K01）：

| 版本 | 时延（ms，Laptop） | 说明 |
|---|---|---|
| v0 | 0.0262 | 起点 |
| v4(float4 + warp shuffle) | **0.0164** | 自写最优 |
| v4.2（去 warp shuffle） | 0.0262 | 反例 1：退回 v0，warp shuffle 是向量化收益的前提 |
| v4.3（main+tail 分离，cols=1500） | 0.0227 | 反例 2：v4 的真正退化点是负载不均 |
| v4.4（故意制造 bank conflict） | 0.0236 | 反例 3：后置退化拖垮前置优化（Amdahl） |
| handwritten_ref（原误标 cublas） | 0.0220 | 自写 warp 参照，非 cuBLAS |

关键贡献：v4.2 / v4.3 / v4.4 三个反例完成精细归因——没有反例就没有归因；这一方法论不依赖外部对照物，依然成立。

NCU 细节（Laptop 采集，行名已更正）见 `softmax/project-proof/profiling/ncu/SUMMARY.md`。

### gemv:极简结构反超,与一次真实工程踩坑

算子：mat(4096x2048 fp32)x vec(2048 fp32)= y(4096 fp32)。

结果（4070 Laptop 口径；4090 端 3 轮口径为 v3 比 `cublasSgemv` 快 34.1%，[EXP-K04](records/EXP-K04_standard_library_baselines.md)，原始数据 `records/data/exp_k04_gemv_3rounds.csv`；前一轮同协议为 37.8%，差异来自 cuBLAS 侧的轮间波动。**该幅度的适用区间是「工作集常驻 L2」**——4096×2048 fp32 = 33.6 MB 放得进 72 MB L2；一旦强制从显存冷读，两者同撞带宽墙、差距收敛到 1.4%，见 [EXP-K09](records/EXP-K09_post_vectorization_sector_ledger.md) §5.8）：

| 版本 | 时延 | 说明 |
|---|---|---|
| baseline | 0.618 ms | 单线程参照 |
| v0-v2(block-per-row) | 0.05-0.055 ms | 访问模式迭代 |
| v3(warp-per-row + warp shuffle) | **0.0325 ms** | 自写最优，比 cuBLAS 快 19% |
| v4(block + shared cache vec) | 0.0628 ms | 工程踩坑：慢一倍，比 cuBLAS 慢 56% |
| cuBLAS(`cublasSgemv`) | 0.0402 ms | 通用库基准 |

v4 是真实的工程踩坑，不是构造反例——用 shared memory 缓存 vec 听起来合理，结果 BankSt 升至 13820，比 v3 多出一万余次 conflict。

NCU 关键数据：

| 指标 | v3 | cuBLAS | 解读 |
|---|---|---|---|
| L2Hit% | 2.7% | **21.0%** | cuBLAS 高 8 倍（column-major tiling） |
| DRAM% | **95.2%** | 94.7% | v3 略高 |
| StallLSB% | **94.8%** | 80.8% | v3 的 SM 全部时间在等 HBM |
| BankLd / BankSt | 0 / 0 | 0 / 0 | v3 完全不用 shared memory |

机制：v3 与 softmax v4 是同一个故事的第二次出现——hardware-optimal beats algorithm-optimal。cuBLAS 用 column-major tiling 获得更好的 L2 复用（命中率 21% 对 2.7%），但被通用性的索引 / 转置开销拖累；v3 用 row-major 加 warp shuffle 的极简结构把 HBM 调度器压到 95%。

v4 的教训：vec 本已在 L1 cache 里（vec 8KB 远小于 L1 的几十 KB），再用 shared memory 缓存等于多做一次「L1 至寄存器至 shared memory 至寄存器」的搬运。这类坑不是设计出来的，而是对硬件状态判断错误的结果；代价是数小时调试加一个慢一倍的 kernel，profiler 是唯一的解药。

延伸：v3 在 mat 不超过 L2 的尺寸上（此处 mat 32MB，在 4090 的 72MB L2 之内）稳定领先；mat 远大于 L2 时 cuBLAS 的 L2 优势会消失，这一领先可能持平甚至反转——reduce 的两区间结果是同一件事的直接演示。这正是 LLM 推理框架（TensorRT-LLM、vLLM）为每个 shape 维护 specialized kernel 的原因——本质是做比 cuBLAS 更激进但 scope 更窄的库。

### int8 quantize:kernel 融合对比 PyTorch eager

算子：fp32 至 int8 的 per-channel symmetric quantize（1024 channels x 1024 hw,4MB 至 1MB）。

结果（4070 Laptop 口径；4090 端 v4 = 5.57±0.03 µs,3 轮，`records/data/exp_k01_int8_quantize_3rounds.csv`）:

| 版本 | 时延 | 说明 |
|---|---|---|
| baseline（GPU 单线程） | 121.35 ms | 参照系 |
| v0(grid-stride) | 0.0148 ms | 已比 PyTorch eager 快 3x |
| v3(block-per-channel + float4 read) | 0.00749 ms | scale 缓存进寄存器 |
| v4(+ char4 vectorized store) | **0.00663 ms** | 自写最优，比 PyTorch eager 快 6.6x |
| PyTorch eager(CUDA) | 0.0437 ms | 工业基准（`(x/s).round().clamp().to(int8)`） |
| PyTorch quantize_per_channel(CPU only) | 2.997 ms | PyTorch 官方 PTQ API |

NCU 关键数据：

| 指标 | v3 | v4 | 含义 |
|---|---|---|---|
| DRAM% | 84% | 85% | 接近算子上限（dtype 不对称使 85% 成为天花板） |
| L2Hit% | 32% | **2.15%** | 命中率暴跌，时延反而更快 |
| StallLSB% | 79% | 82% | SM 越来越纯粹地等 HBM |
| 时延 | 0.0075 ms | **0.0066 ms** | 快 12% |

第一个反直觉现象：L2 命中率暴跌但更快。v3 每次 store 1 字节会触发 read-modify-write（硬件读 4 字节 word、改 1 字节、写回），补偿读被计入 L2 hit；v4 用 char4 做 4 字节整体覆盖，不需要补偿读，命中计数下降的同时实际访存量减少。同样的 L2 命中率暴跌，换一个场景可能是性能劣化，在此处却是优化——profiler 给现象，原因要靠模型推理。

第二个反直觉现象：C++ v4 比 PyTorch eager 快 6.6x。原因不是算法更聪明，而是 PyTorch eager 把 `(x/s).round().clamp().to(int8)` 拆成 4 个独立 kernel，每个都要向 HBM materialize 4MB 中间 tensor，总 HBM 流量 32MB；手写版单 kernel 全程融合在 SRAM 内，流量只有 5MB——6.4x 的流量差对应 6.6x 的时延差。

延伸：这是 Flash Attention 核心思想的最小复现——经典 attention 是 softmax(Q@K^T) @ V 三个独立 kernel，每个 materialize N x N 中间矩阵；Flash Attention 融合在一个 kernel 里，N x N 只存在于 SRAM。避免中间 tensor 往返 HBM，是现代 GPU kernel 优化中最重要的单一原则。

### Tensor Core GEMM:wmma 版本梯

算子：fp16 GEMM 4096³（fp32 累加），对照为真 `cublasGemmEx`（调用点验真——softmax 对照物更正后的标准动作）。RTX 4090。

结果（3 轮，[EXP-K02《CUDA Tensor Core GEMM 版本梯》](records/EXP-K02_cuda_gemm_tc_ladder.md)）:v0 naive 5.2,v1 smem tile 6.5,v2 wmma 89.5,v3 cp.async 双缓冲 95.5,v4 128x128 大 tile 133.1 TFLOPS，即真 cuBLAS 的 85.6%。

- compute-bound 算子的台阶是指令世代（v1 至 v2 为 13.8x），访存微调只有 +25%——与前四个 memory-bound 项目完全相反；先判定 bound 类型再动手。
- v4 理论 occupancy 33% 为全梯最低却最快（92 寄存器 x 256 线程 + 32KB smem，每 SM 仅 2 个 block）——Tensor Core 吞吐依赖 fragment 级 ILP 与 smem 复用，不依赖线程数遮蔽延迟。
- 与自家 Triton 版（triton-kernels#EXP-T02《流水线 GEMM》，160.5 TFLOPS）对照：Triton 编译器发射 mma + ldmatrix + swizzle，而 wmma API 不暴露 smem swizzle——同一硬件上「写 CUDA」不等于「到上限」，API 层级本身是性能变量（剩余差距归因为推断，NCU 不可用）。

### FA2 forward:wmma 架构税的定量测量

算子：Flash Attention 2 前向（在线 softmax，D=128，causal+GQA），协议对齐自家 Triton 版（B=1，Hq=32，Hkv=8，S=4096）。RTX 4090。

结果（3 轮，[EXP-K03《CUDA FA2 forward 简化版版本梯》](records/EXP-K03_cuda_fa2_ladder.md)）:v0 warp-per-row 4.9,v1 smem tile 5.5（+11%,L2 已扛住广播读）,v2 wmma 24.4(4.5x),v3 8 warp 32.5(+33%),v4 cp.async 重叠 34.8 TFLOPS（仅 +7.1%）；全 shape 通过 2e-2 正确性 gate。

GEMM 与 FA2 用同一套 wmma 工具箱得到相反结局——GEMM 够到 cuBLAS 的 86%，FA2 只够到自家 Triton 版（123 TFLOPS，跨 harness）的 28%。原因是结构性的：wmma accumulator fragment 的 lane 到元素的映射未定义，行级 softmax(max/exp/rescale)被迫经由 shared memory 往返，外加每 tile 5 次 `__syncthreads` 的相位链；v4 把 K/V 访存全部预取重叠后只涨 7.1%，说明瓶颈不在访存而在相位链。越是依赖「融合免搬运」的算子，越需要 mma 级寄存器控制——这就是官方 FA2 实现采用 CUTLASS/mma 而非 wmma 的定量理由。

## 跨项目规律

以下四条是各项目里反复出现的硬件规律。

### memory-bound 阶段,shared memory 微优化收益趋近于 0

规律：在 SM 大部分时间等待 HBM 的阶段，shared memory 上的小问题被 HBM 等待时间掩盖，bank conflict 消除、modulo 改位运算之类的小优化几乎没有时延收益。

证据：

- reduce v0 至 v2：bank conflict 削减 50%，时延几乎不变。
- softmax v0 至 v2：bank conflict 削减 70%，时延仅省 2%。
- gemv v0 至 v2：Sec/Ld 变差，时延反而略降（HBM 等待掩盖了 ALU 端的变差）。
- 反例——quantize v0 至 v1：位运算优化把 DRAM% 从 61% 拉到 84%。

反例的解释：quantize 是 memory-bound 加简单 ALU，指令路径短（没有 `__syncthreads` 这类大头），ALU 端的小优化才能浮出来；reduce / softmax / gemv 的指令路径里夹着 `__syncthreads`，那些才是大头，v1 的几个 cycle 完全被掩盖。

（本条证据为 Laptop 时代的 NCU / 时延对，用于讲机制；现行头条数字见前文各项目节。）

推论：优化技巧的有效性，取决于它优化的环节在总时间中的占比。不存在普适有用或普适无用的优化技巧，只存在匹配当前瓶颈的优化。

### 特化 scope 内,hardware-optimal 胜过 algorithm-optimal

规律：当手写 kernel 赢过通用库（cuBLAS / cuDNN / PyTorch）时，很少是因为算法更聪明，绝大多数是因为手写侧能假设通用库不能假设的东西（输入对齐、shape 固定、dtype 固定），从而把硬件压得更狠。前提是还有余量可赢——瓶颈一旦已经是物理墙，双方都只能贴着同一条线。

四类赢法，外加一条边界：

| 项目 | 结果 | 手写侧的赢法 | 通用库的优势 |
|---|---|---|---|
| reduce(4090，HBM-bound) | v7 与官方 CUB 相差 **0.7%**（93.9% 对 94.5% 理论峰值，3 轮，EXP-K04） | 单一 shape 的 two-pass 特化把 DRAM 压到物理墙 | 无余量可赢——同一条 DRAM 带宽线 |
| reduce（4090，L2 常驻） | CUB 比 v7 快 **33.3%**（3 轮，EXP-K04） |—— | 分尺寸 tuning（延迟隐藏 / 展开度 / 两阶段策略） |
| softmax(4090) | v4 比 cuDNN 快 **6.7%**（对齐 1024x1024）；非对齐 1024x1500 反被 cuDNN 快 9.9%（3 轮，EXP-K04） | 对齐形状上的 float4 + warp shuffle 特化 | 所有形状都不塌 |
| gemv(4090) | v3 比 `cublasSgemv` 快 **34.1%**（3 轮，EXP-K04；**限 L2 常驻区间**，冷读时收敛到 1.4%，EXP-K09 §5.8） | DRAM% 95% 对 95%（warp shuffle 极简结构） | L2 命中率 21% 对 2.7%(column-major tiling) |
| quantize | v4 比 PyTorch eager 快 **6.6x** | 1 kernel 对 4 kernel（融合避免中间 tensor） | 灵活性（eager 模式支持动态图） |

推论：

1. gemv 一例是「手写侧赢硬件、通用库赢算法」；softmax 一例是「用更窄的假设换性能」的典型——赢在对齐形状，输在非对齐形状，收益与边界是同一件事的两面。
2. reduce 两行给出这条规律的边界条件：瓶颈已经是物理墙时，谁也赢不了多少（差 0.7%）；数据一旦装进 L2、余量重新出现，厂商库的分尺寸 tuning 就赢回 33.3%。先问「还有多少余量」，再问「谁写得更好」。
3. quantize 一例展示了融合是第三种赢法——与 Flash Attention 的思路一致。
4. scope 意识：v3 的领先在 mat 不超过 L2 的尺寸上稳定成立，mat 远大于 L2 时优势可能消失；softmax 的 6.7% 只在对齐形状上成立。这正是 LLM 推理框架为每个 shape 维护 specialized kernel 的原因——也是它们必须为每个 shape 各测一遍的原因。

### NCU 指标必须与时延趋势联读

规律：单看一个 NCU 指标的变化无法判断好坏，必须配合时延趋势反推机制；同一个指标的相同变化，在不同场景下可能意味着完全相反的事情。

同指标反含义的案例：

| 案例 | 指标变化 | 看似 | 实际 |
|---|---|---|---|
| quantize v4 | L2Hit% 32% 至 2% | 劣化 | 优化：消除了 read-modify-write 的补偿读 |
| gemv v3 | StallLSB% 76% 至 95% | 劣化 | 优化：SM 全部时间在等 HBM，没有资源浪费 |
| softmax v4 | SM% 71% 至 40% | 劣化 | 优化：计算开销被削减，HBM 压满 |
| softmax v3 / gemv v3 | Sec/Ld = 16 | 劣化 | float4 满载（每 lane 16 字节 x 32 lane = 16 sector） |

（原表中 reduce v6/v7 两行随「v6/v7 慢 6x」叙事降级而撤下——旧数据不可确证，EXP-K01 §5；原行见 docs/archive/。）

推论：profiler 给出的是现象，原因要靠推理。看 NCU 的关键不是背指标含义，而是把指标变化、时延变化、算法路径三个证据组合起来反推机制——这是从「会跑 profiler」到「能用 profiler 解决问题」的区别。

### 前置优化需要后置环节激活才能兑现

规律：单链路的优化投资在被激活前收益有限，前置优化（bank conflict 消除、向量化）需要等后置优化（warp shuffle、融合）激活才能体现价值；反过来，后置环节的退化能拖垮所有前置优化（Amdahl 定律）。

证据：

| 项目 | 前置优化 | 激活前 | 激活后 |
|---|---|---|---|
| reduce | float4 | 单看 v3 收益 12% | 加 warp shuffle 后总收益 60%（v3 至 v4 提升 30%） |
| softmax | float4 | v4.2 反例：拿掉 warp shuffle，向量化收益归零 | 加 warp shuffle 后 v4 比 v3 快 30% |
| gemv | warp shuffle 使 shared memory 不再必要 | v0-v2 用 shared memory 慢 | v3 弃用 shared memory 后比 cuBLAS 快 19% |
| quantize | float4 read 与 char4 store 配套 | 只优化 read 端收益有限 | read 与 store 都向量化后才取得最后 12% |

后置退化拖垮前置的反例：

- softmax v4.4：main 循环保留 float4（带宽利用率 100%），归约阶段制造 bank conflict 加全程同步，整体慢于自写 warp 参照 8%。
- gemv v4：v3 已是最优，加 shared memory 缓存 vec，L1 多搬一次反而慢一倍。

推论：优化是链，不是点。工程含义有两条：其一，单链路投资需要等下游环节激活才能兑现，短期看不到收益不代表方向错误；其二，任何被忽略的环节都可能成为新的瓶颈，最热的循环再快也救不了链路上的一个 `__syncthreads`。

## 算子带宽画像

各项目共同验证：不同算子的 DRAM% 上限取决于算法结构本身，而非优化好坏。

| 算子 | 最优版本 DRAM% | 上限原因 |
|---|---|---|
| reduce | 96% | 纯流式读加极少同步，HBM 几乎完全打满 |
| gemv | 95% | mat 流式读，与 reduce 类似 |
| softmax | 76% | reduce、计算、写回的复合算子，两次同步穿插 |
| quantize | 85% | 读 fp32 写 int8，dtype 不对称（4:1），write 端带宽天然低 |

（本表为 Laptop 时代 NCU 采集；4090 容器内 NCU 不可用（EXP-K01 §7），结论按算子结构解读，不依赖具体卡。）

此表说明 DRAM% 不是越高越好的孤立指标，要与算子结构对照解读：76% 对应 softmax 的算法上限，96% 对应 reduce 的算法上限，二者不能横向比较。

## 文档索引

按主题定位仓内的详细材料：

| 主题 | 文档 |
|---|---|
| GPU 执行模型 / SIMT / warp divergence | `cuda-reduce/project-proof/docs/interview-analysis-v7.md` |
| online softmax / Flash Attention | [EXP-K03](records/EXP-K03_cuda_fa2_ladder.md) 与 `flash-attn/README.md` |
| bank conflict 推导 / shared memory 陷阱 | `gemv/GEMV_VS_CUBLAS_ANALYSIS.md` |
| LLM 量化 / 融合的重要性 | `int8-quantize/project-proof/docs/why-each-version-changed.md` |
| 精度 / fp16 / 量化精度 | [EXP-K02](records/EXP-K02_cuda_gemm_tc_ladder.md) §5 与 [EXP-K03](records/EXP-K03_cuda_fa2_ladder.md) §5 的正确性 gate |

每个项目另有独立的详细稿与 NCU 摘要；本文是顶层入口。
