# EXP-K04 · 标准库基准补齐与两区间重测(CUB / cuDNN 入场)

> **一句话结论**：真 HBM-bound 时，手写 reduce 与官方 CUB 只差 **0.7%**（分别达理论带宽的 93.9% 与 94.5%）；数据一旦装进 L2，CUB 反超 **33.3%**。对照物必须同算子、同区间，否则「手写快多少」没有意义。

## 0. 元信息

| 字段 | 值 |
|---|---|
| 环境 | RTX 4090（L2 72.0 MB，HBM 理论 1008.1 GB/s，driver 610.57.04，CUDA 13.2） |
| 状态 | 完成 |
| 关联 | EXP-K01《四 kernel 4090 重基准》（4090 首轮重测）的整改闭环；softmax 对照物勘正的最终落地 |

## 1. 目的与假设(跑前锁定)

四个 memory-bound 项目里，只有 gemv 有同算子的标准库对照（`cublasSgemv`）。另外两个都不合格：softmax 的所谓 "cublas" 其实是自写 kernel（EXP-K01 §5 已判作废）；reduce 用 `cublasSasum`（Σ|x|）代替 Σx，**算子根本不是同一个**。

没有同算子标准库对照，「手写快多少」这句话就不成立。本实验补齐真正的基准并重测，跑前锁定三条可证伪假设：

- **H1（reduce 的正确基准）** —— 应为 CUB `DeviceReduce::Sum`（CCCL 随 toolkit 分发的同算子官方实现），而非 BLAS 的 asum。
- **H2（softmax 的正确基准）** —— 应为 cuDNN `cudnnSoftmaxForward`。BLAS 规范里没有 softmax，深度学习标准算子属 cuDNN。
- **H3（测量效度）** —— 经典 reduce 规模 N=1<<24（67.1 MB）小于 4090 的 72 MB L2，测得的其实是 L2 带宽而非 HBM 带宽；需另设 N=1<<28（1.07 GB）才进入 HBM-bound 区间。**判据**：若等效带宽超过 1008 GB/s 理论峰值，即证明数据没落到 DRAM。

## 2. 环境与配置

新增两个标准库对照实现：

- `cuda-reduce/src/reduce_cub.cu` —— CUB 两段式调用，temp_storage 分配在计时外。
- `softmax/src/softmax_cudnn.cu` —— NCHW=(rows,cols,1,1) + `MODE_INSTANCE` + `SOFTMAX_ACCURATE`，与本仓 v0..v4 同为「减最大值」的数值稳定口径。

配套改动：

- softmax 原 `cublas` 标签全链改名 **`handwritten_ref`**（自写参照，不是厂商库）。
- reduce 规模改为可配（`REDUCE_N`），并加 `REDUCE_SKIP_BASELINE` —— 1 GB 下单线程 baseline 每轮约需 8 分钟。
- 每组 3 轮，`scripts/stability_rebench.sh` 出 UTC 前缀 raw，聚合入 records/data/。

## 3. 原始数据

| 区间 | raw | 聚合 |
|---|---|---|
| L2（N=1<<24） | `*/project-proof/data/2026*_<proj>_stability_r{1,2,3}.csv` | `records/data/exp_k04_{cuda_reduce,softmax,gemv,int8_quantize}_3rounds.csv` |
| HBM（N=1<<28） | `cuda-reduce/project-proof/data/*_hbmbound_N268435456_r{1,2,3}.csv` | `records/data/exp_k04_reduce_hbmbound_3rounds.csv` |

设备参数实测：L2 = 72.0 MB，bus = 384-bit，memclk = 10.50 GHz，据此算得理论带宽 1008.1 GB/s。

## 4. 结果

### 4.1 reduce:两个区间,两个结论(3 轮 mean±std)

**HBM-bound 区间（1.07 GB，数据必须落 DRAM）**

| 版本 | 时延 (ms) | 等效带宽 (GB/s) | 占理论峰值 |
|---|---:|---:|---:|
| **CUB**（官方，同算子） | 1.12730±0.00013 | 952.5 | 94.5% |
| **v7**（自写最优） | 1.13483±0.00023 | 946.2 | **93.9%** |
| cuBLAS Sasum（异算子） | 1.14722±0.00103 | 935.9 | 92.8% |
| v6 | 1.161 | 924.9 | 91.7% |
| v4 | 1.452 | 739.6 | 73.4% |
| v0 | 1.807 | 594.3 | 59.0% |

**L2 常驻区间（67.1 MB，只报时延）**

| 版本 | 时延 (ms) |
|---|---:|
| **CUB** | 0.019828±0.000098 |
| v7 | 0.029740±0.000124 |
| cuBLAS Sasum | 0.037181±0.000079 |

三条读法：

- **HBM-bound：v7 与 CUB 差 0.7%**（1.13483 vs 1.12730），双方都在理论峰值 94% 上下。
- **L2 常驻：CUB 快 33.3%**（时间比 1.50×）。
- 该区间三个版本的等效带宽是 HBM 理论峰值的 1.8–3.4 倍（3384 / 2256 / 1805 GB/s）—— 物理上不可能，**H3 成立**：数据常驻 L2，测的是 L2 带宽，所以此表不报带宽。

### 4.2 softmax:形状敏感性(3 轮 mean±std)

| 形状 | v4 自写最优 (ms) | cuDNN (ms) | 判定 |
|---|---:|---:|---|
| 1024×1024（对齐） | 0.007768±0.000103 | 0.008291±0.000023 | **v4 快 6.7%** |
| 1024×1500（非对齐） | 0.009832±0.000045 | 0.008947±0.000095 | **cuDNN 快 9.9%** |

同表内 `handwritten_ref`（旧 "cublas" 标签）为 0.009651 / 0.012724 ms，仅作历史参照。

### 4.3 gemv(4096×2048,3 轮)

v3 为 0.012740±0.000064 ms，`cublasSgemv` 为 0.017084±0.000156 ms —— **v3 快 34.1%**。

EXP-K01 §7 同协议前一轮测得 37.8%，两轮差异来自 cuBLAS 侧的轮间波动 ±2pp。对外取本轮较保守值，并注明区间。

### 4.4 int8-quantize

该算子不属 BLAS/cuDNN 规范，harness 内没有同算子厂商库可比。v4 为 0.005639±0.000194 ms；PyTorch eager 对照见 triton-kernels#EXP-T03《三件套移植 + torch 绑定》（Laptop 口径，单轮）。

## 5. 分析与结论

**1. 手写打不打得过厂商库，取决于瓶颈是不是已经撞到物理墙。**
真 HBM-bound 时，CUB 与自写 v7 都贴到理论峰值 94% 附近，差距被压到 0.7% —— 两者被同一条 DRAM 带宽线焊住，代码优劣的腾挪空间只剩百分之一量级。这解释了 memory-bound 算子「优化到头」的现象，也说明这类算子的正确目标是**逼近峰值**，而不是超越对手。

**2. L2 常驻区间才是厂商库拉开差距的地方（快 33.3%）。**
数据不再受 DRAM 限制后，胜负回到延迟隐藏、展开度、tile 尺寸与两阶段规约策略 —— CUB 按架构分派的调参正是为此存在。自写 kernel 想追平，需要的是分尺寸调参，而不是再省一次访存。

**3. 测量效度是一等公民。**
经典的 16.7M 元素规模，在 72 MB L2 的卡上会静默变成 L2 基准。不做「等效带宽 vs 理论峰值」的合理性检查，就会把 L2 带宽当成 HBM 带宽汇报。本仓此前跨机比较（4070 Laptop 32 MB L2 vs 4090 72 MB L2）之所以出现版本排序变化，更可能的解释是**两机根本不在同一区间**，而非单纯的 roofline 迁移。

**4. 对照物必须同算子。**
`cublasSasum` 算的是 Σ|x|，与 Σx 语义不同。它在 HBM 区间也慢于 CUB（92.8% vs 94.5%），说明 BLAS 的规约路径本就不是为纯求和优化的。

**5. softmax 的手写优势只在对齐形状成立（+6.7%）。**
非对齐形状下 cuDNN 反超 9.9% —— 厂商库的价值，很大一部分就是「所有形状都不塌」。

## 6. 异常、偏差与开放问题

- **L2 区间不报带宽百分比** —— 等效带宽超理论峰值，报了即错。
- **HBM 区间部分版本 std 偏大** —— v6/v3/v5 在 0.04 ms 量级，疑与 grid-stride 版本对 L2 命中的敏感度有关，未深究；v7/v4/CUB 的 std 均在 1e-3 ms 量级。
- **baseline 只采了一轮** —— 单线程 GPU reduce 在 1 GB 下每轮约 8 分钟，3 轮中仅 r1 采集，r2/r3 用 `REDUCE_SKIP_BASELINE=1` 跳过；该行加速比在 HBM 区间无意义。
- **gemv 两轮 34.1% vs 37.8% 未逐轮溯源**（cuBLAS 侧波动），开放。
- **cuDNN 9.8 未扫 `CUDNN_SOFTMAX_FAST`** —— 非稳定口径，与本仓算法不同，不做对照。

## 7. 下游影响

**作废**：EXP-K01 §5/§7 与各处「reduce v7 反超真 cuBLAS 24.5%」的对外用法 —— 该比较基于异算子 asum，且处于 L2 区间。现行口径见 §4.1 两区间表。

**新增可用口径**：

- HBM-bound 区间手写 reduce 达理论峰值 93.9%，与官方 CUB 差 0.7%
- L2 区间 CUB 快 33.3%
- softmax 对齐 +6.7% / 非对齐 −9.9%，对照 cuDNN
- gemv 快 `cublasSgemv` 34.1%

**待同步**：三份简历、README/PORTFOLIO/LEDGER、讲义中的相关数字。
