# EXP-K07 · 采集主机 NCU 计数器闭环：分管线利用率、GEMM 对照口径核验、fused-norm L2 命题转实测

> **一句话结论**：在计数器权限打开的 4090 虚机上补齐 18 份报告，把 `ncu_reading_guide` §4 中**三条**长期挂着"推断"的命题（第 1、2、5 条）转成实测——GEMM v4 与 cuBLAS 的性能差距（77.9%）与两者 Tensor 管线利用率之比（77.7%）在 0.2 个百分点内重合；fused-norm 的"第二次读不出片"由 DRAM 读恒为 2.000×S（= 算法下界）证实，且机制精确到 **L1**（命中率 83.2%）而非原文所说的 L2（命中率仅 0.94%）；FA2 的 `short_scoreboard`（smem）stall 在 v4 上达 50.13%、而全局访存 stall 已降至 0.31%，坐实"smem 往返是剩余瓶颈"；
> 并从计数器侧独立复现了"16 B 向量化未兑现"——v3/v4 的 L1TEX 全局读扇区是 v1/v2 的 4 倍。

## 0 元信息

| 项 | 值 |
|---|---|
| 日期 | 2026-08-29 |
| 环境 | **采集主机**（非主力机）：RTX 4090 虚机，nvcc 12.8.93，ncu 2025.1.1.0，driver 570.153.02，torch 2.11.0+cu128 |
| 状态 | 完成 |
| 关联 | 解锁 EXP-K01 §7 的 `ERR_NVGPUCTRPERM`；闭合 `docs/ncu_reading_guide.md` §4 第 1、2、5 条；EXP-K05 §7 待办①销账；对照 EXP-K02（GEMM 85.6% 口径） |
| 前置 | `PROFILING_HOST_TASKS.md`（临时任务书，本轮执行对象） |

**口径警告（贯穿全文）**：本记录所有**绝对时延/吞吐**数字来自 CUDA 12.8 工具链，
与主力机（CUDA 13.2）的 benchmark 表**不得混排成同一行**。
**计数器类结论**（管线利用率、扇区计数、stall 分解）跨工具链可用。

## 1 目的与假设

主力机是容器，`RmProfilingAdminOnly=1` 且无 `CAP_SYS_ADMIN`，Nsight Compute 取不到性能计数器。
本轮在一台 GPU 直通的虚机上把 flag 改为 0 并重启，取得计数器权限，做三件主力机做不了的事：

- **H1**（分管线）：v2/v3/v4 用了 wmma，所以"走上了 Tensor Core"——此前只有 SASS 里出现 `HMMA` 这一静态证据，没有运行时利用率。
- **H2**（GEMM 口径）：本机测得 v4 = cuBLAS 的 77.4%，而 README/EXP-K02 写 85.6%。假设：这是 NCU 锁基频造成的假象。
- **H3**（L2 命题）：fused-norm v3→v4 消掉的那次重读"从一开始就被缓存接住、从未出片"，此前是从带宽上界反推的推断。

## 2 环境与配置

| 项 | 采集主机 | 主力机（对照） |
|---|---|---|
| GPU | RTX 4090 · sm_89 · 24564 MiB | 同型号 |
| 驱动 | 570.153.02 | 610.57.04 |
| CUDA | **12.8.93** | 13.2 |
| Nsight Compute | **2025.1.1.0** | 2026.1.0.0（无权限，采不了） |
| 形态 | 虚机，计数器**已开** | 容器，`ERR_NVGPUCTRPERM` |
| power.limit / max | **450.00 W / 450.00 W** | 450 W |
| clocks.max.sm | **3105 MHz** | 3105 MHz |
| torch / ninja / numpy | 2.11.0+cu128 / 1.13.0 / 2.2.6 | — |
| cuDNN | 8.9.7（随 toolkit，在 `/usr/local/cuda-12.8/targets/`） | — |

权限开启方式（一次性，需重启）：

```
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/zz-nvidia-profiler.conf
sudo update-initramfs -u && sudo reboot
grep -i RmProfilingAdminOnly /proc/driver/nvidia/params   # → 0
```

六个 C++ 算子统一以 `-DCMAKE_CUDA_ARCHITECTURES=89` 构建，`cuobjdump -lelf` 核对**六行全为 `sm_89`**
（softmax / gemv / int8-quantize / cuda-reduce 的 CMakeLists 未写编译目标，默认会编成 sm_75）。

## 3 步骤

1. 开计数器权限 → 重启 → `scripts/probe_ncu_permission.sh` 退出码 0。
2. `scripts/setup_profiling_host.sh` 构建六算子并核对编译目标。
3. **GEMM 干净 bench**（不经 ncu，`BENCH_OUT=/dev/null` 保证不写权威数据）3 轮。
4. **bench 期间并发采样时钟/功耗**（`nvidia-smi -lms 200`）。
5. `RUN_NCU_CSV=1 NCU_OPS="gemm flash-attn" bash scripts/run_ncu_all.sh` —— 带分管线指标。
6. 装 torch/ninja/numpy 后 `NCU_OPS="fused-norm"` 采六个臂。
7. `python3 scripts/export_ncu_for_mac.py` 校验导出。

## 4 原始数据

- 报告：各 `*/project-proof/profiling/ncu/*.ncu-rep`，本轮新采 **18 份**，仓内合计 56 份。
- 扩展指标 CSV：`gemm/project-proof/profiling/ncu/gemm_ncu.csv`、`flash-attn/.../fa2_ncu.csv`。
- 导出包：`artifacts/ncu_for_mac/ncu_for_mac_all.tar.gz`（19.8 MB / 56 份），清单 `MANIFEST.md` + `manifest.csv`。
- 校验：**FAIL 0** / WARN 20（全部为"多 grid"，逐条判定见 §7）。
- 硬 gate：`git status --porcelain -- '*/project-proof/data/'` **为空**，权威数据未被 profiler 触碰。

GEMM bench 为 profiler 隔离而走 `BENCH_OUT=/dev/null`，**不落 raw 文件**；
数字仅在本记录内引用，不进任何 benchmark 表（口径不同，见文首警告）。

## 5 结果

### 5.1 分管线利用率（H1）—— 4096³，单一 grid，口径干净

| kernel | grid | Tensor(cyc)% | FMA% | ALU% |
|---|---|---|---|---|
| `ampere_fp16_s1688gemm_fp16_128x128`（真 cuBLAS） | (32, 32, **3**) | **49.02** | 0.36 | 2.93 |
| `gemm_v4_kernel` | (32, 32, 1) | **38.08** | 4.03 | 8.70 |
| `gemm_v3_kernel` | (64, 64, 1) | 26.96 | 5.17 | 10.59 |
| `gemm_v2_kernel` | (64, 64, 1) | 25.72 | 4.43 | 11.90 |
| `gemm_v1_kernel` | (128, 128, 1) | **0.00** | 22.34 | 2.31 |
| `gemm_v0_kernel` | (256, 256, 1) | **0.00** | 27.81 | 15.49 |

FA2（多 grid，按 grid 分组；下表取 S=4096 协议点 `(64,32,1)`）：

| kernel | Tensor(cyc)% @(64,32,1) | 跨 6 种 grid 的区间 |
|---|---|---|
| `fa2_v4_kernel` | **10.30** | 9.38 – 10.30 |
| `fa2_v3_kernel` | 9.63 | 8.54 – 9.63 |
| `fa2_v2_kernel` | 7.23 | 6.30 – 7.23 |
| `fa2_v1` / `fa2_v0` / `ref` | **0.00** | 0.00 |

### 5.2 GEMM 对照口径（H2）—— 3 轮，4096³

| | v4_bigtile | cuBLAS | 比值 |
|---|---|---|---|
| 本机干净 bench（3 轮） | **127.9 ± 0.2** TFLOPS | **164.3 ± 0.8** TFLOPS | **77.9%**（轮间 77.5–78.2%） |
| 本机 NCU 环境（上一轮） | — | — | 77.4% |
| 主力机 CUDA 13.2（EXP-K02） | 133.1 ± 0.97 | ~155.5（反推） | 85.6% |

bench 期间实测：**SM 2625 MHz**（上限 3105）、**260 W**（上限 450）、**47 °C**；
`Clocks Event Reasons` 中 `SW Power Cap` / `HW Slowdown` / `HW Thermal` / `HW Power Brake` **全部 Not Active**。

### 5.3 fused-norm 字节账（H3）—— HBM regime，grid (32768,1,1)，S = 8,388,608 sector

| 版本 | DRAM读/S | DRAM写/S | L1TEX全局读/S | L2读/S | L2读miss/S | local ld/st |
|---|---|---|---|---|---|---|
| v1 | **2.000** | 2.027 | 4.000 | 2.063 | 2.000 | 0 / 0 |
| v2 | **2.000** | 1.885 | 4.000 | 2.059 | 2.000 | 0 / 0 |
| v3 | **2.001** | 1.853 | **16.000** | 2.126 | 2.000 | 0 / 0 |
| v4 命中率 | — | — | **L1 83.19%** | **L2 0.94%** | — | — |
| v3 命中率 | — | — | L1 86.87% | L2 5.71% | — | — |
| v4 | **2.000** | 1.861 | **12.000** | 2.019 | 2.000 | 0 / 0 |

主判据 `R = DRAM读(v3) / DRAM读(v4) = 16,781,504 / 16,779,844 = **1.0001**`
（判据：R ≤ 1.10 证实 / R ≥ 1.35 证伪）→ **证实**。

### 5.4 FA2 bank conflict 与 stall 分解（C2，对应 LEDGER 红线表）

grid `(64,32,1)`（S=4096 协议点）：

| kernel | bank_conf_ld | bank_conf_st | short_sb% | long_sb% | barrier% | occ% |
|---|---|---|---|---|---|---|
| `fa2_v2_kernel` | 553,385,984 | 87,616,301 | 39.95 | 9.40 | 15.06 | 8.33 |
| `fa2_v3_kernel` | 578,420,736 | 90,792,489 | 45.09 | 7.18 | 13.83 | 16.67 |
| `fa2_v4_kernel` | 578,478,080 | 76,702,763 | **50.13** | **0.31** | 13.86 | 16.67 |

### 5.5 EXP-K01 未能复采的 stall / occupancy（C5）

取每 kernel 的最大 grid（多级算法取第一级，反映真实 HBM 行为）：

| 算子 · kernel | grid | SM% | DRAM% | occ% | short_sb% | long_sb% | barrier% |
|---|---|---|---|---|---|---|---|
| reduce_v0 | (65536,1,1) | 90.16 | 39.05 | 87.99 | 16.75 | 17.83 | 34.97 |
| reduce_v3 | (32768,1,1) | 91.89 | 71.57 | 88.70 | 16.26 | 15.26 | 32.77 |
| reduce_v4 | (32768,1,1) | 50.11 | 96.57 | 77.79 | 6.88 | 72.09 | 8.23 |
| **reduce_v7** | (288,1,1) | 8.64 | **96.35** | 86.39 | 0.08 | **96.86** | 0.74 |
| `asum_kernel`（cuBLAS，**异算子 Σ\|x\|，不可作对照**） | (216,1,1) | 9.42 | 94.64 | 98.99 | 0.10 | 94.42 | 0.66 |
| gemv_v0 | (4096,1,1) | 42.73 | 95.30 | 95.09 | 4.54 | 80.73 | 5.24 |
| **gemv_v3** | (1024,1,1) | 17.26 | 95.47 | **88.74** | 0.24 | 94.83 | 0.00 |
| `gemv2T_kernel_val`（真 cuBLAS） | (512,1,1) | 19.48 | 94.84 | **63.75** | 0.69 | 80.76 | 5.31 |
| quantize_v0 | (4096,1,1) | 62.13 | 60.94 | 80.94 | 9.68 | 32.08 | 0.00 |
| quantize_v4 | (1024,1,1) | 25.39 | 84.85 | 85.89 | 3.12 | 82.52 | 0.00 |
| softmax_v0 | (1024,1,1) | 76.00 | 58.85 | 91.12 | 12.77 | 18.67 | 26.00 |
| softmax_v4 | (1024,1,1) | 45.42 | 79.38 | 89.27 | 6.10 | 46.37 | 28.46 |
| `softmax_cublas_kernel`（**自写，非 cuBLAS**） | (1024,1,1) | 52.75 | 59.54 | 91.25 | 10.42 | 53.18 | 8.43 |

`softmax_cublas_kernel` 系自写 warp 原语 kernel（LEDGER 红线：cuBLAS 无 softmax API），
此行仅作画像参考，**不得用于任何"对比 cuBLAS"的表述**。


### 5.6 reduce 同算子对照：v7 vs CUB `DeviceReduce::Sum`（补采）

本轮补采 `cub::DeviceReduceKernel`（原 `PROFILE_TARGETS` 未含 CUB 臂，
且 CSV 因实例数上限截断在 asum 之后，故 CUB 从未被采过）：

| | grid | DRAM 读 | 时长 | 等效带宽 | occ |
|---|---|---|---|---|---|
| `reduce_v7`（第一级） | (1024,1,1) | **67.11 MB** | 73.73 / 73.79 μs | 910.3 / 909.5 GB/s | 87.3% |
| `cub::DeviceReduceKernel` | (3840,1,1) | **67.13 MB** | 73.22 / 73.09 μs | 916.9 / 918.5 GB/s | 91.6% |

两者 DRAM 读字节数差 **0.03%**，时长差 **0.8%**。

### 5.7 gemv 同算子对照：v3 vs 真 cuBLAS `gemv2T_kernel_val`

| | grid | DRAM 读 | 时长 | 等效带宽 | occ |
|---|---|---|---|---|---|
| `gemv_v3` | (1024,1,1) | **33.568 MB** | 37.82 μs | 886.7 GB/s | 62.1% |
| `gemv2T_kernel_val`（真 cuBLAS） | (512,1,1) | **33.571 MB** | 38.35 μs | 875.7 GB/s | 32.6% |

DRAM 读字节数差 **0.01%**，时长差 **1.4%**。


## 6 分析与结论

### 6.1 H1 成立，且给出量化靶子

v0/v1 的 Tensor 利用率**恰为 0.00%**，把"非 wmma 版本不走 Tensor Core"从 SASS 静态证据变成运行时实测。
v2→v3→v4 单调上升（25.72 → 26.96 → 38.08%），wmma 路线确实在把负载往 Tensor 管线上搬。

FA2 的混样虽在（6 种 grid），但**各版本区间完全不重叠**——v2 的最高值 7.23% 低于 v3 的最低值 8.54%，
所以 v2 < v3 < v4 的排序在任意单一 grid 下都成立，混样未污染结论。
FA2 最高仅 **10.30%**，与 EXP-K03"= 自家 Triton 28%"同向。

### 6.2 H2 被推翻：不是 NCU 假象，是工具链差异

三条独立证据：

1. **干净 bench 77.9% 与 NCU 环境 77.4% 仅差 0.5 pp**——若 NCU 锁频是主因，两者应显著分离。
2. **时钟/功耗上限与主力机完全一致**（450 W / 3105 MHz），实测 260 W、47 °C，无任何 throttle 标志，
   本机未被平台压过。
3. **比值对时钟不敏感**：v4 与 cuBLAS 在**同一次 bench、同一时钟条件**下测得，时钟影响分子分母同向、做比值时约掉。
   且实测方向相反——本机 v4 慢 3.8%（133.1→127.9）而 cuBLAS **快** 5.8%（~155.5→164.3）。
   时钟变量只能让两者同向变化，无法解释反向。

因此 85.6% → 77.9% 归因于**工具链**：12.8 的 ptxas 编出的 v4 更差，且 12.8 的 cuBLAS 在此形状上更强。

**最有力的互证**：v4/cuBLAS 的**吞吐比 77.9%** 与两者**Tensor 管线利用率之比 38.08/49.02 = 77.7%**
在 0.2 pp 内重合。这说明差距就是 Tensor 管线喂饱程度的差距，而非调度或访存。
`v5 = mma PTX + ldmatrix + smem swizzle` 由此获得一个具体靶子：**补上 11 pp 的 Tensor 利用率缺口**。

一条待查线索：cuBLAS 的 grid 是 `(32, 32, 3)`，**z = 3 说明用了 split-K**，而 v4 是 `(32,32,1)`。
这可能正是利用率差距的来源之一。

### 6.3 H3 成立，但机制层级须修正：接住它的是 L1，不是 L2

四个版本的 DRAM 读扇区数**全部等于 2.000×S**，与版本无关。
字节账模型预期一趟全张量读 = S，v3 逻辑上读三趟（residual + x + 重读）、v4 读两趟；
若重读出片，R 应为 1.50；实测 R = 1.0001。**那次重读一个扇区都没有到过显存。**

**⚠️ 机制层级修正**：原文（讲义 §3.3、`ncu_reading_guide` §4 第 1 条、白板卡片）写的是"被 **L2** 接住"。
计数器显示这是错的——三层扇区账把收敛点钉在 L1：

| 层级 | v4 | 说明 |
|---|---|---|
| L1TEX 读请求（指令层） | 12.000×S | 未向量化导致的放大，见 §6.4 |
| L2 读（L1 未命中） | 2.019×S | **L1 已把 12× 收敛到 2×** |
| DRAM 读（L2 未命中） | 2.000×S | 恰为算法下界（residual + x 各一遍） |
| L1 命中率 | **83.19%** | 与 1 − 2/12 = 83.33% 吻合 |
| L2 命中率 | **0.94%** | 到达 L2 的扇区 99% 都 miss 到 DRAM |

**L2 在这条路径上几乎没有参与**（命中率不到 1%），收敛全部发生在 L1。
这在硬件上也讲得通：一行 8 KB 本就该落在 L1（Ada 每 SM 128 KB L1/smem），
用不着 72 MB 的 L2。原表述把机制说到了错误的层级。

附注：主判据 R = DRAM(v3)/DRAM(v4) = 1.0001 这个**数字是对的**，但它证明的是"DRAM 读不变"，
**不能**用来证明"L2 接住了"——R 接近 1 与 L2 是否参与无关。
真正的证据是另外两个数：**DRAM = 2.000×S 下界** + **L1 命中率 83.2%**。

自洽校验全部通过：DRAM 写 ≈ 2.0×S（residual + out）；`local ld/st = 0`
说明 v4 的寄存器缓存没有溢出到 local memory——这正是 `fused_norm_v4.cu:21` 自己写下的翻车信号，未触发。

闭环：证实情形下 (2S+2S)×32 B / 1.1667 ms = 920 GB/s，与 3 轮实测 v3 = 920.33 ± 0.67 GB/s 精确吻合；
证伪情形下需 1150 GB/s > 1008 GB/s 物理峰值，本就不可能。上界反推与计数器实测同向。

### 6.4 计划外发现：向量化未兑现，计数器侧独立复现

L1TEX 全局读扇区：**v1/v2 = 4.000×S，而 v3/v4 = 16.000 / 12.000×S**——号称做了 16 B 向量化的版本，
访存事务反而是标量版本的 4 倍。

根因在源码：

```
fused-norm/src/fused_norm_v3.cu:24  // 16 字节 = 8 个 bf16 = 4 个 bf16x2。alignas(16) 让编译器放心发 LDG.E.128;
fused-norm/src/fused_norm_v3.cu:26  struct alignas(16) BF16x8 { __nv_bfloat162 h[4]; };
```

`alignas(16)` 只保证**地址对齐**，不强制向量化 load。nvcc 按成员类型（`__nv_bfloat162`，4 B）逐个生成访存，
出来是四条 `ld.global.v2.u16` 而非一条 128 位。**注释断言的正是没有发生的事。**
`rope/src/rope_v3.cu:18` 是同一个结构体、同一个错。

SASS 侧独立证据（另一路核查）：`fused-norm v3` / `activation v2` / `rope v3` 的 `LDG.E.128` **计数为 0**；
对照 `int8-quantize v4`（原生 `float4`）为 1。两条路径互证。

**它顺带解释了一个旧观测**：README 记载"v3 向量化在 HBM 区间零收益（920.1 → 920.3 GB/s）"，
旧归因是"带宽已饱和"。实际是双重原因——既没有真正向量化，而且即使多发了 4 倍 L1TEX 事务，
**L1/L2 全部吸收，DRAM 侧一字节没多**（2.000×S 不变），所以时间当然不变。
"带宽已饱和"这个归因是对的，但它掩盖了一个更基础的事实：向量化压根没发生。

数量自洽：v3 − v4 的 L1TEX 差值 = 4×S，恰为"一趟读 × 4 倍 sector 放大"，反过来佐证放大倍数就是 4。
（朴素预期的 1×S 只在向量化兑现的前提下成立。）

### 6.5 C2 成立：smem 确是 FA2 的剩余瓶颈（红线可解锁）

LEDGER 红线表中「swizzle / smem 往返是 FA2 剩余差距主因」此前标注"推断，不可当实测说"。

`short_scoreboard` 是 shared memory 访问的 stall 类型，`long_scoreboard` 是全局访存延迟。实测：

- v4 把**全局访存延迟几乎完全藏住**：long_sb 由 v2 的 9.40% 降到 **0.31%**——overlap 设计生效。
- 与此同时 short_sb 升到 **50.13%**，是第二名 barrier（13.86%）的 **3.6 倍**，占全部 stall 的约一半。
- v2 → v3 → v4 的 short_sb 单调上升（39.95 → 45.09 → 50.13%）：每优化一版，瓶颈就更向 smem 集中一分。

**判据满足：在 v4 上 smem 相关 stall 是绝对主导项，且全局访存已不再是瓶颈。该推断转为实测。**

附注：v4 的 bank_conf_st 比 v3 低 15.5%（76.7 M vs 90.8 M）而 bank_conf_ld 基本持平，
说明 v4 的改动作用在写侧；读侧的 578 M 冲突未被触及——这正是 swizzle 要解决的部分。

### 6.6 C5 成立，并给出两条计划外线索

**（a）reduce 的两个 regime 在计数器上泾渭分明**，印证 EXP-K04 的两区间口径不是纸面划分：

| | v0–v3 | v4–v7 |
|---|---|---|
| SM% | 90–92 | 8.6–59.5 |
| DRAM% | 39–72 | 96.3–96.7 |
| barrier% | 31–35 | 0.7–9.5 |
| long_sb% | 15–19 | 72–97 |

v0–v3 受 barrier/计算约束，v4–v7 是纯 DRAM 墙。**在 v0–v3 区间报带宽百分比确实会误导**
（LEDGER 红线「reduce 带宽百分比只在 HBM-bound 区间可报」由此获得计数器层面的支持）。

**v7 与 CUB 的对照（§5.6，同算子）**：两者 DRAM 读字节数差 0.03%（67.11 vs 67.13 MB），
都贴在 910–918 GB/s（峰值 1008 的 90–91%），时长差 **0.8%**——与 EXP-K04 记载的「与 CUB 差 0.7%」吻合。
机理是：**搬同样多的字节、撞同一堵带宽墙，CUB 靠略高的 occupancy（91.6% vs 87.3%）拿走那不到 1%**。

**⚠️ 一处已修正的归因错误**：本记录初版曾用 `asum_kernel` 的画像重合来解释「与 CUB 差 0.7%」。
这是错的，且是 LEDGER 红线第一条（reduce「反超 cuBLAS 24.5%」作废，理由为「异算子 asum」）的同型错误：
`cublasSasum` 算 Σ\|x\|，与 reduce 的 Σx **不是同一个算子**（EXP-K04:115 已判定该类用法作废）；
而「差 0.7%」的对照物是 **CUB `DeviceReduce::Sum`**，不是 asum。两个对照物不可互换。
现已补采 CUB 臂重做（§5.6）。asum 的画像仅作参考，**不得用于任何对照结论**。

**（b）gemv：一条候选归因被自己的数据否掉（EXP-K01 §7 / EXP-K04 开放问题）**

改用 **DRAM 字节总量**（而非吞吐百分比）作判据后（§5.7）：

- **"v3 读得更少"被排除**：两者 DRAM 读字节数差 **0.01%**（33.568 vs 33.571 MB）。搬的字节一样多。
- **occupancy 不是主因**：v3 的 occupancy 接近 cuBLAS 的两倍（62.1% vs 32.6%），
  时长却只差 **1.4%**。纯 DRAM-bound 时带宽已被喂饱，多给 warp 换不来时间。
  本记录初版把 occupancy 列为"v3 快 34.1%"的机理候选，**此处撤回**。
- **更重要的是：34% 在计数器环境下没有复现**。NCU 采到的这个点上 v3 仅快 1.4%，
  与 EXP-K04 的 34.1% 相差一个数量级。两者不是同一个测量点（bench 扫多个 size，
  profiler 只钉住其中一个），**因此不能说"34.1% 被计数器证实"，也不能说它被证伪**——
  它只是没有被本轮覆盖。要剖清 34.1%，须先让 profiler 钉在 bench 报出该数字的同一 size 上。

**结论：EXP-K01 §7 的"cuBLAS gemv 对照慢 35% 未剖"仍是开放问题**，本轮未能推进，
但排除了"读得更少"与"occupancy"两个候选。


## 7 异常、偏差与开放问题

### 7.1 文档中的指标名不存在（须修正）

`docs/ncu_reading_guide.md`（§4 表、第 51 行）、`docs/lectures/03_memory_bound_fusion.md`（183 行）、
`records/EXP-K05`（125 行）、`LEDGER.md`（backlog ①）**四处**写的判据指标是 `lts__t_sectors_op_read`。
该指标名在 raw 页 **1037 列中一个都不存在**，按它去查必然扑空。

实际可用的等价物：

| 文档写法 | 实际指标名 |
|---|---|
| `lts__t_sectors_op_read` | `lts__t_sectors_srcunit_tex_op_read.sum`（及 `_lookup_hit` / `_lookup_miss`） |
| — | `dram__sectors_read.sum` / `dram__sectors_write.sum`（存在，无需改） |

另需注明**读数路径**：这些扇区计数不在 `--page details --csv`，也不在 `NCU_CSV_METRICS`，
必须走 `ncu -i <rep> --page raw --csv`。现有 `NCU_PROFILE_SECTIONS` 已覆盖，**采集脚本无需修改**。

### 7.2 多 grid WARN 20 份的逐条判定

- **cuda-reduce（8 份）**：grid 呈 `65536 → 256 → 1` 阶梯 = **多级归约树各级**，属合法的 (a) 类。
  读图时须知只有第一级反映真实 HBM 行为。
- **flash-attn（6 份）**：`main.cu:91-94` 有 4 个 correctness case，`main.cu:138` 又扫 `S ∈ {512,1024,2048,4096}`
  = **多 regime 混样**，属 (b) 类，**不可跨实例比较**。本记录已按 grid 分组处理（§5.1）。
- **fused-norm（6 份）**：4 种 grid 对应 bench 的 decode / prefill / hbm 等形状，属 (b) 类。
  本记录只取 `(32768,1,1)` 的 HBM regime（§5.3）。

三类均未重采：(a) 类合法；(b) 类已通过按 grid 分组消除影响，且分组后结论不重叠（§6.1）。

### 7.3 环境踩坑（供任务书修订）

| 问题 | 实际情况 |
|---|---|
| 任务书 §4 的 cuDNN 包名 `libcudnn9-cuda-12` | **不存在**。该 repo 里是 `cudnn9-cuda-12-8`；且 cuDNN 8.9.7 已随 toolkit 装在 `/usr/local/cuda-12.8/targets/`，无需另装 |
| torch 扩展算子 | 需 **ninja**（`RuntimeError: Ninja is required`）与 **numpy**，任务书 T3 未提；缺失时脚本 `set -e` 静默退出，只留半份报告 |
| `RUN_NCU_CSV=1` 的验收 | 验收点是"CSV 里真有 `sm__pipe_tensor_cycles_active`"，**不是脚本跑完了** |
| 分管线指标命名不统一 | Tensor 是 `sm__pipe_tensor_cycles_active`，而 FMA/ALU 是 `sm__inst_executed_pipe_fma/alu`——按前者的模式去 grep 后两者会误判为"没采到" |

### 7.4 开放问题

1. **主力机 cuBLAS 选了哪个 kernel**？本机是 `ampere_fp16_s1688gemm_fp16_128x128`、grid `(32,32,3)`（split-K）。
   若 CUDA 13.2 选了不同实现，则 85.6% → 77.9% 的归因彻底确定。主力机无计数器权限，
   但 `nsys profile --trace=cuda` 即可拿到 kernel 名，不需要计数器。
2. **向量化修好之后还有没有收益**？判据干净：`cuobjdump -sass | grep -c LDG.E.128` 由 0 变正，
   然后看 L2 区间。本轮采到的是**未向量化版本**的数据——修改后需再采一次，两份对照。
3. `rope` v2 在 HBM 区间比 v1 慢 1.1%（`ncu_reading_guide` §4 第 4 条）本轮**未采**（rope/activation/w8a8 按指示跳过），仍是推断。
4. `w8a8` "反量化本可融进 GEMM epilogue"（§4 第 3 条）同样未采，仍是推断。
5. FA2 读侧的 578 M bank conflict 未做 swizzle 前后对照——本轮只证明了 smem 是瓶颈，
   **未证明 swizzle 能解决它**。这两件事不同，对外措辞需区分。

## 8 下游影响

**可立即回写（证据已在本记录）**：

| 文件 | 动作 |
|---|---|
| `docs/lectures/03_memory_bound_fusion.md` 177 / 183 / 448 / 499 行 | "推断"改"实测"；更正指标名；**"被 L1/L2 接住"改为"被 L1 接住"**（L2 命中率仅 0.94%，见 §6.3）；"边界四"整条改写为沿革句 |
| `docs/ncu_reading_guide.md` §4 第 1 条、第 51 / 60 行 | 移入"已转实测"；更正指标名；**"被 L2 接住"改为"被 L1 接住"**；补读数路径说明 |
| `docs/talk/whiteboard_card_byte_ledger.md` | 同上，"被 L2 接住"改为"被 L1 接住" |
| `records/EXP-K05` 97 / 125 行 | 史料不覆写，追加"（后续闭环见 EXP-K07）"注记 |
| `LEDGER.md` backlog ① | 销账（fused-norm L2 命题已实测） |
| `LEDGER.md` 红线表"swizzle/smem 是 FA2 剩余差距主因" | **可由"推断"改为"实测"**，依据 §5.4 / §6.5（short_sb 50.13% vs long_sb 0.31%） |
| `LEDGER.md` 开放问题"cuBLAS gemv 对照慢 35% 未剖" | 补入 occupancy 线索（88.74% vs 63.75%，§6.6b），但**标记为候选而非结论** |

**须谨慎处理**：

- README/PORTFOLIO 的 GEMM **85.6%** 是主力机 CUDA 13.2 口径，本轮 77.9% **不取代它**，
  两者是不同工具链下的两个数。对外引用维持 85.6%，但需知该数字对工具链敏感。
- 本记录所有绝对时延数字**不得**进入 benchmark 表或简历。
- `fused-norm` 三个算子 README 声称的"16 B 向量化"在兑现之前，**不应作为优化项对外陈述**。

**不做的事**：本机不改算法、不调参（任务书 §6）。向量化修复在主力机进行——那里有 GPU 可立即三轮验收，
且 `docs/lectures/` 有按行号逐字引用源文件的代码块，改动后须跑 `verify_lectures.py` 重定位。
