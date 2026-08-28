# EXP-K07 · 采集主机 NCU 计数器闭环：分管线利用率、GEMM 对照口径核验、fused-norm L2 命题转实测

> **一句话结论**：在计数器权限打开的 4090 虚机上补齐 18 份报告，把三条长期挂着"推断"的命题转成实测——GEMM v4 与 cuBLAS 的性能差距（77.9%）与两者 Tensor 管线利用率之比（77.7%）在 0.2 个百分点内重合；fused-norm 的"第二次读被缓存接住"由 DRAM 读扇区恒为 2.000×S 直接证实；并从计数器侧独立复现了"16 B 向量化未兑现"——v3/v4 的 L1TEX 全局读扇区是 v1/v2 的 4 倍。

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
| v4 | **2.000** | 1.861 | **12.000** | 2.019 | 2.000 | 0 / 0 |

主判据 `R = DRAM读(v3) / DRAM读(v4) = 16,781,504 / 16,779,844 = **1.0001**`
（判据：R ≤ 1.10 证实 / R ≥ 1.35 证伪）→ **证实**。

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

### 6.3 H3 成立：DRAM 读恒为 2.000×S

四个版本的 DRAM 读扇区数**全部等于 2.000×S**，与版本无关。
字节账模型预期一趟全张量读 = S，v3 逻辑上读三趟（residual + x + 重读）、v4 读两趟；
若重读出片，R 应为 1.50；实测 R = 1.0001。**那次重读一个扇区都没有到过显存。**

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
3. `rope` v2 在 HBM 区间比 v1 慢 1.1%（§4 第 4 条）本轮**未采**（rope/activation/w8a8 按指示跳过），仍是推断。

## 8 下游影响

**可立即回写（证据已在本记录）**：

| 文件 | 动作 |
|---|---|
| `docs/lectures/03_memory_bound_fusion.md` 177 / 183 / 448 / 499 行 | "推断"改"实测"，并更正指标名；"边界四"整条改写为沿革句 |
| `docs/ncu_reading_guide.md` §4 第 1 条、第 51 / 60 行 | 移入"已转实测"；更正指标名；补读数路径说明 |
| `records/EXP-K05` 97 / 125 行 | 史料不覆写，追加"（后续闭环见 EXP-K07）"注记 |
| `LEDGER.md` backlog ① | 销账；红线表中"swizzle/smem 是 FA2 剩余差距主因"仍**不可解锁**（本轮未采 bank conflict 分解） |

**须谨慎处理**：

- README/PORTFOLIO 的 GEMM **85.6%** 是主力机 CUDA 13.2 口径，本轮 77.9% **不取代它**，
  两者是不同工具链下的两个数。对外引用维持 85.6%，但需知该数字对工具链敏感。
- 本记录所有绝对时延数字**不得**进入 benchmark 表或简历。
- `fused-norm` 三个算子 README 声称的"16 B 向量化"在兑现之前，**不应作为优化项对外陈述**。

**不做的事**：本机不改算法、不调参（任务书 §6）。向量化修复在主力机进行——那里有 GPU 可立即三轮验收，
且 `docs/lectures/` 有按行号逐字引用源文件的代码块，改动后须跑 `verify_lectures.py` 重定位。
