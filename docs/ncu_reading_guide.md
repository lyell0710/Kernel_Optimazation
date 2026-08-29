# NCU 报告阅读指南：把「这一版为什么快」读成证据

本文不重复各版本改了什么——那是各算子 README 的版本梯表与 `docs/lectures/` 的事。
本文只做一件事：**把优化手法映射到 Nsight Compute 里该看的那一个数**，
让「我猜它是因为 X 才快的」变成「报告第 N 行显示 X」。

采集入口 `scripts/run_ncu_all.sh`，导出与口径校验 `scripts/export_ncu_for_mac.py`，
Mac 端打开方式见 `artifacts/ncu_for_mac/MANIFEST.md`。

本文引用的每个 section 都必须在 `scripts/ncu_metrics.inc.sh` 的采集列表里，否则报告里根本没有那一节。改本文时同步查那个文件。
`ComputeWorkloadAnalysis` 是后加进采集列表的，早期那批 Laptop 报告没有这一节；重采之后，`cuda-reduce` / `gemv` / `int8-quantize` / `softmax` 的版本梯报告（各 `*/project-proof/profiling/ncu/*.ncu-rep`）都已带上它。两份例外是 `gemv` 的 `*_l2resident_*` 探针——它们为单一开放问题临时采集，只开了 GPU Speed Of Light Throughput 一节；另有 `softmax/project-proof/profiling/ncu/softmax_smoke.ncu-rep` 是冒烟报告、不属版本梯，只有 SOL / Launch Statistics / Occupancy 三节。

## 1 读一份报告的三步

**第一步：认清自己在看哪一次 launch。** 一份报告常含同一 kernel 的多个实例。
先看 Grid Size：
- 归约类算子的多级树（`65536 → 256 → 1`）只有第一级反映真实 HBM 行为，
  后面几级数据量已极小，SOL 低是必然，不是优化空间。
- 逐元素类算子的 bench 扫多个尺寸（decode / l2 / prefill / hbm），
  **不同 grid 之间不可比**。等效带宽超过硬件峰值就是工作集落在 L2 的信号
  （RTX 4090 的 L2 是 72 MB），此时读到的不是显存性能。

**第二步：先用 Speed Of Light 定瓶颈类型，再往下钻。**

| SOL 读数 | 含义 | 接着看 |
|---|---|---|
| Memory 高、Compute 低 | 访存受限 | Memory Workload Analysis |
| Compute 高、Memory 低 | 算力受限 | Compute Workload Analysis |
| 两者都低 | 延迟受限 | Warp State Statistics + Occupancy |

两者都低是最常见也最容易误判的一档：它意味着硬件大部分时间在等，
既没喂饱访存也没喂饱算力，该查的是 stall 分解与占用率，不是继续压访存。

**第三步：版本梯用 Add Baseline 读。** 打开 v0 设为 baseline，再打开 v1，
Details 页每个 metric 旁会显示相对增减。**只看变化的那几个数**——
一次优化如果只动了一件事（本仓的版本梯规则），那么应该只有少数几个 metric 显著变化，
且变化方向必须与你声称的机制一致。若声称"减少了访存"而 DRAM sector 数没降，
这个版本快的原因就不是你以为的那个。

## 2 优化手法与对应证据

本仓十个算子的版本梯归结起来是下面这几类手法。左列的手法出现在哪些版本，
见各算子 README 的版本梯表。

| 手法 | 出现于 | 该看的 section / metric | 期望方向 |
|---|---|---|---|
| 两个 kernel 融合成一个 | `fused-norm` v0→v1、`activation` v0→v1 | Memory Workload Analysis 的 DRAM sector 读写数 | 中间结果不再出片，DRAM 字节数降一档 |
| warp shuffle 取代 shared memory 归约 | `fused-norm` v1→v2 | Memory Workload Analysis 的 Shared Memory 流量；Warp State Statistics 的 Barrier stall | smem 流量降、barrier 等待降（`__syncthreads` 变少） |
| 向量化访存（16 B / `float4`） | `fused-norm` v3、`rope` v3、`activation` v2、`softmax` v3 | `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` 与 `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`（两者相除即 sector/request，都在 `--page raw` 里）；`l1tex__average_t_sectors_per_request_*` 不在报告里，只在 `scripts/ncu_metrics.inc.sh` 的 `NCU_CSV_METRICS` 旁路 CSV 导出里，本行四个算子中只有 `softmax` 采到（`softmax/project-proof/profiling/ncu/softmax_ncu.csv`） | 同样的字节用更少 request 搬完：**request 条数下降、每 request 覆盖的 sector 数相应上升**。实测 `fused-norm` bf16 标量 2.000 → 16 B 向量化 16.000 sector/request（grid `(32768,1,1)` 下扇区数一个不变，request 条数降到 1/8）；`rope` v2→v3 是 2.000 → 10.000。指针：`fused-norm/project-proof/profiling/ncu/fused_norm_v{2,3}_profile.ncu-rep`、`rope/project-proof/profiling/ncu/rope_v{2,3}_profile.ncu-rep` 的 raw 页 |
| 计算换访存（免查表，`__sincosf` 现算） | `rope` v3→v4 | SOL 的 SM Throughput 与 DRAM Throughput 一升一降 | 表不再读，SFU 压力换来 DRAM 压力下降 |
| 寄存器缓存消掉第二次读 | `fused-norm` v3→v4 | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` / `lts__t_sectors_srcunit_tex_op_read.sum` / `dram__sectors_read.sum` 三层对读，配 `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct`(都只在 `--page raw` 里) | 已实测（S = 一趟全张量的扇区数）：L1TEX 全局读扇区降到 3.000×S，DRAM 读不动，恒停在 2.000×S 的算法下界——接住第二次读的是 L1（命中 33.19%），不是 L2（读命中 0.20%）。详见 §4 第 1 条与 EXP-K09 §5.1 |
| Tensor Core（wmma） | `gemm` v1→v2、`flash-attn` v1→v2 | Compute Workload Analysis 的 Tensor pipe 利用率；SOL 的 SM Throughput | Tensor pipe 从零起来，FMA pipe 让位 |
| cp.async 双缓冲 | `gemm` v2→v3、`flash-attn` v3→v4 | Warp State Statistics 的 `long_scoreboard` stall 占比 | 延迟被藏住＝等访存的 stall 占比下降。这是"双缓冲有没有生效"的直接证据 |
| 大 tile / 增加 warp 数 | `gemm` v3→v4、`flash-attn` v2→v3 | Occupancy（Theoretical vs Achieved）；Launch Statistics 的 Waves Per SM、Registers Per Thread、Static Shared Memory | 关注两处：寄存器或 smem 是否把理论占用率压下去；Waves Per SM 是否为接近整数（非整数＝波量化，尾波拖尾） |
| 打包布局（vLLM 风格） | `activation` v2→v3 | Memory Workload Analysis 的 global load 合并度 | 访问更连续。但这一档不要再拿 sector/request 当判据：v2 与 v3 都已是 16 B 向量化，两版实测同为 16.000 sector/request（32 线程 × 16 B = 512 B = 16 扇区，正是完美合并的下界，`activation/project-proof/profiling/ncu/act_v{2,3}_profile.ncu-rep` 的 raw 页），合并度上已无可回收；该看的是 DRAM 扇区数与等效带宽（`activation/project-proof/data/derived_activation_vec-after_stability.csv`：HBM 区间 918.967 → 928.333 GB/s） |

两条通用陷阱，本仓都踩过：

- **字节账要在 HBM 层记，不是指令层。** 指令级静态计数把"发一次 load"等同于
  "搬一次显存"，会高估可优化空间——那次"重读"被 L1 接住，从未出片（计数器实证见 §4 第 1 条）。
  详见 `docs/lectures/03_memory_bound_fusion.md`。
- **对照臂本身也要看。** `softmax_cublas_profile` 里是自写 kernel 不是 cuBLAS
  （BLAS 规范里没有 softmax）；`gemv_cublas_profile` 里才是真 cuBLAS 符号。
  口径见 `artifacts/ncu_for_mac/MANIFEST.md`。

## 3 NCU 看不见什么

NCU 是单 kernel 内部的显微镜，下面这些它一概答不了，别在报告里找：

| 问题 | 该用的工具 |
|---|---|
| kernel 之间的空隙、launch 开销占比 | nsys 时间线 |
| CUDA Graph 塌缩了多少次 launch | `nsys --cuda-graph-trace=node`（不加这个参数，graph 内 kernel 全部不可见） |
| 端到端 TTFT / TPOT 的归因 | 引擎侧计时，见 `llm-engine` |
| 多个 kernel 争抢 SM 的相互影响 | NCU 默认序列化 kernel，恰恰抹掉了这个 |

`rope` v1→v2 的"合并 launch"就是典型：它减少的是 launch 次数，
NCU 里两版的单 kernel 指标几乎一样，收益只在 nsys 时间线上看得见。

## 4 推断转实测的进度

原有五条挂着"推断"的命题，已闭合四条。计数器数据来自一台权限可用的 RTX 4090
（CUDA 12.8），见 `records/EXP-K07_ncu_counter_closure.md`；其中第 1 条在 16 B 向量化修复后
按同一协议复采过，读数以 `records/EXP-K09_post_vectorization_sector_ledger.md` 为准。

| # | 命题 | 状态 | 依据 |
|---|---|---|---|
| 1 | `fused-norm` v4 的"第二次读从未出片" | **实测闭合，且机制层级已更正** | DRAM 读扇区恒停在 2.000×S 的算法下界；L1 命中率 33.19%、L2 读命中率 0.20% —— **接住它的是 L1，不是 L2**（EXP-K09 §5.1） |
| 2 | "smem 是 GEMM/FA2 剩余差距的瓶颈" | **实测闭合** | FA2 v4 `short_scoreboard` 50.13% vs `long_scoreboard` 0.31%（EXP-K07 §5.4/§6.5）；gemm v4 的 shared-load wavefront 里 81.1% 是冲突 wavefront（放大 5.30×），真 cuBLAS 只有 0.28%（1.003×）——口径（分母只取 `op_ld`）与读法见 `docs/sass_evidence_ladder.md` §5.1，报告为 `gemm/project-proof/profiling/ncu/gemm_v4_bigtile_profile.ncu-rep` 与 `gemm_cublas_profile.ncu-rep` |
| 2b | "swizzle 能消除该瓶颈" | **仍是推断** | 本轮只证明瓶颈在 smem，未验证 swizzle 是解法。解锁 = 实现 v5 并同协议复测 |
| 3 | `w8a8`"反量化本可融进 GEMM epilogue" | **判为伪需求** | 融合版本不存在，NCU 也无从比较；代价（26.7%）本就是实测（EXP-K06 §5.1 三步分解） |
| 4 | `rope` v2 在 HBM 区间等效带宽比 v1 低 1.2% 的成因 | **已由 nsys 解决** | v1 对 q/k 各调一次 = 248 次 launch（`rope/project-proof/profiling/nsys/rope_kern_sum.csv`），v2 合并为 124 次；HBM 区间等效带宽的 −1.2%（784.7 → 775.0 GB/s，3 轮 std ±1.5）与 decode 区间的 1.39× 是同一改动的两面（`rope/project-proof/data/derived_rope_vec-after_stability.csv`，3 轮） |
| 5 | EXP-K01 因权限缺失未能复采的 stall / occupancy | **实测补齐** | 四算子全量（EXP-K07 §5.5/§6.6） |

表中两个命中率按扇区数加权（`sum(hit)/sum(total)`，12 次 launch 一并计）读出，与 EXP-K09 §5.1 同口径；若改用 NCU 原生的 `lts__t_sector_op_read_hit_rate.pct` 逐 launch 取平均，L2 读命中率是 0.22% 而不是 0.20%，两者都在说"到 L2 的扇区几乎全 miss 到显存"。L1 侧无此分歧，两种口径同为 33.19%。

**一条计数器答不了的**：`sm__inst_executed_pipe_tensor` 等分管线利用率不在任何 section 的
CLI 导出里，必须显式 `--metrics`（已加入 `scripts/ncu_metrics.inc.sh`）。
它回答的是"wmma 运行时占了多少"，而"wmma 有没有被编出来"由 SASS 的 `HMMA` 计数回答——
两者互补，见 `docs/sass_evidence_ladder.md`。

**报告口径**：`fused-norm` 的六份报告已按 16 B 向量化修复后的代码重采，同批还有 `rope`、
`activation`、`w8a8`，四个算子共 23 份，口径校验 FAIL 0。修复前后的三层扇区账对照见
EXP-K09《向量化修复后的扇区账复采》§5.1：L1TEX 全局读扇区 v3 由 16.000×S 降到 4.000×S、
v4 由 12.000×S 降到 3.000×S，降幅精确 4.00×（正是 16 B / 4 B）；而 DRAM 读一动不动，
仍是 2.000×S 的算法下界。这正是本文第 1 节第三步要求的读法——**声称的机制只该动它该动的那几个数**：
向量化改的是指令怎么发，不是算法要搬多少字节，所以指令层塌了四倍、HBM 层纹丝不动。
顺带也解释了 L1 命中率为什么"变差"：它等于 1 − L2 读扇区 / L1TEX 读扇区，而 L2 读几乎就是 DRAM 那 2.000×S
（L2 读命中率只有 0.20%，到 L2 的扇区几乎全 miss 到显存）；分母从 12.000×S 塌到 3.000×S，
命中率就从 83.19% 掉到 33.19%——搬运的字节数一个没多。命中率变低在这里是好消息，不是回归。

跨仓还有三条同源的（拿到权限后一并做，判据相同）：

- `triton-kernels` 的"双缓冲到底藏住延迟没有"——`long_scoreboard` stall 占比，
  见 `triton-kernels/docs/theory/02_double_buffering.md`。
- `triton-kernels` FA2 的波量化尾波假设与 `tl.exp` 的 SFU 代价，
  见 `triton-kernels/docs/lectures/01_fa2_from_softmax_to_flash.md`。
- `llm-engine` 混合臂 prefill +42% 归因于 wave quantization 而非流水线，
  见 `llm-engine/docs/lectures/02_kernels_meet_system.md`。

转实测后，`triton-kernels/docs/theory/04_perf_without_ncu.md`
（无计数器环境下的四大件平替）仍然有效——它是方法论，不因权限恢复而作废，
但其中"因为没有权限所以只能推断"的措辞需要按实际情况回写。
