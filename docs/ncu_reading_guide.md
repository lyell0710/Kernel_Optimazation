# NCU 报告阅读指南：把「这一版为什么快」读成证据

本文不重复各版本改了什么——那是各算子 README 的版本梯表与 `docs/lectures/` 的事。
本文只做一件事：**把优化手法映射到 Nsight Compute 里该看的那一个数**，
让「我猜它是因为 X 才快的」变成「报告第 N 行显示 X」。

采集入口 `scripts/run_ncu_all.sh`，导出与口径校验 `scripts/export_ncu_for_mac.py`，
Mac 端打开方式见 `artifacts/ncu_for_mac/MANIFEST.md`。

本文引用的每个 section 都必须在 `scripts/ncu_metrics.inc.sh` 的采集列表里，否则报告里根本没有那一节。改本文时同步查那个文件。
注意 2026-05 那批 Laptop 报告采于 `ComputeWorkloadAnalysis` 加入之前，`cuda-reduce` / `gemv` / `int8-quantize` / `softmax` 的现存报告没有这一节；它们都是访存型算子，用不到，但要重采才能补上。

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
| 向量化访存（16 B / `float4`） | `fused-norm` v3、`rope` v3、`activation` v2、`softmax` v3 | `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld` | 每 request 的 sector 数下降＝同样字节用更少 request 搬完 |
| 计算换访存（免查表，`__sincosf` 现算） | `rope` v3→v4 | SOL 的 SM Throughput 与 DRAM Throughput 一升一降 | 表不再读，SFU 压力换来 DRAM 压力下降 |
| 寄存器缓存消掉第二次读 | `fused-norm` v3→v4 | `lts__t_sectors_srcunit_tex_op_read.sum` 与 `dram__sectors_read.sum` 的比值(只在 `--page raw` 里) | 见 §4 第 1 条——这条目前是推断，不是实测 |
| Tensor Core（wmma） | `gemm` v1→v2、`flash-attn` v1→v2 | Compute Workload Analysis 的 Tensor pipe 利用率；SOL 的 SM Throughput | Tensor pipe 从零起来，FMA pipe 让位 |
| cp.async 双缓冲 | `gemm` v2→v3、`flash-attn` v3→v4 | Warp State Statistics 的 `long_scoreboard` stall 占比 | 延迟被藏住＝等访存的 stall 占比下降。这是"双缓冲有没有生效"的直接证据 |
| 大 tile / 增加 warp 数 | `gemm` v3→v4、`flash-attn` v2→v3 | Occupancy（Theoretical vs Achieved）；Launch Statistics 的 Waves Per SM、Registers Per Thread、Static Shared Memory | 关注两处：寄存器或 smem 是否把理论占用率压下去；Waves Per SM 是否为接近整数（非整数＝波量化，尾波拖尾） |
| 打包布局（vLLM 风格） | `activation` v2→v3 | Memory Workload Analysis 的 global load 合并度 | 访问更连续，sector/request 进一步下降 |

两条通用陷阱，本仓都踩过：

- **字节账要在 HBM 层记，不是指令层。** 指令级静态计数把"发一次 load"等同于
  "搬一次显存"，会高估可优化空间——那次"重读"可能被 L2 接住，从未出片。
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

## 4 权限恢复后优先转实测的既有推断

下列结论目前都标着"推断"，因为采不到计数器。它们各自有一个明确的判据，
拿到报告后应当逐条转成实测，并回写对应文件。

| # | 待验证的推断 | 判据 | 出处 |
|---|---|---|---|
| 1 | `fused-norm` v4 的"第二次读被缓存接住、从未出片"**（已实测闭合：接住它的是 L1）** | `lts__t_sectors_srcunit_tex_op_read.sum` 与 `dram__sectors_read.sum` 的比值(只在 `--page raw` 里) | `docs/lectures/03_memory_bound_fusion.md` §3.3 与「边界四」 |
| 2 | "swizzle / smem 往返是 FA2 剩余差距的主因" | Memory Workload Analysis 的 shared bank conflict 计数 | `LEDGER.md`、`docs/lectures/02_wmma_tax_fa2.md` |
| 3 | `w8a8`"反量化本可融进 GEMM epilogue" | 反量化 kernel 的 DRAM 读写字节 vs GEMM epilogue 可省的量 | `records/EXP-K06_w8a8_linear.md` |
| 4 | `rope` v2 在 HBM 区间比 v1 慢 1.1% 的成因 | 指令数与 stall 分解（分支与下标运算的代价） | `records/EXP-K05_llm_fused_elementwise.md` |
| 5 | EXP-K01 因权限缺失未能复采的 stall / occupancy 细节 | 直接重采即可 | `records/EXP-K01_4090_rebench.md` |

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
