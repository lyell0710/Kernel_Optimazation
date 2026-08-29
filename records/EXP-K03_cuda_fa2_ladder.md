# EXP-K03 · CUDA FA2 forward 简化版版本梯(v0→v4,量化 wmma 架构税)

> **一句话结论**：同一套 wmma 工具箱，GEMM 够到 cuBLAS 的 86%，FA2 却只到 34.8 TFLOPS——自家 Triton 版的 28%。越靠融合吃饭的算子，越需要 mma 级的寄存器控制，这就是 wmma 的架构税。

## 0. 元信息

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-24 |
| 环境 | RTX 4090 · CUDA 13.2 · driver 610.57.04（CSV 首行 provenance） |
| 状态 | 完成 |
| 关联清单项 |「 CUDA 手写对应算子」补位 ②：FA2 前向 CUDA 版（Triton 版=triton-kernels#EXP-T01《Triton FA2 forward》） |

## 1. 目的与假设

用 CUDA 原生 API 重走 FA2 forward，与自家 Triton 版同协议对照。核心问题不是「能不能追上」，而是**定量回答：不用 mma PTX（只用 wmma），FA2 能到哪里、卡在哪里**——wmma accumulator fragment 的 lane→元素映射未定义， 行级 softmax(max/exp/α) 必须经 smem 往返，这是 API 层级强加的结构。

可证伪假设（跑前锁定）：
- H1：在线 softmax 三件套（m/l/α） 的 CUDA 实现全 shape 过 2e-2 gate（含 GQA 4:1、 causal/非 causal）。
- H2：Tensor Core(v2) 对 CUDA-core(v0/v1)≥4×；并行度与访存重叠（v3/v4）各有可测收益；但 wmma 路线**到不了** Triton 版（mma+寄存器驻留） 的 60%——smem 往返与相位同步是结构性的。
- 协议 = triton-kernels/scripts/test_fa2.py：B=1，Hq=32，Hkv=8，D=128， causal，S∈{512,1024,2048,4096}；正确性 vs fp32 两遍法参考（算法路径独立）。

## 2. 环境与配置

版本梯（flash-attn/src/，D=128 固定； v2+ 要求 S%64==0，通用尾块见 v0/v1）：
- v0 warp-per-row：1 warp 管 1 行 q，顺序流过 K/V，在线 m/l/α 最直白形态， K/V 复用全交给 L2。
- v1 +smem tile：4 warp/块， K/V 64 键一批进 smem。对 v0 唯一变量=读取层级。
- v2 wmma:QK^T 与 P·V 走 Tensor Core(64×64 tile,4 warp),S→smem→ 标量段 softmax→P→smem→PV,O 驻 smem 逐 tile 按 α 重缩放。动态 smem 90.75KB。
- v3 8 warp：并行组织翻倍（行条带×列半区），softmax 每行 2 线程，算法不变。
- v4 重叠： S/P 统一 half 缓冲（exp 原位改写，省 17KB）；K 双缓冲 cp.async（QK^T 时下一 tile 在途）；V 载入与 QK^T 重叠。 smem 89.75KB。

## 3. 步骤

```bash
cd /root/projects/Kernel_Optimazation/flash-attn
cmake -S . -B build && cmake --build build -j
for r in 1 2 3; do GIT_SHA=<sha> BENCH_OUT=project-proof/data/$(date -u +%Y%m%dT%H%M)_fa2_proto_r$r.csv ./build/fa2_bench; done
```

## 4. 原始数据

- 3 轮 raw:`flash-attn/project-proof/data/20260824T15*_fa2_proto_r{1,2,3}.csv`
- 聚合：`derived_fa2_proto_stability.csv`；资源：`ptxas_resource_usage.txt`

## 5. 结果

正确性 gate（每轮全过）:v0-v4 全 shape PASS,err 4.88e-04（fp32 版）/ 4.88e-04（v4 的 fp16 S/P 未推高误差）≪ 2e-2—— **H1 成立**，含（1,8,8,512,±causal）、 GQA(1,16,8,1024)、（1,32,8,2048）。

S=4096 协议点（3 轮 mean±std）：

| 版本 | latency (ms) | TFLOPS | 逐级归因 |
|---|---|---|---|
| v0 warp-row | 27.795±0.047 | 4.9±0.06 | — |
| v1 smem tile | 25.113±0.071 | 5.5±0.00 | smem 仅 +11%（L2 已扛住广播读）|
| v2 wmma | 5.635±0.017 | 24.4±0.06 | **Tensor Core ×4.5** |
| v3 8warp | 4.229±0.012 | 32.5±0.10 | 并行组织 +33% |
| v4 overlap | **3.949±0.012** | **34.8±0.12** | 预取+half S/P 仅 +7.1% |

跨尺寸（v4）：S=512/1K/2K/4K = 19.9/26.7/31.8/34.9 TFLOPS（r1 单轮值， 小 S 被 wave quantization 与固定开销压低）。

资源（ptxas,sm_89）:v2 72reg/128thr,v3 95reg/256thr,v4 80reg/256thr； 三者 smem ≈90KB → 每 SM 1 block(occupancy 8.3%~16.7%)。

## 6. 分析与结论

- **H2 前半成立**：v1→v2 ×4.5；v3 +33% 说明 v2 的 4 warp/128 线程吃不满 1 block/SM 的机器；**v4 仅 +7.1% 是本实验最有信息量的数字**——K/V 预取能给的都给了，剩下的时间不在全局访存，而在每 tile 5 次 __syncthreads 串起来的「QK^T→S 落 smem→标量 softmax→O 重缩放→PV→O 回写」相位链。
- **H2 后半成立（量化了架构税）**：v4 34.8 TF = Triton 版（triton-kernels#EXP-T01， S=4096 1.119ms ≈ 123 TF，**跨 harness，推断级**） 的 **28%**，= sdpa-flash (≈140 TF) 的 25%。差距解释（推断， NCU 不可用未剖析确认）：Triton 编译到 mma PTX，fragment 布局已知 → softmax/α 直接在寄存器上做、 O 常驻寄存器， 全程无 smem 往返、无相位 sync；wmma 隐藏布局逼出的 smem 中转把 FA2 的「融合免搬运」优势吃掉大半。**结论一句话： GEMM 用 wmma 够到 cuBLAS 86%（EXP-K02《CUDA Tensor Core GEMM 版本梯》），FA2 用 wmma 只够到 28%——越是靠融合吃饭的算子，越需要 mma 级寄存器控制，这就是 FA2 官方实现用 CUTLASS/mma 不用 wmma 的原因。**
- v0→v1 仅 +11%：K/V 广播读 L2 早已扛住（单 kv-head K = S·D·2B = 1MB ≪ 72MB L2），与 gemm v0→v1(compute-bound，+25%) 机理不同但同样「白忙」—— 优化前先想清楚当前瓶颈层。

## 7. 异常、偏差与开放问题

- v2+ 只支持 D=128、 S%64==0（bench 全形状满足）；通用性由 v0/v1 兜底。非整除 S 的 v2 化在 Triton 版已解（mask），CUDA 版列 backlog 不重复做。
- 与 Triton/sdpa 的对照为跨 harness（triton-kernels#EXP-T01 wall-clock vs 本仓 CUDA event， 均为 100 iters 稳态， ms 级 kernel 差异 <1%），记录为推断级。
- 开放： v5 = mma PTX + ldmatrix（拿到确定布局， softmax 上寄存器）——与 EXP-K02 的 gemm v5 是同一张技能票，二者共用一次学习成本。

- **补记（2026-08-24 审计收尾）**：首次单轮验证跑落下的固定名 `project-proof/data/benchmark_results.csv`（sha=unknown，违 CORE bench 铁则） 已 git rm——删除前与 3 轮 UTC 数据逐行核对一致（v4_overlap S=4096: 3.9427ms/34.9TF，3 轮为 3.9404/3.9434/3.9617ms），按「终端级证据，已删」处理；并核验 src/main.cu：设 BENCH_OUT 时仅写 UTC 前缀文件，不落固定名。

## 8. 下游影响

- 面试可说：「 CUDA 手写 FA2 前向（在线 softmax + wmma + cp.async 重叠）， 全 shape 过 2e-2 gate；并量化了 wmma vs mma 的架构税（34.8 vs 123 TFLOPS， 同卡同协议对照自家 Triton 版）」。**不可说**：「 CUDA FA2 达到 sdpa 水平」。 -「 CUDA 手写对应哪些算子」映射表更新： FA2 由「仅 Triton」→「 CUDA 版本梯（性能上限有明确归因）+ Triton 版（87% sdpa）」双证据。
- 教学价值： v2 的 smem 往返设计是讲「为什么 FA 需要 mma」的最好教具。
