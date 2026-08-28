# EXP-K02 · CUDA Tensor Core GEMM 版本梯(v0→v4,vs 真 cuBLAS)

> **一句话结论**：GEMM 的性能台阶来自**指令世代**而非访存微调：smem tiling 只给 +25%，换用 wmma 一步 ×13.8；v4 做到 133.1 TFLOPS，为真 cuBLAS 的 85.6%。

## 0. 元信息

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-24 |
| 环境 | RTX 4090 · CUDA 13.2 · driver 610.57.04（见 CSV 首行 provenance） |
| 状态 | 完成 |
| 关联清单项 |「CUDA 手写对应算子」补位 ①：Tensor Core GEMM（算子计划中此项此前经 Triton 完成，CUDA 路线补齐） |

## 1. 目的与假设

背景：本仓四个 CUDA kernel 全是行核/向量核，唯一的 Tensor Core GEMM 在 triton-kernels 仓（triton-kernels#EXP-T02《流水线 GEMM》，160.5 TFLOPS）。面试「手写 CUDA 用过 Tensor Core 吗」此前只能指本机没有代码的 Llama2 引擎——证据可及性风险。本实验用 CUDA 原生 API 把版本梯补出来。

可证伪假设（跑前锁定）：
- H1：fp16 输入下，CUDA-core 路线（v0/v1）与 Tensor Core 路线（v2+）有数量级差距（>5×），且 v1 的 smem tiling 在 CUDA-core 路线上收益有限（瓶颈在算力不在访存）。
- H2：cp.async 双缓冲（v3）与大 tile(v4)各自带来可测收益，合计使 wmma 路线达到真 cuBLAS 的 ≥75%。
- 判定阈值：正确性 = 对 cuBLAS 输出抽样 max_rel_err < 2e-2（fp16 存储合理界）；性能数字取 3 轮 mean±std。

## 2. 环境与配置

- 尺寸固定 M=N=K=4096，fp16 存储 / fp32 累加，row-major。
- 每版本 3 warmup + 50 iters（v0/v1 慢版 5 iters），CUDA event 计时。
- 对照 = `cublasGemmEx`(CUBLAS_COMPUTE_32F， DEFAULT_TENSOR_OP)， **调用点验真**：gemm/src/gemm_cublas.cu 含 `cublasCreate`+`cublasGemmEx`， 非自写 kernel（EXP-K01《四 kernel 4090 重基准》§5 softmax 勘误的整改动作：凡 "vs cublas" 先验调用点）。
- 版本梯（gemm/src/）:
  - v0 naive：1 thread/输出，直接全局访存。
  - v1 tile:32×32 smem 分块，CUDA core FMA。
  - v2 wmma:BM=BN=64/BK=32,4 warp × (2×2) 16³ fragment,float4 协同加载。
  - v3 dbuf：v2 + `__pipeline_memcpy_async` 双缓冲（smem[2][...]，加载与 mma 重叠）。
  - v4 bigtile:BM=BN=128/BK=32,8 warp × (4×2) fragment + 双缓冲。

## 3. 步骤

```bash
cd /root/projects/Kernel_Optimazation/gemm
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
for r in 1 2 3; do GIT_SHA=<sha> BENCH_OUT=project-proof/data/$(date -u +%Y%m%dT%H%M)_gemm4096_r$r.csv \
  BENCH_ITERS=50 ./build/gemm_bench; done
# 聚合 → derived_gemm4096_stability.csv;资源画像:
nvcc -O3 -arch=sm_89 -Xptxas -v -c src/gemm_v{2,3,4}.cu -Iinclude -o <tmp>
```

## 4. 原始数据

- 3 轮 raw:`gemm/project-proof/data/20260824T1517_gemm4096_r{1,2,3}.csv`（首行 provenance）
- 聚合：`gemm/project-proof/data/derived_gemm4096_stability.csv`
- 资源画像：`gemm/project-proof/data/ptxas_resource_usage.txt`

## 5. 结果(4096³,3 轮 mean±std)

| 版本 | latency (ms) | TFLOPS | vs cuBLAS | 逐级归因 |
|---|---|---|---|---|
| v0 naive | 26.369±0.472 | 5.2±0.12 | 3.4% | — |
| v1 tile | 21.114±0.047 | 6.5±0.00 | 4.2% | smem tiling 仅 +25% |
| v2 wmma | 1.536±0.008 | 89.5±0.46 | 57.6% | **Tensor Core ×13.8** |
| v3 dbuf | 1.439±0.007 | 95.5±0.49 | 61.4% | 双缓冲 +6.7% |
| v4 bigtile | 1.033±0.007 | **133.1±0.97** | **85.6%** | 大 tile +39% |
| cublas（真库） | 0.884±0.004 | 155.4±0.62 | 100% |— |

正确性：全版本 max_rel_err=7.58e-04 < 2e-2，PASS（v0-v4 同值：误差被输入/输出的 fp16 舍入主导，与累加顺序无关）。

资源画像（ptxas -v，sm_89）：

| 版本 | regs | smem | block | 每 SM 驻留 block（限制因子） | 理论 occupancy |
|---|---|---|---|---|---|
| v2 | 54 | 8KB | 128thr | 9(regs) | 75% |
| v3 | 61 | 16KB | 128thr | 6(smem) | 50% |
| v4 | 92 | 32KB | 256thr | 2(regs,92×256=23.5K/64K) | **33%** |

## 6. 分析与结论

- **H1 成立**（实测）：v0→v1 仅 1.25×——fp16 在 CUDA core 上做 fp32 FMA， ~6.5 TFLOPS 已近该路线实际上限，tiling 省下的带宽换不来算力；真正的台阶是指令世代（wmma ×13.8）。这与 gemv/reduce（memory-bound，tiling 才是主菜）形成对照——**先判定 bound 类型再选优化**（PORTFOLIO 原则 3）。
- **H2 成立**：双缓冲 +6.7%（把 smem 加载藏进 mma），大 tile +39%（128² tile 使每字节 smem 复用次数翻倍 + 4×2 fragment 的寄存器级 ILP）， 合计 85.6% cuBLAS，超 75% 阈值。
- **occupancy 反相关再现**：v4 理论 occupancy 33%（全梯最低）却最快——与 EXP-K01 GEMM「occ 17% 打 98% peak」同一结论：Tensor Core 时代吞吐靠 fragment 级 ILP 与数据复用，不靠线程数遮蔽（kperf/占用率分析的标准答案素材）。
- **与自家 Triton 对照**（推断级，跨 harness）：triton-kernels#EXP-T02 同尺寸 Triton 160.5 TFLOPS（其 harness 下 cuBLAS=159.8；本 harness cuBLAS=155.4，harness 差 ~3%）。**Triton 版仍快于本 CUDA v4 约 17%**： Triton 编译器生成 mma.sync+ldmatrix+swizzle 布局，而 wmma API 不暴露 smem swizzle，fragment 加载有 bank conflict 税。
- 剩余 14% 差距去向（推断，标注为未剖析验证）：wmma 固定布局的 smem bank conflict、无 >2 级流水、tile 形状单一（cuBLAS 会选 128×256 等）。 NCU 不可用（EXP-K01 §7 容器限制），未做计数器级确认。

## 7. 异常、偏差与开放问题

- 首次单轮跑（build 后验证）写入了固定名 benchmark_results.csv，与 CORE bench 铁则（UTC 前缀新文件）不符——当场整改：main.cu 加 BENCH_OUT + provenance 首行，固定名文件已删，正式 3 轮全部 UTC 前缀。该文件生前数字与 3 轮一致（v4 132.8/cublas 155.4，终端级证据）。
- v1 的 std≈0(21.11±0.05ms)：慢版仅 5 iters，但绝对波动小，不影响结论。
- 开放问题（backlog，不阻塞）：v5 = mma PTX + ldmatrix + smem swizzle， 验证「wmma→mma 差距」假设；非方阵/LLM 实际形状（如 qwen8b 系列）对照 triton-kernels#EXP-T02 同表。

## 8. 下游影响

- 简历/面试可说（措辞红线）：「手写 CUDA Tensor Core GEMM（wmma+cp.async 双缓冲+大 tile），4096³ 达真 cuBLAS 85.6%（133 TFLOPS，4090,3 轮）」。 **不可说**：超过/追平 cuBLAS（那是 Triton 版的数字，且系跨 harness）。 -「CUDA 手写对应哪些算子」缺口收敛：Tensor Core GEMM 由「仅 Triton/ 本机无 CUDA 证据」→「本仓有完整版本梯+raw」。
- PORTFOLIO 增补项目 5 指针；README 索引 + 红线表更新。
