# Kernel_Optimazation

**手写 CUDA kernel 版本梯证据仓**:6 个算子(reduce / softmax / gemv / int8-quantize / Tensor Core GEMM / FA2 forward)从 naive 到打平/反超通用库,每一步都可测量、可归因、可复现——数字带 raw 指针、≥3 轮 mean±std、对照物调用点验真、勘误公开留痕。

> 叙事入口:[PORTFOLIO.md](PORTFOLIO.md)(方法论四原则 + 六项目 + 跨项目 pattern)。本 README 是**状态与索引的唯一权威**;数字的权威在各 `project-proof/data/` 与 `records/data/`(CORE 铁律 1,只链接不复制)。

## 🎯 Headline(RTX 4090,均为 3 轮 mean±std)

| 结果 | 证据指针 |
|---|---|
| **Tensor Core GEMM 133.1±0.97 TFLOPS = 真 cuBLAS 的 85.6%**(4096³,fp16 输入/fp32 累加,wmma+cp.async 双缓冲+128² 大 tile) | [EXP-K02](records/EXP-K02_cuda_gemm_tc_ladder.md) · `gemm/project-proof/data/derived_gemm4096_stability.csv` |
| **CUDA FA2 forward 34.8±0.12 TFLOPS = 自家 Triton 版的 28%(跨 harness,推断级)**——wmma 架构税的定量测量;全 shape 过 2e-2 正确性 gate | [EXP-K03](records/EXP-K03_cuda_fa2_ladder.md) · `flash-attn/project-proof/data/derived_fa2_proto_stability.csv` |
| **reduce v7 反超真 cuBLAS 24.5%**(0.02988±0.00011 vs 0.03721±0.00022 ms,1600 万 float);端到端 347.6ms→0.291ms ≈ **1193×**(4070 Laptop 口径,授权例外见 PORTFOLIO 勘误段) | [EXP-K01](records/EXP-K01_4090_rebench.md) · `records/data/exp_k01_reduce_3rounds.csv` |
| **gemv v3 比真 cuBLAS gemv 快 37.8%**(4096×2048;单轮虚高值经「对照物也跑 3 轮」复测后勘误作废) | [EXP-K01 §7 闭环](records/EXP-K01_4090_rebench.md) · `records/data/exp_k01_gemv_3rounds.csv` |
| **理论 occupancy 33% 全梯最低,却是最快版本**(gemm v4:92 reg×256 thr+32KB smem)——Tensor Core 吞吐靠 fragment 级 ILP 与 smem 复用,不靠线程数遮蔽 | [EXP-K02 §5](records/EXP-K02_cuda_gemm_tc_ladder.md) · `gemm/project-proof/data/ptxas_resource_usage.txt` |

## 📊 图表(scripts 从 derived/records 数据生成,禁手改)

![GEMM Tensor Core 版本梯](figures/01_gemm_tc_ladder.png)

*GEMM 的性能台阶是指令世代(v1→v2 wmma ×13.8),访存微调只是坡(v0→v1 仅 +25%);v4 达真 cuBLAS 85.6%。source: `gemm/project-proof/data/derived_gemm4096_stability.csv` · 2026-08-24 · [EXP-K02](records/EXP-K02_cuda_gemm_tc_ladder.md)*

![FA2 wmma 版本梯](figures/02_fa2_wmma_ladder.png)

*同一套 wmma 工具箱,FA2 只到 34.8 TFLOPS:v4 把 K/V 访存全部预取重叠后仅 +6.6%,坐实瓶颈在 smem 往返相位链而非访存。source: `flash-attn/project-proof/data/derived_fa2_proto_stability.csv`(S=4096 行)· 2026-08-24 · [EXP-K03](records/EXP-K03_cuda_fa2_ladder.md)*

![reduce v7 vs 真 cuBLAS](figures/03_reduce_v7_vs_cublas.png)

*单一 shape 特化的 grid-stride two-pass(v7)反超真 cuBLAS 24.5%(3 轮,对照物调用点验真)。source: `records/data/exp_k01_reduce_3rounds.csv` · 2026-08-24 · [EXP-K01](records/EXP-K01_4090_rebench.md)*

图表重生成(matplotlib,本机 venv `/root/venvs/kernel-opt/bin/python`):

```bash
python scripts/plot_readme_figures.py
```

## 🔬 代码导览

**① Tensor Core 主循环:wmma 4×2 fragment + cp.async 双缓冲**(摘自 [gemm/src/gemm_v4.cu](gemm/src/gemm_v4.cu) L40-L64;8 warp,每 warp 算 64×32 输出)

```cuda
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K)
            load_tile_async4(As[p ^ 1], Bs[p ^ 1], A, B, N, K,
                             bm, bn, k0 + BK, threadIdx.x, blockDim.x);   // 预取下一 tile
        __pipeline_wait_prior(k0 + BK < K ? 1 : 0);
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                wmma::load_matrix_sync(af[i], &As[p][wr * 64 + i * 16][kk], BK);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[j], &Bs[p][kk][wc * 32 + j * 16], BN);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
        p ^= 1;
    }
```

机制:smem 双缓冲 `As/Bs[2][…]`,mma 吃 buffer `p` 的同时 `cp.async` 往 `p^1` 灌下一 tile(加载完全藏进计算);4×2=8 个 accumulator fragment 常驻寄存器,一次 load 的 fragment 参与多次 `mma_sync`——正是这份 fragment 级 ILP + smem 复用,让 33% occupancy 的 v4 拿到全梯最快(EXP-K02 §6)。

**② FA2 的 wmma 架构税:softmax 被迫走 smem 往返**(摘自 [flash-attn/src/fa2_v2.cu](flash-attn/src/fa2_v2.cu) L72-L92)

```cuda
            wmma::store_matrix_sync(&Ssm[(warp * 16) * LDS + n * 16], sc,
                                    LDS, wmma::mem_row_major);   // S 从 fragment 倒回 smem
        }
        __syncthreads();
        if (tid < BM) {                                // 行级在线 softmax
            const int row = q0 + tid;
            const int jend = min(BN, (causal ? row + 1 : S) - n0);
            float rmax = -1e30f;
            for (int j = 0; j < jend; ++j)
                rmax = fmaxf(rmax, Ssm[tid * LDS + j] * scale);
            const float mn = fmaxf(m_s[tid], rmax);
            const float alpha = __expf(m_s[tid] - mn);
            float sum = 0.f;
            for (int j = 0; j < BN; ++j) {
                float p = j < jend ? __expf(Ssm[tid * LDS + j] * scale - mn) : 0.f;
                Psm[tid * LDP + j] = __float2half(p);
                sum += p;
            }
            l_s[tid] = l_s[tid] * alpha + sum;
            m_s[tid] = mn; a_s[tid] = alpha;
        }
```

机制:wmma accumulator fragment 的 lane→元素映射是编译器私有的,行级 max/exp/α **无法在 fragment 上做**——QK^T 结果必须 `store_matrix_sync` 落 smem,再由标量段逐行重读。这一往返加每 tile 5 次 `__syncthreads` 的相位链,把 FA2「融合免搬运」的优势吃掉大半:GEMM 用 wmma 够到 cuBLAS 86%,FA2 只够到自家 Triton 版(mma+寄存器驻留)的 28%(跨 harness)。这就是官方 FA2 用 CUTLASS/mma 而非 wmma 的定量理由(EXP-K03 §6)。

## 复现 Quickstart

```bash
# 单项目(以 gemm 为例;flash-attn 同构,bench 二进制为 fa2_bench)
cd gemm
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
BENCH_OUT=project-proof/data/$(date -u +%Y%m%dT%H%M)_gemm4096_r1.csv ./build/gemm_bench

# softmax / gemv / int8-quantize 一键 bench + plot
bash scripts/run_bench_and_plot_all.sh

# 3 轮 stability 复测(UTC 前缀新文件 + 首行 provenance)
bash scripts/stability_rebench.sh

# NCU 全项目采集(默认导出扩展 metrics CSV;仅要 .ncu-rep 时 RUN_NCU_CSV=0)
bash scripts/run_ncu_all.sh
# 打包 *_profile.ncu-rep 到 artifacts/ncu_for_mac/ 便于 scp:
bash scripts/pack_ncu_reps_for_mac.sh
```

bench 约定:只写 UTC 前缀新文件(`BENCH_OUT` 控制,首行 provenance),永不覆盖已有文件;profiler 环境的时延数字永不进 benchmark 表。

## 仓库结构

```
Kernel_Optimazation/
├── PORTFOLIO.md        # 叙事入口:方法论 + 六项目 + 跨项目 pattern
├── records/            # EXP 八节记录;records/data/ = 3 轮聚合 CSV
├── figures/            # 本 README 门面图(scripts/plot_readme_figures.py 生成)
├── scripts/            # 一键 bench / plot / NCU / stability 工装
├── gemm/               # Tensor Core GEMM 版本梯 v0→v4(vs 真 cuBLAS)
├── flash-attn/         # CUDA FA2 forward 版本梯 v0→v4(wmma 架构税)
├── cuda-reduce/        # reduce 版本梯 baseline→v7(4090 反超真 cuBLAS)
├── gemv/ softmax/ int8-quantize/   # 行核/向量核三项目
├── artifacts/          # Laptop 时代 ncu-rep 打包(机理参照)
└── docs/               # theory / talk / archive(勘误原文归档处)
```

各子项目自带 `project-proof/data/`(数字权威)与 README(gemm/flash-attn 含子项目级红线表)。

## 实验台账(EXP 索引,本仓状态唯一权威)

| 编号 | slug | 日期 | 状态 | 关键数字(指针) |
|---|---|---|---|---|
| [EXP-K01](records/EXP-K01_4090_rebench.md) | 4090_rebench | 2026-08-23 | 完成(带 8/24 勘误) | 4090 reduce v7 反超 cuBLAS 24.5%(3轮);softmax 对比句作废(对照系自写 kernel,勘误见记录 §5);gemv v3 快 cuBLAS **37.8%**(3 轮;单轮 84% 不可复现——cuBLAS 侧波动,勘误见 §7 闭环)→ 各 project-proof/data/ |
| [EXP-K02](records/EXP-K02_cuda_gemm_tc_ladder.md) | cuda_gemm_tc_ladder | 2026-08-24 | 完成 | Tensor Core GEMM v0→v4:133.1±0.97 TFLOPS = 真 cuBLAS 85.6%(4096³,3轮)→ gemm/project-proof/data/ |
| [EXP-K03](records/EXP-K03_cuda_fa2_ladder.md) | cuda_fa2_ladder | 2026-08-24 | 完成 | CUDA FA2 v0→v4:34.8±0.12 TFLOPS = 自家 Triton 28%(跨harness),wmma 架构税量化 → flash-attn/project-proof/data/ |

## 措辞红线表与方法论

**措辞红线**(对外文本逐条对照;子项目级细表见 `gemm/README.md` 与 `flash-attn/README.md`):

| 红线 | 状态 | 依据 / 解锁条件 |
|---|---|---|
| GEMM「超过/追平 cuBLAS」 | 禁用(现状 85.6%) | Triton 版数字不得挪用;EXP-K02 §8 |
| FA2「达到 sdpa/Triton 水平」 | 禁用(现状 28%) | v5 mma PTX 路线;EXP-K03 §8 |
| 一切 vs Triton/sdpa 数字 | 跨 harness,推断级,引用必须带此限定 | 同 harness 复测;EXP-K03 §7 |
| softmax 的任何「vs cuBLAS」对比句 | 作废(对照物系自写 kernel,cuBLAS 无 softmax API) | 无解锁;EXP-K01 §5 勘误 |
| gemv 单轮领先幅度 | 作废,现行口径 = 快 37.8%(3 轮) | 对照物侧单轮波动已剖;EXP-K01 §7 闭环 |
| reduce「≈1193× 端到端」 | 仅限带「4070 Laptop」定语引用(授权例外) | 4090 端到端口径未测;PORTFOLIO 勘误段 |
| 「swizzle / smem 往返是剩余差距主因」 | 推断,不可当实测说 | NCU 计数器(容器内不可用,EXP-K01 §7) |

**诚实度文化**(本仓的差异化卖点,如实展示):进本 README 的每个数字都有 raw 文件、首行 provenance(env/sha/cmd/date/gpu/driver)、≥3 轮 mean±std,且**对照物同样跑 3 轮并做调用点验真**——「cublas」只准指真实库调用,自写参照叫 handwritten_*。归因靠反例臂:softmax 的 v4.2/v4.3/v4.4 三个故意构造的退化版本把加速拆到具体机制(PORTFOLIO 原则 2)。错误不删不掩:单轮虚高、对照物误标等勘误原文归档 docs/archive/,现行文档只保留降级后的措辞——上面这张红线表就是勘误的执行界面。

**工作流**(对齐 /root/standards CORE 七条铁律):
1. bench 只写 UTC 前缀新文件到各 `project-proof/data/`(`BENCH_OUT` 控制,首行 provenance),永不覆盖已有文件。
2. 数字进 README/简历前 ≥3 轮 mean±std,落 stability/derived 文件。
3. 每个实验一份 `records/` 八节记录(EXP-KNN),并同步上方 EXP 索引表。
4. 对外措辞先过红线表,凡「vs X」声明先验 X 的调用点。
5. 收尾跑 `bash /root/standards/check.sh` 六项自检。

## 相关仓

- [vllmExperience](https://github.com/lyell0710/vllmExperience) — vLLM 推理服务实验证据仓(2×4090)
- [llm-engine](https://github.com/lyell0710/llm-engine) — 手写 Llama2 推理引擎
- triton-kernels(本机仓,无远端)— Triton 版 GEMM/FA2(EXP-T01/T02),本仓「跨 harness 对照」的口径出处
