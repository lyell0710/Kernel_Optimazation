# LEDGER — 状态与措辞账本(对内)

> **本文件是状态与措辞的唯一权威;README 为对外版,措辞以本表为准。**
> 数字的权威在各 `project-proof/data/` 与 `records/data/`(单一事实源,只链接不复制)。
> 对外文档 = README.md(门面,面试官视角)+ PORTFOLIO.md(深读),两者受本文件末节「对外禁词」约束。

## 实验台账(EXP 索引,状态唯一权威)

| 编号 | slug | 日期 | 状态 | 关键数字(指针) |
|---|---|---|---|---|
| [EXP-K01](records/EXP-K01_4090_rebench.md) | 4090_rebench | 2026-08-23 | 完成(带 8/24 勘误) | 4090 reduce v7 反超 cuBLAS 24.5%(3轮);softmax 对比句作废(对照系自写 kernel,勘误见记录 §5);gemv v3 快 cuBLAS **37.8%**(3 轮;单轮 84% 不可复现——cuBLAS 侧波动,勘误见 §7 闭环)→ 各 project-proof/data/ |
| [EXP-K02](records/EXP-K02_cuda_gemm_tc_ladder.md) | cuda_gemm_tc_ladder | 2026-08-24 | 完成 | Tensor Core GEMM v0→v4:133.1±0.97 TFLOPS = 真 cuBLAS 85.6%(4096³,3轮)→ gemm/project-proof/data/ |
| [EXP-K03](records/EXP-K03_cuda_fa2_ladder.md) | cuda_fa2_ladder | 2026-08-24 | 完成 | CUDA FA2 v0→v4:34.8±0.12 TFLOPS = 自家 Triton 28%(跨harness),wmma 架构税量化 → flash-attn/project-proof/data/ |

## 措辞红线表(对外文本逐条对照)

子项目级细表见 `gemm/README.md` 与 `flash-attn/README.md` 的「红线(措辞)」节。

| 红线 | 状态 | 依据 / 解锁条件 |
|---|---|---|
| GEMM「超过/追平 cuBLAS」 | 禁用(现状 85.6%) | Triton 版数字不得挪用;EXP-K02 §8 |
| FA2「达到 sdpa/Triton 水平」 | 禁用(现状 28%) | v5 mma PTX 路线;EXP-K03 §8 |
| 一切 vs Triton/sdpa 数字 | 跨 harness,推断级,引用必须带此限定 | 同 harness 复测;EXP-K03 §7 |
| softmax 的任何「vs cuBLAS」对比句 | 作废(对照物系自写 kernel,cuBLAS 无 softmax API) | 无解锁;EXP-K01 §5 勘误 |
| gemv 单轮领先幅度 | 作废,现行口径 = 快 37.8%(3 轮) | 对照物侧单轮波动已剖;EXP-K01 §7 闭环 |
| reduce「≈1193× 端到端」 | 仅限带「4070 Laptop」定语引用(授权例外) | 4090 端到端口径未测;见下节授权例外条 |
| 「swizzle / smem 往返是剩余差距主因」 | 推断,不可当实测说 | NCU 计数器(容器内不可用,EXP-K01 §7) |

## 勘误 / 审计留痕(横幅集中处;原文与史料见 records/、docs/archive/、LAB_JOURNAL)

- **softmax vs-cuBLAS 红线级勘误(2026-08-24)**:`softmax/src/softmax_cublas.cu` 系自写 warp 原语 kernel,并非 cuBLAS 调用(cuBLAS 无 softmax API)。"softmax 快 cuBLAS 26%/34%" 全链作废,含 L2 命中率 2.4 倍、online softmax 推断等归因叙事;简历/面试禁用。gemv/reduce 的 cuBLAS 对照经调用点验真有效。详见 EXP-K01 §5;作废段落原文 = `docs/archive/2026-08-24_portfolio_laptop_era_sections.md`。
- **gemv 单轮 84% 勘误(2026-08-24 晚闭环)**:v3 自身完全复现,但 cuBLAS 对照单轮 0.02353 → 3 轮 0.017432±0.000169(−26%)——领先幅度一半以上系对照侧单轮波动。现行口径 = 快 37.8%(3 轮)。教训:对照物也要 3 轮。EXP-K01 §7 闭环。
- **reduce Laptop 旧代自相矛盾(2026-08-24 审计)**:旧 results(v7=1.665ms)与旧 stability(0.273ms)矛盾,"v6/v7 回退 → 4090 反转"叙事不可确证、不作主张;旧稿移 `docs/archive/2026-08-24_portfolio_laptop_era_sections.md`,其中数字禁止对外引用。**唯一授权例外**:端到端口径「347.6ms→0.291ms,~1193×,4070 Laptop」经 Resume/Final_Resume/DO_NOT_SEND.md 2026-08-24 处置记录核准用于简历,引用必须带 Laptop 定语。
- **raw driver=13.3 误填勘误**:gemm/flash-attn raw 的 driver 字段误填 13.3(实为 610.57.04;13.3 是 cudaDriverGetVersion 报的 driver-API 版本)。raw 不可变:以两处 `data/manifest.txt` 勘误 + 修 bench 源码 provenance 生成,不重跑。
- **stability 覆盖史**:EXP-K01 首版仅 cuda-reduce 有 3 轮 stability;softmax/gemv/int8-quantize 于 2026-08-24 晚以 `scripts/stability_rebench.sh` 补齐(9 份 UTC raw + 3 份 3rounds 聚合入 records/data/),台账自此全绿。
- **对外化改版**:PORTFOLIO 顶部勘误横幅与正文内嵌勘误注记、README 台账/红线表,已全部收入本文件;README/PORTFOLIO 正文只保留降级后的对外措辞。

## 待办 / backlog(当前全部非阻塞)

- 本仓补测 backlog 已清零(EXP-K01 §7 于 2026-08-24 晚闭环)。
- 可选技能票:gemm/fa2 v5 = mma PTX + ldmatrix + smem swizzle(EXP-K02 §7、EXP-K03 §8;不阻塞)。
- 待非容器环境/NCU 计数器权限(ERR_NVGPUCTRPERM):「swizzle / smem 往返是剩余差距主因」类推断转实测。
- 待同 harness 复测:一切 vs Triton/sdpa 数字(现为跨 harness 推断级)。
- 开放问题:cuBLAS gemv 对照单轮慢 35% 未剖(冷启动/时钟态候选,EXP-K01 §7)。

## 内部约定(工作流,对齐 /root/standards CORE 七条铁律)

1. bench 只写 UTC 前缀新文件到各 `project-proof/data/`(`BENCH_OUT` 控制,首行 provenance),永不覆盖已有文件;profiler 环境的时延数字永不进 benchmark 表。
2. 数字进对外文档前 ≥3 轮 mean±std,落 stability/derived 文件;**对照物同样跑 3 轮**;「cublas」只准指真实库调用,自写参照叫 handwritten_*。
3. 每个实验一份 `records/` 八节记录(EXP-KNN),并同步本文件台账。
4. 对外措辞先过上方红线表;凡「vs X」声明先验 X 的调用点。
5. 收尾跑 `bash /root/standards/check.sh` 六项自检,0 FAIL 才 commit。
6. `figures/` 全部由 `scripts/plot_readme_figures.py` 生成,禁手改;对外图脚注不带日期(日期溯源走 raw 文件名与 manifest)。
7. **对外禁词**(README/PORTFOLIO 逐个 grep 自查,一个不留):任何日期与时间戳(2026-xx、08-2x、UTC 时间戳)、勘误、审计、红线、解锁、台账、停投、待用户、待 GPU 空闲、待建远端、backlog、sibling、Codex、HANDOFF、check.sh、CORE、STANDARDS、DO_NOT_SEND、铁律、终端级证据(对外改写为「单轮」)。records/ 与 docs/archive/ 是史料,不受此限。
