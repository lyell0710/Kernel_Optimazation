# LAB_JOURNAL — Kernel_Optimazation

## §1 4090 重基准 + roofline 迁移(2026-08-23,EXP-K01《4090 重基准》)

- **做了什么**：四 kernel(reduce/softmax/gemv/int8-quantize)在 4090 全量重跑既有 bench（未改源码），与 git HEAD 里的 4070-Laptop CSV 逐版本对照； 重生成全部图。
- **为什么**：阶段一清单"简历数字换桌面卡 + 记录 roofline 位置迁移"。
- **关键数字**：**reduce v6/v7 排序反转**（Laptop 回退版 → 4090 最优且反超 cuBLAS 25%）；softmax v4 快 cuBLAS 26%(aligned)/34%(mis)；gemv 84%（含 cuBLAS gemv 本卡表现平平的对照物因素，如实标注）；int8 仓内 baseline 系 CPU 口径，PyTorch-eager 4090 对照移 triton-kernels#EXP-T03《三件套移植 + torch 绑定》。
- **受阻与整改**：NCU 无计数器权限（容器 ERR_NVGPUCTRPERM，细节重过降级为 Laptop ncu-rep 参照）；本仓 bench 覆盖写 CSV 违 CORE bench 铁则， 本次以提交对作版本锚，harness 改造列为后续整改。
- **产物**：records/EXP-K01、四项目新 CSV/stability/图（本提交）， 旧值锚 = 上一提交。
- **下一步**：triton-kernels 仓（FA2/双缓冲 GEMM/三件套移植）开发中， Triton vs CUDA 对比将回引本仓同尺寸数字。

## §2 红线级勘误:softmax 的"cuBLAS"对照系自写 kernel(2026-08-24)

- **发现（用户审计）**：softmax_cublas.cu 无任何 cuBLAS 调用，是自写 warp 原语 kernel（注释自曝）；cuBLAS 本无 softmax API。
- **影响与处置**："softmax 快 cuBLAS 26%/34%"全链作废（EXP-K01《四 kernel 4090 重基准》表格、 README、PORTFOLIO 归因叙事加勘误横幅）；简历禁用该句。gemv/reduce 的 cublas 验真为真库调用，对照有效。
- **教训**：对照物命名必须开源码验真，不能信文件名——CORE 铁律 6 的反面教材实锤；后续任何 "vs X" 声明先查 X 的调用点。

## §3 CUDA Tensor Core GEMM 版本梯落地(2026-08-24)

- **做了什么**：新建 gemm/ 子项目，CUDA 原生 v0 naive → v1 smem tile → v2 wmma → v3 cp.async 双缓冲 → v4 128² 大 tile，对照真 cublasGemmEx， 4096³ 三轮 + ptxas 资源画像；EXP-K02《CUDA Tensor Core GEMM 版本梯》八节记录。
- **为什么**：「CUDA 手写对应哪些算子」盘点暴露 Tensor Core GEMM 只有 Triton 证据、Llama2 引擎代码不在本机——补 CUDA 路线最高优先项。
- **关键数字**：v4 133.1±0.97 TFLOPS = cuBLAS 85.6%；v1→v2 ×13.8（指令世代）；v4 occupancy 33% 最低却最快；vs 自家 Triton 160.5 有 ~17% 差（wmma 无 swizzle，推断级）。
- **产物**：gemm/（源码+CMake+README）、records/EXP-K02、 gemm/project-proof/data/20260824T1517_*（3 轮 raw + derived + ptxas）。
- **下一步**：v5 = mma PTX + ldmatrix + swizzle（backlog，不阻塞）； commit+push 本里程碑。

## §4 CUDA FA2 版本梯:量化 wmma 架构税(2026-08-24)

- **做了什么**：flash-attn/ 子项目，v0 warp-per-row 在线 softmax → v1 smem tile → v2 wmma（smem 往返 softmax）→ v3 8warp → v4 cp.async 重叠 + half S/P；协议对齐 triton-kernels test_fa2.py,3 轮 + ptxas;EXP-K03《CUDA FA2 forward 简化版版本梯》。
- **为什么**：「CUDA 手写对应算子」映射第 ② 项；FA2 此前只有 Triton 证据。
- **关键数字**：全 shape 过 2e-2 gate(GQA/causal)；S=4096 v4 34.8±0.12 TFLOPS = 自家 Triton 28%（跨 harness）；v4 仅 +6.6% → 瓶颈在 smem 往返相位链，不在访存——wmma fragment 布局不透明的结构性代价，mma 的定量论据。
- **产物**：flash-attn/（5 kernel+参考+bench）、records/EXP-K03、 project-proof/data/（3 轮 raw + derived + ptxas）。
- **下一步**：v5 mma PTX（与 gemm v5 同一张技能票，backlog）；commit+push。

## §5 2026-08-24 · 审计收尾批次

- **做了什么**：逐条落实已确认审计 findings（红→低）：①PORTFOLIO 更名 PORTFOLIO.md、"四个项目"改六个，项目 1/Pattern 2/3/面试节全面换用 4090 现行数字（reduce v7 反超真 cuBLAS 24.5%，3 轮），Laptop 旧代 reduce 叙事与 softmax vs-cuBLAS 段按铁律 7 移 docs/archive/（勘误留痕，不删叙事）； ②softmax NCU SUMMARY 加勘误横幅，cublas 行改 handwritten_ref；③flash-attn 固定名 benchmark_results.csv 与 3 轮 UTC 数据核对一致后 git rm（EXP-K03 §7 补记"终端级证据，已删"）；④gemm/flash-attn raw 的 driver=13.3 误填不改 raw， 以两 data/manifest.txt 勘误 + 修两 main.cu 的 provenance 生成（driver=真实 610.57.04，另立 cuda= 字段，只改源码不重跑）；⑤README 死链（layernorm/、 notes/）删除、Suggested Workflow 改指 records/红线表流程；⑥ENV.md 补 4090 容器节并给 Laptop 节标注旧代；⑦各 profiling/ncu/ 补 manifest.txt（sha256+ Laptop 采集环境+日期）；⑧EXP-K01 §7 登记补测 backlog；⑨删 cuda-reduce 两个 0 字节无引用头文件。
- **为什么**：审计确认问题闭环。硬约束：raw 不可变（用 manifest/勘注/归档）， GPU 被另一实验占用禁跑（一切补测转 backlog，相关数字加"单轮"限定）。
- **关键数字**：reduce v7 0.02988±0.00011 ms vs 真 cuBLAS 0.03721±0.00022 ms（快 24.5%,3 轮，records/data/exp_k01_reduce_3rounds.csv）；现行可说集另含 gemm 133.1 TFLOPS=cuBLAS 85.6%(EXP-K02)、FA2 34.8 TFLOPS wmma 架构税（EXP-K03）、quantize 6.6×（Laptop,PyTorch-eager 口径）。
- **产物路径**：PORTFOLIO.md、docs/archive/2026-08-24_portfolio_laptop_era_sections.md、 gemm|flash-attn/project-proof/data/manifest.txt、五个 profiling/ncu/manifest.txt、 records/EXP-K01 §7、records/EXP-K03 §7、ENV.md、README.md、 softmax/project-proof/profiling/ncu/SUMMARY.md。
- **下一步**：GPU 空闲后按 EXP-K01 §7 backlog 给 softmax/gemv/int8-quantize 补 UTC 前缀 3 轮 stability；gemm/fa2 v5(mma PTX)技能票不变。

## §5 2026-08-24 晚 · 老三样 stability 闭环 + gemv 84% 降级

- **做了什么**：softmax/gemv/int8 各 3 轮（stability_rebench.sh）；聚合入 records/data/；EXP-K01 §7 闭环。
- **关键数字**：gemv v3 快 cuBLAS 37.8%（3 轮）——单轮 84% 不可复现，系 cuBLAS 对照单轮波动（0.0235→0.0174）；softmax v4 快自研参照 23.3%； int8 v4 5.57µs。
- **教训**：对照物也要 3 轮；单轮领先幅度可能一半是对照的坏轮。
- **产物**：9 份 UTC raw + 3 份 3rounds 聚合 + 记录/README/PORTFOLIO 同步。
- **下一步**：无（本仓 backlog 清零；v5 mma 技能票仍为可选项）。

## 2026-08-26 LLM 融合逐元素算子三件套(EXP-K05《LLM 融合逐元素算子三件套》)

**做了什么**：新增三个子项目 `fused-norm/`(v0–v4)、`rope/`(v0–v4)、 `activation/`(v0–v3)，共 14 个手写 CUDA kernel；新增共享工装 `scripts/bench_common.py`（统一计时/provenance/扩展构建）与 `scripts/derive_stability.py`（raw→derived 归并）。三个算子各跑 3 轮 × 4 个工作区间， 七/八条臂（手写 v0–v4 + pytorch_eager + torch_compile + triton）。配套文档：三份 README + 三份 project-intro、深度讲义 `docs/lectures/03_memory_bound_fusion.md`、面试讲稿 `docs/talk/fused_ops_talk.md`、白板卡 `docs/talk/whiteboard_card_byte_ledger.md`、记录 `records/EXP-K05`。

**为什么（决策依据）**：①本仓「什么时候该用手写 CUDA」此前只有两个点（GEMM 85.6% / FA2 28%），缺访存主导这一类；②此前所有「vs Triton/SDPA」数字都是跨 harness 对比、只能标注推断级——这次把手写 kernel 绑进 torch，四类臂共用同一段 CUDA-event 计时，直接把这条限定去掉；③三个算子都是自研引擎正在等的东西， 算子级做完可以立刻接进去验端到端。

**关键数字**：HBM 区间手写最优 920.3 / 905.9 / 927.7 GB/s（91.3% / 89.9% / 92.0% 峰值）， 与 Triton、torch.compile 两两差 <2%；相对 pytorch_eager 5.22x / 5.10x / 1.67x。 L2 区间相对 torch.compile 3.19x / 5.61x / 10.6x。七条跑前锁定的预测：四条成立（H1/H4/H5/H6/H7），两条被推翻（H2/H3，fused-norm 的向量化与寄存器缓存双双零收益， 原因由带宽上界反推出「第二次读从未出片」）。

**产物路径**：`{fused-norm,rope,activation}/`、`records/EXP-K05_llm_fused_elementwise.md`、 `docs/lectures/03_memory_bound_fusion.md`、各 `project-proof/data/`。

**踩坑两条**：①torch cpp_extension 构建被中断会残留 `lock` 文件，之后每次 `load()` 无限等锁且没有编译进程，现象极像死机——已在 `bench_common.build_ext` 里主动清理。 ②就地算子的 bench 必须把 clone 放在计时闭包之外；写进去会让该臂每次迭代白搬整个张量，表现为「恰好慢 2 倍」——**恰好整数倍是 harness bug 的指纹**。

**下一步**：接引擎（已完成，见 llm-engine#EXP-D23《融合逐元素算子接入》）；int8 W8A8 linear 接引擎。

## 2026-08-26(续)W8A8 linear 完整链路(EXP-K06《W8A8 linear 完整链路》)

**做了什么**:新增子项目 `w8a8/` —— per-token 动态量化(v0/v1/v2)、反量化
epilogue(v0/v1)、decode 用的 dp4a int8 GEMV(v0/v1),INT8 GEMM 直接用
`torch._int_mm`。bench 覆盖 prefill 四档形状 + decode 三档(按权重工作集分 L2/HBM)。
配套 README + project-intro + records/EXP-K06。

**为什么**:用户指出面试被问「int8 量化算子性能怎么样?有没有放到引擎里?」。
本仓此前只有量化这一步(int8-quantize),而只插量化算子而后面仍走 bf16 GEMM
是负收益。必须补完整链路才能回答。

**关键数字**:prefill 1.905–2.161x;decode HBM 区间 1.972x(L2 区间 5.30x、
跨层级的 8.82x 为无效数字);三步分解 量化 1.8% / GEMM 78.5% / 反量化 26.7%;
**权重多一次 .contiguous() → 整条链路 2.161x 变 0.734x**;
`torch._int_mm` 要求 M>16,decode 走不通(硬约束)。

**产物路径**:`w8a8/`、`records/EXP-K06_w8a8_linear.md`、
`w8a8/project-proof/data/`。

**踩坑**:①bench 里写了多余的 `acc.copy_(torch._int_mm(...))`,白搬 100 MB,
占整条链路近四分之一——分解计时之和对不上总时间就是这类多余搬运的信号;
②int8 GEMV 的 x_scale 第一版用主机 double 传参,调用方要 `.item()`,
那是设备到主机的隐式同步,decode 逐层放大成每 token 上百次;改传设备指针。
③derive_stability.py 遇到 nan 列会抛 AttributeError,已加剔除。

**下一步**:接引擎(已完成,见 llm-engine#EXP-D24)。
