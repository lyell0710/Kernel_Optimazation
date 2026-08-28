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

**做了什么**：新增子项目 `w8a8/`—— per-token 动态量化（v0/v1/v2）、反量化 epilogue(v0/v1)、decode 用的 dp4a int8 GEMV(v0/v1)，INT8 GEMM 直接用 `torch._int_mm`。bench 覆盖 prefill 四档形状 + decode 三档（按权重工作集分 L2/HBM）。配套 README + project-intro + records/EXP-K06。

**为什么**：用户指出面试被问「int8 量化算子性能怎么样？有没有放到引擎里？」。本仓此前只有量化这一步（int8-quantize），而只插量化算子而后面仍走 bf16 GEMM 是负收益。必须补完整链路才能回答。

**关键数字**：prefill 1.905–2.161x；decode HBM 区间 1.972x（L2 区间 5.30x、跨层级的 8.82x 为无效数字）；三步分解 量化 1.8% / GEMM 78.5% / 反量化 26.7%； **权重多一次 .contiguous() → 整条链路 2.161x 变 0.734x**； `torch._int_mm` 要求 M>16，decode 走不通（硬约束）。

**产物路径**：`w8a8/`、`records/EXP-K06_w8a8_linear.md`、 `w8a8/project-proof/data/`。

**踩坑**：①bench 里写了多余的 `acc.copy_(torch._int_mm(...))`，白搬 100 MB， 占整条链路近四分之一——分解计时之和对不上总时间就是这类多余搬运的信号； ②int8 GEMV 的 x_scale 第一版用主机 double 传参，调用方要 `.item()`， 那是设备到主机的隐式同步，decode 逐层放大成每 token 上百次；改传设备指针。 ③derive_stability.py 遇到 nan 列会抛 AttributeError，已加剔除。

**下一步**：接引擎（已完成，见 llm-engine#EXP-D24）。

## 2026-08-28 NCU 采集设施补齐(十算子全覆盖,待权限)

**做了什么**：①核实本机 NCU 现状——实采烟测返回 `ERR_NVGPUCTRPERM`，`RmProfilingAdminOnly=1` 且容器 CapEff 无 `CAP_SYS_ADMIN`/`CAP_PERFMON`，容器内无解；②核实现存 38 份 `.ncu-rep` 的出处——全部来自笔记本 `ubuntu22`，GPU 是 **RTX 4070 Laptop GPU（36 SM / 8 GB）而非 4090**（桌面 4090 为 128 SM / 24 GB，差 3.6 倍 SM；此前口头误称 4090，已按报告 Session 页更正，ENV.md 与 EXP-K01 标题原本记载正确）；且跨两批采集：2026-05-03 用 NCU 2026.1.1.0（6 份）、05-23 用 2022.4.1.0（32 份），`cuda-reduce` 两批都有。只覆盖 `cuda-reduce`/`gemv`/`int8-quantize`/`softmax` 四个算子；③为零覆盖的六个算子（`gemm`/`flash-attn`/`fused-norm`/`rope`/`activation`/`w8a8`）补齐 `project-proof/scripts/profile_ncu.sh`，共用逻辑抽到 `scripts/ncu_profile_lib.inc.sh`；④`scripts/run_ncu_all.sh` 扩到十算子；⑤新增 `scripts/export_ncu_for_mac.py`（口径校验 + 分类导出 + MANIFEST + tar.gz）；⑥新增 `docs/ncu_reading_guide.md`（优化手法↔NCU 证据的映射，及权限恢复后待转实测的推断清单）。

**为什么（决策依据）**：零覆盖的六个算子恰好都是迁入当前容器之后写的，权限缺失把它们的归因全压成了 roofline 推断——`docs/lectures/03_memory_bound_fusion.md` 等处共有 5 条本仓推断 + 3 条跨仓推断明确写着「想验证但无计数器权限」。采集脚本先写好，权限一到即可一条命令跑完，不必届时再返工。

**关键决定三条**：①`-k regex:` 必须用 `^` 锚定——`w8a8` 的 `v1_kernel` 是 `dequant_v1_kernel`/`gemv_v1_kernel` 的子串，不锚定会把三个不同 kernel 混进同一份报告，跨版本对比直接失效；②`ncu_metrics.inc.sh` 追加 `ComputeWorkloadAnalysis`——原八件是为访存型算子选的，没有算力侧分解，而 `gemm` v1→v2 与 `fa2` v1→v2 的核心就是上没上 Tensor Core，缺这一节那一跳只能靠 SOL 的 SM% 间接猜。追加不破坏兼容，旧报告的原八件仍逐 metric 可比；③python 家族的 bench 逐 regime 跑同一 kernel 且 `timeit` 是 warmup=10+iters，一个 kernel 一次进程会 launch 几十次并横跨多个访存区间，默认全采后由 `export_ncu_for_mac.py` 报出 grid 分布，再用 `NCU_SKIP`/`NCU_COUNT` 钉窗口重采——不这么做就会把 L2 区间与 HBM 区间混为一谈（EXP-K04 的教训）。

**校验器抓到的事**：`export_ncu_for_mac.py` 对现存 38 份报告跑出 FAIL 0 / WARN 8；8 条 WARN 全是 `cuda-reduce` 的多级归约树（grid `65536→256→1` 三级同一 kernel），属合法形态而非混样——脚本因此只报事实、不下断言，判断留给人。另核实 `softmax_cublas_profile` 内是 `softmax_cublas_kernel`（自写 kernel，非 cuBLAS），`gemv_cublas_profile` 内是 `gemv2T_kernel_val<...cublasGemvParams...>`（真 cuBLAS），已写入 MANIFEST 的口径陷阱一节。

**产物路径**：`scripts/ncu_profile_lib.inc.sh`、`scripts/export_ncu_for_mac.py`、`scripts/run_ncu_all.sh`、`{gemm,flash-attn,fused-norm,rope,activation,w8a8}/project-proof/scripts/profile_ncu.sh`、`docs/ncu_reading_guide.md`、`artifacts/ncu_for_mac/`（38 份分类报告 + MANIFEST.md + manifest.csv + 2.1 MB tar.gz）。

**实例形态（决定申请话术）**：`/dev/nvidia4`、`/dev/nvidia5` 表明本容器拿到的是宿主第 5、6 号卡，宿主至少 6 卡且为多租户共享（hostname `cpod-*`，k8s pod）。因此「改宿主 `NVreg_RestrictProfilingToAdminUsers=0`」需重载驱动模块、打断同宿主其他租户，且一开即对全部租户生效；「`--cap-add=SYS_ADMIN`」在共享节点上是提权面。**两者都不应作为申请内容**，应改问「有无支持性能计数器的机型（独占整机/裸金属）」。新增 `scripts/probe_ncu_permission.sh`，换机器先跑它判定能否采集（本机实跑退出码 2）。

**下一步**：向 compshare 询问支持 profiling 的机型；机理类问题（Tensor pipe 是否点亮、long scoreboard stall 是否下降、sector/request 是否下降、融合后中间结果是否出片）不依赖 SM 数与 L2 容量，**可在 4070 Laptop 上先行验证**；占用率、波量化、L2 区间相关结论必须等 4090。获权后 `bash scripts/run_ncu_all.sh` 一次跑完十算子，按 WARN 的 grid 分布钉窗口重采，再逐条转实测 `docs/ncu_reading_guide.md` §4 的 5+3 条推断。
