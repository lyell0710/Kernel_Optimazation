# LAB_JOURNAL — Kernel_Optimazation

## §1 4090 重基准 + roofline 迁移(2026-08-23,EXP-K01)

- **做了什么**:四 kernel(reduce/softmax/gemv/int8-quantize)在 4090 全量
  重跑既有 bench(未改源码),与 git HEAD 里的 4070-Laptop CSV 逐版本对照;
  重生成全部图。
- **为什么**:阶段一清单"简历数字换桌面卡 + 记录 roofline 位置迁移"。
- **关键数字**:**reduce v6/v7 排序反转**(Laptop 回退版 → 4090 最优且反超
  cuBLAS 25%);softmax v4 快 cuBLAS 26%(aligned)/34%(mis);gemv 84%
  (含 cuBLAS gemv 本卡表现平平的对照物因素,如实标注);int8 仓内 baseline
  系 CPU 口径,PyTorch-eager 4090 对照移 triton-kernels#EXP-T03。
- **受阻与整改**:NCU 无计数器权限(容器 ERR_NVGPUCTRPERM,细节重过降级
  为 Laptop ncu-rep 参照);本仓 bench 覆盖写 CSV 违 CORE bench 铁则,
  本次以提交对作版本锚,harness 改造列为后续整改。
- **产物**:records/EXP-K01、四项目新 CSV/stability/图(本提交),
  旧值锚 = 上一提交。
- **下一步**:triton-kernels 仓(FA2/双缓冲 GEMM/三件套移植)开发中,
  Triton vs CUDA 对比将回引本仓同尺寸数字。

## §2 红线级勘误:softmax 的"cuBLAS"对照系自写 kernel(2026-08-24)

- **发现(用户审计)**:softmax_cublas.cu 无任何 cuBLAS 调用,是自写
  warp 原语 kernel(注释自曝);cuBLAS 本无 softmax API。
- **影响与处置**:"softmax 快 cuBLAS 26%/34%"全链作废(EXP-K01 表格、
  README、PORTFOLIO 归因叙事加勘误横幅);简历禁用该句。gemv/reduce 的
  cublas 验真为真库调用,对照有效。
- **教训**:对照物命名必须开源码验真,不能信文件名——CORE 铁律 6 的
  反面教材实锤;后续任何 "vs X" 声明先查 X 的调用点。

## §3 CUDA Tensor Core GEMM 版本梯落地(2026-08-24)

- **做了什么**:新建 gemm/ 子项目,CUDA 原生 v0 naive → v1 smem tile →
  v2 wmma → v3 cp.async 双缓冲 → v4 128² 大 tile,对照真 cublasGemmEx,
  4096³ 三轮 + ptxas 资源画像;EXP-K02 八节记录。
- **为什么**:「CUDA 手写对应哪些算子」盘点暴露 Tensor Core GEMM 只有
  Triton 证据、Llama2 引擎代码不在本机——补 CUDA 路线最高优先项。
- **关键数字**:v4 133.1±0.97 TFLOPS = cuBLAS 85.6%;v1→v2 ×13.8(指令
  世代);v4 occupancy 33% 最低却最快;vs 自家 Triton 160.5 有 ~17% 差
  (wmma 无 swizzle,推断级)。
- **产物**:gemm/(源码+CMake+README)、records/EXP-K02、
  gemm/project-proof/data/20260824T1517_*(3 轮 raw + derived + ptxas)。
- **下一步**:v5 = mma PTX + ldmatrix + swizzle(backlog,不阻塞);
  commit+push 本里程碑。

## §4 CUDA FA2 版本梯:量化 wmma 架构税(2026-08-24)

- **做了什么**:flash-attn/ 子项目,v0 warp-per-row 在线 softmax → v1 smem
  tile → v2 wmma(smem 往返 softmax)→ v3 8warp → v4 cp.async 重叠 + half
  S/P;协议对齐 triton-kernels test_fa2.py,3 轮 + ptxas;EXP-K03。
- **为什么**:「CUDA 手写对应算子」映射第 ② 项;FA2 此前只有 Triton 证据。
- **关键数字**:全 shape 过 2e-2 gate(GQA/causal);S=4096 v4 34.8±0.12
  TFLOPS = 自家 Triton 28%(跨 harness);v4 仅 +6.6% → 瓶颈在 smem 往返
  相位链,不在访存——wmma fragment 布局不透明的结构性代价,mma 的定量论据。
- **产物**:flash-attn/(5 kernel+参考+bench)、records/EXP-K03、
  project-proof/data/(3 轮 raw + derived + ptxas)。
- **下一步**:v5 mma PTX(与 gemm v5 同一张技能票,backlog);commit+push。

## §5 2026-08-24 · 审计收尾批次

- **做了什么**:逐条落实已确认审计 findings(红→低):①PORTFOLIO 更名
  PORTFOLIO.md、"四个项目"改六个,项目 1/Pattern 2/3/面试节全面换用 4090
  现行数字(reduce v7 反超真 cuBLAS 24.5%,3 轮),Laptop 旧代 reduce 叙事与
  softmax vs-cuBLAS 段按铁律 7 移 docs/archive/(勘误留痕,不删叙事);
  ②softmax NCU SUMMARY 加勘误横幅,cublas 行改 handwritten_ref;③flash-attn
  固定名 benchmark_results.csv 与 3 轮 UTC 数据核对一致后 git rm(EXP-K03 §7
  补记"终端级证据,已删");④gemm/flash-attn raw 的 driver=13.3 误填不改 raw,
  以两 data/manifest.txt 勘误 + 修两 main.cu 的 provenance 生成(driver=真实
  610.57.04,另立 cuda= 字段,只改源码不重跑);⑤README 死链(layernorm/、
  notes/)删除、Suggested Workflow 改指 records/红线表流程;⑥ENV.md 补 4090
  容器节并给 Laptop 节标注旧代;⑦各 profiling/ncu/ 补 manifest.txt(sha256+
  Laptop 采集环境+日期);⑧EXP-K01 §7 登记补测 backlog;⑨删 cuda-reduce 两个
  0 字节无引用头文件。
- **为什么**:审计确认问题闭环。硬约束:raw 不可变(用 manifest/勘注/归档),
  GPU 被另一实验占用禁跑(一切补测转 backlog,相关数字加"单轮"限定)。
- **关键数字**:reduce v7 0.02988±0.00011 ms vs 真 cuBLAS 0.03721±0.00022 ms
  (快 24.5%,3 轮,records/data/exp_k01_reduce_3rounds.csv);现行可说集另含
  gemm 133.1 TFLOPS=cuBLAS 85.6%(EXP-K02)、FA2 34.8 TFLOPS wmma 架构税
  (EXP-K03)、quantize 6.6×(Laptop,PyTorch-eager 口径)。
- **产物路径**:PORTFOLIO.md、docs/archive/2026-08-24_portfolio_laptop_era_sections.md、
  gemm|flash-attn/project-proof/data/manifest.txt、五个 profiling/ncu/manifest.txt、
  records/EXP-K01 §7、records/EXP-K03 §7、ENV.md、README.md、
  softmax/project-proof/profiling/ncu/SUMMARY.md。
- **下一步**:GPU 空闲后按 EXP-K01 §7 backlog 给 softmax/gemv/int8-quantize
  补 UTC 前缀 3 轮 stability;gemm/fa2 v5(mma PTX)技能票不变。

## §5 2026-08-24 晚 · 老三样 stability 闭环 + gemv 84% 降级

- **做了什么**:softmax/gemv/int8 各 3 轮(stability_rebench.sh);聚合入
  records/data/;EXP-K01 §7 闭环。
- **关键数字**:gemv v3 快 cuBLAS 37.8%(3 轮)——单轮 84% 不可复现,系
  cuBLAS 对照单轮波动(0.0235→0.0174);softmax v4 快自研参照 23.3%;
  int8 v4 5.57µs。
- **教训**:对照物也要 3 轮;单轮领先幅度可能一半是对照的坏轮。
- **产物**:9 份 UTC raw + 3 份 3rounds 聚合 + 记录/README/PORTFOLIO 同步。
- **下一步**:无(本仓 backlog 清零;v5 mma 技能票仍为可选项)。
