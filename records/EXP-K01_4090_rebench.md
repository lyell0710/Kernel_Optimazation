# EXP-K01 · 四 kernel 4090 重基准:roofline 迁移(4070 Laptop → 4090)

## 0. 元信息

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-23 |
| 环境 | RTX 4090(桌面,~1008GB/s GDDR6X)· CUDA 13.2 · driver 610.57.04;原基线 = RTX 4070 Laptop(git HEAD 内 CSV) |
| 状态 | 完成(NCU 部分受阻,见 §7) |
| 关联清单项 | 阶段一"4 个 kernel 在 4090 重新 benchmark"+"(NVIDIA线)细节重过"(部分) |

## 1. 目的与假设

简历数字换桌面卡;记录 roofline 位置迁移。可证伪假设:各 kernel 相对
cuBLAS/baseline 的**版本间排序在 4090 上可能改变**(带宽/算力/L2 比例不同)。

## 2. 环境与配置

`scripts/run_bench_and_plot_all.sh`(softmax/gemv/int8-quantize)+
cuda-reduce 单独 cmake+`reduce_bench`。各 bench 自带 100 iters + warmup
(stability CSV 仅 cuda-reduce 产出,见 §4 勘误);尺寸与版本矩阵与原仓
完全一致(未改任何源码)。
旧值来源:`git show HEAD:<project>/project-proof/data/benchmark_results.csv`。

## 3. 步骤

构建 → bench → 四组 plot 重生成;新 CSV 覆盖工作区(旧值以 git 历史为锚,
见 §7 整改项)。

## 4. 原始数据

各 `<project>/project-proof/data/benchmark_results.csv`(本次提交即 4090 版;
上一提交即 Laptop 版——版本对 = git 两个 commit)。
**勘误(8/24 审计)**:stability CSV 仅 cuda-reduce 项目存在;softmax/gemv/
int8-quantize 的 bench 不产 stability 文件——4090 头条数字为单轮 100-iter
均值,**无 ≥3 轮 mean±std 支撑**,按 CORE 降级为"单轮实测",进简历前
需补 3 轮(整改项)。

## 5. 结果(末档尺寸,ms;倍率均为同机内比值)

| kernel | 最优版 | 4090 | Laptop | 4090 vs cuBLAS | Laptop vs cuBLAS |
|---|---|---|---|---|---|
| reduce | **v7** | 0.0296 | 1.665(旧 results;与旧 stability 0.273 矛盾,存疑) | **快 25%**(0.0371/0.0296) | v4 慢 1.7%(旧 results 口径) |
| softmax(aligned 1024²) | v4 | 0.0078 | 0.0164 | 快自写参照 26%(0.0098/0.0078,见下勘误) | 同左 |
| softmax(mis 1024×1500) | v4 | 0.0099 | 0.0223 | 快自写参照 34%(0.0133/0.0099) | 同左 |
| gemv | v3 | 0.0128 | 0.0325 | **快 84%**(0.0235/0.0128) | 快 19% |
| int8-quantize | v4 | 0.0059 | — | vs 仓内 baseline 1.8e4×(见 §7 口径) | — |

**Roofline 迁移(本实验最有价值的发现,带勘注)**:
- **4090 上 v7 最优且反超 cuBLAS 25%(0.0296 vs 0.0371)——这是本机实测,
  立得住**。但"Laptop 上 v6/v7 是回退版"的说法**存疑**:旧 results CSV
  记 v7=1.665ms(回退),旧 stability CSV 却记 v7=0.2734ms(全场最快)
  ——两个旧文件自相矛盾,无法确证 Laptop 端排序。故"排序反转"降级为
  "4090 端最优版本为 v7(旧数据端不可考)";简历只引 4090 侧
  "最优实现反超 cuBLAS 25%",不讲反转故事(除非旧机复测)。
- gemv 领先幅度 19%→84%:一半来自我方提速,一半来自 **cuBLAS gemv 在
  4090 上相对表现变差**(0.0402→0.0235 仅 1.7× 硬件增益,低于其余对照)
  ——对照物状态如实标注,不吹成纯自身优势。
- **重大勘误(8/24,红线级)**:softmax 的"cublas"对照**不是 cuBLAS**——
  `softmax/src/softmax_cublas.cu` 是自写 kernel(注释自曝 "cuBLAS-like
  patterns";cuBLAS 亦无 softmax API,该命名从源头不成立)。因此
  "softmax 快 cuBLAS 26%/34%"**全部作废**,降级为"快一个自写优化参照
  26%/34%"——此表述无对外引用价值,softmax 的简历句只能保留绝对时延与
  正确性,不得含任何 cuBLAS 对比。旧 PORTFOLIO 中围绕该对照的 L2/online
  softmax 归因叙事随之作废(那是在归因自家参照 kernel)。
  **验真结果**:gemv_cublas.cu 与 reduce_cublas.cu 为真库调用
  (cublasCreate/handle 在源码),gemv 84% 与 reduce 24.5% 的对照有效。
  违反 CORE 铁律 6(对照物命名诚实)的实例,入方法论教材。

## 6. 分析与结论

排序反转假设**不可判定**(旧数据自相矛盾,见 §5 勘注);4090 端结论独立
成立。简历数字迁移方案:reduce 347.6ms→0.291ms(Laptop 叙事保留)+
4090 版"最优实现(v7)反超 cuBLAS **24.5%**(0.02988±0.0001 vs
0.03721±0.0002 ms,**3 轮 mean±std 已转正**,records/data/
exp_k01_reduce_3rounds.csv)";
softmax 对比句**撤下**(对照系自写 kernel,见 §5 勘误);gemv 84% 须带
"cuBLAS gemv 该卡表现平平"限定或改引 19%(Laptop)保守值。

## 7. 异常、偏差与开放问题

- **NCU 不可用**:本容器 ERR_NVGPUCTRPERM(性能计数器无权限,与 py-spy
  ptrace 同类平台限制)→"细节重过(stall/occupancy)"无法在 4090 复采;
  沿用 `artifacts/ncu_for_mac/` 的 Laptop 时代 ncu-rep 作机理参照,
  4090 归因以带宽模型推断并注明证据等级。
- **int8 baseline 口径**:仓内 baseline 为 CPU 级实现(106ms/1024²),
  1.8e4× 不可对外引用;简历的"比 PyTorch eager 快 6.6×"是另一口径,
  4090 版 PyTorch-eager 对照在 triton-kernels 仓补测(EXP-T03)。
- **整改项(违 CORE bench 铁则)**:本仓 bench 直接覆盖 CSV(trunc 旧值),
  旧值仅靠 git 提交锚定——后续应改 UTC 前缀新文件;本次以"提交对"作
  版本锚,未改 harness(老项目缺哪补哪,不推倒)。
- **stability 覆盖勘误**:仅 cuda-reduce 有 stability CSV(新旧皆有,
  且旧版含 v7=0.2734 与旧 results 的 1.665 矛盾——矛盾本身入档);
  其余三项目 bench 不产 stability 文件。§4 首版"4090 版已覆盖全版本"
  的说法错误,已改。整改项:四 bench 补 ≥3 轮 stability 输出——**reduce 头条已补**(8/24,
  records/data/exp_k01_reduce_3rounds.csv,24.5%);其余三项目待补。

- **backlog(2026-08-24 审计收尾登记;GPU 被另一实验占用,禁跑)**:
  softmax/gemv/int8-quantize 的 4090 数字补测,按 gemm/fa2 模式
  (BENCH_OUT=UTC 前缀新文件 + 首行 provenance + 3 轮 mean/std 落 stability)
  执行,**待 GPU 空闲**;在此之前上述三项目的 4090 数字(含 README 索引的
  gemv 84%)一律带「单轮」限定。

## 8. 下游影响

- 简历 CUDA 段数字按 §6 方案迁移(红线:gemv 84% 必须带对照物限定)。
- 面试素材改为两条:①4090 上 v7 反超 cuBLAS 25%(实测硬);②旧数据
  自相矛盾被审计抓出并如实降级——"数据考古的诚实处理"本身是素材。
- int8 的 PyTorch-eager 4090 对照 → triton-kernels#EXP-T03 一并出。

## §7 整改闭环(2026-08-24 晚:三项目 ≥3 轮 stability 补测)

工装 `scripts/stability_rebench.sh`(UTC 前缀+provenance,既有 raw 原样复位);
raw = 各 `project-proof/data/2026*_stability_r{1,2,3}.csv`,聚合 =
`records/data/exp_k01_{softmax,gemv,int8_quantize}_3rounds.csv`。

- **gemv:"快 84%" 单轮值不可复现,降级为"快 37.8%(3 轮)"**。v3 自身完全
  复现(0.012651±0.000031 vs 单轮 0.012818),但 cuBLAS 对照从单轮 0.02353
  变为 3 轮 0.017432±0.000169(−26%)——领先幅度的一半以上原系 cuBLAS 侧
  单轮波动。§5 表的 84% 作废,现行口径 = **v3 比 cuBLAS gemv 快 37.8%
  (4096×2048,3 轮)**;cuBLAS 侧单轮为何慢 35% 未剖(冷启动/时钟态候选),
  列开放问题。§6 "领先 19%→84%" 的归因叙事一并作废。
- softmax:v4 0.007691±0.000153 vs handwritten_ref 0.009482±0.000133
  (快 23.3%,3 轮;对照系自研 warp 参照,非 cuBLAS——勘误口径不变)。
- int8-quantize:v4 0.005570±0.000031 ms(单轮 0.0059 复现向好),
  baseline 104.487±0.028 ms → 1.88e4×。
- 教训写进方法论:**对照物也要 3 轮**——自家 kernel 稳定不代表对照稳定,
  单轮领先幅度里可能一半是对照的坏轮。
