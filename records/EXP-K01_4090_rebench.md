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
cuda-reduce 单独 cmake+`reduce_bench`。各 bench 自带 100 iters + warmup +
stability CSV(mean±std);尺寸与版本矩阵与原仓完全一致(未改任何源码)。
旧值来源:`git show HEAD:<project>/project-proof/data/benchmark_results.csv`。

## 3. 步骤

构建 → bench → 四组 plot 重生成;新 CSV 覆盖工作区(旧值以 git 历史为锚,
见 §7 整改项)。

## 4. 原始数据

各 `<project>/project-proof/data/benchmark_results.csv` + `benchmark_stability.csv`
(本次提交即 4090 版;上一提交即 Laptop 版——版本对 = git 两个 commit)。

## 5. 结果(末档尺寸,ms;倍率均为同机内比值)

| kernel | 最优版 | 4090 | Laptop | 4090 vs cuBLAS | Laptop vs cuBLAS |
|---|---|---|---|---|---|
| reduce | **v7** | 0.0296 | 1.665(彼时回退版) | **快 25%**(0.0371/0.0296) | v4 慢 1.7%(彼时最优) |
| softmax(aligned 1024²) | v4 | 0.0078 | 0.0164 | **快 26%**(0.0098/0.0078) | 快 26% |
| softmax(mis 1024×1500) | v4 | 0.0099 | 0.0223 | **快 34%**(0.0133/0.0099) | 快 26% |
| gemv | v3 | 0.0128 | 0.0325 | **快 84%**(0.0235/0.0128) | 快 19% |
| int8-quantize | v4 | 0.0059 | — | vs 仓内 baseline 1.8e4×(见 §7 口径) | — |

**Roofline 迁移(本实验最有价值的发现)**:
- **reduce v6/v7 排序反转**:Laptop 上 v6/v7 是回退版(1.66ms,慢于 v4/v5
  的 0.29),4090 上反成最优(0.0296)且**反超 cuBLAS 25%**——同一代码,
  硬件带宽/SM 配比变了,最优实现随之改变。简历的"与 cuBLAS 差 1.7%"
  在 4090 上应升级为"快 25%"(版本号从 v4 换 v7,须注明)。
- gemv 领先幅度 19%→84%:一半来自我方提速,一半来自 **cuBLAS gemv 在
  4090 上相对表现变差**(0.0402→0.0235 仅 1.7× 硬件增益,低于其余对照)
  ——对照物状态如实标注,不吹成纯自身优势。
- softmax 结论稳定(26%→26/34%),misaligned 组扩大——4090 带宽更高,
  cuBLAS 的 L2 策略优势相对缩水,与 Laptop 时代 NCU 归因(它赢在 L2)自洽。

## 6. 分析与结论

排序反转假设成立(reduce)。简历数字迁移方案:reduce 347.6ms→0.291ms
(Laptop 叙事保留)+ 4090 版"最优实现反超 cuBLAS 25%(0.0296 vs 0.0371ms)";
softmax 26%(4090 aligned 同值,mis 34%);gemv 84% 须带"cuBLAS gemv 该卡
表现平平"限定或改引 19%(Laptop)保守值。

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
- reduce 的 stability 旧 CSV 未含 v6/v7 行(彼时回退未入稳定表),4090 版
  已覆盖全版本。

## 8. 下游影响

- 简历 CUDA 段数字按 §6 方案迁移(红线:gemv 84% 必须带对照物限定)。
- reduce 排序反转 = 面试"roofline 迁移"最佳素材(同码不同卡,最优解变)。
- int8 的 PyTorch-eager 4090 对照 → triton-kernels#EXP-T03 一并出。
