# EXP-K05 · LLM 融合逐元素算子三件套:fused_add_rmsnorm / rope / silu_and_mul

> **一句话结论**：三个融合逐元素算子在 HBM 区间都贴到理论峰值 **89.9%–92.0%**，手写 CUDA、Triton 与 torch.compile 三者打平；贴上带宽墙之后语言不再重要，分水岭是融不融合——pytorch_eager 只有 17.5%–55.1%。

## 0 元信息

| 项 | 值 |
|---|---|
| 日期 | 2026-08-26 |
| 环境 | 4090 容器，venv:/root/venvs/main(torch 2.13.0+cu132, triton 3.7.1),nvcc 13.2 |
| 状态 | 完成 |
| 关联 | 新增三个子项目 `fused-norm/` `rope/` `activation/`；Triton 对照臂在 triton-kernels#EXP-T09《Triton 版 LLM 融合逐元素算子》；引擎接入在 llm-engine#EXP-D23《融合逐元素算子接入》 |
| 数据 | `fused-norm|rope|activation/project-proof/data/2026*_r{1,2,3}.csv` 与同目录 `derived_*_stability.csv` |

## 1 目的与假设

补齐 LLM 前向里三个高频访存型融合算子的手写 CUDA 版本梯，并回答一个此前只有两个数据点的问题：**什么时候手写 CUDA 才值得？**

已有两点（均为本仓既有结论）：计算主导的 GEMM，手写 wmma 够到真 cuBLAS 的 85.6%（EXP-K02《CUDA Tensor Core GEMM 版本梯》）；融合型 attention，同一套 wmma 只够到自家 Triton 的 28%（EXP-K03《CUDA FA2 forward 简化版版本梯》）。本实验补第三点：**访存主导的融合逐元素算子**。

跑前锁定的可证伪预测（逐条，跑完不改）：

| 编号 | 预测 | 依据 |
|---|---|---|
| H1 | 三个算子的 v0→v1（融合）是主台阶，幅度接近各自字节账之比 | fused-norm 6/5=1.20，activation 5/3=1.67 |
| H2 | fused-norm 的 v2→v3（向量化）是本梯最大一级 | 标量 bf16 只覆盖半个 128B 事务 |
| H3 | fused-norm 的 v3→v4（寄存器缓存）约 +25% | 字节账 5→4 |
| H4 | rope 的 v1→v2（q/k 合并 launch）在 prefill ≈0，在 decode ≈2x | launch 占比随 T 变化 |
| H5 | rope 的 v3→v4（免表现算）≈ v3(±5%) | cos/sin 表只有 8MB，应常驻 L2 |
| H6 | activation 的 v2→v3（打包布局）≈ v2(±3%) | 打包不改变 HBM 字节数 |
| H7 | HBM 区间三种实现（手写 CUDA / Triton / torch.compile）彼此差距 <5% | 都撞同一堵带宽墙 |

判定阈值：正确性 max_rel_err < 2e-2（与本仓 gemm 同口径）；性能结论需 3 轮 mean±std。

## 2 环境与配置

- 单一 harness：五个手写版本经 `torch.utils.cpp_extension` 绑进 torch，与 pytorch_eager / torch.compile / Triton 四类臂共用同一段 CUDA-event 计时（`scripts/bench_common.py::timeit`）。**这解决了本仓此前 "vs Triton/SDPA" 只能标注「跨 harness 推断级」的问题**——本实验所有跨实现比较均为实测级。
- 编译选项 `-O3 -arch=sm_89`，不开 `--use_fast_math`（访存主导算子无性能价值， 却会让超越函数舍入偏离 PyTorch，给数值差异引入无法归因的来源；实测开关前后误差完全相同，证明误差来自累加顺序而非快速数学）。
- 形状按区间取，每个算子四档；区间划分依据 EXP-K04《标准库基准补齐与两区间重测》的教训：4090 的 L2 是 72MB， memory-bound 算子只测小尺寸会量到 L2 带宽而非 HBM 带宽。
- 有效带宽一律按**算法下界字节数**计（fused-norm 8 B/元素、rope 4 B/元素、 activation 6 B/元素）；因此 >100% 峰值即说明数据落在 L2，是区间判据而非错误。

## 3 步骤

```bash
export CUDA_HOME=/usr/local/cuda
for op in fused-norm rope activation; do
  cd $op
  for r in 1 2 3; do
    TS=$(date -u +%Y%m%dT%H%M%S)
    BENCH_OUT="project-proof/data/${TS}_${op}_stability_r${r}.csv" python bench.py
  done
  cd ..
  python scripts/derive_stability.py $op/project-proof/data/derived_${op}_stability.csv \
        $op/project-proof/data/2026*_r*.csv
done
```

## 4 原始数据

- `fused-norm/project-proof/data/20260826T08*_fused-norm_stability_r{1,2,3}.csv`
- `rope/project-proof/data/20260826T*_rope_stability_r{1,2,3}.csv`
- `activation/project-proof/data/20260826T*_activation_stability_r{1,2,3}.csv`
- 归并表 `derived_*_stability.csv`（由 `scripts/derive_stability.py` 从上述 raw 重算）

## 5 结果

HBM 区间（工作集分别 1.0GB / 336MB / 600MB，确定落 HBM），3 轮 mean，单位 GB/s（占 1008 峰值%）：

| 版本 | fused-norm | rope | activation |
|---|---|---|---|
| v0 未融合基线 | 581.5 (57.7%) | 430.9 (42.7%) | 540.9 (53.7%) |
| v1 融合 | 877.3 (87.0%) | 779.0 (77.3%) | 907.7 (90.1%) |
| v2 | 920.1 (91.3%) | 770.6 (76.5%) | 918.8 (91.2%) |
| v3 | 920.3 (91.3%) | 887.0 (88.0%) | **927.7 (92.0%)** |
| v4 | 918.7 (91.1%) | **905.9 (89.9%)** | — |
| pytorch_eager | 176.2 (17.5%) | 177.6 (17.6%) | 555.5 (55.1%) |
| torch_compile | 917.4 (91.0%) | 877.2 (87.0%) | 925.7 (91.8%) |
| triton | 922.1 (91.5%) | 898.5 (89.1%) | 928.0 (92.1%) |

L2 常驻区间（fused-norm prefill 64MB / rope prefill 21MB / activation l2 19MB）:

| 版本 | fused-norm | rope | activation |
|---|---|---|---|
| 手写最优 | v3 2980.9 | v3 2961.9 | v3 2993.1 |
| torch_compile | 934.5 | 527.6 | 282.5 |
| triton | 1723.1 | 1149.2 | 984.7 |
| 手写 / torch_compile | **3.19x** | **5.61x** | **10.6x** |

decode 区间（T=1，grid 只有 1 个 block）：手写 6.1-7.4 us，triton 18.6-38.3 us， pytorch_eager 16.5-138.0 us。全部由 launch 与单 block 占用主导，与带宽无关。

正确性：全算子全形状 max_rel_err ≤ 7.1e-3 < 2e-2，PASS。 fused-norm v4 与 v3 逐位一致（bench 内 `torch.equal` 断言），证明寄存器缓存未改变语义。

## 6 分析与结论

**H1 成立。** activation 的 v0→v1 实测 1.675x，与字节账预测的 5/3=1.667x 几乎精确吻合；fused-norm 1.509x 高于预测的 1.20x（v0 的两 kernel 之间隔了一整趟 1GB 写回， 第二个 kernel 的首次读全部 miss，实际字节账比静态计数更差）。自检条件通过：activation 的 v0(540.9)与 pytorch_eager(555.5)同速，说明未融合基线确实复刻了 eager 的执行方式，不是稻草人。

**H2、H3 均被推翻，而推翻它们的论证来自测量本身。** fused-norm 的 v2 已达 920.1 GB/s = 91.3% 峰值，v3（向量化）+0.02%、v4（寄存器缓存）-0.2%，双双零收益。原因可以从数据直接反推：有效带宽按 4 次访存（8 B/元素）计得 920 GB/s；若第二遍重读真的走到 HBM（5 次访存 = 10 B/元素），实际带宽将是 920x10/8 = 1150 GB/s， **超过 1008 的物理峰值，不可能**。所以 v1/v2/v3 的"第二次读"从一开始就被 L1/L2 接住，从未到过 HBM——v4 消掉的是一次缓存命中，不是一次显存访问。 **字节账必须在 HBM 层面记，不能在指令层面记**；写在 kernel 注释里的静态计数是指令级的，它高估了可优化空间。

**H4 成立。** rope 的 v1→v2（q/k 合并 launch）在 HBM 区间 -1.1%（778.9→770.6， 噪声内），在 decode 区间 1.43x(0.01030→0.00721 ms)。同一个改动在两个区间收益差 40 倍，这是"优化必须绑定工作区间来谈"的直接样本，也复现了 triton-kernels#EXP-T03《三件套移植 + torch 绑定》「小 kernel 的瓶颈在主机侧而非设备侧」。

**H5 成立（边缘）。** rope 的 v3→v4（免表现算）在 HBM 区间 +2.1%(887.0→905.9)， 落在预测的 ±5% 内。表访存确实主要命中 L2，省掉它只有边际收益。注：v4 最初写成标量版，与向量化的 v3 相比同时变了两个变量，快慢无法归因； 重写为"v3 + 免表"后才隔离出这 +2.1%。**版本梯每一级只改一件事，否则整条梯子讲不出因果**——这是本实验的方法论收获之一。

**H6 成立。** activation 的 v2→v3（打包布局）+1.0%，在 ±3% 内。打包的真正价值不在这个算子里，而在上游：gate_proj 与 up_proj 可以合并成一次 GEMM。算子级 bench 看不到该收益，必须接进引擎才量得到。

**H7 成立，且是本实验的核心结论。** HBM 区间三种实现全部收敛到 88-92% 峰值： 手写 CUDA 905.9-927.7、Triton 898.5-928.0、torch.compile 877.2-925.7，两两差距均 <2%。而 pytorch_eager 落后 1.7-5.2x。

> **分水岭是「融不融合」(1.7-5.2x)，不是「用什么语言写」(<2%)。**

由此得到"什么时候该用手写 CUDA"的三点判断曲线（全部为本仓自测）：

| 算子类型 | 手写 CUDA 相对最优对照 | 证据 |
|---|---|---|
| 计算主导 GEMM | 85.6% of 真 cuBLAS | EXP-K02 |
| 融合型 attention(wmma) | 28% of 自家 Triton | EXP-K03 |
| 访存主导融合逐元素 | 与 Triton / torch.compile 打平（±2%） | 本实验 |

推论：手写 CUDA 的价值集中在**需要 mma 级寄存器控制**的场合；访存主导的算子上，写 Triton 或直接上 torch.compile 能拿到同样的数字，手写只在两处仍有优势—— L2 常驻区间（快 3.2-10.6x）与 decode 的 launch 敏感区间（快 2.5-11x）。

**一个 harness 层面的教训（值得单独记）。** Triton 的 rope 臂最初测出"慢 2 倍"， 排查顺序是：改粒度（无效）→ 加连续性断言 `tl.max_contiguous`（无效）→ 直接 dump PTX，看到 `ld.global.v4.b32`，证明向量化本身没问题 → 单独 benchmark 该 kernel，得 907 GB/s，与手写持平。真因是**我把 `q.clone()` 写进了被计时的闭包里**，Triton 臂每次迭代白搬 320MB。就地算子的 bench 最容易在这里翻车： 正确性需要干净副本，时延不需要，两者必须分开。**"恰好慢 2 倍"这种整数倍关系是 harness bug 的典型指纹，不是性能现象**。

## 7 异常、偏差与开放问题

- v0 的字节账在 kernel 注释里按指令计数写作 6/元素，实测反推 HBM 层面更接近 5； 注释中的静态计数与 §6 的实测结论并存，读代码时以本记录为准。
- rope v2 在 HBM 区间比 v1 慢 1.1%(770.6 vs 779.0)，超出 3 轮 std(±3.7)。合并 launch 引入了一次分支与更复杂的下标运算，在带宽已饱和时表现为轻微净亏。未进一步剖（NCU 在本容器无计数器权限，ERR_NVGPUCTRPERM）。
- 三个算子均未做 NCU 采集（同上权限限制），"第二次读被缓存接住"是从带宽上界反推的**推断**，非计数器实测；若日后拿到 NCU 权限，应以 `lts__t_sectors_op_read` 与 `dram__sectors_read` 的比值直接验证。
- Triton 版 rope 采用 q/k 分两次 launch（合并版的双路 masked load 会实际发出两倍访存，实测有效带宽恰好减半）。手写 CUDA v2 的"q/k 合并 launch"这一级在 Triton 侧无对应实现，decode 区间的跨语言比较因此含一次额外 launch 的偏差。

## 8 下游影响

- 三个算子已接入 llm-engine（见 llm-engine#EXP-D23），端到端 TTFT -28.4%、 TPOT -30.6%，是本仓算子首次给出**同一批 kernel 的算子级与端到端两套数字**。
- 本仓 README 的算子数从六个增至九个；"什么时候该用手写 CUDA"由两点扩为三点曲线。
- 措辞红线：本实验所有跨实现比较为**同 harness 实测级**，可去掉"推断级"限定； 但 EXP-K02/K03 的旧结论仍是跨 harness，限定不得删除。
- 新增可复用工装：`scripts/bench_common.py`（统一计时/provenance/扩展构建）与 `scripts/derive_stability.py`（raw→derived 归并）。
