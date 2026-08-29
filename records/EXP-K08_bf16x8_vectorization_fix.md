# EXP-K08 · BF16x8 向量化未兑现的定位与修复：从 alignas 到 union

> 一句话结论：三个逐元素算子声称的「16 B 向量化」在 SASS 层从未兑现——`alignas(16)`
> 只保证地址对齐、不强制向量化访存。修复 `fused-norm` 后 L2 常驻区间 v3 **+21.3%**、
> v4 **+41.8%**（同环境同轮数 A/B，未改动的 v1/v2 对照组 +0.1%）。

## 0 元信息

| 项 | 值 |
|---|---|
| 日期 | 2026-08-29 |
| 环境 | RTX 4090 · driver 610.57.04 · CUDA 13.0 · torch 2.11.0+cu130 · miniconda py312 |
| 代码版本 | 基线 = `f95a0a8`；改动见本记录 §3 |
| 状态 | 部分完成（`fused-norm` 已修并验证；`rope` / `activation` / `w8a8` 未改） |
| 关联 | [EXP-K05《LLM 融合逐元素算子三件套》](EXP-K05_llm_fused_elementwise.md)、[EXP-K07《NCU 计数器闭环》](EXP-K07_ncu_counter_closure.md)、`docs/sass_evidence_ladder.md` |

## 1 目的与假设

`docs/sass_evidence_ladder.md` 的 SASS 静态分析发现：`fused-norm` v3/v4、`rope` v3/v4、
`activation` v2/v3、`w8a8` quant.v2 的 SASS 里 `LDG.E.128` **全为 0**，
而各自 README 与源码注释均声称做了 16 字节向量化访存。
`fused_norm_v3.cu` 的注释更直接断言「`alignas(16)` 让编译器放心发 `LDG.E.128`」。

- **H1**：向量化未兑现的根因是类型选择，不是对齐属性缺失。
  判据：改用编译器认得的向量类型后 `LDG.E.128` 出现。
- **H2**：兑现后在 **L2 常驻区间**有可观收益，在 **HBM 区间**没有。
  判据：prefill/l2 区间提升显著且超出 3 轮 std；hbm 区间变化落在 std 内。
  依据是 EXP-K04 的两区间口径——HBM 区间已在 91-92% 峰值，访存指令条数不是瓶颈。
- **H3（跑前锁定的证伪条件）**：若 v1/v2（不使用 `BF16x8`、代码未改）
  在同一轮次里也出现同向变化，则增益不可归因到向量化。

## 2 环境与配置

同一台机器、同一个 python 环境、同一组 bench 参数，**只有源码一处不同**。

早先曾用 `venv:/root/venvs/main`（torch 2.13.0+cu132 / CUDA 13.2）的历史基线做过对比，
但那是跨环境比较：decode 区间出现 ~10% 的齐降，而 v1/v2 也同样下降——环境噪声。
本记录的全部结论改用 §3 的同环境 A/B，跨环境那轮不作证据。

## 3 步骤

改动只在结构体定义，**调用点一行未改**：

```cuda
// 改前
struct alignas(16) BF16x8 { __nv_bfloat162 h[4]; };

// 改后
struct alignas(16) BF16x8 {
    union { float4 raw; __nv_bfloat162 h[4]; };
    __device__ __forceinline__ BF16x8() {}
    __device__ __forceinline__ BF16x8(const BF16x8& o) { raw = o.raw; }
    __device__ __forceinline__ BF16x8& operator=(const BF16x8& o) { raw = o.raw; return *this; }
};
```

匿名 union 会删除隐式拷贝构造，所以必须显式给出拷贝语义；给出之后
`BF16x8 v = p[i];` 与 `p[i] = v;` 自动走 `raw` 这条 128 位通路，调用点无须改写。

A/B 协议：
1. 用当前源码构建，`BENCH_ITERS=0 python bench.py` 跑 3 轮 → after
2. `git show HEAD:` 取回旧版 v3/v4，同环境重建，同样 3 轮 → before
3. 还原源码

## 4 原始数据

| 用途 | 文件 |
|---|---|
| 修复后 3 轮 | `fused-norm/project-proof/data/20260829T0304_fused-norm_vec-after_r{1,2,3}.csv` |
| 对照 3 轮 | `fused-norm/project-proof/data/20260829T0304_fused-norm_vec-before_r{1,2,3}.csv` |
| 归并 | `derived_fused-norm_vec-{after,before}_stability.csv` |

`vec-after` 三份的 provenance 首行 `sha` 指向 `f95a0a8`，但该轮源码含未提交改动，
文件内已加 `# note:` 说明。EXP-K05 时代的 `derived_fused-norm_stability.csv` 未被触碰。

## 5 结果

SASS（`cuobjdump -sass`，纯静态）：

| 版本 | LDG.128 | STG.128 | LDG.16 |
|---|---:|---:|---:|
| v3 改前 | 0 | 0 | 4 |
| v3 改后 | **4** | **2** | 0 |
| v4⟨1⟩ 改后 | **3** | **2** | 0 |

带宽（同环境，3 轮 mean±std，GB/s）：

| regime | 版本 | 改前 | 改后 | 变化 |
|---|---|---:|---:|---:|
| prefill（L2 常驻） | v1 *对照* | 1564.0±3.6 | 1565.7±2.9 | +0.1% |
| | v2 *对照* | 2344.4±5.6 | 2347.2±5.1 | +0.1% |
| | **v3** | 2983.8±3.0 | **3619.4±5.2** | **+21.3%** |
| | **v4** | 2586.8±1.8 | **3669.0±2.8** | **+41.8%** |
| hbm | v3 | 921.0±0.1 | 921.3±0.1 | +0.0% |
| | v4 | 920.2±0.2 | 920.8±0.2 | +0.1% |
| decode | v3 | 3.9±0.0 | 3.9±0.0 | +0.0% |
| | v4 | 3.9±0.0 | 3.9±0.1 | −0.8% |

正确性 3 轮全通过，`max_rel_err` 与改前逐位相同（prefill 4.545e-03、hbm 6.452e-03）。

`decode64` 区间三轮 std 达 ±35~39 GB/s（改前 v1/v2 尤甚），**该区间本轮数据不可用**，
不作任何结论。

## 6 分析与结论

**H1 成立。** `alignas(16)` 只约束地址对齐；nvcc 按**成员类型**逐个生成访存，
成员是 4 字节的 `__nv_bfloat162`，于是编出 4 条 32 位 `LDG.E`。
PTX 层可直接看见：`ld.global.v2.u16 ×4`，而非一条 128 位。
同仓 `int8-quantize` v4 用内建 `float4`，`LDG.E.128` 一直是有的——**同一台机器、
同一个工具链下的对照，说明这不是编译器能力问题，是类型选择问题**。

**H2 成立。** L2 常驻区间 v3 +21.3%、v4 +41.8%；HBM 区间 +0.0%/+0.1%（落在 std 内）。
与 EXP-K04 的两区间口径一致：HBM 区间已在 91-92% 峰值，瓶颈是带宽不是指令条数；
L2 区间带宽富余，访存指令条数才成为瓶颈。

**H3 未触发。** v1/v2 不使用 `BF16x8`、代码未改，prefill 仅 +0.1%，增益可归因。

**一处定性反转。** v4 在 prefill 原本**慢于** v3（2586.8 vs 2983.8），修复后**快于** v3
（3669.0 vs 3619.4）。v4 的卖点是「寄存器缓存消掉第二次读」，但在未向量化时每次访存要发
4 条指令，省下的那次读被指令开销淹没。**该优化在本次修复之前一直是净亏的**，
这解释了 EXP-K05 记录的「v4 相对 v3 无收益」。

**EXP-K05 的一处归因需要更正**（数字不变，机制变）：原记录把「v3 向量化在 HBM 区间
零收益」归因为「带宽已饱和所以向量化无收益」。带宽饱和这半句对，但前提不成立——
当时根本没有向量化。正确表述是「未兑现，且即便兑现在该区间也不会有收益」。

## 7 异常、偏差与开放问题

- `rope` v3/v4、`activation` v2/v3、`w8a8` quant.v2 用**同一个 `BF16x8` 定义、
  同一个根因**，本次未改。`activation` v3 是「打包布局」，预期收益可能更大。
- `fused-norm` 现有 6 份 `.ncu-rep`（EXP-K07，RTX 4090 / CUDA 12.8）是**修复前**版本的实测。
  L1/L2/DRAM 的字节账在修复后必然改变，需重采才能对照。采集主机已释放，待下次。
- `decode64` 区间方差过大（±35~39 GB/s），成因未查。
- 跨环境那轮（torch 2.13.0+cu132 基线 vs 2.11.0+cu130）在 decode 区间有 ~10% 齐降，
  v1/v2 同样下降，判为环境噪声；成因未查，不影响本记录结论。

## 8 下游影响

- `fused-norm/README.md` 的版本梯表与 `README.md` 主表中 fused_add_rmsnorm 的
  L2 区间数字需更新（HBM 区间数字不变）。
- `docs/lectures/03_memory_bound_fusion.md`：v3/v4 的代码引用行号已重定位
  （v3 43-58 → 53-68、v4 49-66 → 60-77），`verify_lectures.py` 0 MISMATCH。
- `docs/sass_evidence_ladder.md` §5.3 / §6.1 的「向量化未兑现」条目：
  `fused-norm` 移出，`rope`/`activation`/`w8a8` 保留。
- 简历与 PORTFOLIO 若引用 fused-norm 的 L2 区间数字，需同步。HBM 区间的
  91.3% / 920.3 GB/s 不受影响。
