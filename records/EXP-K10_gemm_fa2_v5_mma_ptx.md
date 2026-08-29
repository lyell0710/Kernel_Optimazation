# EXP-K10 · gemm/fa2 v5 = mma PTX + ldmatrix + smem padding

> **一句话结论**：gemm 与 fa2 的 v5 都用 `mma.sync.m16n8k16` PTX + `ldmatrix` 重写 v4 微内核，**正确性全 PASS、性能与 v4 持平**（gemm 135.1±2.1 TFLOPS = cuBLAS 88.1% vs v4 85.6%；fa2 35.0 vs v4 34.8 TFLOPS）。**诚实负结果**：ldmatrix/padding 消掉 smem 读 bank conflict 后性能几乎不变，说明 EXP-K07 报告的「smem 冲突波前占 77.1%（gemm）」「short_scoreboard 50.13%（fa2）」**都不是 bank conflict 可解的瓶颈**——「swizzle 能消除该瓶颈」这一推断在 gemm 和 fa2 两侧都被证伪（端到端口径；计数器口径需采集机复验）。

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-30 |
| 环境 | 2×RTX 4090，CUDA 13.2（主力机），sm_89，driver 610.57.04 |
| 状态 | **gemm + fa2 均完成** |
| 关联 | EXP-K02/K03；EXP-K07；LEDGER 红线「swizzle 能消除该瓶颈」「FA2 达到 sdpa/Triton 水平」 |

## 1. 目的与假设

v4 用 wmma API，固定布局不暴露 smem 地址计算 → 无法 swizzle。v5 用 mma PTX + ldmatrix 重写微内核，换取 smem 布局控制权。tile 尺寸完全对齐 v4（128×128、8 warp 2×4、BK=32），**只换微内核**（控制变量）。

跑前锁定判据（沿用 LEDGER）：v5 的 bank conflict 与 short_scoreboard 较 v4 显著下降，且端到端有正收益——二者缺一，「swizzle 能消除该瓶颈」仍只能是推断。

## 2. 环境与配置

- `gemm/src/gemm_v5.cu`：BM=128/BN=128/BK=32，256 线程 8 warp，双缓冲 cp.async（继承 v4）。
- 微内核：`ldmatrix.x4`（A）、`ldmatrix.x4.trans`（B）、`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`。
- smem 行宽 padding：A 32→40、B 128→136（各 +8 half = +16B，消 ldmatrix 8 行读取的 bank 冲突）。

## 3. 步骤

1. 写 v5 微内核，接入 main.cu/CMakeLists/gemm_common.h。
2. 编译通过。
3. 正确性调试：`dump_ldmatrix.cu` / `mma_test.cu` 两个最小实验钉死 fragment 布局。
4. 3 轮正式 bench + derived stability。

## 4. 原始数据

- `project-proof/data/20260829T193704_gemm4096_v5_r{1,2,3}.csv`（3 轮 raw，首行 provenance；sha=unknown 系 GIT_SHA 环境变量未设，已修但本次未重跑）。
- `project-proof/data/derived_gemm4096_v5_stability.csv`（derived，3 轮 mean±std）。

## 5. 结果

| 版本 | latency_mean (ms) | latency_std | TFLOPS_mean | TFLOPS_std | pct of cuBLAS |
|---|---|---|---|---|---|
| v4_bigtile | 1.03523 | 0.0041 | 132.8 | 0.49 | 85.6% |
| **v5_mmaPTX** | **1.01747** | **0.0153** | **135.1** | **2.08** | **88.1%** |
| cublas | 0.89673 | 0.0009 | 153.3 | 0.15 | 100.0% |

正确性：v5 maxrel 7.58e-04，与 v0–v4 完全一致，PASS。

### 5.2 fa2（S=4096 协议点，3 轮 mean±std）

| 版本 | latency_mean (ms) | latency_std | TFLOPS |
|---|---|---|---|
| v4_overlap | 3.9485 | 0.0115 | 34.8 |
| **v5_mmaPTX** | **3.9283** | **0.0115** | **35.0** |

正确性：v5 err 4.88e-04，与 v0–v4 完全一致，PASS（全 shape 族）。

## 6. 分析与结论

**① 布局正确性闭环。** 两个最小实验（`dump_ldmatrix` 验证 ldmatrix 输出 = mma fragment 期望；`mma_test` 用单位阵 A 直接钉死 D fragment 布局）把调试从「猜」变成「实测」。最终写回段修正一处：mma.m16n8k16 的 D fragment 是 `c0=D[g][2t] c1=D[g][2t+1] c2=D[g+8][2t] c3=D[g+8][2t+1]`——即 c0/c1 同一 M 行相邻 N 列、c2/c3 是 M+8 行。此前按「c1=D[g+8][2t]」写回，导致偶数列对、奇数列错。

**② 诚实负结果：padding 消 bank conflict 后性能几乎不变（+1.7%，且 std 从 0.49 涨到 2.08）。** EXP-K07 报告 gemm v4 smem 冲突波前占 77.1%（放大 4.37×），据此推断「swizzle 能消除该瓶颈」。v5 用 padding 消掉 ldmatrix 的 bank conflict 后，性能只 +1.7%，且轮间 std 变大（r2 异常 137.5 TFLOPS 疑似热时钟），**不足以支撑「显著下降」的判据**。结论：**gemm 的 smem bank conflict 不是瓶颈**——冲突波前占比高，但 smem 带宽有富余、冲突被其他 stall 掩盖，消掉它不转化为性能。剩余 12% 差距（88.1% → 100%）在别处（多级流水 kStages、tile 形状、cp.async 调度），非 swizzle 可解。

**③ 一个关键澄清（相位作用域，与讲义 01 的 phase 结论互证）。** ldmatrix 的 8 行 × 8 half 读取，每行 16B 落在 4 个 bank；行宽 32 half 时行 r 起始 bank = 16r mod 32，r=0 与 r=2 撞同 bank——这正是 EXP-K07 报告冲突的来源。padding 到 40 half 后行 r 起始 bank = 20r mod 32，8 行互不重叠。这是「bank conflict 作用域是 phase」在 ldmatrix 场景的直接应用。

**④ v5 相对 v4 的提升若报，须带「+1.7%、std 2.08、未达显著」的完整限定，不得报「v5 快 X%」。** 这 +1.7% 与轮间噪声同量级（v4 std 0.49、v5 std 2.08），按红线「相对变化 <5% 或与跨轮波动同量级时写无可区分改善」应写「持平」。

**⑤ fa2 v5 结论同 gemm：ldmatrix 消 bank conflict 后性能持平（35.0 vs 34.8，+0.6% 噪声内）。** fa2 的 short_scoreboard 50.13% 不是 padding/ldmatrix 可解的——端到端口径与 gemm 一致。要坐实「是否仍 bank conflict」需采集机 NCU 复验。

**⑥ fa2 的 K 加载是本次最有价值的布局教训。** K 存储是 `[S][D]`（row-major），而 mma 的 B 操作数要 K^T=[D][S] 的 col-major。两种 ldmatrix 的语义：
- `ldmatrix.trans` 读 row-major `[K][N]` → 输出 col-major `[K][N]` fragment（矩阵语义不变，只换存储序）——用于 gemm 的 B 与 fa2 的 V。
- `ldmatrix`（non-trans）读 row-major `[M][K]` → 输出 row-major A fragment，其 `a0=A[g][2t]`、`a2=A[g][2t+8]` 恰好等于 K^T col-major B 的 `b0=K^T[2t][g]=K[g][2t]`、`b1=K^T[2t+8][g]=K[g][2t+8]`——**用于 fa2 的 K（K^T 的转置关系）**。初版误用 `ldmatrix.trans` 导致 err ~5e-2~7e-1（每行 98/128 列错、幅度 O(0.1)），改 non-trans 后 err 落回 4.88e-04。

## 7. 异常、偏差与开放问题

- **机器 GPU 硬挂起一次**（调试 dump_ldmatrix 时，极小 grid + ldmatrix 读非法 smem 地址触发驱动级死锁，nvidia-smi/ls 均不返回，约 15 分钟后超时自愈）。教训：**ldmatrix 读到非法 smem 地址会整机挂起，不是普通 CUDA error**——与本仓「smem 边界 bug 不报错只错位」是同族，但后果严重一个量级。
- r2 的 137.5 TFLOPS 是异常值（疑似热时钟/boost），导致 v5 std 2.08 远大于 v4 的 0.49。若进对外数字，按「3 轮 mean±std」报，不得摘最好轮。
- raw provenance 的 `sha=unknown`：GIT_SHA 环境变量未设（main.cu 读 `getenv("GIT_SHA")`）。已登记，本次不重跑（raw 不可变）。
- 采集机已停机，v5 的 NCU 计数器（short_scoreboard/bank_conflicts）**未采**——「bank conflict 已消」是布局推导 + 性能旁证，不是计数器实测。

## 8. 下游影响

- **「swizzle 能消除该瓶颈」推断在 gemm 与 fa2 两侧都被证伪（端到端口径）**：ldmatrix/padding 已消 bank conflict，两仓性能均持平。LEDGER 红线表该条改写——bank conflict 波前占比高 ≠ 性能瓶颈（smem 带宽有富余，冲突被其他 stall 掩盖）。剩余差距在别处（gemm：多级流水/tile 形状/cp.async；fa2：相位链/barrier），非 swizzle 可解。
- **计数器口径待采集机复验**：端到端持平已证伪「swizzle 是解法」，但「bank conflict 是否真被 ldmatrix/padding 消掉」需 NCU 的 `bank_conflicts`/`short_scoreboard` 实测。采集机已停机，指标攒着（LEDGER 采集清单不变）。
- gemm/fa2 v5 代码（`gemm_v5.cu`、`fa2_v5.cu`）正确性 PASS、已接入 bench，保留。对外数字：gemm「cuBLAS 88.1%（CUDA 13.2）」、fa2 仍是「Triton 28%」（v5 持平 v4，未解锁「达到 sdpa/Triton 水平」）。
- 调试临时文件（`debug_v5.cu`、`dump_ldmatrix.cu`、`mma_test.cu`、`debug_fa2_v5.cu`）不入库，删除。
