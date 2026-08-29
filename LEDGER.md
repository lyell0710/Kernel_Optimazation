# LEDGER — 状态与措辞账本(对内)

> **本文件是状态与措辞的唯一权威；README 为对外版，措辞以本表为准。** 数字的权威在各 `project-proof/data/` 与 `records/data/`（单一事实源，只链接不复制）。对外文档 = README.md（门面，面试官视角）+ PORTFOLIO.md（深读），两者受本文件末节「对外禁词」约束。

## 实验台账(EXP 索引,状态唯一权威)

| 编号 | 名称 | slug | 日期 | 状态 | 关键数字（指针） |
|---|---|---|---|---|---|
| [EXP-K01](records/EXP-K01_4090_rebench.md) | 四 kernel 4090 重基准：roofline 迁移（4070 Laptop → 4090） | 4090_rebench | 2026-08-23 | 完成（带 8/24 勘误） | 4090 reduce v7 反超 cuBLAS 24.5%（3 轮）；softmax 对比句作废（对照系自写 kernel，勘误见记录 §5）；gemv v3 快 cuBLAS **37.8%**（3 轮；单轮 84% 不可复现——cuBLAS 侧波动，勘误见 §7 闭环）→ 各 project-proof/data/ |
| [EXP-K04](records/EXP-K04_standard_library_baselines.md) | 标准库基准补齐与两区间重测（CUB / cuDNN 入场） | standard_library_baselines | 2026-08-25 | 完成 | 补 CUB/cuDNN 同算子基准并两区间重测：HBM-bound v7 93.9% 峰值（与 CUB 差 0.7%）、L2 区间 CUB 快 33.3%、softmax vs cuDNN +6.7%/−9.9%、gemv 34.1%；**作废** "reduce 反超 cuBLAS 24.5%" 的对外用法 → records/data/exp_k04_* |
| [EXP-K02](records/EXP-K02_cuda_gemm_tc_ladder.md) | CUDA Tensor Core GEMM 版本梯（v0→v4,vs 真 cuBLAS） | cuda_gemm_tc_ladder | 2026-08-24 | 完成 | Tensor Core GEMM v0→v4:133.1±0.97 TFLOPS = 真 cuBLAS 85.6%（4096³,3 轮）→ gemm/project-proof/data/ |
| [EXP-K03](records/EXP-K03_cuda_fa2_ladder.md) | CUDA FA2 forward 简化版版本梯（v0→v4，量化 wmma 架构税） | cuda_fa2_ladder | 2026-08-24 | 完成 | CUDA FA2 v0→v4:34.8±0.12 TFLOPS = 自家 Triton 28%（跨 harness）,wmma 架构税量化 → flash-attn/project-proof/data/ |
| [EXP-K05](records/EXP-K05_llm_fused_elementwise.md) | LLM 融合逐元素算子三件套：fused_add_rmsnorm / rope / silu_and_mul | llm_fused_elementwise | 2026-08-26 | 完成 | 三个算子 HBM 区间贴到理论峰值 **89.9%–92.0%**，手写 CUDA / Triton / torch.compile 打平；分水岭是融不融合（pytorch_eager 仅 17.5%–55.1%）→ 各 project-proof/data/ |
| [EXP-K06](records/EXP-K06_w8a8_linear.md) | W8A8 linear 完整链路：per-token 量化 + INT8 GEMM/GEMV + 融合反量化 | w8a8_linear | 2026-08-26 | 完成 | prefill **2.161x** bf16 cuBLAS；decode 的 M=1 库路径不可用，自写 dp4a GEMV **1.972x**；多一次 `.contiguous()` 则整链路跌到 **0.734x**（3.6 倍单步差距全来自 stride）；quant 仅占链路 **1.8%** → w8a8/project-proof/data/ |
| [EXP-K07](records/EXP-K07_ncu_counter_closure.md) | 采集主机 NCU 计数器闭环：分管线利用率、GEMM 对照口径核验、fused-norm L2 命题 | ncu_counter_closure | 2026-08-29 | 完成 | 闭合 `ncu_reading_guide` §4 第 1/2/5 条：GEMM v4/cuBLAS 吞吐比 **77.9%** 与 Tensor 管线利用率比 **77.7%** 在 0.2pp 内重合；fused-norm「第二次读不出片」证实且机制精确到 **L1**（命中 83.2%，L2 仅 0.94%）；FA2 v4 `short_scoreboard` **50.13%** vs `long_scoreboard` **0.31%** → 各 profiling/ncu/ |
| [EXP-K08](records/EXP-K08_bf16x8_vectorization_fix.md) | BF16x8 向量化未兑现的定位与修复：从 alignas 到 union | bf16x8_vectorization_fix | 2026-08-29 | 完成 | `alignas(16)` 只保证地址对齐、不强制向量化访存，四算子声称的 16 B 向量化在 SASS 层从未兑现；修复后全部兑现但收益分化：L2 区间 fused-norm v3 **+21.3%** / v4 **+41.8%**、rope v3 **+15.0%** / v4 **+48.6%**，而 activation 与 w8a8 **≈0** → 各 data/*vec-after* |
| [EXP-K09](records/EXP-K09_post_vectorization_sector_ledger.md) | 向量化修复后的扇区账复采：守卫验证与「浪费比」判据 | post_vectorization_sector_ledger | 2026-08-29 | 完成 | L1TEX 请求降幅精确 **4.00×**（=16 B/4 B），DRAM 读仍为 **2.000×S** 算法下界（守卫通过）；修复前 v3/v4 的 16/12×S **比标量版 4×S 还差**，即当时是负优化；「兑现≠收益」获得浪费比判据（§6.3）→ 各 profiling/ncu/ |

## 措辞红线表(对外文本逐条对照)

子项目级细表见 `gemm/README.md` 与 `flash-attn/README.md` 的「红线（措辞）」节。

| 红线 | 状态 | 依据 / 解锁条件 |
|---|---|---|
| reduce「反超 cuBLAS 24.5%」 | **作废**（异算子 asum + L2 区间） | 现行口径见 EXP-K04 §4.1 两区间表 |
| softmax「快 cuBLAS X%」 | 永久禁用（对照系自写） | cuDNN 对照见 EXP-K04 §4.2 |
| softmax「快 cuDNN 6.7%」 | 可用，但必须连非对齐形状慢 9.9% 一起报 | EXP-K04 §4.2（两形状同表） |
| reduce 带宽百分比 | 只在 HBM-bound(1.07 GB)区间可报；L2 常驻区间报带宽即错 | EXP-K04 §4.1 / §6（等效带宽超理论峰值） |
| GEMM「超过/追平 cuBLAS」 | 禁用（现状 85.6%，**该数字对工具链敏感**） | 85.6% = 主力机 CUDA 13.2（EXP-K02 §8）；采集主机 CUDA 12.8 同协议测得 **77.9%**（EXP-K07《NCU 计数器闭环》 §5.2/§6.2）。两者是不同工具链下的两个数，对外引用维持 85.6% 并知悉该敏感性。Triton 版数字不得挪用 |
| FA2「达到 sdpa/Triton 水平」 | 禁用（现状 28%） | v5 mma PTX 路线；EXP-K03 §8 |
| 一切 vs Triton/sdpa 数字 | 跨 harness，推断级，引用必须带此限定 | 同 harness 复测；EXP-K03 §7 |
| softmax 的任何「vs cuBLAS」对比句 | 作废（对照物系自写 kernel，cuBLAS 无 softmax API） | 无解锁；EXP-K01 §5 勘误 |
| gemv 单轮领先幅度 | 作废，现行口径 = 快 **34.1%**（3 轮；可写「34%，轮间 34–38%」）——**必须带 L2 常驻限定，见下一行** | EXP-K04 §4.3 取代 EXP-K01 §7 的 37.8%；差异为 cuBLAS 侧轮间波动 |
| gemv「快 34.1%」的适用区间 | **只在 L2 常驻区间可报**；HBM 区间两者持平（1.4%） | 工作集 4096×2048 fp32 = 33.55 MB < 72 MB L2；bench 等效带宽 2633 GB/s = DRAM 峰值 2.6 倍，关掉 cache flush 后 `dram__bytes_read` = **0**。与 reduce 同型判据（超峰值即落 L2）。**真实大形状复核**：8192×4096 = 134.2 MB 下 3 轮仅 **+2.4%**，轮间 −2.9%~+7.6% **跨越零点**，std 与领先同量级 → 该 regime 下应报「持平」而非「快 2.4%」。EXP-K09 §5.8/§6.7/§5.15/§6.15 |
| softmax「快 cuDNN 6.7%」 | **降级：只在 L2 常驻区间成立**，引用必须与 HBM 区间持平结论同表出现 | HBM 区间（8192×4096，工作集 268.4 MB）实测**领先完全消失**：bench 3 轮 **0.0%**，ncu 两种 cache 模式下 v4 还慢 2.6%–3.4%。两者都贴在 912 GB/s（峰值 90.5%）。禁止写成「手写 softmax 比 cuDNN 快」这种无限定陈述。EXP-K09 §5.10/§6.9 |
| softmax 的两个限定不可混淆 | 形状敏感性 ≠ regime 敏感性 | 「非对齐 1024×1500 反被 cuDNN 快 9.9%」是**形状**敏感；「HBM 区间领先归零」是**工作集**敏感。两条独立，须分别陈述。EXP-K09 §6.9 |
| `int8-quantize` v4 的 5.57 μs | **L2 常驻区间数字**，需标注 | 1024² 工作集 5.24 MB < 72 MB L2。**注意它未触发「超峰值」判据**（941.3 GB/s < 1008），却实测 DRAM 读为 0——判据漏报的实例，见下一行。EXP-K09 §5.9 |
| **L2 常驻区间的判定方法**（取代原先单一的「超峰值」说法） | 三层，强度递增 | ①**必要**：工作集 < L2 容量（4090 = 72 MB）；②**充分**：等效带宽 > DRAM 物理峰值（满足即落 L2，**不满足不能排除**）；③**决定性**：`ncu --cache-control all` 与 `none` 的 `dram__bytes_read.sum` 对比。①命中而②未命中时**必须上③**——`int8-quantize` 即此情形。EXP-K09 §6.8 |
| `w8a8` prefill **2.161×** | **已过 regime 检验，无需区间限定** | 对外形状 O=12288 处 int8 权重 50.3 MB 在 L2 内、bf16 100.7 MB 在 HBM（两臂不同层级）；补测 O=32768（134.2 / 268.4 MB，两边都超 L2）得 **2.267×**，不低反高。机理：`step_int8gemm_col` 跨 L2 边界仅 3.180→3.202×，优势来自算力（int8 525 TOPS vs bf16 165 TFLOPS）而非 cache。EXP-K09 §5.12/§6.12 |
| ~~reduce v7 跨工具链行为不一致~~ | **已闭环，归因被推翻**（EXP-K09 §5.17/§6.19） | 元凶不是 two-pass 结构，而是 **v6/v7 在计时区内调 `cudaGetDeviceProperties`**（v1–v5 没有），该调用在 driver 570 上单次达毫秒级。判决性检验是核 v6——实测 v6 = 1.716 ms **同慢**。修正计时口径后本机 v7 = **0.024545 ms** 与主力机 0.024707 ms 差 **0.7%**，跨工具链一致。**主力机数据从未测错。** |
| **EXP-K04「L2 区间 CUB 快 33.3%」** | **须修正为 12.1%（主力机口径）** | 该数以 v7 为对照，而 v7 侧计时含分配器开销、CUB 侧不含 —— 口径不对称。修正后 L2 区间 CUB 领先 **12.1%（主力机）/ 13.5%（本机）**，**33.3 个点里约 20 个来自分配器**；HBM 区间由「差 0.7%」变为 **−0.0%（分辨率内贴平）**。讲义 §8.3.4 早已预判此事，现已实测闭合。**由主力机侧统一落地 README/EXP-K04。** |
| reduce「≈1193× 端到端」 | 仍需硬件+工具链定语，**但定语可换** | 4090/CUDA 12.8 口径已测 = **2709×**（3 轮，baseline 151.478 ms ÷ v5 0.05592 ms）。**不可摘定语**：该值跨环境 1196×（Laptop）→2709×（本机）→4909×（主力机），跨度 4 倍；分子是 `<<<1,1>>>` 单线程，比值绝大部分来自硬件差距而非代码；且「自写最优」在三环境里分别是 v5/v5/v7。EXP-K09 §5.13/§6.13 |
|「smem 是 GEMM/FA2 剩余差距的瓶颈」 | **可用**（4090 计数器实测） | EXP-K07《NCU 计数器闭环》 §5.4/§6.5：FA2 v4 `short_scoreboard` 50.13% vs `long_scoreboard` 0.31%；gemm v4 smem 冲突波前占 77.1%（放大 4.37×），cuBLAS 2.4%（1.02×） |
|「swizzle 能消除该瓶颈」 | **推断，不可当实测说** | 本轮只证明瓶颈在 smem，未验证 swizzle 是解法；解锁条件 = 实现 v5（mma PTX + ldmatrix + swizzle）并同协议复测 |
| 「向量化兑现 ⇒ 有收益」 | **禁用该推论** | 判据单向:浪费比(L1TEX/DRAM)≈1 → **必然无** sector 层面收益;>1 → **不保证**有收益。反例:`w8a8` quant_v2 浪费比 2.00× 仍 ≈0 收益(量化只占链路 1.8%,属 Amdahl 上限)。三者缺一不可:`LDG.E.128>0` 只证兑现,浪费比证有无可回收量,端到端占比证值不值。EXP-K09 §6.3 |

## 勘误 / 审计留痕(横幅集中处;原文与史料见 records/、docs/archive/、LAB_JOURNAL)

- **softmax vs-cuBLAS 红线级勘误（2026-08-24）**：`softmax/src/softmax_cublas.cu` 系自写 warp 原语 kernel，并非 cuBLAS 调用（cuBLAS 无 softmax API）。"softmax 快 cuBLAS 26%/34%" 全链作废，含 L2 命中率 2.4 倍、online softmax 推断等归因叙事；简历/面试禁用。gemv/reduce 的 cuBLAS 对照经调用点验真有效。详见 EXP-K01 §5；作废段落原文 = `docs/archive/2026-08-24_portfolio_laptop_era_sections.md`。
- **gemv 单轮 84% 勘误（2026-08-24 晚闭环）**：v3 自身完全复现，但 cuBLAS 对照单轮 0.02353 → 3 轮 0.017432±0.000169(−26%)——领先幅度一半以上系对照侧单轮波动。该轮口径 = 快 37.8%（3 轮），其后由 EXP-K04 §4.3 同协议复测更新为 **34.1%**（轮间 34–38%，cuBLAS 侧波动）。教训：对照物也要 3 轮。EXP-K01 §7 闭环。
- **reduce Laptop 旧代自相矛盾（2026-08-24 审计）**：旧 results(v7=1.665ms)与旧 stability(0.273ms)矛盾，"v6/v7 回退 → 4090 反转"叙事不可确证、不作主张；旧稿移 `docs/archive/2026-08-24_portfolio_laptop_era_sections.md`，其中数字禁止对外引用。**唯一授权例外**：端到端口径「347.6ms→0.291ms，~1193×，4070 Laptop」经 Resume/Final_Resume/DO_NOT_SEND.md 2026-08-24 处置记录核准用于简历，引用必须带 Laptop 定语。
- **raw driver=13.3 误填勘误**：gemm/flash-attn raw 的 driver 字段误填 13.3（实为 610.57.04；13.3 是 cudaDriverGetVersion 报的 driver-API 版本）。raw 不可变：以两处 `data/manifest.txt` 勘误 + 修 bench 源码 provenance 生成，不重跑。
- **stability 覆盖史**：EXP-K01 首版仅 cuda-reduce 有 3 轮 stability；softmax/gemv/int8-quantize 于 2026-08-24 晚以 `scripts/stability_rebench.sh` 补齐（9 份 UTC raw + 3 份 3rounds 聚合入 records/data/），台账自此全绿。
- **两处对外数字勘误（2026-08-29，采集主机核对）**：
  ① `activation` 融合一级 **1.675x → 1.678x**。指针文件 `derived_activation_stability.csv` 的 `hbm,v1`
  三口径一致给 1.678（`speedup_vs_v0_mean`=1.67833、延迟比 1.11674/0.665349=1.6784、带宽比 907.733/540.867=1.6783）。
  源头是 `records/EXP-K05` 的笔误，已扩散 10 处，本轮统一订正。结论方向不变（与 5/3=1.667 的吻合度由 0.48% 变 0.66%，同量级）。
  ② FA2 v3→v4 增量 **+6.6% → +7.1%**。v3 32.5 → v4 34.8，正确口径 (34.8−32.5)/32.5 = 7.08%；
  6.6% 是 2.3/34.8，**分母误用终值**。已扩散 20 处（含 `PORTFOLIO`、两处源码注释、绘图脚本），本轮统一订正。
  同句的 v2→v3 **+33%** 经核为 33.2%，分母用法正确，不动。
  两条均不改变任何结论方向（7.1% 依然是很小的增量，"排除访存假设"的论证不受影响）。
  **连带待办已销账**：`figures/02_fa2_wmma_ladder.png` 已在主力机（`/root/venvs/kernel-opt`，
  matplotlib 3.11.1 + uming 字体）经 `scripts/plot_readme_figures.py` 重绘，标题与 v4 标签均为 +7.1%。
  重绘后 `01`/`03` 两图 sha256 逐字节不变、只有 `02` 变化——这正是把重绘留到主力机的理由：
  同环境重绘不会用无意义的二进制差异淹没真正的那一处变更。

- **对外化改版**：PORTFOLIO 顶部勘误横幅与正文内嵌勘误注记、README 台账/红线表，已全部收入本文件；README/PORTFOLIO 正文只保留降级后的对外措辞。

## 待办 / backlog(当前全部非阻塞)

- 2026-08-26 EXP-K05《LLM 融合逐元素算子三件套》新增待办 —— **①② 已闭环，仅剩 ③**：
  ~~①fused-norm「第二次读被缓存接住」待实测~~ → **已实测**（EXP-K07 §6.3）：DRAM 读恒为 2.000×S 算法下界；且机制层级由「L2」**更正为 L1**（L1 命中 83.2%，L2 命中仅 0.94%）。向量化修复后复采见 EXP-K09 §5.1。
  ~~②rope v2 在 HBM 区间比 v1 慢 1.1% 未剖~~ → **已完整闭合**：nsys 侧解释了 decode 为何快（248→124 次 launch）；计数器补上 HBM 为何慢（EXP-K09 §6.17）——Σ 指令 **+12.0%** 而 Σ DRAM 读几乎不变（−1.1%）。净效应取决于 launch 开销占比：decode 档 kernel 仅约 2 μs、launch 占大头故大赚；HBM 档单 kernel 约 214 μs、launch 可忽略，只剩指令代价故变慢。
  ~~④activation 兑现向量化却零收益、成因未剖（EXP-K08 §7）~~ → **已剖**（EXP-K09 §6.16）：指令降 39.8% 而 sector 数 12,582,912 **一个未省**、DRAM 402.71→402.73 MB —— EXP-K08 列的两个候选（指令密度低 / L2 近饱和）**都不对**，真因是访存本就完美合并（浪费比 1.00×），向量化只换指令不换事务。
  ~~⑤decode64 方差过大（±35~39 GB/s）成因未查（EXP-K08 §7）~~ → **已查**（EXP-K09 §6.18）：非自写的 `pytorch_eager`/`torch_compile` CV 反而最高（25.4%/20.8%）；NCU 实测该档 kernel 仅 2.23–2.46 μs 而 bench 报 10.82 μs —— **75–80% 时间在 launch**，该 regime 本质是 launch-bound，测的是 launch 开销分布。降方差的正解是 CUDA Graph。
  **③三个算子均未做 autotune**（仅 rope 单独扫过 BLOCK×num_warps，最优与所用配置差 0.6%）—— 仍开着，不阻塞。
  - **①已销账**（EXP-K07《NCU 计数器闭环》 §6.3）：DRAM 读扇区恒为 2.000 倍算法下界、L1 命中率 83.2%、L2 读命中率 0.94%——接住第二次读的是 **L1 不是 L2**，机制层级已更正。
  - **③相关**：向量化本身此前从未兑现，根因与修复见 EXP-K08《BF16x8 向量化未兑现的定位与修复》；`rope`/`activation`/`w8a8` 同一根因已于 `bc18547` 一并修复（EXP-K08）。
- 本仓补测 backlog 已清零（EXP-K01 §7 于 2026-08-24 晚闭环）。
- 可选技能票：gemm/fa2 v5 = mma PTX + ldmatrix + smem swizzle（EXP-K02 §7、EXP-K03 §8；不阻塞）。
- ~~待 NCU 计数器权限：「swizzle / smem 往返」类推断转实测~~ —— **已完成**（EXP-K07《NCU 计数器闭环》）。在一台计数器可用的 4090 虚机上补齐七个算子共 51 份报告；`docs/ncu_reading_guide.md` §4 五条推断闭合四条（第 1、2、5 条实测闭合，第 4 条由 nsys 解决，第 3 条判为伪需求——融合版不存在，NCU 也证不了）。
- 待同 harness 复测：一切 vs Triton/sdpa 数字（现为跨 harness 推断级）。
- ~~开放问题：cuBLAS gemv 对照单轮慢 35% 未剖~~ —— **降级为「未复现，两候选已排除」**（EXP-K09 §5.11/§6.10）：从 P8 冷态起连跑 5 次，cuBLAS 全落在 15.60–16.66 μs，无一接近当时的 23.53 μs；**第 1 次（冷启动）反而最快**，且 GPU 升频快于 250 ms 采样间隔——冷启动与时钟态均不成立。该现象非环境无关的系统性效应，若要续查须回原环境（CUDA 13.2/610）。护栏（对照物也跑 3 轮）已在内部约定第 2 条。
- ~~开放问题：gemv「34.1%」在计数器环境未被覆盖~~ —— **已闭环**（EXP-K09 §6.7）：并非「钉错 size」（bench 只有一个形状），而是 **cache regime 不同**——34.1% 是 L2 常驻区间的数字，HBM 区间两者持平（1.4%）。红线表已补区间限定。
- ~~待重采~~ **已完成**（EXP-K09《向量化修复后的扇区账复采》）：`fused-norm` 六份及 rope/activation/w8a8 共 23 份已于 `6f320f2` 重采。前后对照见 EXP-K09 §5.1——L1TEX 降幅精确 4.00×，DRAM 读仍为 2.000×S 算法下界。
- ~~待修~~ **已完成**（`bc18547`）：`rope` v3/v4、`activation` v2/v3、`w8a8` quant.v2 与 dequant 六处向量化载体已全部替换为 union 定义，SASS 判据全部转正。

## 内部约定(工作流,对齐 /root/standards CORE 七条铁律)

1. bench 只写 UTC 前缀新文件到各 `project-proof/data/`（`BENCH_OUT` 控制，首行 provenance），永不覆盖已有文件；profiler 环境的时延数字永不进 benchmark 表。
2. 数字进对外文档前 ≥3 轮 mean±std，落 stability/derived 文件；**对照物同样跑 3 轮**；「cublas」只准指真实库调用，自写参照叫 handwritten_*。
3. 每个实验一份 `records/` 八节记录（EXP-KNN），并同步本文件台账。
4. 对外措辞先过上方红线表；凡「vs X」声明先验 X 的调用点。
5. 收尾跑 `bash /root/standards/check.sh` 六项自检，0 FAIL 才 commit。
6. `figures/` 全部由 `scripts/plot_readme_figures.py` 生成，禁手改；对外图脚注不带日期（日期溯源走 raw 文件名与 manifest）。
7. **对外禁词**（README/PORTFOLIO 逐个 grep 自查，一个不留）：任何日期与时间戳（2026-xx、08-2x、UTC 时间戳）、勘误、审计、红线、解锁、台账、停投、待用户、待 GPU 空闲、待建远端、backlog、sibling、Codex、HANDOFF、check.sh、CORE、STANDARDS、DO_NOT_SEND、铁律、终端级证据（对外改写为「单轮」）。records/ 与 docs/archive/ 是史料，不受此限。

## NCU 报告代际

- 2026-08-28 起,`{softmax,gemv,int8-quantize,cuda-reduce}/project-proof/profiling/ncu/`
  下 33 份报告由 **RTX 4090 / CUDA 12.8 / ncu 2025.1.1** 采集,取代原 RTX 4070 Laptop
  (36 SM / 2026-05-23 / ncu 2022.4.1) 的同名报告。
- 4070 代报告未复制副本,权威归档在 git ref `13fdaa3`;
  取回:`git show 13fdaa3:<路径> > <目标>`。
- `cuda-reduce/profiling/ncu/` 下 5 份仍为 4070 代(新脚本不写该路径)。
- 逐份出处以 `artifacts/ncu_for_mac/manifest.csv` 的 gpu/sm_count/ncu_ver/created 列为准。
