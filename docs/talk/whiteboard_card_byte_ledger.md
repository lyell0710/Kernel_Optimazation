# 白板推导卡 · 访存型算子的字节账


1. **记账**：写下算法下界——无论怎么实现都省不掉的读与写。 fused_add_rmsnorm：读 x + 读 res + 写 res + 写 out = 4 次 = 8 B/元素； rope 配对后 = 4 B/元素；silu_and_mul = 6 B/输出元素。
2. **定上限**：某级优化把每元素字节从 B1 降到 B2，加速上限就是 B1/B2，一分不多。 silu 融合 5→3 预测 1.667x，实测 **1.680x**（EXP-K05《LLM 融合逐元素算子三件套》，`activation/project-proof/data/derived_activation_vec-after_stability.csv`）。
3. **量位置**：有效带宽 = N·B/t；除以峰值（4090 = 1008 GB/s）即"离最优多远"。 **超过 100% 不是错，是数据落在 L2 的指示器**(4090 L2 = 72 MB)。
4. **记在哪一层**：字节账要在 **HBM 层面**记，不是指令层面。反例：fused-norm v3→v4 按指令账该 +25%，实测 0%——因为 v2 的 917 GB/s x 10/8 = 1146 > 1008 峰值，证明那次"重读"从未出片；**接住它的是 L1，不是 L2**——同一形状下 v4 的 L1 命中率 33.19%、L2 读命中率仅 0.20%，DRAM 读停在算法下界（residual + x 各一遍，实测 2.001×S）；带宽账见 `fused-norm/project-proof/data/derived_fused-norm_vec-after_stability.csv`，命中率与扇区账见 EXP-K09《向量化修复后的扇区账复采》§5.1。
5. **面试点**：贴墙之后语言不重要。HBM 区间手写 CUDA / Triton / torch.compile 两两差最大 3.3%（87-92% 峰值），而未融合的 eager 落后 1.7-5.2x。 **分水岭是融不融合，不是用什么写**；手写只在 L2 区间（相对 torch.compile 3.25-7.41x）与 decode 的 launch 区间（相对 Triton 2.49-4.86x）仍有优势——而推理引擎恰好常驻这两处。（本条数字取自三个算子 `project-proof/data/derived_*_vec-after_stability.csv`，EXP-K05《LLM 融合逐元素算子三件套》）
