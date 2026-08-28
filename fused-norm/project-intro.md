# fused_add_rmsnorm 项目介绍

## 为什么会有这个项目

写自研推理引擎时，我用 nsys 看了一遍 Qwen3-0.6B 的前向时间线，发现一件反直觉的事： GEMM 之外的空隙里密密麻麻挤满了小 kernel。放大看，一次 RMSNorm 在 PyTorch 里被展开成六个 kernel——add、pow、mean、rsqrt、mul、mul，每一个都把整个隐藏状态从显存读进来、算完再写回去。

而 RMSNorm 每个元素只做两次乘加。

这就是这个子项目的起点：**一个算力上完全微不足道的算子，凭什么占掉可观的时间？** 答案是它根本不由算力决定，由搬运字节数决定；而"一个 kernel 只做一件事"这个实现约束，让它搬了远超必要的字节。

## 这个算子是什么

pre-norm Transformer 的每一层里，这个组合出现两次（attention 后一次、MLP 后一次）：

```
residual = residual + x          # 残差流,下一层要用
out      = rmsnorm(residual) * w # 本层分支的归一化输入
```

两个输出都要。它是 LLM 前向里调用次数第二多的访存型算子，仅次于逐元素激活。

## 版本化方法论

五个版本，**每个版本只改一件事**，否则整条梯子讲不出因果：

| 版本 | 只改的那一件事 | 想验证的假设 |
|---|---|---|
| v0 |（基线）两个 kernel，复刻 PyTorch 的执行方式 | 与 eager 同速，证明基线不是稻草人 |
| v1 | 融合成单 kernel | 省一次全局读，加速接近字节账之比 |
| v2 | smem 树形归约 → warp shuffle 两级归约 | 访存主导下，片上开销的优化只是坡不是台阶 |
| v3 | 标量访存 → 16 B 向量化 | 标量 bf16 只覆盖半个事务，应有大收益 |
| v4 | 寄存器缓存，消掉第二遍重读 | 字节账 5→4，应有 +25% |

跑之前把每条预测写进 kernel 的文件头注释，跑完不改。结果是 v1、v2 的预测大致成立， **v3 与 v4 双双被推翻**——而推翻它们的论证，恰好是这个子项目最有价值的部分（见 [README](README.md) 的「关键发现」与深度讲义 [docs/lectures/03](../docs/lectures/03_memory_bound_fusion.md) 的第 3.3 节）。

## 工程结构

- `include/fused_norm.h`— 共用声明 + warp/block 两级归约原语
- `src/fused_norm_v0.cu ~ v4.cu`— 五级版本梯，每级文件头写清"改了什么、字节账、预测"
- `src/binding.cpp`— torch 绑定，让手写 kernel 与 PyTorch/Triton 进同一个 harness
- `bench.py`— 七条臂、四个工作区间的统一 benchmark
- `project-proof/data/`— 每轮独立落盘的原始 CSV 与归并表

## 与其他子项目的关系

- 归约原语与 [cuda-reduce](../cuda-reduce/) 同源，但结论相反：同一手法在 compute-bound 的 reduce 上是台阶，在这里只是坡。
- 与 [rope](../rope/)、[activation](../activation/) 共用一套 harness 与方法论， 三者合起来构成"访存主导算子"这一类的完整样本（EXP-K05《LLM 融合逐元素算子三件套》）。
- 接进自研推理引擎的端到端结果见 llm-engine 的 llm-engine#EXP-D23《融合逐元素算子接入》。
