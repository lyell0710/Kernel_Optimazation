# GEMV 项目介绍

这个项目聚焦 `y = A * x` 的 CUDA kernel 优化，目标是用可复现的版本演进记录：

- 每版仅改变一个核心优化点
- 每版都做 correctness + latency 验证
- 用 CSV 和图表沉淀结果

主要用于面试场景中展示从 baseline 到高性能实现的工程化过程。
