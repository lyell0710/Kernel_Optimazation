# INT8 Per-channel Quantize 项目介绍

项目目标是把 per-channel symmetric INT8 量化 kernel 做成可复现的版本优化实验：

- 同一输入分布与 scale 策略
- 每版保持 correctness 校验
- 指标统一输出到 CSV 和图表

适合在简历中承接“INT8 per-channel quantize”相关描述，并展示从基础实现到访存/并行优化的过程。
