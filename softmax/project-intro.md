# Softmax Kernel 项目介绍

## 项目定位
`cuda-softmax` 是一个“手写 CUDA kernel + 版本化迭代 + 基准验证”的学习型工程。  
目标不是一次性写出最优实现，而是像 `cuda-reduce` 一样，把优化过程拆成可解释、可复现、可回归的多个版本。

## 核心任务
对输入矩阵按行做 softmax：

- 数值稳定写法：`exp(x - row_max) / sum(exp(x - row_max))`
- 每一行独立并行
- 同时关注正确性和性能

## 工程结构
- `src/softmax_baseline.cu`：正确性基线（单线程串行一行）
- `src/softmax_v0.cu ~ src/softmax_v4.cu`：逐版优化实现
- `src/main.cu`：统一 benchmark 入口（正确性 + 时延 + CSV）
- `project-proof/data/benchmark_results.csv`：指标沉淀
- `project-proof/docs/why-each-version-changed.md`：每版优化动机与收益预期

## 版本化方法论
- 每个版本只改一个主要优化点，降低定位难度
- 先保证正确，再优化速度
- 每次改动后都跑同一套 benchmark，保证横向可比

## 如何使用
```bash
cmake -S . -B build
cmake --build build -j
./build/softmax_bench
```

运行后会打印每版 `PASS/FAIL`、平均时延、最大误差，并更新 CSV。
