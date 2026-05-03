# Project Proof

这个目录用于沉淀 softmax 项目的可复现实验产物。

## 目录说明
- `data/benchmark_results.csv`：基准结果数据
- `scripts/*.py`：从 CSV 生成图表的脚本
- `scripts/profile_ncu.sh`：关键 kernel 的 Nsight Compute 采集脚本
- `docs/why-each-version-changed.md`：版本迭代动机说明
- `docs/figures/01-benchmark/*.png`：图表输出
- `profiling/ncu/*.csv`：NCU 导出结果

## 一键流程（手动执行）
```bash
./build/softmax_bench
python project-proof/scripts/plot_latency.py
python project-proof/scripts/plot_latency_log.py
python project-proof/scripts/plot_speedup.py
python project-proof/scripts/plot_correctness.py
bash project-proof/scripts/profile_ncu.sh
```
