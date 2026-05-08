# Project Proof

这个目录用于沉淀 softmax 项目的可复现实验产物。

## 目录说明
- `data/benchmark_results.csv`：基准结果数据
- `scripts/*.py`：从 CSV 生成图表的脚本
- `scripts/profile_ncu.sh`：按版本生成 NCU `.ncu-rep`（可选 `RUN_NCU_CSV=1` 导出 CSV 供画图）
- `docs/why-each-version-changed.md`：版本迭代动机说明
- `docs/figures/01-benchmark/*.png`：图表输出
- `profiling/ncu/*.ncu-rep`：NCU 报告（主）；`*.csv`：可选 metrics（`RUN_NCU_CSV=1`）

## 一键流程（手动执行）
```bash
./build/softmax_bench
python project-proof/scripts/plot_latency.py
python project-proof/scripts/plot_latency_log.py
python project-proof/scripts/plot_speedup.py
python project-proof/scripts/plot_correctness.py
bash project-proof/scripts/profile_ncu.sh
RUN_NCU_CSV=1 bash project-proof/scripts/profile_ncu.sh   # 仅当需要 plot_ncu_summary 时
python project-proof/scripts/plot_ncu_summary.py
```