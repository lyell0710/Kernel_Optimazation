# Figures

图表由 `project-proof/scripts/*.py` 生成。

## 当前图表
### 01-benchmark
- `01-latency.png`：各版本延迟柱状图
- `02-latency-log.png`：各版本延迟对数坐标图
- `03-speedup-vs-baseline.png`：相对 baseline 的加速比
- `04-correctness.png`：正确性摘要（误差 + PASS/FAIL）

### 02-profiling
- `00-ncu-unavailable.png`：无 NCU 权限/数据时的说明图
- `01-ncu-throughput.png`：SM/DRAM 吞吐对比
- `02-ncu-occupancy-inst.png`：活跃 warp 与指令数
- `03-ncu-bound-scatter.png`：compute/memory/latency bound 启发式散点图

## 生成命令
```bash
python project-proof/scripts/plot_latency.py
python project-proof/scripts/plot_latency_log.py
python project-proof/scripts/plot_speedup.py
python project-proof/scripts/plot_correctness.py
bash project-proof/scripts/profile_ncu.sh
RUN_NCU_CSV=1 bash project-proof/scripts/profile_ncu.sh   # 若需 NCU CSV 汇总图
python project-proof/scripts/plot_ncu_summary.py
```
