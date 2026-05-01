import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "project-proof" / "data" / "benchmark_results.csv"
FIG_PATH = ROOT / "project-proof" / "docs" / "figures" / "latency_comparison.png"
VERSION_ORDER = ("baseline", "v0", "v1", "v2", "v3", "v4", "v5", "v6")


def load_benchmark_rows():
    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def aggregate_rows_by_version(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["version"], []).append(row)

    aggregated = []
    for version, samples in grouped.items():
        latency_values = [float(r["latency_ms"]) for r in samples]
        cpu_values = [float(r["cpu_result"]) for r in samples]
        gpu_values = [float(r["gpu_result"]) for r in samples]
        diff_values = [float(r["diff"]) for r in samples]
        correctness_values = [str(r["correctness_pass"]).lower() == "true" for r in samples]
        aggregated.append(
            {
                "version": version,
                "latency_ms": f"{sum(latency_values) / len(latency_values):.6f}",
                "cpu_result": f"{sum(cpu_values) / len(cpu_values):.6e}",
                "gpu_result": f"{sum(gpu_values) / len(gpu_values):.6e}",
                "diff": f"{sum(diff_values) / len(diff_values):.6e}",
                "correctness_pass": str(all(correctness_values)).lower(),
            }
        )
    return aggregated


def pick_colors(count: int):
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(count)]


rows = aggregate_rows_by_version(load_benchmark_rows())
row_by_version = {row["version"]: row for row in rows}
ordered_rows = [row_by_version[v] for v in VERSION_ORDER if v in row_by_version]
extra_rows = [row for row in rows if row["version"] not in VERSION_ORDER]
plot_rows = ordered_rows + extra_rows

labels = [row["version"] for row in plot_rows]
latency_ms = [float(row["latency_ms"]) for row in plot_rows]
colors = pick_colors(len(labels))

baseline_latency = (
    float(row_by_version["baseline"]["latency_ms"])
    if "baseline" in row_by_version
    else latency_ms[0]
)
speedups = [baseline_latency / value for value in latency_ms]

early_versions = ["baseline", "v0", "v1", "v2"]
late_versions = ["v3", "v4", "v5", "v6"]

early_rows = [row_by_version[v] for v in early_versions if v in row_by_version]
late_rows = [row_by_version[v] for v in late_versions if v in row_by_version]

fig, (ax_early, ax_late) = plt.subplots(1, 2, figsize=(11.5, 4.8), width_ratios=[1.1, 1.6])

# 左图：只做量级参考
early_labels = [r["version"] for r in early_rows]
early_latency = [float(r["latency_ms"]) for r in early_rows]
early_bars = ax_early.bar(early_labels, early_latency, color="#9AA0A6")
ax_early.set_yscale("log")
ax_early.set_title("Early Versions (Scale Reference)")
ax_early.set_ylabel("Latency (ms, log scale)")
ax_early.grid(True, axis="y", linestyle="--", alpha=0.35)
for bar, value in zip(early_bars, early_latency):
    ax_early.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.3f} ms",
        ha="center",
        va="bottom",
        fontsize=8,
    )
ax_early.text(
    0.02,
    0.02,
    "Only for scale context\n(not optimization focus)",
    transform=ax_early.transAxes,
    fontsize=8,
    color="#555555",
)

# 右图：重点比较后段优化版本
late_labels = [r["version"] for r in late_rows]
late_latency = [float(r["latency_ms"]) for r in late_rows]
late_x = np.arange(len(late_labels))
late_bars = ax_late.bar(late_x, late_latency, color=["#4C78A8", "#59A14F", "#F28E2B", "#E15759"])
ax_late.set_xticks(late_x, late_labels)
ax_late.set_title("Optimization Focus: v3 ~ v6")
ax_late.set_ylabel("Latency (ms)")
ax_late.grid(True, axis="y", linestyle="--", alpha=0.35)

if late_latency:
    y_min = min(late_latency)
    y_max = max(late_latency)
    pad = (y_max - y_min) * 0.35 if y_max > y_min else y_max * 0.1
    ax_late.set_ylim(max(0, y_min - pad), y_max + pad)

v3_latency = float(row_by_version["v3"]["latency_ms"]) if "v3" in row_by_version else None
for bar, value, label in zip(late_bars, late_latency, late_labels):
    speedup_vs_v3 = (v3_latency / value) if v3_latency else 1.0
    ax_late.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.6f} ms\n({speedup_vs_v3:.2f}x vs v3)",
        ha="center",
        va="bottom",
        fontsize=8,
    )

fig.suptitle("CUDA Reduction Latency Comparison")
fig.tight_layout()
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_PATH, dpi=220)
plt.close(fig)

for version, value, speedup in zip(labels, latency_ms, speedups):
    print(f"{version}: {value:.6f} ms, {speedup:.2f}x vs baseline")
