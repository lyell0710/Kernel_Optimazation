import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "project-proof" / "data" / "benchmark_results.csv"
FIG_PATH = ROOT / "project-proof" / "docs" / "figures" / "latency_comparison_line.png"
VERSION_ORDER = ("baseline", "v0", "v1", "v2", "v3", "v4", "v5")


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


rows = aggregate_rows_by_version(load_benchmark_rows())
row_by_version = {row["version"]: row for row in rows}
ordered_rows = [row_by_version[v] for v in VERSION_ORDER if v in row_by_version]
extra_rows = [row for row in rows if row["version"] not in VERSION_ORDER]
plot_rows = ordered_rows + extra_rows

labels = [row["version"] for row in plot_rows]
latency_ms = [float(row["latency_ms"]) for row in plot_rows]
baseline_latency = (
    float(row_by_version["baseline"]["latency_ms"])
    if "baseline" in row_by_version
    else latency_ms[0]
)
speedups = [baseline_latency / value for value in latency_ms]
x = np.arange(len(labels))

fig, (ax_all, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4.5), width_ratios=[1.1, 1])

# Left: full trend including baseline
ax_all.plot(x, latency_ms, marker="o", linewidth=2.2, color="#4C78A8")
ax_all.set_xticks(x, labels)
ax_all.set_title("Full Latency Trend")
ax_all.set_ylabel("Latency (ms)")
ax_all.set_xlabel("Version")
ax_all.grid(True, axis="y", linestyle="--", alpha=0.3)

for xi, value, speedup in zip(x, latency_ms, speedups):
    ax_all.annotate(
        f"{value:.6f} ms\n({speedup:.2f}x)",
        (xi, value),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=8,
    )

# Right: zoom in on optimized versions only
zoom_labels = labels[1:]
zoom_latency = latency_ms[1:]
zoom_speedups = speedups[1:]
zoom_x = np.arange(len(zoom_labels))

ax_zoom.plot(zoom_x, zoom_latency, marker="o", linewidth=2.2, color="#E45756")
ax_zoom.set_xticks(zoom_x, zoom_labels)
ax_zoom.set_title("Zoomed Optimized Versions")
ax_zoom.set_ylabel("Latency (ms)")
ax_zoom.set_xlabel("Version")
ax_zoom.grid(True, axis="y", linestyle="--", alpha=0.3)

y_min = min(zoom_latency)
y_max = max(zoom_latency)
padding = (y_max - y_min) * 0.25 if y_max > y_min else y_max * 0.1
ax_zoom.set_ylim(y_min - padding, y_max + padding)

for xi, value, speedup in zip(zoom_x, zoom_latency, zoom_speedups):
    ax_zoom.annotate(
        f"{value:.6f} ms\n({speedup:.2f}x)",
        (xi, value),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=8,
    )

fig.suptitle("CUDA Reduction Latency Comparison", fontsize=13)
fig.tight_layout()
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_PATH, dpi=200)
plt.close()
