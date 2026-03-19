import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "project-proof" / "data" / "benchmark_results.csv"
FIG_PATH = ROOT / "project-proof" / "docs" / "figures" / "latency_comparison_line.png"


def load_benchmark_rows():
    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


rows = load_benchmark_rows()
labels = [row["version"] for row in rows]
latency_ms = [float(row["latency_ms"]) for row in rows]
baseline_latency = latency_ms[0]
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
fig.savefig(FIG_PATH, dpi=200)
plt.show()
