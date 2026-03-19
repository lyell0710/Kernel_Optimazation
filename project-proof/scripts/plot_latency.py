import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "project-proof" / "data" / "benchmark_results.csv"
FIG_PATH = ROOT / "project-proof" / "docs" / "figures" / "latency_comparison.png"


def load_benchmark_rows():
    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


rows = load_benchmark_rows()
labels = [row["version"] for row in rows]
latency_ms = [float(row["latency_ms"]) for row in rows]
colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"][: len(labels)]

baseline_latency = latency_ms[0]
speedups = [baseline_latency / value for value in latency_ms]

plt.figure(figsize=(8, 4.5))
bars = plt.bar(labels, latency_ms, color=colors)

for bar, value, speedup in zip(bars, latency_ms, speedups):
    label = f"{value:.6f} ms\n({speedup:.2f}x)"
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        label,
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.title("CUDA Reduction Latency Comparison")
plt.ylabel("Latency (ms)")
plt.xlabel("Version")
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=200)
plt.show()

for version, value, speedup in zip(labels, latency_ms, speedups):
    print(f"{version}: {value:.6f} ms, {speedup:.2f}x vs baseline")
