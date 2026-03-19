import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "project-proof" / "data" / "benchmark_results.csv"
FIG_PATH = ROOT / "project-proof" / "docs" / "figures" / "latency_comparison_log.png"


def load_benchmark_rows():
    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


rows = load_benchmark_rows()
labels = [row["version"] for row in rows]
latency_ms = [float(row["latency_ms"]) for row in rows]
colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"][: len(labels)]

plt.figure(figsize=(8, 4.5))
bars = plt.bar(labels, latency_ms, color=colors)
plt.yscale("log")

for bar, value in zip(bars, latency_ms):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.6f} ms",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.title("CUDA Reduction Latency Comparison (Log Scale)")
plt.ylabel("Latency (ms, log scale)")
plt.xlabel("Version")
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=200)
plt.show()
