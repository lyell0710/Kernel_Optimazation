import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "project-proof" / "data" / "benchmark_results.csv"
FIG_PATH = ROOT / "project-proof" / "docs" / "figures" / "correctness_check.png"


def load_benchmark_rows():
    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


rows = load_benchmark_rows()
cpu_value = float(rows[0]["cpu_result"])
labels = ["CPU"] + [f"{row['version']} GPU" for row in rows]
values = [cpu_value] + [float(row["gpu_result"]) for row in rows]
colors = ["#72B7B2", "#4C78A8", "#F58518", "#54A24B", "#E45756"][: len(labels)]

plt.figure(figsize=(8.5, 4.5))
bars = plt.bar(labels, values, color=colors)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.5e}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.title("CUDA Reduction Correctness Check")
plt.ylabel("Result Value")
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=200)
plt.show()
