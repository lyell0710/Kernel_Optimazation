import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "project-proof" / "data" / "benchmark_results.csv"
FIG_PATH = ROOT / "project-proof" / "docs" / "figures" / "01-benchmark" / "03-speedup-vs-baseline.png"
VERSION_ORDER = ("baseline", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7")


def load_rows():
    with CSV_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def ordered_rows(rows):
    row_by_version = {r["version"]: r for r in rows}
    ordered = [row_by_version[v] for v in VERSION_ORDER if v in row_by_version]
    extra = [r for r in rows if r["version"] not in VERSION_ORDER]
    return ordered + extra


rows = ordered_rows(load_rows())
labels = [r["version"] for r in rows]
latency = [float(r["latency_ms"]) for r in rows]
baseline = latency[0]
speedup = [baseline / v for v in latency]

plt.figure(figsize=(9.6, 4.8))
bars = plt.bar(labels, speedup, color=plt.get_cmap("tab10").colors[: len(labels)])
for bar, value in zip(bars, speedup):
    plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}x", ha="center", va="bottom", fontsize=8)

plt.title("CUDA Softmax Speedup vs Baseline")
plt.xlabel("Version")
plt.ylabel("Speedup (x)")
plt.grid(True, axis="y", linestyle="--", alpha=0.35)
plt.tight_layout()
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(FIG_PATH, dpi=220)
plt.close()

print(f"Saved: {FIG_PATH}")
