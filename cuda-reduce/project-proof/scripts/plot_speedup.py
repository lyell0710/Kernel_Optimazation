import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "project-proof" / "data" / "benchmark_results.csv"
FIG_PATH = ROOT / "project-proof" / "docs" / "figures" / "01-benchmark" / "04-speedup-vs-baseline.png"
VERSION_ORDER = ("baseline", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7")


def load_rows():
    with CSV_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def aggregate_by_version(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["version"], []).append(row)

    merged = []
    for version, samples in grouped.items():
        latency_values = [float(r["latency_ms"]) for r in samples]
        merged.append({"version": version, "latency_ms": sum(latency_values) / len(latency_values)})
    return merged


rows = aggregate_by_version(load_rows())
row_by_version = {r["version"]: r for r in rows}
ordered = [row_by_version[v] for v in VERSION_ORDER if v in row_by_version]
extra = [r for r in rows if r["version"] not in VERSION_ORDER]
plot_rows = ordered + extra

labels = [r["version"] for r in plot_rows]
latency = [r["latency_ms"] for r in plot_rows]
baseline_latency = row_by_version["baseline"]["latency_ms"] if "baseline" in row_by_version else latency[0]
speedup = [baseline_latency / v for v in latency]

plt.figure(figsize=(9.2, 4.8))
bars = plt.bar(labels, speedup, color=plt.get_cmap("tab10").colors[: len(labels)])
plt.title("CUDA Reduction Speedup vs Baseline")
plt.xlabel("Version")
plt.ylabel("Speedup (x)")
plt.grid(True, axis="y", linestyle="--", alpha=0.35)

for bar, value in zip(bars, speedup):
    plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}x", ha="center", va="bottom", fontsize=8)

FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=220)
plt.close()
