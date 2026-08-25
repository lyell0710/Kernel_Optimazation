#!/usr/bin/env python
"""README 门面图 3 张(全部从 derived/records 数据读取,禁手改)。

  01_gemm_tc_ladder.png       <- gemm/project-proof/data/derived_gemm4096_stability.csv
  02_fa2_wmma_ladder.png      <- flash-attn/project-proof/data/derived_fa2_proto_stability.csv (S=4096)
  03_reduce_v7_vs_cublas.png  <- records/data/exp_k01_reduce_3rounds.csv

用法: /root/venvs/kernel-opt/bin/python scripts/plot_readme_figures.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_manager.fontManager.addfont("/usr/share/fonts/truetype/arphic/uming.ttc")
plt.rcParams["font.family"] = font_manager.FontProperties(
    fname="/usr/share/fonts/truetype/arphic/uming.ttc").get_name()
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

C_OURS = "#1a6fb8"      # 我方版本
C_BEST = "#0f4c81"      # 我方最优(次强调)
C_BASE = "#c0392b"      # 基线/对照(真库)
C_NEUT = "#999999"      # 中性参照
DPI = 220


def read_csv(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(
            line for line in f if not line.startswith("#"))]
    return rows


def style_ax(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.xaxis.grid(True, color="#e6e6e6", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def hbar(ax, labels, means, stds, colors, unit_fmt, extra=None):
    y = range(len(labels))[::-1]
    ax.barh(list(y), means, xerr=stds, color=colors, height=0.62, zorder=3,
            error_kw=dict(ecolor="#333333", elinewidth=1.0, capsize=3))
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    span = max(means)
    for yi, m, s, i in zip(y, means, stds, range(len(labels))):
        txt = unit_fmt(m, s) + (f"  {extra[i]}" if extra and extra[i] else "")
        ax.text(m + s + span * 0.015, yi, txt, va="center", ha="left",
                fontsize=9, color="#333333")
    ax.set_xlim(0, span * 1.30)


def footnote(fig, text):
    fig.text(0.01, 0.012, text, fontsize=7.5, color="#777777")


# ---- 图 1:GEMM Tensor Core 版本梯 --------------------------------------
rows = read_csv(os.path.join(
    ROOT, "gemm/project-proof/data/derived_gemm4096_stability.csv"))
name_map = {
    "v0": "v0 naive", "v1": "v1 smem tile(CUDA core)", "v2_wmma": "v2 wmma",
    "v3_dbuf": "v3 +cp.async 双缓冲", "v4_bigtile": "v4 +128² 大 tile·8 warp",
    "cublas": "cuBLAS(真库调用)"}
labels = [name_map[r["version"]] for r in rows]
means = [float(r["tflops_mean"]) for r in rows]
stds = [float(r["tflops_std"]) for r in rows]
pcts = [r["pct_of_cublas"] if r["version"] != "cublas" else "" for r in rows]
colors = [C_BEST if r["version"] == "v4_bigtile"
          else C_BASE if r["version"] == "cublas" else C_OURS for r in rows]
fig, ax = plt.subplots(figsize=(8.6, 3.9))
fig.patch.set_facecolor("white")
style_ax(ax)
hbar(ax, labels, means, stds, colors,
     lambda m, s: f"{m:.1f}±{s:.2f}", extra=pcts)
ax.set_xlabel("TFLOPS(fp16 输入 / fp32 累加,4096³)", fontsize=9)
ax.set_title("GEMM 的台阶是指令世代(v1→v2 wmma ×13.8),v4 达真 cuBLAS 85.6%",
             fontsize=11.5, pad=10)
footnote(fig, "source: gemm/project-proof/data/derived_gemm4096_stability.csv"
              " · RTX 4090 · 3 轮 mean±std(误差条)· EXP-K02")
fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.savefig(os.path.join(FIGDIR, "01_gemm_tc_ladder.png"), dpi=DPI,
            facecolor="white")
plt.close(fig)

# ---- 图 2:FA2 wmma 版本梯(S=4096 协议点) -----------------------------
rows = read_csv(os.path.join(
    ROOT, "flash-attn/project-proof/data/derived_fa2_proto_stability.csv"))
rows = [r for r in rows if r["S"] == "4096"]
name_map = {
    "v0_warp_row": "v0 warp-per-row", "v1_smem_tile": "v1 K/V 进 smem",
    "v2_wmma": "v2 wmma(×4.5)", "v3_8warp": "v3 8 warp(+33%)",
    "v4_overlap": "v4 cp.async 重叠(仅 +6.6%)"}
order = ["v0_warp_row", "v1_smem_tile", "v2_wmma", "v3_8warp", "v4_overlap"]
rows.sort(key=lambda r: order.index(r["version"]))
labels = [name_map[r["version"]] for r in rows]
means = [float(r["tflops_mean"]) for r in rows]
stds = [float(r["tflops_std"]) for r in rows]
colors = [C_BEST if r["version"] == "v4_overlap" else C_OURS for r in rows]
fig, ax = plt.subplots(figsize=(8.6, 3.6))
fig.patch.set_facecolor("white")
style_ax(ax)
hbar(ax, labels, means, stds, colors, lambda m, s: f"{m:.1f}±{s:.2f}")
ax.set_xlabel("TFLOPS(B=1 Hq=32 Hkv=8 D=128 causal,S=4096)", fontsize=9)
ax.set_title("FA2 用 wmma:访存全预取后仅 +6.6%——瓶颈在 smem 往返相位链,不在访存",
             fontsize=11.5, pad=10)
footnote(fig, "source: flash-attn/project-proof/data/derived_fa2_proto_stability.csv"
              "(S=4096 行)· RTX 4090 · 3 轮 mean±std · EXP-K03")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(FIGDIR, "02_fa2_wmma_ladder.png"), dpi=DPI,
            facecolor="white")
plt.close(fig)

# ---- 图 3:reduce v7 vs 真 cuBLAS(3 轮) -------------------------------
rows = read_csv(os.path.join(ROOT, "records/data/exp_k01_reduce_3rounds.csv"))
name_map = {"v7": "v7(grid-stride two-pass)",
            "cuBLAS": "cuBLAS(真库调用)", "v4": "v4(中间版本参照)"}
labels = [name_map[r["version"]] for r in rows]
means = [float(r["mean_ms"]) for r in rows]
stds = [float(r["std_ms"]) for r in rows]
colors = {"v7": C_BEST, "cuBLAS": C_BASE, "v4": C_NEUT}
colors = [colors[r["version"]] for r in rows]
fig, ax = plt.subplots(figsize=(8.6, 2.9))
fig.patch.set_facecolor("white")
style_ax(ax)
hbar(ax, labels, means, stds, colors, lambda m, s: f"{m:.5f}±{s:.5f} ms")
ax.set_xlabel("时延 ms(1600 万 float 求和,越低越好)", fontsize=9)
ax.set_title("reduce v7 反超真 cuBLAS 24.5%(4090,3 轮 mean±std)",
             fontsize=11.5, pad=10)
footnote(fig, "source: records/data/exp_k01_reduce_3rounds.csv"
              " · RTX 4090 · N=1<<24 · 3 轮 mean±std · EXP-K01")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(FIGDIR, "03_reduce_v7_vs_cublas.png"), dpi=DPI,
            facecolor="white")
plt.close(fig)

print("figures written to", FIGDIR)
