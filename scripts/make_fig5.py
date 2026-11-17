"""图 5：证据深度的空间分布。
(a) 各城低深度图像占比 vs 视觉冗余
(b) 城内证据深度的 Moran's I 分布 vs 随机期望
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\city")
FIG = Path(r"D:\mycode\crime-temporal-spatio\paper_ceus\figures")
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73",
         red="#D55E00", purple="#CC79A7", grey="#7F7F7F", black="#222222")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.2,
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

d = pd.read_csv(OUT / "evidence_depth_spatial.csv")
v = pd.read_csv(OUT / "city_visual_homogeneity.csv")[["city", "sim_mean"]]
m = d.merge(v, on="city")
r = stats.pearsonr(m.low_share, m.sim_mean)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9),
                         gridspec_kw=dict(width_ratios=[1.15, 1]))

ax = axes[0]
ax.scatter(m.sim_mean, m.low_share, s=16, color=C["blue"], alpha=0.75,
           edgecolor="white", linewidth=0.4)
k, b = np.polyfit(m.sim_mean, m.low_share, 1)
xs = np.linspace(m.sim_mean.min(), m.sim_mean.max(), 50)
ax.plot(xs, k * xs + b, color=C["red"], lw=1.1, ls="--")
ax.set_xlabel("Visual redundancy of the city's image pool")
ax.set_ylabel("Share of images with <= 3 comparisons")
ax.text(0.03, 0.95, f"r = {r.statistic:+.2f}, p = {r.pvalue:.1e}",
        transform=ax.transAxes, va="top", fontsize=7.5)
ax.text(-0.17, 1.04, "(a)", transform=ax.transAxes, fontsize=9,
        fontweight="bold", va="bottom")

ax = axes[1]
ax.hist(d.moran_depth.dropna(), bins=18, color=C["grey"], alpha=0.85,
        edgecolor="white", linewidth=0.5)
ax.axvline(0.0, color=C["red"], lw=1.2, ls="--")
ax.set_xlabel("Moran's I of evidence depth, within city")
ax.set_ylabel("Cities")
ax.text(-0.17, 1.04, "(b)", transform=ax.transAxes, fontsize=9,
        fontweight="bold", va="bottom")
ax.text(0.97, 0.95, f"mean {d.moran_depth.mean():+.3f}\n"
        f"{int((d.p_depth<0.05).sum())}/{len(d)} at p < 0.05",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5)

fig.tight_layout()
fig.savefig(FIG / "fig5_evidence_spatial.pdf")
fig.savefig(FIG / "fig5_evidence_spatial.png")
print(f"  r={r.statistic:+.3f} p={r.pvalue:.2e}  Moran mean {d.moran_depth.mean():+.4f}")
print(f"  已保存 fig5_evidence_spatial.{{pdf,png}}")
