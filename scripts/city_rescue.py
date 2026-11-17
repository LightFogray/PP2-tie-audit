"""救城市同质性：三项处理能否让 r=0.48 站住。

三处攻击
--------
O2 内生剪枝：平局导致图像被移除，池子构成是平局的结果
O4 二项噪声：每城约 9k 票，SE≈0.35pp，而城际 SD 仅 0.47pp
O5 只是弱相关：r=0.48 即 23% 方差，不构成「代理」

对应处理
--------
1. 剪枝无关的估计：只用每张图的**第一次投票**（此时该图尚未因平局被移除），
   并控制季度固定效应。这样每张图贡献一个观测，不受其后续生涯影响。
2. 噪声校正：自助与留一城；再用测量误差模型去衰减（disattenuation）：
   r_true = r_obs / sqrt(rel_x · rel_y)，其中 rel 由各城自身的二项方差估计。
3. 增量效度：平局率能否解释视觉同质性解释不了的部分？以及反过来。

若三项都过，城市那条可以保留；否则建议降到附录或不发。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")
OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\city")


def wls_r(x, y, w):
    mx = np.average(x, weights=w)
    my = np.average(y, weights=w)
    cx = np.average((x - mx) ** 2, weights=w)
    cy = np.average((y - my) ** 2, weights=w)
    cxy = np.average((x - mx) * (y - my), weights=w)
    return cxy / np.sqrt(cx * cy), cx, cy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA / "pairs.parquet")
    df["_tie"] = df.choice.eq("equal").astype(int)
    df["t"] = pd.to_datetime(df.timestamp, errors="coerce")
    df = df.dropna(subset=["t"]).copy()
    df["q"] = df.t.dt.to_period("Q").astype(str)
    eco = pd.read_csv(args.out / "city_visual_homogeneity.csv")
    sim_map = dict(zip(eco.city, eco.sim_mean))

    # ---------- 基准：全部投票 ----------
    long = []
    for side in ("left", "right"):
        long.append(pd.DataFrame({"img": df[side].values, "tie": df._tie.values,
                                  "t": df.t.values, "q": df.q.values,
                                  "city": df[f"{side}_city"].values}))
    L = pd.concat(long, ignore_index=True)
    L["sim"] = L.city.map(sim_map)
    base_recs = []
    for c, g in L.dropna(subset=["sim"]).groupby("city"):
        if len(g) < 500:
            continue
        base_recs.append(dict(city=c, n=len(g), tie=g.tie.mean(),
                              sim=sim_map[c]))
    base = pd.DataFrame(base_recs)
    r_base, _, _ = wls_r(base.sim.values, base.tie.values,
                         base.n.values.astype(float))
    print(f"=== 基准（全部投票，{len(base)} 城）===")
    print(f"  平局率 {base.tie.min():.4f}–{base.tie.max():.4f}  SD {base.tie.std():.5f}")
    print(f"  加权 r = {r_base:+.4f}")

    # ---------- 处理 1：每图首次投票 ----------
    Ls = L.sort_values(["img", "t"])
    first = Ls.groupby("img").head(1)
    # 季度固定效应：减季度均值
    first = first.copy()
    first["qmean"] = first.groupby("q")["tie"].transform("mean")
    first["tie_adj"] = first.tie - first.qmean
    recs = []
    for c, g in first.dropna(subset=["sim"]).groupby("city"):
        if len(g) < 200:
            continue
        recs.append(dict(city=c, n=len(g), tie_adj=g.tie_adj.mean(),
                         tie_raw=g.tie.mean(), sim=sim_map[c],
                         qmean=g.qmean.mean()))
    f = pd.DataFrame(recs)
    r_first, _, _ = wls_r(f.sim.values, f.tie_adj.values, f.n.values.astype(float))
    r_first_raw, _, _ = wls_r(f.sim.values, f.tie_raw.values, f.n.values.astype(float))
    print(f"\n=== 处理 1：首次投票（{len(f)} 城，每城 {f.n.median():.0f} 票）===")
    print(f"  r（原值）   {r_first_raw:+.4f}")
    print(f"  r（控季度） {r_first:+.4f}")
    print("  设计：每图只贡献一票，该图此时尚未因平局被移除")

    # ---------- 处理 2：噪声校正 ----------
    print(f"\n=== 处理 2：噪声与不确定性 ===")
    # 各城二项 SE
    p = base.tie.values
    se = np.sqrt(p * (1 - p) / base.n.values)
    print(f"  各城二项 SE 中位 {np.median(se)*100:.3f} pp  "
          f"范围 {se.min()*100:.3f}–{se.max()*100:.3f} pp")
    print(f"  城际 SD {base.tie.std()*100:.3f} pp")
    # 去衰减：真方差 = 观测方差 - 平均噪声方差
    var_obs = base.tie.var(ddof=1)
    var_noise = np.mean(se ** 2)
    rel_y = max(0.0, 1 - var_noise / var_obs)
    print(f"  观测方差 {var_obs:.3e}  噪声方差 {var_noise:.3e}  "
          f"信度比 rel_y = {rel_y:.3f}")
    r_dis = r_base / np.sqrt(max(rel_y, 1e-6))
    print(f"  去衰减后 r_true ≈ {r_dis:+.4f}")

    # 自助
    rng = np.random.RandomState(args.seed)
    bs = []
    for _ in range(args.boot):
        idx = rng.randint(0, len(base), len(base))
        b = base.iloc[idx]
        if b.sim.std() == 0:
            continue
        bs.append(wls_r(b.sim.values, b.tie.values, b.n.values.astype(float))[0])
    bs = np.array(bs)
    print(f"  自助（{args.boot} 次）95% CI [{np.percentile(bs,2.5):+.3f}, "
          f"{np.percentile(bs,97.5):+.3f}]")

    # 留一城
    loo = []
    for c in base.city:
        b = base[base.city != c]
        loo.append(wls_r(b.sim.values, b.tie.values, b.n.values.astype(float))[0])
    loo = np.array(loo)
    print(f"  留一城 r 范围 [{loo.min():+.3f}, {loo.max():+.3f}]  "
          f"均值 {loo.mean():+.3f}")

    # ---------- 处理 3：增量效度 ----------
    print(f"\n=== 处理 3：增量效度 ===")
    dec = args.out / "city_tie_decomposition.csv"
    if dec.exists():
        d = pd.read_csv(dec)[["city", "gamma_mean"]]
        m = base.merge(d, on="city")
        # 平局率 ~ sim —— 残差能否被 gamma 解释
        b1 = np.polyfit(m.sim, m.tie, 1)
        res_tie = m.tie - np.polyval(b1, m.sim)
        r_res = stats.pearsonr(res_tie, m.gamma_mean)
        print(f"  控制 sim 后，平局率残差 与 投票者构成 的相关 "
              f"{r_res.statistic:+.3f}（p={r_res.pvalue:.4f}）")
        b2 = np.polyfit(m.gamma_mean, m.tie, 1)
        res2 = m.tie - np.polyval(b2, m.gamma_mean)
        r_res2 = stats.pearsonr(res2, m.sim)
        print(f"  控制构成后，平局率残差 与 sim 的相关 "
              f"{r_res2.statistic:+.3f}（p={r_res2.pvalue:.4f}）")
        X = np.column_stack([np.ones(len(m)), m.sim, m.gamma_mean])
        bb, *_ = np.linalg.lstsq(X, m.tie.values, rcond=None)
        r2 = 1 - ((m.tie.values - X @ bb) ** 2).sum() / \
            ((m.tie.values - m.tie.mean()) ** 2).sum()
        print(f"  联合模型 R² = {r2:.3f}")

    base.to_csv(args.out / "city_rescue_base.csv", index=False)
    f.to_csv(args.out / "city_rescue_first.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
