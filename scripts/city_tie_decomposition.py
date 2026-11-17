"""城市层面分析：一个城市的平局率是城市属性还是投票者构成？

为什么城市分析刊会问
--------------------
各城平局率从 0.1155（台北）到 0.1360（波特兰）。若按城市比较平局率，
需要先知道这个差异来自图像内容还是「谁给这个城市投了票」。

做法
----
1. 用投票者在**其他城市**的投票估计其平局倾向 τ_r（收缩、并中心化）
2. 对每个城市：
   - 观测平局率
   - 构成预测平局率 = 该城投票的 τ_r 均值（加总体基准）
   - 两者之差 = 城市效应（内容成分）
3. 方差分解：跨城变异中有多少由构成解释
4. 重加权标准化：把每城投票按 τ_r 分布重加权到全局分布，
   得到标准化平局率，比较城际排名是否改变

若构成解释了大部分变异、且标准化后排名大幅变动，
则「按平局率比较城市」是无效的——这是对城市分析读者的直接告诫。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")
OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\city")
LAB = {"left": 0, "equal": 1, "right": 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA / "pairs.parquet")
    df["_tie"] = df.choice.eq("equal").astype(int)

    # 一个城市的所有投票 = 任一张图来自该城
    long = []
    for side in ("left", "right"):
        tmp = df[[f"{side}_city", "voter_uniqueid", "_tie"]].copy()
        tmp.columns = ["city", "voter", "_tie"]
        long.append(tmp)
    long = pd.concat(long, ignore_index=True)
    # 同城比较会在两侧各出现一次，按侧计数即可
    print(f"  长表投票 {len(long):,}（每侧一份）")

    p0 = float(df._tie.mean())
    logit_p0 = np.log(p0 / (1 - p0))
    agg = df.groupby("voter_uniqueid")["_tie"].agg(["sum", "count"])
    rate = (agg["sum"] + args.alpha * p0) / (agg["count"] + args.alpha)
    rr = np.clip(rate, 1e-4, 1 - 1e-4)
    # 按 γ 自身的加权均值中心化（logit 的均值 ≠ 均值的 logit）
    lg_all = np.log(rr / (1 - rr))
    lg_center = float(np.average(lg_all.values,
                                 weights=agg["count"].values.astype(float)))
    gamma = (lg_all - lg_center).to_dict()

    long["gamma"] = long["voter"].map(gamma).fillna(0.0)

    rows = []
    for c, g in long.groupby("city"):
        if len(g) < 500:
            continue
        rows.append(dict(
            city=c, n=len(g),
            tie_obs=float(g._tie.mean()),
            gamma_mean=float(g.gamma.mean()),
            gamma_sd=float(g.gamma.std()),
        ))
    t = pd.DataFrame(rows).sort_values("tie_obs")

    print("\n=== 各城平局率与投票者构成 ===")
    print(f"  {'city':<18}{'票数':>8}{'观测平局':>10}{'γ均值':>9}{'γ sd':>8}")
    for _, r in t.iterrows():
        print(f"  {r.city:<18}{int(r.n):>8}{r.tie_obs:>10.4f}"
              f"{r.gamma_mean:>+9.3f}{r.gamma_sd:>8.3f}")

    # 方差分解
    r_pear = stats.pearsonr(t.tie_obs, t.gamma_mean)
    rho = stats.spearmanr(t.tie_obs, t.gamma_mean)
    slope = np.polyfit(t.gamma_mean, t.tie_obs, 1)
    resid = t.tie_obs - np.polyval(slope, t.gamma_mean)
    ss_tot = ((t.tie_obs - t.tie_obs.mean()) ** 2).sum()
    ss_res = (resid ** 2).sum()
    print(f"\n=== 方差分解（{len(t)} 城）===")
    print(f"  观测平局率范围 {t.tie_obs.min():.4f} – {t.tie_obs.max():.4f}"
          f"（跨度 {(t.tie_obs.max()-t.tie_obs.min())*100:.2f} pp）")
    print(f"  与 γ 均值的 Pearson r = {r_pear.statistic:+.3f}  p={r_pear.pvalue:.4f}")
    print(f"  Spearman ρ = {rho.statistic:+.3f}  p={rho.pvalue:.4f}")
    print(f"  构成解释的方差 R² = {1 - ss_res/ss_tot:.3f}")
    print(f"  残差 SD = {resid.std():.5f}（跨城总 SD {t.tie_obs.std():.5f}）")

    # 重加权标准化
    qs = np.linspace(0, 1, args.bins + 1)
    long["bin"] = pd.qcut(long.gamma, args.bins, labels=False, duplicates="drop")
    w_global = long.bin.value_counts(normalize=True).sort_index()

    std = []
    for c, g in long.groupby("city"):
        if len(g) < 500:
            continue
        w_city = g.bin.value_counts(normalize=True).sort_index()
        w_city = w_city.reindex(w_global.index).fillna(1e-6)
        w = (w_global / w_city).reindex(g.bin.values).values
        w = w / w.sum()
        std.append(dict(city=c, tie_std=float(np.average(g._tie.values, weights=w))))
    s = pd.DataFrame(std)
    m = t.merge(s, on="city")
    m["rank_obs"] = m.tie_obs.rank()
    m["rank_std"] = m.tie_std.rank()
    m["drank"] = (m.rank_obs - m.rank_std).abs()

    print("\n=== 重加权到共同投票者构成后 ===")
    print(f"  观测 vs 标准化 相关 {stats.pearsonr(m.tie_obs, m.tie_std).statistic:+.3f}")
    print(f"  排名平均变动 {m.drank.mean():.1f} 位，最大 {m.drank.max():.0f} 位（共 {len(m)} 城）")
    print(f"  观测跨度 {(m.tie_obs.max()-m.tie_obs.min())*100:.2f} pp  ->  "
          f"标准化跨度 {(m.tie_std.max()-m.tie_std.min())*100:.2f} pp")
    print("\n  变动最大的 5 城：")
    for _, r in m.reindex(m.drank.sort_values(ascending=False).index).head(5).iterrows():
        print(f"    {r.city:<18} {r.tie_obs:.4f} -> {r.tie_std:.4f}  "
              f"排名 {int(r.rank_obs)} -> {int(r.rank_std)}")

    m.to_csv(args.out / "city_tie_decomposition.csv", index=False)
    print(f"\n已写入 {args.out/'city_tie_decomposition.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
