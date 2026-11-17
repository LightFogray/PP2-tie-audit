"""生态学检验：城市平局率是否反映街景的视觉同质性？

为什么这可能救回 CEUS 定位
--------------------------
单条投票层面投票者占优（AUC 0.81 vs 0.53），城市构成又只解释 11%，
那么城际平局率差异的其余部分来自哪里？

一个城市层面的机制：街景越同质，任意两图越可能看起来差不多 → 平局越多。
这条在单票层面信号很弱（AUC 0.53），但聚合到上万票的城市均值后可以显现。

检验
----
对每个城市，抽样其图像，计算
  - 视觉同质性 = 图像特征两两余弦相似度的均值
  - 视觉分散度 = 特征的第一主成分方差占比
再与该城平局率做相关，控制图像数与城市规模。

若相关显著，则「城市平局率」是一个可解释的城市形态指标——这是城市分析的落点。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")
OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\city")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-img", type=int, default=400)
    ap.add_argument("--pairs", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA / "pairs.parquet")
    df["_tie"] = df.choice.eq("equal").astype(int)
    z = np.load(DATA / "features_convnext_tiny.npz")
    fidx = {k: i for i, k in enumerate(z["ids"].tolist())}
    F = z["features"].astype(np.float32)
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)

    # 图像 -> 城市
    city_of = {}
    for side in ("left", "right"):
        for i, c in zip(df[f"{side}"].values, df[f"{side}_city"].values):
            city_of[i] = c

    rng = np.random.RandomState(args.seed)
    rows = []
    for c in sorted(set(city_of.values())):
        ids = [i for i, cc in city_of.items() if cc == c and i in fidx]
        if len(ids) < 150:
            continue
        ids = list(rng.choice(ids, min(len(ids), args.max_img), replace=False))
        X = Fn[[fidx[i] for i in ids]]

        # 两两余弦相似度（抽样）
        n = len(ids)
        k = min(args.pairs, n * (n - 1) // 2)
        a = rng.randint(0, n, k)
        b = rng.randint(0, n, k)
        m = a != b
        sim = (X[a[m]] * X[b[m]]).sum(1)

        # 主成分方差占比
        Xc = X - X.mean(0, keepdims=True)
        s = np.linalg.svd(Xc, compute_uv=False)
        ev = (s ** 2) / (s ** 2).sum()

        in_c = (df.left_city == c) | (df.right_city == c)
        rows.append(dict(
            city=c, n_img=len(ids), n_votes=int(in_c.sum()),
            tie_rate=float(df._tie[in_c].mean()),
            sim_mean=float(sim.mean()), sim_sd=float(sim.std()),
            pc1=float(ev[0]), pc_eff=float(1 / (ev[:20] ** 2).sum()),
        ))

    t = pd.DataFrame(rows)
    t.to_csv(args.out / "city_visual_homogeneity.csv", index=False)
    print(f"  城市 {len(t)}")

    print("\n=== 城际相关 ===")
    for col in ("sim_mean", "sim_sd", "pc1", "pc_eff"):
        r = stats.pearsonr(t.tie_rate, t[col])
        rho = stats.spearmanr(t.tie_rate, t[col])
        # 控制图像数
        def partial(x, y, zz):
            def resid(a, b):
                return a - np.polyval(np.polyfit(b, a, 1), b)
            return np.corrcoef(resid(x, zz), resid(y, zz))[0, 1]
        pr = partial(t.tie_rate.values, t[col].values,
                     np.log(t.n_img.values.astype(float)))
        print(f"  {col:<10} Pearson {r.statistic:+.3f} (p={r.pvalue:.4f})  "
              f"Spearman {rho.statistic:+.3f}  偏相关(控图像数) {pr:+.3f}")

    print("\n=== 平局率最高/最低各 5 城（含同质性）===")
    s = t.sort_values("tie_rate")
    for _, r in pd.concat([s.head(5), s.tail(5)]).iterrows():
        print(f"  {r.city:<18} 平局 {r.tie_rate:.4f}  相似度 {r.sim_mean:.4f}  "
              f"PC1 {r.pc1:.3f}")

    # 与投票者构成一起做多元回归
    print("\n=== 多元回归：平局率 ~ 视觉同质性 + 投票者构成 + 图像数 ===")
    dec = Path(args.out) / "city_tie_decomposition.csv"
    if dec.exists():
        d = pd.read_csv(dec)[["city", "gamma_mean"]]
        t2 = t.merge(d, on="city")
        X = np.column_stack([np.ones(len(t2)), t2.sim_mean, t2.gamma_mean,
                             np.log(t2.n_img.astype(float))])
        y = t2.tie_rate.values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        ss = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"  R² = {ss:.3f}   (n={len(t2)})")
        print(f"  视觉同质性 β = {beta[1]:+.5f}"
              f"  投票者构成 β = {beta[2]:+.5f}"
              f"  图像数 β = {beta[3]:+.6f}")
        # 单独 R²
        for name, cols in (("仅同质性", [1]), ("仅构成", [2]), ("两者", [1, 2])):
            Xs = np.column_stack([np.ones(len(t2))] + [X[:, i] for i in cols])
            bs, *_ = np.linalg.lstsq(Xs, y, rcond=None)
            r2 = 1 - ((y - Xs @ bs) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            print(f"    {name}: R² = {r2:.3f}")
        t2.to_csv(args.out / "city_eco.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
