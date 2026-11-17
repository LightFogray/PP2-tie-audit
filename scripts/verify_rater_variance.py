"""复现数据探针：哪些投票者个体差异在稀疏数据下可估？

背景
----
若每个投票者只有 7 次投票，「个体偏好」不可估，但「个体平局倾向」可能可估。
两者对应完全不同的方法方向，必须先自己确认。

检验
----
1. 平局倾向的 ICC 与分半信度（按投票者）
2. 非平局时左右偏好的 ICC
3. 全局位置偏差
4. 同城 vs 跨城的平局率
5. 控制同城占比后，平局 ICC 是否仍在
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DATA = r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data\pairs.parquet"


def icc_moment(p, n):
    """二项结果的矩估计 ICC：组间方差 / 总方差。"""
    w = n / n.sum()
    pbar = float(np.sum(w * p))
    between = float(np.sum(w * (p - pbar) ** 2))
    total = pbar * (1 - pbar)
    return between / total if total > 0 else np.nan


def split_half(df, voter_col, outcome, min_votes=6, reps=20, seed=0):
    """把每个投票者的投票随机对半分，算两半的比率相关（Fisher z 平均）。"""
    rng = np.random.RandomState(seed)
    zs = []
    for _ in range(reps):
        a, b = [], []
        for _, g in df.groupby(voter_col, sort=False):
            k = len(g)
            if k < min_votes:
                continue
            idx = rng.permutation(k)
            h = k // 2
            a.append(g.iloc[idx[:h]][outcome].mean())
            b.append(g.iloc[idx[h:2 * h]][outcome].mean())
        a, b = np.array(a), np.array(b)
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if -0.999 < r < 0.999:
            zs.append(np.arctanh(r))
    r = float(np.tanh(np.mean(zs))) if zs else np.nan
    # Spearman-Brown 校正到全长
    return r, (2 * r / (1 + r) if r == r else np.nan)


def main() -> int:
    df = pd.read_parquet(DATA)
    n_vote = len(df)
    print(f"  投票 {n_vote:,}  投票者 {df.voter_uniqueid.nunique():,}")
    print(f"  平局率 {df.choice.eq('equal').mean()*100:.2f}%")
    nt = df[df.choice.ne("equal")]
    print(f"  非平局中 right {nt.choice.eq('right').mean()*100:.2f}%  "
          f"left {nt.choice.eq('left').mean()*100:.2f}%  "
          f"偏差 {(nt.choice.eq('right').mean()-nt.choice.eq('left').mean())*100:+.2f} pp")

    df["_tie"] = df.choice.eq("equal").astype(float)
    df["_right"] = df.choice.eq("right").astype(float)

    print("\n=== 1) 平局倾向（个体差异）===")
    g = df.groupby("voter_uniqueid")["_tie"].agg(["mean", "count"])
    print(f"  投票者 {len(g):,}  每人中位 {g['count'].median():.0f} 次")
    print(f"  个体平局率分布: P5={g['mean'].quantile(.05):.3f} "
          f"P50={g['mean'].median():.3f} P95={g['mean'].quantile(.95):.3f}")
    for lo, hi in ((6, 7), (8, 10), (11, 15), (16, 25), (26, 999)):
        k = (g["count"] >= lo) & (g["count"] <= hi)
        print(f"    n∈[{lo},{hi}]: {k.sum():,} 人  ICC={icc_moment(g.loc[k,'mean'].values, g.loc[k,'count'].values):.3f}")
    rh, rf = split_half(df, "voter_uniqueid", "_tie")
    print(f"  分半信度 r={rh:.3f}  Spearman-Brown={rf:.3f}")

    print("\n=== 2) 左右偏好（个体差异）===")
    d2 = df[df.choice.ne("equal")]
    g2 = d2.groupby("voter_uniqueid")["_right"].agg(["mean", "count"])
    g2 = g2[g2["count"] >= 6]
    print(f"  可用投票者 {len(g2):,}")
    print(f"  right 率分布: P5={g2['mean'].quantile(.05):.3f} "
          f"P50={g2['mean'].median():.3f} P95={g2['mean'].quantile(.95):.3f}")
    print(f"  ICC={icc_moment(g2['mean'].values, g2['count'].values):.3f}")
    rh2, rf2 = split_half(d2, "voter_uniqueid", "_right")
    print(f"  分半信度 r={rh2:.3f}  Spearman-Brown={rf2:.3f}")

    print("\n=== 3) 同城 vs 跨城平局率 ===")
    for name, sub in (("同城", df[df.same_city]), ("跨城", df[~df.same_city])):
        print(f"  {name}: n={len(sub):,}  平局率 {sub._tie.mean()*100:.2f}%")

    print("\n=== 4) 控制同城占比后的平局 ICC ===")
    share = df.groupby("voter_uniqueid")["same_city"].mean()
    g3 = g.join(share.rename("same_share"))
    g3 = g3[g3["count"] >= 6]
    # 按同城占比分层，层内算 ICC 后加权
    g3["bin"] = pd.qcut(g3["same_share"], 5, labels=False, duplicates="drop")
    tot_w, ics = 0.0, []
    for b, sub in g3.groupby("bin"):
        if len(sub) < 100:
            continue
        ics.append((len(sub), icc_moment(sub["mean"].values, sub["count"].values)))
        tot_w += len(sub)
    print("  分层 ICC: " + "  ".join(f"{n}:{v:.3f}" for n, v in ics))
    print(f"  加权平均 {sum(n*v for n,v in ics)/tot_w:.3f}")
    print(f"  全体（≥6 次）ICC {icc_moment(g3['mean'].values, g3['count'].values):.3f}")

    print("\n=== 5) 投票者是否跨城投票 ===")
    cities = df.groupby("voter_uniqueid")["left_city"].nunique()
    multi = (cities > 1).mean()
    print(f"  投过 ≥2 个 city 字段的投票者 {multi*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
