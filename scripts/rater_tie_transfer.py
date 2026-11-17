"""判决实验：一个人的平局风格能否迁移到他在别的城市的投票？

为什么这是判决点
----------------
模型类（Davidson + 投票者特定平局阈值）可能已有先例，
但「跨城市人群的稀疏平局风格 + 迁移」没有——Place Pulse 是少数
能让同一个人在多城投票的数据集（86% 的投票者投过 ≥2 个 city）。

若迁移成立，方法贡献可以落在「跨城迁移 + 稀疏可识别性 + 城市评分偏移」，
而不依赖模型类本身是否新。

两个层次
--------
A. 同分布检验：用一个人一半投票估其平局倾向，预测另一半
B. 跨城检验（关键）：只用他在**其他城市**的投票估，预测他在**本城**的投票

对照
----
global   全局平局率
rater    收缩后的个体平局率
rater+img 再叠加图像侧信号（对图的视觉差异范数）

指标
----
二分类 tie/非 tie 的 log loss、Brier、AUC。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")
OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\rater")
GRID_SH = 10.0


def shrunk(rate_num, n, p0, alpha):
    return (rate_num + alpha * p0) / (n + alpha)


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    # AUC
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    auc = float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan
    return ll, brier, auc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-votes", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA / "pairs.parquet")
    df["_tie"] = df.choice.eq("equal").astype(float)
    df["_non"] = 1 - df._tie
    rng = np.random.RandomState(args.seed)
    p0 = float(df._tie.mean())
    print(f"  投票 {len(df):,}  投票者 {df.voter_uniqueid.nunique():,}  全局平局率 {p0:.4f}")

    # ---------- A. 同分布分半 ----------
    print("\n=== A. 同分布分半检验 ===")
    hist_n, hist_t, hold_y, hold_r = [], [], [], []
    per_voter = []
    for v, g in df.groupby("voter_uniqueid", sort=False):
        k = len(g)
        if k < args.min_votes:
            continue
        idx = rng.permutation(k)
        h = k // 2
        gh, go = g.iloc[idx[:h]], g.iloc[idx[h:]]
        per_voter.append((v, len(gh), gh._tie.sum(), go._tie.values))
    print(f"  可用投票者 {len(per_voter):,}  "
          f"每人历史票中位 {np.median([x[1] for x in per_voter]):.0f}")

    for v, n_h, t_h, go in per_voter:
        tau = shrunk(t_h, n_h, p0, args.alpha)
        hold_y.append(go)
        hold_r.append(np.full(len(go), tau))
    y = np.concatenate(hold_y)
    pr = np.concatenate(hold_r)
    for name, p in (("global", np.full(len(y), p0)), ("rater", pr)):
        ll, br, auc = metrics(y, p)
        print(f"  {name:<8} logloss={ll:.5f}  Brier={br:.5f}  AUC={auc:.4f}")

    # ---------- B. 跨城迁移 ----------
    print("\n=== B. 跨城迁移检验（用他在其他城市的投票估）===")
    cities = sorted(set(df.left_city) | set(df.right_city))
    rows = []
    for c in cities:
        in_c = (df.left_city == c) | (df.right_city == c)
        sub_out = df[~in_c]     # 该投票者不涉及 c 的投票
        sub_in = df[in_c]       # 涉及 c 的投票
        if len(sub_in) < 200:
            continue
        # 每个投票者在城内与城外的历史
        outside = sub_out.groupby("voter_uniqueid")["_tie"].agg(["sum", "count"])
        inside = sub_in.groupby("voter_uniqueid")["_tie"].agg(["sum", "count"])
        joined = inside.join(outside, how="inner", lsuffix="_in", rsuffix="_out")
        joined = joined[(joined["count_out"] >= args.min_votes) & (joined["count_in"] >= 1)]
        if len(joined) < 50:
            continue
        tau = shrunk(joined["sum_out"].values, joined["count_out"].values,
                     p0, args.alpha)
        w = joined["count_in"].values
        ybar = joined["sum_in"].values / w          # 每人在城内的平局率
        # 加权相关（每人一行，权重=城内票数）
        def wcorr(a, b, w):
            ma, mb = np.average(a, weights=w), np.average(b, weights=w)
            ca = np.average((a - ma) ** 2, weights=w)
            cb = np.average((b - mb) ** 2, weights=w)
            cab = np.average((a - ma) * (b - mb), weights=w)
            return cab / np.sqrt(ca * cb) if ca > 0 and cb > 0 else np.nan
        r = wcorr(tau, ybar, w)
        rows.append(dict(city=c, n_voters=len(joined), n_votes=int(w.sum()),
                         r_tau_outside_in=r,
                         tie_in=float(np.average(ybar, weights=w))))
        print(f"  {c:<16} 人数 {len(joined):>5}  票 {int(w.sum()):>6}  "
              f"加权相关 {r:+.4f}  城内平局率 {np.average(ybar, weights=w):.4f}")

    t = pd.DataFrame(rows)
    t.to_csv(args.out / "rater_transfer.csv", index=False)
    if len(t):
        print(f"\n  加权相关均值 {t.r_tau_outside_in.mean():+.4f}  "
              f"中位 {t.r_tau_outside_in.median():+.4f}  "
              f"为正的城市 {int((t.r_tau_outside_in > 0).sum())}/{len(t)}")
        pv = stats.wilcoxon(t.r_tau_outside_in.dropna()).pvalue
        print(f"  Wilcoxon（相关>0）p={pv:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
