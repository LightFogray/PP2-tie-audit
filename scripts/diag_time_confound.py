"""在控制「池子状态」后，投票者平局倾向是否依然成立？

为什么必须查
------------
已确认图像在平局后被停用（末位投票平局率 35.6% vs 其余 9.6%）。
这意味着池子随时间推移越来越「不像对」，平局率有共同的时间趋势。

若投票者效应只是「哪些投票者恰好在早期投票」的代理，那它就不是个体特质。

检验
----
1. 平局率随时间的变化有多大
2. 在**同一时间段内**，投票者之间的平局率差异是否依然存在（分层 ICC）
3. 用投票者的「同期相对平局倾向」（减去该时段的总体平局率）重做跨城迁移，
   若仍为正，则个体效应不是时间趋势的代理
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DATA = r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data\pairs.parquet"


def icc_moment(p, n):
    w = n / n.sum()
    pbar = float(np.sum(w * p))
    between = float(np.sum(w * (p - pbar) ** 2))
    total = pbar * (1 - pbar)
    return between / total if total > 0 else np.nan


def main() -> int:
    df = pd.read_parquet(DATA)
    df["_tie"] = df.choice.eq("equal").astype(int)
    df["t"] = pd.to_datetime(df.timestamp, errors="coerce")
    df = df.dropna(subset=["t"]).copy()
    df["q"] = df.t.dt.to_period("Q").astype(str)
    print(f"  投票 {len(df):,}  季度 {df.q.nunique()}")

    # 1) 时间趋势
    q = df.groupby("q")["_tie"].agg(["mean", "size"])
    q = q[q["size"] >= 3000]
    print(f"\n=== 平局率的季度变化（n≥3000 的 {len(q)} 个季度）===")
    print(f"  范围 {q['mean'].min():.4f} – {q['mean'].max():.4f}  "
          f"SD {q['mean'].std():.4f}")

    # 2) 分层 ICC：在每个季度内算 ICC 再按票数加权
    ics, ws = [], []
    for period, g in df.groupby("q"):
        if len(g) < 3000:
            continue
        t = g.groupby("voter_uniqueid")["_tie"].agg(["mean", "count"])
        t = t[t["count"] >= 3]
        if len(t) < 200:
            continue
        ics.append(icc_moment(t["mean"].values, t["count"].values))
        ws.append(t["count"].sum())
    ics, ws = np.array(ics), np.array(ws)
    print(f"\n=== 季度内 ICC（{len(ics)} 个季度）===")
    print(f"  加权均值 {np.average(ics, weights=ws):.4f}   "
          f"范围 {ics.min():.4f} – {ics.max():.4f}")
    print("  若个体效应真实，季度内 ICC 应仍显著大于 0")

    # 3) 扣除同期均值后的跨城迁移
    df["qmean"] = df.groupby("q")["_tie"].transform("mean")
    df["_adj"] = df["_tie"] - df["qmean"]
    p0 = float(df._tie.mean())
    agg = df.groupby("voter_uniqueid")["_adj"].agg(["sum", "count"])
    rate = (agg["sum"] + 5.0 * 0.0) / (agg["count"] + 5.0)
    tau = rate.to_dict()

    cities = sorted(set(df.left_city) | set(df.right_city))
    rs = []
    for c in cities:
        in_c = (df.left_city == c) | (df.right_city == c)
        inside = df[in_c].groupby("voter_uniqueid")["_adj"].agg(["sum", "count"])
        outside = df[~in_c].groupby("voter_uniqueid")["_adj"].agg(["sum", "count"])
        j = inside.join(outside, how="inner", lsuffix="_in", rsuffix="_out")
        j = j[(j["count_out"] >= 6) & (j["count_in"] >= 1)]
        if len(j) < 50:
            continue
        a = j["sum_out"] / j["count_out"]
        b = j["sum_in"] / j["count_in"]
        w = j["count_in"].values.astype(float)
        ma, mb = np.average(a, weights=w), np.average(b, weights=w)
        ca = np.average((a - ma) ** 2, weights=w)
        cb = np.average((b - mb) ** 2, weights=w)
        cab = np.average((a - ma) * (b - mb), weights=w)
        if ca > 0 and cb > 0:
            rs.append(cab / np.sqrt(ca * cb))
    rs = np.array(rs)
    print(f"\n=== 扣除季度均值后的跨城迁移（{len(rs)} 城）===")
    print(f"  加权相关均值 {rs.mean():+.4f}  中位 {np.median(rs):+.4f}  "
          f"为正 {int((rs > 0).sum())}/{len(rs)}")
    print(f"  Wilcoxon p = {stats.wilcoxon(rs).pvalue:.3e}")
    print("  若仍显著为正，说明个体平局倾向不是时间趋势的代理")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
