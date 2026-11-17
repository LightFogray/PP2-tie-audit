"""诊断：Place Pulse 2.0 的平局是否触发了「近重复停用」机制？

背景
----
平台代码里，平局会调用 updateQScores：把票数少的那张图绑定到另一张、
标记 inactive。若该规则在 PP2 生产运行中生效，平局就不只是「判断」，
还导致了图像的移除。

可检验的痕迹
------------
H1 若停用生效，被停用的图只出现在少数票里 → 票数分布应左偏、
   且大量图票数极少。实际：中位 9 票，p10=2，p90=16（偏紧）
H2 若停用生效，一次平局后该图不再出现 → 图像的平局数与总票数应负相关
H3 若停用生效，被停用的图应与对手极相似 → 平局集中在高相似度对上
H4 若停用生效，活跃池会持续收缩 → 按时间看，平局率应随时段下降

若不支持 H1–H4，则更可能的情况是：该规则未在 PP2 生产运行中生效，
或只影响少数研究。此时平局是纯粹的回答行为，我们的分析不受剪枝污染。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAW = Path(r"D:\mycode\crime-temporal-spatio\data\raw\placepulse2\from_zip")
DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")
OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\diag")
LAB = {"left": 0, "equal": 1, "right": 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA / "pairs.parquet")
    df["_tie"] = df.choice.eq("equal").astype(int)
    print(f"  safer 投票 {len(df):,}  图像 {pd.unique(pd.concat([df.left, df.right])).size:,}")

    # ---------- H1：每图票数分布 ----------
    cnt = pd.concat([df.left, df.right]).value_counts()
    print("\n=== H1 每图票数分布 ===")
    print("  " + "  ".join(f"P{q}={np.percentile(cnt, q):.0f}"
                           for q in (5, 10, 25, 50, 75, 90, 95)))
    print(f"  票数=1 的图 {int((cnt == 1).sum()):,}（{(cnt == 1).mean()*100:.1f}%）")
    print(f"  票数≤2 的图 {int((cnt <= 2).sum()):,}（{(cnt <= 2).mean()*100:.1f}%）")
    print("  若停用生效，应有大量票数为 1–2 的图（被早早移除）")

    # ---------- H2：平局数与总票数 ----------
    part = []
    for side in ("left", "right"):
        part.append(pd.DataFrame({"img": df[side].values,
                                  "tie": df._tie.values}))
    part = pd.concat(part, ignore_index=True)
    g = part.groupby("img").agg(n=("tie", "size"), ties=("tie", "sum"))
    g["tie_rate"] = g.ties / g.n
    print("\n=== H2 每图平局率 vs 总票数 ===")
    print(f"  Spearman(票数, 平局率) = {stats.spearmanr(g.n, g.tie_rate).statistic:+.4f}")
    print("  若停用生效，应为显著负相关（平局多者票数少）")
    # 置换对照：在保持每图票数的前提下随机重排平局标签，
    # 可分离「停用」与「小样本比率方差大」这一机制假象
    rng = np.random.RandomState(0)
    nulls = []
    part_arr = part.values
    for _ in range(20):
        sh = part_arr.copy()
        sh[:, 1] = rng.permutation(sh[:, 1])
        gg = pd.DataFrame(sh, columns=["img", "tie"]).groupby("img").agg(
            n=("tie", "size"), ties=("tie", "sum"))
        nulls.append(stats.spearmanr(gg.n, gg.ties / gg.n).statistic)
    print(f"  置换零分布（20 次）: 均值 {np.mean(nulls):+.4f}  "
          f"5–95% [{np.percentile(nulls,5):+.4f}, {np.percentile(nulls,95):+.4f}]")
    print(f"  观测值 {stats.spearmanr(g.n, g.tie_rate).statistic:+.4f} "
          f"是否落在零分布外: "
          f"{abs(stats.spearmanr(g.n, g.tie_rate).statistic) > np.percentile(np.abs(nulls),95)}")
    for lo, hi in ((2, 4), (5, 8), (9, 12), (13, 20), (21, 999)):
        m = (g.n >= lo) & (g.n <= hi)
        if m.sum() < 200:
            continue
        print(f"    票数 {lo}-{hi}: {int(m.sum()):>6} 图  "
              f"该层平局率 {g.ties[m].sum()/g.n[m].sum():.4f}")

    # ---------- H3：平局对是否更相似 ----------
    z = np.load(DATA / "features_convnext_tiny.npz")
    fidx = {k: i for i, k in enumerate(z["ids"].tolist())}
    F = z["features"].astype(np.float32)
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    ok = df.left.isin(fidx) & df.right.isin(fidx)
    sub = df[ok]
    sim = (Fn[sub.left.map(fidx).values] * Fn[sub.right.map(fidx).values]).sum(1)
    sub = sub.assign(sim=sim)
    print("\n=== H3 平局对的视觉相似度 ===")
    print(f"  平局对   n={int(sub._tie.sum()):>7}  相似度均值 {sub.sim[sub._tie==1].mean():.4f}")
    print(f"  非平局对 n={int((1-sub._tie).sum()):>7}  相似度均值 {sub.sim[sub._tie==0].mean():.4f}")
    d = sub.sim[sub._tie == 1].mean() - sub.sim[sub._tie == 0].mean()
    print(f"  差 {d:+.4f}   若停用生效且平局=近重复，差值应明显为正")
    # AUC：相似度能否区分平局
    from scipy.stats import rankdata
    r = rankdata(sub.sim.values)
    n1 = int(sub._tie.sum()); n0 = len(sub) - n1
    a = (r[sub._tie.values == 1].sum() - n1*(n1+1)/2) / (n1*n0)
    print(f"  相似度预测平局的 AUC = {a:.4f}")

    # ---------- H4：平局率随时间 ----------
    print("\n=== H4 平局率随时间 ===")
    d2 = df.copy()
    d2["t"] = pd.to_datetime(d2.timestamp, errors="coerce")
    d2 = d2.dropna(subset=["t"])
    d2["q"] = d2.t.dt.to_period("Q").astype(str)
    qq = d2.groupby("q")["_tie"].agg(["mean", "size"])
    qq = qq[qq["size"] >= 2000]
    for q, r_ in qq.iterrows():
        print(f"    {q}  n={int(r_['size']):>7}  平局率 {r_['mean']:.4f}")
    print("  若停用生效，池子收缩会让平局率随时间上升（相似对被提前剪掉后剩余仍有…）")

    g.to_csv(args.out / "per_image_votes.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
