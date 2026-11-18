"""平局倾向 ICC 的正式估计：扣除二项零假设 + 投票者自助区间。

估计量
------
对投票者 i（票数 n_i、平局比例 p_i），加权 w_i = n_i / Σn：
    between = Σ w_i (p_i - p̄)²,   total = p̄(1-p̄)
    R = between / total                       （朴素比值）
    b = Σ w_i / n_i                           （同质概率下的期望比值）
    ICC = (R - b) / (1 - b)
b 的推导：若所有 p_i = p，则 E[Σ w_i (p_i-p̄)²] = p(1-p) Σ w_i/n_i，
故 R 的期望恰为 Σ w_i/n_i。

假设
----
1) 每位投票者的平局次数 ~ Binomial(n_i, p_i)，相互独立；
2) 投票者间 p_i 的分布由单一 ICC 概括（交换性）；
3) 仅纳入 n_i >= 6 的投票者，使 p_i 有足够分辨率；
4) 左右偏好用非平局比较中的"选右"指示，与平局用同一批投票者。

区间
----
按投票者重抽（保留重数），重复 500 次取 2.5%–97.5% 分位。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")
OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\rater\clean")
RAW = Path(r"D:\mycode\crime-temporal-spatio\data\raw\placepulse2\from_zip")
OUT.mkdir(parents=True, exist_ok=True)


def icc_corrected(p, n):
    w = n / n.sum()
    pbar = float(np.sum(w * p))
    if pbar <= 0 or pbar >= 1:
        return np.nan, np.nan, np.nan
    R = float(np.sum(w * (p - pbar) ** 2)) / (pbar * (1 - pbar))
    b = float(np.sum(w / n))
    return R, b, (R - b) / (1 - b)


def boot_icc(df, col, min_votes=6, reps=500, seed=0):
    g = df.groupby("voter_uniqueid")[col].agg(["mean", "count"])
    g = g[g["count"] >= min_votes]
    p, n = g["mean"].values, g["count"].values.astype(float)
    R, b, icc = icc_corrected(p, n)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(reps):
        k = rng.randint(0, len(p), len(p))
        _, _, v = icc_corrected(p[k], n[k])
        if v == v:
            vals.append(v)
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan))
    return dict(n_voters=int(len(p)), R=R, null=b, icc=icc, lo=lo, hi=hi,
                n_votes=int(n.sum()), mean_votes=float(n.mean()))


df = pd.read_parquet(DATA / "pairs.parquet")
df["_tie"] = df.choice.eq("equal").astype(float)
nt = df[df.choice.ne("equal")].copy()
nt["_right"] = nt.choice.eq("right").astype(float)

# 共同投票者集合：同时满足两种口径
g_t = df.groupby("voter_uniqueid").agg(n_all=("_tie", "size"))
g_r = nt.groupby("voter_uniqueid").agg(n_nt=("_right", "size"))
common = g_t.join(g_r, how="left").fillna(0)
keep = set(common[(common.n_all >= 6) & (common.n_nt >= 6)].index)
print(f"  共同投票者（两种口径均 >=6）: {len(keep):,}")

d_t = df[df.voter_uniqueid.isin(keep)]
d_r = nt[nt.voter_uniqueid.isin(keep)]
a = boot_icc(d_t, "_tie")
c = boot_icc(d_r, "_right")
print(f"  平局 ICC ... 朴素 {a['R']:.4f}  零基线 {a['null']:.4f}  校正 {a['icc']:.4f}"
      f"  [{a['lo']:.4f}, {a['hi']:.4f}]")
print(f"  偏好 ICC ... 朴素 {c['R']:.4f}  零基线 {c['null']:.4f}  校正 {c['icc']:.4f}"
      f"  [{c['lo']:.4f}, {c['hi']:.4f}]")
print(f"  比率 {a['icc']/c['icc']:.1f}×")

# 六属性
st = pd.read_csv(RAW / "studies.tsv", sep="\t", dtype=str)
votes = pd.read_csv(RAW / "votes.tsv", sep="\t", dtype=str,
                    usecols=["choice", "left", "right", "study_id", "voter_uniqueid"])
votes = votes.dropna(subset=["choice", "study_id", "voter_uniqueid"])
qmap = dict(zip(st["_id"], st["study_question"]))
votes["q"] = votes.study_id.map(qmap)
votes["_tie"] = votes.choice.eq("equal").astype(float)

rows = []
for q, g in votes.groupby("q"):
    if len(g) < 5000:
        continue
    r = boot_icc(g, "_tie")
    r["question"] = q
    r["n"] = len(g)
    r["tie_rate"] = float(g._tie.mean())
    rows.append(r)
    print(f"  {q:<18} n={len(g):>7} 校正 ICC {r['icc']:.4f} "
          f"[{r['lo']:.4f}, {r['hi']:.4f}]", flush=True)
t = pd.DataFrame(rows)
t.to_csv(OUT / "icc_formal.csv", index=False)
json.dump(dict(safer=a, preference_binary=c), open(OUT / "icc_main.json", "w"), indent=2)
print(f"\n  六属性 ICC 范围 {t.icc.min():.4f}–{t.icc.max():.4f}")
