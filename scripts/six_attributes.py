"""六属性复现：平局个体差异与「城市平局率 ~ 视觉同质性」是否跨属性稳健。

为什么需要
----------
目前所有结论都只在 safer 一个属性上验证。若六种感知问题
（safer / livelier / more boring / wealthier / more depressing / more beautiful）
都给出同样的结构，结论就从「一个任务的观察」变成「这类数据的一般性质」。

数据链路
--------
votes.tsv（choice, left, right, study_id, voter_uniqueid）
  + studies.tsv（study_id -> study_question）
  + locations.tsv（图像 _id -> place_id）
  + places.tsv（place_id -> place_name）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAW = Path(r"D:\mycode\crime-temporal-spatio\data\raw\placepulse2\from_zip")
OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\city")


def icc_moment(p, n):
    w = n / n.sum()
    pbar = float(np.sum(w * p))
    between = float(np.sum(w * (p - pbar) ** 2))
    total = pbar * (1 - pbar)
    return between / total if total > 0 else np.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-votes", type=int, default=6)
    ap.add_argument("--min-city-votes", type=int, default=500)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # 全部按 str 读，避免 Arrow 字符串里含 NA 时 astype 失败
    studies = pd.read_csv(RAW / "studies.tsv", sep="\t", dtype=str)
    print("=== studies ===")
    print(studies[["study_name", "study_question", "num_votes"]].to_string(index=False))

    loc = pd.read_csv(RAW / "locations.tsv", sep="\t",
                      usecols=["_id", "place_id"], dtype=str)
    loc = loc.fillna("")
    places = pd.read_csv(RAW / "places.tsv", sep="\t",
                         usecols=["_id", "place_name"], dtype=str)
    places = places.fillna("")
    p2n = dict(zip(places["_id"], places["place_name"]))
    city_of = {i: p2n[pid] for i, pid in zip(loc["_id"], loc["place_id"])
               if pid in p2n}
    print(f"\n  图像 {len(city_of):,} 张映射到城市；城市 {len(set(city_of.values()))} 个")

    votes = pd.read_csv(RAW / "votes.tsv", sep="\t",
                        usecols=["choice", "left", "right", "study_id",
                                 "voter_uniqueid"], dtype=str)
    votes = votes.dropna(subset=["choice", "left", "right", "study_id",
                                 "voter_uniqueid"])
    print(f"  原始投票 {len(votes):,}")
    votes["city_l"] = votes["left"].map(city_of)
    votes["city_r"] = votes["right"].map(city_of)
    votes["_tie"] = votes["choice"].eq("equal").astype(int)
    qmap = dict(zip(studies["_id"], studies["study_question"]))

    eco = pd.read_csv(args.out / "city_visual_homogeneity.csv")
    sim_map = dict(zip(eco.city, eco.sim_mean))

    rows, per_attr = [], {}
    for sid, g in votes.groupby("study_id"):
        q = qmap.get(str(sid), str(sid))
        if len(g) < 5000:
            continue
        t = g.groupby("voter_uniqueid")["_tie"].agg(["mean", "count"])
        t = t[t["count"] >= args.min_votes]
        icc = icc_moment(t["mean"].values, t["count"].values)

        # 城际：把「任一端在该城」的票归给该城
        recs = []
        for side in ("city_l", "city_r"):
            for c, sub in g.groupby(side):
                if pd.isna(c) or len(sub) < args.min_city_votes:
                    continue
                recs.append((c, sub["_tie"].mean(), len(sub)))
        cs = pd.DataFrame(recs, columns=["city", "tie_rate", "k"])
        cs = cs.groupby("city", as_index=False).agg(
            tie_rate=("tie_rate", "mean"), k=("k", "sum"))
        cs["sim"] = cs.city.map(sim_map)
        cs = cs.dropna(subset=["sim"])
        if len(cs) < 20:
            print(f"  {q:<18} 城市不足（{len(cs)}），跳过")
            continue
        r = stats.pearsonr(cs.tie_rate, cs.sim)
        rho = stats.spearmanr(cs.tie_rate, cs.sim)

        rows.append(dict(question=q, n=len(g), tie_rate=float(g._tie.mean()),
                         n_voters=int(g.voter_uniqueid.nunique()),
                         voter_median=float(t["count"].median()),
                         icc=icc, n_cities=len(cs),
                         r_sim=r.statistic, p_sim=r.pvalue,
                         rho_sim=rho.statistic))
        per_attr[q] = cs
        print(f"  {q:<18} n={len(g):>9}  平局 {g._tie.mean():.4f}  "
              f"ICC {icc:.3f}  城 {len(cs):>2}  r={r.statistic:+.3f} "
              f"(p={r.pvalue:.4f})", flush=True)

    res = pd.DataFrame(rows).sort_values("icc", ascending=False)
    res.to_csv(args.out / "six_attributes.csv", index=False)
    for q, cs in per_attr.items():
        cs.assign(question=q).to_csv(
            args.out / f"city_tie_{q.replace(' ', '_')}.csv", index=False)

    print("\n=== 六属性汇总 ===")
    print(res[["question", "n", "tie_rate", "icc", "r_sim", "p_sim"]]
          .to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print(f"\n  ICC 范围 {res.icc.min():.3f} – {res.icc.max():.3f}"
          f"（中位 {res.icc.median():.3f}）")
    print(f"  城际相关 {res.r_sim.min():+.3f} – {res.r_sim.max():+.3f}"
          f"（中位 {res.r_sim.median():+.3f}）")
    npos = int((res.r_sim > 0).sum())
    p_sign = stats.binomtest(npos, len(res), 0.5, alternative="greater").pvalue
    print(f"  正相关 {npos}/{len(res)} 个属性，符号检验 p = {p_sign:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
