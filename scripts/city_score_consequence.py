"""城市层面的后果：模型是否改变或锐化了地点得分？

为什么这是 CEUS 的生死线
------------------------
若 rater-aware 模型既不改变城市/物品排序，也不收窄不确定性，
那它对城市分析没有可交付的后果，编辑会判 out of scope。

三项检验
--------
1. 物品得分是否移动：同一批测试城图像，global ν 与 hier 的 θ 相关
2. 城市层面是否移动：城市平均 θ 的排名变化
3. 不确定性是否收窄：对每城做投票者自助重采样，比较 θ 的分散度

若 1 的相关接近 1、2 的排名几乎不变，则必须承认「无推断后果」，
把论文定位为「预测层的改进」而不是「地点测量的更正」。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from history_ablation import fit_davidson
from hier_transfer_kappa import (city_data, fit_global_nu, fit_theta_nu)

DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")
OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\city")
LAB = {"left": 0, "equal": 1, "right": 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-img", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--boot", type=int, default=8)
    ap.add_argument("--alpha-kappa", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA / "pairs.parquet")
    df["_tie"] = df.choice.eq("equal").astype(int)
    folds = json.loads((DATA / "splits" / "logo_folds.json").read_text())
    df["_y"] = df["choice"].map(LAB).astype(np.int8)
    ids = sorted(set(df["left"]) | set(df["right"]))
    idx = {s: i for i, s in enumerate(ids)}
    df["_ia"] = df["left"].map(idx).astype(np.int32)
    df["_ib"] = df["right"].map(idx).astype(np.int32)
    p0 = float(df._tie.mean())

    rows, city_rows = [], []
    for f in folds:
        c, v = f["test_city"], f["val_city"]
        in_t = (df["left_city"] == c) | (df["right_city"] == c)
        in_v = (df["left_city"] == v) | (df["right_city"] == v)
        tr = df[(~in_t) & (~in_v)]
        th_all, _, _ = fit_davidson(tr["_ia"].values, tr["_ib"].values,
                                    tr["_y"].values, len(ids))
        agg = df[~in_t].groupby("voter_uniqueid")["_tie"].agg(["sum", "count"])
        rate = (agg["sum"] + args.alpha * p0) / (agg["count"] + args.alpha)
        rr = np.clip(rate, 1e-4, 1 - 1e-4)
        lg = np.log(rr / (1 - rr))
        lg_c = float(np.average(lg.values, weights=agg["count"].values.astype(float)))
        gmap = (lg - lg_c).to_dict()

        d = city_data(c, v, df, None, np.random.RandomState(args.seed), args.max_img)
        if d is None:
            continue
        theta_ref = th_all[d["ref_i"]]
        gamma = np.array([gmap.get(r, 0.0) for r in d["voters"]])

        th_g, nu_g = fit_global_nu(theta_ref, d["test_i"], d["on_left"], d["y"])
        th_h, nu_h = fit_theta_nu(theta_ref, d["test_i"], d["on_left"], d["y"],
                                  args.alpha_kappa * gamma)

        keys = sorted(set(th_g) & set(th_h))
        a = np.array([th_g[k] for k in keys])
        b = np.array([th_h[k] for k in keys])
        rho = stats.spearmanr(a, b).statistic
        rows.append(dict(city=c, n_item=len(keys), rho_items=rho,
                         mean_g=float(a.mean()), mean_h=float(b.mean()),
                         sd_g=float(a.std()), sd_h=float(b.std()),
                         pear=float(np.corrcoef(a, b)[0, 1])))

        # 投票者自助：θ 的稳定性
        vids = np.unique(d["voters"])
        bs = []
        rng = np.random.RandomState(args.seed)
        for _ in range(args.boot):
            pick = rng.choice(vids, len(vids), replace=True)
            keep = np.isin(d["voters"], pick)
            if keep.sum() < 200:
                continue
            tg, _ = fit_global_nu(theta_ref[keep], d["test_i"][keep],
                                  d["on_left"][keep], d["y"][keep])
            th2, _ = fit_theta_nu(theta_ref[keep], d["test_i"][keep],
                                  d["on_left"][keep], d["y"][keep],
                                  args.alpha_kappa * gamma[keep])
            kk = sorted(set(tg) & set(th2))
            if len(kk) < 50:
                continue
            bs.append((np.array([tg[k] for k in kk]),
                       np.array([th2[k] for k in kk])))
        if bs:
            sg = np.mean([x[0].std() for x in bs])
            sh = np.mean([x[1].std() for x in bs])
            city_rows.append(dict(city=c, boot_sd_global=sg, boot_sd_hier=sh))
        print(f"  {c:<16} 物品 {len(keys):>4}  ρ={rho:.4f}  "
              f"均值 {a.mean():+.3f}->{b.mean():+.3f}", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(args.out / "city_score_change.csv", index=False)
    print("\n=== 物品得分是否移动 ===")
    print(f"  物品层面 Spearman ρ 均值 {t.rho_items.mean():.4f}  "
          f"最小 {t.rho_items.min():.4f}")
    print(f"  Pearson 相关均值 {t.pear.mean():.4f}")
    print(f"  城均 θ 的变化：均值 {t.mean_g.mean():+.4f} -> {t.mean_h.mean():+.4f}")

    if city_rows:
        ct = pd.DataFrame(city_rows)
        ct.to_csv(args.out / "city_bootstrap_sd.csv", index=False)
        print("\n=== 自助重采样下 θ 的分散度 ===")
        print(f"  global {ct.boot_sd_global.mean():.4f}   "
              f"hier {ct.boot_sd_hier.mean():.4f}   "
              f"比值 {ct.boot_sd_hier.mean()/ct.boot_sd_global.mean():.4f}")
        dlt = ct.boot_sd_hier - ct.boot_sd_global
        print(f"  差 {dlt.mean():+.5f}  p={stats.wilcoxon(dlt).pvalue:.4f}  "
              f"更稳 {int((dlt < 0).sum())}/{len(ct)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
