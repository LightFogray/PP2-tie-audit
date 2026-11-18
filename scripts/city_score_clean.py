"""地点得分不变性：严格隔离 + 正确的投票者自助。

修正两处
--------
1) 原 bootstrap 用 np.isin 做布尔掩码，把重复抽中的投票者的重数丢掉了；
   这里改为按抽中次数拼接行索引，保留重数。
2) 原代码对每次重复取"物品间的标准差"，衡量的是得分分散程度；
   这里改为对每个物品收集其跨重复的 θ，再取该物品的标准差，
   衡量的是同一物品估计的不确定性。

隔离：p0 与 gamma 只用训练城（排除测试城与验证城）。
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from history_ablation import fit_davidson
from hier_transfer_kappa import (city_data, fit_global_nu, fit_theta_nu)

DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")
OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\rater\clean")
LAB = {"left": 0, "equal": 1, "right": 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-img", type=int, default=150)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--kappa", type=float, default=0.25)
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out = OUT; args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA / "pairs.parquet")
    df["_tie"] = df.choice.eq("equal").astype(int)
    df["_y"] = df["choice"].map(LAB).astype(np.int8)
    folds = json.loads((DATA / "splits" / "logo_folds.json").read_text())
    ids = sorted(set(df["left"]) | set(df["right"]))
    idx = {s: i for i, s in enumerate(ids)}
    df["_ia"] = df["left"].map(idx).astype(np.int32)
    df["_ib"] = df["right"].map(idx).astype(np.int32)

    rows, boots = [], []
    for f in folds:
        c, v = f["test_city"], f["val_city"]
        in_t = (df["left_city"] == c) | (df["right_city"] == c)
        in_v = (df["left_city"] == v) | (df["right_city"] == v)
        tr = df[(~in_t) & (~in_v)]
        th_all, _, _ = fit_davidson(tr["_ia"].values, tr["_ib"].values,
                                    tr["_y"].values, len(ids))
        p0 = float(tr["_tie"].mean())
        agg = tr.groupby("voter_uniqueid")["_tie"].agg(["sum", "count"])
        rate = (agg["sum"] + args.alpha * p0) / (agg["count"] + args.alpha)
        rr = np.clip(rate, 1e-4, 1 - 1e-4)
        lg = np.log(rr / (1 - rr))
        w = agg["count"].values.astype(float)
        lg_c = float(np.average(lg.values, weights=w)) if w.sum() > 0 else 0.0
        gmap = (lg - lg_c).to_dict()

        d = city_data(c, v, df, None, np.random.RandomState(args.seed), args.max_img)
        if d is None:
            continue
        thref = th_all[d["ref_i"]]
        gamma = np.array([gmap.get(r, 0.0) for r in d["voters"]])

        th_g, nu_g = fit_global_nu(thref, d["test_i"], d["on_left"], d["y"])
        th_h, _ = fit_theta_nu(thref, d["test_i"], d["on_left"], d["y"],
                               nu_g * np.exp(args.kappa * gamma))
        keys = sorted(set(th_g) & set(th_h))
        a = np.array([th_g[k] for k in keys]); b = np.array([th_h[k] for k in keys])
        rows.append(dict(city=c, n_item=len(keys),
                         rho_items=stats.spearmanr(a, b).statistic,
                         mean_g=a.mean(), mean_h=b.mean(),
                         max_abs_shift=float(np.abs(b - a).max())))

        # 正确的投票者自助：保留重数
        vids = np.unique(d["voters"])
        idx_of = {u: np.flatnonzero(d["voters"] == u) for u in vids}
        rng = np.random.RandomState(args.seed)
        G, H = [], []
        for _ in range(args.boot):
            pick = rng.choice(vids, len(vids), replace=True)
            sel = np.concatenate([idx_of[u] for u in pick])
            tg, ng = fit_global_nu(thref[sel], d["test_i"][sel],
                                   d["on_left"][sel], d["y"][sel])
            th2, _ = fit_theta_nu(thref[sel], d["test_i"][sel], d["on_left"][sel],
                                  d["y"][sel],
                                  ng * np.exp(args.kappa * gamma[sel]))
            kk = sorted(set(tg) & set(th2))
            if len(kk) < 30:
                continue
            G.append(np.array([tg[k] for k in kk]))
            H.append(np.array([th2[k] for k in kk]))
        if len(G) >= 20:
            # 只保留在全部重复中都出现的物品
            common = set(range(len(G[0])))
            for x in G: common &= set(range(len(x)))
            common = sorted(common)
            G = np.array([x[common] for x in G]); H = np.array([x[common] for x in H])
            sd_g_item = G.std(axis=0); sd_h_item = H.std(axis=0)
            boots.append(dict(city=c, n_boot=len(G), n_item=len(common),
                              med_sd_global=float(np.median(sd_g_item)),
                              med_sd_hier=float(np.median(sd_h_item)),
                              ratio=float(np.median(sd_h_item)/np.median(sd_g_item))))
            print(f"  {c:<16} rho={rows[-1]['rho_items']:.4f} "
                  f"SD_g={np.median(sd_g_item):.4f} SD_h={np.median(sd_h_item):.4f} "
                  f"ratio={boots[-1]['ratio']:.3f}", flush=True)

    pd.DataFrame(rows).to_csv(args.out / "city_score_clean.csv", index=False)
    bt = pd.DataFrame(boots)
    bt.to_csv(args.out / "city_bootstrap_clean.csv", index=False)
    print(f"\n  城市 {len(rows)}  rho 均值 {np.mean([r['rho_items'] for r in rows]):.5f}")
    print(f"  每物品估计 SD 比值（层级/全局）: 中位 {bt.ratio.median():.4f}  "
          f"范围 {bt.ratio.min():.3f}–{bt.ratio.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
