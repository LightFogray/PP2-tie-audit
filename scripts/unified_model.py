"""统一模型：严格隔离 + 逐票预测输出。

修正两处泄漏
------------
1) 先验 p0 与受试者历史 gamma 只用训练城（排除测试城与验证城）
2) kappa 只在验证城选，测试城不参与任何选择

输出
----
clean/unified_votes.parquet  每条留出比较一行：
  fold, city, voter, n_train, y, p_discard[3], p_global[3], p_hier[3]
clean/unified_folds.csv      每折汇总
表 4 与表 5 都从同一个 parquet 派生。
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from history_ablation import fit_davidson
from hier_transfer_kappa import (city_data, fit_discard, fit_global_nu,
                                 fit_theta, ll3, p3mat, tie_ap)

DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")
OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\rater\clean")
LAB = {"left": 0, "equal": 1, "right": 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-img", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kappas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA / "pairs.parquet")
    df["_tie"] = df.choice.eq("equal").astype(int)
    df["_y"] = df["choice"].map(LAB).astype(np.int8)
    folds = json.loads((DATA / "splits" / "logo_folds.json").read_text())
    ids = sorted(set(df["left"]) | set(df["right"]))
    idx = {s: i for i, s in enumerate(ids)}
    df["_ia"] = df["left"].map(idx).astype(np.int32)
    df["_ib"] = df["right"].map(idx).astype(np.int32)

    rng = np.random.RandomState(args.seed)
    vote_rows, fold_rows = [], []

    for f in folds:
        c, v = f["test_city"], f["val_city"]
        in_t = (df["left_city"] == c) | (df["right_city"] == c)
        in_v = (df["left_city"] == v) | (df["right_city"] == v)
        tr = df[(~in_t) & (~in_v)]

        th_all, _, _ = fit_davidson(tr["_ia"].values, tr["_ib"].values,
                                    tr["_y"].values, len(ids))
        p0_tr = float(tr["_tie"].mean())
        agg = tr.groupby("voter_uniqueid")["_tie"].agg(["sum", "count"])
        rate = (agg["sum"] + args.alpha * p0_tr) / (agg["count"] + args.alpha)
        rr = np.clip(rate, 1e-4, 1 - 1e-4)
        gamma = np.log(rr / (1 - rr))
        gamma_map = gamma.to_dict()
        n_train = agg["count"].to_dict()

        def prep(city, other):
            d = city_data(city, other, df, None,
                          np.random.RandomState(args.seed), args.max_img)
            if d is None:
                return None
            perm = np.random.RandomState(args.seed).permutation(len(d["y"]))
            h = len(d["y"]) // 2
            return d, perm[:h], perm[h:]

        pv = prep(v, c)
        if pv is None:
            continue
        dv, fv, ev = pv
        thref = th_all[dv["ref_i"]]
        best, kbest = None, None
        for kap in args.kappas:
            th_h = fit_theta(thref[fv], dv["test_i"][fv], dv["on_left"][fv],
                             dv["y"][fv],
                             np.exp(kap * np.array([gamma_map.get(r, 0.0)
                                                    for r in dv["voters"][fv]])))
            # 用验证折内的全局 nu 作基准
            _, nu_g = fit_global_nu(thref[fv], dv["test_i"][fv],
                                    dv["on_left"][fv], dv["y"][fv])
            g_ev = np.array([gamma_map.get(r, 0.0) for r in dv["voters"][ev]])
            p = p3mat(th_h, thref[ev], dv["test_i"][ev], dv["on_left"][ev],
                      nu_r=nu_g * np.exp(kap * g_ev))
            s = ll3(p, dv["y"][ev])
            if best is None or s < best:
                best, kbest = s, kap
        if kbest is None:
            continue

        pv = prep(c, v)
        if pv is None:
            continue
        d, fi, ei = pv
        thref = th_all[d["ref_i"]]
        y_e = d["y"][ei]

        th_d, tconst = fit_discard(thref[fi], d["test_i"][fi], d["on_left"][fi],
                                   d["y"][fi])
        p_d = p3mat(th_d, thref[ei], d["test_i"][ei], d["on_left"][ei],
                    tie_const=tconst)
        th_g, nu_g = fit_global_nu(thref[fi], d["test_i"][fi], d["on_left"][fi],
                                   d["y"][fi])
        p_g = p3mat(th_g, thref[ei], d["test_i"][ei], d["on_left"][ei],
                    nu_r=np.full(len(ei), nu_g))
        g_fi = np.array([gamma_map.get(r, 0.0) for r in d["voters"][fi]])
        g_e = np.array([gamma_map.get(r, 0.0) for r in d["voters"][ei]])
        th_h = fit_theta(thref[fi], d["test_i"][fi], d["on_left"][fi], d["y"][fi],
                         nu_g * np.exp(kbest * g_fi))
        p_h = p3mat(th_h, thref[ei], d["test_i"][ei], d["on_left"][ei],
                    nu_r=nu_g * np.exp(kbest * g_e))

        for j in range(len(ei)):
            r = d["voters"][ei[j]]
            vote_rows.append(dict(
                fold=c, city=c, voter=r, n_train=int(n_train.get(r, 0)), y=int(y_e[j]),
                pd0=p_d[j, 0], pd1=p_d[j, 1], pd2=p_d[j, 2],
                pg0=p_g[j, 0], pg1=p_g[j, 1], pg2=p_g[j, 2],
                ph0=p_h[j, 0], ph1=p_h[j, 1], ph2=p_h[j, 2]))

        fold_rows.append(dict(city=c, kappa=kbest, n_ev=len(ei),
                              ll_discard=ll3(p_d, y_e), ll_global=ll3(p_g, y_e),
                              ll_hier=ll3(p_h, y_e),
                              ap_discard=tie_ap(p_d[:, 1], y_e == 1),
                              ap_global=tie_ap(p_g[:, 1], y_e == 1),
                              ap_hier=tie_ap(p_h[:, 1], y_e == 1)))
        print(f"  {c:<16} k*={kbest:<5g} n={len(ei):>6} "
              f"{fold_rows[-1]['ll_discard']:.4f} "
              f"{fold_rows[-1]['ll_global']:.4f} "
              f"{fold_rows[-1]['ll_hier']:.4f}", flush=True)

    pd.DataFrame(vote_rows).to_parquet(args.out / "unified_votes.parquet")
    t = pd.DataFrame(fold_rows)
    t.to_csv(args.out / "unified_folds.csv", index=False)
    print(f"\n  投票行 {len(vote_rows):,}  折 {len(t)}")
    print(f"  discard {t.ll_discard.mean():.5f}  global {t.ll_global.mean():.5f}"
          f"  hier {t.ll_hier.mean():.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
