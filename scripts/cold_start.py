"""冷启动实验：没有历史的投票者会怎样？

先修的一个 bug
--------------
收缩后的平局率 rate = (s + α·p0)/(n + α)。n=0 时 rate = p0，
取 logit 得 logit(p0) ≈ −1.95 —— **不是 0**。
但冷投票者本应拿到总体均值，即 γ = 0。
修正：γ_r = logit(rate_r) − logit(p0)，使无历史者恰为 0。

实验
----
对每条留出比较，记录该投票者在**其他城市**的票数 n_out，
按 n_out 分层比较 global ν 与 hier 的逐票 log loss。
这回答「增益是否只来自有历史的投票者」以及「冷启动代价多大」。
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
from hier_transfer_kappa import (city_data, fit_discard, fit_global_nu,
                                 fit_theta, fit_theta_nu, ll3, p3mat, tie_ap)

DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")
OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\rater")
LAB = {"left": 0, "equal": 1, "right": 2}


def per_vote_ll(p, y):
    p = np.clip(p, 1e-9, 1)
    return -np.log(p[np.arange(len(y)), y])


def run_city_votes(city, other, df, th_all, gamma_map, nout_map,
                   rng, max_img, kappa):
    d = city_data(city, other, df, None, rng, max_img)
    if d is None:
        return None
    perm = rng.permutation(len(d["y"]))
    n_fit = len(d["y"]) // 2
    fit_i, ev_i = perm[:n_fit], perm[n_fit:]
    theta_ref = th_all[d["ref_i"]]

    th_g, nu_g = fit_global_nu(theta_ref[fit_i], d["test_i"][fit_i],
                               d["on_left"][fit_i], d["y"][fit_i])
    g_fit = np.array([gamma_map.get(r, 0.0) for r in d["voters"][fit_i]])
    g_ev = np.array([gamma_map.get(r, 0.0) for r in d["voters"][ev_i]])

    p_g = p3mat(th_g, theta_ref[ev_i], d["test_i"][ev_i], d["on_left"][ev_i],
                nu_r=np.full(len(ev_i), nu_g))
    # 层级模型自己估 ν，不沿用全局 ν（单一 ν 与 γ 不匹配）
    th_h, nu_h = fit_theta_nu(theta_ref[fit_i], d["test_i"][fit_i],
                              d["on_left"][fit_i], d["y"][fit_i],
                              kappa * g_fit)
    p_h = p3mat(th_h, theta_ref[ev_i], d["test_i"][ev_i], d["on_left"][ev_i],
                nu_r=nu_h * np.exp(kappa * g_ev))

    y = d["y"][ev_i]
    return pd.DataFrame(dict(
        city=city,
        n_out=np.array([nout_map.get(r, 0) for r in d["voters"][ev_i]]),
        y=y,
        ll_global=per_vote_ll(p_g, y),
        ll_hier=per_vote_ll(p_h, y),
        p_tie_global=p_g[:, 1], p_tie_hier=p_h[:, 1],
    ))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-img", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kappas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 1.0])
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
    logit_p0 = np.log(p0 / (1 - p0))

    frames = []
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
        # 关键修正：按 γ 自身的加权均值中心化。
        # 用 logit(p0) 中心化是错的——logit 的均值不等于均值的 logit
        # （Jensen），会让所有投票者被系统性压低约 −0.46。
        lg = np.log(rr / (1 - rr))
        w = agg["count"].values.astype(float)
        lg_center = float(np.average(lg.values, weights=w))
        gamma_map = (lg - lg_center).to_dict()
        nout_map = agg["count"].to_dict()

        # 验证城选 κ
        val = run_city_votes(v, c, df, th_all, gamma_map, nout_map,
                             np.random.RandomState(args.seed),
                             args.max_img, 0.5)
        if val is None:
            continue
        best, bestk = 9e9, 0.5
        for kap in args.kappas:
            tmp = run_city_votes(v, c, df, th_all, gamma_map, nout_map,
                                 np.random.RandomState(args.seed),
                                 args.max_img, kap)
            if tmp is None:
                continue
            l = tmp.ll_hier.mean()
            if l < best:
                best, bestk = l, kap

        out = run_city_votes(c, v, df, th_all, gamma_map, nout_map,
                             np.random.RandomState(args.seed),
                             args.max_img, bestk)
        if out is None:
            continue
        out["kappa"] = bestk
        frames.append(out)
        print(f"  {c:<16} κ*={bestk:<5g} 票 {len(out):>6}  "
              f"global {out.ll_global.mean():.4f}  hier {out.ll_hier.mean():.4f}",
              flush=True)

    d = pd.concat(frames, ignore_index=True)
    d.to_csv(args.out / "cold_start_votes.csv", index=False)

    print(f"\n=== 全样本（{len(d):,} 条留出比较）===")
    print(f"  global {d.ll_global.mean():.5f}   hier {d.ll_hier.mean():.5f}   "
          f"差 {(d.ll_hier - d.ll_global).mean():+.5f}")

    print("\n=== 按投票者城外历史票数分层 ===")
    d["stratum"] = pd.cut(d.n_out, [-1, 0, 5, 15, 40, 10**9],
                          labels=["0", "1-5", "6-15", "16-40", ">40"])
    d["d"] = d.ll_hier - d.ll_global
    print(f"  {'n_out':<8}{'票数':>9}{'占比':>8}{'global':>10}{'hier':>10}"
          f"{'差':>10}{'平局率':>9}")
    for s, g in d.groupby("stratum", observed=True):
        print(f"  {str(s):<8}{len(g):>9}{len(g)/len(d)*100:>7.1f}%"
              f"{g.ll_global.mean():>10.5f}{g.ll_hier.mean():>10.5f}"
              f"{g.d.mean():>+10.5f}{g.y.eq(1).mean():>9.4f}")

    print("\n=== 增益随历史长度的趋势 ===")
    sm = d.groupby("stratum", observed=True).d.mean()
    print("  " + "  ".join(f"{k}:{v:+.5f}" for k, v in sm.items()))
    rho, pv = stats.spearmanr(d.n_out, -d.d)
    print(f"  Spearman(n_out, 每票增益) = {rho:+.4f}  p={pv:.2e}")

    cold = d[d.n_out == 0]
    print(f"\n=== 冷启动（n_out=0，{len(cold):,} 条，占 {len(cold)/len(d)*100:.1f}%）===")
    print(f"  global {cold.ll_global.mean():.5f}  hier {cold.ll_hier.mean():.5f}  "
          f"差 {cold.d.mean():+.5f}")
    print("  （冷投票者自身 γ=0，但 θ 由全体投票者共同估计，")
    print("    拟合集中有历史者的 γ≠0 会改变共享的 θ，故差值不严格为 0）")
    print(f"  最大逐票差异 {np.abs(cold.d).max():.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
