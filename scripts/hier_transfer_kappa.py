"""层级 Davidson（跨城迁移 + 标定版）。

关键修正
--------
Davidson 里的 ν_r 是**平局尺度**，不是平局概率本身；它与 |d| 交互。
因此不能把投票者的平局率取 logit 直接当 γ_r 用，必须标定缩放系数 κ：

    ν_r = ν · exp(κ · γ_r)

κ 在**验证城**上选（验证城做与测试城完全相同的估计流程），
再原封不动用到测试城。这样测试城从不接触自己的留出标签。

四个模型
--------
discard  只用非平局票，平局概率取拟合集常数
global   全体票，全局 ν
hier     ν_r = ν·exp(κ·γ_r)，κ 在验证城标定
hier0    ν_r = ν·exp(γ_r) 不标定（负面对照，展示标定的必要性）
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
from hier_davidson import NU_GRID, THETA_GRID

DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")
OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\rater")
LAB = {"left": 0, "equal": 1, "right": 2}


def groups_of(sorted_ids, n):
    uniq, start = np.unique(sorted_ids, return_index=True)
    end = np.append(start[1:], n)
    return {u: (s, e) for u, s, e in zip(uniq, start, end)}


def item_mle(theta_ref, on_left, y, nu_r):
    R = theta_ref[None, :]
    D = np.where(on_left[None, :], THETA_GRID[:, None] - R, R - THETA_GRID[:, None])
    s = nu_r[None, :] * np.exp(D / 2)
    ed = np.exp(D)
    den = ed + 1.0 + s
    P = np.stack([ed / den, s / den, 1.0 / den], 0)
    idx = np.arange(D.shape[1])[None, :]
    sel = np.clip(P[y[None, :], np.arange(P.shape[1])[:, None], idx], 1e-12, None)
    return float(THETA_GRID[int(np.argmax(np.log(sel).sum(1)))])


def fit_theta(theta_ref, item_ids, on_left, y, nu_r):
    order = np.argsort(item_ids, kind="stable")
    ii = item_ids[order]
    g = groups_of(ii, len(ii))
    th = {int(u): 0.0 for u in g}
    for u, (s, e) in g.items():
        idx = order[s:e]
        th[int(u)] = item_mle(theta_ref[idx], on_left[idx], y[idx], nu_r[idx])
    return th


def fit_theta_nu(theta_ref, item_ids, on_left, y, gamma, n_iter=4, nu0=0.3):
    """层级版：γ 固定，交替估 θ 与 ν。

    不能用全局模型估出的 ν 直接配 γ —— 单一的 ν 在投票者异质时
    是一个折中值，与 γ 不匹配。这里让 ν 在给定 γ 下重新估计。
    """
    order = np.argsort(item_ids, kind="stable")
    ii = item_ids[order]
    g = groups_of(ii, len(ii))
    th = {int(u): 0.0 for u in g}
    nu = nu0
    for _ in range(n_iter):
        nu_r = nu * np.exp(gamma)
        for u, (s, e) in g.items():
            idx = order[s:e]
            th[int(u)] = item_mle(theta_ref[idx], on_left[idx], y[idx], nu_r[idx])
        ta = np.array([th[i] for i in item_ids])
        d = np.where(on_left, ta - theta_ref, theta_ref - ta)
        lls = np.zeros_like(NU_GRID)
        for ni, nv in enumerate(NU_GRID):
            s2 = nv * np.exp(gamma) * np.exp(d / 2)
            ed = np.exp(d)
            den = ed + 1 + s2
            P = np.stack([ed / den, s2 / den, 1.0 / den], 1)
            lls[ni] = np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).sum()
        nu = float(NU_GRID[int(np.argmax(lls))])
    # 用最终 ν 再更新一次 θ
    nu_r = nu * np.exp(gamma)
    for u, (s, e) in g.items():
        idx = order[s:e]
        th[int(u)] = item_mle(theta_ref[idx], on_left[idx], y[idx], nu_r[idx])
    return th, nu


def fit_global_nu(theta_ref, item_ids, on_left, y, n_iter=5, nu0=0.3):
    order = np.argsort(item_ids, kind="stable")
    ii = item_ids[order]
    g = groups_of(ii, len(ii))
    th = {int(u): 0.0 for u in g}
    nu = nu0
    for _ in range(n_iter):
        for u, (s, e) in g.items():
            idx = order[s:e]
            th[int(u)] = item_mle(theta_ref[idx], on_left[idx], y[idx],
                                  np.full(len(idx), nu))
        ta = np.array([th[i] for i in item_ids])
        d = np.where(on_left, ta - theta_ref, theta_ref - ta)
        lls = np.zeros_like(NU_GRID)
        for ni, nv in enumerate(NU_GRID):
            s2 = nv * np.exp(d / 2)
            ed = np.exp(d)
            den = ed + 1 + s2
            P = np.stack([ed / den, s2 / den, 1.0 / den], 1)
            lls[ni] = np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).sum()
        nu = float(NU_GRID[int(np.argmax(lls))])
    # 用最终 ν 再更新一次 θ：否则返回的 θ 用的是倒数第二轮 ν，
    # 与「固定 ν 后估 θ」的路径不一致，会让 γ=0 的投票者出现虚假差异。
    for u, (s, e) in g.items():
        idx = order[s:e]
        th[int(u)] = item_mle(theta_ref[idx], on_left[idx], y[idx],
                              np.full(len(idx), nu))
    return th, nu


def fit_discard(theta_ref, item_ids, on_left, y):
    keep = y != 1
    order = np.argsort(item_ids[keep], kind="stable")
    ii = item_ids[keep][order]
    g = groups_of(ii, len(ii))
    th = {int(u): 0.0 for u in np.unique(item_ids)}
    for u, (s, e) in g.items():
        idx = order[s:e]
        th[int(u)] = item_mle(theta_ref[keep][idx], on_left[keep][idx],
                              y[keep][idx], np.full(len(idx), 1e-6))
    return th, float(np.mean(y == 1))


def p3mat(th, theta_ref, item_ids, on_left, nu_r=None, tie_const=None):
    t = np.array([th.get(int(i), 0.0) for i in item_ids])
    d = np.where(on_left, t - theta_ref, theta_ref - t)
    ed = np.exp(d)
    if tie_const is not None:
        den = ed + 1.0
        return np.stack([ed / den * (1 - tie_const),
                         np.full(len(d), tie_const),
                         1.0 / den * (1 - tie_const)], 1)
    s = nu_r * np.exp(d / 2)
    den = ed + 1.0 + s
    return np.stack([ed / den, s / den, 1.0 / den], 1)


def ll3(p, y):
    p = np.clip(p, 1e-9, 1)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def tie_ap(p_tie, y_tie):
    order = np.argsort(-p_tie)
    yy = y_tie[order]
    tp, s = 0, 0.0
    for i, v in enumerate(yy):
        if v:
            tp += 1
            s += tp / (i + 1)
    return s / max(yy.sum(), 1)


def city_data(city, other, df, th_all_hook, rng, max_img):
    """取出某城市做测试时的比较数组（去掉另一城市）。"""
    c, v = city, other
    in_t = (df["left_city"] == c) | (df["right_city"] == c)
    in_v = (df["left_city"] == v) | (df["right_city"] == v)
    t2 = df[in_t & (~in_v) & (~df["same_city"])].reset_index(drop=True)
    if len(t2) < 200:
        return None
    left_is_test = (t2["left_city"] == c).values
    test_i = np.where(left_is_test, t2["_ia"], t2["_ib"])
    ref_i = np.where(left_is_test, t2["_ib"], t2["_ia"])
    on_left, y = left_is_test, t2["_y"].values
    voters = t2["voter_uniqueid"].values
    order = np.argsort(test_i, kind="stable")
    test_i, ref_i = test_i[order], ref_i[order]
    on_left, y, voters = on_left[order], y[order], voters[order]
    uniq, start = np.unique(test_i, return_index=True)
    end = np.append(start[1:], len(test_i))
    ok = [(u, s, e) for u, s, e in zip(uniq, start, end) if e - s >= 8]
    rng.shuffle(ok)
    ok = ok[:max_img]
    if not ok:
        return None
    keep = np.concatenate([np.arange(s, e) for _, s, e in ok])
    return dict(test_i=test_i[keep], ref_i=ref_i[keep], on_left=on_left[keep],
                y=y[keep], voters=voters[keep])


def run_city(city, other, df, th_all, gamma_map, rng, max_img, kappas):
    d = city_data(city, other, df, None, rng, max_img)
    if d is None:
        return None
    perm = rng.permutation(len(d["y"]))
    n_fit = len(d["y"]) // 2
    fit_i, ev_i = perm[:n_fit], perm[n_fit:]
    theta_ref = th_all[d["ref_i"]]

    out = {"city": city, "n_fit": len(fit_i), "n_ev": len(ev_i)}

    th_d, tconst = fit_discard(theta_ref[fit_i], d["test_i"][fit_i],
                               d["on_left"][fit_i], d["y"][fit_i])
    p = p3mat(th_d, theta_ref[ev_i], d["test_i"][ev_i], d["on_left"][ev_i],
              tie_const=tconst)
    out["ll_discard"] = ll3(p, d["y"][ev_i])
    out["ap_discard"] = tie_ap(p[:, 1], (d["y"][ev_i] == 1).astype(int))

    th_g, nu_g = fit_global_nu(theta_ref[fit_i], d["test_i"][fit_i],
                               d["on_left"][fit_i], d["y"][fit_i])
    p = p3mat(th_g, theta_ref[ev_i], d["test_i"][ev_i], d["on_left"][ev_i],
              nu_r=np.full(len(ev_i), nu_g))
    out["ll_global"] = ll3(p, d["y"][ev_i])
    out["ap_global"] = tie_ap(p[:, 1], (d["y"][ev_i] == 1).astype(int))
    out["nu"] = nu_g

    g_fit = np.array([gamma_map.get(r, 0.0) for r in d["voters"][fit_i]])
    g_ev = np.array([gamma_map.get(r, 0.0) for r in d["voters"][ev_i]])
    for kap in kappas:
        th_h = fit_theta(theta_ref[fit_i], d["test_i"][fit_i],
                         d["on_left"][fit_i], d["y"][fit_i],
                         nu_g * np.exp(kap * g_fit))
        p = p3mat(th_h, theta_ref[ev_i], d["test_i"][ev_i], d["on_left"][ev_i],
                  nu_r=nu_g * np.exp(kap * g_ev))
        out[f"ll_k{kap:g}"] = ll3(p, d["y"][ev_i])
        out[f"ap_k{kap:g}"] = tie_ap(p[:, 1], (d["y"][ev_i] == 1).astype(int))
    out["gamma_sd"] = float(np.std(g_ev))
    return out


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
    folds = json.loads((DATA / "splits" / "logo_folds.json").read_text())
    df["_y"] = df["choice"].map(LAB).astype(np.int8)
    ids = sorted(set(df["left"]) | set(df["right"]))
    idx = {s: i for i, s in enumerate(ids)}
    df["_ia"] = df["left"].map(idx).astype(np.int32)
    df["_ib"] = df["right"].map(idx).astype(np.int32)
    p0 = float(df._tie.mean())

    rng = np.random.RandomState(args.seed)
    rows = []
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
        gamma_map = np.log(rr / (1 - rr)).to_dict()

        # 1) 在验证城上选 κ
        val = run_city(v, c, df, th_all, gamma_map, np.random.RandomState(args.seed),
                       args.max_img, args.kappas)
        if val is None:
            continue
        score = {k: val[f"ll_k{k:g}"] for k in args.kappas}
        kbest = min(score, key=score.get)

        # 2) 用同一个 κ 在测试城上评估
        test = run_city(c, v, df, th_all, gamma_map,
                        np.random.RandomState(args.seed), args.max_img,
                        args.kappas)
        if test is None:
            continue
        rec = dict(city=c, kappa=kbest,
                   ll_discard=test["ll_discard"], ll_global=test["ll_global"],
                   ap_discard=test["ap_discard"], ap_global=test["ap_global"],
                   ll_hier=test[f"ll_k{kbest:g}"], ap_hier=test[f"ap_k{kbest:g}"],
                   ll_hier0=test["ll_k0"], gamma_sd=test["gamma_sd"],
                   n_ev=test["n_ev"])
        rows.append(rec)
        print(f"  {c:<16} κ*={kbest:<5g} discard {rec['ll_discard']:.4f}  "
              f"global {rec['ll_global']:.4f}  hier {rec['ll_hier']:.4f}  "
              f"ap {rec['ap_global']:.3f}->{rec['ap_hier']:.3f}", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(args.out / "hier_transfer_kappa.csv", index=False)
    print("\n=== 三分类 log loss（15 折均值）===")
    for k in ("discard", "global", "hier0", "hier"):
        print(f"  {k:<9}{t[f'll_{k}'].mean():.5f}")
    print("\n=== tie AP ===")
    for k in ("discard", "global", "hier"):
        print(f"  {k:<9}{t[f'ap_{k}'].mean():.4f}")
    print("\n=== 配对检验 ===")
    for k in ("global", "hier"):
        dlt = t[f"ll_{k}"] - t["ll_discard"]
        print(f"  {k} vs discard: {dlt.mean():+.5f}  p={stats.wilcoxon(dlt).pvalue:.4f}"
              f"  更好 {int((dlt < 0).sum())}/15")
    dlt = t.ll_hier - t.ll_global
    print(f"  hier vs global: {dlt.mean():+.5f}  p={stats.wilcoxon(dlt).pvalue:.4f}"
          f"  更好 {int((dlt < 0).sum())}/15")
    print(f"\n  选中 κ: " + "  ".join(f"{k:g}×{int((t.kappa==k).sum())}"
                                      for k in args.kappas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
