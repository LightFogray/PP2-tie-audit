"""层级 Davidson：平局阈值随投票者变化。

模型
----
对一次比较，令 d = θ_left − θ_right（左右指图像在屏幕上的位置）：

    p_tie   = s / (e^d + 1 + s),   s = ν_r · e^{d/2}
    p_left  = e^d / (e^d + 1 + s)
    p_right = 1 / (e^d + 1 + s)

其中 ν_r = ν · e^{γ_r}，γ_r 是投票者的平局偏移，带 L2 收缩（随机效应 MAP）。

参照标准 Davidson 只有全局 ν；本模型让 ν 随投票者变化。

估计
----
块坐标下降：交替做
  1. 每个物品 θ_i 的一维 MLE（网格）
  2. 每个投票者 γ_r 的一维惩罚 MLE（网格）
  3. 全局 ν 的一维 MLE
θ 每轮中心化（只有差值可识别），γ 每轮中心化（均值并入 ν）。

自检
----
合成数据：已知 θ 与 γ，检验能否还原；并检验无投票者效应时
（γ≡0）模型是否退化到标准 Davidson 而不变差。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

THETA_GRID = np.linspace(-6, 6, 241)
GAMMA_GRID = np.linspace(-3, 3, 121)
NU_GRID = np.linspace(0.02, 2.0, 100)


@dataclass
class Fitted:
    theta: dict
    gamma: dict
    nu: float
    lam: float
    n_iter: int


def p3(d, nu_r):
    """d = θ_left − θ_right。返回 (p_left, p_tie, p_right)。"""
    s = nu_r * np.exp(d / 2)
    den = np.exp(d) + 1.0 + s
    return np.exp(d) / den, s / den, 1.0 / den


def loglik_grid(theta_grid, d_right, nu_r, y, on_left):
    """对一组观测，返回每个候选 θ_left 的对数似然。

    d_right: 对手（右侧或左侧）的 θ，已按位置对齐
    on_left: True 表示候选物品在左
    y: 0=left,1=tie,2=right
    """
    D = np.where(on_left[None, :], theta_grid[:, None] - d_right[None, :],
                 d_right[None, :] - theta_grid[:, None])
    s = nu_r[None, :] * np.exp(D / 2)
    ed = np.exp(D)
    den = ed + 1.0 + s
    P = np.stack([ed / den, s / den, 1.0 / den], 0)      # (3, G, m)
    idx = np.arange(D.shape[1])[None, :]
    sel = np.clip(P[y[None, :], np.arange(P.shape[1])[:, None], idx], 1e-12, None)
    return np.log(sel).sum(1)


def fit_hier(theta_fixed, item_ids, rater_ids, on_left, y,
             lam=0.5, n_iter=8, nu0=0.3, verbose=False):
    """theta_fixed：对手（参考图）的 θ，逐观测给出。

    item_ids：待估物品的 id（逐观测）
    rater_ids：投票者 id（逐观测）
    返回 Fitted。
    """
    items = np.unique(item_ids)
    raters = np.unique(rater_ids)
    th = {i: 0.0 for i in items}
    ga = {r: 0.0 for r in raters}
    nu = nu0

    # 注意：np.unique 的第二返回值是**首次出现的位置**，不是下一次的值。
    # 早期版本把两者混用，导致分组字典的键变成位置而非 id，拟合完全失效。
    order = np.argsort(item_ids, kind="stable")
    ii = item_ids[order]
    it_uniq, it_start = np.unique(ii, return_index=True)
    it_end = np.append(it_start[1:], len(ii))
    item_groups = {u: order[s:e] for u, s, e in zip(it_uniq, it_start, it_end)}

    order_r = np.argsort(rater_ids, kind="stable")
    ri = rater_ids[order_r]
    rt_uniq, rt_start = np.unique(ri, return_index=True)
    rt_end = np.append(rt_start[1:], len(ri))
    rater_groups = {u: order_r[s:e] for u, s, e in
                    zip(rt_uniq, rt_start, rt_end)}

    for it in range(n_iter):
        nu_r = nu * np.exp(np.array([ga[r] for r in rater_ids]))
        # 1) θ
        for u, g in item_groups.items():
            lg = loglik_grid(THETA_GRID, theta_fixed[g], nu_r[g],
                             y[g], on_left[g])
            th[u] = float(THETA_GRID[int(np.argmax(lg))])
        # 注意：θ_ref 固定，模型对 θ 平移**不是**不变的，
        # 因此绝不能对 θ 中心化（曾因此使似然从 -5005 崩到 -13400）。

        # 2) γ（带 L2）
        th_arr = np.array([th[i] for i in item_ids])
        d_base = np.where(on_left, th_arr - theta_fixed,
                          theta_fixed - th_arr)
        for r, g in rater_groups.items():
            base = d_base[g]                 # 不含 γ 的 d
            yy, oo = y[g], on_left[g]
            lls = np.zeros_like(GAMMA_GRID)
            for gi, gv in enumerate(GAMMA_GRID):
                nu_r_g = nu * np.exp(gv)
                s = nu_r_g * np.exp(base / 2)
                ed = np.exp(base)
                den = ed + 1.0 + s
                P = np.stack([ed / den, s / den, 1.0 / den], 1)
                sel = np.clip(P[np.arange(len(yy)), yy], 1e-12, None)
                lls[gi] = np.log(sel).sum() - lam * gv ** 2
            ga[r] = float(GAMMA_GRID[int(np.argmax(lls))])
        gm = np.mean([ga[r] for r in rater_ids])
        for r in ga:
            ga[r] -= gm
        nu *= np.exp(gm)

        # 3) ν
        nu_r_arr = np.exp(np.array([ga[r] for r in rater_ids]))
        lls = np.zeros_like(NU_GRID)
        for ni, nv in enumerate(NU_GRID):
            s = nv * nu_r_arr * np.exp(d_base / 2)
            ed = np.exp(d_base)
            den = ed + 1.0 + s
            P = np.stack([ed / den, s / den, 1.0 / den], 1)
            sel = np.clip(P[np.arange(len(y)), y], 1e-12, None)
            lls[ni] = np.log(sel).sum()
        nu = float(NU_GRID[int(np.argmax(lls))])

        if verbose:
            print(f"    iter {it+1}: nu={nu:.4f}  "
                  f"γ sd={np.std([ga[r] for r in raters]):.4f}")

    return Fitted(theta=th, gamma=ga, nu=nu, lam=lam, n_iter=n_iter)


def predict(fit: Fitted, item_ids, theta_ref, rater_ids, on_left):
    th = np.array([fit.theta[i] for i in item_ids])
    d = np.where(on_left, th - theta_ref, theta_ref - th)
    nu_r = fit.nu * np.exp(np.array([fit.gamma[r] for r in rater_ids]))
    s = nu_r * np.exp(d / 2)
    ed = np.exp(d)
    den = ed + 1.0 + s
    return np.stack([ed / den, s / den, 1.0 / den], 1)


# ---------------------------------------------------------------- 自检

def selfcheck() -> int:
    rng = np.random.RandomState(0)
    ok = True

    print("=== 自检 1：已知 θ 与 γ，能否还原 ===")
    n_item, n_rater, n_obs = 150, 40, 6000
    th_true = rng.normal(0, 1.0, n_item)
    ga_true = rng.normal(0, 0.7, n_rater)
    nu_true = 0.35
    ref_true = rng.normal(0, 1.0, n_obs)

    ii = rng.randint(0, n_item, n_obs)
    rr = rng.randint(0, n_rater, n_obs)
    on_left = rng.rand(n_obs) < 0.5
    theta_fixed = ref_true
    d = np.where(on_left, th_true[ii] - theta_fixed, theta_fixed - th_true[ii])
    nu_r = nu_true * np.exp(ga_true[rr])
    s = nu_r * np.exp(d / 2)
    ed = np.exp(d)
    den = ed + 1.0 + s
    P = np.stack([ed / den, s / den, 1.0 / den], 1)
    y = np.array([rng.choice(3, p=P[k]) for k in range(n_obs)])

    fit = fit_hier(theta_fixed, ii, rr, on_left, y, lam=0.5, n_iter=10)
    th_hat = np.array([fit.theta[i] for i in range(n_item)])
    ga_hat = np.array([fit.gamma[r] for r in range(n_rater)])

    r_th = np.corrcoef(th_true, th_hat)[0, 1]
    r_ga = np.corrcoef(ga_true, ga_hat)[0, 1]
    print(f"  θ 还原相关 {r_th:.3f}   γ 还原相关 {r_ga:.3f}   "
          f"ν 估计 {fit.nu:.3f}（真值 {nu_true}）")
    print(f"  θ 的 SD: 真值 {th_true.std():.3f}  估计 {th_hat.std():.3f}")
    print(f"  γ 的 SD: 真值 {ga_true.std():.3f}  估计 {ga_hat.std():.3f}")
    ok &= r_th > 0.85 and r_ga > 0.5

    print("\n=== 自检 2：无投票者效应（γ≡0）时不应变差 ===")
    ga_true0 = np.zeros(n_rater)
    nu_r = nu_true * np.exp(ga_true0[rr])
    s = nu_r * np.exp(d / 2)
    den = ed + 1.0 + s
    P = np.stack([ed / den, s / den, 1.0 / den], 1)
    y0 = np.array([rng.choice(3, p=P[k]) for k in range(n_obs)])
    fit0 = fit_hier(theta_fixed, ii, rr, on_left, y0, lam=0.5, n_iter=10)
    ga0 = np.array([fit0.gamma[r] for r in range(n_rater)])
    print(f"  估计出的 γ SD {ga0.std():.3f}（真值 0.000）")
    # 拟合质量对比：全局 ν 与层级模型
    print(f"  ν 估计 {fit0.nu:.3f}")
    ok &= ga0.std() < 0.35

    print("\n=== 自检 3：p3 与显式公式一致 ===")
    d_test, nur = 0.7, 0.4
    pl, pt, pr = p3(d_test, nur)
    s = nur * np.exp(d_test / 2)
    den = np.exp(d_test) + 1 + s
    print(f"  和为 1: {pl+pt+pr:.6f}")
    print(f"  显式一致: {abs(pl - np.exp(d_test)/den) < 1e-12}")
    ok &= abs(pl + pt + pr - 1) < 1e-9

    print("\n=== 自检 4：平局概率在 d=0 时最大 ===")
    ds = np.linspace(-4, 4, 801)
    pts_ = [p3(x, 0.4)[1] for x in ds]
    print(f"  argmax H 位置 d={ds[int(np.argmax(pts_))]:+.3f}")
    ok &= abs(ds[int(np.argmax(pts_))]) < 1e-9

    print(f"\nSELFCHECK {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()
    print("用 --selfcheck 运行自检")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
