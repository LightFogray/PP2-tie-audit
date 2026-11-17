"""两项防御性检验。

检验 1：平局阈值是不是「噪声」换个名字？
----------------------------------------
Thurstonian 模型里，个体的判别标准差 σ_i 会机械地产生个体平局率差异
（P(tie)=P(|D|<τ)，D 的方差含 σ_i）。若如此，我们的「阈值」只是 σ_i 的改写。

判据：若平局倾向由 σ_i 驱动，那么平局率高的人，其**二元选择也应更不可预测**。
做法：对每个投票者，用图像模型给其非平局票打分（模型赋予实际所选方向的概率），
得到「一致性」指标，再看它与平局率的相关。
  - 相关接近 0  → 阈值与二元噪声可分离 → 反驳成立
  - 显著负相关 → 二者同源 → 必须承认是 σ_i 的改写

检验 2：AUC 0.81 是否来自泄漏？
-------------------------------
γ 由投票者在**其他城市**的投票估得，与测试城的票不相交，本无泄漏。
但还需证明「信息确实来自历史」：按投票者的城外历史票数分层看 AUC。
  - 冷投票者（无历史，γ=0）的 AUC 应 ≈ 0.5
  - 若冷投票者仍高，说明信息来自别处（可疑）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.stats import rankdata

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from converge_heads import Head

DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")
OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\rater")
LAB = {"left": 0, "equal": 1, "right": 2}


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", type=Path,
                    default=DATA / "features_convnext_tiny.npz")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet(DATA / "pairs.parquet")
    df["_tie"] = df.choice.eq("equal").astype(int)
    folds = json.loads((DATA / "splits" / "logo_folds.json").read_text())
    z = np.load(args.features)
    idx = {k: i for i, k in enumerate(z["ids"].tolist())}
    X = torch.from_numpy(z["features"].astype(np.float32)).to(device)
    df = df[df["left"].isin(idx) & df["right"].isin(idx)].reset_index(drop=True)
    df["_ia"] = df["left"].map(idx).astype(np.int32)
    df["_ib"] = df["right"].map(idx).astype(np.int32)
    df["_y"] = df["choice"].map(LAB).astype(np.int8)
    p0 = float(df._tie.mean())

    per_rater, strat = [], []
    for f in folds:
        c, v = f["test_city"], f["val_city"]
        in_t = (df["left_city"] == c) | (df["right_city"] == c)
        in_v = (df["left_city"] == v) | (df["right_city"] == v)
        tr = df[(~in_t) & (~in_v)]
        te = df[in_t]

        agg = df[~in_t].groupby("voter_uniqueid")["_tie"].agg(["sum", "count"])
        rate = (agg["sum"] + args.alpha * p0) / (agg["count"] + args.alpha)
        rr = np.clip(rate, 1e-4, 1 - 1e-4)
        lg = np.log(rr / (1 - rr))
        lg_c = float(np.average(lg.values, weights=agg["count"].values.astype(float)))
        gmap = (lg - lg_c).to_dict()

        torch.manual_seed(args.seed)
        model = Head(dim=int(X.shape[1]), mode="diff").to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        ia = torch.tensor(tr["_ia"].values, device=device)
        ib = torch.tensor(tr["_ib"].values, device=device)
        yy = torch.tensor(tr["_y"].values, dtype=torch.long, device=device)
        for _ in range(args.epochs):
            model.train()
            perm = torch.randperm(len(ia), device=device)
            for k in range(0, len(ia), 8192):
                j = perm[k:k + 8192]
                loss = F.cross_entropy(model(X[ia[j]], X[ib[j]]), yy[j])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            l = model(X[torch.tensor(te["_ia"].values, device=device)],
                      X[torch.tensor(te["_ib"].values, device=device)]).cpu().numpy()
        e = np.exp(l - l.max(1, keepdims=True))
        P = e / e.sum(1, keepdims=True)

        te = te.copy()
        te["p_tie"] = P[:, 1]
        te["p_obs"] = P[np.arange(len(te)), te["_y"].values]
        te["gamma"] = te["voter_uniqueid"].map(gmap).fillna(0.0)
        te["n_out"] = te["voter_uniqueid"].map(agg["count"]).fillna(0).astype(int)

        # 检验 1：每个投票者的平局率 vs 二元选择一致性
        nt = te[te._y != 1]
        g = nt.groupby("voter_uniqueid").agg(
            n=("p_obs", "size"),
            consist=("p_obs", lambda s: float(np.mean(np.log(np.clip(s, 1e-6, 1))))),
        )
        g2 = te.groupby("voter_uniqueid")["_tie"].mean().rename("tie_rate")
        g3 = g.join(g2).dropna()
        g3["city"] = c
        per_rater.append(g3[g3.n >= 3])
        print(f"  {c:<16} 投票者 {len(g3)}  "
              f"相关 {stats.spearmanr(g3.tie_rate, g3.consist).statistic:+.3f}",
              flush=True)

        # 检验 2：按城外历史票数分层看 AUC
        for lo, hi in ((0, 0), (1, 5), (6, 15), (16, 40), (41, 10 ** 9)):
            m = (te.n_out >= lo) & (te.n_out <= hi)
            if m.sum() < 100:
                continue
            strat.append(dict(city=c, stratum=f"{lo}-{hi}", n=int(m.sum()),
                              auc_rater=auc(te._tie.values[m], te.p_tie.values[m]),
                              auc_gamma=auc(te._tie.values[m],
                                            rankdata(te.gamma.values[m])),
                              cold=bool(lo == 0)))

    # ---------------- 检验 1 汇总 ----------------
    r = pd.concat(per_rater)
    print(f"\n=== 检验 1：平局倾向 vs 二元选择不一致性（{len(r):,} 投票者）===")
    rho = stats.spearmanr(r.tie_rate, r.consist)
    # 控制票数后的偏相关
    def partial(x, y, z):
        rx = stats.rankdata(x); ry = stats.rankdata(y); rz = stats.rankdata(z)
        def resid(a, b):
            b1 = np.polyfit(b, a, 1)
            return a - np.polyval(b1, b)
        return np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1]
    pr = partial(r.tie_rate.values, r.consist.values, r.n.values.astype(float))
    print(f"  平局率 与 log P(实际选择) 的 Spearman ρ = {rho.statistic:+.4f}  "
          f"p={rho.pvalue:.3e}")
    print(f"  控制票数后的偏相关 = {pr:+.4f}")
    print(f"  解释：ρ≈0 表示阈值与二元噪声可分离；显著负相关表示二者同源")

    t1 = pd.DataFrame(dict(tie_rate=r.tie_rate, consist=r.consist, n=r.n))
    t1.to_csv(args.out / "threshold_vs_noise.csv", index=False)

    # ---------------- 检验 2 汇总 ----------------
    s = pd.DataFrame(strat)
    print(f"\n=== 检验 2：按城外历史票数分层的 AUC ===")
    print(f"  {'n_out':<10}{'票数':>9}{'AUC(图像p_tie)':>16}{'AUC(γ)':>12}")
    for name in ("0-0", "1-5", "6-15", "16-40", "41-1000000000"):
        sub = s[s.stratum == name]
        if len(sub) == 0:
            continue
        print(f"  {name:<10}{sub.n.sum():>9}{sub.auc_rater.mean():>16.4f}"
              f"{sub.auc_gamma.mean():>12.4f}")
    s.to_csv(args.out / "auc_by_history.csv", index=False)

    cold = s[s.cold]
    if len(cold):
        print(f"\n  冷投票者（无历史）AUC(γ) = {cold.auc_gamma.mean():.4f}"
              f"  （γ 恒为 0，按定义应为 0.5）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
