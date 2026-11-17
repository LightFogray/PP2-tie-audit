"""平局到底由什么决定：投票者、配对构成、还是图像内容？

为什么必须查
------------
审计显示同城配对平局率 15.58%、跨城 12.38%，差 3.2 pp。
这说明「配对是否同城」携带信息，与「平局纯属投票者」有张力。
必须把三种来源分开量化，否则论文的核心命题会被一条简单的对照推翻。

设计
----
对每条比较预测 tie/非 tie，比较：
  global      全局平局率
  rater       投票者的 γ（由其在**其他城市**的投票估得，中心化）
  samecity    配对是否同城
  rater+same  两者
  image       图像特征模型（三分类头的 p_tie）
  全部         三者

所有预测子都在**训练城**上拟合/估计，在测试城上评估，无泄漏。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from converge_heads import Head

DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")
OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\rater")
LAB = {"left": 0, "equal": 1, "right": 2}


def auc(y, p):
    """必须用平均秩处理并列值，否则恒定预测子会得到随机 AUC。"""
    from scipy.stats import rankdata
    y, p = np.asarray(y), np.asarray(p)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logit(X, y, l2=1.0, iters=300):
    Xt = torch.tensor(np.concatenate([np.ones((len(X), 1)), X], 1),
                      dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(Xt.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        p = torch.sigmoid(Xt @ w)
        loss = F.binary_cross_entropy(p, yt) + l2 * (w[1:] ** 2).sum() / len(X)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy()


def p_logit(w, X):
    return 1 / (1 + np.exp(-np.concatenate([np.ones((len(X), 1)), X], 1) @ w))


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
    logit_p0 = np.log(p0 / (1 - p0))

    rows = []
    for f in folds:
        c, v = f["test_city"], f["val_city"]
        in_t = (df["left_city"] == c) | (df["right_city"] == c)
        in_v = (df["left_city"] == v) | (df["right_city"] == v)
        tr = df[(~in_t) & (~in_v)]
        va = df[in_v & (~in_t)]
        te = df[in_t]

        agg = df[~in_t].groupby("voter_uniqueid")["_tie"].agg(["sum", "count"])
        rate = (agg["sum"] + args.alpha * p0) / (agg["count"] + args.alpha)
        rr = np.clip(rate, 1e-4, 1 - 1e-4)
        lg = np.log(rr / (1 - rr))
        lg_c = float(np.average(lg.values, weights=agg["count"].values.astype(float)))
        gmap = (lg - lg_c).to_dict()

        def g_of(sub):
            return sub["voter_uniqueid"].map(gmap).fillna(0.0).values

        # 图像模型
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

        def p_tie(d):
            with torch.no_grad():
                l = model(X[torch.tensor(d["_ia"].values, device=device)],
                          X[torch.tensor(d["_ib"].values, device=device)]).cpu().numpy()
            e = np.exp(l - l.max(1, keepdims=True))
            return (e / e.sum(1, keepdims=True))[:, 1]

        gv, gt = g_of(va), g_of(te)
        sv = va["same_city"].astype(float).values[:, None]
        st = te["same_city"].astype(float).values[:, None]
        pv, pt = p_tie(va), p_tie(te)
        eps = 1e-6
        zv = np.log(np.clip(pv, eps, 1 - eps) / (1 - np.clip(pv, eps, 1 - eps)))[:, None]
        zt = np.log(np.clip(pt, eps, 1 - eps) / (1 - np.clip(pt, eps, 1 - eps)))[:, None]
        yv, yt = va["_tie"].values, te["_tie"].values

        def fit_pred(train_X, ev_X):
            w = fit_logit(train_X, yv)
            return p_logit(w, ev_X)

        cand = {
            "global": np.full(len(yt), p0),
            "rater": 1 / (1 + np.exp(-np.log(p0 / (1 - p0)) - gt)),
            "samecity": fit_pred(sv, st),
            "rater+same": fit_pred(np.concatenate([gv[:, None], sv], 1),
                                   np.concatenate([gt[:, None], st], 1)),
            "image": pt,
            "all": fit_pred(np.concatenate([gv[:, None], sv, zv], 1),
                            np.concatenate([gt[:, None], st, zt], 1)),
        }
        rec = {"city": c, "n": len(te), "tie_rate": float(yt.mean())}
        for k, p in cand.items():
            rec[f"auc_{k}"] = auc(yt, p)
        rows.append(rec)
        print(f"  {c:<16} " + "  ".join(f"{k}={rec[f'auc_{k}']:.3f}"
                                        for k in cand), flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(args.out / "tie_sources.csv", index=False)
    print("\n=== 平局预测 AUC（15 折均值）===")
    for k in ("global", "image", "samecity", "rater", "rater+same", "all"):
        print(f"  {k:<12}{t[f'auc_{k}'].mean():.4f}")

    from scipy import stats
    print("\n=== 配对检验 ===")
    for a, b in (("rater", "image"), ("rater+same", "rater"), ("all", "rater+same")):
        d = t[f"auc_{a}"] - t[f"auc_{b}"]
        print(f"  {a} - {b}: {d.mean():+.4f}  p={stats.wilcoxon(d).pvalue:.4f}  "
              f"更好 {int((d > 0).sum())}/15")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
