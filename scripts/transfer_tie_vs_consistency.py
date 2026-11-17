"""判别检验：平局倾向与二元一致性是两个可分离的特质吗？

问题的严重性
------------
全样本（n≥3）ρ = +0.006，但限制到 n≥10 后 ρ = −0.094（p=0.01）。
负相关与「个体判别标准差 σ 同时产生高平局率与高二元噪声」的
Thurstonian 解释一致，因此单看这个相关无法区分：

  噪声假说    高 σ → 平局多 且 二元选择更不可预测
  无差异假说  偏好弱 → 平局多 且 二元选择更接近随机

两种都预测负相关。需要一个能分开它们的检验。

判别逻辑
--------
噪声参数 σ 是与刺激无关的个体属性，应当跨城稳定。
偏好强度则是相对于具体刺激分布的，换一批城市就可能变。

若二元一致性跨城迁移很弱、而平局倾向迁移很强，
则二者是不同特质——即使它们在个体层面弱相关。
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent
                      / "STREETVIEW-PP2" / "scripts"))
from converge_heads import Head

DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")
OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\rater")
LAB = {"left": 0, "equal": 1, "right": 2}


def wcorr(a, b, w):
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    ca = np.average((a - ma) ** 2, weights=w)
    cb = np.average((b - mb) ** 2, weights=w)
    cab = np.average((a - ma) * (b - mb), weights=w)
    return cab / np.sqrt(ca * cb) if ca > 0 and cb > 0 else np.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--min-votes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", type=Path,
                    default=DATA / "features_convnext_tiny.npz")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

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

    rows = []
    for f in folds:
        c, v = f["test_city"], f["val_city"]
        in_t = (df["left_city"] == c) | (df["right_city"] == c)
        in_v = (df["left_city"] == v) | (df["right_city"] == v)
        tr = df[(~in_t) & (~in_v)]

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

        def add_scores(sub):
            with torch.no_grad():
                lg = model(X[torch.tensor(sub["_ia"].values, device=device)],
                           X[torch.tensor(sub["_ib"].values, device=device)]
                           ).cpu().numpy()
            e = np.exp(lg - lg.max(1, keepdims=True))
            P = e / e.sum(1, keepdims=True)
            out = sub.copy()
            out["p_obs"] = P[np.arange(len(sub)), sub["_y"].values]
            return out

        outside = add_scores(df[~in_t])
        inside = add_scores(df[in_t])

        # 城外的平局率与二元一致性
        o_tie = outside.groupby("voter_uniqueid")["_tie"].agg(["mean", "count"])
        o_nt = outside[outside._y != 1].groupby("voter_uniqueid").agg(
            cons=("p_obs", lambda s: float(np.mean(np.log(np.clip(s, 1e-6, 1))))),
            n_nt=("p_obs", "size"))
        # 城内的
        i_tie = inside.groupby("voter_uniqueid")["_tie"].agg(["mean", "count"])
        i_nt = inside[inside._y != 1].groupby("voter_uniqueid").agg(
            cons=("p_obs", lambda s: float(np.mean(np.log(np.clip(s, 1e-6, 1))))),
            n_nt=("p_obs", "size"))

        j = o_tie.join(i_tie, lsuffix="_o", rsuffix="_i")
        j = j[(j["count_o"] >= args.min_votes) & (j["count_i"] >= 2)]
        jc = o_nt.join(i_nt, lsuffix="_o", rsuffix="_i")
        jc = jc[(jc["n_nt_o"] >= 4) & (jc["n_nt_i"] >= 2)]
        if len(j) < 50 or len(jc) < 50:
            continue

        r_tie = wcorr(j["mean_o"].values, j["mean_i"].values,
                      j["count_i"].values.astype(float))
        r_cons = wcorr(jc["cons_o"].values, jc["cons_i"].values,
                       jc["n_nt_i"].values.astype(float))
        rows.append(dict(city=c, n_voter=len(j), n_cons=len(jc),
                         r_tie=r_tie, r_cons=r_cons))
        print(f"  {c:<16} 平局迁移 {r_tie:+.3f}   一致性迁移 {r_cons:+.3f}",
              flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(args.out / "transfer_tie_vs_consistency.csv", index=False)
    print(f"\n=== 跨城迁移（{len(t)} 城）===")
    print(f"  平局倾向  均值 {t.r_tie.mean():+.4f}  中位 {t.r_tie.median():+.4f}  "
          f"为正 {int((t.r_tie > 0).sum())}/{len(t)}")
    print(f"  二元一致性 均值 {t.r_cons.mean():+.4f}  中位 {t.r_cons.median():+.4f}  "
          f"为正 {int((t.r_cons > 0).sum())}/{len(t)}")
    d = stats.wilcoxon(t.r_tie, t.r_cons)
    print(f"  差值 {t.r_tie.mean() - t.r_cons.mean():+.4f}  p={d.pvalue:.4f}")
    print("\n  解读：若一致性迁移也强，二者可能是同一特质的两种表现；")
    print("        若一致性迁移弱而平局迁移强，则它们是可分离的特质。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
