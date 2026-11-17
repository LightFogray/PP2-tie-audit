"""全文数字复核：把准备引用的每个数从原始数据或已存结果重算一遍。

原则
----
1. 能从原始数据重算的，一律重算，不信任中间文件
2. 每个数与预期比对，不一致就标红
3. 输出一张可直接放进论文的核对表
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")
OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs")

CHECKS: list[tuple[str, str, object, object, bool]] = []


def chk(name, got, expect, tol=0.02, note=""):
    ok = expect is None or (abs(float(got) - float(expect)) <= tol
                            if isinstance(expect, (int, float)) else got == expect)
    CHECKS.append((name, f"{got:.4f}" if isinstance(got, float) else str(got),
                   str(expect), ok, note))
    return got


def main() -> int:
    df = pd.read_parquet(DATA / "pairs.parquet")
    df["_tie"] = df.choice.eq("equal").astype(int)
    df["t"] = pd.to_datetime(df.timestamp, errors="coerce")
    df["q"] = df.t.dt.to_period("Q").astype(str)
    p0 = float(df._tie.mean())

    print("=" * 74)
    print("A. 数据集基本量与溯源证据")
    print("=" * 74)
    n_vote = len(df)
    imgs = pd.unique(pd.concat([df.left, df.right]))
    n_voter = df.voter_uniqueid.nunique()
    chk("投票数", n_vote, 506165, 5)
    chk("图像数", len(imgs), 111389, 5)
    chk("投票者数", n_voter, 48400, 600, "不同筛选下略有差异")
    chk("总体平局率", p0, 0.1247, 0.001)
    print(f"  投票 {n_vote:,}  图像 {len(imgs):,}  投票者 {n_voter:,}  "
          f"平局率 {p0:.4f}")

    long = []
    for s in ("left", "right"):
        long.append(pd.DataFrame({"img": df[s].values, "tie": df._tie.values,
                                  "t": df.t.values}))
    L = pd.concat(long, ignore_index=True).sort_values(["img", "t"])
    L["pos_end"] = L.groupby("img").cumcount(ascending=False)
    last = L[L.pos_end == 0]
    rest = L[L.pos_end > 0]
    chk("末位投票平局率", float(last.tie.mean()), 0.356, 0.01)
    chk("其余投票平局率", float(rest.tie.mean()), 0.096, 0.01)
    print(f"  末位平局率 {last.tie.mean():.4f}  其余 {rest.tie.mean():.4f}  "
          f"比值 {last.tie.mean()/rest.tie.mean():.2f}×")

    g = L.groupby("img").agg(n=("tie", "size"), ties=("tie", "sum"))
    s2 = g[g.n == 2]
    chk("2票图中恰1次平局占比", float((s2.ties == 1).mean()), 0.888, 0.01)
    chk("2票图中0次平局占比", float((s2.ties == 0).mean()), 0.002, 0.005)
    print(f"  2票图 {len(s2):,}: 0次平局 {(s2.ties==0).mean()*100:.1f}%  "
          f"1次 {(s2.ties==1).mean()*100:.1f}%")

    print("\n" + "=" * 74)
    print("B. 投票者平局倾向")
    print("=" * 74)

    def icc(p, n):
        w = n / n.sum(); pb = float(np.sum(w * p))
        b = float(np.sum(w * (p - pb) ** 2)); t = pb * (1 - pb)
        return b / t if t > 0 else np.nan

    t = df.groupby("voter_uniqueid")["_tie"].agg(["mean", "count"])
    t6 = t[t["count"] >= 6]
    chk("平局 ICC（n≥6）", icc(t6["mean"].values, t6["count"].values), 0.263, 0.02)
    print(f"  平局 ICC（n≥6）{icc(t6['mean'].values, t6['count'].values):.4f}")

    nt = df[df.choice.ne("equal")]
    t2 = nt.groupby("voter_uniqueid").apply(
        lambda x: (x.choice.eq("right").mean(), len(x)), include_groups=False)
    t2 = pd.DataFrame(t2.tolist(), index=t2.index, columns=["r", "n"])
    t2 = t2[t2.n >= 6]
    chk("左右偏好 ICC", icc(t2.r.values, t2.n.values), 0.086, 0.03)
    print(f"  左右偏好 ICC {icc(t2.r.values, t2.n.values):.4f}")

    ics, ws = [], []
    for q, gg in df.groupby("q"):
        if len(gg) < 3000:
            continue
        tt = gg.groupby("voter_uniqueid")["_tie"].agg(["mean", "count"])
        tt = tt[tt["count"] >= 3]
        if len(tt) < 200:
            continue
        ics.append(icc(tt["mean"].values, tt["count"].values))
        ws.append(tt["count"].sum())
    chk("季度内 ICC", float(np.average(ics, weights=ws)), 0.263, 0.02)
    print(f"  季度内 ICC {np.average(ics, weights=ws):.4f}（{len(ics)} 个季度）")

    p = OUT / "rater" / "tie_sources.csv"
    if p.exists():
        s = pd.read_csv(p)
        chk("AUC 投票者", float(s.auc_rater.mean()), 0.812, 0.02)
        chk("AUC 图像", float(s.auc_image.mean()), 0.532, 0.02)
        chk("AUC 配对构成", float(s.auc_samecity.mean()), 0.502, 0.02)
        print(f"  AUC 投票者 {s.auc_rater.mean():.3f}  图像 {s.auc_image.mean():.3f}"
              f"  配对 {s.auc_samecity.mean():.3f}")

    p = OUT / "rater" / "cold_start_votes.csv"
    if p.exists():
        c = pd.read_csv(p)
        dg = c.groupby("city").apply(
            lambda x: x.ll_hier.mean() - x.ll_global.mean(), include_groups=False)
        rel = abs(dg.mean()) / c.groupby("city").ll_global.mean().mean() * 100
        chk("层级模型相对改善 %", float(rel), 4.71, 0.3)
        chk("层级模型更好折数", int((dg < 0).sum()), 15, 0)
        print(f"  层级模型 相对 {rel:.2f}%  更好 {int((dg<0).sum())}/15  "
              f"p={stats.wilcoxon(dg).pvalue:.6f}")
        c["n_out"] = c.n_out.fillna(0)
        for lo, hi, nm in ((0, 0, "冷"), (1, 5, "1-5"), (6, 15, "6-15"),
                           (16, 40, "16-40"), (41, 10**9, ">40")):
            m = (c.n_out >= lo) & (c.n_out <= hi)
            if m.sum() > 50:
                print(f"    n_out {nm:<6} {int(m.sum()):>6} 票  "
                      f"差 {(c.ll_hier[m]-c.ll_global[m]).mean():+.4f}")

    p = OUT / "rater" / "threshold_vs_noise.csv"
    if p.exists():
        r = pd.read_csv(p)
        rho = stats.spearmanr(r.tie_rate, r.consist)
        chk("阈值vs噪声 rho", float(rho.statistic), 0.006, 0.03)
        print(f"  阈值 vs 二元噪声 ρ = {rho.statistic:+.4f}（p={rho.pvalue:.3f}）")

    p = OUT / "rater" / "rater_transfer.csv"
    if p.exists():
        r = pd.read_csv(p)
        chk("跨城迁移 r", float(r.r_tau_outside_in.mean()), 0.510, 0.03)
        print(f"  跨城迁移 r = {r.r_tau_outside_in.mean():+.4f}  "
              f"为正 {int((r.r_tau_outside_in>0).sum())}/{len(r)}")

    p = OUT / "city" / "six_attributes.csv"
    if p.exists():
        r = pd.read_csv(p)
        chk("六属性 ICC 下界", float(r.icc.min()), 0.234, 0.02)
        chk("六属性 ICC 上界", float(r.icc.max()), 0.277, 0.02)
        chk("六属性正相关数", int((r.r_sim > 0).sum()), 6, 0)
        print(f"  六属性 ICC {r.icc.min():.3f}–{r.icc.max():.3f}  "
              f"正相关 {int((r.r_sim>0).sum())}/6  "
              f"显著 {int((r.p_sim<0.05).sum())}/6")

    print("\n" + "=" * 74)
    print("C. 城市 WVR")
    print("=" * 74)
    p = OUT / "city" / "city_rescue_base.csv"
    if p.exists():
        b = pd.read_csv(p)
        def wr(x, y, w):
            mx = np.average(x, weights=w); my = np.average(y, weights=w)
            cx = np.average((x-mx)**2, weights=w); cy = np.average((y-my)**2, weights=w)
            return np.average((x-mx)*(y-my), weights=w)/np.sqrt(cx*cy)
        r = wr(b.sim.values, b.tie.values, b.n.values.astype(float))
        chk("WVR 基准 r", float(r), 0.508, 0.03)
        print(f"  基准 r = {r:+.4f}（{len(b)} 城）")
    p = OUT / "city" / "city_rescue_first.csv"
    if p.exists():
        f = pd.read_csv(p)
        r = stats.pearsonr(f.sim, f.tie_adj)
        chk("WVR 剪枝无关 r", float(r.statistic), 0.365, 0.05)
        print(f"  剪枝无关（首次投票+控季度）r = {r.statistic:+.4f}")

    print("\n" + "=" * 74)
    print("核对表")
    print("=" * 74)
    print(f"  {'指标':<28}{'实测':>12}{'预期':>12}{'':>4}")
    bad = 0
    for name, got, exp, ok, note in CHECKS:
        flag = "OK" if ok else "**不符**"
        if not ok:
            bad += 1
        print(f"  {name:<28}{got:>12}{exp:>12}   {flag}")
    print(f"\n  {len(CHECKS)} 项检查，{bad} 项不符")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
