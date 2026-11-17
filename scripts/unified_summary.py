"""从 unified_votes.parquet 派生表 4 与表 5（同一批预测）。"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\rater\clean")
v = pd.read_parquet(OUT / "unified_votes.parquet")
y = v.y.values

def ll(P, y):
    p = np.clip(P[np.arange(len(y)), y], 1e-12, 1)
    return float(-np.log(p).mean())

def auc_p(t):
    from scipy.stats import rankdata
    n1 = int(t.sum()); n0 = len(t) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(t)
    return float((r[t == 1].sum() - n1*(n1+1)/2)/(n1*n0))

def ap(score, label):
    o = np.argsort(-score, kind="stable"); s, l = score[o], label[o]
    tp = np.cumsum(l); prec = tp/np.arange(1, len(l)+1)
    return float((prec*l).sum()/l.sum()) if l.sum() else np.nan

models = {"discard": ("pd0","pd1","pd2"), "global": ("pg0","pg1","pg2"),
          "hier": ("ph0","ph1","ph2")}
res = {}
for k,(a,b,c) in models.items():
    P = np.column_stack([v[a], v[b], v[c]])
    res[k] = dict(ll=ll(P, y), ap=ap(P[:,1], (y==1).astype(int)),
                  auc=auc_p((y==1).astype(int)) if False else None)

print("=== 表 4：主表（同一批逐票预测）===")
print(f"  {'model':<10}{'log loss':>10}{'tie AP':>9}")
for k,r in res.items():
    print(f"  {k:<10}{r['ll']:>10.5f}{r['ap']:>9.4f}")

# 逐折配对（按 fold 汇总）
print("\n=== 逐折配对检验 ===")
rows=[]
for f,g in v.groupby("fold"):
    yy=g.y.values
    d={}
    for k,(a,b,c) in models.items():
        P=np.column_stack([g[a],g[b],g[c]])
        d[k]=ll(P, yy)
    rows.append(dict(city=f, **d))
t=pd.DataFrame(rows)
for a_,b_ in (("global","discard"),("hier","discard"),("hier","global")):
    dlt=t[a_]-t[b_]
    print(f"  {a_} vs {b_}: {dlt.mean():+.5f}  p={stats.wilcoxon(dlt).pvalue:.2g}  "
          f"更好 {int((dlt<0).sum())}/{len(t)}")

print("\n=== 表 5：按训练城历史长度分层（与表 4 同源）===")
bins=[(-1,0,"0"),(0,5,"1-5"),(5,15,"6-15"),(15,40,"16-40"),(40,10**9,">40")]
print(f"  {'history':<9}{'votes':>8}{'discard':>10}{'global':>10}{'hier':>10}{'gain':>9}")
for lo,hi,lab in bins:
    m=(v.n_train>lo)&(v.n_train<=hi)
    if m.sum()<50: continue
    sub=v[m]; yy=sub.y.values
    vals={}
    for k,(a,b,c) in models.items():
        P=np.column_stack([sub[a],sub[b],sub[c]])
        vals[k]=float(-np.log(np.clip(P[np.arange(len(yy)),yy],1e-12,1)).mean())
    print(f"  {lab:<9}{int(m.sum()):>8}{vals['discard']:>10.4f}"
          f"{vals['global']:>10.4f}{vals['hier']:>10.4f}"
          f"{vals['global']-vals['hier']:>+9.4f}")
print(f"\n  n_train 中位 {v.n_train.median():.0f}  均值 {v.n_train.mean():.1f}")
print(f"  无训练历史票数 {int((v.n_train==0).sum())} ({100*(v.n_train==0).mean():.1f}%)")
