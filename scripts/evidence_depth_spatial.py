"""证据深度的空间结构（稀疏矩阵版）。

问题：被退役规则截断证据的图像，是否在空间上成簇？
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy import sparse

DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")
OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\city")
RNG = np.random.RandomState(0)

df = pd.read_parquet(DATA / "pairs.parquet")
L = df[["left","left_city","left_lat","left_lon"]].rename(
    columns={"left":"img","left_city":"city","left_lat":"lat","left_lon":"lon"})
R = df[["right","right_city","right_lat","right_lon"]].rename(
    columns={"right":"img","right_city":"city","right_lat":"lat","right_lon":"lon"})
app = pd.concat([L,R], ignore_index=True).dropna(subset=["lat","lon"])
per = app.groupby("img").agg(city=("city","first"), lat=("lat","first"),
                             lon=("lon","first"), n=("city","size")).reset_index()
print(f"  图像 {len(per):,}   城市 {per.city.nunique()}", flush=True)


def moran_sparse(coords, vals, k=6, n_perm=999, rng=RNG):
    n = len(vals)
    if n < 50:
        return np.nan, np.nan
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=min(k+1, n))
    idx = idx[:, 1:]
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.ravel()
    W = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    W = (W + W.T) * 0.5
    rs = np.asarray(W.sum(1)).ravel(); rs[rs == 0] = 1
    W = sparse.diags(1.0/rs) @ W
    S0 = W.sum()
    x = vals - vals.mean()
    den = (x*x).sum()
    if den == 0:
        return np.nan, np.nan
    I = (n/S0) * float(x @ (W @ x)) / den
    null = np.empty(n_perm)
    for i in range(n_perm):
        xp = rng.permutation(x)
        null[i] = (n/S0) * float(xp @ (W @ xp)) / den
    return float(I), float((np.abs(null) >= abs(I)).mean())


rows = []
for city, g in per.groupby("city"):
    if len(g) < 100:
        continue
    coords = np.column_stack([g.lat.values, g.lon.values])
    coords = coords - coords.mean(0)
    depth = g.n.values.astype(float)
    low = (g.n.values <= 3).astype(float)
    I_d, p_d = moran_sparse(coords, depth)
    I_l, p_l = moran_sparse(coords, low)
    rows.append(dict(city=city, n_img=len(g), mean_depth=depth.mean(),
                     low_share=low.mean(), moran_depth=I_d, p_depth=p_d,
                     moran_low=I_l, p_low=p_l))
    print(f"  {city:<18}{len(g):>7} I_depth={I_d:+.3f} p={p_d:.3f}", flush=True)

t = pd.DataFrame(rows).sort_values("moran_depth", ascending=False)
t.to_csv(OUT / "evidence_depth_spatial.csv", index=False)
print(f"\n  城市 {len(t)}")
print(f"  深度 Moran's I 均值 {t.moran_depth.mean():+.4f}  "
      f"显著 {int((t.p_depth<0.05).sum())}/{len(t)}")
print(f"  低深度 Moran's I 均值 {t.moran_low.mean():+.4f}  "
      f"显著 {int((t.p_low<0.05).sum())}/{len(t)}")
print(f"  低深度占比 均值 {t.low_share.mean():.3f}  "
      f"范围 {t.low_share.min():.3f}–{t.low_share.max():.3f}")
