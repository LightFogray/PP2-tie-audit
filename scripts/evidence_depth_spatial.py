"""证据深度的空间结构（投影版 + 尺度敏感性）。

修正
----
1) 邻居不再用经纬度欧氏距离，改为按城市中心纬度换算为局部米制平面：
   x = (lon - lon0) * cos(lat0) * 111320,  y = (lat - lat0) * 110540
2) 对 k = 4, 6, 8, 10 四个邻域尺度分别检验，报告 Moran's I 的分布，
   避免以单一尺度下结论。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy import sparse

DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")
OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\city")
RNG = np.random.RandomState(0)
KS = (4, 6, 8, 10)

df = pd.read_parquet(DATA / "pairs.parquet")
L = df[["left","left_city","left_lat","left_lon"]].rename(
    columns={"left":"img","left_city":"city","left_lat":"lat","left_lon":"lon"})
R = df[["right","right_city","right_lat","right_lon"]].rename(
    columns={"right":"img","right_city":"city","right_lat":"lat","right_lon":"lon"})
app = pd.concat([L,R], ignore_index=True).dropna(subset=["lat","lon"])
per = app.groupby("img").agg(city=("city","first"), lat=("lat","first"),
                             lon=("lon","first"), n=("city","size")).reset_index()
print(f"  图像 {len(per):,}   城市 {per.city.nunique()}", flush=True)


def local_metres(lat, lon):
    lat0 = np.deg2rad(lat.mean())
    x = (lon - lon.mean()) * np.cos(lat0) * 111320.0
    y = (lat - lat.mean()) * 110540.0
    return np.column_stack([x, y])


def moran(coords, vals, k, n_perm=999, rng=RNG):
    n = len(vals)
    if n < max(50, k + 5):
        return np.nan, np.nan
    _, idx = cKDTree(coords).query(coords, k=min(k + 1, n))
    idx = idx[:, 1:]
    rows = np.repeat(np.arange(n), idx.shape[1])
    W = sparse.coo_matrix((np.ones(len(rows)), (rows, idx.ravel())),
                          shape=(n, n)).tocsr()
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
    coords = local_metres(g.lat.values, g.lon.values)
    depth = g.n.values.astype(float)
    low = (g.n.values <= 3).astype(float)
    rec = dict(city=city, n_img=len(g), mean_depth=depth.mean(),
               low_share=low.mean())
    for k in KS:
        i1, p1 = moran(coords, depth, k)
        i2, p2 = moran(coords, low, k)
        rec[f"I_depth_k{k}"], rec[f"p_depth_k{k}"] = i1, p1
        rec[f"I_low_k{k}"], rec[f"p_low_k{k}"] = i2, p2
    rows.append(rec)
    print(f"  {city:<18}{len(g):>7} I6={rec['I_depth_k6']:+.3f}", flush=True)

t = pd.DataFrame(rows)
t.to_csv(OUT / "evidence_depth_spatial.csv", index=False)

print(f"\n  城市 {len(t)}")
print(f"  {'k':<4}{'I_depth 均值':>14}{'显著数':>8}{'I_low 均值':>14}{'显著数':>8}")
for k in KS:
    print(f"  {k:<4}{t[f'I_depth_k{k}'].mean():>+14.4f}"
          f"{int((t[f'p_depth_k{k}']<0.05).sum()):>8}"
          f"{t[f'I_low_k{k}'].mean():>+14.4f}"
          f"{int((t[f'p_low_k{k}']<0.05).sum()):>8}")
print(f"\n  低深度占比 {t.low_share.mean():.3f}  "
      f"范围 {t.low_share.min():.3f}–{t.low_share.max():.3f}")
