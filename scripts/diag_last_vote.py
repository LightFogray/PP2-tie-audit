"""判决性检验：图像是否在「第一次平局」时被停用？

若平台代码里的规则生效（平局 → 票数少的那张被标记 inactive），
那么每张图的投票序列应当**以平局结尾**。
因此：最后一次投票的平局率应显著高于同图其余投票。

这是最直接的检验，不依赖任何模型假设。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\data")
OUT = Path(r"D:\mycode\crime-temporal-spatio\STREETVIEW-PP2\outputs\diag")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA / "pairs.parquet")
    df["_tie"] = df.choice.eq("equal").astype(int)
    df["t"] = pd.to_datetime(df.timestamp, errors="coerce")
    df = df.dropna(subset=["t"])

    long = []
    for side in ("left", "right"):
        long.append(pd.DataFrame({"img": df[side].values,
                                  "tie": df._tie.values,
                                  "t": df.t.values}))
    long = pd.concat(long, ignore_index=True).sort_values(["img", "t"])
    long["pos_from_end"] = long.groupby("img").cumcount(ascending=False)

    per_img = long.groupby("img").agg(n=("tie", "size"), ties=("tie", "sum"))
    print(f"  图像 {len(per_img):,}   投票 {len(long):,}")

    print("\n=== 判决检验：最后一次投票的平局率 ===")
    last = long[long.pos_from_end == 0]
    prev = long[long.pos_from_end > 0]
    print(f"  最后一次投票  n={len(last):>7}  平局率 {last.tie.mean():.4f}")
    print(f"  其余投票      n={len(prev):>7}  平局率 {prev.tie.mean():.4f}")
    print(f"  比值 {last.tie.mean()/prev.tie.mean():.2f}×")

    # 只在票数>=3 的图上比，排除小样本
    big = per_img[per_img.n >= 3].index
    lb = last[last.img.isin(big)]
    pb = prev[prev.img.isin(big)]
    print(f"\n  仅限票数≥3 的图（{len(big):,} 张）:")
    print(f"    最后一次 {lb.tie.mean():.4f}   其余 {pb.tie.mean():.4f}   "
          f"比值 {lb.tie.mean()/pb.tie.mean():.2f}×")

    # 按 n 分层，看平局率随位置的变化
    print("\n=== 按票数分层：末位 vs 其余 ===")
    print(f"  {'n':>4}{'图数':>8}{'末位平局率':>12}{'其余平局率':>12}{'差':>9}")
    for n in range(2, 11):
        ids = per_img[per_img.n == n].index
        if len(ids) < 200:
            continue
        l = last[last.img.isin(ids)]
        p = prev[prev.img.isin(ids)]
        print(f"  {n:>4}{len(ids):>8}{l.tie.mean():>12.4f}"
              f"{p.tie.mean():>12.4f}{l.tie.mean()-p.tie.mean():>+9.4f}")
    ids = per_img[per_img.n > 10].index
    if len(ids) >= 200:
        l = last[last.img.isin(ids)]
        p = prev[prev.img.isin(ids)]
        print(f"  >10{len(ids):>8}{l.tie.mean():>12.4f}"
              f"{p.tie.mean():>12.4f}{l.tie.mean()-p.tie.mean():>+9.4f}")

    # 若「末位必为平局」，则末位平局率应接近 1；
    # 若只是「平局后不再出现」，末位平局率应在各层都显著高于其余
    print("\n=== 若停用生效的预期 ===")
    print("  末位平局率应显著高于其余；且随 n 增大，差值应缩小（因为平局概率固定）")

    long.to_csv(args.out / "vote_sequence.csv", index=False)
    per_img.to_csv(args.out / "per_image_counts.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
