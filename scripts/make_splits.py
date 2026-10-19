#!/usr/bin/env python3
"""生成 leave-maps-out 划分并冻结为 data/splits.json。

用法：
    python scripts/make_splits.py --data data/radiomapseer --out data/splits.json \
        --n-test 10 --n-val 5 --seed 20260101

为什么按地图切：按样本随机切会让同一张地图的建筑布局同时出现在训练与测试中，
模型只需记住布局再换一个 Tx 位置插值，测出来的指标不代表任何泛化能力。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/radiomapseer")
    ap.add_argument("--out", default="data/splits.json")
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--n-val", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--tx-per-map", type=int, default=80)
    args = ap.parse_args()

    bdir = Path(args.data) / "buildings_complete"
    maps = sorted(int(p.stem) for p in bdir.glob("*.png"))
    if not maps:
        raise SystemExit(f"{bdir} 下没有找到地图")
    if args.n_test + args.n_val >= len(maps):
        raise SystemExit(f"test({args.n_test}) + val({args.n_val}) 必须小于地图总数 {len(maps)}")

    rng = np.random.RandomState(args.seed)
    order = list(maps)
    rng.shuffle(order)

    test_maps = sorted(order[: args.n_test])
    val_maps = sorted(order[args.n_test : args.n_test + args.n_val])
    train_maps = sorted(order[args.n_test + args.n_val :])

    splits = {
        "seed": args.seed,
        "split_unit": "map",          # 关键：划分单位是地图，不是样本
        "n_tx_per_map": args.tx_per_map,
        "all_maps": maps,
        "train_maps": train_maps,
        "val_maps": val_maps,
        "test_maps": test_maps,
        "counts": {
            "train_samples": len(train_maps) * args.tx_per_map,
            "val_samples": len(val_maps) * args.tx_per_map,
            "test_samples": len(test_maps) * args.tx_per_map,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(f"写入 {out}")
    print(f"  train {len(train_maps)} maps ({splits['counts']['train_samples']} 样本)")
    print(f"  val   {len(val_maps)} maps ({splits['counts']['val_samples']} 样本)")
    print(f"  test  {len(test_maps)} maps ({splits['counts']['test_samples']} 样本)")
    print("  冻结后禁止修改；如需改动请换文件名并记录在 runs/*/config.yaml 里。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
