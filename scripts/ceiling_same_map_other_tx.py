#!/usr/bin/env python3
"""天花板参考：同一张地图内，用"另一个 Tx 的 pathloss"当预测，RMSE 是多少。

这不是模型的天花板，而是"地图内 Tx 之间的变化有多大"这个**参考量**：
如果两张图（同 map 不同 Tx）平均就差 X dB，那么一个只知道"这是哪张地图"、
不知道"是哪个 Tx"的预测器，最好也只能做到 ~X dB。它给 8.3 dB 一个量级参照，
但不参与任何"先验有没有用"的归因（那要靠 bigger-model 天花板）。

用法（零 GPU，只读数据）：
    python scripts/ceiling_same_map_other_tx.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from src.data.normalize import gain_to_pathloss_norm, pathloss_norm_to_db  # noqa: E402

DATA = ROOT / "data" / "radiomapseer"


def main() -> int:
    splits = json.loads((ROOT / "data" / "splits.json").read_text(encoding="utf-8"))
    test_maps = [int(m) for m in splits["test_maps"]]

    rmse_db = []
    rmse_db_nonsat = []
    n_pairs = 0
    for m in test_maps:
        ants = json.loads((DATA / "antenna" / f"{m}.json").read_text(encoding="utf-8"))
        # 预加载整张地图的所有 Tx 目标
        targets = []
        for t in range(len(ants)):
            g = np.array(Image.open(DATA / "gain" / "carsDPM" / f"{m}_{t}.png").convert("L"))
            targets.append(gain_to_pathloss_norm(g))
        targets = np.stack(targets)  # (T, H, W)
        T = targets.shape[0]
        for t in range(T):
            other = targets[(t + 1) % T]          # 同图另一个 Tx（无不动点）
            cur = targets[t]
            d = (pathloss_norm_to_db(cur) - pathloss_norm_to_db(other))
            rmse_db.append(float(np.sqrt((d ** 2).mean())))
            sat = cur >= 1.0 - 1e-7
            nsat = ~sat
            d_ns = d[nsat]
            rmse_db_nonsat.append(float(np.sqrt((d_ns ** 2).mean())))
            n_pairs += 1

    rmse_db = np.array(rmse_db)
    rmse_db_ns = np.array(rmse_db_nonsat)
    print(f"同图其它 Tx 天花板参考（test {len(test_maps)} 张地图，{n_pairs} 个配对）：")
    print(f"  RMSE = {rmse_db.mean():.2f} ± {rmse_db.std():.2f} dB")
    print(f"  RMSE 非饱和 = {rmse_db_ns.mean():.2f} ± {rmse_db_ns.std():.2f} dB")
    print("\n解读：这是'地图内 Tx 间变化'的量级，不是模型天花板。")
    print("若模型 RMSE 接近这个数，说明模型已逼近'只知道地图、不知道 Tx'的下限；")
    print("若模型明显好于它，说明模型用上了 Tx 位置信息（输入通道里确实有 Tx 坐标）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
