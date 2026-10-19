#!/usr/bin/env python3
"""校验 RadioMapSeer 资产是否完整、可用于训练。

用法：
    python scripts/verify_data.py --data data/radiomapseer

退出码非 0 = 校验失败，禁止进入后续阶段。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:  # Windows 控制台默认 GBK，统一按 UTF-8 输出（Linux 上本来就是 UTF-8）
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def _load_gray(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        return np.array(im.convert("L"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/radiomapseer")
    ap.add_argument("--maps", type=int, default=50)
    ap.add_argument("--tx-per-map", type=int, default=80)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    root = Path(args.data)
    bdir = root / "buildings_complete"
    gdir = root / "gain" / "carsDPM"
    adir = root / "antenna"

    errors: list[str] = []
    warnings: list[str] = []

    for d in (bdir, gdir, adir):
        if not d.is_dir():
            errors.append(f"缺少目录: {d}")
    if errors:
        print("\n".join(errors))
        return 1

    # 1) 建筑掩码：数量、尺寸、是否二值
    masks = sorted(bdir.glob("*.png"), key=lambda p: int(p.stem))
    if len(masks) != args.maps:
        errors.append(f"建筑掩码数量 {len(masks)} != {args.maps}")
    for p in masks:
        a = _load_gray(p)
        if a.shape != (args.size, args.size):
            errors.append(f"{p.name} 尺寸 {a.shape} != {(args.size, args.size)}")
        uniq = np.unique(a)
        if not np.all(np.isin(uniq, [0, 255])):
            warnings.append(f"{p.name} 掩码不是严格二值（前几个值 {uniq[:5].tolist()}），请确认二值化阈值")

    # 2) gain 与 antenna 一一对应
    missing_gain = 0
    for m in range(args.maps):
        jf = adir / f"{m}.json"
        if not jf.exists():
            errors.append(f"缺少 antenna/{m}.json")
            continue
        coords = json.loads(jf.read_text(encoding="utf-8"))
        if len(coords) != args.tx_per_map:
            errors.append(f"antenna/{m}.json 有 {len(coords)} 个 Tx != {args.tx_per_map}")
        for t in range(args.tx_per_map):
            if not (gdir / f"{m}_{t}.png").exists():
                missing_gain += 1
    if missing_gain:
        errors.append(f"gain/carsDPM 缺少 {missing_gain} 个文件")

    # 3) 抽样检查 gain 尺寸与非退化
    all_gain = sorted(gdir.glob("*.png"))
    for p in all_gain[:20]:
        a = _load_gray(p)
        if a.shape != (args.size, args.size):
            errors.append(f"{p.name} 尺寸 {a.shape} != {(args.size, args.size)}")
        if a.min() == a.max():
            warnings.append(f"{p.name} 是常量图（值={int(a.min())}）")

    print(f"建筑掩码 {len(masks)} / gain {len(all_gain)} / antenna {len(list(adir.glob('*.json')))}")
    for w in warnings:
        print("WARN:", w)
    if errors:
        print("\n校验失败：")
        for e in errors:
            print("  -", e)
        return 1
    print("校验通过 [OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
