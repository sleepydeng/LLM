"""方位扇区几何：prompt 画图、dataset 楔形通道、audit 一致性检查三处共用。

统一约定（与 `src/prior/schema.py` 的方位角定义**必须**一致）：
    azimuth 逆时针为正，0° = 右/东(+x)，90° = 上/北(−y)，180° = 左/西，270° = 下/南。
    像素坐标 y 轴向下，所以射线方向 = (dx, dy) = (cos(az), −sin(az))。

像素尺度：RadioMapSeer 的 antenna/*.json 里存的就是像素坐标（实测 x∈[54,198]，
y∈[54,201]，图像 256×256），即 **1 px = 1 m**。所以 100 m 半径 = 100 px。
该值通过 `px_per_m` 参数暴露，不要硬编码到调用点。
"""
from __future__ import annotations

import numpy as np

from src.prior.schema import AZIMUTHS, OBSTRUCTIONS

# 遮挡等级 → [0,1] 标量（越大越封闭）
OBSTRUCTION_LEVEL: dict[str, float] = {
    ob: i / (len(OBSTRUCTIONS) - 1) for i, ob in enumerate(OBSTRUCTIONS)
}

# 几何代理判定阈值（粗检，不是严格电磁遮挡）
BLOCKED_NEAREST_M = 10.0   # 最近建筑 ≤ 10 m → blocked
DENSE_FRACTION = 0.30      # 扇区内建筑像素占比 ≥ 30% → dense


def angular_index_map(
    height: int,
    width: int,
    tx_xy: tuple[float, float] | list[float],
    azimuths: tuple[int, ...] = AZIMUTHS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (扇区索引图, 半径图像素, 方位角图[度])。

    索引 k 对应 `azimuths[k]`，归属规则是"离哪个方位最近就归哪个"（每个扇区 45° 宽，
    以方位角为中心）。
    """
    ys, xs = np.mgrid[0:height, 0:width]
    dx = xs.astype(np.float32) - float(tx_xy[0])
    dy = ys.astype(np.float32) - float(tx_xy[1])
    # 逆时针、y 向上为正 → 取 −dy
    az = np.degrees(np.arctan2(-dy, dx)) % 360.0
    radius = np.sqrt(dx * dx + dy * dy)
    step = 360.0 / len(azimuths)
    idx = np.floor(((az + step / 2.0) % 360.0) / step).astype(np.int16)
    return idx, radius, az


def building_mask_bool(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    """建筑掩码 → 布尔图。掩码实测严格二值 {0,255}。"""
    return np.asarray(mask) > threshold


def local_building_ratio(
    mask: np.ndarray,
    tx_xy: tuple[float, float] | list[float],
    radius_m: float,
    px_per_m: float = 1.0,
) -> float:
    """以 Tx 为中心、半径 radius_m 的圆内建筑像素占比。"""
    h, w = mask.shape
    idx, radius, _ = angular_index_map(h, w, tx_xy)
    sel = radius <= radius_m * px_per_m
    if not sel.any():
        return 0.0
    b = building_mask_bool(mask)
    return float(b[sel].mean())


def geometric_obstruction(
    mask: np.ndarray,
    tx_xy: tuple[float, float] | list[float],
    azimuths: tuple[int, ...] = AZIMUTHS,
    radius_m: float = 100.0,
    px_per_m: float = 1.0,
) -> dict[int, dict[str, float | str]]:
    """从建筑掩码算出每个方位的期望遮挡等级（粗检代理）。

    判定规则（刻意简单、可复现，且已在文档里写明是代理而非严格几何遮挡）：
        1. 该扇区 100 m 内没有建筑像素           → open,    nearest = 500
        2. 最近建筑距离 ≤ 10 m                    → blocked
        3. 扇区内建筑像素占比 ≥ 30%               → dense
        4. 其余                                   → sparse

    返回：{azimuth: {"obstruction": str, "nearest_building_m": float, "building_fraction": float}}
    """
    h, w = mask.shape
    idx, radius, _ = angular_index_map(h, w, tx_xy, azimuths)
    b = building_mask_bool(mask)
    within = radius <= radius_m * px_per_m

    out: dict[int, dict[str, float | str]] = {}
    for k, az in enumerate(azimuths):
        sel = within & (idx == k)
        if not sel.any():
            out[az] = {"obstruction": "open", "nearest_building_m": 500.0,
                       "building_fraction": 0.0}
            continue
        nb = b & sel
        frac = float(nb.sum()) / float(sel.sum())
        if not nb.any():
            out[az] = {"obstruction": "open", "nearest_building_m": 500.0,
                       "building_fraction": frac}
            continue
        nearest_px = float(radius[nb].min())
        nearest_m = nearest_px / px_per_m
        if nearest_m <= BLOCKED_NEAREST_M:
            label = "blocked"
        elif frac >= DENSE_FRACTION:
            label = "dense"
        else:
            label = "sparse"
        out[az] = {"obstruction": label, "nearest_building_m": nearest_m,
                   "building_fraction": frac}
    return out


def wedge_level_maps(
    sector_levels: dict[int, float],
    height: int,
    width: int,
    tx_xy: tuple[float, float] | list[float],
    azimuths: tuple[int, ...] = AZIMUTHS,
) -> np.ndarray:
    """把 8 个方位的遮挡等级铺成 (8, H, W) 楔形图，供 dataset 拼接成额外通道。

    第 k 个通道 = 该方向覆盖的像素取 `sector_levels[azimuths[k]]`，其余为 0。
    这是"信号往哪些方向传得出去"的空间化表达，而不是把标量广播成 40 通道。
    """
    idx, _, _ = angular_index_map(height, width, tx_xy, azimuths)
    out = np.zeros((len(azimuths), height, width), dtype=np.float32)
    for k, az in enumerate(azimuths):
        out[k] = np.where(idx == k, float(sector_levels.get(az, 0.0)), 0.0)
    return out


def obstruction_level(label: str) -> float:
    """遮挡等级标签 → [0,1] 标量。未知标签抛错（不静默当 0）。"""
    if label not in OBSTRUCTION_LEVEL:
        raise ValueError(f"未知遮挡等级 {label!r}，可选 {OBSTRUCTIONS}")
    return OBSTRUCTION_LEVEL[label]


if __name__ == "__main__":  # 手工自检
    from PIL import Image
    from pathlib import Path

    p = Path(__file__).resolve().parents[2] / "data" / "radiomapseer"
    m = np.array(Image.open(p / "buildings_complete" / "0.png").convert("L"))
    import json
    tx = json.loads((p / "antenna" / "0.json").read_text())[0]
    print("Tx 像素坐标:", tx)
    print("Tx 周围 50 m 建筑占比: %.3f" % local_building_ratio(m, tx, 50))
    print("Tx 周围 100 m 建筑占比: %.3f" % local_building_ratio(m, tx, 100))
    g = geometric_obstruction(m, tx)
    for az in AZIMUTHS:
        d = g[az]
        print(f"  az={az:<4} {d['obstruction']:<8} nearest={d['nearest_building_m']:6.1f}m "
              f"frac={d['building_fraction']:.3f}")
    wm = wedge_level_maps({az: obstruction_level(g[az]["obstruction"]) for az in AZIMUTHS},
                          256, 256, tx)
    print("楔形通道形状:", wm.shape, "非零通道数:", int((wm.reshape(8, -1).max(1) > 0).sum()))
