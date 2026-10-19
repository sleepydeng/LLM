"""RadioMapSeer gain → pathloss 归一化，以及 dB 口径的唯一定义处。

**这个文件是评测口径的唯一来源。** 上一版的问题正是口径散落在多处：
训练日志一套（5.38/5.09）、独立复核一套（5.11/4.86），`metrics.json` 里 MAE 还全是 0.0。
新项目要求"一个口径、一处代码、一次报告"，所以：

    pathloss_norm = 1 - (max(gain/255, 0.2) - 0.2) / 0.8        ∈ [0, 1]
    pathloss_db   = pathloss_norm * (PATHLOSS_MAX - PATHLOSS_MIN) + PATHLOSS_MIN

两条必须写进论文的事实
----------------------
1. **dB 数依赖人为区间**：本数据集取 `PATHLOSS_MIN=40, PATHLOSS_MAX=200`。
   换一个区间，RMSE 的 dB 值就会变（归一化 RMSE 不变）。与 RadioUNet 等文献对比时
   必须同时给出归一化口径。5.11 dB ≈ 归一化 0.032。
2. **噪声地板会制造"饱和像素"**：`gain/255 < 0.2` 的像素被钉到 `pathloss_norm = 1`
   （即 200 dB）。这批像素数量大、误差也大，会主导 RMSE。因此除了主指标，
   必须额外报告一个 `rmse_db_nonsaturated`（排除饱和像素）。见 `saturation_mask()`。
"""
from __future__ import annotations

import numpy as np

# ===== 全局唯一的归一化常量 =====
PATHLOSS_MIN = 40.0    # dB
PATHLOSS_MAX = 200.0   # dB
GAIN_THRESH = 0.2      # RadioMapSeer 噪声地板（与 RadioUNet 一致）
MAP_SIZE = 256         # 图像边长（1 px = 1 m）


def gain_to_pathloss_norm(gain_pixel: np.ndarray) -> np.ndarray:
    """RadioMapSeer gain 像素（uint8，255=信号最强）→ 归一化 pathloss（float32，[0,1]）。

    步骤：归一化 → 噪声地板 → 重新缩放 → 取反。
    """
    g = np.asarray(gain_pixel, dtype=np.float32) / 255.0
    g = np.maximum(g, GAIN_THRESH)
    g = (g - GAIN_THRESH) / (1.0 - GAIN_THRESH)
    return (1.0 - g).astype(np.float32)


def pathloss_norm_to_gain_pixel(pl_norm: np.ndarray) -> np.ndarray:
    """归一化 pathloss → gain 像素值（仅用于可视化对比）。"""
    g = 1.0 - np.asarray(pl_norm, dtype=np.float32)
    g = g * (1.0 - GAIN_THRESH) + GAIN_THRESH
    return np.clip(g * 255.0, 0, 255).astype(np.uint8)


def saturation_mask(gain_pixel: np.ndarray) -> np.ndarray:
    """标记被噪声地板钉住的像素（True = 饱和，其真值就是 200 dB）。

    这些像素的真值不是真实路损，而是"低于噪声地板"的占位值，所以它们
    单独统计出来的 RMSE 才反映模型对**有效区域**的拟合能力。

    注意这里**不写成 `gain/255 < GAIN_THRESH`**：`gain == 51` 时
    `(0.2-0.2)/0.8 = 0`，`pathloss_norm` 恰好等于 1.0，也就是它同样被钉在 200 dB。
    为了和上面的映射严格一致，这里沿用同一个公式反推，
    避免"边界那一列像素算不算饱和"这种会悄悄影响指标的差异。
    """
    return gain_to_pathloss_norm(gain_pixel) >= 1.0 - 1e-7


def pathloss_norm_to_db(pl_norm: np.ndarray) -> np.ndarray:
    """归一化 pathloss → dB。区间由本模块的 PATHLOSS_MIN/MAX 决定。"""
    return np.asarray(pl_norm, dtype=np.float64) * (PATHLOSS_MAX - PATHLOSS_MIN) + PATHLOSS_MIN


def db_to_pathloss_norm(pl_db: np.ndarray) -> np.ndarray:
    """dB → 归一化 pathloss。"""
    return (np.asarray(pl_db, dtype=np.float64) - PATHLOSS_MIN) / (PATHLOSS_MAX - PATHLOSS_MIN)


def db_span() -> float:
    """dB 与归一化单位之间的换算系数，报告里要写明。"""
    return PATHLOSS_MAX - PATHLOSS_MIN


def norm_rmse_to_db(norm_rmse: float) -> float:
    """归一化 RMSE → dB RMSE，便于与文献的数字互相换算。"""
    return float(norm_rmse) * db_span()
