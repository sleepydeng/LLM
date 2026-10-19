"""评测指标（纯 numpy 实现）。

为什么用 numpy 而不是 torch
--------------------------
上一版的评测脚本体现在训练环境里，导致 ① 口径与训练日志分叉（5.38/5.09 vs 5.11/4.86），
② `avg_mae` 恒为 0.0 而没人发现，③ EMA 权重分支永远走不到。
这里把指标写成纯 numpy：**不依赖 GPU、可在本地对拍**，训练侧只需要把 tensor 转成
`float32` 数组即可，训练与评测共享同一份口径。

指标清单（报告里必须给全，不能只给一个 RMSE）
-------------------------------------------
* `rmse_db` / `mae_db`：反归一化到 dB 后逐样本计算。**dB 数依赖**
  `PATHLOSS_MIN/MAX=40/200` 这个人为区间，论文里必须写明；
* `rmse_db_nonsaturated`：排除被噪声地板钉在 200 dB 的像素。
  这批像素真值不是真实路损，却会显著拉低 RMSE；
* `ssim`：结构与 `src/train/losses.py:ssim_torch` 对齐（11×11、σ=1.5、C1/C2 同值），
  且在 `tests/test_local.py` 里与 torch 版对拍过；
* `grad_error`：一阶差分 L1，与训练里的梯度损失同一公式；
* `rmse_norm`：归一化口径，便于与 RadioUNet 等文献换算。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from src.data.normalize import PATHLOSS_MAX, PATHLOSS_MIN, db_span

SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
SSIM_C1 = (0.01 * 1.0) ** 2
SSIM_C2 = (0.03 * 1.0) ** 2

METRIC_KEYS = (
    "rmse_db", "mae_db", "rmse_db_nonsaturated", "rmse_norm", "ssim", "grad_error",
)


def _gauss1d(size: int = SSIM_WINDOW, sigma: float = SSIM_SIGMA) -> np.ndarray:
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


_G1D = _gauss1d()


def _sep_filter(x: np.ndarray) -> np.ndarray:
    """与 torch 的 F.conv2d(padding=k//2) 等价的零填充可分离高斯滤波。"""
    y = ndimage.correlate1d(x, _G1D, axis=0, mode="constant", cval=0.0)
    return ndimage.correlate1d(y, _G1D, axis=1, mode="constant", cval=0.0)


def ssim_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """单通道（或逐通道后取均值）SSIM。输入形状 (H, W) 或 (C, H, W)。"""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.ndim == 3:
        return float(np.mean([ssim_numpy(p[c], t[c]) for c in range(p.shape[0])]))
    mu_p, mu_t = _sep_filter(p), _sep_filter(t)
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    sig_p2 = _sep_filter(p * p) - mu_p2
    sig_t2 = _sep_filter(t * t) - mu_t2
    sig_pt = _sep_filter(p * t) - mu_pt
    num = (2 * mu_pt + SSIM_C1) * (2 * sig_pt + SSIM_C2)
    den = (mu_p2 + mu_t2 + SSIM_C1) * (sig_p2 + sig_t2 + SSIM_C2)
    return float(np.mean(num / den))


def gradient_error(pred: np.ndarray, target: np.ndarray) -> float:
    """与 `src/train/losses.py:gradient_loss` 同一公式。"""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    dp_x = p[..., :, 1:] - p[..., :, :-1]
    dt_x = t[..., :, 1:] - t[..., :, :-1]
    dp_y = p[..., 1:, :] - p[..., :-1, :]
    dt_y = t[..., 1:, :] - t[..., :-1, :]
    return float(0.5 * (np.abs(dp_x - dt_x).mean() + np.abs(dp_y - dt_y).mean()))


def _db(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) * db_span() + PATHLOSS_MIN


def sample_metrics(pred_norm: np.ndarray, target_norm: np.ndarray,
                   sat_mask: np.ndarray | None = None) -> dict[str, float]:
    """单个样本的指标。所有形状为 (1, H, W) 或 (H, W)。"""
    p = np.squeeze(np.asarray(pred_norm, dtype=np.float64))
    t = np.squeeze(np.asarray(target_norm, dtype=np.float64))
    err_norm = p - t
    out = {
        "rmse_norm": float(np.sqrt(np.mean(err_norm ** 2))),
        "rmse_db": float(np.sqrt(np.mean((_db(p) - _db(t)) ** 2))),
        "mae_db": float(np.mean(np.abs(_db(p) - _db(t)))),
        "ssim": ssim_numpy(p, t),
        "grad_error": gradient_error(p, t),
    }
    if sat_mask is not None:
        keep = ~np.squeeze(np.asarray(sat_mask)).astype(bool)
        if keep.any():
            out["rmse_db_nonsaturated"] = float(
                np.sqrt(np.mean((_db(p)[keep] - _db(t)[keep]) ** 2)))
            out["n_nonsaturated"] = float(keep.sum())
        else:
            out["rmse_db_nonsaturated"] = float("nan")
            out["n_nonsaturated"] = 0.0
    else:
        out["rmse_db_nonsaturated"] = float("nan")
        out["n_nonsaturated"] = float(p.size)
    out["n_pixels"] = float(p.size)
    return out


def aggregate(sample_rows: list[dict[str, float]]) -> dict[str, float]:
    """逐样本指标 → 数据集级汇总（均值 + 标准差，便于报 mean±std）。"""
    if not sample_rows:
        return {}
    keys = [k for k in METRIC_KEYS if k in sample_rows[0]]
    out: dict[str, float] = {"n_samples": len(sample_rows)}
    for k in keys:
        vals = np.array([r[k] for r in sample_rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out[f"mean_{k}"] = float("nan")
            out[f"std_{k}"] = float("nan")
            continue
        out[f"mean_{k}"] = float(vals.mean())
        out[f"std_{k}"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    return out


def norm_to_db(norm_value: float) -> float:
    """归一化 RMSE → dB RMSE（= norm × 160），便于与文献数字互相换算。"""
    return float(norm_value) * db_span()


def db_bounds() -> tuple[float, float]:
    return PATHLOSS_MIN, PATHLOSS_MAX
