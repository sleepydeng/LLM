"""训练损失：SmoothL1 + 梯度 + SSIM（权重全部来自 config）。

沿用上一版验证过的组合（平滑 L1 + 0.5·梯度 + 0.3·SSIM），但做了两件事：
1. 权重、窗口、常数全部可配，不再散落在代码里；
2. SSIM 的实现与 `src/eval/metrics.py` 的 numpy 版**在自检里交叉验证过**
   （`tests/test_local.py`），避免出现"训练时优化的 SSIM"和"报告里写的 SSIM"
   不是同一个量的情况 —— 上一版的报告数字与图表对不上就是这类问题的后果。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# SSIM 常数（数据已归一化到 [0,1]，故 L=1）
SSIM_C1 = (0.01 * 1.0) ** 2
SSIM_C2 = (0.03 * 1.0) ** 2
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5


def gaussian_window(size: int = SSIM_WINDOW, sigma: float = SSIM_SIGMA,
                    device=None, dtype=torch.float32) -> torch.Tensor:
    """一维高斯核，展成 (1, 1, size, size) 的二维可分离窗口。"""
    coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None, :, :]


def _filter(x: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    return F.conv2d(x, window, padding=window.shape[-1] // 2, groups=x.shape[1])


def ssim_torch(pred: torch.Tensor, target: torch.Tensor,
               window: torch.Tensor | None = None) -> torch.Tensor:
    """逐通道 SSIM 的均值（返回标量，越大越好）。pred/target 形状 (B, C, H, W)。"""
    if window is None:
        window = gaussian_window(device=pred.device, dtype=pred.dtype)
    mu_p = _filter(pred, window)
    mu_t = _filter(target, window)
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t

    sigma_p2 = _filter(pred * pred, window) - mu_p2
    sigma_t2 = _filter(target * target, window) - mu_t2
    sigma_pt = _filter(pred * target, window) - mu_pt

    ssim_map = ((2 * mu_pt + SSIM_C1) * (2 * sigma_pt + SSIM_C2)) / (
        (mu_p2 + mu_t2 + SSIM_C1) * (sigma_p2 + sigma_t2 + SSIM_C2)
    )
    return ssim_map.mean()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor,
              window: torch.Tensor | None = None) -> torch.Tensor:
    """1 − SSIM，作为可最小化的损失。"""
    return 1.0 - ssim_torch(pred, target, window)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """梯度（边缘）误差：x/y 方向一阶差分的 L1 均值。

    这是上一版 V1→V2 唯一改动的那个损失项，报告称它"最关键"但没有做权重搜索与
    消融（§5.3/§7.3 自承）。本项目把它作为可单独开关的项，便于做真正的消融。
    """
    dp_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dt_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    dp_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dt_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    return 0.5 * ((dp_x - dt_x).abs().mean() + (dp_y - dt_y).abs().mean())


def total_loss(pred: torch.Tensor, target: torch.Tensor, loss_cfg: dict) -> dict[str, torch.Tensor]:
    """按 config 组合损失。返回各项与总和，便于逐项记进日志。

    **在 fp32 下计算损失**：混合精度（AMP）开启时 pred 可能是 fp16，而 SSIM 里有
    多次卷积与除法，fp16 下会掉精度、梯度也变成噪声。把损失入口统一升到 fp32
    是数值敏感损失的常规做法，代价可以忽略（只影响损失那几步，不影响主干前向）。
    """
    pred = pred.float()
    target = target.float()
    w_l1 = float(loss_cfg.get("smooth_l1", 1.0))
    w_grad = float(loss_cfg.get("grad", 0.5))
    w_ssim = float(loss_cfg.get("ssim", 0.3))
    beta = float(loss_cfg.get("smooth_l1_beta", 0.1))

    parts: dict[str, torch.Tensor] = {}
    total = pred.new_zeros(())
    if w_l1:
        parts["smooth_l1"] = F.smooth_l1_loss(pred, target, beta=beta)
        total = total + w_l1 * parts["smooth_l1"]
    if w_grad:
        parts["grad"] = gradient_loss(pred, target)
        total = total + w_grad * parts["grad"]
    if w_ssim:
        parts["ssim"] = ssim_loss(pred, target)
        total = total + w_ssim * parts["ssim"]
    parts["total"] = total
    return parts


def masked_loss(pred: torch.Tensor, target: torch.Tensor,
                sat_mask: torch.Tensor | None, loss_cfg: dict) -> dict[str, torch.Tensor]:
    """可选：把被噪声地板钉住的像素排除在损失之外。

    上一版 `gain_norm = max(gain_norm, 0.2)` 把大量低增益像素钉在 200 dB，
    这些像素真值不是真实路损，却主导了 RMSE。主实验仍用全图损失（与旧结果可比），
    这个开关用于"排除饱和像素训练"的补充消融。
    """
    if sat_mask is None or not loss_cfg.get("exclude_saturated", False):
        return total_loss(pred, target, loss_cfg)
    pred = pred.float()
    target = target.float()
    keep = (~sat_mask.bool()).to(pred.dtype)
    denom = keep.sum().clamp_min(1.0)
    cfg = dict(loss_cfg)
    cfg["exclude_saturated"] = False

    parts: dict[str, torch.Tensor] = {}
    w_l1 = float(cfg.get("smooth_l1", 1.0))
    w_grad = float(cfg.get("grad", 0.5))
    w_ssim = float(cfg.get("ssim", 0.3))
    total = pred.new_zeros(())
    if w_l1:
        parts["smooth_l1"] = (F.smooth_l1_loss(pred, target, beta=float(cfg.get("smooth_l1_beta", 0.1)),
                                               reduction="none") * keep).sum() / denom
        total = total + w_l1 * parts["smooth_l1"]
    if w_grad:
        parts["grad"] = gradient_loss(pred, target)
        total = total + w_grad * parts["grad"]
    if w_ssim:
        parts["ssim"] = ssim_loss(pred, target)
        total = total + w_ssim * parts["ssim"]
    parts["total"] = total
    return parts
