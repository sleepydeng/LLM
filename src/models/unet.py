"""U-Net 主干（与上一版结构保持一致，只做必要清理）。

保留的设计（上一版验证过、没问题）
---------------------------------
* GroupNorm + GELU（小 batch 友好）、残差 ConvBlock、Dropout2d；
* Encoder 4 次下采样 256→128→64→32→16，Decoder 4 次上采样；
* MaxPool 下采样 + bilinear 上采样；
* 输出层 1×1 卷积，**不加 Sigmoid**。

相对上一版的清理
---------------
* `in_channels` / `base_channels` 完全来自 config，不再在模型里硬编码 26 或 4
  （上一版 `train_radiomapseer.py:117` 建的是 26 通道模型，而 dataset 给的是 4 通道，
  于是必然 `RuntimeError`）；
* 输出是否 clamp 由 config 显式声明（`out_clamp`），不再"让梯度自由流动"这种隐含约定。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """双卷积块：Conv→GN→GELU→Conv→GN→GELU，带残差连接。"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)
        self.residual = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.act(self.norm2(self.conv2(out)))
        return out + res


class DownBlock(nn.Module):
    """MaxPool → ConvBlock。"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsample → concat(skip) → ConvBlock。"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """纯场景图 U-Net（无先验、无条件分支）= `prior_none` 配置。"""

    def __init__(self, in_channels: int = 4, out_channels: int = 1,
                 base_channels: int = 32, dropout: float = 0.15,
                 out_clamp: tuple[float, float] | list[float] | None = None):
        super().__init__()
        c = base_channels
        self.out_clamp = tuple(out_clamp) if out_clamp else None

        self.inc = ConvBlock(in_channels, c, dropout)
        self.down1 = DownBlock(c, c * 2, dropout)
        self.down2 = DownBlock(c * 2, c * 4, dropout)
        self.down3 = DownBlock(c * 4, c * 8, dropout)
        self.down4 = DownBlock(c * 8, c * 8, dropout)
        self.bottleneck = ConvBlock(c * 8, c * 8, dropout)
        self.up1 = UpBlock(c * 8, c * 8, c * 8, dropout)
        self.up2 = UpBlock(c * 8, c * 4, c * 4, dropout)
        self.up3 = UpBlock(c * 4, c * 2, c * 2, dropout)
        self.up4 = UpBlock(c * 2, c, c, dropout)
        self.outc = nn.Conv2d(c, out_channels, 1)

    def forward(self, x: torch.Tensor, prior_vec: torch.Tensor | None = None,
                cot_text: list[str] | None = None) -> torch.Tensor:
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)
        b = self.bottleneck(e5)
        d1 = self.up1(b, e4)
        d2 = self.up2(d1, e3)
        d3 = self.up3(d2, e2)
        d4 = self.up4(d3, e1)
        out = self.outc(d4)
        if self.out_clamp is not None:
            out = out.clamp(*self.out_clamp)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
