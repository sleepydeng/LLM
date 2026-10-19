"""FiLM 条件注入：`out = γ·x + β`，逐样本独立，**带 identity 初始化**。

两个必须修掉的上一版 bug
----------------------
1. **FiLM 没有 identity 初始化**（`unet.py` 原版直接让最后一个 Linear 随机初始化）。
   标准 FiLM 应让初始状态等价于恒等变换（γ=1, β=0），否则训练一开始 FiLM 分支
   就是一个随机仿射噪声源。上一版 epoch 1 的验证 RMSE：FiLM 118.26 dB vs
   消融 75.85 dB，差 42.4 dB，而报告只写"稳定、温和地提升"。这里初始化后再跑，
   前期劣化是否消失本身就是一条要报告的结论。

2. **参数量不对齐**。上一版 FiLM 比 baseline 多 282,688 个参数（正好是 `film_mlp`
   的大小），所以 +0.25 dB 无法排除"单纯多了容量"。本实现的 `prior_masked` 配置
   **用完全相同的结构**，只把先验向量换成 `torch.zeros`，两模型参数量逐位相同，
   差值可以干净地归因于先验信息（交接文档 §8.3）。

注意 γ 的布局不是前缀
--------------------
`film_total = 2·Σ(decoder_channels)`，并且按层交错存放 `[γ_ch, β_ch]`。
所以 identity 初始化必须按层设置 γ 的下标，不能简单写 `bias[:gamma_total] = 1`。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.unet import ConvBlock, DownBlock, UpBlock


class FiLMLayer(nn.Module):
    """通道级仿射调制。"""

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return x * gamma + beta


class UNetFiLM(nn.Module):
    """U-Net + FiLM（只调制 decoder 的 4 个上采样层）。

    prior 向量（结构化字段）与可选的 CoT 文本嵌入拼在一起后，经 MLP 生成
    4 组 (γ, β)。CoT 是否参与由 `cot_mode` 决定：
        none      只用结构化字段（默认，对应"H3 声明不注入"）
        cot_only  只用 CoT 文本嵌入
        both      两者拼接（可回答"CoT 是否带来额外信息"）
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 32,
        prior_dim: int = 64,
        dropout: float = 0.15,
        film_identity_init: bool = True,
        allow_no_identity_init: bool = False,
        out_clamp: tuple[float, float] | list[float] | None = None,
        cot_encoder: nn.Module | None = None,
        cot_mode: str = "none",
        film_hidden: tuple[int, int] = (128, 256),
    ):
        super().__init__()
        if cot_mode not in ("none", "cot_only", "both"):
            raise ValueError(f"cot_mode 必须是 none/cot_only/both，收到 {cot_mode!r}")
        if cot_mode in ("cot_only", "both") and cot_encoder is None:
            raise ValueError(f"cot_mode={cot_mode} 需要传入 cot_encoder")
        if cot_mode == "cot_only" and prior_dim:
            prior_dim_used = 0
        else:
            prior_dim_used = prior_dim

        self.cot_mode = cot_mode
        self.cot_encoder = cot_encoder
        self.prior_dim = prior_dim
        self.out_clamp = tuple(out_clamp) if out_clamp else None
        cot_dim = int(getattr(cot_encoder, "out_dim", 0)) if cot_encoder is not None else 0
        film_in = prior_dim_used + (cot_dim if cot_mode in ("cot_only", "both") else 0)
        if film_in <= 0:
            raise ValueError("FiLM 的输入维度为 0：prior_dim 与 cot_dim 不能同时为空")
        self.film_in_dim = film_in

        c = base_channels
        self.inc = ConvBlock(in_channels, c, dropout)
        self.down1 = DownBlock(c, c * 2, dropout)
        self.down2 = DownBlock(c * 2, c * 4, dropout)
        self.down3 = DownBlock(c * 4, c * 8, dropout)
        self.down4 = DownBlock(c * 8, c * 8, dropout)
        self.bottleneck = ConvBlock(c * 8, c * 8, dropout)

        self._decoder_channels = [c * 8, c * 4, c * 2, c]      # [256,128,64,32] @ c=32
        self.film_total = 2 * sum(self._decoder_channels)

        h1, h2 = film_hidden
        self.film_mlp = nn.Sequential(
            nn.Linear(film_in, h1), nn.GELU(),
            nn.Linear(h1, h2), nn.GELU(),
            nn.Linear(h2, self.film_total),
        )

        self.up1 = UpBlock(c * 8, c * 8, c * 8, dropout)
        self.film1 = FiLMLayer()
        self.up2 = UpBlock(c * 8, c * 4, c * 4, dropout)
        self.film2 = FiLMLayer()
        self.up3 = UpBlock(c * 4, c * 2, c * 2, dropout)
        self.film3 = FiLMLayer()
        self.up4 = UpBlock(c * 2, c, c, dropout)
        self.film4 = FiLMLayer()
        self.outc = nn.Conv2d(c, out_channels, 1)

        if film_identity_init:
            self.apply_film_identity_init()
        elif not allow_no_identity_init:
            raise ValueError(
                "film_identity_init=False 正是上一版的 bug 来源（epoch 1 FiLM 118 dB vs "
                "消融 76 dB）。如果这是有意做的消融对照，请显式传 "
                "allow_no_identity_init=True 并把它记进 config。"
            )

    # ---------- identity 初始化 ----------

    def gamma_beta_indices(self) -> tuple[list[int], list[int]]:
        """返回（γ 的下标列表, β 的下标列表）。按层交错，不是前缀。"""
        gamma_idx: list[int] = []
        beta_idx: list[int] = []
        off = 0
        for ch in self._decoder_channels:
            gamma_idx.extend(range(off, off + ch))
            beta_idx.extend(range(off + ch, off + 2 * ch))
            off += 2 * ch
        return gamma_idx, beta_idx

    @torch.no_grad()
    def apply_film_identity_init(self) -> None:
        """γ=1、β=0：让 FiLM 分支在初始状态等价于恒等变换。

        ⚠️ 这里**只允许置零最后一层的权重，绝不能连权重一起置零** —— 这是实测
        踩出来的坑，而且它不报错、只在训练日志里表现为"两组实验永远逐位相同"：

            plot: 若把最后一层 `weight` 与 `bias` 一起置零、只把 γ 的 bias 设成 1，
            那么 film_mlp 的输出恒为 [1, 0, …]，**与输入完全无关**；
            更致命的是最后一层权重的梯度 ∝ (上游梯度 × 该层输入)，而该层输入恒为 0，
            于是**梯度永远是 0、权重永久冻结在 0** ⇒ 先验从头到尾不影响前向。

            实测后果：`prior_real_v3` 与 `prior_v3_masked_em` 训练 5 个 epoch 后
            **全部 84 个张量仍逐位相同**（max|Δ| = 0.000000e+00），
            连 train total loss 都一模一样 —— 整轮机时全部无效。

        正确做法：bias 保持 identity（γ 位 1、β 位 0），权重用**很小的随机值**。
        初始 γ ≈ 1 + ε（ε ~ 1e-4 量级，相对通道值可忽略），identity 性质在数值上保留，
        但梯度可以正常流回整个 MLP 分支。
        """
        last = self.film_mlp[-1]
        # 决定性随机小权重：既保留"初始≈恒等"，又保证梯度能回传（见上）
        nn.init.normal_(last.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(last.bias)
        gamma_idx, _ = self.gamma_beta_indices()
        last.bias[gamma_idx] = 1.0

    @torch.no_grad()
    def check_identity(self, atol: float = 1e-5) -> dict[str, float]:
        """自检：随机输入下，MLP 输出的 γ 是否恒为 1、β 是否恒为 0。

        注意 `atol` 的默认值：最后一层权重是 1e-4 量级的随机小值（为了让梯度能回传，
        见 `apply_film_identity_init`），所以 γ 与 1 的偏差量级是 1e-4～1e-3，
        **不是** 0。调用方判断"identity 是否成立"时应当用 1e-2 量级的容差；
        默认的 1e-5 只用于检查 bias 布局本身（γ 位必须精确等于 1）。
        """
        dev = next(self.parameters()).device
        dummy = torch.zeros(3, self.film_in_dim, device=dev)
        out = self.film_mlp(dummy)
        gamma_idx, beta_idx = self.gamma_beta_indices()
        return {
            "max_abs_gamma_minus_1": float((out[:, gamma_idx] - 1.0).abs().max()),
            "max_abs_beta": float(out[:, beta_idx].abs().max()),
            "ok": float(max((out[:, gamma_idx] - 1.0).abs().max(),
                            out[:, beta_idx].abs().max()) <= atol),
        }

    # ---------- 前向 ----------

    def _film_params(self, prior_vec: torch.Tensor | None,
                     cot_text: list[str] | None) -> list[tuple[torch.Tensor, torch.Tensor]]:
        parts: list[torch.Tensor] = []
        if self.cot_mode == "cot_only":
            if self.cot_encoder is None or cot_text is None:
                raise ValueError("cot_mode=cot_only 需要 cot_text")
            parts.append(self.cot_encoder(cot_text))
        else:
            if prior_vec is None:
                raise ValueError("该模型需要 prior_vec")
            if prior_vec.dim() == 1:
                prior_vec = prior_vec.unsqueeze(0)
            parts.append(prior_vec)
            if self.cot_mode == "both":
                if self.cot_encoder is None or cot_text is None:
                    raise ValueError("cot_mode=both 需要 cot_text")
                parts.append(self.cot_encoder(cot_text))
        film_in = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        film_out = self.film_mlp(film_in)

        params: list[tuple[torch.Tensor, torch.Tensor]] = []
        off = 0
        for ch in self._decoder_channels:
            chunk = film_out[:, off:off + 2 * ch]
            params.append((chunk[:, :ch].view(-1, ch, 1, 1),
                           chunk[:, ch:].view(-1, ch, 1, 1)))
            off += 2 * ch
        return params

    def forward(self, x: torch.Tensor, prior_vec: torch.Tensor | None = None,
                cot_text: list[str] | None = None) -> torch.Tensor:
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)
        b = self.bottleneck(e5)

        fp = self._film_params(prior_vec, cot_text)

        d1 = self.film1(self.up1(b, e4), fp[0][0], fp[0][1])
        d2 = self.film2(self.up2(d1, e3), fp[1][0], fp[1][1])
        d3 = self.film3(self.up3(d2, e2), fp[2][0], fp[2][1])
        d4 = self.film4(self.up4(d3, e1), fp[3][0], fp[3][1])
        out = self.outc(d4)
        if self.out_clamp is not None:
            out = out.clamp(*self.out_clamp)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_conditional_parameters(self) -> int:
        """条件分支（film_mlp + cot_encoder）的参数量，用于报告里说明容量差异。"""
        n = sum(p.numel() for p in self.film_mlp.parameters())
        if self.cot_encoder is not None:
            n += sum(p.numel() for p in self.cot_encoder.parameters())
        return n
