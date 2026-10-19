"""模型工厂：把 config 变成一个模型实例，并把参数量信息一并返回。

`prior_none` 与其它三种配置的关键区别
-----------------------------------
    none     → `UNet`：没有条件分支，也没有 FiLM 参数
    masked / real / shuffled / 各种 CoT 变体 → `UNetFiLM`：结构**逐位相同**，
               差别只在喂进去的向量是零、真值、被打乱的值，或 CoT 文本。

这样 `real − masked` 才是先验信息的净贡献（交接文档 §8.3）。
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.models.cot_encoder import HashingTextEncoder
from src.models.film import UNetFiLM
from src.models.unet import UNet

MODEL_NAMES = ("unet", "unet_film")


def build_cot_encoder(model_cfg: dict) -> HashingTextEncoder | None:
    cot = (model_cfg.get("cot") or {})
    if not cot.get("enabled", False):
        return None
    return HashingTextEncoder(
        out_dim=int(cot.get("out_dim", 32)),
        buckets=int(cot.get("buckets", 8192)),
        hidden=int(cot.get("hidden", 256)),
        max_tokens=int(cot.get("max_tokens", 512)),
    )


def cot_mode(model_cfg: dict) -> str:
    cot = (model_cfg.get("cot") or {})
    if not cot.get("enabled", False):
        return "none"
    mode = str(cot.get("mode", "both"))
    if mode not in ("cot_only", "both"):
        raise ValueError(f"cot.enabled=true 时 cot.mode 只能是 cot_only/both，收到 {mode!r}")
    return mode


def build_model(
    model_cfg: dict,
    in_channels: int,
    prior_dim: int,
    out_channels: int = 1,
) -> tuple[nn.Module, dict[str, Any]]:
    """构造模型。返回 (model, 参数量/配置摘要)。"""
    name = str(model_cfg.get("name", "unet_film"))
    if name not in MODEL_NAMES:
        raise ValueError(f"model.name 必须是 {MODEL_NAMES}，收到 {name!r}")

    base_channels = int(model_cfg.get("base_channels", 32))
    dropout = float(model_cfg.get("dropout", 0.15))
    out_clamp = model_cfg.get("out_clamp", None)
    cot = model_cfg.get("cot") or {}
    mode = cot_mode(model_cfg)

    if name == "unet":
        if mode != "none":
            raise ValueError("model.name=unet 没有条件分支，不能开启 cot")
        if prior_dim:
            # 纯 U-Net 配置下不允许悄悄把先验丢掉：要对比请用 prior_masked。
            raise ValueError(
                f"model.name=unet 但 prior_dim={prior_dim}（>0）。"
                f"要做参数量对齐的对照请用 model.name=unet_film + prior.mode=masked，"
                f"而不是把先验丢掉。"
            )
        model = UNet(in_channels=in_channels, out_channels=out_channels,
                     base_channels=base_channels, dropout=dropout, out_clamp=out_clamp)
        info = {"name": name, "in_channels": in_channels, "prior_dim": 0,
                "cot_mode": "none", "base_channels": base_channels, "dropout": dropout}
    else:
        film = model_cfg.get("film") or {}
        cot_enc = build_cot_encoder(model_cfg)
        model = UNetFiLM(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            prior_dim=prior_dim,
            dropout=dropout,
            film_identity_init=bool(film.get("identity_init", True)),
            allow_no_identity_init=bool(film.get("allow_no_identity_init", False)),
            out_clamp=out_clamp,
            cot_encoder=cot_enc,
            cot_mode=mode,
            film_hidden=tuple(film.get("hidden", (128, 256))),
        )
        info = {
            "name": name, "in_channels": in_channels, "prior_dim": prior_dim,
            "cot_mode": mode, "base_channels": base_channels, "dropout": dropout,
            "film_hidden": list(film.get("hidden", (128, 256))),
            "film_identity_init": bool(film.get("identity_init", True)),
            "film_in_dim": getattr(model, "film_in_dim", None),
            "cot_dim": int(getattr(cot_enc, "out_dim", 0)) if cot_enc is not None else 0,
        }

    info["n_params_total"] = int(model.count_parameters())
    info["n_params_conditional"] = int(
        model.count_conditional_parameters() if hasattr(model, "count_conditional_parameters") else 0
    )
    info["n_params_main"] = info["n_params_total"] - info["n_params_conditional"]
    return model, info


def identity_check(model: nn.Module, atol: float = 1e-2) -> dict[str, float] | None:
    """FiLM identity 初始化自检（纯 U-Net 返回 None）。

    `atol` 默认 1e-2 而不是 1e-5：identity 初始化**必须**给最后一层权重留一个
    1e-4 量级的小随机值，否则该层梯度恒为 0、先验永久失效（实测踩过：
    `real_v3` 与 `masked_em` 跑 5 个 epoch 后 84 个张量仍逐位相同）。
    所以 γ 与 1 的偏差在 1e-4 量级是**正常**的，不是初始化失败。
    """
    if hasattr(model, "check_identity"):
        return model.check_identity(atol=atol)
    return None
