"""单一训练入口：yaml 驱动，四种 prior 模式 + CoT 消融。

为什么只有一个入口
-----------------
上一版有 `train_radiomapseer.py` / `_v2.py` / `_ablation.py` / `_film.py` 四份脚本，
它们互相漂移，最后连 dataset 的返回值都对不上（4 元组 vs 3 元组）、
26 通道模型配 4 通道数据，**当前仓库跑不出那批 checkpoint**（交接文档 §2 问题 4）。
现在用 `prior.mode` 与 `model.cot` 两个开关覆盖全部配置，不给脚本漂移的机会。

本入口强制执行的规则
-------------------
* `set_seed()` 第一个调用，seed 写进 `runs/<exp_id>/config.yaml`（规则 2）；
* 数据划分只从 `data/splits.json` 读（规则 3）；
* 报告用 test、选模型用 val，两者不混（规则 3）；
* checkpoint 里同时存 `model_state_dict` 与 `ema_shadow`，并**显式记录**
  `eval_weights`，避免上一版"EMA 分支永远走不到"的口径混乱（§8.4）。

用法
----
    python -m src.train.train --config configs/prior_real.yaml --seed 0
    python -m src.train.train --config configs/prior_none.yaml --seed 0 --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from src.data.dataset import RadioMapSeerDataset, build_dataloader  # noqa: E402
from src.data.splits import splits_hash  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.models.factory import build_model, identity_check  # noqa: E402
from src.train.losses import masked_loss  # noqa: E402
from src.utils.config import (  # noqa: E402
    freeze_run_dir,
    git_rev,
    load_config,
    rel_or_abs,
)
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.seed import describe_seed, set_seed  # noqa: E402


# ===== EMA =====


class EMA:
    """参数指数滑动平均。训练时记录的"最好成绩"走这条路，评测必须能选它（§8.4）。"""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> None:
        """把 EMA 权重量临时装进模型（用 swap 方式，之后可 restore）。"""
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: (self.shadow[k].to(v.dtype) if k in self.shadow else v)
                               for k, v in model.state_dict().items()}, strict=True)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        if self._backup:
            model.load_state_dict(self._backup, strict=True)
            self._backup = {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}


# ===== 学习率调度 =====


def lr_at(step: int, total_steps: int, warmup_steps: int, base_lr: float,
          min_ratio: float = 0.05) -> float:
    """线性 warmup + cosine 衰减。"""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    prog = min(max(prog, 0.0), 1.0)
    return base_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog)))


# ===== 训练 / 验证一步 =====


def forward_model(model, batch, cot_enabled: bool) -> torch.Tensor:
    return model(batch["inputs"], batch["prior_vec"],
                 batch["cot_text"] if cot_enabled else None)


@torch.no_grad()
def evaluate(model, loader, device, cot_enabled: bool,
             max_samples: int = 0) -> tuple[list[dict], dict]:
    """在给定 loader 上逐样本评测。返回 (逐样本行, 汇总)。"""
    model.eval()
    rows: list[dict] = []
    for batch in loader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        prior_vec = batch["prior_vec"].to(device, non_blocking=True)
        cot_text = batch["cot_text"] if cot_enabled else None
        pred = model(inputs, prior_vec, cot_text)
        p = pred.detach().float().cpu().numpy()
        t = target.detach().float().cpu().numpy()
        s = batch["sat_mask"].numpy()
        for i in range(p.shape[0]):
            row = M.sample_metrics(p[i], t[i], s[i])
            row["sample_id"] = batch["sample_id"][i]
            row["map_id"] = int(batch["map_id"][i])
            row["tx_id"] = int(batch["tx_id"][i])
            rows.append(row)
        if max_samples and len(rows) >= max_samples:
            rows = rows[:max_samples]
            break
    return rows, M.aggregate(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LLM-prior RadioMap 训练入口")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp-id", default=None, help="默认 <config名>_seed<seed>")
    ap.add_argument("--device", default=None, help="cuda / cpu（默认自动）")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖 config")
    ap.add_argument("--limit-train", type=int, default=0, help="只用前 N 个训练样本（调试）")
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：2 epoch + 小样本 + 小 batch，仅用于验证代码能跑")
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    exp_id = args.exp_id or f"{cfg['_config_name']}_seed{args.seed}"
    run_dir = ROOT / cfg.get("runs_dir", "runs") / exp_id
    logger = get_logger("train", run_dir / "train.log")

    seed = int(args.seed)
    set_seed(seed, deterministic=args.deterministic)
    logger.info(f"实验 {exp_id}｜seed={seed}｜config={args.config}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"设备 {device}｜torch {torch.__version__}｜cuda 可用 {torch.cuda.is_available()}")
    if device.type == "cpu":
        logger.warning("在 CPU 上训练：正式结果必须来自 Linux + NVIDIA GPU（AGENTS 规则 7）")

    # ---------- 数据 ----------
    dcfg = cfg["data"]
    pcfg = cfg["prior"]
    tcfg = cfg["train"]
    data_root = ROOT / dcfg["root"]
    splits_path = ROOT / dcfg["splits"]
    s_hash = splits_hash(splits_path)

    prior_dir = None
    if pcfg["mode"] in ("real", "shuffled"):
        prior_dir = ROOT / pcfg["dir"]
        if not prior_dir.is_dir():
            logger.error(f"先验目录不存在: {prior_dir}。先跑 src/prior/generate.py 并过 audit 门槛。")
            return 2

    prior_groups = tuple(pcfg.get("prior_groups", ["global_num", "global_cat", "sectors"]))
    cot_cfg = (cfg["model"].get("cot") or {})
    cot_enabled = bool(cot_cfg.get("enabled", False))
    cot_mode = str(cot_cfg.get("mode", "both")) if cot_enabled else "none"
    base_model_cfg = dict(cfg["model"])
    base_model_cfg["cot"] = dict(cot_cfg, enabled=cot_enabled, mode=cot_mode)

    common = dict(
        data_root=data_root, splits_path=splits_path, prior_mode=pcfg["mode"],
        prior_dir=prior_dir, prior_groups=prior_groups,
        sectors_channels=bool(pcfg.get("sectors_channels", False)),
        shuffle_seed=int(pcfg.get("shuffle_seed", 12345)),
        shuffle_cot=bool(pcfg.get("shuffle_cot", False)),
        zero_groups=tuple(pcfg.get("zero_groups", ()) or ()),
        freq_ghz=float(dcfg.get("freq_ghz", 0.9)),
        px_per_m=float(dcfg.get("px_per_m", 1.0)),
        tx_encoding=str(dcfg.get("tx_encoding", "constant")),
        return_cot_text=cot_enabled,
    )
    train_ds = RadioMapSeerDataset(split="train", **common)
    val_ds = RadioMapSeerDataset(split="val", **common)
    if args.limit_train:
        train_ds.samples = train_ds.samples[: args.limit_train]
        if train_ds.prior_vectors is not None:
            train_ds.prior_vectors = train_ds.prior_vectors[: args.limit_train]
        # 截断样本后错位映射会指向被切掉的样本，所以调试模式下关掉两种打乱
        if train_ds.prior_shuffle_index is not None or train_ds.cot_shuffle_index is not None:
            train_ds.prior_shuffle_index = None
            train_ds.cot_shuffle_index = None
            logger.warning("--limit-train 下关闭了 shuffled / shuffle_cot 的错位映射（仅调试用）")
    if args.limit_val:
        val_ds.samples = val_ds.samples[: args.limit_val]

    logger.info(f"train: {train_ds.describe()}")
    logger.info(f"val  : {val_ds.describe()}")

    batch_size = 4 if args.smoke else int(tcfg["batch_size"])
    num_workers = 0 if args.smoke else int(dcfg.get("num_workers", 8))
    train_loader = build_dataloader(train_ds, batch_size, True, num_workers, seed,
                                   pin_memory=(device.type == "cuda"), drop_last=False)
    val_loader = build_dataloader(val_ds, batch_size, False, max(0, num_workers // 2 or 0), seed,
                                 pin_memory=(device.type == "cuda"))

    # ---------- 模型 ----------
    model, minfo = build_model(base_model_cfg, in_channels=train_ds.describe()["in_channels"],
                              prior_dim=train_ds.prior_dim)
    model = model.to(device)
    logger.info(f"模型 {minfo['name']}｜总参数 {minfo['n_params_total']:,}"
                f"｜其中条件分支 {minfo['n_params_conditional']:,}"
                f"｜主干 {minfo['n_params_main']:,}｜film_in_dim {minfo.get('film_in_dim')}")
    ic = identity_check(model)
    if ic is not None:
        logger.info(f"FiLM identity 自检: γ 最大偏差 {ic['max_abs_gamma_minus_1']:.2e}，"
                    f"β 最大 {ic['max_abs_beta']:.2e}，通过={bool(ic['ok'])}")

    # ---------- 优化器 / 调度 / EMA ----------
    epochs = 2 if args.smoke else int(args.epochs or tcfg["epochs"])
    opt = AdamW(model.parameters(), lr=float(tcfg["lr"]),
                weight_decay=float(tcfg.get("weight_decay", 1e-4)))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * int(tcfg.get("warmup_epochs", 0))
    loss_cfg = dict(tcfg.get("loss", {}))
    use_ema = bool(tcfg.get("ema", True))
    ema = EMA(model, float(tcfg.get("ema_decay", 0.999))) if use_ema else None
    amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema_eval_weights = str(cfg.get("eval", {}).get("weights", "ema"))
    if ema is None and ema_eval_weights == "ema":
        logger.warning("config 要求用 EMA 权重评测但训练未启用 EMA，回退为 raw")
        ema_eval_weights = "raw"

    # ---------- 运行目录记录 ----------
    runtime = {
        "exp_id": exp_id, "seed": seed, "device": str(device),
        "splits_hash": s_hash, "git_rev": git_rev(ROOT),
        "model_info": minfo, "train_ds": train_ds.describe(), "val_ds": val_ds.describe(),
        "n_train": len(train_ds), "n_val": len(val_ds),
        "db_bounds": list(M.db_bounds()), "eval_weights": ema_eval_weights,
        "smoke": bool(args.smoke), "limit_train": args.limit_train, "limit_val": args.limit_val,
        **describe_seed(seed, args.deterministic),
    }
    freeze_run_dir(run_dir, cfg, runtime)
    logger.info(f"运行目录 {run_dir}｜splits_hash {s_hash}｜eval_weights {ema_eval_weights}")

    # ---------- 训练循环 ----------
    history: list[dict] = []
    best = {"val_rmse_db": float("inf"), "epoch": -1}
    patience = int(tcfg.get("patience", 25))
    bad_epochs = 0
    gstep = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        sums: dict[str, float] = {}
        nb = 0
        for batch in train_loader:
            lr = lr_at(gstep, total_steps, warmup_steps, float(tcfg["lr"]))
            for g in opt.param_groups:
                g["lr"] = lr
            inputs = batch["inputs"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            prior_vec = batch["prior_vec"].to(device, non_blocking=True)
            cot_text = batch["cot_text"] if cot_enabled else None
            sat = batch["sat_mask"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                pred = model(inputs, prior_vec, cot_text)
                parts = masked_loss(pred, target, sat, loss_cfg)
            scaler.scale(parts["total"]).backward()
            if float(tcfg.get("grad_clip", 0.0)):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)

            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            sums["lr"] = sums.get("lr", 0.0) + lr
            nb += 1
            gstep += 1

        train_avg = {k: v / max(1, nb) for k, v in sums.items()}

        # ---- 验证：raw 权重 ----
        val_rows, val_agg = evaluate(model, val_loader, device, cot_enabled, args.limit_val)
        # ---- 验证：EMA 权重（训练时记录的"最好成绩"走这条路）----
        ema_rmse = None
        if ema is not None:
            ema.apply_to(model)
            _, ema_agg = evaluate(model, val_loader, device, cot_enabled, args.limit_val)
            ema.restore(model)
            ema_rmse = float(ema_agg["mean_rmse_db"])

        selected = ema_rmse if ema_eval_weights == "ema" else float(val_agg["mean_rmse_db"])
        is_best = selected < best["val_rmse_db"]
        if is_best:
            best = {"val_rmse_db": selected, "epoch": epoch}
            bad_epochs = 0
        else:
            bad_epochs += 1

        rec = {
            "epoch": epoch, "lr": train_avg.get("lr"),
            "train": {k: v for k, v in train_avg.items() if k != "lr"},
            "val": {k: float(v) for k, v in val_agg.items()},
            "ema_val_rmse_db": ema_rmse, "is_best": bool(is_best),
            "selected_val_rmse_db": selected,
        }
        history.append(rec)
        logger.info(
            f"epoch {epoch:>3}/{epochs}｜lr {train_avg.get('lr', 0):.2e}"
            f"｜train total {train_avg.get('total', float('nan')):.4f}"
            f" ({train_avg.get('smooth_l1', 0):.3f}/{train_avg.get('grad', 0):.4f}/"
            f"{train_avg.get('ssim', 0):.4f})"
            f"｜val RMSE {val_agg['mean_rmse_db']:.3f} dB"
            f" (non-sat {val_agg.get('mean_rmse_db_nonsaturated', float('nan')):.3f})"
            f"｜MAE {val_agg['mean_mae_db']:.3f}｜SSIM {val_agg['mean_ssim']:.4f}"
            + (f"｜EMA {ema_rmse:.3f}" if ema_rmse is not None else "")
            + ("  ← best" if is_best else "")
        )

        if is_best:
            ckpt = {
                "model_state_dict": model.state_dict(),
                "ema_shadow": ema.state_dict() if ema is not None else None,
                "eval_weights": ema_eval_weights,
                "config": cfg, "model_cfg": base_model_cfg, "model_info": minfo,
                "prior_mode": pcfg["mode"], "prior_groups": list(prior_groups),
                "prior_dim": train_ds.prior_dim,
                "in_channels": train_ds.describe()["in_channels"],
                "sectors_channels": train_ds.sectors_channels,
                "shuffle_cot": bool(pcfg.get("shuffle_cot", False)),
                "zero_groups": list(pcfg.get("zero_groups", ()) or ()),
                "cot_mode": cot_mode, "cot_enabled": cot_enabled,
                "seed": seed, "epoch": epoch, "best_val_rmse_db": best["val_rmse_db"],
                "splits_hash": s_hash, "git_rev": runtime["git_rev"],
                "db_bounds": list(M.db_bounds()), "runtime": runtime,
            }
            ck_dir = run_dir / "ckpt"
            ck_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, ck_dir / "best.pth")
            logger.info(f"  保存 best checkpoint（epoch {epoch}, "
                        f"selected val RMSE {selected:.4f} dB, 权重口径 {ema_eval_weights}）")

        if patience > 0 and bad_epochs >= patience:
            logger.info(f"早停：连续 {bad_epochs} 个 epoch 没有改善（patience={patience}）")
            break

    elapsed = time.time() - t0
    summary = {
        "exp_id": exp_id, "config_name": cfg["_config_name"], "seed": seed,
        "best_epoch": best["epoch"], "best_val_rmse_db": best["val_rmse_db"],
        "eval_weights": ema_eval_weights, "epochs": history,
        "n_params_total": minfo["n_params_total"],
        "n_params_conditional": minfo["n_params_conditional"],
        "splits_hash": s_hash, "elapsed_sec": elapsed,
        "prior_mode": pcfg["mode"], "cot_mode": cot_mode,
    }
    (run_dir / "history.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"训练结束：{elapsed / 60:.1f} min｜best epoch {best['epoch']}"
                f"｜best val RMSE {best['val_rmse_db']:.4f} dB（口径 {ema_eval_weights}）")
    logger.info(f"history 写入 {run_dir / 'history.json'}")
    logger.info("下一步：python -m src.eval.run_eval --run-dir "
                f"{rel_or_abs(run_dir, ROOT)} --split test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
