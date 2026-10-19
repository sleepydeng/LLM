"""在指定 split 上评测一个 checkpoint，产出 `runs/<exp_id>/metrics.json`。

口径纪律（这是上一版最乱的地方）
-------------------------------
* **test maps 只在最后评一次**；如果要评 val，脚本会明确警告"val 参与过选模型，
  在上面报结果属于乐观偏差"（交接文档 §8.6 禁止项）。
* 权重口径必须显式：checkpoint 里同时有 `model_state_dict` 与 `ema_shadow`，
  上一版的 `if "model_state_dict" in state: ... elif "ema_shadow" in state:` 让
  EMA 分支**永远走不到**，于是"训练时记的是 EMA 成绩、评测用的是原始权重"。
  这里按 `--weights` 严格二选一，缺 EMA 时**报错而不是静默回退**。
* 逐样本指标与汇总一起落盘，供配对检验与"最坏样本"分析使用。

用法
----
    python -m src.eval.run_eval --run-dir runs/prior_real_seed0 --split test
    python -m src.eval.run_eval --run-dir runs/prior_real_seed0 --split test --weights ema
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from src.data.dataset import RadioMapSeerDataset, build_dataloader  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.train.train import evaluate  # noqa: E402
from src.utils.config import rel_or_abs  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def load_checkpoint(ckpt_path: Path, device: torch.device, which: str,
                    logger) -> tuple[torch.nn.Module, dict, dict]:
    """按显式口径载入模型。返回 (model, state, 载入信息)。"""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state.get("config") or {}
    model_cfg = state.get("model_cfg") or cfg.get("model") or {}
    if not model_cfg:
        raise KeyError(f"{ckpt_path} 里没有 model_cfg/config.model，无法重建模型")

    model, minfo = build_model(model_cfg, in_channels=int(state["in_channels"]),
                              prior_dim=int(state["prior_dim"]))
    if which == "ema":
        if state.get("ema_shadow") is None:
            raise RuntimeError(
                f"{ckpt_path} 里没有 ema_shadow，但请求了 --weights ema。"
                f"上一版就是在这里静默回退到原始权重、导致两份数字对不上；"
                f"这里选择直接报错。请改用 --weights raw，或重新训练（config train.ema=true）。"
            )
        sd = {k: v.to(next(iter(model.state_dict().values())).dtype)
              for k, v in state["ema_shadow"].items()}
        info = {"weights": "ema", "n_keys_loaded": len(sd)}
    else:
        sd = state["model_state_dict"]
        info = {"weights": "raw", "n_keys_loaded": len(sd)}
    model.load_state_dict(sd, strict=True)          # strict=True 让不匹配立刻暴露
    model = model.to(device).eval()
    info["model_info"] = minfo
    info["ckpt_epoch"] = state.get("epoch")
    info["ckpt_best_val_rmse_db"] = state.get("best_val_rmse_db")
    info["ckpt_eval_weights"] = state.get("eval_weights")
    return model, state, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="评测一个 run 的 checkpoint")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--weights", default="ema", choices=["ema", "raw"],
                    help="必须显式选择；缺 EMA 会报错而不是静默回退")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 个样本（调试）")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=None, help="默认 runs/<exp_id>/metrics_<split>.json")
    args = ap.parse_args(argv)

    run_dir = (ROOT / args.run_dir) if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    logger = get_logger("eval", run_dir / "eval.log")
    ckpt_path = run_dir / "ckpt" / "best.pth"
    if not ckpt_path.is_file():
        logger.error(f"找不到 checkpoint: {ckpt_path}")
        return 2

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, state, load_info = load_checkpoint(ckpt_path, device, args.weights, logger)
    cfg = state.get("config") or {}
    set_seed(int(state.get("seed", 0)), deterministic=True)

    if args.split != "test":
        logger.warning(f"正在 {args.split} 上评测。注意 {args.split} 参与过选模型，"
                       f"在此报告结果属于乐观偏差；论文里的结论只能用 test（AGENTS 规则 3）。")

    dcfg = cfg.get("data", {})
    pcfg = cfg.get("prior", {})
    prior_mode = state.get("prior_mode", pcfg.get("mode", "none"))
    prior_dir = None
    if prior_mode in ("real", "shuffled"):
        prior_dir = ROOT / pcfg["dir"]

    ds = RadioMapSeerDataset(
        data_root=ROOT / dcfg.get("root", "data/radiomapseer"),
        splits_path=ROOT / dcfg.get("splits", "data/splits.json"),
        split=args.split, prior_mode=prior_mode, prior_dir=prior_dir,
        prior_groups=tuple(state.get("prior_groups", ["global_num", "global_cat", "sectors"])),
        sectors_channels=bool(state.get("sectors_channels", False)),
        shuffle_seed=int(pcfg.get("shuffle_seed", 12345)),
        # 必须与训练时一致，否则 shuffled 对照会被评成一个不同的实验
        shuffle_cot=bool(state.get("shuffle_cot", pcfg.get("shuffle_cot", False))),
        # 容量对齐对照：与训练时同样的置零分组，否则评测会把置零组当成真值
        zero_groups=tuple(state.get("zero_groups", pcfg.get("zero_groups", ())) or ()),
        freq_ghz=float(dcfg.get("freq_ghz", 0.9)),
        px_per_m=float(dcfg.get("px_per_m", 1.0)),
        tx_encoding=str(dcfg.get("tx_encoding", "constant")),
        return_cot_text=bool(state.get("cot_enabled", False)),
    )
    if args.limit:
        ds.samples = ds.samples[: args.limit]
        if ds.prior_vectors is not None:
            ds.prior_vectors = ds.prior_vectors[: args.limit]
        if ds.prior_shuffle_index is not None:
            ds.prior_shuffle_index = ds.prior_shuffle_index[: args.limit]
        if ds.cot_shuffle_index is not None:
            ds.cot_shuffle_index = ds.cot_shuffle_index[: args.limit]
    loader = build_dataloader(ds, args.batch_size, False, 0, seed=0)

    logger.info(f"评测 {run_dir.name}｜split={args.split}｜{ds.describe()}")
    logger.info(f"载入权重口径 {load_info['weights']}"
                f"（ckpt 里记录的口径是 {load_info['ckpt_eval_weights']}）"
                f"｜ckpt epoch {load_info['ckpt_epoch']}"
                f"｜ckpt best val RMSE {load_info['ckpt_best_val_rmse_db']}")

    rows, agg = evaluate(model, loader, device, bool(state.get("cot_enabled", False)))
    agg = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in agg.items()}

    metrics = {
        "exp_id": run_dir.name,
        "config_name": (cfg.get("_config_name")
                        or run_dir.name.rsplit("_seed", 1)[0]),
        "seed": int(state.get("seed", 0)),
        "split": args.split,
        "weights": load_info["weights"],
        "n_samples": len(rows),
        "aggregate": agg,
        "per_sample": rows,
        "meta": {
            "prior_mode": prior_mode,
            "cot_mode": state.get("cot_mode", "none"),
            "n_params_total": int(load_info["model_info"]["n_params_total"]),
            "n_params_conditional": int(load_info["model_info"]["n_params_conditional"]),
            "model": load_info["model_info"],
            "ckpt": rel_or_abs(ckpt_path, ROOT),
            "ckpt_epoch": load_info["ckpt_epoch"],
            "ckpt_best_val_rmse_db": load_info["ckpt_best_val_rmse_db"],
            "splits_hash": state.get("splits_hash"),
            "db_bounds": list(M.db_bounds()),
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "split_used_for_model_selection": args.split != "test",
        },
    }
    out = Path(args.out) if args.out else run_dir / f"metrics_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"RMSE {agg['mean_rmse_db']:.4f} ± {agg['std_rmse_db']:.4f} dB"
                f"（排除饱和像素 {agg.get('mean_rmse_db_nonsaturated', float('nan')):.4f} dB）")
    logger.info(f"MAE {agg['mean_mae_db']:.4f} dB｜SSIM {agg['mean_ssim']:.4f}"
                f"｜grad_err {agg['mean_grad_error']:.5f}"
                f"｜归一化 RMSE {agg['mean_rmse_norm']:.5f}")
    logger.info(f"写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
