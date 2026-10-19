#!/usr/bin/env python3
"""端到端流水线自检：在 CPU 上把 train.py 与 run_eval.py 的真实命令行跑通。

它验证的不是模型效果，而是**代码链路与产物契约**：
    config → dataset → model → 训练 → best ckpt → 按显式口径载入 → test 评测 → metrics_test.json

关键设计（有意为之）
------------------
* 全部假数据（小划分 + 合成先验 + 运行目录）都在**临时目录**里，结束时整个删掉。
  上一版最危险的一个做法是"先验缓存未命中就规则生成并写回先验目录"
  （`model/radiomapseer_dataset.py:332-344`），结果程序造的假先验混进了真先验目录，
  让"先验质量"这件事事后无法审计。这里从结构上避免同一件事。
* 四种配置都跑一遍，并检查 **masked 与 real 的参数量逐位相同** —— 这是
  `real − masked` 可归因的前提，也检查 prior_none 确实更小。
* 检查 `metrics_test.json` 里的口径字段齐全（含 non-saturated RMSE、db_bounds、
  weights），因为报告脚本要靠这些字段自动生成表格。

运行：python -m tests.test_pipeline
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from src.prior import schema as S  # noqa: E402
from src.utils.config import deep_merge, load_config  # noqa: E402

DATA_ROOT = ROOT / "data" / "radiomapseer"
N_TX = 4                       # 每张地图只用 4 个 Tx，够跑通就行
CONFIGS = [
    ("prior_none", {"model": {"name": "unet"}, "prior": {"mode": "none"}}),
    ("prior_masked", {"model": {"name": "unet_film"}, "prior": {"mode": "masked"}}),
    ("prior_real", {"model": {"name": "unet_film"}, "prior": {"mode": "real"}}),
    ("prior_shuffled", {"model": {"name": "unet_film"},
                        "prior": {"mode": "shuffled", "shuffle_seed": 12345}}),
]


def build_tiny_workspace(tmp: Path) -> dict:
    """造小划分 + 合成先验，返回供 config 覆盖用的路径字典。"""
    data = tmp / "data"
    (data / "buildings_complete").mkdir(parents=True)
    (data / "gain" / "carsDPM").mkdir(parents=True)
    (data / "antenna").mkdir(parents=True)
    for m in (0, 1, 2):
        shutil.copy(DATA_ROOT / "buildings_complete" / f"{m}.png",
                    data / "buildings_complete" / f"{m}.png")
        ants = json.loads((DATA_ROOT / "antenna" / f"{m}.json").read_text(encoding="utf-8"))
        (data / "antenna" / f"{m}.json").write_text(json.dumps(ants[:N_TX]), encoding="utf-8")
        for t in range(N_TX):
            shutil.copy(DATA_ROOT / "gain" / "carsDPM" / f"{m}_{t}.png",
                        data / "gain" / "carsDPM" / f"{m}_{t}.png")

    splits = {
        "seed": 20260101, "split_unit": "map", "n_tx_per_map": N_TX, "all_maps": [0, 1, 2],
        "train_maps": [0], "val_maps": [1], "test_maps": [2],
        "counts": {"train_samples": N_TX, "val_samples": N_TX, "test_samples": N_TX},
    }
    splits_path = tmp / "splits.json"
    splits_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    # ---- 合成先验（明确标注是假的，且只存在于临时目录）----
    priors = tmp / "priors_synthetic"
    priors.mkdir()
    for m in range(3):
        for t in range(N_TX):
            rng = np.random.RandomState(m * 100 + t)
            sectors = []
            for i, az in enumerate(S.AZIMUTHS):
                # 让 0 度附近的遮挡随 tx 变化，模拟"随 Tx 变化的空间先验"
                near = 5.0 + ((t * 13 + i * 7) % 60)
                obs = S.OBSTRUCTIONS[min(3, (t + i) % 4)]
                sectors.append({"azimuth_deg": az, "obstruction": obs,
                                "nearest_building_m": float(near)})
            prior = {
                "scene_type": S.SCENE_TYPES[(m + t) % 5],
                "los_ratio": float(rng.uniform(0.1, 0.7)),
                "reflection_ratio": float(rng.uniform(0.1, 0.4)),
                "diffraction_ratio": float(rng.uniform(0.1, 0.4)),
                "scattering_ratio": float(rng.uniform(0.05, 0.3)),
                "dominant_material": S.MATERIALS[(m + t) % 6],
                "permittivity": float(rng.uniform(2, 12)),
                "penetration_loss_db": float(rng.uniform(3, 30)),
                "pathloss_min_db": float(rng.uniform(45, 70)),
                "pathloss_max_db": float(rng.uniform(130, 190)),
                "confidence": S.CONFIDENCES[(m + t) % 3],
                "tx_local_building_ratio_50m": float(rng.uniform(0, 1)),
                "tx_local_building_ratio_100m": float(rng.uniform(0, 1)),
                "sectors": sectors,
                "reasoning": f"合成自检先验 map{m} tx{t}：0.9GHz，逐方位遮挡见 sectors。",
            }
            rec = {"schema_version": "v2", "sample_id": f"{m}_{t}", "map_id": m, "tx_id": t,
                   "prior": prior,
                   "provenance": {"synthetic": True, "for_pipeline_test_only": True,
                                  "field_consistency": {"overall": 0.9}}}
            (priors / f"{m}_{t}.json").write_text(
                json.dumps(rec, ensure_ascii=False), encoding="utf-8")

    return {"data_root": str(data), "splits": str(splits_path), "priors": str(priors)}


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="radiomap_pipeline_"))
    failures: list[str] = []
    rows: list[tuple[str, float, float, float, int]] = []
    try:
        ws = build_tiny_workspace(tmp)
        runs_dir = tmp / "runs"
        base = load_config(ROOT / "configs" / "base.yaml")
        base.pop("_base_", None)

        from src.eval.run_eval import main as run_eval_main
        from src.train.train import main as train_main

        for name, override in CONFIGS:
            cfg = deep_merge(base, override)
            cfg["runs_dir"] = str(runs_dir)
            cfg["data"] = deep_merge(cfg["data"], {
                "root": ws["data_root"], "splits": ws["splits"], "num_workers": 0})
            cfg["prior"] = deep_merge(cfg["prior"], {"dir": ws["priors"]})
            cfg["train"] = deep_merge(cfg["train"], {"epochs": 2, "batch_size": N_TX})
            cfg.pop("_config_path", None)
            cfg.pop("_config_name", None)
            cfg_path = tmp / f"{name}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                                encoding="utf-8")

            print(f"\n{'=' * 70}\n[{name}] 训练\n{'=' * 70}")
            rc = train_main(["--config", str(cfg_path), "--seed", "0", "--smoke",
                             "--device", "cpu"])
            if rc != 0:
                failures.append(f"{name}: train 退出码 {rc}")
                continue

            run_dir = runs_dir / f"{name}_seed0"
            hist = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
            if len(hist["epochs"]) != 2:
                failures.append(f"{name}: history 应有 2 个 epoch，实际 {len(hist['epochs'])}")
            if not (run_dir / "ckpt" / "best.pth").is_file():
                failures.append(f"{name}: 没有产出 best.pth")
            if not (run_dir / "config.yaml").is_file():
                failures.append(f"{name}: 没有冻结 config.yaml")

            print(f"\n[{name}] 在 test split 上评测")
            rc = run_eval_main(["--run-dir", str(run_dir), "--split", "test",
                                "--weights", "ema", "--device", "cpu",
                                "--batch-size", str(N_TX)])
            if rc != 0:
                failures.append(f"{name}: run_eval 退出码 {rc}")
                continue

            m = json.loads((run_dir / "metrics_test.json").read_text(encoding="utf-8"))
            agg = m["aggregate"]
            for key in ("mean_rmse_db", "mean_mae_db", "mean_ssim",
                        "mean_rmse_db_nonsaturated", "mean_rmse_norm", "mean_grad_error"):
                if key not in agg or not np.isfinite(agg[key]):
                    failures.append(f"{name}: metrics 缺少或不有限 {key}")
            for key in ("db_bounds", "prior_mode", "n_params_total"):
                if key not in m["meta"]:
                    failures.append(f"{name}: meta 缺少 {key}")
            if m.get("weights") != "ema":
                failures.append(f"{name}: 权重口径不是 ema（实际 {m.get('weights')!r}）")
            if m["n_samples"] != N_TX:
                failures.append(f"{name}: 样本数 {m['n_samples']} != {N_TX}")
            if len(m["per_sample"]) != N_TX:
                failures.append(f"{name}: 逐样本行数不对")
            if m["meta"]["split_used_for_model_selection"]:
                failures.append(f"{name}: test 被标记成了参与选模型")

            rows.append((name, agg["mean_rmse_db"], agg["mean_mae_db"],
                         agg["mean_ssim"], m["meta"]["n_params_total"]))

        # ---- 参数量对齐检查 ----
        pm = {n: p for n, _r, _m, _s, p in rows}
        if "prior_masked" in pm and "prior_real" in pm:
            if pm["prior_masked"] != pm["prior_real"]:
                failures.append(
                    f"参数量未对齐：masked {pm['prior_masked']} vs real {pm['prior_real']}；"
                    f"real − masked 将无法归因于先验信息")
        if "prior_none" in pm and "prior_real" in pm:
            if not pm["prior_none"] < pm["prior_real"]:
                failures.append("prior_none 不该有 FiLM 分支，参数量应更小")

        print(f"\n{'=' * 70}\n四配置流水线结果（2 epoch 的**冒烟**数字，无任何科学含义）\n{'=' * 70}")
        print(f"{'config':<16}{'RMSE(dB)':>12}{'MAE(dB)':>12}{'SSIM':>10}{'参数量':>14}")
        for n, r, ma, s, p in rows:
            print(f"{n:<16}{r:>12.3f}{ma:>12.3f}{s:>10.4f}{p:>14,}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'=' * 70}")
    if failures:
        print("端到端自检失败：")
        for f in failures:
            print("  -", f)
        return 1
    print("端到端自检通过 [OK]：四配置的 训练→ckpt→按 EMA 口径评测→metrics 契约 全部走通")
    return 0


if __name__ == "__main__":
    sys.exit(main())
