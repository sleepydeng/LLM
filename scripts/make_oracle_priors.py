#!/usr/bin/env python3
"""构造 oracle 先验（阳性对照）：把 LLM 的 22 个 EM 维度替换成**目标 pathloss 图**
的真实统计量。

为什么需要它（而不只靠 `prior_geometry_only`）
-----------------------------------------------
`geometry_only > none` 证明的是"程序几何先验 + FiLM 通路"有用；但它用的是
sectors/local_ratio 这 42 维，**不是** LLM 占用的那 22 个 EM 维度。所以它无法回答：
"那 22 个 EM 维度本身能不能携带信号"。

这个 oracle 直接把**真值统计量**灌进那 22 维：
  * `pathloss_min_db` / `pathloss_max_db` ← 真值 pathloss 的最小/最大值（dB）
  * `los/reflection/diffraction/scattering_ratio` ← 真值 pathloss 落在四个区间的占比
  * `permittivity` / `penetration_loss_db` ← 真值 pathloss 的均值 / 标准差（映射到量程）
  * `scene_type` / `dominant_material` ← 由真值均值粗分档

判读（这是关键的三分结果）：
  * oracle **明显好于** masked_em，且 real_v3 ≈ masked_em
        → 判据与通路都没坏，只是 **LLM 产出的 EM 数值本身没有信息**（负面结果成立）。
  * oracle 也 ≈ masked_em
        → 那是"FiLM 通路断了 / 判据没区分度"的**警报**，必须回头查，而不是发表负面结果。

只读数据、零 GPU。输出目录 priors/oracle/（每个文件复制自真实先验、只替换 EM 字段，
sectors 与 tx_local_building_ratio 保留程序几何）。

用法：
    python scripts/make_oracle_priors.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from src.data.normalize import gain_to_pathloss_norm, pathloss_norm_to_db  # noqa: E402

DATA = ROOT / "data" / "radiomapseer"
REAL = ROOT / "priors" / "prompt_v3" / "glm-4v-flash"
OUT = ROOT / "priors" / "oracle"


def oracle_fields(pl_norm: np.ndarray) -> dict:
    """真值 pathloss 归一化图 → oracle 的 22 维 EM 字段（全部符合 schema 量程）。

    关键：只用**随样本变化**的统计量。实测（160 个样本）：
      * 全局 min/max 恒为 40/200 dB（每张图都有强信号像素与饱和像素）→ 不能用；
      * sat_frac（饱和占比）0.34–0.78、ns_mean（非饱和均值）0.74–0.90、
        ns_std（非饱和标准差）0.11–0.19、f_lt_06（强信号占比）0.01–0.15 都在变。
    这些才是"真值里、但模型要费劲才能从掩码+Tx 推出来"的全局摘要。
    """
    pl = pl_norm.astype(np.float64)
    sat = pl >= 1.0 - 1e-7
    ns = pl[~sat]
    sat_frac = float(sat.mean())
    ns_mean = float(ns.mean()) if len(ns) else 0.0
    ns_std = float(ns.std()) if len(ns) else 0.0
    ns_min = float(ns.min()) if len(ns) else 0.0
    ns_med = float(np.median(ns)) if len(ns) else 0.0

    # 四个机制权重 = 强信号占比 / 中段占比 / 高段占比 / 饱和占比（各自 ∈[0,1]）
    los = float((pl < 0.40).mean())
    refl = float(((pl >= 0.40) & (pl < 0.60)).mean())
    diff = float(((pl >= 0.60) & (pl < 0.85)).mean())
    scat = sat_frac
    # 材料/穿透：均值与标准差映射到量程
    perm = float(np.clip(1.0 + 19.0 * ns_mean, 1.0, 20.0))
    pen = float(np.clip(60.0 * ns_std, 0.0, 60.0))
    plmin = float(np.clip(pathloss_norm_to_db(np.array(ns_min)), 30.0, 199.0))
    plmax = float(np.clip(pathloss_norm_to_db(np.array(ns_med + 2.0 * ns_std)),
                          61.0, 250.0))
    if plmax <= plmin:
        plmax = plmin + 1.0
    scene = ("dense_urban" if sat_frac > 0.62 else "urban" if sat_frac > 0.52
             else "suburban" if sat_frac > 0.42 else "rural")
    mat = "concrete" if ns_mean > 0.82 else "brick"
    return {
        "scene_type": scene,
        "los_ratio": float(np.clip(los, 0.0, 1.0)),
        "reflection_ratio": float(np.clip(refl, 0.0, 1.0)),
        "diffraction_ratio": float(np.clip(diff, 0.0, 1.0)),
        "scattering_ratio": float(np.clip(scat, 0.0, 1.0)),
        "dominant_material": mat,
        "permittivity": perm,
        "penetration_loss_db": pen,
        "pathloss_min_db": plmin,
        "pathloss_max_db": plmax,
        "confidence": "high",          # oracle 是完美先验
    }


def main() -> int:
    files = sorted(REAL.glob("*.json"))
    if not files:
        print(f"真实先验目录为空：{REAL}")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        m, t = rec["map_id"], rec["tx_id"]
        gain = np.array(Image.open(DATA / "gain" / "carsDPM" / f"{m}_{t}.png").convert("L"))
        pl = gain_to_pathloss_norm(gain)
        fields = oracle_fields(pl)
        # 只替换 EM 字段；sectors / tx_local_building_ratio_*（程序几何）原样保留
        rec["prior"].update(fields)
        rec["prior"]["reasoning"] = "oracle prior: EM fields encode ground-truth pathloss stats"
        rec["provenance"]["field_sources"] = {
            "em": "oracle", "sectors": "program", "local_ratios": "program"}
        rec["provenance"]["design"] = "v3_geometry_oracle"
        (OUT / f.name).write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        n += 1
    print(f"写入 {n} 条 oracle 先验 → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
