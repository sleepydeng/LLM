#!/usr/bin/env python3
"""门槛区分度自检：确认 audit 的每条硬门槛都**因为正确的原因**通过/失败。

为什么需要这个测试
-----------------
`reports/audit_legacy_v2.json` 只证明了门槛会在**彻底坍缩**的上一版先验上 FAIL。
但那只说明"门槛会亮红灯"，没说明两件更要紧的事：

  1. **阈值是否可达**？如果 `unique_ratio >= 0.5`、`sector_consistency >= 0.7`
     这些要求一份正常工作的 prompt 根本达不到，那么试生成会永远卡在门口，
     白白烧掉整轮 API 调用与时间 —— 这是当前最贵的失败模式。
  2. **门槛是否只对正确的病敏感**？如果某个数据集的多样性不错、却因为
     某个无关指标误判 FAIL，我们会去改一个本来没问题的 prompt。

所以这里构造六份"只犯一种病"的先验集，逐条核对：

    A  健康（sectors 来自真实几何 + 逐 Tx 变化的数值）   → 全部硬门槛 PASS
    B  每个 map 内所有字段恒定（= 上一版的病）           → unique_ratio / intra_map FAIL
    C  confidence 全是 medium（类别字段退化成常量）       → categorical_min_unique FAIL
    D  向量多样但 sectors 与几何随机不符                 → sector_consistency FAIL
    E  单个数值字段（penetration_loss_db）恒定            → numeric_unique/entropy FAIL
    F  恒答 "blocked"（偷懒策略）                        → sector_consistency_gain FAIL

F 组是这次新增的：实测 maps 0-3 的真值分布里 `blocked` 占 59.3%、`open` 仅 0.7%，
于是"永远回答 blocked"的**绝对**一致率就有 0.593。绝对门槛 0.70 在这批数据上还能拦住它，
余量却只有 0.107 —— 换一批建筑更密的地图就会越线。自校准的 gain 判据
（本 Tx 一致率 − 同图另一 Tx 一致率）对任何偷懒策略必然 ≈ 0，所以 F 组必须被它拦住。

运行：python -m tests.test_gate
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
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

from src.data.sectors import geometric_obstruction, local_building_ratio  # noqa: E402
from src.prior import schema as S  # noqa: E402
from src.prior.audit import main as audit_main  # noqa: E402

DATA_ROOT = ROOT / "data" / "radiomapseer"
MAPS = [0, 1, 2, 3, 7, 11]      # 6 张地图
TX_PER_MAP = 12                  # 每张 12 个 Tx → 72 个样本


def _geometry(map_id: int, tx_id: int) -> tuple[dict, float, float]:
    """返回 (几何遮挡, 50m 建筑占比, 100m 建筑占比)。"""
    mask = np.array(Image.open(DATA_ROOT / "buildings_complete" / f"{map_id}.png").convert("L"))
    tx = json.loads((DATA_ROOT / "antenna" / f"{map_id}.json").read_text(encoding="utf-8"))[tx_id]
    geo = geometric_obstruction(mask, tx, radius_m=100.0, px_per_m=1.0)
    return geo, local_building_ratio(mask, tx, 50.0), local_building_ratio(mask, tx, 100.0)


def _blank_sectors() -> list[dict]:
    return [{"azimuth_deg": a, "obstruction": "open", "nearest_building_m": 500.0}
            for a in S.AZIMUTHS]


def build_set(out: Path, kind: str) -> None:
    """按 kind 生成一份先验集。

    A 健康；B 每个 map 内所有字段恒定；C confidence 全是 medium；
    D sectors 与几何随机不符；E 单个数值字段（penetration_loss_db）恒定。
    """
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(20260101)
    per_map_geo = {m: _geometry(m, 0) for m in MAPS}
    # "B" 组要求同一 map 内一切恒定，所以任何按 Tx 取值的东西都要退化成 map 级
    jitter = kind != "B"

    def noise(scale: float) -> float:
        """需要"逐 Tx 变化"时才加扰动；否则返回 0，保证 map 内恒定。"""
        return float(rng.normal(0, scale)) if jitter else 0.0

    for map_id in MAPS:
        for tx_id in range(TX_PER_MAP):
            if jitter:
                geo, r50, r100 = _geometry(map_id, tx_id)
            else:
                geo, r50, r100 = per_map_geo[map_id]

            # ---- sectors ----
            sectors = []
            for az in S.AZIMUTHS:
                if kind == "D":
                    ob = S.OBSTRUCTIONS[rng.randint(0, 4)]      # 病：与几何无关
                    nb = float(rng.uniform(0, 300))
                elif kind == "F":
                    # 病：偷懒策略 —— 恒答真值里最常见的标签，完全不看图
                    ob = "blocked"
                    nb = 2.0
                else:
                    ob = str(geo[az]["obstruction"])
                    if jitter and rng.rand() < 0.15:
                        i = S.OBSTRUCTIONS.index(ob)
                        ob = S.OBSTRUCTIONS[min(3, max(0, i + int(rng.choice([-1, 1]))))]
                    nb = float(geo[az]["nearest_building_m"])
                    if jitter and np.isfinite(nb) and nb < 400:
                        nb = float(np.clip(nb + rng.normal(0, 6.0), 0, 500))
                sectors.append({"azimuth_deg": az, "obstruction": ob,
                                "nearest_building_m": nb})

            # ---- 四个传播机制权重：先取受几何影响的原始权重，再**归一化** ----
            # 初版写成 scat = 1 - los - refl - diff 再 clip，结果 scat 几乎恒被裁到
            # 下限、退化成一个常量，反而把"健康集"判成了坍缩。归一化才是对的。
            raw = np.array([
                max(0.05, (1.2 - 1.0 * r100) + noise(0.08)),   # los：越密越少
                max(0.05, 0.60 + noise(0.08)),                  # 反射
                max(0.05, (0.50 + 0.30 * r50) + noise(0.08)),   # 衍射：越密越多
                max(0.05, 0.30 + noise(0.08)),                  # 散射
            ])
            los, refl, diff, scat = (raw / raw.sum()).tolist()

            perm = float(np.clip(3.0 + 6.0 * r100 + noise(0.30), 1.5, 15))
            pen = 12.0 if kind in ("E", "G") else float(
                np.clip(4.0 + 20.0 * r100 + noise(0.80), 0.5, 40))
            # E 组：让一个**几何条件**数值字段恒定（该字段必须逐样本变化）
            # G 组：让一个**材料属性**字段恒定（在单一材质数据集上是正确答案）
            pl_max = 130.0 if kind == "E" else float(
                np.clip(120 + r100 * 60 + noise(2.0), 61, 249))

            prior = {
                "scene_type": ("dense_urban" if r100 > 0.34 else
                               "urban" if r100 > 0.22 else
                               "suburban" if r100 > 0.12 else "rural"),
                "los_ratio": los, "reflection_ratio": refl,
                "diffraction_ratio": diff, "scattering_ratio": scat,
                "dominant_material": "concrete" if r100 > 0.25 else "brick",
                "permittivity": perm, "penetration_loss_db": pen,
                "pathloss_min_db": float(np.clip(50 + r50 * 30 + noise(1.0), 30, 199)),
                "pathloss_max_db": pl_max,
                # C 组：所有样本都写 medium，制造类别字段退化成常量
                # B 组：只用 map_id 决定，保证 map 内恒定
                "confidence": "medium" if kind == "C" else
                              S.CONFIDENCES[((0 if not jitter else tx_id) + map_id) % 3],
                "tx_local_building_ratio_50m": float(np.clip(r50 + noise(0.02), 0, 1)),
                "tx_local_building_ratio_100m": float(np.clip(r100 + noise(0.02), 0, 1)),
                "sectors": sectors,
                "reasoning": (f"map{map_id} tx{tx_id}: at 0.9 GHz the 100 m disc around the "
                              f"transmitter is {r100:.3f} built-up; the 50 m disc is "
                              f"{r50:.3f}. Azimuth 0 deg has obstruction "
                              f"{sectors[0]['obstruction']} at "
                              f"{sectors[0]['nearest_building_m']:.1f} m; azimuth 180 deg has "
                              f"{sectors[4]['obstruction']} at "
                              f"{sectors[4]['nearest_building_m']:.1f} m."),
            }
            rec = {"schema_version": "v2", "sample_id": f"{map_id}_{tx_id}",
                   "map_id": map_id, "tx_id": tx_id, "prior": prior,
                   "provenance": {"synthetic": True, "for_gate_test_only": True,
                                  "field_consistency": {"overall": 0.9}}}
            (out / f"{map_id}_{tx_id}.json").write_text(
                json.dumps(rec, ensure_ascii=False), encoding="utf-8")


def run_audit(priors: Path, out_json: Path) -> tuple[int, dict]:
    rc = audit_main(["--priors", str(priors), "--data", str(DATA_ROOT),
                     "--out", str(out_json)])
    return rc, json.loads(out_json.read_text(encoding="utf-8"))


def gate_status(rep: dict) -> dict[str, bool | None]:
    return {g["gate"]: g["pass"] for g in rep.get("gate", {}).get("results", [])}


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="radiomap_gate_"))
    failures: list[str] = []
    try:
        summary = []
        for kind, expect_fail_keys, expect_pass_all in (
            ("A", [], True),
            ("B", ["unique_ratio", "intra_map_unique_ratio_mean"], False),
            ("C", ["categorical_min_unique", "confidence_top1_share"], False),
            ("D", ["sector_consistency_strict", "sector_consistency_gain"], False),
            ("E", ["numeric_unique_min", "numeric_entropy_min"], False),
            ("F", ["sector_consistency_gain"], False),
            # G 组反向验证：只让**材料属性**字段（穿透损耗）恒定。
            # 在"全是混凝土城市"的数据集上，0.9 GHz 下混凝土的穿透损耗本来就该是固定值，
            # 恒定是**正确答案**，所以这条**不允许**把数值门槛判成 FAIL。
            ("G", [], True),
        ):
            priors = tmp / f"priors_{kind}"
            build_set(priors, kind)
            rc, rep = run_audit(priors, tmp / f"audit_{kind}.json")
            st = gate_status(rep)
            hard_names = ("unique_ratio", "intra_map_unique_ratio_mean",
                          "numeric_unique_min", "numeric_entropy_min",
                          "categorical_min_unique", "sector_consistency_strict",
                          "sector_consistency_gain")
            hard = {k: v for k, v in st.items() if k in hard_names}
            summary.append((kind, rc, rep, st))

            print(f"\n===== 集合 {kind} =====")
            print(f"  唯一向量 {rep['n_unique_vector']}/{rep['n_files']}"
                  f"（unique_ratio {rep['unique_ratio']:.3f}）"
                  f"｜同 map 内唯一占比 {rep['intra_map_unique_ratio_mean']:.3f}")
            if "sector_check" in rep:
                sc = rep["sector_check"]
                print(f"  sectors：本 Tx 严格 {sc['agreement_strict']:.3f}"
                      f"｜错位对照 {sc.get('agreement_strict_shifted', float('nan')):.3f}"
                      f"｜**增益 {sc.get('agreement_gain', float('nan')):+.3f}**"
                      f"（相邻容差 {sc['agreement_adjacent']:.3f}）")
            num = rep.get("numeric", {})
            cat = rep.get("categorical", {})
            print(f"  数值字段 唯一值最少 {num.get('unique_min')}"
                  f"｜熵最低 {None if num.get('entropy_min') is None else round(num['entropy_min'], 2)}"
                  f"｜类别字段 取值数最少 {cat.get('unique_min')}")
            print(f"  硬门槛：{hard}")

            if expect_pass_all:
                bad = [g["gate"] for g in rep["gate"]["results"]
                       if g["severity"] == "fail" and g["pass"] is False]
                if bad or rc != 0:
                    detail = []
                    for name in bad:
                        rec = next(g for g in rep["gate"]["results"] if g["gate"] == name)
                        detail.append(f"{name}={rec['value']}({rec.get('note','')})")
                    failures.append(
                        f"集合 A（健康先验）本应全部通过，实际未通过：{'；'.join(detail)}。"
                        f"这说明门槛阈值**不可达**，试生成会永远卡住 —— 必须先调阈值")
            else:
                if rep["gate"]["pass"] or rc == 0:
                    failures.append(f"集合 {kind} 本应 FAIL，实际通过了 —— 门槛对该病不敏感")
                for key in expect_fail_keys:
                    if st.get(key) is not False:
                        failures.append(f"集合 {kind} 的 {key} 本应 FAIL，实际 {st.get(key)}")

        # 额外：确认 A 的 sector 一致率明显高于随机猜测（4 类约 0.25）
        strict_a = summary[0][2]["sector_check"]["agreement_strict"]
        strict_d = summary[3][2]["sector_check"]["agreement_strict"]
        gain_a = summary[0][2]["sector_check"]["agreement_gain"]
        gain_f = summary[5][2]["sector_check"]["agreement_gain"]
        print(f"\n区分度：健康集一致率 {strict_a:.3f} vs 随机 sectors 集 {strict_d:.3f}"
              f"（4 类随机猜测期望约 0.25）")
        print(f"自校准增益：健康集 {gain_a:+.3f} vs 恒答 blocked（偷懒）{gain_f:+.3f}")
        if not (strict_a >= 0.70 > strict_d):
            failures.append(
                f"sector 一致率没有区分度：健康 {strict_a:.3f}、随机 {strict_d:.3f}")
        # 偷懒策略的自动校准增益必须接近 0（这正是引入该判据的理由）
        if not (gain_f <= 0.02 < gain_a):
            failures.append(
                f"自校准增益没有区分度：健康 {gain_a:+.3f}、偷懒 {gain_f:+.3f}；"
                f"偷懒策略的增益必须接近 0")

        # G 组：材料属性恒定必须被**容忍**（不能因为它恒定就判坍缩）
        st_g = summary[6][3]
        for key in ("numeric_unique_min", "numeric_entropy_min"):
            if st_g.get(key) is not True:
                failures.append(
                    f"集合 G（只有材料属性字段恒定）的 {key} 本应 PASS，实际 {st_g.get(key)} —— "
                    f"把'物理上该恒定'误判成了坍缩")
        rep_g = summary[6][2]
        if rep_g.get("material_numeric", {}).get("unique_min") != 1:
            failures.append("集合 G 的穿透损耗应只有 1 个取值，构造有误")

        print()
        if failures:
            print("门槛区分度自检失败：")
            for f in failures:
                print("  -", f)
            return 1
        print("门槛区分度自检通过 [OK]：健康集全过、每种病都被对应的门槛精确抓住，"
              "且阈值可达（不会把正常 prompt 永远挡在门外）")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
