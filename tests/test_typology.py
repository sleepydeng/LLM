#!/usr/bin/env python3
"""L1 三仪器类型学的自检：确认**每个判据分支都会因为正确的原因被触发**。

为什么需要这个测试
----------------
`reports/llm_l1_typology.md` 是针对某一批真实先验算出来的一张表。表本身不能说明
下面这些事 —— 而它们全都是"会悄悄给出错误结论"的失败模式：

  1. **几何解析对不对？** `provenance.geometry_facts.facts_text` 是从文本里正则式解析的
     （因为这个脚本刻意不依赖 `data/radiomapseer/`）。解析错位不会报错，
     只会让所有 ρ 都变小，看上去像"LLM 没接地"。
  2. **扇形数组顺序**：`prior.sectors` 只保证覆盖 8 个方位角，**不保证数组顺序**。
     若按数组下标取，驱动量就会与方位错位 —— 而且这种错位极难察觉。
  3. **分类器的每个分支**：常数 / 噪声 / 真实且有残余信息 / 接地不稳（地图间混淆）/
     自信地错 / 幻觉 —— 六个分支必须各自被正确触发，否则报告会静默地贴错标签。
  4. **小样本保护**：实测同一个 `los_ratio` 在 24 条子集上被判"常数"、在 371 条全量上
     被判"真实且有残余信息"。报告必须显式标注"样本量不够，标签不可作定论"。
  5. **对拍机制本身能失败**：`--validate-against` 若永远返回"通过"，它就没有任何用。

所以这里用**合成先验记录**（几何与电磁字段的关系是我写死的）覆盖上面每一条，
全部不需要 GPU / API key / 数据盘。

运行：python -m tests.test_typology
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from scripts.probe_l1_typology import (  # noqa: E402
    LOW_N,
    build_typology,
    geometry_drivers,
    load_priors,
    main as typology_main,
    parse_facts_text,
)
from src.prior.schema import AZIMUTHS, EM_NUM_KEYS  # noqa: E402

PASS, FAIL = [], []


def check(name: str, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  [OK]   {name}")
    except Exception as exc:  # noqa: BLE001
        FAIL.append((name, exc))
        print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


def _facts_text(r50: float, r100: float, on_building: bool, secs: list[dict]) -> str:
    lines = [
        f"Building fraction inside the 50 m disc around the Tx: {r50:.3f}",
        f"Building fraction inside the 100 m disc around the Tx: {r100:.3f}",
        ("The Tx pixel itself lies ON a building footprint."
         if on_building else "The Tx pixel lies on open ground (not on a building)."),
        "Per-azimuth obstruction within 100 m "
        "(azimuth in degrees, obstruction class, distance to the first building):",
    ]
    for s in secs:
        lines.append(f"  {s['azimuth_deg']} deg: {s['obstruction']}, "
                     f"{s['nearest_building_m']:.0f} m")
    counts: dict[str, int] = {}
    for s in secs:
        counts[s["obstruction"]] = counts.get(s["obstruction"], 0) + 1
    lines.append("Count of azimuths by obstruction class: "
                 + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return "\n".join(lines)


#: 方位 → 与其所属建筑的距离**增量**。刻意每个方位都不同，这样"按数组下标取"
#: 与"按方位角取"会给出不同的驱动量 —— 顺序 bug 会被立刻抓住。
AZ_DELTA = {0: 0.0, 45: 3.0, 90: 7.0, 135: 18.0, 180: 31.0, 225: 48.0,
            270: 69.0, 315: 96.0}


def _azimuth_specs(tx_id: int, base_dist: float) -> dict[int, dict]:
    """按 tx 决定每个方位的遮挡类别与距离。

    设计要点（否则测不出东西）：
      * `min_dist` = 最近的那个 blocked 方位 = `base_dist + min(被挡方位的增量)`
        —— 随 tx 变化，所以 min_dist 有方差；
      * 每 4 个 Tx 里只有 2 个方位是 blocked → `n_blocked` 也随 tx 变化；
      * 类别按方位轮换，避免"所有方位同类别"把 sector 驱动量退化掉。
    """
    n_blocked = [3, 2, 4, 2][tx_id % 4]
    specs: dict[int, dict] = {}
    for az in AZIMUTHS:
        idx = int(az) // 45
        if idx < n_blocked:
            kind = "blocked"
        elif idx % 3 == 0:
            kind = "open"
        elif idx % 3 == 1:
            kind = "sparse"
        else:
            kind = "dense"
        delta = float(AZ_DELTA[int(az)])
        if kind == "blocked":
            d = base_dist + delta
        elif kind == "sparse":
            d = 120.0 + delta
        elif kind == "dense":
            d = 80.0 + delta
        else:                                   # open：扇区内没有建筑
            d = 500.0
        specs[int(az)] = {"azimuth_deg": int(az), "obstruction": kind,
                          "nearest_building_m": round(d, 1)}
    return specs


def synth_record(map_id: int, tx_id: int, *, sector_perm: list[int] | None = None,
                 r50: float = 0.30, r100: float = 0.25,
                 on_building: bool = True, freq_ghz: float = 0.9,
                 base_dist: float | None = None) -> dict:
    """构造一条 v3 先验记录：几何驱动量由调用方给定的 `base_dist` 决定。

    字段与几何的关系（这是测试的"真值"）：
        los_ratio         = min_dist / 300     → 与 min_dist 完美单调（预期 +）
        pathloss_max_db   = 250 - 20*n_blocked → 与 n_blocked 单调（预期 +，故方向反）
        reflection_ratio  = 0.5 + 0.2*sin(tx)  → 与几何无关
        diffraction/scattering/pathloss_min    → 常数
    """
    if base_dist is None:
        # 单调递增（不是取模循环）：同一张地图内 min_dist 与 tx 严格单调，
        # 这样"随 tx 变化"的驱动量与目标字段之间的关系在同一张图内不会被周期性混叠破坏。
        # n_blocked 仍然每 4 个 Tx 循环一次（这是**故意**的：两个驱动量彼此部分解耦）。
        base_dist = 5.0 + 2.0 * tx_id
    specs = _azimuth_specs(tx_id, float(base_dist))
    secs = [specs[int(az)] for az in AZIMUTHS]
    if sector_perm is not None:
        secs = [secs[i] for i in sector_perm]
    labels = [specs[int(az)]["obstruction"] for az in AZIMUTHS]
    dists = [specs[int(az)]["nearest_building_m"] for az in AZIMUTHS]
    n_blocked = labels.count("blocked")
    min_dist = min(dists)
    prior = {
        "scene_type": "urban",
        "los_ratio": float(min_dist / 300.0),
        "reflection_ratio": float(0.5 + 0.2 * np.sin(tx_id)),
        "diffraction_ratio": 0.3,
        "scattering_ratio": 0.15,
        "dominant_material": "brick",
        "permittivity": float(5.0 + 0.5 * np.sin(tx_id)),
        "penetration_loss_db": 10.0,
        "pathloss_min_db": 40.0,
        "pathloss_max_db": float(250.0 - 20.0 * n_blocked),
        "confidence": "high",
        "tx_local_building_ratio_50m": float(r50),
        "tx_local_building_ratio_100m": float(r100),
        "sectors": secs,
        "reasoning": "synthetic record for self-test",
    }
    fc = {k: 1.0 for k in list(EM_NUM_KEYS) + ["scene_type", "dominant_material"]}
    fc["overall"] = 1.0
    fc["llm_reported_confidence_mode"] = "medium"
    fc["llm_reported_confidence_unique"] = 1.0
    fc["reasoning_pick_distance"] = 0.1
    fc["n_samples"] = 3
    return {
        "schema_version": "v2+geomfacts",
        "sample_id": f"{map_id}_{tx_id}",
        "map_id": int(map_id),
        "tx_id": int(tx_id),
        "prior": prior,
        "provenance": {
            "design": "v3_geometry",
            "field_sources": {"em": "llm", "sectors": "program",
                              "local_ratios": "program"},
            "model": "synthetic",
            "base_url": "about:blank",
            "prompt_version": "prompt_v3",
            "temperature": 0.7,
            "samples_per_call": 3,
            "n_parsed": 3,
            "n_errors": 0,
            "errors": [],
            "field_consistency": fc,
            "geometry_facts": {"tx_on_building": bool(on_building),
                               "facts_text": _facts_text(r50, r100, on_building, secs)},
            "v3_include_image": True,
            "tx_xy": [100.0, 100.0],
            "freq_ghz": freq_ghz,
            "px_per_m": 1.0,
            "prompt_sha1": "deadbeefdeadbeef",
            "generated_at": "2026-01-01T00:00:00+00:00",
        },
    }


def write_synth_dir(tmp: Path, n_maps: int = 5, per_map: int = 24,
                    sector_perm: list[int] | None = None,
                    level_step: float = 30.0) -> Path:
    """写出一批合成先验：几何**在地图内**与**地图之间**都变化。

    为什么两个方向都要有变化（这是踩过的坑）：
      * `min_dist` 若在地图内恒定，逐图 ρ 就是 NaN/退化，`within_map` 诊断失效；
      * `min_dist` 若没有地图级水平差异，就构造不出"总体 ρ 显著但逐图变号"
        的地图间混淆场景。
    所以 base_dist = 5 + level_step*m + 4*t：`m` 给地图级水平，`t` 给地图内变化。
    """
    d = tmp
    d.mkdir(parents=True, exist_ok=True)
    for m in range(n_maps):
        for t in range(per_map):
            # r50/r100 用正弦：既不与 tx 单调（避免与 min_dist 完全共线，
            # 否则"最强驱动量"会随机落到它们头上），也不恒定
            r50 = 0.30 + 0.15 * np.sin(0.7 * t)
            r100 = 0.25 + 0.12 * np.cos(0.5 * t)
            base_dist = 5.0 + level_step * m + 4.0 * t
            rec = synth_record(m, t, sector_perm=sector_perm, r50=float(r50),
                               r100=float(r100), base_dist=base_dist)
            (d / f"{m}_{t}.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return d


def field_by_name(t: dict, name: str) -> dict:
    for f in t["fields"]:
        if f["field"] == name:
            return f
    raise AssertionError(f"类型学表里没有字段 {name}")


# ===== 各项自检 =====

def t_parse_facts_text() -> None:
    rec = synth_record(0, 0)
    f = parse_facts_text(rec["provenance"]["geometry_facts"]["facts_text"])
    assert approx(f["r50"], 0.30, 1e-3), f"r50 解析错：{f['r50']}"
    assert approx(f["r100"], 0.25, 1e-3), f"r100 解析错：{f['r100']}"
    assert f["tx_on_building"] is True, "Tx 落在建筑内没被解析出来"
    assert len(f["obstruction"]) == 8, f"方位数错：{len(f['obstruction'])}"
    assert f["obstruction"][90] == "blocked", f"90° 的类别解析错：{f['obstruction']}"
    # 反例：文本被截断时必须抛错，不许静默给默认值
    broken = rec["provenance"]["geometry_facts"]["facts_text"].replace(
        "The Tx pixel itself lies ON a building footprint.", "")
    try:
        parse_facts_text(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("facts_text 残缺时没有抛错（会静默产出错的驱动量）")


def t_tx_on_open_ground() -> None:
    rec = synth_record(0, 0, on_building=False)
    f = parse_facts_text(rec["provenance"]["geometry_facts"]["facts_text"])
    assert f["tx_on_building"] is False, "Tx 在空地时被解析成在建筑内"


def t_geometry_drivers_alignment() -> None:
    """**核心防错**：sectors 数组被打乱后，驱动量必须与"按方位角对齐"一致。"""
    rec_ok = synth_record(0, 0)
    perm = [3, 7, 1, 5, 0, 6, 2, 4]        # 打乱数组顺序（方位角仍是同一集合）
    rec_sh = synth_record(0, 0, sector_perm=perm)
    assert [s["azimuth_deg"] for s in rec_sh["prior"]["sectors"]] != list(AZIMUTHS), \
        "构造失败：数组顺序没有被真的打乱"
    d_ok = geometry_drivers(rec_ok)
    d_sh = geometry_drivers(rec_sh)
    for key in ("min_dist", "mean_dist", "n_blocked"):
        assert approx(d_ok[key], d_sh[key]), (
            f"打乱 sectors 数组后 {key} 变了（{d_ok[key]} → {d_sh[key]}）："
            f"说明代码按数组下标而不是按方位角取值")
    # tx 0 的 0° 方位是 blocked，增量 0 → min_dist 应等于 base_dist
    assert approx(d_ok["min_dist"], 5.0), f"min_dist 算错：{d_ok['min_dist']}"
    assert approx(d_ok["n_blocked"], 3.0), f"n_blocked 算错：{d_ok['n_blocked']}"


def t_geometry_drivers_detects_real_change() -> None:
    """反向对照：如果**真的**把方位与距离配错，驱动量必须变（否则上一个自检没有功效）。"""
    rec = synth_record(0, 0)
    secs = [dict(s) for s in rec["prior"]["sectors"]]
    for i, s in enumerate(secs):                # 距离与类别整体右移一位，方位不动
        src = secs[(i + 1) % len(secs)]
        s["nearest_building_m"] = src["nearest_building_m"]
        s["obstruction"] = src["obstruction"]
    rec["prior"]["sectors"] = secs
    d = geometry_drivers(rec)
    assert not approx(d["min_dist"], 5.0), \
        "方位↔距离被真的错配后 min_dist 没变：说明该自检没有区分度"


def t_empty_and_missing_dir() -> None:
    with tempfile.TemporaryDirectory() as td:
        empty = Path(td) / "empty"
        empty.mkdir()
        try:
            load_priors(empty)
        except SystemExit:
            pass
        else:
            raise AssertionError("空目录没有抛错")


def t_load_priors_and_field_count() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = write_synth_dir(Path(td) / "priors", n_maps=2, per_map=5)
        rows = load_priors(d)
        assert len(rows) == 10, f"载入样本数错：{len(rows)}"
        need = {"consistency", "overall", "r50", "r100", "n_blocked", "min_dist",
                "mean_dist", "_lab_0", "_dist_7", "_tx_on_building"}
        missing = need - set(rows[0].keys())
        assert not missing, f"载入结果缺字段：{missing}"


def t_classifier_branches() -> None:
    """六个分类分支必须各自被正确触发，且方向、驱动量、同号数都对。"""
    with tempfile.TemporaryDirectory() as td:
        d = write_synth_dir(Path(td) / "priors", n_maps=5, per_map=24)
        t = build_typology(load_priors(d))

        # (1) 接地且**不能被几何复现** → 真实且有残余信息。
        # 注意：合成数据里 los_ratio 恰好是 min_dist 的单调函数，仪器 3 的 R² 会打到 1.0，
        # 于是分类器会正当地判成"真实但冗余"。这里要测的是"残余信息"这一支，
        # 所以显式把 R² 压到阈值以下。逐字段的 6 个分支由
        # `t_classifier_unit_branches` 用直接构造的输入逐个覆盖（不依赖合成数据）。
        from scripts.probe_l1_typology import classify
        f = field_by_name(t, "los_ratio")
        assert f["grounding"]["best_driver"] == "min_dist", \
            f"最强驱动量错：{f['grounding']['best_driver']}"
        assert approx(f["grounding"]["rho_absmax_signed"], 1.0, 1e-9), \
            f"ρ 应为 +1：{f['grounding']['rho_absmax_signed']}"
        assert f["grounding"]["any_supported"], "接地成立却没有被判定为 supported"
        assert f["within_map"]["sign_agreement_with_expected"] == "5/5", \
            f"逐图同号数错：{f['within_map']['sign_agreement_with_expected']}"
        d_low = dict(f["reproducibility"], r2_best=0.40)      # 仪器 3 解释不了
        got = classify("los_ratio", f["repeatability"], f["grounding"], d_low,
                       f["variance"], f["within_map"])
        assert got["type"] == "真实且有残余信息", f"残余信息分支错：{got['type']}"
        assert got["in_condition_vector"] is True
        assert got["signal"] == "residual"
        # 同一输入但 R² 高 → 必须换成"真实但冗余"
        d_high = dict(f["reproducibility"], r2_best=0.90)
        got2 = classify("los_ratio", f["repeatability"], f["grounding"], d_high,
                        f["variance"], f["within_map"])
        assert got2["type"] == "真实但冗余", f"冗余分支错：{got2['type']}"
        assert got2["in_condition_vector"] is False

        # (2) 常数：diffraction_ratio 取值完全不变
        f = field_by_name(t, "diffraction_ratio")
        assert f["type"] == "常数（无区分度）", f"diffraction 类型错：{f['type']}"
        assert f["in_condition_vector"] is False

        # (3) 幻觉 / 无信号：reflection_ratio 有方差但整体 ρ 不显著（与几何无关）
        f = field_by_name(t, "reflection_ratio")
        assert f["type"] in ("幻觉（自信地错）", "弱/无信号"), \
            f"reflection 类型错：{f['type']}"
        assert f["in_condition_vector"] is False
        assert not f["grounding"]["any_supported"], "该字段不该被判为接地成立"

        # (4) 类别字段且只有一个取值 → 常数（不是"弱/无信号"）
        f = field_by_name(t, "scene_type")
        assert f["type"] == "常数（无区分度）", f"scene_type 类型错：{f['type']}"
        assert approx(f["variance"]["top1_share"], 1.0), "scene_type 应当只有一个取值"

        # 类型计数必须覆盖到这些字段
        assert sum(t["type_counts"].values()) == len(t["fields"]), "类型计数与字段数不符"


def t_classifier_unit_branches() -> None:
    """六个判读分支：用**直接构造的输入**逐个覆盖，不依赖合成数据的巧合。

    这是必要的一层：合成数据里"某个字段恰好落在哪一支"会随着系数变动而漂移，
    用它来测分支会得到脆弱的测试。所以这里直接把分类器的输入摆好。
    """
    from scripts.probe_l1_typology import classify

    cons_hi = {"median": 1.0, "mean": 1.0, "min": 1.0, "std": 0.0,
               "share_all_rounds_agree": 1.0}
    cons_lo = dict(cons_hi, median=0.67, mean=0.70, min=0.67)

    def ground(rho, p, drv="min_dist", sign="+", supported=None, dir_ok=None):
        dir_ok = (rho > 0) if dir_ok is None else dir_ok
        return {"best_driver": drv, "best_expected_sign": sign,
                "rho_absmax_signed": rho, "rho_absmax_shuffled": 0.0,
                "p_permutation": p, "tested": 1, "n_testable": 1,
                "any_supported": (dir_ok and p < 0.05) if supported is None else supported,
                "direction_correct_any": dir_ok, "n_pairs_supported": 0}

    num_var = {"std_over_range": 0.10}
    # 散度既不为 0 也不有限（例如类别字段拿不到任何散度信息）→ 只有这一支能落到"弱/无信号"。
    # 注意：数值字段只要散度 > CV_EPS 就一定进"幻觉"那一支，所以这里必须用退化输入。
    num_tiny = {"std_over_range": float("nan")}
    const_var = {"std_over_range": 0.01}
    cat_const = {"top1_share": 1.0}
    within_ok = {"sign_agreement_with_expected": "5/5"}
    within_bad = {"sign_agreement_with_expected": "1/5"}
    within_amb = {"sign_agreement_with_expected": "3/5"}
    dist_low = {"r2_best": 0.40, "std": 0.1, "r2_shuffled": 0.0}
    dist_high = {"r2_best": 0.90, "std": 0.1, "r2_shuffled": 0.0}

    cases = [
        # (名称, 期望类型, 期望 signal, 期望进向量, 参数)
        ("常数（数值）", "常数（无区分度）", "none", False,
         (cons_hi, ground(0.1, 0.9), dist_low, const_var, within_ok)),
        ("常数（类别）", "常数（无区分度）", "none", False,
         (cons_hi, ground(0.1, 0.9), dist_low, cat_const, within_ok)),
        ("噪声", "噪声", "none", False,
         (cons_lo, ground(0.1, 0.9), dist_low, num_var, within_ok)),
        ("残余信息", "真实且有残余信息", "residual", True,
         (cons_hi, ground(0.6, 0.001), dist_low, num_var, within_ok)),
        ("真实但冗余", "真实但冗余", "redundant", False,
         (cons_hi, ground(0.6, 0.001), dist_high, num_var, within_ok)),
        ("地图间混淆", "接地不稳（地图间混淆）", "confounded", False,
         (cons_hi, ground(0.6, 0.001), dist_low, num_var, within_bad)),
        ("自信地错", "自信地错（系统性幻觉）", "harmful", False,
         (cons_hi, ground(-0.4, 0.001, dir_ok=False), dist_low, num_var, within_ok)),
        ("幻觉", "幻觉（自信地错）", "harmful", False,
         (cons_hi, ground(0.05, 0.60, supported=False, dir_ok=True), dist_low,
          num_var, within_ok)),
        ("弱/无信号（散度未知）", "弱/无信号", "none", False,
         (cons_hi, ground(0.05, 0.60, supported=False, dir_ok=True), dist_low,
          num_tiny, within_ok)),
        ("幻觉（类别·多取值）", "幻觉（自信地错）", "harmful", False,
         (cons_hi, ground(0.05, 0.60, supported=False, dir_ok=True), dist_low,
          {"top1_share": 0.90, "unique": 2}, within_ok)),
    ]
    for name, want_type, want_signal, want_in, (c, g, d, v, w) in cases:
        got = classify("f", c, g, d, v, w, n=500)
        assert got["type"] == want_type, f"{name}: 类型错 {got['type']}"
        assert got["signal"] == want_signal, f"{name}: signal 错 {got['signal']}"
        assert got["in_condition_vector"] is want_in, f"{name}: 进向量标记错"
        assert got["verdict"], f"{name}: 没有判读文字"

    # 模糊的逐图同号（3/5）**不**触发混淆判定：只有明确占多数的反向才触发。
    # 这是刻意的保守设定：宁可不标 confounded，也不要把真实字段误标掉。
    got = classify("f", cons_hi, ground(0.6, 0.001), dist_low, num_var, within_amb, n=500)
    assert got["type"] == "真实且有残余信息", f"3/5 同号不该判混淆：{got['type']}"

    # 小样本标注必须挂在每一支上
    got = classify("f", cons_hi, ground(0.6, 0.001), dist_low, num_var, within_ok, n=30)
    assert got["low_n_warning"] and got["n"] == 30, "小样本标注没生效"
    got = classify("f", cons_hi, ground(0.6, 0.001), dist_low, num_var, within_ok, n=500)
    assert "low_n_warning" not in got, "样本充足时不该有小样本标注"


def t_confounded_branch() -> None:
    """接地不稳（地图间混淆）分支：总体 ρ 显著，但地图内部反复变号。

    构造办法：让字段的水平**跟着地图走**（所以与地图级几何差异强相关 → 总体 ρ 显著），
    而地图内部只给一个与几何无关、幅度很小的抖动（所以逐图 ρ 是随机符号）。
    这正是"上一版地图级常量"的形态：它是地图的属性，不是这个 Tx 的属性。
    """
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "priors"
        write_synth_dir(d, n_maps=5, per_map=24)
        # 重写 reflection_ratio：
        #   跨地图 = 地图级水平、**方向与预期一致**（地图越大 → 越小，因为 min_dist 越大）
        #            → 总体 ρ 显著且方向正确，所以不会被"自信地错"那一支截走；
        #   地图内 = 与 min_dist 同向 / 反向**交替**（c 按地图取 ±）→ 逐图 ρ 变号。
        # 这正是"地图级常量"的形态：字段在描述地图，而不是这个 Tx。
        for m in range(5):
            c = -0.0025 if m < 2 else +0.0025      # 2/5 张图与预期同号 → 明确变号
            for t in range(24):
                f = d / f"{m}_{t}.json"
                rec = json.loads(f.read_text(encoding="utf-8"))
                min_dist = min(s["nearest_building_m"] for s in rec["prior"]["sectors"])
                level = 0.90 - 0.16 * m            # 地图级水平：与 min_dist 反向
                # 不加噪声：地图内必须是**干净的**单调关系，否则 ρ 的符号会被抖动翻来翻去，
                # 测到的就不是"地图间混淆"而是"随机噪声"（我自己就踩过这个坑）。
                val = level + c * (min_dist - (5.0 + 30.0 * m))
                rec["prior"]["reflection_ratio"] = float(np.clip(val, 0.0, 1.0))
                f.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")

        t = build_typology(load_priors(d))
        f = field_by_name(t, "reflection_ratio")
        agree = f["within_map"]["sign_agreement_with_expected"]
        num, den = (int(x) for x in agree.split("/"))
        # 注意：`any_supported` 要求"方向与物理预期一致"（该字段预期为 -）。
        # 这里造的是"地图间混淆"，总体 ρ 可以显著但方向与预期**相反**，
        # 所以判据不能挂在 any_supported 上，而要挂在"逐图大面积变号"上。
        assert f["grounding"]["p_permutation"] < 0.05, (
            f"构造失败：总体 ρ 应当显著才测得到混淆分支"
            f"（ρ={f['grounding']['rho_absmax_signed']}, "
            f"p={f['grounding']['p_permutation']}）")
        assert num / den < 0.6, (
            f"构造失败：逐图 ρ 应大面积变号才测得到混淆分支，实际同号 {agree}"
            f"（逐图 ρ={ {m: round(e['rho'], 3) for m, e in f['within_map']['per_map'].items()} }）")
        assert f["type"] == "接地不稳（地图间混淆）", (
            f"类型错：{f['type']}（ρ={f['grounding']['rho_absmax_signed']}, "
            f"p={f['grounding']['p_permutation']}, 同号 {agree}）")
        assert f["signal"] == "confounded"
        assert f["in_condition_vector"] is False, "被地图间差异驱动的字段不该进条件向量"
        assert f["within_map"]["std_over_range_min"] > 0.0, "逐图散度不该是 0"


def t_oracle_redundancy() -> None:
    """仪器 3b：单调可复现的字段必须被 oracle 抓成"单几何量的重新编码"。"""
    with tempfile.TemporaryDirectory() as td:
        d = write_synth_dir(Path(td) / "priors", n_maps=5, per_map=24)
        t = build_typology(load_priors(d))
        f = field_by_name(t, "los_ratio")
        o = f["oracle"]
        assert o["best_driver"] == "min_dist", f"oracle 选错驱动量：{o['best_driver']}"
        assert o["best_in_sample_r2"] >= 0.98, \
            f"完美单调关系的样本内 R² 应≈1，实际 {o['best_in_sample_r2']}"
        assert o["encoded_by_single_driver"] is True, "没被判定为单几何量重编码"
        assert f["redundant_confirmed_by_oracle"] is True
        assert t["fields_redundant_confirmed_by_oracle"] == ["los_ratio"] or \
            "los_ratio" in t["fields_redundant_confirmed_by_oracle"]
        # 反例：纯常数字段不该被说成"某个几何量的重新编码"
        fd = field_by_name(t, "diffraction_ratio")
        assert fd["oracle"]["best_in_sample_r2"] < 0.85, \
            "常数字段被误判为单几何量重编码"


def t_instrument3_shuffle_control() -> None:
    """仪器 3 的错位对照必须正常（R² ≈0 或负），否则流程有泄漏。"""
    with tempfile.TemporaryDirectory() as td:
        d = write_synth_dir(Path(td) / "priors", n_maps=5, per_map=24)
        t = build_typology(load_priors(d))
        for name in ("los_ratio", "reflection_ratio", "pathloss_max_db"):
            f = field_by_name(t, name)
            r2_sh = f["reproducibility"]["r2_shuffled"]
            assert r2_sh <= 0.05, f"{name} 的错位对照 R²={r2_sh} 异常（可能泄漏）"
            assert f["reproducibility"]["shuffle_ok"] is True


def t_grounding_pairs_are_prespecified() -> None:
    """接地检验必须只用**预先指定**的配对，逐字段不漏、也不多测。"""
    with tempfile.TemporaryDirectory() as td:
        d = write_synth_dir(Path(td) / "priors", n_maps=5, per_map=24)
        t = build_typology(load_priors(d))
        assert len(t["grounding_pairs"]) == 10, \
            f"预指定配对数错：{len(t['grounding_pairs'])}"
        for p in t["grounding_pairs"]:
            assert p["field"] in EM_NUM_KEYS, f"配对里出现了非电磁字段：{p['field']}"
            assert p["expected_sign"] in ("+", "-")
            # 退化输入（字段或驱动量恒定）必须被显式标注为"不可检验"，
            # 而不是留一个 NaN 让下游把"没测"读成"测出 0"
            if p["testable"]:
                assert 0.0 <= p["p_permutation"] <= 1.0, \
                    f"可检验的配对 p 值越界：{p['p_permutation']}"
                assert np.isfinite(p["rho"]), "可检验的配对 ρ 是 NaN"
            else:
                assert p["not_testable_reason"], "不可检验的配对没有给出原因"
                assert not np.isfinite(p["p_permutation"]), "不可检验却给了 p 值"
        assert t["grounding_summary"]["tested_pairs"] == 10
        assert t["grounding_summary"]["testable_pairs"] >= 1, "一个可检验的配对都没有"


def t_low_n_warning() -> None:
    """小样本时必须显式标注"结论不可作定论"，并在每一行字段上留下痕迹。"""
    with tempfile.TemporaryDirectory() as td:
        d = write_synth_dir(Path(td) / "priors", n_maps=2, per_map=12)  # 24 条 < LOW_N
        t = build_typology(load_priors(d))
        assert t["n_samples"] == 24
        assert t["low_n_warning"], "24 条样本却没有小样本警告"
        assert f"{LOW_N}" in t["low_n_warning"]
        for f in t["fields"]:
            assert f.get("low_n_warning"), f"{f['field']} 缺小样本标注"
            assert f["n"] == 24
        # 反向对照：样本足够时不该有警告
        d2 = write_synth_dir(Path(td) / "priors2", n_maps=5, per_map=24)  # 120 条
        t2 = build_typology(load_priors(d2))
        assert t2["low_n_warning"] is None, "样本充足时不该有小样本警告"
        assert "low_n_warning" not in t2["fields"][0], "样本充足时字段不该带小样本标注"


def t_map_filter() -> None:
    """--limit-maps 过滤必须真的生效（对拍历史结果时全靠它）。"""
    with tempfile.TemporaryDirectory() as td:
        d = write_synth_dir(Path(td) / "priors", n_maps=4, per_map=6)
        rows = load_priors(d, maps={0, 1})
        assert len(rows) == 12, f"过滤后样本数错：{len(rows)}"
        assert {r["map_id"] for r in rows} == {0, 1}


def t_validate_against_matches_and_fails() -> None:
    """对拍机制：一致时退出码 0；人为改坏参考结果时必须退出码 3（否则对拍毫无意义）。"""
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        d = write_synth_dir(td_p / "priors", n_maps=5, per_map=24)
        rows = load_priors(d)
        n = len(rows)
        ref = {"n": n, "direction_correct": 4,
               "significant_vs_shuffled": 2,
               "spearman_se": 1.0 / np.sqrt(n - 1)}
        # 参考文件里的计数必须与真实算出来的一致，才谈得上"对拍"
        from scripts.probe_l1_typology import grounding_stats
        real = grounding_stats(rows)
        ref["direction_correct"] = real["direction_correct"]
        ref_path = td_p / "ref_ok.json"
        ref_path.write_text(json.dumps(ref), encoding="utf-8")

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = typology_main(["--priors", str(d), "--validate-against", str(ref_path),
                                "--no-write"])
        assert rc == 0, f"一致时退出码应为 0，实际 {rc}"
        assert "对拍通过" in buf.getvalue(), "没有打印对拍结果"

        ref_bad = dict(ref)
        ref_bad["direction_correct"] = real["direction_correct"] + 1
        bad_path = td_p / "ref_bad.json"
        bad_path.write_text(json.dumps(ref_bad), encoding="utf-8")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = typology_main(["--priors", str(d), "--validate-against", str(bad_path),
                                "--no-write"])
        assert rc == 3, f"不一致时退出码应为 3，实际 {rc}"
        assert "不一致" in buf.getvalue(), "不一致时没有给出警告"


def t_report_render_smoke() -> None:
    """报告渲染不得抛错，且必须包含三仪器表与结论段（渲染崩溃=没有交付物）。"""
    with tempfile.TemporaryDirectory() as td:
        d = write_synth_dir(Path(td) / "priors", n_maps=5, per_map=24)
        out_md = Path(td) / "r.md"
        out_js = Path(td) / "r.json"
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = typology_main(["--priors", str(d), "--out", str(out_md),
                                "--json", str(out_js)])
        assert rc == 0, f"退出码 {rc}"
        md = out_md.read_text(encoding="utf-8")
        for token in ("## 1. 三仪器表", "## 4. 结论", "仪器 3b", "## 5. 本次分析的已知限制"):
            assert token in md, f"报告里缺 {token}"
        payload = json.loads(out_js.read_text(encoding="utf-8"))
        assert payload["n_samples"] == 120, "JSON 里样本数错"
        assert len(payload["fields"]) == len(EM_NUM_KEYS) + 2


def t_scene_gate_three_branches() -> None:
    """`scene_type_follows_geometry` 的三分支：只有"分档落在真实断层上"才该 FAIL。

    为什么必须测这一支：这条判据修过两次，两次都是把"数据本身的性质"当成"LLM 偷懒"。
    实测 prompt_v3 全量 4000 条 / 50 张地图的整图建筑占比是 0.242→0.435 的**平滑**
    分布（最大相邻间隔仅为典型间隔的 ~5.5 倍），0.35 这个阈值劈出来的 26/24
    "两类"是阈值产物，LLM 恒定答 urban 是对的。判据不能凭空要求 LLM 跨类。
    """
    from src.prior.audit import scene_follows_geometry, scene_split_has_natural_gap

    # (1) 平滑分布（复刻实测形状）→ 无天然空隙 → LLM 恒定合法
    smooth = [0.242 + (0.435 - 0.242) * i / 49 for i in range(50)]
    has, mx, med, ratio = scene_split_has_natural_gap(smooth)
    assert not has, f"平滑分布被误判为有天然空隙（ratio={ratio:.2f}）"
    assert mx <= 0.01, f"最大间隔算错：{mx}"
    ok, why = scene_follows_geometry(["urban", "dense_urban"], ["urban"], has, ratio)
    assert ok is True, f"平滑分布 + LLM 恒定时应通过，理由：{why}"
    assert "阈值劈出来" in why or "平滑" in why

    # (2) 真双峰（有断层）→ 有天然空隙 → LLM 恒定必须 FAIL
    bimodal = [0.10, 0.11, 0.12, 0.105, 0.115, 0.40, 0.41, 0.42, 0.405, 0.415]
    has2, mx2, med2, ratio2 = scene_split_has_natural_gap(bimodal)
    assert has2, f"真双峰被误判为平滑（ratio={ratio2:.2f}）"
    ok2, why2 = scene_follows_geometry(["urban", "dense_urban"], ["urban"], has2, ratio2)
    assert ok2 is False, f"真双峰 + LLM 恒定时应 FAIL，理由：{why2}"
    # 同一个断层上，只要 LLM 给出两类就该通过
    ok2b, _ = scene_follows_geometry(["urban", "dense_urban"], ["urban", "dense_urban"],
                                     has2, ratio2)
    assert ok2b is True, "真双峰且 LLM 也跨类时应当通过"

    # (3) 程序侧本身只有一类 → LLM 恒定合法
    ok3, why3 = scene_follows_geometry(["urban"], ["urban"], False, float("nan"))
    assert ok3 is True, f"程序侧单类时应通过，理由：{why3}"

    # (4) 样本太少时不硬判（分档统计量无意义）
    has4, *_ = scene_split_has_natural_gap([0.1, 0.2])
    assert has4 is False, "样本过少时不该声称存在天然空隙"


def t_prior_actually_conditions_model() -> None:
    """真实数据 + 真实 config：**换先验必须改变模型输出**。

    这是本会话最贵的一个 bug 的直接防线。`apply_film_identity_init()` 初版把最后一层
    的权重与 bias 一起置零，于是 film_mlp 输出恒为 [1,0,…]：
      * 先验完全不影响前向；
      * 最后一层权重的梯度 ∝ 该层输入（恒为 0）⇒ 梯度恒为 0、权重永久冻结。
    后果是 `prior_real_v3` 与 `prior_v3_masked_em` **数学恒等**（实测 5 epoch 后
    84 个张量 max|Δ|=0，连 train loss 都逐位相同），整轮实验无效，且不报任何错。

    所以必须有一条端到端断言：喂两组**真实**先验向量，输出不能逐位相同。
    有数据盘时才跑（没有则跳过并说明）。
    """
    data_root = ROOT / "data" / "radiomapseer"
    if not (data_root / "buildings_complete").exists():
        print("      (跳过：本机没有 data/radiomapseer，此检查在远端 GPU 机器上跑)")
        return
    prior_dir = ROOT / "priors" / "prompt_v3" / "glm-4v-flash"
    if len(list(prior_dir.glob("*.json"))) < 100:
        print("      (跳过：本机先验不全（数据集加载要求 train split 全部就位），"
              "此检查在远端 GPU 机器上跑；合成版见 tests/test_local.py:"
              "models.film_prior_actually_conditions)")
        return
    import torch

    from src.data.dataset import RadioMapSeerDataset
    from src.models.factory import build_model
    from src.utils.seed import set_seed

    set_seed(0, deterministic=True)
    ds = RadioMapSeerDataset(
        split="train", data_root=str(data_root),
        splits_path=str(ROOT / "data" / "splits.json"), prior_mode="real",
        prior_dir=str(ROOT / "priors" / "prompt_v3" / "glm-4v-flash"),
        prior_groups=["em_num", "local_ratio", "global_cat", "sectors"],
        freq_ghz=0.9, px_per_m=1.0, tx_encoding="constant")
    ds.samples = ds.samples[:2]
    if ds.prior_vectors is not None:
        ds.prior_vectors = ds.prior_vectors[:2]

    item = ds[0]
    x = item["inputs"].unsqueeze(0)
    pv = item["prior_vec"].unsqueeze(0)

    cfg = {"name": "unet_film", "base_channels": 8, "dropout": 0.0,
           "film": {"hidden": [64, 128], "identity_init": True},
           "cot": {"enabled": False}}
    m, _ = build_model(cfg, in_channels=int(x.shape[1]), prior_dim=int(pv.shape[1]))
    m.eval()
    with torch.no_grad():
        y_real = m(x, pv)
        y_zero = m(x, torch.zeros_like(pv))
    d = float((y_real - y_zero).abs().max())
    assert d > 0.0, (
        "真实先验与全零先验给出**逐位相同**的输出 → 先验没有接到前向上"
        "（identity 初始化把最后一层权重置零的经典陷阱）")

    # 梯度也必须真的流到 FiLM 的每一层
    m.train()
    m(x, pv).sum().backward()
    for name, p in m.film_mlp.named_parameters():
        assert p.grad is not None and float(p.grad.abs().max()) > 0.0, (
            f"{name} 的梯度恒为 0 —— 该层永远不会被训练，先验分支等于没有")


def main(argv=None) -> int:
    only = argv[0] if argv else None
    print("=" * 78)
    print("test_typology：L1 三仪器类型学的自检（不需要 GPU / API key / 数据盘）")
    print("=" * 78)
    tests = [
        ("parse.facts_text", t_parse_facts_text),
        ("parse.tx_on_open_ground", t_tx_on_open_ground),
        ("geometry.drivers_alignment", t_geometry_drivers_alignment),
        ("geometry.drivers_detect_real_change", t_geometry_drivers_detects_real_change),
        ("load.empty_dir_raises", t_empty_and_missing_dir),
        ("load.rows_and_fields", t_load_priors_and_field_count),
        ("classify.all_branches", t_classifier_branches),
        ("classify.unit_branches", t_classifier_unit_branches),
        ("classify.confounded_branch", t_confounded_branch),
        ("scene_gate.three_branches", t_scene_gate_three_branches),
        ("prior.actually_conditions_model", t_prior_actually_conditions_model),
        ("oracle.redundancy", t_oracle_redundancy),
        ("instrument3.shuffle_control", t_instrument3_shuffle_control),
        ("instrument2.prespecified_pairs", t_grounding_pairs_are_prespecified),
        ("samplesize.low_n_warning", t_low_n_warning),
        ("load.map_filter", t_map_filter),
        ("validate.matches_and_fails", t_validate_against_matches_and_fails),
        ("report.render_smoke", t_report_render_smoke),
    ]
    for name, fn in tests:
        if only and only not in name:
            continue
        check(name, fn)
    print("-" * 78)
    if FAIL:
        print(f"通过 {len(PASS)} / {len(PASS) + len(FAIL)}")
        for name, exc in FAIL:
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
        print("存在失败项 [FAIL]")
        return 1
    print(f"通过 {len(PASS)} / {len(PASS)}")
    print("全部通过 [OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
