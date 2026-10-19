#!/usr/bin/env python3
"""L1 类型学：把三个正交仪器合成一张表，判定每个 LLM 电磁字段属于哪一类。

为什么要有这个脚本（交接文档 `NEXT_TASK_LLM_prior_information.md` §1.3/§1.4）
--------------------------------------------------------------------------------
"LLM 先验有没有信息"这个问题**不能靠单个指标回答**：低 R²（几何解释不了）同时与
"有独立信息"和"纯噪声"相容；一致率高同时与"学到真规律"和"输出模板"相容。
所以必须三角定位。本脚本把三个**互相不蕴含**的测量放在一起：

  仪器 1 可重复性（repeatability）—— 同一样本多轮采样的字段一致率。
        来源：`prior.provenance.field_consistency`（生成时已记录，**零 API 成本**）。
        口径与 `src/prior/generate.py:aggregate_em()` 完全一致：数值字段落在样本均值
        ±10% 量程内算一致；类别字段与众数相同算一致。
  仪器 2 接地（grounding）—— 字段与**给定几何事实**的 Spearman ρ，配对预先指定，
        并与"错位对照"（打乱的几何）比较，另加置换检验给 p 值。几何事实直接解析
        `provenance.geometry_facts.facts_text`，所以**不需要数据盘**
        （这批先验的几何本来就是程序算出来、写进 facts_text 再喂给 LLM 的）。
  仪器 3 可复现性（reproducibility）—— 用几何特征做 5 折 CV 拟合该字段的 R²
        （三档特征集 × Ridge/GBM + 错位对照）。

判读规则（§1.4 的表，逐行实现在 `classify()`）：

  | 可重复性 | 接地 ρ | 可复现 R² | 类型 | 对条件向量的意义 |
  |---|---|---|---|---|
  | 高 | 高 | 高 | 真实但冗余 | LLM 在做特征工程，直接喂输入更好 |
  | 高 | 高 | 低 | 真实且有残余信息 | ✅ 唯一支持继续的类型 |
  | 高 | ≈0 | 低 | 自信地错（系统性幻觉） | 有害 |
  | 低 | ≈0 | — | 噪声 | 有害 |
  | 高 | ≈0 | 低、方差大 | 幻觉 | 比恒定更糟：下游得额外学会忽略它 |
  | 高 | — | — | 方差≈0 | 合法常数（材料属性） | 无区分度 |

另外补两个**判定性**的诊断（都不依赖人工阈值）：

  * **地图层内接地**：把样本按地图分层，在地图内部重算 ρ。若 ρ 在地图内塌到 0，
    则总体 ρ 全部来自"地图之间几何分布不同"，即该字段只是**地图级常量**，
    对同图不同 Tx 没有分辨力 —— 这正是上一版先验的失败模式（同图 80 个 Tx
    拿到几乎同一个向量），必须单独查。
  * **散度与一致率容差**：一致率是"落在 ±10% 量程内"，对 `pathloss_max_db`
    这种量程 190 dB 的字段，容差高达 ±19 dB —— 一致率 1.0 **不代表**确定性。
    所以额外报"散度/量程"（std ÷ 量程），并把每个数值字段的容差写进表里。

产出
----
    reports/llm_l1_typology.md    人读（三仪器表 + 逐字段判读 + 对条件向量的结论）
    reports/llm_l1_typology.json  机读（供跨模型迁移那一步做逐字段对比）

用法
----
    python scripts/probe_l1_typology.py --priors priors/prompt_v3/glm-4v-flash
    # 校验模式：与 probe_llm_grounding.py 的 240 样本结果对拍
    python scripts/probe_l1_typology.py --priors priors/prompt_v3/glm-4v-flash \
        --limit-maps 0:4 --validate-against reports/llm_grounding_240.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # Windows 控制台默认 GBK，中文表格会乱码
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from src.prior.generate import derive_confidence  # noqa: E402
from src.prior.schema import EM_NUM_KEYS, NUM_RANGES, vector_dim  # noqa: E402

# 复用接地检验里**已经跑过、已经对拍过**的统计量与配对表，避免两处实现漂移
from scripts.probe_llm_grounding import MOTIVATED_PAIRS, spearman  # noqa: E402

# ===== 常量：口径写死在这里，报告里必须原样引用 =====

#: 仪器 1/2/3 都只用 LLM 产出的字段。sectors 与 tx_local_building_ratio_*
#: 在 v3 里是**程序**算的（provenance.field_sources 标记为 "program"），
#: 它们的一致性是结构性的 1.0，混进类型学会把表稀释掉，所以单列（见 JSON
#: 的 program_computed_fields）。
LLM_CAT_KEYS: tuple[str, ...] = ("scene_type", "dominant_material")

#: 判定"实质为常数"的阈值（归一化散度 std ÷ 量程）。取 5% 量程：比 0 宽容
#: （允许浮点末位抖动），又远低于任何有物理意义的变化幅度。改它必须同步改报告文字。
CV_EPS = 0.05
#: "一致率高/低"的分界：与 generate.py:derive_confidence() 的 medium 门槛一致。
CONSISTENCY_HI = 0.70
#: 类别字段"实质恒定"的判据：众数占比 ≥ 此值（等价于归一化散度 1−share ≤ 0.05）。
#: 数值字段用 std ÷ 量程 ≤ CV_EPS。
CAT_TOP1_CONST = 0.95
#: 仪器 2 的"强接地"分界（|ρ| 分档）。**只用于把结论写成强/弱**，
#: 是否"成立"一律由置换检验的 p 值决定，不用裸阈值。
RHO_STRONG = 0.50
#: 仪器 3 的分界：沿用 probe_llm_distillability.py 的三档口径。
R2_HI = 0.50
R2_ENC = 0.80
#: 仪器 3b（单驱动量 oracle）的"就是那个几何量的重新编码"判据：**样本内** R² ≥ 此值。
ORACLE_R2 = 0.85
#: 样本量下限：低于它，**方差估计本身**就不可靠（std 的相对标准误 ≈ 1/√(2n)）。
#: 实测：24 条子集上 `los_ratio` 的散度是 0.038，落在 0.05 之下而被判"常数"；
#: 而在 371 条全量上它是 0.071，远高于阈值 —— 同一个字段两种结论，
#: 差别只来自样本量。所以低于此值时报告必须显式标注"方差估计不可靠"。
LOW_N = 100
#: 置换检验次数（打乱字段取值重算 |ρ|）与种子
N_PERM = 4000
PERM_SEED = 20260101
#: 错位对照的随机种子：与 probe_llm_grounding.py 一致，便于对拍
SHUFFLE_SEED = 20260101
#: 字段是否"物理上应当随 Tx 位置变化"。材料/场景属性恒定是常识，不参与判定。
GEO_CONDITIONED: tuple[str, ...] = (
    "los_ratio", "reflection_ratio", "diffraction_ratio", "scattering_ratio",
    "pathloss_min_db", "pathloss_max_db",
)


# ===== 几何事实解析（只用 provenance，不需要数据盘） =====


def parse_facts_text(text: str) -> dict[str, Any]:
    """从 `provenance.geometry_facts.facts_text` 还原几何事实。

    这段文本由 `src/prior/prompt_v3.py:compute_geometry_facts()` 生成，格式固定：
        Building fraction inside the 50 m disc around the Tx: 0.352
        Building fraction inside the 100 m disc around the Tx: 0.334
        The Tx pixel itself lies ON a building footprint.   (或 ... on open ground)
        Per-azimuth obstruction within 100 m (...):
          0 deg: blocked, 0 m
          ...
        Count of azimuths by obstruction class: blocked=8

    解析失败一律抛错（不静默填默认值，项目规则 6）。
    """
    r50 = r100 = None
    on_building = None
    obs: dict[int, str] = {}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Building fraction inside the 50 m disc"):
            r50 = float(s.rsplit(":", 1)[1])
        elif s.startswith("Building fraction inside the 100 m disc"):
            r100 = float(s.rsplit(":", 1)[1])
        elif "Tx pixel itself lies ON a building" in s:
            on_building = True
        elif "Tx pixel lies on open ground" in s:
            on_building = False
        elif s and s[0].isdigit() and " deg:" in s:
            head, rest = s.split(" deg:", 1)
            obs[int(head)] = rest.strip().split(",")[0].strip()
    if r50 is None or r100 is None or on_building is None or len(obs) != 8:
        raise ValueError(f"facts_text 解析不完整：r50={r50} r100={r100} "
                         f"on_building={on_building} sectors={len(obs)}")
    return {"r50": r50, "r100": r100, "tx_on_building": on_building, "obstruction": obs}


def geometry_drivers(rec: dict[str, Any]) -> dict[str, float]:
    """先验记录 → 7 个几何驱动量。

    口径与 `scripts/probe_llm_grounding.py:geometry_drivers()` 一致：驱动量来自
    `geometric_obstruction()` 的输出，其中 `nearest_building_m` 在
    `compute_geometry_facts()` 里被 `round(..., 1)`（差别 ≤0.05 m，对秩相关无影响；
    `--validate-against` 会对拍确认）。这里改用同一份 provenance
    （`prior.sectors` 与几何事实同源），所以本脚本不需要 `data/radiomapseer/`。
    """
    facts = parse_facts_text(rec["provenance"]["geometry_facts"]["facts_text"])
    sec = sorted(rec["prior"]["sectors"], key=lambda s: int(s["azimuth_deg"]))
    labels = [str(s["obstruction"]) for s in sec]
    dists = [float(s["nearest_building_m"]) for s in sec]
    finite = [d for d in dists if d <= 400]
    return {
        "r50": float(rec["prior"]["tx_local_building_ratio_50m"]),
        "r100": float(rec["prior"]["tx_local_building_ratio_100m"]),
        "n_blocked": float(labels.count("blocked")),
        "n_dense": float(labels.count("dense")),
        "n_sparse": float(labels.count("sparse")),
        "min_dist": float(min(dists)),
        "mean_dist": float(np.mean(finite)) if finite else 500.0,
        # 下面两个不进 MOTIVATED_PAIRS，只用于报告里的几何背景描述
        "_tx_on_building": float(facts["tx_on_building"]),
        "_r50_text": float(facts["r50"]),
        "_r100_text": float(facts["r100"]),
    }


# ===== 数据加载 =====


def load_priors(priors_dir: Path, maps: set[int] | None = None) -> list[dict[str, Any]]:
    files = sorted(priors_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"目录里没有 *.json：{priors_dir}")
    rows: list[dict[str, Any]] = []
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        if "prior" not in rec or "provenance" not in rec:
            raise SystemExit(f"{f.name} 不是 v3 先验记录（缺 prior/provenance）")
        if maps is not None and int(rec["map_id"]) not in maps:
            continue
        fc = rec["provenance"].get("field_consistency")
        if not fc:
            raise SystemExit(f"{f.name} 缺 provenance.field_consistency，无法做仪器 1")
        prior = rec["prior"]
        row: dict[str, Any] = {
            "sample_id": rec["sample_id"],
            "map_id": int(rec["map_id"]),
            "tx_id": int(rec["tx_id"]),
            "consistency": fc,
            "n_samples": int(fc.get("n_samples", 0)),
            "overall": float(fc["overall"]),
            "model": str(rec["provenance"].get("model", "?")),
            "temperature": float(rec["provenance"].get("temperature", float("nan"))),
            "generated_at": str(rec["provenance"].get("generated_at", "")),
            "v3_include_image": rec["provenance"].get("v3_include_image"),
            "sectors": prior["sectors"],
        }
        for k in EM_NUM_KEYS:
            row[k] = float(prior[k])
        for k in LLM_CAT_KEYS:
            row[k] = str(prior[k])
        # 供 build_features 使用的逐扇形列。
        # 必须**按方位角排序**再摊平：schema 只要求 sectors 恰好覆盖 8 个方位角，
        # 不保证数组顺序；按数组下标取会与驱动量错位（且这种错位极难察觉）。
        sec_sorted = sorted(prior["sectors"], key=lambda s: int(s["azimuth_deg"]))
        if [int(s["azimuth_deg"]) for s in sec_sorted] != list(range(0, 360, 45)):
            raise SystemExit(f"{f.name} 的 sectors 方位角不是 0..315 步长 45")
        for i, s in enumerate(sec_sorted):
            row[f"_lab_{i}"] = str(s["obstruction"])
            row[f"_dist_{i}"] = float(s["nearest_building_m"])
        if "geometry_facts" not in rec["provenance"]:
            raise SystemExit(f"{f.name} 缺 provenance.geometry_facts（v3 设计应当都有）")
        row.update(geometry_drivers(rec))
        rows.append(row)
    if not rows:
        raise SystemExit(f"按 --limit-maps 过滤后没有样本了（地图 {sorted(maps or [])}）")
    if len(rows) < 30:
        print(f"[警告] 样本只有 {len(rows)} 个，统计量置信区间会很宽\n")
    return rows


# ===== 仪器 1：可重复性（逐字段，零 API 成本） =====


def consistency_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """逐字段汇总多轮采样一致率。

    `field_consistency[k]` 的定义（见 generate.py:aggregate_em）：
        数值字段 = 各轮取值落在**样本均值 ±10% 量程**内的比例；
        类别字段 = 与众数相同的比例。
    它**不是**位级确定性：落盘的是各轮均值，所以"完全一致"只意味着各轮彼此
    都落在 ±10% 量程内。为防误读，这里把每个数值字段的绝对容差一起报出来，
    并反推"各轮极差至少有多大"：一致率 <1.0 ⟺ 至少一轮与均值差 >0.1·量程
    ⟹ 各轮极差 ≥ 0.2·量程（2 轮）/ ≥0.3·量程（3 轮，最坏情形）。
    """
    out: dict[str, dict[str, Any]] = {}
    for k in list(EM_NUM_KEYS) + list(LLM_CAT_KEYS):
        vals = [float(r["consistency"][k]) for r in rows]
        if k in EM_NUM_KEYS:
            rng = NUM_RANGES[k][1] - NUM_RANGES[k][0]
            tol = 0.10 * rng
            min_range = 0.2 * rng if len(rows) else float("nan")
            note = (f"±{_num(tol)} 量程单位（= 量程 {NUM_RANGES[k][0]:g}..{NUM_RANGES[k][1]:g} 的 10%）")
        else:
            tol = float("nan")
            min_range = float("nan")
            note = "类别字段：与众数相同的比例"
        out[k] = {
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "std": float(np.std(vals)),
            "share_all_rounds_agree": float(np.mean([v >= 1.0 - 1e-12 for v in vals])),
            "share_below_2of3": float(np.mean([v <= 2.0 / 3.0 + 1e-12 for v in vals])),
            "share_zero": float(np.mean([v <= 1e-12 for v in vals])),
            "tolerance_abs": tol,
            "min_within_sample_range_implied": min_range,
            "tolerance_note": note,
        }
    out["_overall"] = {
        "n": len(rows),
        "mean": float(np.mean([r["overall"] for r in rows])),
        "median": float(np.median([r["overall"] for r in rows])),
        "min": float(np.min([r["overall"] for r in rows])),
        "max": float(np.max([r["overall"] for r in rows])),
        "std": float(np.std([r["overall"] for r in rows])),
        "derived_confidence_counts": dict(Counter(
            derive_confidence(r["overall"]) for r in rows)),
        "llm_reported_confidence_mode_counts": dict(Counter(
            str(r["consistency"]["llm_reported_confidence_mode"]) for r in rows)),
    }
    return out


# ===== 仪器 2：接地（ρ + 错位对照 + 置换检验） =====


def permutation_p(a: np.ndarray, b: np.ndarray, rho: float, n_perm: int = N_PERM,
                  seed: int = PERM_SEED) -> float:
    """|ρ| 的双侧置换检验 p 值。不假设分布，n 小时也有效。"""
    if not np.isfinite(rho) or len(a) < 5:
        return float("nan")
    rng = np.random.RandomState(seed)
    target = abs(rho)
    hits = 0
    for _ in range(n_perm):
        if abs(spearman(a, rng.permutation(b))) >= target:
            hits += 1
    return float((hits + 1) / (n_perm + 1))


def _corr_pair(y: np.ndarray, g: np.ndarray) -> tuple[float, float, bool, str]:
    """(ρ, 置换 p, 是否可检验, 说明)。**退化输入必须显式说明**，不留 NaN 悬念。

    实测教训：`min_dist` 在一批样本上恒定时，ρ 与 p 都是 NaN（`spearman()` 对常数
    返回 NaN）。NaN 在报告里会显示成"—"，读者无法区分"没测"与"测不了"，
    而在 JSON 里 `0 <= nan <= 1` 又是 False —— 会把"不可检验"误诊成"数值越界"。
    所以这里显式返回可检验标志与原因。
    """
    if len(y) < 5 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), False, "该字段在本批样本上恒定，无法做相关检验"
    if len(np.unique(g)) < 2:
        return float("nan"), float("nan"), False, "该驱动量在本批样本上恒定，无法做相关检验"
    rho = spearman(y, g)
    if not np.isfinite(rho):
        return rho, float("nan"), False, "ρ 算不出来（输入退化）"
    return rho, permutation_p(y, g, rho), True, ""


def grounding_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """预先指定的物理配对的 ρ、错位对照 ρ、置换 p 值。"""
    n = len(rows)
    # 与 probe_llm_grounding.py 完全相同的错位构造（无不动点）
    perm = np.roll(np.random.RandomState(SHUFFLE_SEED).permutation(n), 1)

    pairs: list[dict[str, Any]] = []
    for field, drv, sign in MOTIVATED_PAIRS:
        y = np.array([r[field] for r in rows], dtype=float)
        g = np.array([r[drv] for r in rows], dtype=float)
        rho, p_perm, testable, why = _corr_pair(y, g)
        rho_sh = spearman(y, g[perm]) if testable else float("nan")
        want = 1.0 if sign == "+" else -1.0
        dir_ok = bool(np.isfinite(rho) and np.sign(rho) == want)
        pairs.append({
            "field": field, "driver": drv, "expected_sign": sign,
            "rho": rho, "rho_shuffled": rho_sh,
            "delta": (float(rho - rho_sh) if np.isfinite(rho) and np.isfinite(rho_sh)
                      else float("nan")),
            "direction_correct": dir_ok,
            "p_permutation": p_perm,
            "significant_p05": bool(np.isfinite(p_perm) and p_perm < 0.05),
            "testable": testable,
            "not_testable_reason": why,
            # 比旧的 2*SE 判据更严：方向正确 **且** 置换检验显著
            "supported": bool(dir_ok and np.isfinite(p_perm) and p_perm < 0.05),
        })

    per_field: dict[str, dict[str, Any]] = {}
    for k in EM_NUM_KEYS:
        my = [p for p in pairs if p["field"] == k]
        if not my:
            per_field[k] = {"tested": 0, "any_supported": False}
            continue
        testable = [p for p in my if p["testable"]]
        if not testable:
            per_field[k] = {
                "tested": len(my), "n_testable": 0, "any_supported": False,
                "direction_correct_any": False, "n_pairs_supported": 0,
                "rho_absmax": float("nan"), "rho_absmax_signed": float("nan"),
                "rho_absmax_shuffled": float("nan"), "p_permutation": float("nan"),
                "best_driver": my[0]["driver"], "best_expected_sign": my[0]["expected_sign"],
                "not_testable_reason": my[0]["not_testable_reason"],
            }
            continue
        best = max(testable, key=lambda p: abs(p["rho"]) if np.isfinite(p["rho"]) else -1)
        per_field[k] = {
            "tested": len(my), "n_testable": len(testable),
            "best_driver": best["driver"],
            "best_expected_sign": best["expected_sign"],
            "rho_absmax": float(abs(best["rho"])) if np.isfinite(best["rho"]) else float("nan"),
            "rho_absmax_signed": best["rho"],
            "rho_absmax_shuffled": best["rho_shuffled"],
            "p_permutation": best["p_permutation"],
            "any_supported": bool(any(p["supported"] for p in my)),
            "direction_correct_any": bool(any(p["direction_correct"] for p in my)),
            "n_pairs_supported": int(sum(p["supported"] for p in my)),
        }

    se = 1.0 / np.sqrt(max(1, n - 1))
    return {
        "n": n, "spearman_se": float(se), "tested_pairs": len(pairs),
        "testable_pairs": int(sum(p["testable"] for p in pairs)),
        "direction_correct": int(sum(p["direction_correct"] for p in pairs)),
        "supported": int(sum(p["supported"] for p in pairs)),
        "pairs": pairs, "per_field": per_field,
        "shuffle_seed": SHUFFLE_SEED, "n_permutation": N_PERM,
    }


# ===== 诊断：地图层内接地（区分"真随 Tx"与"地图级常量"） =====


def within_map_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """按地图分层重算 ρ 与散度，回答"该字段随 Tx 变，还是只在描述地图"。

    为什么关键：371 条先验来自 5 张地图。跨地图比较会把"地图之间的差异"算成
    "字段随几何变化"。上一版先验的失败模式恰恰是**地图级常量**（同图 80 个 Tx
    几乎同一个向量）。地图内 n≤80，ρ 标准误 ~0.11，所以这里只作**描述**
    （逐地图是否同号、量级是否一致），不逐地图做显著性检验。
    """
    maps = sorted({r["map_id"] for r in rows})
    per_field: dict[str, Any] = {}
    for k in list(EM_NUM_KEYS) + list(LLM_CAT_KEYS):
        first = next((p for p in MOTIVATED_PAIRS if p[0] == k), None)
        per_map: dict[str, Any] = {}
        signs: list[int] = []
        for m in maps:
            sub = [r for r in rows if r["map_id"] == m]
            if len(sub) < 20:
                per_map[str(m)] = {"n": len(sub), "status": "样本太少，跳过"}
                continue
            if k in EM_NUM_KEYS:
                y = np.array([r[k] for r in sub], dtype=float)
                lo, hi = NUM_RANGES[k]
                entry: dict[str, Any] = {
                    "n": len(sub), "std": float(y.std()),
                    "std_over_range": float(y.std() / (hi - lo)),
                    "unique": int(len(set(np.round(y, 9)))),
                }
                if first is not None:
                    g = np.array([r[first[1]] for r in sub], dtype=float)
                    rho = spearman(y, g)
                    entry.update({"driver": first[1], "expected_sign": first[2], "rho": rho})
                    if np.isfinite(rho) and abs(rho) > 1e-9:
                        signs.append(int(np.sign(rho) == (1 if first[2] == "+" else -1)))
            else:
                cnt = Counter(str(r[k]) for r in sub)
                top1, top1n = cnt.most_common(1)[0]
                entry = {"n": len(sub), "unique": len(cnt),
                         "top1_share": top1n / len(sub), "top1_value": top1,
                         "counts": dict(cnt.most_common())}
            per_map[str(m)] = entry
        dispersions = [e["std_over_range"] for e in per_map.values()
                       if "std_over_range" in e]
        tops = [e["top1_share"] for e in per_map.values() if "top1_share" in e]
        per_field[k] = {
            "per_map": per_map,
            "sign_agreement_with_expected": (f"{sum(signs)}/{len(signs)}"
                                             if signs else "无预指定配对"),
            "std_over_range_min": float(min(dispersions)) if dispersions else float("nan"),
            "std_over_range_max": float(max(dispersions)) if dispersions else float("nan"),
            "top1_share_min": float(min(tops)) if tops else float("nan"),
            "top1_share_max": float(max(tops)) if tops else float("nan"),
        }
    return {"maps": maps, "per_field": per_field}


# ===== 仪器 3：可复现性（几何 → 字段的 CV R²） =====


def build_features(rows: list[dict[str, Any]]) -> np.ndarray:
    """S1(5) / S2(13) / S3(45) 特征矩阵，逐列顺序与 distillability 脚本一致。"""
    az = list(range(8))
    ob_order = ("open", "sparse", "dense", "blocked")
    feats: list[list[float]] = []
    for r in rows:
        labs = [str(r[f"_lab_{i}"]) for i in az]
        dists = [float(r[f"_dist_{i}"]) for i in az]
        row = [float(r["r50"]), float(r["r100"]), float(r["_tx_on_building"]),
               float(labs.count("blocked")), float(min(dists))]
        row += dists
        for i in az:
            for ob in ob_order:
                row.append(1.0 if labs[i] == ob else 0.0)
        feats.append(row)
    return np.asarray(feats, dtype=np.float64)


def cv_r2(X: np.ndarray, y: np.ndarray, model: str, seed: int = 0) -> float:
    """5 折交叉验证 R²。**与 probe_llm_distillability.py 逐参数一致**，便于对拍。"""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if model == "ridge":
        est = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    else:
        est = GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                        learning_rate=0.05, random_state=seed)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    return float(r2_score(y, cross_val_predict(est, X, y, cv=kf)))


def in_sample_r2(X: np.ndarray, y: np.ndarray, model: str = "gbm",
                 seed: int = 0) -> float:
    """**样本内** R²（非线性上限估计）。

    为什么不只用 5 折 CV：字段若被量化成少数几个取值（本项目 `los_ratio` 就是），
    CV R² 会因为"离散目标的噪声"被低估，样本内 R² 才反映
    "这个字段到底是不是几何量的单调函数"。
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score

    est = GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                   learning_rate=0.05, random_state=seed)
    est.fit(X, y)
    return float(r2_score(y, est.predict(X)))


def oracle_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """冗余复核：**单个**原始几何驱动量能不能把该字段解释掉。

    为什么不能只看仪器 3：仪器 3 用整组几何特征做 5 折 CV，目标是"泛化"。但对
    "这个字段是不是几何量的重新编码"这个问题，CV R² 会被目标的**量化**压低估
    （`los_ratio` 逐图只有 6~9 个取值），而且 R² 在拟合很差时无下界、可以出现
    很大的负值，做差值会放大噪声。所以这里改用两个**有界**的量的组合：

      * `in_sample_r2`：只用一个驱动量（+二次项）在小树模型上的**样本内** R²。
        它**不是**泛化估计，只回答"这条关系存不存在、有多强"。
        小树（depth=2）容量极低，所以样本内 R² 不会像深树那样爆掉。
      * `spearman`：该驱动量与该字段的秩相关（单调关系强度）。

    判据：`in_sample_r2 ≥ ORACLE_R2`（0.85）→ 该字段基本就是那个驱动量的一个
    单调/分段函数；而那个驱动量在 v3 里**本来就已经喂给下游了**
    （`sectors` / `local_ratio` 就在条件向量里），所以它的边际贡献 ≈ 0。
    """
    if len(rows) < 30:
        return {"status": "样本太少，跳过", "per_field": {}}
    X = build_features(rows)
    # 每个驱动量在 S2 特征块里的列位置（与 build_features 的顺序对应）
    drv_col = {"r50": 0, "r100": 1, "tx_on_building": 2, "n_blocked": 3,
               "min_dist": 4}
    for i in range(8):
        drv_col[f"dist_az{45 * i}"] = 5 + i
    per_field: dict[str, Any] = {}
    for k in EM_NUM_KEYS:
        y = np.array([r[k] for r in rows], dtype=float)
        rows_out: list[dict[str, Any]] = []
        for name, col in drv_col.items():
            x = X[:, col]
            if np.all(x == x[0]):
                continue
            # 只用 (x, x^2)：小树模型自身已能表达单调与阈值关系
            r2_in = in_sample_r2(np.column_stack([x, x ** 2]), y)
            rho = spearman(y, x)
            rows_out.append({"driver": name, "in_sample_r2": r2_in,
                             "spearman": rho})
        if not rows_out:
            per_field[k] = {"tested": 0}
            continue
        best = max(rows_out, key=lambda e: e["in_sample_r2"])
        per_field[k] = {
            "tested": len(rows_out),
            "per_driver": sorted(rows_out, key=lambda e: -e["in_sample_r2"]),
            "best_driver": best["driver"],
            "best_in_sample_r2": best["in_sample_r2"],
            "best_spearman": best["spearman"],
            "encoded_by_single_driver": bool(best["in_sample_r2"] >= ORACLE_R2),
        }
    return {
        "n": len(rows),
        "threshold_in_sample_r2": ORACLE_R2,
        "note": "in_sample_r2 是**样本内**（非泛化）指标，容量极低的小树；"
                "它只回答「这条关系有多强」，越高越说明该字段是那个几何量的重新编码。",
        "per_field": per_field,
    }


def distillability_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) < 30:
        return {"n": len(rows), "status": "样本太少，跳过仪器 3", "per_field": {}}
    X = build_features(rows)
    sets = {"S1": slice(0, 5), "S2": slice(0, 13), "S3": slice(0, 45)}
    rng = np.random.RandomState(0)
    per_field: dict[str, Any] = {}
    for k in EM_NUM_KEYS:
        y = np.array([r[k] for r in rows], dtype=float)
        r2 = {name: cv_r2(X[:, sl], y, "ridge") for name, sl in sets.items()}
        r2["gbm_S3"] = cv_r2(X[:, :45], y, "gbm")
        r2_sh = cv_r2(X[:, :45], rng.permutation(y), "ridge")
        per_field[k] = {
            "std": float(y.std()),
            "r2": {kk: float(vv) for kk, vv in r2.items()},
            "r2_best": float(max(r2.values())),
            "r2_in_sample_gbm": in_sample_r2(X[:, :45], y),
            "r2_shuffled": float(r2_sh),
            "shuffle_ok": bool(r2_sh <= 0.05),
        }
    return {
        "n": len(rows), "n_features": int(X.shape[1]),
        "feature_sets": {"S1": "5 个标量", "S2": "S1 + 8 方位最近距离",
                         "S3": "S2 + 8 方位遮挡 one-hot", "gbm_S3": "S3 + GBM（非线性上限）"},
        "per_field": per_field,
    }


# ===== 类型学判定 =====


def _num(v: float) -> str:
    """数值的短格式：≤5 保留两位，>5 保留一位。用于报告里的量程/容差。"""
    if not np.isfinite(v):
        return "—"
    return f"{v:.2f}" if abs(v) <= 5 else f"{v:.1f}"


def dispersion(variance: dict[str, Any]) -> tuple[float, str]:
    """字段的归一化散度与判据说明。

    数值字段：std ÷ 量程（对 0.9 GHz 的四个机制权重，量程 = 1，所以直接就是 std）。
    类别字段：1 − 众数占比（0 = 只有一个取值）。
    """
    if "top1_share" in variance:
        return float(1.0 - variance["top1_share"]), f"1−top1占比 (top1={variance['top1_share']:.2f})"
    return float(variance.get("std_over_range", float("nan"))), "std÷量程"


def classify(k: str, cons: dict[str, Any], ground: dict[str, Any],
             dist: dict[str, Any], variance: dict[str, Any],
             within: dict[str, Any] | None = None,
             n: int | None = None) -> dict[str, Any]:
    """§1.4 的判读表 → 机器可判定的类型标签。

    判定顺序（顺序即理由，且每一支都只用**两个正交测量**）：
      1. 散度 ≈0        → 常数（无区分度）。与 ρ 无关：没有方差的字段不可能是信息。
      2. 可重复性低     → 噪声。
      3. 接地成立（方向对 + 置换 p<0.05）：
           **但逐图 ρ 与总体 ρ 符号大面积不一致** → 接地不稳（地图间混淆）：
             总体 ρ 来自"地图之间的几何差异 + 地图内的偏置"，不是字段跟着
             本 Tx 的几何走。对"给这个 Tx 的条件向量"没有价值（Simpson 式混淆）。
           逐图一致：
             R² 高 → 真实但冗余（LLM 在做特征工程，直接喂几何更好）
             R² 低 → 真实且有残余信息（唯一支持继续的类型）
      4. 接地不成立：
           方向**反了**且显著    → 自信地错（系统性幻觉）
           其余                  → 散度不小 = 幻觉；散度小 = 弱/无信号
    """
    within = within or {}
    disp, disp_note = dispersion(variance)
    best_r2 = dist.get("r2_best", float("nan"))
    rho = ground.get("rho_absmax_signed", float("nan"))
    p = ground.get("p_permutation", float("nan"))
    p_txt = f"{p:.4f}" if np.isfinite(p) else "—"
    rho_txt = f"{rho:+.2f}" if np.isfinite(rho) else "—"
    drv = ground.get("best_driver", "—")
    # 逐图符号一致性：只有真正算过逐图 ρ 的字段才有（材料类字段无预指定配对）
    sign_txt = str(within.get("sign_agreement_with_expected", ""))
    sign_den = int(sign_txt.split("/")[-1]) if "/" in sign_txt else 0
    sign_num = int(sign_txt.split("/")[0]) if "/" in sign_txt else 0
    inconsistent = bool(sign_den >= 3 and sign_num / sign_den < 0.6)

    if np.isfinite(disp) and disp <= CV_EPS:
        return _add_low_n_caveat({
            "type": "常数（无区分度）",
            "verdict": f"归一化散度 {disp:.3f} ≤ {CV_EPS}（{disp_note}）→ 取值基本恒定，"
                       f"不携带逐样本信息；恒定本身可能是对的（材料/场景属性属常识），"
                       f"但下游不能指望它带信息",
            "in_condition_vector": False, "signal": "none"}, n)
    if not np.isfinite(cons["median"]) or cons["median"] < CONSISTENCY_HI:
        return _add_low_n_caveat({
            "type": "噪声",
            "verdict": f"多轮一致率中位数 {cons['median']:.2f} < {CONSISTENCY_HI}"
                       f" → 同一输入下取值不稳，下游只能当噪声",
            "in_condition_vector": False, "signal": "none"}, n)
    if ground.get("any_supported"):
        if inconsistent:
            return _add_low_n_caveat({
                "type": "接地不稳（地图间混淆）",
                "verdict": f"总体 ρ={rho_txt}（↔ {drv}，p={p_txt}，方向与预期"
                           f"{ground.get('best_expected_sign', '?')} 相符）**成立**，"
                           f"但地图内部单独算时只有 {sign_txt} 张图的 ρ 与预期同号"
                           f"（逐图 ρ 反复变号）→ 这段总体相关主要来自"
                           f"**地图之间的几何分布差异**，而不是字段跟着本 Tx 的几何走；"
                           f"对「同图不同 Tx」的条件向量没有分辨力",
                "in_condition_vector": False, "signal": "confounded"}, n)
        if np.isfinite(best_r2) and best_r2 >= R2_ENC:
            return _add_low_n_caveat({
                "type": "真实但冗余",
                "verdict": f"接地成立（ρ={rho_txt} ↔ {drv}，p={p_txt}；逐图 ρ 同号 "
                           f"{sign_txt or '—'}）且 R²={best_r2:.2f} ≥ {R2_ENC} → "
                           f"几何可直接复现它，喂几何比喂它更好",
                "in_condition_vector": False, "signal": "redundant"}, n)
        return _add_low_n_caveat({
            "type": "真实且有残余信息",
            "verdict": f"接地成立（ρ={rho_txt} ↔ {drv}，p={p_txt}）且 "
                       f"R²={best_r2:.2f} < {R2_ENC} → 随几何变化、但几何特征拟合不出来"
                       + (f"；逐图 ρ 同号 {sign_txt}" if sign_txt else ""),
            "in_condition_vector": True, "signal": "residual"}, n)
    # ---- 接地不成立 ----
    significant_wrong = bool(np.isfinite(p) and p < 0.05 and not ground.get("direction_correct_any"))
    if significant_wrong:
        res = {"type": "自信地错（系统性幻觉）",
               "verdict": f"最强配对 ρ={rho_txt}（↔ {drv}，p={p_txt}）**方向与物理预期相反**，"
                          f"且散度 {disp:.3f} 不小 → 它确实在随几何变，但朝错的方向变",
               "in_condition_vector": False, "signal": "harmful"}
    elif np.isfinite(disp) and disp > CV_EPS:
        res = {"type": "幻觉（自信地错）",
               "verdict": f"最强预指定配对 ρ={rho_txt}（↔ {drv}，p={p_txt}）不成立，"
                          f"但散度 {disp:.3f} 不小 → 数值在乱跳却不反映给定几何",
               "in_condition_vector": False, "signal": "harmful"}
    else:
        res = {"type": "弱/无信号",
               "verdict": f"ρ 不显著（{rho_txt}, p={p_txt}）且散度小（{disp:.3f}）"
                          f"→ 既不反映场景也无害，等同无信息",
               "in_condition_vector": False, "signal": "none"}
    return _add_low_n_caveat(res, n)


def _add_low_n_caveat(res: dict[str, Any], n: int | None) -> dict[str, Any]:
    """样本量太小时给结论加保留：方差/ρ 的估计本身就不稳，不能当作定论。

    为什么必须加：实测同一字段在 24 条子集上被判"常数"、在 371 条全量上被判
    "真实且有残余信息"，差别只来自样本量。没有这条标注，读报告的人会把
    "n 太小导致的分类抖动"当成先验的性质。
    """
    res = dict(res)
    res["n"] = n
    if n is not None and n < LOW_N:
        res["low_n_warning"] = (f"样本 {n} < {LOW_N}：散度与 ρ 的估计不稳"
                                f"（std 的相对标准误 ≈ 1/√(2n)），"
                                f"本行结论**不可作为定论**，只用于流程自检")
    return res


def build_typology(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cons = consistency_stats(rows)
    ground = grounding_stats(rows)
    dist = distillability_stats(rows)
    oracle = oracle_stats(rows)
    wmap = within_map_stats(rows)

    fields: list[dict[str, Any]] = []
    for k in list(EM_NUM_KEYS) + list(LLM_CAT_KEYS):
        if k in EM_NUM_KEYS:
            vals = np.array([r[k] for r in rows], dtype=float)
            lo, hi = NUM_RANGES[k]
            variance: dict[str, Any] = {
                "std": float(vals.std()), "range": hi - lo,
                "std_over_range": float(vals.std() / (hi - lo)),
                "unique": int(len(set(np.round(vals, 9)))),
                "min": float(vals.min()), "max": float(vals.max()),
                "mean": float(vals.mean()),
            }
        else:
            cnt = Counter(str(r[k]) for r in rows)
            n = len(rows)
            probs = np.array([c / n for c in cnt.values()], dtype=float)
            top1, top1n = cnt.most_common(1)[0]
            variance = {
                "unique": len(cnt),
                "top1_share": top1n / n,
                "top1_value": top1,
                "entropy_bits": float(-(probs * np.log2(probs)).sum()),
                "counts": dict(cnt.most_common()),
            }
        g = ground["per_field"].get(k, {"tested": 0, "any_supported": False,
                                        "rho_absmax_signed": float("nan"),
                                        "p_permutation": float("nan")})
        d = dist.get("per_field", {}).get(k, {})
        w = wmap["per_field"].get(k, {})
        o = oracle.get("per_field", {}).get(k, {})
        # 冗余复核：仪器 3 是"用**整组**几何特征拟合"，仪器 3b 是"只用**单个**驱动量拟合"。
        # 后者高而前者不算高，说明该字段是某个几何量的重新编码 —— 而那个几何量
        # 在 v3 里本来就已经喂给下游了（sectors/local_ratio），所以边际贡献 ≈ 0。
        redundant_confirmed = bool(o.get("encoded_by_single_driver", False))
        fields.append({
            "field": k,
            "kind": "LLM 数值" if k in EM_NUM_KEYS else "LLM 类别",
            "physically_geo_conditioned": k in GEO_CONDITIONED,
            "repeatability": cons[k],
            "grounding": g,
            "reproducibility": d,
            "oracle": o,
            "variance": variance,
            "within_map": w,
            "redundant_confirmed_by_oracle": redundant_confirmed,
            **classify(k, cons[k], g, d, variance, w, n=len(rows)),
        })

    counts = Counter(f["type"] for f in fields)
    return {
        "n_samples": len(rows),
        "maps": sorted({r["map_id"] for r in rows}),
        "samples_per_map": dict(sorted(Counter(r["map_id"] for r in rows).items())),
        "model": rows[0]["model"],
        "temperature": rows[0]["temperature"],
        "n_rounds_per_sample": rows[0]["n_samples"],
        "v3_include_image": rows[0]["v3_include_image"],
        "generated_at_range": [min(r["generated_at"] for r in rows),
                               max(r["generated_at"] for r in rows)],
        "thresholds": {"CV_EPS": CV_EPS, "CAT_TOP1_CONST": CAT_TOP1_CONST,
                       "CONSISTENCY_HI": CONSISTENCY_HI, "RHO_STRONG": RHO_STRONG,
                       "R2_HI": R2_HI, "R2_ENC": R2_ENC,
                       "N_PERM": N_PERM, "SHUFFLE_SEED": SHUFFLE_SEED,
                       "WITHIN_MAP_MIN_SIGN_AGREEMENT": 0.6},
        "fields": fields,
        "type_counts": dict(counts),
        "low_n_warning": (f"样本 {len(rows)} < {LOW_N}：散度/ρ/R² 的估计都不稳，"
                          f"本报告的**类型标签不可作为定论**"
                          if len(rows) < LOW_N else None),
        "fields_with_residual_information": [f["field"] for f in fields
                                             if f["type"] == "真实且有残余信息"],
        "fields_harmful": [f["field"] for f in fields if f["signal"] == "harmful"],
        "fields_redundant": [f["field"] for f in fields if f["signal"] == "redundant"],
        "fields_confounded": [f["field"] for f in fields if f["signal"] == "confounded"],
        # ✅ 字段的强度明细：类型相同但边际价值可以差很多，这里把两个方向
        # （逐图一致性 vs 单几何量可解释度）一起给出，避免把"强接地"直接读成"强增量"。
        "fields_residual_strength": [
            {"field": f["field"],
             "sign_agreement": f.get("within_map", {}).get("sign_agreement_with_expected"),
             "r2_cv": (f["reproducibility"] or {}).get("r2_best"),
             "r2_in_sample_single_driver": (f.get("oracle") or {}).get("best_in_sample_r2"),
             "best_driver": (f.get("oracle") or {}).get("best_driver"),
             "redundant_confirmed_by_oracle": f["redundant_confirmed_by_oracle"]}
            for f in fields if f["type"] == "真实且有残余信息"],
        "overall_consistency": cons["_overall"],
        "grounding_pairs": ground["pairs"],
        "grounding_summary": {kk: vv for kk, vv in ground.items()
                              if kk not in ("pairs", "per_field")},
        "distillability_summary": {kk: vv for kk, vv in dist.items() if kk != "per_field"},
        "oracle_summary": {kk: vv for kk, vv in oracle.items() if kk != "per_field"},
        "fields_redundant_confirmed_by_oracle": [
            f["field"] for f in fields if f["redundant_confirmed_by_oracle"]],
        "within_map": wmap,
        "program_computed_fields": {
            "note": "v3 里 sectors(40 维) 与 tx_local_building_ratio_*(2 维) 由程序精确"
                    "测量（provenance.field_sources=program），不在本表的 LLM 类型学里；"
                    "它们的一致性是结构性的 1.0",
            "fields": ["sectors", "tx_local_building_ratio_50m",
                       "tx_local_building_ratio_100m"],
        },
        "prior_dim_all_groups": vector_dim(),
    }


def validate_against(rows: list[dict[str, Any]], ref_path: Path) -> dict[str, Any]:
    """与 probe_llm_grounding.py 已跑过的结果对拍（同一子集、同一配对）。

    参考 JSON 只存了 verdict 与计数（不存逐配对 ρ），所以能自动核对的是
    "方向正确数"、"显著数"与 Spearman 标准误这三个计数/常量。
    """
    report = json.loads(ref_path.read_text(encoding="utf-8"))
    g = grounding_stats(rows)
    return {
        "reference": str(ref_path),
        "n_ref": report.get("n"), "n_here": g["n"],
        "direction_correct_ref": report.get("direction_correct"),
        "direction_correct_here": g["direction_correct"],
        # 注意：旧脚本的"显著"判据是 |ρ−ρ_错位| > 2·SE，本脚本用的是置换检验 p<0.05。
        # 两者可以不同，这不是矛盾，是两个判据；只作为提示打印。
        "supported_old_criterion_ref": report.get("significant_vs_shuffled"),
        "supported_permutation_here": g["supported"],
        "se_ref": report.get("spearman_se"), "se_here": g["spearman_se"],
        "match_direction": report.get("direction_correct") == g["direction_correct"],
        "match_se": (report.get("spearman_se") is not None
                     and abs(float(report["spearman_se"]) - g["spearman_se"]) < 1e-9),
    }


# ===== 报告 =====


def fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}" if np.isfinite(v) else "—"
    return str(v)


def render_markdown(t: dict[str, Any], priors_dir: str) -> str:
    L: list[str] = []
    A = L.append
    A("# L1 三仪器类型学：LLM 电磁先验到底有没有信息")
    A("")
    A("> 由 `scripts/probe_l1_typology.py` **自动生成**，禁止手改数字（AGENTS.md 规则 5）。")
    A("> 判读规则来自交接文档 §1.4。本文件回答的是「哪几个字段值得进条件向量」。")
    A("")
    A("## 0. 数据与口径")
    A("")
    A(f"- 先验目录：`{priors_dir}`")
    A(f"- 模型 `{t['model']}`｜temperature={t['temperature']}｜每样本 "
      f"{t['n_rounds_per_sample']} 轮采样｜送图={t['v3_include_image']}")
    A(f"- 样本 {t['n_samples']} 条，覆盖地图 {t['maps']}（逐图 {t['samples_per_map']}）")
    A(f"- 生成时间 {t['generated_at_range'][0]} ～ {t['generated_at_range'][1]}")
    if t.get("low_n_warning"):
        A("")
        A(f"> ⚠️ **样本量警告**：{t['low_n_warning']}")
        A("> 实测证据：同一个 `los_ratio` 在 24 条子集上被判「常数」（散度 0.038 < 0.05），"
          "在 371 条全量上被判「真实且有残余信息」（散度 0.071）。"
          "差别只来自样本量，不是先验的性质 —— 小样本结论只能当流程自检。")
    A("")
    A("三个仪器**互相不蕴含**（§1.3：低 R² 同时与「有信息」和「纯噪声」相容）：")
    A("")
    A("| 仪器 | 测量 | 来源 | 成本 |")
    A("|---|---|---|---|")
    A("| 1 可重复性 | 同一样本多轮采样的逐字段一致率 | `provenance.field_consistency` | "
      "0（生成时已记录） |")
    A("| 2 接地 | 字段 ↔ 给定几何事实的 Spearman ρ（预指定配对 + 错位对照 + 置换检验） | "
      "`provenance.geometry_facts` 重新解析 | 0（纯本地计算） |")
    A("| 3 可复现性 | 几何特征 5 折 CV 拟合该字段的 R²（三档特征集 × Ridge/GBM） | 同上 | 0 |")
    A("")
    A("仪器 1 的口径（`src/prior/generate.py:aggregate_em`）：数值字段 = 各轮取值落在"
      "**样本均值 ±10% 量程**内的比例；类别字段 = 与众数相同的比例。")
    A("")
    A("> ⚠️ 一致率 1.0 **不等于**确定性。对 `pathloss_max_db`（量程 190 dB）那个容差是 ±19 dB，"
      "对 `permittivity`（量程 19）是 ±1.9。本表的「散度/量程」列就是为了揭穿这个误读："
      "散度/量程 ≤ 0.05 才算实质恒定。")
    A("")
    A("## 1. 三仪器表（正文）")
    A("")
    A("| 字段 | 类型 | 可重复性(中位) | 接地 \\|ρ\\| (驱动量) | p(置换) | 可复现 R²(best) | "
      "散度/量程 | 唯一值 | 进条件向量 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for f in t["fields"]:
        g, d, v = f["grounding"], f["reproducibility"] or {}, f["variance"]
        rho = g.get("rho_absmax_signed")
        rho_s = ("—" if rho is None or not np.isfinite(rho)
                 else f"{abs(rho):.2f} ({g.get('best_driver', '—')})")
        A(f"| `{f['field']}` | {f['type']} | {fmt(f['repeatability']['median'], 3)} | {rho_s} | "
          f"{fmt(g.get('p_permutation'), 4)} | {fmt(d.get('r2_best'), 2)} | "
          f"{fmt(v.get('std_over_range'), 3)} | {v.get('unique', '—')} | "
          f"{'✅' if f['in_condition_vector'] else '❌'} |")
    A("")
    A(f"类型计数：{t['type_counts']}")
    A("")
    A("## 2. 逐字段判读")
    A("")
    for f in t["fields"]:
        A(f"### `{f['field']}` → **{f['type']}**")
        A("")
        A(f"- 判读：{f['verdict']}")
        rep = f["repeatability"]
        A(f"- 仪器 1：一致率 均值 {fmt(rep['mean'])} / 中位 {fmt(rep['median'])} / "
          f"最小 {fmt(rep['min'])}；各轮全落在容差内的样本占比 "
          f"{fmt(rep['share_all_rounds_agree'])}（容差：{rep['tolerance_note']}）")
        g = f["grounding"]
        if g.get("tested"):
            A(f"- 仪器 2：最强预指定配对 `{f['field']}` ↔ `{g['best_driver']}`"
              f"（预期 {g['best_expected_sign']}）ρ={fmt(g['rho_absmax_signed'])}、"
              f"错位对照 ρ={fmt(g['rho_absmax_shuffled'])}、置换 p={fmt(g['p_permutation'], 4)}"
              f"（n={t['n_samples']}，标准误 {fmt(t['grounding_summary']['spearman_se'])}）；"
              f"该字段 {g['n_pairs_supported']}/{g['tested']} 个配对同时满足"
              f"「方向对 + 置换显著」")
        d = f["reproducibility"] or {}
        if d:
            A(f"- 仪器 3：R² 最好 {fmt(d['r2_best'], 2)}（std={fmt(d['std'], 3)}、"
              f"样本内 GBM 上限 {fmt(d.get('r2_in_sample_gbm'), 2)}）、"
              f"错位对照 R²={fmt(d['r2_shuffled'], 2)}"
              f"{'（正常）' if d.get('shuffle_ok') else '（**异常，可能泄漏**）'}")
        o = f.get("oracle") or {}
        if o and o.get("tested"):
            A(f"- 仪器 3b（单驱动量复核）：只用 `{o.get('best_driver')}` 一个几何量"
              f"（+二次项、容量极低的小树）就能**样本内**解释该字段 "
              f"{fmt(o.get('best_in_sample_r2'), 2)} 的方差"
              f"（秩相关 ρ={fmt(o.get('best_spearman'))}）"
              + ("　→ **它就是该几何量的重新编码**，而那个几何量本来就已经在条件向量里了"
                 if f.get("redundant_confirmed_by_oracle") else ""))
        wm = f.get("within_map", {})
        if wm.get("per_map"):
            cells = []
            for m, e in wm["per_map"].items():
                if "rho" in e:
                    cells.append(f"map{m}: ρ={e['rho']:+.2f}(n={e['n']})")
                elif "unique" in e:
                    cells.append(f"map{m}: {e['unique']} 个取值(n={e['n']})")
                else:
                    cells.append(f"map{m}: {e.get('status', '?')}")
            A(f"- 地图层内：{'；'.join(cells)}｜ρ 与预期同号 "
              f"{wm.get('sign_agreement_with_expected', '—')}")
        A("")

    A("## 3. 两个衍生诊断")
    A("")
    A("### 3.1 字段到底随 Tx 变，还是只在描述地图")
    A("")
    A("同图不同 Tx 之间该字段的取值数、以及地图层内 ρ 的方向一致性 —— 这是"
      "「先验是不是地图级常量」的直接检查（上一版的失败模式）。")
    A("")
    A("| 字段 | 逐图唯一值数 | 逐图 ρ 与预期同号 | 逐图 散度/量程 范围 | 逐图 众数占比范围 |")
    A("|---|---|---|---|---|")
    for f in t["fields"]:
        wm = f.get("within_map", {})
        pm = wm.get("per_map", {})
        uniq = "/".join(str(e.get("unique", "-")) for e in pm.values()) if pm else "—"
        disp = (f"{fmt(wm.get('std_over_range_min'), 3)} ~ {fmt(wm.get('std_over_range_max'), 3)}"
                if np.isfinite(wm.get("std_over_range_min", float("nan"))) else "—")
        top1 = (f"{fmt(wm.get('top1_share_min'), 2)} ~ {fmt(wm.get('top1_share_max'), 2)}"
                if np.isfinite(wm.get("top1_share_min", float("nan"))) else "—")
        A(f"| `{f['field']}` | {uniq} | {wm.get('sign_agreement_with_expected', '—')} | "
          f"{disp} | {top1} |")
    A("")
    A("> 读法：**逐图唯一值数**是「同图 80 个 Tx 里该字段有几个不同取值」——"
      "若某个字段在每张图里只有个位数取值，它基本是**地图级**信息，"
      "对「这个 Tx 与同图别的 Tx 有何不同」没有分辨力。")
    A("> **逐图 ρ 同号数**是地图内部单独算的接地相关方向；"
      "若总体 ρ 显著而地图内 ρ 反复变号，那段总体相关就主要来自"
      "**地图之间的几何差异**，而不是字段真的跟着本 Tx 的几何走。")
    A("")

    A("### 3.2 整体一致性 vs 派生 confidence")
    A("")
    oc = t["overall_consistency"]
    A(f"- 整体一致率：均值 {fmt(oc['mean'])}、中位 {fmt(oc['median'])}、"
      f"范围 [{fmt(oc['min'])}, {fmt(oc['max'])}]")
    A(f"- 由一致率**派生**的 confidence 分布：{oc['derived_confidence_counts']}")
    A(f"- LLM **自报** confidence 的众数分布：{oc['llm_reported_confidence_mode_counts']}")
    A("")
    A("## 4. 结论：条件向量里该放什么")
    A("")
    if t["fields_with_residual_information"]:
        A("- ✅ 有残余信息（唯一支持继续的类型）："
          + "、".join(f"`{x}`" for x in t["fields_with_residual_information"]))
    else:
        A("- ❌ **没有任何字段**落在「真实且有残余信息」：LLM 的电磁输出要么是常数、"
          "要么能被几何复现、要么不接地。")
    if t["fields_redundant"]:
        A("- 🔁 真实但冗余（几何可直接复现）："
          + "、".join(f"`{x}`" for x in t["fields_redundant"]))
    if t["fields_confounded"]:
        A("- 🌀 接地不稳（总体 ρ 成立，但地图内部反复变号 → 主要来自地图之间的差异，"
          "不是跟着本 Tx 的几何走）："
          + "、".join(f"`{x}`" for x in t["fields_confounded"]))
    if t["fields_harmful"]:
        A("- ⚠️ 不接地但有方差（注入后下游得学会忽略）："
          + "、".join(f"`{x}`" for x in t["fields_harmful"]))
    if t["fields_residual_strength"]:
        A("")
        A("- 📏 ✅ 字段的**冗余复核**（仪器 3b）：LLM 输出的接地是强的，但它与条件向量里"
          "**已有的**几何量高度单调相关，这限制了它的边际贡献上限：")
        A("")
        A("  | 字段 | 逐图 ρ 同号 | 仪器 3 CV R² | 仪器 3b 单驱动量样本内 R² | 最强驱动量 |")
        A("  |---|---|---|---|---|")
        for e in t["fields_residual_strength"]:
            A(f"  | `{e['field']}` | {e['sign_agreement']} | {fmt(e['r2_cv'], 2)} | "
              f"{fmt(e['r2_in_sample_single_driver'], 2)} | `{e['best_driver']}` |")
        A("")
    A("")
    A("### 判定强度的分档（同一「类型」内部并不等强）")
    A("")
    A("| 字段 | 类型 | 总体 ρ | 逐图 ρ 同号 | 逐图 散度/量程 | 结论强度 |")
    A("|---|---|---|---|---|---|")
    for f in t["fields"]:
        if f["type"] not in ("真实且有残余信息", "真实但冗余", "接地不稳（地图间混淆）",
                            "自信地错（系统性幻觉）", "幻觉（自信地错）"):
            continue
        wm = f.get("within_map", {})
        agree = str(wm.get("sign_agreement_with_expected", "—"))
        strong = agree.split("/")[0] == agree.split("/")[-1] if "/" in agree else False
        A(f"| `{f['field']}` | {f['type']} | {fmt(f['grounding'].get('rho_absmax_signed'), 2)} | "
          f"{agree} | {fmt(wm.get('std_over_range_min'), 3)} ~ "
          f"{fmt(wm.get('std_over_range_max'), 3)} | "
          f"{'强（逐图同号）' if strong else '弱（逐图变号）'} |")
    A("")
    A("### 给下一步的口径建议")
    A("")
    A("- 条件向量应当**只保留**上表 ✅ 的字段，其余字段的贡献必须用"
      "`prior_v3_masked_em`（几何真、电磁置零、参数量逐位相同）来对照。")
    A("- ❌ 的字段不应当被静默删掉后再宣称「LLM 先验有效」：它们要么已被几何覆盖、"
      "要么不接地；删掉它们等于换了一个更小的先验，必须作为一个**新的配置**报出来，"
      "而不是当作「清理」。")
    A("- 仪器 3b 提示的**主要风险**：✅ 字段的接地 ρ 很强（如 `los_ratio` ρ=+0.73），"
      "但它同时与条件向量里**已有的**几何量 `min_dist`/`sectors` 高度单调相关"
      "（单驱动量样本内 R² 见上）。所以真正的对照不是"
      "「LLM 字段 vs 零」，而是「LLM 字段 vs 它所编码的那个几何量」。"
      "若只报前者，很容易把「几何量的重新编码」误读成「LLM 的贡献」。")
    A("- 核心表至少要同时报：`real_v3`（原样 22 维电磁）、"
      "`masked_em`（电磁置零、参数量逐位相同）、`geometry_only`（只有 42 维几何）、"
      "`real_v3_shuffled`（内容打乱）。四个数放在一起，「LLM 有没有正贡献」才可归因。")
    A("")
    A("## 5. 本次分析的已知限制（结论必须与这些限制一起引用）")
    A("")
    A(f"- 只覆盖地图 {t['maps']}；跨地图泛化的结论受这批地图的几何分布限制。")
    A("- 几何驱动量取自先验自带的 `provenance`，其中 `nearest_building_m` 在"
      "写入 prompt 前被四舍五入到 0.1 m（对秩相关无影响；有数据盘时用 "
      "`--validate-against` 对拍）。")
    A("- `sectors` 与 `tx_local_building_ratio_*` 是**程序**测量，不参与本表的"
      "LLM 类型学判定（它们的一致性由构造保证为 1.0）。")
    A("- 频率只有单值 0.9 GHz，无法检验频率条件化。")
    A("- 仪器 2 的配对是**预先指定**的（`MOTIVATED_PAIRS`，10 条），"
      "不是扫一遍挑最显著的；因此本表报的 p 值不需要多重比较校正，"
      "但也意味着**没被指定的关系不会被发现**。")
    A("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="L1 三仪器类型学（零 API 成本）")
    ap.add_argument("--priors", required=True, help="先验目录，例如 priors/prompt_v3/glm-4v-flash")
    ap.add_argument("--out", default="reports/llm_l1_typology.md")
    ap.add_argument("--json", default="reports/llm_l1_typology.json")
    ap.add_argument("--limit-maps", default=None, help="只用地图像 a:b（用于历史结果对拍）")
    ap.add_argument("--validate-against", default=None,
                    help="参考 JSON（probe_llm_grounding.py --out 的产物）做计数对拍")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    priors_dir = Path(args.priors)
    if not priors_dir.is_absolute():
        priors_dir = ROOT / priors_dir
    if not priors_dir.is_dir():
        raise SystemExit(f"先验目录不存在：{priors_dir}")
    maps = None
    if args.limit_maps:
        a, b = args.limit_maps.split(":")
        maps = set(range(int(a), int(b)))

    rows = load_priors(priors_dir, maps=maps)
    t = build_typology(rows)
    t["priors_dir"] = str(priors_dir)
    md = render_markdown(t, str(priors_dir))
    print(md)

    if not args.no_write:
        out_md, out_json = Path(args.out), Path(args.json)
        out_md = out_md if out_md.is_absolute() else ROOT / out_md
        out_json = out_json if out_json.is_absolute() else ROOT / out_json
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md, encoding="utf-8")
        out_json.write_text(json.dumps(t, ensure_ascii=False, indent=2, default=str),
                            encoding="utf-8")
        print(f"\n写入 {out_md}")
        print(f"写入 {out_json}")

    if args.validate_against:
        ref = Path(args.validate_against)
        ref = ref if ref.is_absolute() else ROOT / ref
        report = validate_against(rows, ref)
        print("\n---- 与已有接地检验对拍（probe_llm_grounding.py 的输出） ----")
        for k, v in report.items():
            print(f"  {k}: {v}")
        ok = bool(report["match_direction"] and report["match_se"])
        print(f"  对拍{'通过' if ok else '**不一致，需先查清**'}")
        return 0 if ok else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
