"""先验多样性体检 + 准入门槛判定（v2）。

为什么这一步是硬门槛（AGENTS.md 规则 1）
---------------------------------------
上一版 4000 条先验只有 **55 个唯一向量**（unique_ratio 0.0138）、最常见 2 个占 48.3%、
`confidence` 97.9% 都是 medium、同一张地图内不同 Tx 的唯一向量占比只有 0.104。
模型学到的是"按地图切换的常量偏置"，不是电磁先验——上一版的核心论点因此不成立。
所以：**生成先验后必须先跑本脚本，不达标禁止进入训练阶段。**

v2 相对 starter 版新增的检查
---------------------------
1. 兼容 v2 外层结构（`{"prior": {...}, "provenance": {...}}`）与旧扁平结构；
2. **sectors 专项**：8 方位遮挡剖面的唯一性、逐方位分布；
3. **同 map 内字段变化数**：直接对应"至少 3 个字段随 Tx 变化"这条验收；
4. **内容污点自动回归**：把上一版的具体病症做成机器检查 ——
   `reasoning` 里出现 "28GHz / 毫米波 / 建筑高度 / Tx 高度 25m" 这类与数据集
   （0.9 GHz、无高度信息、2D）矛盾的说法，以及 reasoning 的逐字复读率。
   上一版有 475 条（12%）先验写着 "28GHz…Tx 高度 25m"，这项检查就是为了让同一个
   错误不可能第二次悄悄通过；
5. 门槛分 `fail`（硬）与 `warn`（建议）：硬门槛不过退出码 1。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:  # Windows 控制台默认 GBK，统一按 UTF-8 输出
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from src.data.sectors import geometric_obstruction, local_building_ratio
from src.prior.schema import AZIMUTHS

# 与 schema v2 对齐的键列表（顺序即报告里的展示顺序）
NUM_KEYS = [
    "los_ratio", "reflection_ratio", "diffraction_ratio", "scattering_ratio",
    "permittivity", "penetration_loss_db", "pathloss_min_db", "pathloss_max_db",
    "tx_local_building_ratio_50m", "tx_local_building_ratio_100m",
]
CAT_KEYS = ["scene_type", "dominant_material", "confidence"]
ALL_KEYS = CAT_KEYS + NUM_KEYS

# 数值字段必须再分两类（第四条实测教训）。把"物理上该恒定"当成"模型偷懒"是错的：
#
#   几何条件字段 —— 随 Tx 位置变化，**必须**逐样本不同；恒定就是坍缩。
#   材料属性字段 —— 在"全是混凝土城市"这类单一材质数据集上，0.9 GHz 下混凝土的
#                   介电常数与穿透损耗本来就该是固定值，near-constant 是**正确答案**。
#
# 实测：把精确几何当事实喂给 glm-4v-flash 后，四个机制权重唯一值 5-7（范围很宽），
# 而 penetration_loss_db 只有 2 个取值（10/15 dB）—— 后者不是缺陷，是物理常识。
# 所以只有几何条件字段参加硬门槛，材料属性字段作为建议级并附说明。
GEOMETRY_NUM_KEYS = [
    "los_ratio", "reflection_ratio", "diffraction_ratio", "scattering_ratio",
    "pathloss_min_db", "pathloss_max_db",
    "tx_local_building_ratio_50m", "tx_local_building_ratio_100m",
]
MATERIAL_NUM_KEYS = ["permittivity", "penetration_loss_db"]

# ===== 准入门槛 =====
#
# 阈值怎么定出来的（可复现，不要凭感觉改）
# ------------------------------------
# 这套阈值是在**两批数据**上标定的：
#   (a) 上一版真实的 4000 条先验（已知无效）—— 硬门槛必须拦住它；
#   (b) `tests/test_gate.py` 构造的一份"健康"先验集（sectors 取自真实几何、
#       数值逐 Tx 变化）—— 硬门槛必须放它过。
#
# 一个已经踩过的坑：初版把"任一单维 top1_share < 0.40"设为**硬**门槛。
# 结果健康集被判 FAIL，原因是 `scattering_ratio` 真的恒定（这一条判对了），
# 但更要紧的是这个指标本身有问题：它把"LLM 偷懒"和"数据分布本来就不均衡"混为一谈。
# 例如 50 张城市地图真实就是 60% urban / 40% dense_urban，`scene_type` 的
# top1 = 0.60 > 0.40，一份完全正确的 prompt 会被永远挡在训练门外。
# 所以改成按字段类型分工：
#   * 数值字段 —— 硬要求"真的在变"（唯一值数 + 熵）；恒定值就是坍缩，没有歧义；
#   * 类别字段 —— 硬要求"不能是单一常量"；不均衡是数据的性质，不是错误；
#   * top1_share —— 降级为**建议级**，只提示、不阻断。
GATES_FAIL = {
    "unique_ratio": 0.50,               # 唯一字段向量 / 样本数
    "intra_map_unique_ratio_mean": 0.50,  # 同一地图内不同 Tx 的唯一向量占比
    "numeric_unique_min": 5,            # 每个数值字段的最少唯一值数
    "numeric_entropy_min": 1.0,         # 每个数值字段的熵下限（bits）
    "categorical_min_unique": 2,        # 每个类别字段的最少取值数（>=2 即非常量）
    "sector_consistency_strict": 0.70,  # 与**本 Tx** 几何的绝对一致率
    # 自校准判据：与同图另一 Tx 几何的一致率必须明显更低。
    # 偷懒策略（恒答某标签 / 只记方位先验）两侧得分相同 → gain ≈ 0。
    # 实测 maps 0-3：'blocked' 占 59.3%、'open' 仅 0.7%，"永远回答 blocked"
    # 的绝对一致率就有 0.593，离 0.70 只差 0.107 —— 换个更密的地图集就会越线。
    # gain 不依赖几何代理的标定精度，所以把它也设为硬门槛。
    "sector_consistency_gain": 0.10,
}

# scene_type 地图级判据里"分档处是否存在天然空隙"的门槛（见 scene_check 注释）。
# 定义：把各图的整图建筑占比排序后，**最大相邻间隔 ÷ 相邻间隔中位数**。
#   ≥ 此值 → 分布在该处有真实断层，"跨类"是数据本来的结构，LLM 必须跟上；
#   <  此值 → 分档落在一段平滑分布中间，是阈值劈出来的类，不据此否决 LLM。
# 实测 prompt_v3 全量 4000 条 / 50 张地图：比例 ≈ 5.5（0.021 / 0.0038）→ 判为无天然空隙。
SCENE_GAP_RATIO_MIN = 8.0
GATES_WARN = {
    "max_top1_share": 0.40,             # 建议级：可能只是数据不均衡
    "confidence_top1_share": 0.90,      # 自报置信度几乎恒定 = 上一版的病症之一
    "fields_varying_within_map": 3,
    "scene_type_classes": 3,            # 若数据确实存在多类
    "sector_profile_unique_ratio": 0.50,
    "reasoning_repeat_share": 0.10,        # 单个 reasoning 文本的最大占比
    "content_contamination_share": 0.05,   # 含 28GHz / 高度类矛盾描述的比例
    "mmwave_hallucination_share": 0.02,    # 0.9 GHz 数据集里谈 28GHz/毫米波
    "height_hallucination_share": 0.05,    # 二值掩码里"看出"建筑高度
    "field_consistency_mean": 0.80,        # 多轮采样一致性（来自 provenance）
}

# 与数据集事实矛盾的表述，分两类（数据集是 0.9 GHz、2D、掩码无高度信息）
CONTAMINATION_GROUPS: dict[str, list[str]] = {
    # 频段幻觉：上一版有 475 条整段复读 "28GHz…"，但即使不写 28GHz 也不能谈毫米波
    "mmwave": [r"28\s*ghz", r"毫米波", r"mmwave", r"millimeter", r"mm\s*wave"],
    # 高度幻觉：建筑掩码是严格二值 {0,255}，不含任何高度信息，
    # 实测 82% 的先验却写出了"建筑高度 10-30m"这类具体数字
    "height": [r"建筑高度", r"楼高", r"楼层", r"building height", r"taller",
               r"天线高度", r"tx height", r"发射机高度"],
}
CONTAMINATION_PATTERNS = [p for pats in CONTAMINATION_GROUPS.values() for p in pats]


def load_priors(prior_dir: Path):
    """读取先验目录，兼容 v2 外层结构与旧扁平结构。"""
    files = sorted(prior_dir.glob("*.json"))
    records, bad = [], []
    for f in files:
        if f.name.startswith("_"):
            continue
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            bad.append({"file": f.name, "error": repr(exc)})
            continue
        if isinstance(raw, dict) and "prior" in raw and "sample_id" in raw:
            d, prov = raw["prior"], raw.get("provenance", {})
        else:
            d, prov = raw, {}
        parts = f.stem.split("_")
        records.append({
            "file": f,
            "map_id": parts[0] if parts else "?",
            "tx_id": parts[1] if len(parts) > 1 else "?",
            "data": d,
            "provenance": prov,
            "format": "v2" if prov else "flat",
        })
    return records, bad


def _sector_profile(d: dict) -> str:
    """把 8 方位遮挡压成一个字符串指纹，用于统计"方位剖面"的唯一性。"""
    secs = d.get("sectors")
    if not isinstance(secs, list) or len(secs) != 8:
        return "MISSING"
    by_az = {}
    for s in secs:
        if isinstance(s, dict) and "azimuth_deg" in s:
            by_az[int(s["azimuth_deg"])] = str(s.get("obstruction", "?"))
    return "|".join(by_az.get(a, "?") for a in AZIMUTHS)


def diversity_report(records) -> dict:
    per_key = defaultdict(list)
    per_map = defaultdict(list)
    per_map_key = defaultdict(lambda: defaultdict(list))
    missing: Counter = Counter()
    sector_profiles = []
    reasonings = []

    for r in records:
        d = r["data"]
        vec = tuple(str(d.get(k)) for k in ALL_KEYS) + (_sector_profile(d),)
        per_map[r["map_id"]].append(vec)
        for k in ALL_KEYS:
            if d.get(k) is None:
                missing[k] += 1
            else:
                per_key[k].append(str(d.get(k)))
                per_map_key[r["map_id"]][k].append(str(d.get(k)))
        sector_profiles.append(_sector_profile(d))
        rz = d.get("reasoning")
        if isinstance(rz, str) and rz.strip():
            reasonings.append(rz.strip())

    all_vecs = {v for vs in per_map.values() for v in vs}
    rep: dict = {
        "n_files": len(records),
        "n_unique_vector": len(all_vecs),
        "unique_ratio": len(all_vecs) / len(records) if records else 0.0,
        "per_key": {},
        "missing_keys": dict(missing),
    }

    for k, v in per_key.items():
        cnt = Counter(v)
        p = np.array(list(cnt.values()), dtype=float) / len(v)
        rep["per_key"][k] = {
            "unique": len(cnt),
            "top1_value": cnt.most_common(1)[0][0],
            "top1_share": float(p.max()),
            "entropy_bits": float(-(p * np.log2(p)).sum()),
        }

    intra = [len(set(vs)) / len(vs) for vs in per_map.values() if len(vs) > 1]
    rep["intra_map_unique_ratio_mean"] = float(np.mean(intra)) if intra else 0.0
    rep["n_maps"] = len(per_map)
    rep["samples_per_map_mean"] = (
        float(np.mean([len(v) for v in per_map.values()])) if per_map else 0.0
    )

    # ---- 数值字段 / 类别字段的分型诊断 ----
    # 数值字段与类别字段的"健康"标准不同，所以分开统计：
    #   几何条件数值字段：唯一值少 或 熵低 → 该维度实际是常量（坍缩）
    #   材料属性数值字段：恒定是物理常识   → 只报告，不判定
    #   类别字段：唯一值数 == 1          → 单一常量（坍缩）；取值不均衡不算错误
    geo_present = [k for k in GEOMETRY_NUM_KEYS if k in rep["per_key"]]
    mat_present = [k for k in MATERIAL_NUM_KEYS if k in rep["per_key"]]
    cat_present = [k for k in CAT_KEYS if k in rep["per_key"]]
    rep["numeric"] = {
        "keys": geo_present,
        "unique_min": min((rep["per_key"][k]["unique"] for k in geo_present), default=None),
        "entropy_min": min((rep["per_key"][k]["entropy_bits"] for k in geo_present), default=None),
        "failing_unique": [k for k in geo_present
                           if rep["per_key"][k]["unique"] < GATES_FAIL["numeric_unique_min"]],
        "failing_entropy": [k for k in geo_present
                            if rep["per_key"][k]["entropy_bits"] < GATES_FAIL["numeric_entropy_min"]],
    }
    rep["material_numeric"] = {
        "keys": mat_present,
        "unique_min": min((rep["per_key"][k]["unique"] for k in mat_present), default=None),
        "note": "材料属性（介电常数 / 穿透损耗）在单一材质数据集上本就应近乎恒定，"
                "因此不参加硬门槛；报告出来只为让读者判断物理是否合理。",
        "values": {k: {"unique": rep["per_key"][k]["unique"],
                       "top1_value": rep["per_key"][k]["top1_value"],
                       "top1_share": rep["per_key"][k]["top1_share"]} for k in mat_present},
    }
    rep["categorical"] = {
        "keys": cat_present,
        "unique_min": min((rep["per_key"][k]["unique"] for k in cat_present), default=None),
        "constant_keys": [k for k in cat_present if rep["per_key"][k]["unique"] < 2],
    }

    # ---- 每个字段"在同一张地图内是否随 Tx 变化" ----
    varying = {}
    for k in ALL_KEYS:
        hit = total = 0
        for _m, kv in per_map_key.items():
            vals = kv.get(k, [])
            if len(vals) < 2:
                continue
            total += 1
            if len(set(vals)) > 1:
                hit += 1
        varying[k] = {"maps_with_variation": hit, "maps_considered": total,
                      "share": (hit / total) if total else 0.0}
    rep["within_map_variation"] = varying
    rep["fields_varying_within_map"] = sum(
        1 for v in varying.values() if v["share"] >= 0.5
    )

    # ---- sectors 专项 ----
    sc = Counter(s for s in sector_profiles if s != "MISSING")
    rep["sectors"] = {
        "n_with_sectors": sum(1 for s in sector_profiles if s != "MISSING"),
        "unique_profiles": len(sc),
        "sector_profile_unique_ratio": (len(sc) / len(records)) if records else 0.0,
        "top1_profile_share": (sc.most_common(1)[0][1] / len(records)) if (sc and records) else 0.0,
        "top1_profile": sc.most_common(1)[0][0] if sc else None,
    }
    if sc:
        per_az: dict[int, Counter] = defaultdict(Counter)
        for r in records:
            secs = r["data"].get("sectors")
            if isinstance(secs, list):
                for s in secs:
                    if isinstance(s, dict) and "azimuth_deg" in s:
                        per_az[int(s["azimuth_deg"])][str(s.get("obstruction"))] += 1
        rep["sectors"]["per_azimuth"] = {str(a): dict(per_az[a]) for a in AZIMUTHS if a in per_az}

    # ---- reasoning 复读与内容污点 ----
    if reasonings:
        rc = Counter(reasonings)
        top_text, top_n = rc.most_common(1)[0]
        rep["reasoning"] = {
            "n": len(reasonings),
            "unique": len(rc),
            "unique_ratio": len(rc) / len(reasonings),
            "top1_share": top_n / len(reasonings),
            "top1_head": top_text[:200],
            "mean_len": float(np.mean([len(x) for x in reasonings])),
        }
    pats = [re.compile(p, re.IGNORECASE) for p in CONTAMINATION_PATTERNS]
    hits = [rz for rz in reasonings if any(p.search(rz) for p in pats)]
    rep["content_contamination"] = {
        "n": len(reasonings),
        "n_hit": len(hits),
        "share": (len(hits) / len(reasonings)) if reasonings else 0.0,
        "examples": [h[:160] for h in hits[:3]],
        "by_group": {},
    }
    for gname, gpats in CONTAMINATION_GROUPS.items():
        gcomp = [re.compile(p, re.IGNORECASE) for p in gpats]
        ghits = [rz for rz in reasonings if any(p.search(rz) for p in gcomp)]
        rep["content_contamination"]["by_group"][gname] = {
            "n_hit": len(ghits),
            "share": (len(ghits) / len(reasonings)) if reasonings else 0.0,
            "patterns": gpats,
            "example": ghits[0][:160] if ghits else None,
        }

    # ---- 多轮采样一致性（只有 v2 记录才有）----
    cons = [r["provenance"].get("field_consistency", {}).get("overall")
            for r in records if r["provenance"]]
    cons = [c for c in cons if isinstance(c, (int, float))]
    rep["field_consistency_mean"] = float(np.mean(cons)) if cons else None
    rep["n_with_provenance"] = sum(1 for r in records if r["provenance"])
    return rep


# ---------- scene_type 的地图级判据（可单独自检）----------


def scene_split_has_natural_gap(fracs: list[float]) -> tuple[bool, float, float, float]:
    """各图整图建筑占比的分布里，是否存在"天然类间空隙"。

    返回 (是否有天然空隙, 最大相邻间隔, 相邻间隔中位数, 比值)。

    为什么需要这个量：程序侧把连续的建筑占比按固定阈值切成 dense_urban/urban/…，
    "程序侧跨类"本身**不能**证明数据真的有两类场景 —— 阈值切在平滑分布中间时，
    切出来的"类"只是人为的。实测 prompt_v3 全量 4000 条 / 50 张地图：
    占比从 0.242 近乎均匀地铺到 0.435，最大相邻间隔 0.021、典型间隔 0.0038，
    比值 ≈5.5 —— 远低于 SCENE_GAP_RATIO_MIN，所以那 26/24 的"两类"是阈值产物。
    这个量只依赖几何（掩码），与 LLM 输出无关，因此不会被"LLM 恰好答对/答错"污染。
    """
    fs = sorted(float(x) for x in fracs)
    if len(fs) < 4:
        return False, float("nan"), float("nan"), float("nan")
    gaps = [b - a for a, b in zip(fs, fs[1:])]
    med = float(np.median(gaps))
    mx = float(max(gaps))
    ratio = (mx / med) if med > 0 else float("inf")
    return bool(np.isfinite(ratio) and ratio >= SCENE_GAP_RATIO_MIN), mx, med, ratio


def scene_follows_geometry(prog_classes: list[str], llm_classes: list[str],
                           has_natural_gap: bool, gap_ratio: float) -> tuple[bool, str]:
    """scene_type 地图级判据的**三分支**规则 → (是否通过, 理由)。

    这条判据修过两次，两次都是同一个病：把"数据本身的性质"当成"LLM 偷懒"。

      1. 程序侧本身只有一类 → LLM 恒定合法（这一批城市都一样密）。
      2. 程序侧跨类，但分档处**没有天然空隙** → 那些"类"是阈值劈出来的，
         不据此否决 LLM。否则一份正确的 prompt 会被永远挡在训练门外 ——
         这与 AGENTS.md 记的第二条教训（`max_top1_share` 把"LLM 偷懒"与
         "数据本来不均衡"混为一谈）是同一族缺陷。
      3. 程序侧跨类、且分档落在真实断层上 → LLM 必须给出 ≥2 类，否则 FAIL。
    """
    if len(prog_classes) <= 1:
        return True, "程序侧本身只有一类，LLM 恒定属合法"
    if not has_natural_gap:
        return True, (f"程序侧虽跨 {len(prog_classes)} 类，但整图建筑占比是**平滑连续分布**"
                      f"（最大相邻间隔仅为典型间隔的 {gap_ratio:.1f} 倍，"
                      f"门槛 {SCENE_GAP_RATIO_MIN:g}）→ 这个分档是阈值劈出来的，"
                      f"不能据此判 LLM 没在区分场景")
    return (len(llm_classes) >= 2), (
        f"程序侧跨 {len(prog_classes)} 类且分档处存在天然空隙"
        f"（最大间隔是典型间隔的 {gap_ratio:.1f} 倍）"
        f"→ LLM 必须给出 ≥2 类，实际 {len(llm_classes)} 类")


def _scene_type_check(per_map_idx, masks, scene_by) -> dict:
    """地图级 scene_type 比对：程序侧用**整图**建筑占比分档，LLM 侧取每张图的众数。

    为什么只在**地图级**比：`scene_type` 是地图级/城市级属性。初版拿它比"每个 Tx 局部
    100 m 的密度分档"，于是只要同一张图里有的 Tx 周围空旷、有的密集，就会判
    LLM"没在区分场景"—— 而实测 maps 0-2 在地图级全是 urban 档（整图占比
    0.253 / 0.332 / 0.291），LLM 恒定答 urban 本来就是对的。
    """
    prog_map_class: dict[str, str] = {}
    prog_map_frac: dict[str, float] = {}
    for mid in per_map_idx:
        mask = masks.get(mid)
        if mask is None:
            continue
        frac = float((mask > 127).mean())          # 整图建筑占比
        prog_map_frac[mid] = frac
        prog_map_class[mid] = (
            "dense_urban" if frac > 0.35 else
            "urban" if frac > 0.20 else
            "suburban" if frac > 0.10 else "rural")
    llm_map_class: dict[str, str] = {}
    for mid, idxs in per_map_idx.items():
        vals = [scene_by[(mid, t)] for t in idxs if (mid, t) in scene_by]
        if vals:
            llm_map_class[mid] = Counter(vals).most_common(1)[0][0]
    common_maps = sorted(set(prog_map_class) & set(llm_map_class))
    map_pairs = [{"map": m, "program": prog_map_class[m], "llm": llm_map_class[m],
                  "agree": prog_map_class[m] == llm_map_class[m]} for m in common_maps]
    prog_classes = sorted({v for v in prog_map_class.values()})
    llm_classes = sorted({v for v in llm_map_class.values()})
    map_agree = [p["agree"] for p in map_pairs]

    has_gap, max_gap, med_gap, gap_ratio = scene_split_has_natural_gap(
        list(prog_map_frac.values()))
    follows, why = scene_follows_geometry(prog_classes, llm_classes, has_gap, gap_ratio)
    return {
        "level": "map",
        "n_maps": len(common_maps),
        "program_classes": prog_classes,
        "llm_classes": llm_classes,
        "agreement_rate": (sum(map_agree) / len(map_agree)) if map_agree else 0.0,
        "per_map": map_pairs,
        "follows_geometry": bool(follows),
        "why": why,
        "natural_gap_max": max_gap,
        "natural_gap_median": med_gap,
        "natural_gap_ratio": gap_ratio,
        "has_natural_gap": bool(has_gap),
        "note": "程序侧按**整图**建筑占比分档：>0.35 dense_urban / >0.20 urban / "
                ">0.10 suburban / 其余 rural。scene_type 是地图级属性，"
                "所以只在**地图级**比对，不逐 Tx 比局部密度。"
                "跨类但分档处没有天然空隙时，不拿它否决 LLM（阈值劈出来的类）。",
    }


# ---------- sectors 与程序几何的一致率 ----------


def sector_consistency(records, data_dir: Path, px_per_m: float,
                       radius_m: float = 100.0) -> dict:
    """sectors 与几何的一致率，以及**同一地图内错位对照**的增益。

    为什么要加错位对照（真实数据逼出来的）
    ------------------------------------
    只用"与几何的绝对一致率 >= 0.70"有一个脆弱之处：这个绝对值的偷懒下限
    取决于几何真值的标签分布。实测 maps 0-3 的 2560 个扇区判定里
    `blocked` 占 59.3%、`open` 只占 0.7%，于是"永远回答 blocked"就能拿 0.593 ——
    离门槛 0.70 只差 0.107。换个建筑更密的地图集，偷懒基线就会越过门槛。

    所以这里再算一个**自校准**的量：
        gain = 与**本 Tx** 几何的一致率 − 与**同一地图另一个 Tx** 几何的一致率
    偷懒策略（恒答某标签、或只按方位记先验）在两侧的得分必然相同 → gain ≈ 0；
    只有真的逐 Tx 读图，本 Tx 那一侧才会明显更高。这个判据不依赖几何代理的
    标定精度，也不需要我预设"答案该长什么样"。
    """
    from PIL import Image

    masks: dict[str, np.ndarray] = {}
    antennas: dict[str, list] = {}
    geo_cache: dict[tuple[str, int], dict] = {}
    geo_missing = 0

    # ---- 先把每个 (map, tx) 的几何真值算出来并缓存 ----
    per_map_idx: dict[str, list[int]] = defaultdict(list)
    labels_by: dict[tuple[str, int], dict[int, str]] = {}
    scene_by: dict[tuple[str, int], str] = {}
    prog_scene: dict[tuple[str, int], str] = {}
    for r in records:
        secs = r["data"].get("sectors")
        if not isinstance(secs, list) or len(secs) != 8:
            continue
        mid = r["map_id"]
        try:
            tidx = int(r["tx_id"])
        except (TypeError, ValueError):
            continue
        if mid not in masks:
            mp = data_dir / "buildings_complete" / f"{mid}.png"
            if not mp.exists():
                continue
            masks[mid] = np.array(Image.open(mp).convert("L"))
            aj = data_dir / "antenna" / f"{mid}.json"
            antennas[mid] = json.loads(aj.read_text(encoding="utf-8")) if aj.exists() else []
        tx_list = antennas.get(mid, [])
        if not (0 <= tidx < len(tx_list)):
            continue
        by_az = {int(s["azimuth_deg"]): str(s.get("obstruction")) for s in secs
                 if isinstance(s, dict) and "azimuth_deg" in s}
        if len(by_az) != 8:
            continue
        labels_by[(mid, tidx)] = by_az
        scene_by[(mid, tidx)] = str(r["data"].get("scene_type", "?"))
        per_map_idx[mid].append(tidx)
        if (mid, tidx) not in geo_cache:
            geo_cache[(mid, tidx)] = geometric_obstruction(
                masks[mid], tx_list[tidx], radius_m=radius_m, px_per_m=px_per_m)
            # 程序侧场景分类（按 100 m 建筑占比分档，与 prompt 给 LLM 的口径一致）
            frac = local_building_ratio(masks[mid], tx_list[tidx], 100.0, px_per_m)
            prog_scene[(mid, tidx)] = (
                "dense_urban" if frac > 0.35 else
                "urban" if frac > 0.22 else
                "suburban" if frac > 0.12 else "rural")

    # ---- 同一地图内的错位映射（无不动点）----
    shifted: dict[tuple[str, int], int] = {}
    for mid, idxs in per_map_idx.items():
        uniq = sorted(set(idxs))
        if len(uniq) < 2:
            continue
        order = np.random.RandomState(20260101).permutation(len(uniq))
        der = np.empty(len(uniq), dtype=np.int64)
        der[order] = np.roll(order, 1)
        for i, t in enumerate(uniq):
            shifted[(mid, t)] = uniq[int(der[i])]

    levels = {"open": 0, "sparse": 1, "dense": 2, "blocked": 3}
    own_strict = own_adj = sh_strict = 0
    total = n_shift = 0
    per_az = {a: [0, 0] for a in AZIMUTHS}
    confusion: Counter = Counter()

    # 分部统计：把"Tx 落在建筑像素内"与"Tx 在街上"分开算。
    # 前者（实测占 30.3%）的几何真值往往是 8 个方位全 blocked，**本来就没有逐方位
    # 可区分的方向信息**；如果模型在这类样本上失分，说明问题是"没认出 Tx 压在建筑上"，
    # 而不是"没在逐方位读图"。混在一起统计会把两类问题误判成同一种。
    def _blank() -> dict:
        return {"n": 0, "own": 0, "adj": 0, "shift": 0, "n_shift": 0}

    subsets: dict[str, dict] = {"tx_on_building": _blank(), "tx_on_street": _blank()}

    # Tx 是否落在建筑像素内
    tx_on_building: dict[tuple[str, int], bool] = {}
    for (mid, tidx) in labels_by:
        tx_list = antennas.get(mid, [])
        mask = masks.get(mid)
        if mask is None or not (0 <= tidx < len(tx_list)):
            continue
        tx = tx_list[tidx]
        ty, txp = int(round(tx[1])), int(round(tx[0]))
        tx_on_building[(mid, tidx)] = bool(
            0 <= ty < mask.shape[0] and 0 <= txp < mask.shape[1] and mask[ty, txp] > 127)

    for (mid, tidx), got in labels_by.items():
        exp_geo = geo_cache.get((mid, tidx))
        if exp_geo is None:
            geo_missing += 1
            continue
        key = "tx_on_building" if tx_on_building.get((mid, tidx)) else "tx_on_street"
        sub = subsets[key]
        for az in AZIMUTHS:
            exp = str(exp_geo[az]["obstruction"])
            g = got[az]
            total += 1
            per_az[az][1] += 1
            confusion[(exp, g)] += 1
            sub["n"] += 1
            if exp == g:
                own_strict += 1
                sub["own"] += 1
                per_az[az][0] += 1
            if abs(levels.get(exp, 0) - levels.get(g, 0)) <= 1:
                own_adj += 1
                sub["adj"] += 1
        # 错位对照：拿同一地图另一个 Tx 的几何来比
        donor = shifted.get((mid, tidx))
        if donor is not None:
            exp_geo_s = geo_cache.get((mid, donor))
            if exp_geo_s is not None:
                for az in AZIMUTHS:
                    n_shift += 1
                    sub["n_shift"] += 1
                    if str(exp_geo_s[az]["obstruction"]) == got[az]:
                        sh_strict += 1
                        sub["shift"] += 1

    own = (own_strict / total) if total else 0.0
    sh = (sh_strict / n_shift) if n_shift else 0.0

    # ---- 场景类型：在**地图级**判定，而不是逐 Tx ----
    # 判据与三次修正的理由都写在 `scene_follows_geometry()` 的 docstring 里
    # （那是这条判据唯一的一处实现，便于 `tests/test_typology.py` 直接自检）。
    scene_check = _scene_type_check(per_map_idx, masks, scene_by)
    # 顺带保留逐 Tx 局部密度分档的对照（仅参考，不参与判定）
    local_classes = sorted({v for v in prog_scene.values()})
    scene_check["local_per_tx_classes_for_reference"] = local_classes
    subset_out = {}
    for k, v in subsets.items():
        if not v["n"]:
            continue
        s_own = v["own"] / v["n"]
        s_sh = (v["shift"] / v["n_shift"]) if v["n_shift"] else 0.0
        subset_out[k] = {
            "n_sectors": v["n"],
            "agreement_strict": s_own,
            "agreement_adjacent": v["adj"] / v["n"],
            "agreement_strict_shifted": s_sh,
            "agreement_gain": s_own - s_sh,
        }
    return {
        "n_checked": total,
        "agreement_strict": own,
        "agreement_adjacent": (own_adj / total) if total else 0.0,
        "n_shifted_checked": n_shift,
        "agreement_strict_shifted": sh,
        "agreement_gain": own - sh,
        "subsets": subset_out,
        "scene_type_check": scene_check,
        "per_azimuth_strict": {str(a): (v[0] / v[1] if v[1] else None) for a, v in per_az.items()},
        "confusion_expected_vs_llm": {f"{k[0]}->{k[1]}": v for k, v in
                                      sorted(confusion.items(), key=lambda kv: -kv[1])[:12]},
        "note": "粗检：按扇区内最近建筑距离与建筑像素占比判定，不是严格几何遮挡；"
                "agreement_adjacent 允许相邻等级（±1 档）。"
                "agreement_gain = 本 Tx 一致率 − 同图另一 Tx 一致率，是自校准判据："
                "偷懒（恒答某标签 / 只记方位先验）在两侧得分相同，gain ≈ 0。"
                "subsets 把「Tx 落在建筑像素内」与「Tx 在街上」分开报告，"
                "因为前者的真值常是 8 方位全 blocked，本就没有可区分的方向信息。",
    }


def evaluate_gates(rep: dict) -> dict:
    results: list[dict] = []

    def add(name, value, threshold, op, severity, note=None):
        if value is None:
            results.append({"gate": name, "value": None, "threshold": threshold,
                            "op": op, "severity": severity, "pass": None,
                            "note": note or "数据不足，未判定"})
            return
        passed = (value >= threshold) if op == ">=" else (value < threshold)
        rec = {"gate": name, "value": round(float(value), 4),
               "threshold": threshold, "op": op, "severity": severity,
               "pass": bool(passed)}
        if note:
            rec["note"] = note
        results.append(rec)

    # ---------- 硬门槛：向量整体有信息量 ----------
    add("unique_ratio", rep["unique_ratio"], GATES_FAIL["unique_ratio"], ">=", "fail",
        "唯一字段向量 / 样本数；上一版是 0.0138（55 / 4000）")
    add("intra_map_unique_ratio_mean", rep["intra_map_unique_ratio_mean"],
        GATES_FAIL["intra_map_unique_ratio_mean"], ">=", "fail",
        "同一张地图内不同 Tx 的唯一向量占比；上一版是 0.104")

    # ---------- 硬门槛：几何条件数值字段真的在变 ----------
    num = rep.get("numeric", {})
    add("numeric_unique_min", num.get("unique_min"), GATES_FAIL["numeric_unique_min"],
        ">=", "fail",
        ("恒定或量化的几何条件字段：" + ", ".join(num.get("failing_unique", [])))
        if num.get("failing_unique")
        else "所有几何条件数值字段都有足够多的取值"
             "（材料属性字段 permittivity / penetration_loss_db 不参加此项，见 GATES_FAIL 注释）")
    add("numeric_entropy_min", num.get("entropy_min"), GATES_FAIL["numeric_entropy_min"],
        ">=", "fail",
        ("熵过低的几何条件字段：" + ", ".join(num.get("failing_entropy", [])))
        if num.get("failing_entropy") else "所有几何条件数值字段的熵都达标")

    # ---------- 硬门槛：场景类型在**地图级**跟得上几何 ----------
    # 这条判据修过两次，两次都是同一个病：把"数据本身的性质"与"LLM 偷懒"混为一谈。
    #   第一次：拿 scene_type 去比"每个 Tx 局部 100 m 的密度分档" → 同一城图内疏密不同，
    #           于是正常 prompt 也会被判 FAIL。改成只在**地图级**比。
    #   第二次：只要程序侧跨类就要求 LLM 也跨类。实测 50 张地图的整图建筑占比是
    #           0.242→0.435 的平滑连续分布（最大相邻间隔只有典型间隔的 ~5.5 倍），
    #           0.35 这个阈值落在这段分布中间，26/24 的"两类"是阈值劈出来的，
    #           LLM 恒定答 urban 是正确的。→ 增加"分档处有无天然空隙"这个客观判据。
    if "sector_check" in rep and rep["sector_check"].get("scene_type_check"):
        sc = rep["sector_check"]["scene_type_check"]
        add("scene_type_follows_geometry", 1.0 if sc["follows_geometry"] else 0.0,
            1.0, ">=", "fail",
            f"地图级：程序侧分档 {sc['program_classes']}"
            f"（{len(sc['program_classes'])} 类），LLM 侧 {sc['llm_classes']}"
            f"（{len(sc['llm_classes'])} 类），逐地图一致率 {sc['agreement_rate']:.3f}"
            f"（{sc['n_maps']} 张地图）。判定：{sc.get('why', '')}")
        # 分档处没有天然空隙时要如实报告：这不是错误，但必须让读者知道
        # "program_classes 是阈值产物"，否则会把 26/24 当成真实的两类场景。
        if not sc.get("has_natural_gap", True) and len(sc["program_classes"]) > 1:
            add("scene_type_program_split_is_threshold_artifact", 1.0, 1.0, ">=", "warn",
                f"仅参考：程序侧把整图建筑占比 0.242→0.435 的**平滑分布**在 0.35 处劈成 "
                f"{len(sc['program_classes'])} 类（{sc['program_classes']}），"
                f"最大相邻间隔仅为典型间隔的 {sc.get('natural_gap_ratio', float('nan')):.1f} 倍"
                f"（< {SCENE_GAP_RATIO_MIN:g}）→ 这是**阈值产物**，"
                f"报告里不要把它当作真实的场景类别")
        # 逐 Tx 局部密度分档仅作参考：它跨多类是正常的（同一城图内疏密不同）
        ref = sc.get("local_per_tx_classes_for_reference") or []
        if len(ref) > 1:
            add("scene_type_vs_local_density_for_reference", 1.0, 1.0, ">=", "warn",
                f"仅参考：逐 Tx 局部 100 m 密度分档跨 {len(ref)} 类（{ref}），"
                f"这不要求 scene_type 跟着变 —— 它是地图级属性")

    # ---------- 硬门槛：类别字段不能是单一常量 ----------
    # 只把 `scene_type` 排除在外 —— 它是**地图级**属性，已由上面的地图级判据接手
    # （本批地图只有一类时恒定是合法的）。其余类别字段仍须非常量：
    # `confidence` 在 v3 里由采样一致性推导，恒定就意味着一致性没有变化（可疑）；
    # `dominant_material` 是材质属性，单一材质城市里也应当至少能与几何产生区分。
    cat = rep.get("categorical", {})
    hard_constant = [k for k in cat.get("constant_keys", []) if k != "scene_type"]
    if not hard_constant and cat.get("constant_keys"):
        add("categorical_min_unique", float(cat.get("unique_min") or 1), 2.0, ">=", "warn",
            f"只有 scene_type 取值为单一常量（{cat['constant_keys']}），"
            f"已由地图级判据 scene_type_follows_geometry 接手")
    else:
        add("categorical_min_unique", float(cat.get("unique_min") or 1), 2.0, ">=", "fail",
            ("退化成单一常量的类别字段：" + ", ".join(hard_constant))
            if hard_constant else "没有任何类别字段退化成常量")

    # ---------- 硬门槛：sectors 与几何一致 ----------
    if "sector_check" in rep and rep["sector_check"]["n_checked"]:
        add("sector_consistency_strict", rep["sector_check"]["agreement_strict"],
            GATES_FAIL["sector_consistency_strict"], ">=", "fail",
            "4 类随机猜测的期望严格一致率约 0.25；但注意偷懒基线取决于真值分布，"
            "实测 maps 0-3 上「永远回答 blocked」就能拿 0.593")
        if rep["sector_check"].get("n_shifted_checked"):
            add("sector_consistency_gain", rep["sector_check"]["agreement_gain"],
                GATES_FAIL["sector_consistency_gain"], ">=", "fail",
                f"自校准判据：本 Tx 一致率 {rep['sector_check']['agreement_strict']:.3f} − "
                f"同图另一 Tx 一致率 "
                f"{rep['sector_check']['agreement_strict_shifted']:.3f}；"
                f"偷懒策略在此项上必然接近 0")

    # ---------- 建议级 ----------
    if rep["per_key"]:
        worst = max(rep["per_key"].items(), key=lambda kv: kv[1]["top1_share"])
        add(f"max_top1_share[{worst[0]}]", worst[1]["top1_share"],
            GATES_WARN["max_top1_share"], "<", "warn",
            "仅提示：类别字段的取值不均衡可能只是数据本身的性质，不阻断训练")
        cf = rep["per_key"].get("confidence")
        if cf is not None:
            add("confidence_top1_share", cf["top1_share"],
                GATES_WARN["confidence_top1_share"], "<", "warn",
                "自报置信度几乎恒定是上一版的病症之一（97.9% 都是 medium）")
    add("fields_varying_within_map", rep["fields_varying_within_map"],
        GATES_WARN["fields_varying_within_map"], ">=", "warn",
        "注意：上一版此指标是 9/13（通过）却依然完全坍缩 —— 它不足以判定健康，仅供参考")
    add("scene_type_classes", rep["per_key"].get("scene_type", {}).get("unique"),
        GATES_WARN["scene_type_classes"], ">=", "warn",
        "若数据集本身只有两类，这一条不达标属正常")
    add("sector_profile_unique_ratio", rep["sectors"]["sector_profile_unique_ratio"],
        GATES_WARN["sector_profile_unique_ratio"], ">=", "warn")
    if "reasoning" in rep:
        add("reasoning_repeat_share", rep["reasoning"]["top1_share"],
            GATES_WARN["reasoning_repeat_share"], "<", "warn")
    add("content_contamination_share", rep["content_contamination"]["share"],
        GATES_WARN["content_contamination_share"], "<", "warn")
    for gname, key in (("mmwave", "mmwave_hallucination_share"),
                       ("height", "height_hallucination_share")):
        g = rep["content_contamination"]["by_group"].get(gname)
        add(key, g["share"] if g else None, GATES_WARN[key], "<", "warn")
    add("field_consistency_mean", rep.get("field_consistency_mean"),
        GATES_WARN["field_consistency_mean"], ">=", "warn")

    hard_fail = [r for r in results if r["severity"] == "fail" and r["pass"] is False]
    soft_fail = [r for r in results if r["severity"] == "warn" and r["pass"] is False]
    return {"pass": len(hard_fail) == 0, "hard_failures": len(hard_fail),
            "warnings": len(soft_fail), "results": results}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="先验多样性体检 + 准入门槛")
    ap.add_argument("--priors", required=True)
    ap.add_argument("--out", default="reports/audit.json")
    ap.add_argument("--data", default=None)
    ap.add_argument("--px-per-m", type=float, default=1.0)
    ap.add_argument("--px-per-100m", type=float, default=None,
                    help="兼容 starter 版参数：100 米对应多少像素")
    ap.add_argument("--local-radius-m", type=float, default=100.0)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--weak-ok", action="store_true",
                    help="只做统计；即使硬门槛不过也返回 0（仅用于探索）")
    args = ap.parse_args(argv)

    px_per_m = args.px_per_100m / 100.0 if args.px_per_100m else args.px_per_m

    prior_dir = Path(args.priors)
    if not prior_dir.is_dir():
        print(f"目录不存在: {prior_dir}")
        return 2

    records, bad = load_priors(prior_dir)
    if not records:
        print(f"{prior_dir} 下没有可解析的 prior JSON")
        return 2

    rep = diversity_report(records)
    rep["n_parse_failed"] = len(bad)
    rep["parse_failed"] = bad[:20]
    rep["formats"] = dict(Counter(r["format"] for r in records))

    if args.data:
        rep["sector_check"] = sector_consistency(
            records, Path(args.data), px_per_m, radius_m=args.local_radius_m)

    if not args.no_gate:
        rep["gate"] = evaluate_gates(rep)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 控制台摘要 ----------
    print(f"样本 {rep['n_files']}（解析失败 {rep['n_parse_failed']}）｜格式 {rep['formats']}")
    print(f"唯一向量 {rep['n_unique_vector']}｜unique_ratio {rep['unique_ratio']:.4f}"
          f"｜同 map 内唯一占比 {rep['intra_map_unique_ratio_mean']:.4f}")
    print(f"同 map 内会随 Tx 变化的字段数 {rep['fields_varying_within_map']} / {len(ALL_KEYS)}")
    if rep["missing_keys"]:
        print(f"  缺失字段（不计入统计）：{rep['missing_keys']}")
    for k, v in sorted(rep["per_key"].items(), key=lambda kv: -kv[1]["top1_share"])[:5]:
        print(f"  {k:<26} unique={v['unique']:<5} top1={v['top1_share']:.3f} "
              f"H={v['entropy_bits']:.2f} bits")
    num, cat = rep.get("numeric", {}), rep.get("categorical", {})
    mat = rep.get("material_numeric", {})
    emin = num.get("entropy_min")
    print(f"几何条件数值字段：唯一值最少 {num.get('unique_min')}"
          f"（门槛 >= {GATES_FAIL['numeric_unique_min']}）"
          f"｜熵最低 {'—' if emin is None else round(emin, 2)}"
          f"（门槛 >= {GATES_FAIL['numeric_entropy_min']}）")
    if num.get("failing_unique"):
        print(f"  唯一值不足的字段：{num['failing_unique']}")
    if num.get("failing_entropy"):
        print(f"  熵过低的字段：{num['failing_entropy']}")
    if mat.get("values"):
        short = ", ".join(f"{k}={v['top1_value']}({v['top1_share']:.2f})"
                          for k, v in mat["values"].items())
        print(f"材料属性字段（不参加硬门槛，恒定为物理常识）：{short}")
    print(f"类别字段：取值数最少 {cat.get('unique_min')}"
          f"（门槛 >= {GATES_FAIL['categorical_min_unique']}）")
    if cat.get("constant_keys"):
        print(f"  退化成单一常量的字段：{cat['constant_keys']}")
    s = rep["sectors"]
    print(f"sectors：有 sectors 的样本 {s['n_with_sectors']}｜唯一剖面 {s['unique_profiles']}"
          f"｜剖面唯一率 {s['sector_profile_unique_ratio']:.4f}"
          f"｜最常见剖面占比 {s['top1_profile_share']:.4f}")
    if "reasoning" in rep:
        r = rep["reasoning"]
        print(f"reasoning：唯一 {r['unique']}/{r['n']}（唯一率 {r['unique_ratio']:.4f}）"
              f"｜最常见占比 {r['top1_share']:.4f}｜平均长度 {r['mean_len']:.0f} 字")
        print(f"  最常见文本开头：{r['top1_head']!r}")
    cc = rep["content_contamination"]
    print(f"内容污点合计：{cc['n_hit']}/{cc['n']} = {cc['share']:.4f}")
    for gname, g in cc["by_group"].items():
        label = "频段幻觉(28GHz/毫米波)" if gname == "mmwave" else "高度幻觉(掩码无高度)"
        print(f"  {label}: {g['n_hit']}/{cc['n']} = {g['share']:.4f}")
        if g["example"]:
            print(f"    例：{g['example']!r}")
    if rep.get("field_consistency_mean") is not None:
        print(f"多轮采样一致性均值：{rep['field_consistency_mean']:.4f}"
              f"（{rep['n_with_provenance']} 条含 provenance）")
    if "sector_check" in rep:
        sc = rep["sector_check"]
        print(f"sectors 与几何一致率：严格 {sc['agreement_strict']:.4f}"
              f"｜相邻容差 {sc['agreement_adjacent']:.4f}（检查 {sc['n_checked']} 个扇区）")
        if sc.get("n_shifted_checked"):
            print(f"  错位对照（同图另一 Tx）严格一致率 {sc['agreement_strict_shifted']:.4f}"
                  f" → **增益 {sc['agreement_gain']:+.4f}**"
                  f"（门槛 >= {GATES_FAIL['sector_consistency_gain']}；"
                  f"恒答某标签 / 只记方位先验的增益必为 0）")
        for sname, sv in (sc.get("subsets") or {}).items():
            label = "Tx 落在建筑内" if sname == "tx_on_building" else "Tx 在街上"
            print(f"    [{label}] 扇区 {sv['n_sectors']}｜本 Tx {sv['agreement_strict']:.3f}"
                  f"｜错位 {sv['agreement_strict_shifted']:.3f}"
                  f"｜增益 {sv['agreement_gain']:+.3f}")
        print(f"  混淆 top：{sc['confusion_expected_vs_llm']}")
    print(f"报告写入 {out}")

    if "gate" in rep:
        print("\n门槛判定：")
        for r in rep["gate"]["results"]:
            tag = "SKIP" if r["pass"] is None else ("PASS" if r["pass"] else "FAIL")
            sev = "硬" if r["severity"] == "fail" else "建议"
            print(f"  [{tag}][{sev}] {r['gate']} = {r['value']} "
                  f"(要求 {r['op']} {r['threshold']})")
        if not rep["gate"]["pass"]:
            print(f"\n[FAIL] {rep['gate']['hard_failures']} 条硬门槛未通过："
                  f"请先改 prompt / 输入图（交接文档 §5.5），**不要进入训练阶段**。")
            return 0 if args.weak_ok else 1
        if rep["gate"]["warnings"]:
            print(f"\n[OK] 硬门槛全部通过；但有 {rep['gate']['warnings']} 条建议级告警，"
                  f"请在报告里如实说明。")
        else:
            print("\n[OK] 全部通过，可以进入下一阶段。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
