"""LLM 电磁先验的 Schema v2：数据结构 + 分组向量化。

为什么要有 v2（相对上一版的 22 维）：
    上一版 4000 条先验只有 55 个唯一向量，22 维里 19 维基本是常量。根因是
    LLM 只看到一张**全局**建筑掩码 + Tx 坐标，它没有任何办法推断"这个 Tx 周围
    的遮挡情况"，所以只能给整张地图一个笼统描述，`los_ratio` 实际退化成 per-map 常量。

v2 的增量（对应交接文档 §5.5）：
    1. `tx_local_building_ratio_50m / _100m`：以 Tx 为中心的局部建筑占比；
    2. `sectors`：8 个方位扇区（每 45°）的遮挡等级 + 最近建筑距离 —— 这是真正随
       Tx 变化的**空间先验**，回答"信号往哪些方向传得出去"，而不是"这张图是城区"。

设计约定
--------
方位角约定（prompt、schema、audit 三者必须一致，否则 sectors 一致率检查无意义）：
    逆时针为正，0° = 右/东(+x)，90° = 上/北(−y)，180° = 左/西，270° = 下/南，
    即"人看图片"的标准数学约定。像素坐标 y 轴向下，故射线方向为
    (dx, dy) = (cos(az), −sin(az))。见 `src/data/sectors.py:azimuth_ray_mask`。

静默修正被禁止
--------------
任何字段越界、缺字段、sectors 不是 8 个，一律抛 `ValidationError`，
由 `src/prior/generate.py` 记录到 `_failed/` 或带错误反馈重试一次，
**不允许**填默认值后写进先验目录（AGENTS.md 规则 6）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = "v2"

# ===== 枚举取值（顺序即 one-hot 顺序，改动会让已生成的先验失效）=====
SCENE_TYPES: tuple[str, ...] = ("dense_urban", "urban", "suburban", "rural", "indoor")
MATERIALS: tuple[str, ...] = ("concrete", "brick", "glass", "wood", "metal", "mixed")
CONFIDENCES: tuple[str, ...] = ("high", "medium", "low")
OBSTRUCTIONS: tuple[str, ...] = ("open", "sparse", "dense", "blocked")
AZIMUTHS: tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315)

# ===== 数值字段的物理量程 =====
# 归一化用 (x - lo) / (hi - lo)；同时也是 audit 判断"是否越界"的依据。
# 注意：这里的 pathloss 量程是**先验自己声称的范围**，与目标图的
# PATHLOSS_MIN/MAX=40/200（src/data/normalize.py）是两件不同的事，不要混。
NUM_RANGES: dict[str, tuple[float, float]] = {
    "los_ratio": (0.0, 1.0),
    "reflection_ratio": (0.0, 1.0),
    "diffraction_ratio": (0.0, 1.0),
    "scattering_ratio": (0.0, 1.0),
    "permittivity": (1.0, 20.0),
    "penetration_loss_db": (0.0, 60.0),
    "pathloss_min_db": (30.0, 200.0),
    "pathloss_max_db": (60.0, 250.0),
    "tx_local_building_ratio_50m": (0.0, 1.0),
    "tx_local_building_ratio_100m": (0.0, 1.0),
}
NEAREST_BUILDING_RANGE: tuple[float, float] = (0.0, 500.0)

# ===== 分组：支持"只全局 / 只 sectors / 全用"的消融 =====
# v3 追加两个分组，用来表达"程序测量 vs LLM 电磁推理"的分工与对照：
#   em_num        LLM 产出的电磁数值（机制权重 / 材料电参数 / pathloss 区间）
#   local_ratio   程序算出的 Tx 局部建筑占比（精确，不依赖 LLM 看图）
# global_num 保留为 em_num + local_ratio 的并集，便于与 prompt_v2 的历史结果对齐。
GROUP_EM_NUM = "em_num"
GROUP_LOCAL_RATIO = "local_ratio"
GROUP_GLOBAL_NUM = "global_num"
GROUP_GLOBAL_CAT = "global_cat"
GROUP_SECTORS = "sectors"
ALL_GROUPS: tuple[str, ...] = (
    GROUP_EM_NUM, GROUP_LOCAL_RATIO, GROUP_GLOBAL_CAT, GROUP_SECTORS,
)
# 分组 → 该组包含的数值字段（顺序即维度顺序）
EM_NUM_KEYS: tuple[str, ...] = (
    "los_ratio", "reflection_ratio", "diffraction_ratio", "scattering_ratio",
    "permittivity", "penetration_loss_db", "pathloss_min_db", "pathloss_max_db",
)
LOCAL_RATIO_KEYS: tuple[str, ...] = (
    "tx_local_building_ratio_50m", "tx_local_building_ratio_100m",
)
# 便捷别名：这几个组合在消融里反复用到
GEOMETRY_GROUPS: tuple[str, ...] = (GROUP_LOCAL_RATIO, GROUP_SECTORS)
EM_GROUPS: tuple[str, ...] = (GROUP_EM_NUM, GROUP_GLOBAL_CAT)

_GROUP_DIMS: dict[str, int] = {
    GROUP_EM_NUM: len(EM_NUM_KEYS),                                # 8（LLM）
    GROUP_LOCAL_RATIO: len(LOCAL_RATIO_KEYS),                      # 2（程序，精确）
    GROUP_GLOBAL_NUM: len(EM_NUM_KEYS) + len(LOCAL_RATIO_KEYS),    # 10（兼容旧结果）
    GROUP_GLOBAL_CAT: len(SCENE_TYPES) + len(MATERIALS) + len(CONFIDENCES),  # 14（LLM）
    GROUP_SECTORS: len(AZIMUTHS) * (len(OBSTRUCTIONS) + 1),        # 40（程序，精确）
}


class Sector(BaseModel):
    """以 Tx 为中心的一个方位扇区（8 个扇区，每 45°）。"""

    model_config = {"extra": "forbid"}

    azimuth_deg: int = Field(description="方位角，必须取 0/45/90/.../315")
    obstruction: Literal["open", "sparse", "dense", "blocked"] = Field(
        description="该方向 100 m 内的建筑遮挡等级：open=基本无遮挡，"
                    "sparse=零星建筑，dense=连片建筑，blocked=紧邻建筑几乎封死"
    )
    nearest_building_m: float = Field(
        ge=0.0, le=500.0,
        description="该方向上最近建筑的距离（米）；若无建筑取 500 作为「无遮挡」上限",
    )

    @field_validator("azimuth_deg")
    @classmethod
    def _check_azimuth(cls, v: int) -> int:
        if v not in AZIMUTHS:
            raise ValueError(f"azimuth_deg 必须是 {AZIMUTHS} 之一，收到 {v}")
        return v


class RadioPriorV2(BaseModel):
    """LLM 输出的电磁先验（v2）。

    分三块：
      A. 全局场景字段（上一版有，继续保留，允许在报告中与 v1 对比）
      B. Tx 局部字段（新增，随 Tx 变化）
      C. sectors 8 方位遮挡剖面（本次核心增量）
      D. reasoning：CoT 文本。**不进 FiLM 向量**，只在开启 CoT 编码器时进模型，
         否则仅用于审计与 case study（由 config 的 cot_mode 决定）。
    """

    model_config = {"extra": "forbid"}

    # ---- A. 全局场景 ----
    scene_type: Literal["dense_urban", "urban", "suburban", "rural", "indoor"]
    los_ratio: float = Field(ge=0.0, le=1.0)
    reflection_ratio: float = Field(ge=0.0, le=1.0)
    diffraction_ratio: float = Field(ge=0.0, le=1.0)
    scattering_ratio: float = Field(ge=0.0, le=1.0)
    dominant_material: Literal["concrete", "brick", "glass", "wood", "metal", "mixed"]
    permittivity: float = Field(ge=1.0, le=20.0)
    penetration_loss_db: float = Field(ge=0.0, le=60.0)
    pathloss_min_db: float = Field(ge=30.0, le=200.0)
    pathloss_max_db: float = Field(ge=60.0, le=250.0)
    confidence: Literal["high", "medium", "low"]

    # ---- B. Tx 局部（随 Tx 变化）----
    tx_local_building_ratio_50m: float = Field(ge=0.0, le=1.0)
    tx_local_building_ratio_100m: float = Field(ge=0.0, le=1.0)

    # ---- C. sectors ----
    sectors: list[Sector] = Field(min_length=8, max_length=8)

    # ---- D. CoT ----
    reasoning: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> "RadioPriorV2":
        az = sorted(s.azimuth_deg for s in self.sectors)
        if az != list(AZIMUTHS):
            raise ValueError(
                f"sectors 的方位角必须恰好覆盖 {list(AZIMUTHS)}，收到 {az}"
            )
        if not self.pathloss_min_db < self.pathloss_max_db:
            raise ValueError(
                f"pathloss_min_db({self.pathloss_min_db}) 必须小于 "
                f"pathloss_max_db({self.pathloss_max_db})"
            )
        return self


# ===== 向量化 =====


def _norm(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        raise ValueError(f"非法量程 ({lo}, {hi})")
    return (float(x) - lo) / (hi - lo)


def vector_dim(groups: list[str] | tuple[str, ...] = ALL_GROUPS) -> int:
    """给定分组返回向量维度。"""
    unknown = [g for g in groups if g not in _GROUP_DIMS]
    if unknown:
        raise ValueError(f"未知的 prior 分组 {unknown}，可选 {ALL_GROUPS}")
    return sum(_GROUP_DIMS[g] for g in groups)


def prior_to_vector(
    prior: RadioPriorV2 | dict[str, Any],
    groups: list[str] | tuple[str, ...] = ALL_GROUPS,
) -> list[float]:
    """把先验转成归一化向量（全部落在 [0,1]，供 FiLM/条件分支使用）。

    与上一版的关键差别：
      * 每个数值维度都按**自己的**物理量程归一化，不会出现"permittivity=7 和
        场景图=1.0 混在同一个输入里"这种量纲污染（旧版 26 通道正是死在这里）；
      * 新增 sectors 40 维，随 Tx 变化，是本次让先验"有信息量"的主力；
      * CoT 文本不在这里，走 `src/models/cot_encoder.py`。
    """
    if isinstance(prior, dict):
        prior = RadioPriorV2(**prior)

    vec: list[float] = []

    for group, keys in ((GROUP_EM_NUM, EM_NUM_KEYS),
                        (GROUP_LOCAL_RATIO, LOCAL_RATIO_KEYS),
                        (GROUP_GLOBAL_NUM, EM_NUM_KEYS + LOCAL_RATIO_KEYS)):
        # global_num 是历史并集分组：只有它被显式要求、且两个细分组都没被要求时才展开，
        # 避免同时传 (global_num, em_num) 时把同一批字段算两遍。
        if group not in groups:
            continue
        if group == GROUP_GLOBAL_NUM and (
                GROUP_EM_NUM in groups or GROUP_LOCAL_RATIO in groups):
            continue
        for key in keys:
            lo, hi = NUM_RANGES[key]
            vec.append(_norm(getattr(prior, key), lo, hi))

    if GROUP_GLOBAL_CAT in groups:
        for st in SCENE_TYPES:
            vec.append(1.0 if prior.scene_type == st else 0.0)
        for mat in MATERIALS:
            vec.append(1.0 if prior.dominant_material == mat else 0.0)
        for cf in CONFIDENCES:
            vec.append(1.0 if prior.confidence == cf else 0.0)

    if GROUP_SECTORS in groups:
        by_az = {s.azimuth_deg: s for s in prior.sectors}
        lo, hi = NEAREST_BUILDING_RANGE
        for az in AZIMUTHS:  # 固定顺序，保证向量语义稳定
            s = by_az[az]
            for ob in OBSTRUCTIONS:
                vec.append(1.0 if s.obstruction == ob else 0.0)
            vec.append(_norm(s.nearest_building_m, lo, hi))

    expect = vector_dim(groups)
    if len(vec) != expect:
        raise AssertionError(f"向量维度算错：得到 {len(vec)}，期望 {expect}")
    return vec


def sector_levels(prior: RadioPriorV2 | dict[str, Any]) -> dict[int, float]:
    """把 8 个扇区的遮挡等级映射成 [0,1] 标量（open=0 … blocked=1）。

    供 `src/data/dataset.py` 构造 8 个方向楔形通道使用。
    """
    if isinstance(prior, dict):
        prior = RadioPriorV2(**prior)
    order = {ob: i / (len(OBSTRUCTIONS) - 1) for i, ob in enumerate(OBSTRUCTIONS)}
    return {s.azimuth_deg: order[s.obstruction] for s in prior.sectors}


# ===== 落盘记录（先验文件的外层结构）=====


class PriorRecord(BaseModel):
    """先验 JSON 文件的外层结构：把"内容"和"来源"分开，便于审计与复现。"""

    model_config = {"extra": "forbid"}

    schema_version: str = SCHEMA_VERSION
    sample_id: str
    map_id: int
    tx_id: int
    prior: RadioPriorV2
    provenance: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            self.model_dump(), ensure_ascii=False, indent=2, sort_keys=False
        )


class RadioPriorEM(BaseModel):
    """v3：LLM 只负责的**电磁**部分（空间部分由程序精确测量，不经过 LLM）。

    为什么拆出来：实测表明可用视觉模型无法从这批建筑掩码里读出逐方位遮挡
    （详见 `reports/vlm_perception_probe.md`），所以把"测量"交给 numpy、
    把"电磁推理"留给 LLM。这个模型的字段就是 LLM 在本设计里的全部产出。
    """

    model_config = {"extra": "forbid"}

    scene_type: Literal["dense_urban", "urban", "suburban", "rural", "indoor"]
    los_ratio: float = Field(ge=0.0, le=1.0)
    reflection_ratio: float = Field(ge=0.0, le=1.0)
    diffraction_ratio: float = Field(ge=0.0, le=1.0)
    scattering_ratio: float = Field(ge=0.0, le=1.0)
    dominant_material: Literal["concrete", "brick", "glass", "wood", "metal", "mixed"]
    permittivity: float = Field(ge=1.0, le=20.0)
    penetration_loss_db: float = Field(ge=0.0, le=60.0)
    pathloss_min_db: float = Field(ge=30.0, le=200.0)
    pathloss_max_db: float = Field(ge=60.0, le=250.0)
    confidence: Literal["high", "medium", "low"]
    reasoning: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "RadioPriorEM":
        if not self.pathloss_min_db < self.pathloss_max_db:
            raise ValueError(
                f"pathloss_min_db({self.pathloss_min_db}) 必须小于 "
                f"pathloss_max_db({self.pathloss_max_db})")
        return self


def merge_em_with_geometry(em: RadioPriorEM | dict, geometry: dict) -> RadioPriorV2:
    """把 LLM 的电磁字段与程序算出的精确几何合成一个完整的 prior（v2 形状）。

    合成后 `sectors` 与 `tx_local_building_ratio_*` 来自 `geometry`（精确），
    其余字段来自 `em`（LLM）。provenance 里必须记清这个分工，否则事后无法判断
    "先验的信息量"到底来自哪一侧 —— 这正是 `prior_geometry_only` 对照能成立的前提。
    """
    if isinstance(em, dict):
        em = RadioPriorEM(**em)
    payload = em.model_dump()
    payload["sectors"] = geometry["sectors"]
    payload["tx_local_building_ratio_50m"] = geometry["tx_local_building_ratio_50m"]
    payload["tx_local_building_ratio_100m"] = geometry["tx_local_building_ratio_100m"]
    return RadioPriorV2(**payload)


def group_index_map(
    groups: list[str] | tuple[str, ...] = ALL_GROUPS,
) -> dict[str, list[int]]:
    """返回每个分组在拼接向量里占据的**下标列表**。

    用途：做"容量对齐"的对照。例如 geometry + LLM 电磁（dim=64）与
    geometry + 电磁**置零**（dim 也是 64）对比时，需要知道哪些下标属于电磁部分。
    没有这个工具就只能拿 dim=42 的纯几何配置去比，那样 prior_dim 不同、
    差值里混着容量效应 —— 正是本项目反复踩到的那个坑。
    """
    idx: dict[str, list[int]] = {}
    pos = 0
    for group, keys in ((GROUP_EM_NUM, EM_NUM_KEYS),
                        (GROUP_LOCAL_RATIO, LOCAL_RATIO_KEYS),
                        (GROUP_GLOBAL_NUM, EM_NUM_KEYS + LOCAL_RATIO_KEYS)):
        if group not in groups:
            continue
        if group == GROUP_GLOBAL_NUM and (
                GROUP_EM_NUM in groups or GROUP_LOCAL_RATIO in groups):
            continue
        idx[group] = list(range(pos, pos + len(keys)))
        pos += len(keys)
    if GROUP_GLOBAL_CAT in groups:
        n = len(SCENE_TYPES) + len(MATERIALS) + len(CONFIDENCES)
        idx[GROUP_GLOBAL_CAT] = list(range(pos, pos + n))
        pos += n
    if GROUP_SECTORS in groups:
        n = len(AZIMUTHS) * (len(OBSTRUCTIONS) + 1)
        idx[GROUP_SECTORS] = list(range(pos, pos + n))
        pos += n
    expect = vector_dim(groups)
    if pos != expect:
        raise AssertionError(f"分组下标算错：覆盖 {pos} 维，期望 {expect} 维")
    return idx


def indices_for_groups(
    groups: list[str] | tuple[str, ...],
    want: list[str] | tuple[str, ...],
) -> list[int]:
    """在 `groups` 的拼接向量里，取出 `want` 这些分组占用的下标。"""
    m = group_index_map(groups)
    out: list[int] = []
    for g in want:
        out.extend(m.get(g, []))
    return sorted(out)


def make_sample_id(map_id: int, tx_id: int) -> str:
    return f"{map_id}_{tx_id}"


def parse_sample_id(sample_id: str) -> tuple[str, str]:
    """解析样本 id → (map_id, tx_id)，保持字符串形态（沿用旧 audit 的行为）。"""
    parts = Path(sample_id).stem.split("_")
    map_id = parts[0] if parts else "?"
    tx_id = parts[1] if len(parts) > 1 else "?"
    return map_id, tx_id


def load_prior_file(path: str | Path) -> tuple[RadioPriorV2, dict[str, Any]]:
    """读取先验文件，兼容 v2 外层结构与上一版的扁平结构。

    返回 (prior, provenance)。provenance 在旧格式下为空 dict。
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "prior" in raw and "schema_version" in raw:
        rec = PriorRecord(**raw)
        return rec.prior, rec.provenance
    # 旧格式：整个文件就是一个 prior（且没有 v2 字段，会直接抛错——这是有意的：
    # 旧先验本来就不该被新代码当成有效输入，见交接文档 §5.3 C 档）。
    return RadioPriorV2(**raw), {}


def vector_from_file(
    path: str | Path,
    groups: list[str] | tuple[str, ...] = ALL_GROUPS,
) -> list[float]:
    """便捷入口：先验文件 → 归一化向量。"""
    prior, _ = load_prior_file(path)
    return prior_to_vector(prior, groups)


if __name__ == "__main__":  # 手工自检
    demo = {
        "scene_type": "dense_urban",
        "los_ratio": 0.3, "reflection_ratio": 0.25,
        "diffraction_ratio": 0.3, "scattering_ratio": 0.15,
        "dominant_material": "concrete",
        "permittivity": 7.0, "penetration_loss_db": 22.0,
        "pathloss_min_db": 70.0, "pathloss_max_db": 150.0,
        "confidence": "medium",
        "tx_local_building_ratio_50m": 0.42,
        "tx_local_building_ratio_100m": 0.31,
        "sectors": [
            {"azimuth_deg": a, "obstruction": "dense", "nearest_building_m": 18.0}
            for a in AZIMUTHS
        ],
        "reasoning": "示例推理文本。",
    }
    p = RadioPriorV2(**demo)
    for groups in (ALL_GROUPS, (GROUP_GLOBAL_NUM, GROUP_GLOBAL_CAT), (GROUP_SECTORS,)):
        v = prior_to_vector(p, groups)
        print(f"{str(groups):<60} dim={len(v):<3} 范围=[{min(v):.3f}, {max(v):.3f}]")
    print("全部维度 =", vector_dim(ALL_GROUPS))
