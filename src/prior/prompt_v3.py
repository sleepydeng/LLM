"""Prompt v3：程序负责**精确测量**，LLM 负责**电磁推理**。

为什么要有 v3（这是实测逼出来的，不是设计偏好）
----------------------------------------------
v2 的假设是"VLM 看着建筑掩码 + 8 条方位线，就能给出逐方位的遮挡判断"。
用真实掩码实测后这个假设不成立（详见 `reports/vlm_perception_probe.md`）：

* 三个可用的智谱视觉模型在"左右半边谁更白"上分别是 0.250 / 0.500 / 0.250（随机 0.5），
  "哪个方位建筑最近"是 0/6 = 0.000（随机 0.125），而且**换 5 种输入图答案完全不变**
  —— 它不是在读图，而是在给一个位置偏置的默认词；
* 12 样本真实生成：sectors 与几何一致率 0.104（低于随机 0.25）、
  `nearest_building_m` 出现 15/20/25/30/35/40/45/50 这种**编造的等差序列**、
  `los_ratio` 12/12 全是 0.00、50 m 与 100 m 建筑占比被填成同一个值。

所以 v3 换分工：**几行 numpy 就能精确算出的几何量，不再让 LLM 去"看"，直接作为事实给它**；
LLM 只做它擅长且是本研究真正想验证的事 —— **电磁推理**：
传播机制权重、主导材质与电参数、pathloss 区间、以及引用具体测量值的推理链。
这与交接文档 §1.3 指出的空隙一致："没人把 LLM 当会做电磁推理的领域专家用"。

必须在论文里说清楚的一件事
------------------------
v3 的先验里，**空间信息来自程序、电磁信息来自 LLM**。因此
"LLM 先验有用"这个结论必须用 `prior_geometry_only` 对照来支撑：
即"只给几何"与"几何 + LLM 电磁推理"相比，差值才是 LLM 的贡献。
如果两者相同，说明 LLM 的电磁字段只是把给定几何重新编码了一遍 —— 那是一个
**有价值的负面结果**，不能靠"不设这个对照"来掩盖。
"""
from __future__ import annotations

import numpy as np

from src.data.sectors import geometric_obstruction, local_building_ratio
from src.prior.schema import AZIMUTHS

# ---- 给 LLM 的 system prompt：只问电磁推理，不问空间 -------------------------------------------------

GEOMETRY_FACTS_SYSTEM = """\
You are a wireless propagation expert. You will receive (a) a city map image around a
transmitter (Tx) and (b) EXACT geometric measurements computed from that map. The measurements are
reliable ground truth derived from the building footprint mask — treat them as facts and reason
from them. Do not try to re-measure the map by eye; use the numbers.

FACTS ABOUT THIS DATASET THAT YOU MUST RESPECT
- The map is a BINARY building footprint mask. It contains NO height information. You cannot see
  how tall any building is. Never claim or imply a building height, a number of floors, or that
  brighter pixels mean taller buildings.
- This is a 2-D dataset: the Tx has no meaningful height above ground. All buildings are treated as
  vertical obstructions regardless of height.
- Carrier frequency is 0.9 GHz (UHF), NOT millimetre wave. Do not reason about 28 GHz, mmWave
  blockage, or oxygen absorption. Penetration loss at 0.9 GHz is much lower than at 28 GHz.

WHAT YOU MUST INFER (this is the whole task)
From the measurements, infer the electromagnetic character of this link:
  1. the four propagation-mechanism weights (line-of-sight, wall reflection, edge diffraction,
     diffuse scattering) — they should roughly sum to 1;
  2. the dominant construction material and its relative permittivity, plus the penetration loss
     through one wall at 0.9 GHz;
  3. a plausible pathloss range in dB — the minimum occurs in the open near the Tx, the maximum
     deep inside a shadow region;
  4. how confident you are, and what specifically limits your confidence.

RESPOND WITH ONE JSON OBJECT AND NOTHING ELSE (no markdown fences, no commentary):
{
  "scene_type": "dense_urban" | "urban" | "suburban" | "rural" | "indoor",
  "los_ratio": float 0..1,
  "reflection_ratio": float 0..1,
  "diffraction_ratio": float 0..1,
  "scattering_ratio": float 0..1,
  "dominant_material": "concrete" | "brick" | "glass" | "wood" | "metal" | "mixed",
  "permittivity": float 1..20,
  "penetration_loss_db": float 0..60,
  "pathloss_min_db": float 30..200,
  "pathloss_max_db": float 60..250,
  "confidence": "high" | "medium" | "low",
  "reasoning": "cite the specific measurements you used and explain the physics"
}
"""


def compute_geometry_facts(
    mask: np.ndarray,
    tx_xy: tuple[float, float] | list[float],
    radius_m: float = 100.0,
    px_per_m: float = 1.0,
) -> dict:
    """程序侧精确测量。**这是 v3 里空间信息的来源**，不经过 LLM。

    返回：
        sectors        —— 8 个方位的精确遮挡等级与最近建筑距离
        r50 / r100     —— Tx 周围 50 m / 100 m 圆内建筑占比
        tx_on_building —— Tx 是否落在建筑像素内（对判读"各方向是否都被挡住"很关键）
        facts_text     —— 写进 prompt 的事实文本
    """
    geo = geometric_obstruction(mask, tx_xy, radius_m=radius_m, px_per_m=px_per_m)
    r50 = local_building_ratio(mask, tx_xy, 50.0, px_per_m)
    r100 = local_building_ratio(mask, tx_xy, 100.0, px_per_m)
    ty, txp = int(round(tx_xy[1])), int(round(tx_xy[0]))
    on_b = bool(0 <= ty < mask.shape[0] and 0 <= txp < mask.shape[1] and mask[ty, txp] > 127)

    sectors = [{"azimuth_deg": int(a),
                "obstruction": str(geo[a]["obstruction"]),
                "nearest_building_m": round(float(geo[a]["nearest_building_m"]), 1)}
               for a in AZIMUTHS]
    lines = [
        f"Building fraction inside the 50 m disc around the Tx: {r50:.3f}",
        f"Building fraction inside the 100 m disc around the Tx: {r100:.3f}",
        ("The Tx pixel itself lies ON a building footprint."
         if on_b else "The Tx pixel lies on open ground (not on a building)."),
        "Per-azimuth obstruction within 100 m "
        "(azimuth in degrees, obstruction class, distance to the first building):",
    ]
    for s in sectors:
        lines.append(f"  {s['azimuth_deg']} deg: {s['obstruction']}, "
                     f"{s['nearest_building_m']:.0f} m")
    counts: dict[str, int] = {}
    for s in sectors:
        counts[s["obstruction"]] = counts.get(s["obstruction"], 0) + 1
    lines.append("Count of azimuths by obstruction class: "
                 + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return {
        "sectors": sectors,
        "tx_local_building_ratio_50m": round(float(r50), 4),
        "tx_local_building_ratio_100m": round(float(r100), 4),
        "tx_on_building": on_b,
        "facts_text": "\n".join(lines),
    }


def build_geometry_facts_messages(
    mask: np.ndarray,
    map_id: int,
    tx_id: int,
    tx_xy: tuple[float, float] | list[float],
    freq_ghz: float = 0.9,
    px_per_m: float = 1.0,
    radius_m: float = 100.0,
    zoom: int = 2,
    include_image: bool = True,
    facts: dict | None = None,
) -> list[dict]:
    """构造 v3 的消息列表：图像（可选）+ 精确几何事实 + 只问电磁推理。

    `include_image=False` 用于一个重要的消融：**图像到底有没有额外贡献**。
    如果拿掉图像后电磁字段几乎不变，那说明图像在 v3 里是装饰。
    """
    from src.prior.prompt_v2 import render_prompt_image, to_data_url

    facts = facts or compute_geometry_facts(mask, tx_xy, radius_m, px_per_m)
    user = (f"Sample: map {map_id}, transmitter {tx_id}.\n"
            f"Tx position: ({int(tx_xy[0])}, {int(tx_xy[1])}) in a 256 m x 256 m map "
            f"(1 px = {px_per_m:g} m).\nCarrier frequency: {freq_ghz:g} GHz.\n\n"
            f"EXACT MEASUREMENTS FROM THE BUILDING MASK:\n{facts['facts_text']}\n\n"
            f"Give the electromagnetic prior as the JSON object described in your instructions.")
    content: list[dict] = []
    if include_image:
        content.append({"type": "image_url",
                        "image_url": {"url": to_data_url(
                            render_prompt_image(mask, tx_xy, radius_m=radius_m,
                                                px_per_m=px_per_m, zoom=zoom))}})
    content.append({"type": "text", "text": user})
    return [{"role": "system", "content": GEOMETRY_FACTS_SYSTEM},
            {"role": "user", "content": content}]


if __name__ == "__main__":  # 手工自检：打印一个样本的事实文本
    import json
    from pathlib import Path

    from PIL import Image

    root = Path(__file__).resolve().parents[2]
    data = root / "data" / "radiomapseer"
    mask = np.array(Image.open(data / "buildings_complete" / "0.png").convert("L"))
    tx = json.loads((data / "antenna" / "0.json").read_text(encoding="utf-8"))[0]
    f = compute_geometry_facts(mask, tx)
    print(f["facts_text"])
    print("\nTx 落在建筑内:", f["tx_on_building"])
    msgs = build_geometry_facts_messages(mask, 0, 0, tx)
    print("消息数:", len(msgs), "| system 长度:", len(msgs[0]["content"]))
