"""Prompt v2：给 LLM 看"全局掩码 + Tx 中心局部裁剪"，且不给答案。

相对上一版 prompt 的三处关键改动（交接文档 §5.5）
------------------------------------------------
1. **不再把程序算好的答案塞进 prompt**。上一版用
   `estimate_scene_type()` 算好 `scene_hint` 并以 `scene_type 应为 '{scene_hint}'`
   直接告知 LLM，LLM 只是在抄答案。v2 完全删掉这件事；程序算出的真实统计量只在
   **生成之后**用来核对（`src/prior/audit.py` 的 sector 一致率），
   LLM 的"判断与真值是否一致"本身就是一个可报告的指标。
2. **不只给全局掩码**。额外给以 Tx 为中心、半径 100 m 的局部放大图 —— 上一版 LLM
   看不到"这个 Tx 周围"的情况，只能给整图一个笼统描述，这是先验退化成 per-map
   常量的根因。
3. **不再说"brighter pixels 表示更高建筑"**。掩码实测严格二值 {0,255}，不含任何
   高度信息，上一版这句话是事实错误，也是 475 条先验里出现"Tx 高度 25 m"的直接来源。

单图设计
--------
GLM-4V-Flash 对单次请求的图片数量有限制，因此这里把两块内容**合成一张 PNG**：
左panel = 全局掩码（标出 Tx 与 100 m 裁剪圈），右panel = 局部放大（标出 Tx、
50/100 m 圈、以及 8 个方位的方位线）。这样单图模型也能看到全部信息。

方位线是刻意画上去的：它消除"0° 在哪、是顺时针还是逆时针"的歧义，
否则 sectors 一致率检查测的就不是"LLM 会不会看图"，而是"LLM 猜没猜中坐标系"。
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.prior.schema import AZIMUTHS

# ===== 渲染常量 =====
GLOBAL_PANEL = 256          # 全局面板边长（保持原始尺度，1px=1m）
LOCAL_RADIUS_M = 100.0      # 局部裁剪半径（米）
LOCAL_ZOOM = 2              # 局部放大倍数 → 400×400
GAP = 12                    # 两面板之间的间隔
AZ_LABELS = {0: "0E", 90: "90N", 180: "180W", 270: "270S",
             45: "45", 135: "135", 225: "225", 315: "315"}

COLOR_BG = (0, 0, 0)
COLOR_BUILDING = (255, 255, 255)
COLOR_TX = (255, 40, 40)
COLOR_GUIDE = (255, 190, 0)
COLOR_TEXT = (0, 255, 120)


def _font(size: int = 12) -> ImageFont.ImageFont:
    try:  # Pillow >= 10.1 支持指定字号
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - 老版本 Pillow
        return ImageFont.load_default()


def _pad_mask(mask: np.ndarray, pad: int) -> np.ndarray:
    """四周补 0（空地）后返回，保证任何 Tx 位置都能裁出完整半径。"""
    if pad <= 0:
        return mask
    return np.pad(mask, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)


def _text_bg(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str,
             fill=COLOR_TEXT, font: ImageFont.ImageFont | None = None,
             bg: tuple[int, int, int] = (0, 0, 0)) -> None:
    """带底色矩形地写字：掩码是白底建筑，直接写绿字会看不清。"""
    font = font or _font(10)
    x, y = xy
    try:
        l, t, r, b = draw.textbbox((x, y), text, font=font)
    except Exception:  # noqa: BLE001  老版本 Pillow
        l, t, r, b = x, y, x + 8 * len(text), y + 12
    draw.rectangle([l - 2, t - 1, r + 2, b + 1], fill=bg)
    draw.text((x, y), text, fill=fill, font=font)


def _draw_azimuth_guides(draw: ImageDraw.ImageDraw, cx: float, cy: float,
                         radius: float, font: ImageFont.ImageFont) -> None:
    """画出 8 条方位线 + 方位标签（0°=右/东，逆时针）。

    标签刻意画在 0.86·radius 处并带底色：0° 是最关键的方向，不能像初版那样
    被面板边缘裁掉，也不能压在白花花的建筑掩码上看不清。
    """
    import math

    for az in AZIMUTHS:
        th = math.radians(az)
        dx, dy = math.cos(th), -math.sin(th)      # 像素 y 向下，故取 −sin
        draw.line([cx, cy, cx + dx * radius, cy + dy * radius],
                  fill=COLOR_GUIDE, width=1)
        lx, ly = cx + dx * radius * 0.86, cy + dy * radius * 0.86
        label = AZ_LABELS[az]
        _text_bg(draw, (lx - 9, ly - 6), label, fill=COLOR_GUIDE, font=font)


def render_panels(
    mask: np.ndarray,
    tx_xy: tuple[float, float] | list[float],
    radius_m: float = LOCAL_RADIUS_M,
    px_per_m: float = 1.0,
    zoom: int = LOCAL_ZOOM,
    guides: bool = True,
    captions: bool = True,
) -> tuple[Image.Image, Image.Image]:
    """分别渲染（全局面板, 局部放大面板）。

    拆成独立函数有两个理由：
    * 自检可以只看面板内容，不受画布拼接布局影响；
    * 报告里要单独用某一块时不必再裁图。
    """
    mask = np.asarray(mask)
    h, w = mask.shape
    font = _font(12)
    small = _font(10)
    tx_x, ty = float(tx_xy[0]), float(tx_xy[1])
    r_px = radius_m * px_per_m

    # ---------- 左：全局面板 ----------
    g = np.where(mask > 127, 255, 0).astype(np.uint8)
    left = Image.fromarray(g).convert("RGB")
    gl = ImageDraw.Draw(left)
    if guides:
        gl.ellipse([tx_x - r_px, ty - r_px, tx_x + r_px, ty + r_px],
                   outline=(0, 200, 255), width=1)
        gl.line([tx_x - 8, ty, tx_x + 8, ty], fill=COLOR_TX, width=2)
        gl.line([tx_x, ty - 8, tx_x, ty + 8], fill=COLOR_TX, width=2)
        gl.ellipse([tx_x - 3, ty - 3, tx_x + 3, ty + 3], fill=COLOR_TX)
    if captions:
        _text_bg(gl, (3, 3), "GLOBAL 256x256, 1px=1m", font=small)
        _text_bg(gl, (3, 17), "white=building  black=open", font=small)
        _text_bg(gl, (3, h - 15), f"red=Tx({int(tx_x)},{int(ty)})  cyan=100m", font=small)

    # ---------- 右：局部放大面板 ----------
    # 以 Tx 为中心裁 2·r 见方；Tx 靠边时用 0（空地）补边，保证裁剪窗口完整
    pad = int(np.ceil(r_px))
    padded = Image.fromarray(_pad_mask(g, pad))
    px = int(round(tx_x)) + pad
    py = int(round(ty)) + pad
    crop = padded.crop((px - pad, py - pad, px + pad, py + pad))
    side = crop.width * zoom
    right = crop.resize((side, side), Image.NEAREST).convert("RGB")
    rr = ImageDraw.Draw(right)
    c = side / 2.0
    scale = zoom * px_per_m                    # 放大后的 像素/米
    if guides:
        rr.ellipse([c - 50 * scale, c - 50 * scale, c + 50 * scale, c + 50 * scale],
                   outline=(0, 200, 255), width=1)
        rr.ellipse([c - 100 * scale, c - 100 * scale, c + 100 * scale, c + 100 * scale],
                   outline=(0, 140, 200), width=1)
        _draw_azimuth_guides(rr, c, c, side / 2.0 - 12, font)
        rr.line([c - 8, c, c + 8, c], fill=COLOR_TX, width=2)
        rr.line([c, c - 8, c, c + 8], fill=COLOR_TX, width=2)
        rr.ellipse([c - 3, c - 3, c + 3, c + 3], fill=COLOR_TX)
    if captions:
        _text_bg(rr, (3, 3), f"LOCAL {radius_m:.0f}m radius, {zoom}x zoom", font=small)
        _text_bg(rr, (3, 17), "cyan=50m  blue=100m", font=small)
        _text_bg(rr, (3, side - 15), "azimuth 0=right(E), CCW", font=small)
    return left, right


def render_prompt_image(
    mask: np.ndarray,
    tx_xy: tuple[float, float] | list[float],
    radius_m: float = LOCAL_RADIUS_M,
    px_per_m: float = 1.0,
    zoom: int = LOCAL_ZOOM,
    guides: bool = True,
    captions: bool = True,
) -> Image.Image:
    """把两块内容合成一张 PNG（单图模型也能一次看到全部信息）。

    全局面板垂直居中，避免默认 zoom=2 时下方留一大块无意义的黑边。
    """
    left, right = render_panels(mask, tx_xy, radius_m, px_per_m, zoom, guides, captions)
    height = max(left.height, right.height)
    canvas = Image.new("RGB", (left.width + GAP + right.width, height), COLOR_BG)
    canvas.paste(left, (0, (height - left.height) // 2))
    canvas.paste(right, (left.width + GAP, 0))
    return canvas


def to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    """PIL 图像 → base64 data URL（OpenAI 兼容协议可直接用）。"""
    buf = io.BytesIO()
    img.save(buf, format=fmt, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


SYSTEM_PROMPT = """\
You are a wireless propagation expert. You will be given ONE image that shows a city map
around a transmitter (Tx), and you must output an electromagnetic propagation prior as JSON.

WHAT THE IMAGE SHOWS (read this carefully — do not guess beyond it)
- The image has TWO panels side by side.
- LEFT panel "GLOBAL": the whole map, 256x256 pixels. 1 pixel = 1 metre.
  WHITE = building footprint. BLACK = open area (street, plaza, water, vegetation).
- RIGHT panel "LOCAL": a zoomed view of the 100 m radius disc centred on the Tx.
  The yellow lines are 8 azimuth guides, one every 45 degrees. The text labels on them
  are the azimuth values in degrees. The cyan circle is 50 m, the blue circle is 100 m.
- The red cross marks the Tx position (it is the same physical point in both panels).

HARD FACTS ABOUT THIS DATASET (violating them makes your answer wrong)
- The map is a BINARY building footprint mask. It contains NO height information.
  You CANNOT see how tall any building is. Never claim or imply a building height,
  a number of floors, or that "brighter pixels mean taller buildings".
- This is a 2-D dataset: the Tx has no meaningful height above ground.
- Carrier frequency is 0.9 GHz (UHF), NOT millimetre wave. Do not reason about 28 GHz,
  mmWave blockage, or oxygen absorption. Penetration loss at 0.9 GHz is much lower than
  at 28 GHz.
- Distances: use the circles. The wide view is 256 m across; the local view is 200 m across.

AZIMUTH CONVENTION (must match the guides drawn in the image)
- 0 = right / East. 90 = up / North. 180 = left / West. 270 = down / South.
- Angles increase counter-clockwise.
- For each of the 8 azimuths you must look outward from the Tx along that guide and judge
  what the signal meets within roughly 100 m.

THREE CHECKS THAT DECIDE MOST OF THE ANSWER (do these before writing anything)
1. Where does the red Tx cross sit in the LOCAL panel — on WHITE (building) or on BLACK
   (open ground)? If it sits on white, the transmitter is inside a building footprint.
   Then the signal is obstructed in EVERY direction at essentially zero distance:
   report obstruction "blocked" and nearest_building_m 0 for all eight azimuths and say
   so in your reasoning. Do not describe open directions in that case.
2. Your eight nearest_building_m values must come from what you actually see along each
   guide. They must NOT form a regular sequence. An answer like 15, 20, 25, 30, 35, 40,
   45, 50 is a made-up ramp, not a measurement. Real geometry is irregular: some
   directions have a wall a few metres away, others have nothing for a hundred metres.
   If you cannot see any building along a guide, write 500.
3. The 50 m and 100 m local building fractions are averages of overlapping discs — the
   100 m disc contains the 50 m disc. In this dataset they are normally within about
   0.1 of each other. If your two numbers differ by much more than that, you have
   misread one of them; go back and re-estimate.

HOW TO REASON (this text goes into the "reasoning" field)
Work through these steps and cite concrete evidence from the picture (distances in metres,
azimuths, and what you actually see along each direction). Do not write generic filler.
  1. Overall built-up density of the GLOBAL panel (rough fraction that is white).
  2. The local surroundings of the Tx: what is within 50 m and within 100 m? Which
     azimuths are open, which are closed by a building wall, which have a building very close?
  3. Which propagation mechanisms dominate at 0.9 GHz in this geometry: line-of-sight,
     wall reflection, edge diffraction, diffuse scattering.
  4. Dominant construction material, and a physically plausible relative permittivity and
     penetration loss for it at 0.9 GHz.
  5. A plausible pathloss range (min in the open near the Tx, max deep in shadow) in dB.
  6. How confident you are, and what in the image makes you less confident.

OUTPUT FORMAT
- Reply with ONE JSON object and nothing else: no markdown fences, no commentary.
- Required keys, in this order:

{
  "scene_type": "dense_urban" | "urban" | "suburban" | "rural" | "indoor",
  "los_ratio": float 0..1,              // share of the map with a clear direct path
  "reflection_ratio": float 0..1,
  "diffraction_ratio": float 0..1,
  "scattering_ratio": float 0..1,       // these four should roughly sum to 1
  "dominant_material": "concrete" | "brick" | "glass" | "wood" | "metal" | "mixed",
  "permittivity": float 1..20,          // relative permittivity of that material
  "penetration_loss_db": float 0..60,   // one wall, at 0.9 GHz
  "pathloss_min_db": float 30..200,
  "pathloss_max_db": float 60..250,     // must be strictly greater than pathloss_min_db
  "confidence": "high" | "medium" | "low",
  "tx_local_building_ratio_50m": float 0..1,   // white fraction inside the 50 m circle
  "tx_local_building_ratio_100m": float 0..1,  // white fraction inside the 100 m circle
  "sectors": [                          // EXACTLY 8 entries, one per azimuth, any order
    {"azimuth_deg": 0,   "obstruction": "open"|"sparse"|"dense"|"blocked",
     "nearest_building_m": float 0..500},
    ... one for each of 0, 45, 90, 135, 180, 225, 270, 315 ...
  ],
  "reasoning": "your step-by-step reasoning, citing the picture"
}

DEFINITIONS
- obstruction: "open" = no building within 100 m along that guide;
  "sparse" = buildings present but with gaps, sky/open path still exists;
  "dense" = a continuous built frontage along that direction;
  "blocked" = a building wall is essentially on top of the Tx in that direction.
- nearest_building_m: distance from the Tx to the first building along that guide.
  If there is no building within 100 m, write 500.

FORMAT ILLUSTRATION ONLY — the numbers below are arbitrary placeholders, they are NOT
answers and you must NOT copy them. They only show the shape and the value types:

{"scene_type":"urban","los_ratio":0.33,"reflection_ratio":0.27,"diffraction_ratio":0.26,
 "scattering_ratio":0.14,"dominant_material":"concrete","permittivity":5.5,
 "penetration_loss_db":12.0,"pathloss_min_db":62.0,"pathloss_max_db":148.0,
 "confidence":"medium","tx_local_building_ratio_50m":0.27,
 "tx_local_building_ratio_100m":0.22,
 "sectors":[{"azimuth_deg":0,"obstruction":"sparse","nearest_building_m":41.0},
            {"azimuth_deg":45,"obstruction":"open","nearest_building_m":500.0},
            {"azimuth_deg":90,"obstruction":"dense","nearest_building_m":17.0},
            {"azimuth_deg":135,"obstruction":"dense","nearest_building_m":13.0},
            {"azimuth_deg":180,"obstruction":"blocked","nearest_building_m":6.0},
            {"azimuth_deg":225,"obstruction":"sparse","nearest_building_m":55.0},
            {"azimuth_deg":270,"obstruction":"open","nearest_building_m":500.0},
            {"azimuth_deg":315,"obstruction":"sparse","nearest_building_m":38.0}],
 "reasoning":"placeholder text"}
"""


def build_user_text(
    map_id: int,
    tx_id: int,
    tx_xy: tuple[float, float] | list[float],
    freq_ghz: float = 0.9,
    px_per_m: float = 1.0,
) -> str:
    """用户消息文本。**只给事实，不给程序算出的答案**（无 scene_hint）。"""
    return (
        f"Sample: map {map_id}, transmitter {tx_id}.\n"
        f"Tx pixel position: ({int(tx_xy[0])}, {int(tx_xy[1])}), "
        f"i.e. about {int(tx_xy[0])} m from the left edge and {int(tx_xy[1])} m from the top edge "
        f"of the 256 m x 256 m map (1 px = {px_per_m:g} m).\n"
        f"Carrier frequency: {freq_ghz:g} GHz.\n"
        f"Analyse the two panels and reply with the JSON object only."
    )


def build_messages(
    mask: np.ndarray,
    map_id: int,
    tx_id: int,
    tx_xy: tuple[float, float] | list[float],
    freq_ghz: float = 0.9,
    px_per_m: float = 1.0,
    radius_m: float = LOCAL_RADIUS_M,
    zoom: int = LOCAL_ZOOM,
) -> list[dict]:
    """构造 OpenAI 兼容协议的消息列表（单图 + 文本）。"""
    img = render_prompt_image(mask, tx_xy, radius_m=radius_m, px_per_m=px_per_m, zoom=zoom)
    url = to_data_url(img)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text",
                 "text": build_user_text(map_id, tx_id, tx_xy, freq_ghz, px_per_m)},
            ],
        },
    ]


if __name__ == "__main__":  # 手工自检：渲染一张样例图
    import json
    from pathlib import Path

    from PIL import Image as _I

    root = Path(__file__).resolve().parents[2]
    data = root / "data" / "radiomapseer"
    mask = np.array(_I.open(data / "buildings_complete" / "0.png").convert("L"))
    tx = json.loads((data / "antenna" / "0.json").read_text(encoding="utf-8"))[7]
    out = render_prompt_image(mask, tx)
    dst = root / "reports" / "prompt_v2_example.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)
    print("渲染样例图:", dst, out.size)
    msgs = build_messages(mask, 0, 7, tx)
    print("消息数:", len(msgs), "| system 长度:", len(msgs[0]["content"]),
          "| 图像 data URL 前缀:", msgs[1]["content"][0]["image_url"]["url"][:30])
