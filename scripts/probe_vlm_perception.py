#!/usr/bin/env python3
"""测量 VLM 在这批建筑掩码上到底能读出什么，以及换一种分工是否可行。

为什么需要这个脚本
-----------------
项目的原始设计假设"VLM 看着建筑掩码就能给出逐方位的遮挡判断"。第一轮真实 API 调用
（12 个样本）显示：sectors 是编造的等差序列、与几何一致率 0.104（低于随机 0.25）、
`los_ratio` 恒为 0、50 m 与 100 m 建筑占比被填成同一个值。在改 prompt 之前必须先确定
**这到底是 prompt 的问题，还是模型能力的上限** —— 否则会陷入"反复调 prompt、
每次都在跟模型能力打架"的循环。

三个子命令
---------
  sanity   人造图对照：单张清晰色块。用来证明"图像确实送到了模型"。
  perceive 真实图上的两个**有方差**的感知题（左右半边谁更白 / 哪个方位建筑最近），
           对多个视觉模型各测一遍。用来定位能力上限。
  facts    换分工的可行性：把程序算出的精确几何当**事实**喂给 LLM，
           只让它做电磁推理（机制权重 / 材质 / 介电常数 / 穿透损耗 / pathloss 区间），
           然后统计这些字段有没有逐样本差异。

用法：
    python scripts/probe_vlm_perception.py sanity
    python scripts/probe_vlm_perception.py perceive --models glm-4v-flash glm-4v glm-4v-plus
    python scripts/probe_vlm_perception.py facts --n 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from src.data.sectors import geometric_obstruction, local_building_ratio  # noqa: E402
from src.prior.prompt_v2 import render_panels, render_prompt_image, to_data_url  # noqa: E402
from src.prior.schema import AZIMUTHS  # noqa: E402

DATA = ROOT / "data" / "radiomapseer"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


def client():
    from openai import OpenAI
    key = os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("ZHIPU_API_KEY")
    if not key:
        raise SystemExit("未设置 ZHIPUAI_API_KEY")
    return OpenAI(api_key=key, base_url=BASE_URL, timeout=120)


def local_crop(mask: np.ndarray, tx, r: int = 100) -> np.ndarray:
    g = np.where(mask > 127, 255, 0).astype(np.uint8)
    p = np.pad(g, ((r, r), (r, r)))
    px, py = int(round(tx[0])) + r, int(round(tx[1])) + r
    return p[py - r:py + r, px - r:px + r]


def load(m: int, t: int):
    mask = np.array(Image.open(DATA / "buildings_complete" / f"{m}.png").convert("L"))
    ants = json.loads((DATA / "antenna" / f"{m}.json").read_text(encoding="utf-8"))
    return mask, ants[t]


def ask(cli, model: str, img, question: str, max_tokens: int = 16,
        system: str | None = None) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": to_data_url(img)}},
        {"type": "text", "text": question}]})
    r = cli.chat.completions.create(model=model, temperature=0.0,
                                    max_tokens=max_tokens, messages=msgs)
    return (r.choices[0].message.content or "").strip()


# ---------------------------------------------------------------- sanity

def cmd_sanity(args) -> int:
    """人造图：证明图像管道正常。答对说明模型能定位红叉并判读大块色块。"""
    cli = client()

    def make(kind: str) -> Image.Image:
        img = Image.new("RGB", (256, 256), (0, 0, 0))
        d = ImageDraw.Draw(img)
        if kind == "white":
            d.rectangle([60, 60, 200, 200], fill=(255, 255, 255))
        elif kind == "leftwhite":
            d.rectangle([0, 0, 120, 256], fill=(255, 255, 255))
        elif kind == "rightwhite":
            d.rectangle([136, 0, 256, 256], fill=(255, 255, 255))
        d.line([120, 128, 136, 128], fill=(255, 0, 0), width=3)
        d.line([128, 120, 128, 136], fill=(255, 0, 0), width=3)
        return img

    q = ("Answer with exactly two words: first answer the colour question then the side question.\n"
         "Q1: is the pixel at the centre (the red cross) white or black?\n"
         "Q2: which half of the image contains more white pixels, left or right?")
    ok = 0
    for kind in ("white", "black", "leftwhite", "rightwhite"):
        a = ask(cli, args.model, make(kind), q, max_tokens=16).replace("\n", " ")
        print(f"  {kind:<11} → {a[:80]}")
        ok += 1
    print("\n  期望：white→white,  leftwhite→left,  rightwhite→right,  black→black")
    print("  只要答案随图变化（不是恒答同一个词），就说明图像内容确实进入了模型。")
    return 0


# -------------------------------------------------------------- perceive

PERCEIVE_SAMPLES = [(0, 0), (0, 3), (0, 2), (0, 6), (0, 7), (0, 9), (1, 0), (2, 0)]


def cmd_perceive(args) -> int:
    """真实图上两个有方差的感知题：粗尺度（左右）与细尺度（最近方位）。"""
    import re
    cli = client()
    q_lr = ("Look at this image. Is there MORE white area in the LEFT half or in the RIGHT half?\n"
            "Answer with exactly one word: LEFT or RIGHT.")
    q_frac = ("Look at this image. Roughly what fraction of the image is white? "
              "Answer with a single number between 0 and 1, nothing else.")
    q_az = ("The red cross is the transmitter and there are 8 yellow guides, each labelled with an "
            "angle. Walking outward from the cross, along WHICH guide does a white (building) area "
            "begin CLOSEST to the cross?\nAnswer with the angle number only "
            "(one of 0,45,90,135,180,225,270,315).")

    base = None
    for model in args.models:
        try:
            cli.chat.completions.create(model=model, max_tokens=4,
                                        messages=[{"role": "user", "content": "hi"}])
        except Exception as exc:  # noqa: BLE001
            print(f"  {model:<26} 不可用: {str(exc)[:70]}")
            continue
        hit_lr = n_lr = hit_az = n_az = 0
        frac_err, frac_pairs, lr_ans = [], [], []
        az_ans = []
        for (m, t) in PERCEIVE_SAMPLES:
            mask, tx = load(m, t)
            crop = local_crop(mask, tx)
            is_b = bool(mask[int(round(tx[1])), int(round(tx[0]))] > 127)
            truth_lr = ("LEFT" if (crop[:, :100] > 127).mean() > (crop[:, 100:] > 127).mean()
                        else "RIGHT")
            _, plain = render_panels(mask, tx, guides=False, captions=False)
            a = ask(cli, model, plain, q_lr, max_tokens=8).upper()
            pick = "LEFT" if "LEFT" in a else ("RIGHT" if "RIGHT" in a else "?")
            lr_ans.append(pick)
            n_lr += 1
            hit_lr += int(pick == truth_lr)
            f = ask(cli, model, plain, q_frac, max_tokens=8)
            try:
                est = float("".join(c for c in f if c.isdigit() or c == "."))
            except ValueError:
                est = None
            if est is not None:
                frac_err.append(abs(est - float((crop > 127).mean())))
                frac_pairs.append(f"{est:.2f}/{(crop > 127).mean():.2f}")
            if not is_b:
                with_guides = render_prompt_image(mask, tx)
                g = ask(cli, model, with_guides, q_az, max_tokens=8)
                nums = [int(x) for x in re.findall(r"\d+", g)]
                got = next((x for x in nums if x in AZIMUTHS), None)
                geo = geometric_obstruction(mask, tx, radius_m=100.0)
                nb = {a2: geo[a2]["nearest_building_m"] for a2 in AZIMUTHS}
                truth_az = min(nb, key=lambda a2: nb[a2])
                n_az += 1
                hit_az += int(got == truth_az)
                az_ans.append(got)
        print(f"  {model:<26} 左右 {hit_lr}/{n_lr} = {hit_lr / n_lr:.3f}"
              f"（随机 0.5）| 最近方位 {hit_az}/{n_az} = "
              f"{(hit_az / n_az if n_az else float('nan')):.3f}（随机 0.125）")
        print(f"      左右答案: {lr_ans}")
        if az_ans:
            print(f"      方位答案: {az_ans}")
        if frac_err:
            print(f"      白占比 MAE {np.mean(frac_err):.3f}  估计/真值: {frac_pairs[:4]}")
        base = base or (hit_lr / n_lr)
    print("\n  判读：左右题若接近 0.5 且答案恒定 → 模型不做面积聚合；")
    print("        方位题若远低于 0.125 → 模型读不出逐方位的遮挡结构。")
    return 0


# ----------------------------------------------------------------- facts

FACTS_SYSTEM = """You are a wireless propagation expert. You will be given a city map image around a
transmitter (Tx) AND the exact geometric measurements taken from that map. The measurements were
computed from the building footprint mask and are reliable — treat them as facts.

Facts about the dataset you must respect:
- The map is a BINARY building footprint mask (1 px = 1 m, 256x256 m). It contains NO height
  information: you cannot see how tall any building is, and you must never claim a building height,
  a number of floors, or that brighter pixels mean taller buildings.
- This is a 2-D dataset: the Tx has no meaningful height above ground.
- Carrier frequency is 0.9 GHz (UHF), NOT millimetre wave. Do not reason about 28 GHz or mmWave.

Your job is the ELECTROMAGNETIC reasoning: from the measurements below, infer the propagation
mechanisms, the dominant construction material and its electrical parameters, and a plausible
pathloss range. Reply with ONE JSON object and nothing else:

{"scene_type": "dense_urban"|"urban"|"suburban"|"rural"|"indoor",
 "los_ratio": 0..1, "reflection_ratio": 0..1, "diffraction_ratio": 0..1, "scattering_ratio": 0..1,
 "dominant_material": "concrete"|"brick"|"glass"|"wood"|"metal"|"mixed",
 "permittivity": 1..20, "penetration_loss_db": 0..60,
 "pathloss_min_db": 30..200, "pathloss_max_db": 60..250,
 "confidence": "high"|"medium"|"low",
 "reasoning": "your reasoning, citing the specific measurements"}
"""

NUM_KEYS = ["los_ratio", "reflection_ratio", "diffraction_ratio", "scattering_ratio",
            "permittivity", "penetration_loss_db", "pathloss_min_db", "pathloss_max_db"]
CAT_KEYS = ["scene_type", "dominant_material", "confidence"]


def cmd_facts(args) -> int:
    """把精确几何当事实喂进去，只让 LLM 做电磁推理，看字段有没有逐样本差异。"""
    cli = client()
    cases = [(0, t) for t in (0, 2, 3, 6, 7, 9, 12, 20, 40, 60)] + [(1, 0), (2, 0)]
    cases = cases[: args.n]
    rows = []
    for (m, t) in cases:
        mask, tx = load(m, t)
        geo = geometric_obstruction(mask, tx, radius_m=100.0)
        r50 = local_building_ratio(mask, tx, 50.0)
        r100 = local_building_ratio(mask, tx, 100.0)
        facts = (f"50 m disc building fraction: {r50:.3f}\n"
                 f"100 m disc building fraction: {r100:.3f}\n"
                 f"per-azimuth obstruction within 100 m "
                 f"(class, distance to first building):\n"
                 + "\n".join(f"  {a} deg: {geo[a]['obstruction']}, "
                             f"{geo[a]['nearest_building_m']:.0f} m" for a in AZIMUTHS))
        img = render_prompt_image(mask, tx)
        try:
            r = cli.chat.completions.create(
                model=args.model, temperature=0.7, max_tokens=1024,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": FACTS_SYSTEM},
                          {"role": "user", "content": [
                              {"type": "image_url", "image_url": {"url": to_data_url(img)}},
                              {"type": "text", "text": facts}]}])
            txt = r.choices[0].message.content or ""
            d = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
            d["_ok"] = True
        except Exception as exc:  # noqa: BLE001
            d = {"_ok": False, "_err": f"{type(exc).__name__}: {str(exc)[:70]}"}
        d["_sid"] = f"{m}_{t}"
        rows.append(d)
        print(f"  {d['_sid']:<6} " + ("ok" if d["_ok"] else d["_err"]))

    ok = [r for r in rows if r.get("_ok")]
    print(f"\n成功 {len(ok)}/{len(rows)}")
    if not ok:
        return 1
    print(f"\n{'field':<22}{'unique':>8}{'min':>9}{'max':>9}{'top1':>8}")
    hard_fail = []
    for k in NUM_KEYS:
        v = [str(r[k]) for r in ok]
        c = Counter(v)
        p = np.array(list(c.values())) / len(v)
        lo, hi = min(float(x) for x in v), max(float(x) for x in v)
        print(f"{k:<22}{len(c):>8}{lo:>9.2f}{hi:>9.2f}{p.max():>8.2f}")
        if len(c) < 5:
            hard_fail.append(f"{k}: unique={len(c)} < 5")
    for k in CAT_KEYS:
        c = Counter(str(r[k]) for r in ok)
        print(f"{k:<22}{len(c):>8}{'—':>9}{'—':>9}{max(c.values()) / len(ok):>8.2f}")
        if len(c) < 2:
            hard_fail.append(f"{k}: 退化成单一常量")
    vec = len({tuple(str(r[k]) for k in NUM_KEYS + CAT_KEYS) for r in ok})
    print(f"\n唯一电磁向量 {vec}/{len(ok)} = {vec / len(ok):.3f}"
          f"（门槛 >= 0.5；注意这批样本的**几何**是精确给定的）")
    print("\n结论：")
    if hard_fail:
        print("  [不可行] 即使把精确几何作为事实给出，电磁字段仍然坍缩：")
        for f in hard_fail:
            print("    -", f)
        print("  → 说明瓶颈不只是「读图」，LLM 对这组物理量的输出本身缺乏逐样本信息。")
    else:
        print("  [可行] 给定精确几何后，电磁字段有充分的逐样本差异 —— "
              "「程序测量 + LLM 做电磁推理」这条分工可行。")
    print(f"\n推理示例：{ok[0]['reasoning'][:260]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="测量 VLM 在这批掩码上的能力边界")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sanity")
    s.add_argument("--model", default="glm-4v-flash")
    p = sub.add_parser("perceive")
    p.add_argument("--models", nargs="+",
                   default=["glm-4v-flash", "glm-4v", "glm-4v-plus"])
    f = sub.add_parser("facts")
    f.add_argument("--model", default="glm-4v-flash")
    f.add_argument("--n", type=int, default=12)
    args = ap.parse_args(argv)
    return {"sanity": cmd_sanity, "perceive": cmd_perceive, "facts": cmd_facts}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
