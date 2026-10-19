#!/usr/bin/env python3
"""试生成前的可行性检查：几何本身能给 sectors 提供多少信号。

为什么必须在花 API 调用之前跑这一步
----------------------------------
`sector_consistency_strict >= 0.70` 是本项目唯一用来检验"LLM 到底有没有在按方位线
看图"的硬门槛（`AGENTS.md` 规则 1）。但它能不能达到，取决于**几何真值本身的分布**：

* 如果 8 个方位里 70% 都是 "open"，那么"永远回答 open"这种完全不看图的策略
  就能拿到 0.70，门槛形同虚设 —— 而且会让一个没看图的模型混过准入；
* 如果真值分布非常不均衡且集中在极少数类别上，那么一个**认真看图**的模型
  反而可能因为边界情况多而拿不到 0.70，门槛不可达。

两种情况的处置完全不同（前者要收紧门槛，后者要放宽或换指标），
所以必须在生成 320 个样本之前先把这件事量出来。

本脚本给出三个基准：
    majority_global   —— 永远回答全局最常见标签，能拿到的一致率（**偷懒下限**）
    majority_per_az   —— 每个方位都回答该方位的众数、但忽略具体 Tx（**不看图上限**）
    perfect           —— 1.0（真正逐 Tx 读图的理论上限）

同时给出方位剖面的唯一性（决定 `sector_profile_unique_ratio` 好不好过）
与 `tx_local_building_ratio_*` 的取值范围（决定数值字段有没有变化空间）。

用法：
    python scripts/pilot_preflight.py --maps 0:4 --data data/radiomapseer
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
from src.prior.audit import GATES_FAIL  # noqa: E402
from src.prior.schema import AZIMUTHS, OBSTRUCTIONS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="试生成前的几何可行性检查")
    ap.add_argument("--data", default="data/radiomapseer")
    ap.add_argument("--maps", default="0:4", help="地图范围，例如 0:4（不含右端）")
    ap.add_argument("--tx-per-map", type=int, default=80)
    ap.add_argument("--radius-m", type=float, default=100.0)
    ap.add_argument("--px-per-m", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="可选：把结果写成 JSON")
    args = ap.parse_args(argv)

    a, b = (int(x) for x in args.maps.split(":"))
    maps = list(range(a, b))
    data = Path(args.data)
    print(f"数据 {data}｜地图 {maps}｜每图 {args.tx_per_map} 个 Tx｜"
          f"扇区半径 {args.radius_m:g} m")

    # 记录 (map, tx) → 8 个方位的真值标签
    truth: dict[tuple[int, int], list[str]] = {}
    r50s, r100s = [], []
    nblank = 0
    n_tx_on_building = 0
    nearest0 = []

    for m in maps:
        mask = np.array(Image.open(data / "buildings_complete" / f"{m}.png").convert("L"))
        ants = json.loads((data / "antenna" / f"{m}.json").read_text(encoding="utf-8"))
        for t in range(min(args.tx_per_map, len(ants))):
            tx = ants[t]
            geo = geometric_obstruction(mask, tx, radius_m=args.radius_m,
                                       px_per_m=args.px_per_m)
            labels = [str(geo[az]["obstruction"]) for az in AZIMUTHS]
            truth[(m, t)] = labels
            r50s.append(local_building_ratio(mask, tx, 50.0, args.px_per_m))
            r100s.append(local_building_ratio(mask, tx, 100.0, args.px_per_m))
            if all(v == "open" for v in labels):
                nblank += 1
            # Tx 是否恰好落在建筑像素里 —— 这决定"8 方位全 blocked"是不是常态
            ty, txp = int(round(tx[1])), int(round(tx[0]))
            if 0 <= ty < mask.shape[0] and 0 <= txp < mask.shape[1] and mask[ty, txp] > 127:
                n_tx_on_building += 1
            nearest0.append(float(np.min([geo[az]["nearest_building_m"] for az in AZIMUTHS])))

    n_tx = len(truth)
    flat = [v for labels in truth.values() for v in labels]
    print(f"\n样本 {n_tx} 个 Tx × 8 方位 = {len(flat)} 个扇区判定")

    # ---------- 1. 标签边缘分布 ----------
    cnt = Counter(flat)
    print("\n---- 真值标签分布（全部扇区）----")
    for ob in OBSTRUCTIONS:
        c = cnt.get(ob, 0)
        print(f"  {ob:<8} {c:>6}  {c / len(flat):>7.3f}")

    # ---------- 2. 三种策略的期望一致率 ----------
    # 注意：`most_common` 返回的是 (标签, 计数)，必须自己除以总数才是占比。
    # 初版把计数直接当占比打印成 "1519.000"，于是把一个**有区分度**的门槛
    # 误报成"偷懒就能过"—— 结论完全反了。这类错误比崩溃危险得多。
    global_major, global_count = cnt.most_common(1)[0]
    global_share = global_count / len(flat)
    per_az_major = {}
    hit = 0
    for i, az in enumerate(AZIMUTHS):
        c = Counter(labels[i] for labels in truth.values())
        lab, _ = c.most_common(1)[0]
        per_az_major[az] = lab
        hit += sum(1 for labels in truth.values() if labels[i] == lab)
    per_az_acc = hit / len(flat)

    print("\n---- sector_consistency 的三个基准 ----")
    print(f"  永远回答全局众数 '{global_major}'        → {global_share:.3f}   （偷懒下限）")
    print(f"  每方位回答该方位众数、忽略具体 Tx        → {per_az_acc:.3f}   （不看图上限）")
    print(f"  真正逐 Tx 读图（理论上限）               → 1.000")
    thr = GATES_FAIL["sector_consistency_strict"]
    gain_thr = GATES_FAIL["sector_consistency_gain"]
    headroom = 1.0 - max(global_share, per_az_acc)
    print(f"  绝对门槛 = {thr}；偷懒基线与门槛的余量 = {thr - max(global_share, per_az_acc):+.3f}"
          f"（可用区分空间上限 {headroom:.3f}）")
    print(f"\n  另有一条**自校准**门槛 sector_consistency_gain >= {gain_thr}：")
    print(f"    它 = 「与**本 Tx**几何的一致率」−「与**同一地图另一个 Tx**几何的一致率」。")
    print(f"    任何偷懒策略（恒答某标签、或只记方位先验）在两侧得分相同 → 增益恒为 0.000，")
    print(f"    因此即使本批地图的偷懒基线高于 {thr}，这条门槛依然拦得住。")

    verdict = []
    if global_share >= thr:
        verdict.append(
            f"[风险] 只用「永远回答 {global_major}」这一招就能达到 {global_share:.3f} >= {thr}："
            f"该门槛在这一批地图上**几乎没有区分度**，一个完全没看图的模型也能过。"
            f"→ 建议：门槛提高到 {min(0.95, round(per_az_acc + 0.10, 2))} 左右，"
            f"或在报告里同时给出 adjacent 一致率并说明这一点。")
    if per_az_acc >= thr:
        verdict.append(
            f"[风险] 「每方位众数但不看图」就能达到 {per_az_acc:.3f} >= {thr}："
            f"门槛无法区分「按方位读图」与「只记方位先验」。")
    if per_az_acc < thr <= 1.0:
        verdict.append(
            f"[可行] 不看图的上限 {per_az_acc:.3f} < 门槛 {thr}，"
            f"要通过必须真的按 Tx 逐方位读图 —— 门槛有区分度。")

    # ---------- 3. 方位剖面唯一性 ----------
    profs = Counter("|".join(v) for v in truth.values())
    uniq = len(profs)
    print(f"\n---- 方位剖面唯一性 ----")
    print(f"  唯一剖面 {uniq} / {n_tx} = {uniq / max(1, n_tx):.3f}"
          f"（建议级门槛 >= {0.50}）")
    print(f"  最常见剖面占比 {profs.most_common(1)[0][1] / max(1, n_tx):.3f}")
    print(f"  剖面示例：{profs.most_common(1)[0][0]}")
    if uniq / max(1, n_tx) < 0.50:
        verdict.append(f"[注意] 几何本身的剖面唯一率只有 {uniq / max(1, n_tx):.3f}，"
                       f"低于建议级门槛 0.50 —— 这是**数据**的性质，不是 prompt 的错，"
                       f"报告里要如实说明。")

    # ---------- 4. 数值字段的变化空间 ----------
    print("\n---- 局部建筑占比（数值字段的变化空间）----")
    for name, arr in (("tx_local_building_ratio_50m", r50s),
                      ("tx_local_building_ratio_100m", r100s)):
        arr = np.asarray(arr)
        print(f"  {name:<30} min={arr.min():.3f} 中位={np.median(arr):.3f} "
              f"max={arr.max():.3f} 唯一值={len(np.unique(np.round(arr, 3)))}")
    print(f"  8 个方位全为 open 的 Tx：{nblank} / {n_tx} = {nblank / max(1, n_tx):.3f}")

    # ---------- 5. Tx 落点诊断 ----------
    # 如果大量 Tx 直接落在建筑像素里，那么"最近建筑距离 = 0 → blocked"会成为
    # 几何真值的默认答案。这不是判据的错，但它决定了 sector 信息的**上限**：
    # 一个被建筑包住的 Tx，逐方位读图也读不出方向差异。
    near = np.asarray(nearest0)
    print("\n---- Tx 落点与最近建筑距离 ----")
    print(f"  Tx 落在建筑像素内：{n_tx_on_building} / {n_tx} = "
          f"{n_tx_on_building / max(1, n_tx):.3f}")
    print(f"  最近建筑距离（取 8 方位最小值）分位："
          f"p10={np.percentile(near, 10):.1f} p50={np.percentile(near, 50):.1f} "
          f"p90={np.percentile(near, 90):.1f} m")
    print(f"  最近距离 <= 10 m（判 blocked）的比例：{(near <= 10).mean():.3f}")
    if n_tx_on_building / max(1, n_tx) > 0.20:
        print("  [数据性质] 相当一部分 Tx 直接位于建筑轮廓内 —— 这些样本的"
              "8 个方位大概率全判 blocked，逐方位读图在这里没有区分空间。"
              "报告里要说明，并建议单独统计「Tx 不在建筑内」子集的一致率。")

    print("\n---- 结论 ----")
    for v in verdict:
        print("  " + v)
    if not verdict:
        print("  没有发现风险项。")
    if global_share >= thr or per_az_acc >= thr:
        print("\n  **建议在跑全量之前先处理上面的门槛区分度问题**，"
              "否则 0.70 通过并不能证明 LLM 在按方位读图。")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "maps": maps, "n_tx": n_tx, "n_sectors": len(flat),
            "label_distribution": dict(cnt),
            "majority_global": {"label": global_major, "accuracy": global_share},
            "majority_per_azimuth": {str(k): v for k, v in per_az_major.items()},
            "accuracy_majority_per_azimuth": per_az_acc,
            "gate": thr, "sector_profile_unique_ratio": uniq / max(1, n_tx),
            "n_all_open_tx": nblank,
            "verdict": verdict,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
