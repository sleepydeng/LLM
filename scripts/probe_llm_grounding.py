#!/usr/bin/env python3
"""测量 LLM 的电磁字段到底有没有"接地"在它拿到的几何事实上。

为什么这是决定性的
-----------------
v3 设计把精确几何作为**事实**喂给 LLM，只让它做电磁推理。这个设计只有在
"LLM 的输出确实随那些事实变化"时才有意义。否则它就是在输出一个模板，
先验不携带任何来自几何的信息，而**直接把几何喂给模型会严格更好**。

判定方法：把每个 LLM 电磁数值字段与几何驱动量做 **Spearman 秩相关**，
并同时算一个**错位对照**（与该字段所属样本无关的另一个样本的几何量）。
- 真相关明显高于错位相关  → 模型在用这些事实（接地成立）
- 两者都接近 0            → 模型没在用（路线 A 的 LLM 贡献为零，是个负面结果）

用秩相关而不是 Pearson：我们只关心"单调方向对不对"，且不假设线性。

用法：
    python scripts/probe_llm_grounding.py --priors priors/<...>
"""
from __future__ import annotations

import argparse
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

from src.data.sectors import geometric_obstruction, local_building_ratio  # noqa: E402
from src.prior.schema import AZIMUTHS, EM_NUM_KEYS  # noqa: E402

DATA_ROOT = ROOT / "data" / "radiomapseer"

# 几何驱动量：LLM 拿到的"事实"就是这些
DRIVERS = ("r50", "r100", "n_blocked", "n_dense", "n_sparse", "min_dist", "mean_dist")

# **物理上有动机**的配对：只检验这些。
# 为什么必须限制：把 56 个"字段×驱动量"组合（含 permittivity vs 最近距离这类
# 物理上无关的组合）一起取平均，会把真信号稀释掉 —— 初版就是这么做的，
# 于是明明 los_ratio↔最近距离有 ρ=+0.82 的强相关，汇总却只显示 0.268 vs 0.228。
# 检验应当是"预先指定假设"，而不是"扫一遍再挑最显著的"。
MOTIVATED_PAIRS: tuple[tuple[str, str, str], ...] = (
    # (LLM 字段, 几何驱动量, 预期符号)
    ("los_ratio", "min_dist", "+"),        # 建筑越远 → 直射机会越多
    ("los_ratio", "n_blocked", "-"),       # 被挡的方位越多 → 直射越少
    ("los_ratio", "r100", "-"),            # 越密 → 直射越少
    ("reflection_ratio", "r100", "+"),     # 越密 → 反射越多
    ("reflection_ratio", "min_dist", "-"),  # 建筑越近 → 反射越多
    ("diffraction_ratio", "n_blocked", "+"),  # 遮挡越多 → 衍射越重要
    ("diffraction_ratio", "min_dist", "-"),
    ("scattering_ratio", "r100", "+"),     # 越密/越粗糙 → 散射越多
    ("pathloss_max_db", "n_blocked", "+"),  # 遮挡越多 → 最大路损越大
    ("pathloss_max_db", "min_dist", "-"),
)


def _finite_corr(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    return spearman(x[m], y[m])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """秩相关。样本太少或全为常数时返回 nan。"""
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def geometry_drivers(mask: np.ndarray, tx) -> dict[str, float]:
    geo = geometric_obstruction(mask, tx, radius_m=100.0)
    labs = [str(geo[a]["obstruction"]) for a in AZIMUTHS]
    dists = [float(geo[a]["nearest_building_m"]) for a in AZIMUTHS]
    finite = [d for d in dists if d <= 400]
    return {
        "r50": float(local_building_ratio(mask, tx, 50.0)),
        "r100": float(local_building_ratio(mask, tx, 100.0)),
        "n_blocked": float(labs.count("blocked")),
        "n_dense": float(labs.count("dense")),
        "n_sparse": float(labs.count("sparse")),
        "min_dist": float(min(dists)),
        "mean_dist": float(np.mean(finite)) if finite else 500.0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LLM 电磁字段是否接地在给定几何上")
    ap.add_argument("--priors", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    files = sorted(Path(args.priors).glob("*.json"))
    rows = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        p = d.get("prior")
        if not p:
            continue
        m, t = int(d["map_id"]), int(d["tx_id"])
        mask = np.array(Image.open(DATA_ROOT / "buildings_complete" / f"{m}.png").convert("L"))
        ants = json.loads((DATA_ROOT / "antenna" / f"{m}.json").read_text(encoding="utf-8"))
        if t >= len(ants):
            continue
        drv = geometry_drivers(mask, ants[t])
        rows.append({"sid": d["sample_id"], "map": m, "tx": t,
                     **{k: float(p.get(k, np.nan)) for k in EM_NUM_KEYS},
                     "scene_type": str(p.get("scene_type")),
                     "dominant_material": str(p.get("dominant_material")),
                     **drv})
    n = len(rows)
    if n < 5:
        print(f"样本太少（{n}），无法判定")
        return 2
    print(f"样本 {n} 个\n")

    # LLM 字段自身的唯一值数（先看有没有变化）
    print("---- LLM 电磁字段的取值情况 ----")
    for k in EM_NUM_KEYS:
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        print(f"  {k:<22} unique={len(set(np.round(v, 6))):<4}"
              f"min={v.min():<8.2f} max={v.max():<8.2f} std={v.std():<7.3f}")
    for k in ("scene_type", "dominant_material"):
        vals, cnt = np.unique([r[k] for r in rows], return_counts=True)
        print(f"  {k:<22} unique={len(vals):<4} {dict(zip(vals.tolist(), cnt.tolist()))}")

    # ---- 接地检验：只查**预先指定**的物理配对 ----
    # n 小的时候 Spearman 的标准误约 1/sqrt(n-1)，必须一起报出来，
    # 否则会把"没有功效"误读成"没有效应"。
    se = 1.0 / np.sqrt(max(1, n - 1))
    print(f"\n---- 接地检验（预先指定的物理配对）----")
    print(f"     n={n}，Spearman 的标准误约 {se:.3f}；"
          f"|真实 ρ − 错位 ρ| 需明显大于此值才算有信号")
    print(f"{'LLM 字段':<22}{'驱动量':<12}{'预期':<6}{'真实ρ':>8}{'错位ρ':>8}{'差值':>8}{'方向':>6}")
    rng = np.random.RandomState(20260101)
    perm = np.roll(rng.permutation(n), 1)      # 无不动点的错位
    hits, tested, deltas, dir_ok = 0, 0, [], 0
    for field, drv, sign in MOTIVATED_PAIRS:
        real = np.array([r[field] for r in rows], dtype=float)
        g = np.array([r[drv] for r in rows], dtype=float)
        gs = g[perm]
        rho = _finite_corr(real, g)
        rho_s = _finite_corr(real, gs)
        if not np.isfinite(rho):
            continue
        tested += 1
        d = rho - (rho_s if np.isfinite(rho_s) else 0.0)
        deltas.append(abs(d))
        want = 1.0 if sign == "+" else -1.0
        ok_dir = (np.sign(rho) == want)
        dir_ok += int(ok_dir)
        # "命中"= 方向正确 且 真实相关比错位明显更强
        hit = bool(ok_dir and abs(d) > 2 * se)
        hits += int(hit)
        print(f"{field:<22}{drv:<12}{sign:<6}{rho:>8.3f}"
              f"{(rho_s if np.isfinite(rho_s) else float('nan')):>8.3f}{d:>8.3f}"
              f"{('对' if ok_dir else '反'):>6}{'  *' if hit else ''}")
    print(f"\n  方向正确 {dir_ok}/{tested}；同时显著强于错位对照 {hits}/{tested}")
    if tested == 0:
        print("  无法检验（数据不足）")
        return 2
    power_ok = n >= 60
    if hits >= max(3, tested // 3) and dir_ok >= tested * 0.6:
        verdict = (f"接地成立（{hits}/{tested} 个物理配对方向正确且明显强于错位对照）："
                   f"LLM 的电磁字段确实随给定几何变化")
    elif not power_ok:
        verdict = (f"**功效不足，无法下结论**：n={n}，标准误 {se:.2f}。"
                   f"方向正确 {dir_ok}/{tested}、显著 {hits}/{tested}。"
                   f"需要更大的试生成（建议 ≥ 120 样本，maps 0-2 全 80 Tx 即可）再判定。")
    else:
        verdict = (f"接地不足：仅 {hits}/{tested} 个物理配对成立、方向正确 {dir_ok}/{tested}。"
                   f"此时把几何直接喂给下游模型会更好，LLM 的边际贡献接近零（负面结果）")
    print(f"  结论：{verdict}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "n": n, "spearman_se": se, "tested_pairs": tested,
            "direction_correct": dir_ok, "significant_vs_shuffled": hits,
            "verdict": verdict,
            "motivated_pairs": [{"field": f, "driver": d, "expected_sign": s}
                                for f, d, s in MOTIVATED_PAIRS]},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
