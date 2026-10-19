#!/usr/bin/env python3
""""LLM 是不是只是在再编码给定几何"——用可解释方差来判定。

要回答的问题
-----------
prompt_v3 把程序算出的**精确几何**作为事实喂给 LLM，LLM 输出电磁字段。
接地检验已表明 LOS 与反射确实随几何变化（ρ=+0.72 / −0.33，远高于错位对照）。
但"随几何变化"还不够 —— 如果 LLM 输出的只是这些几何量的一个（可学习的）函数，
那么**把几何直接喂给下游模型会严格更好，LLM 这一环的边际贡献 ≈ 0**。

判定方法
-------
用几何特征去**拟合** LLM 的每个电磁字段，看交叉验证能解释多少方差（R²）：
  * R² 高（如 ≥ 0.80）→ LLM 输出基本可由几何复现 ⇒ 它就是再编码器，不值得做训练对照
  * R² 低（如 ≤ 0.50）→ LLM 提供了几何之外的东西 ⇒ 值得做训练对照
同时给三个层次的特征集，区分"几个标量就够"与"必须用完整方位剖面"：
  S1  少量标量：r50, r100, Tx是否在建筑内, 被挡方位数, 最近建筑距离
  S2  S1 + 8 方位最近距离
  S3  完整几何：S2 + 8 方位遮挡等级 one-hot
两个模型：
  Ridge（线性，含标准化）与 GradientBoosting（非线性上限估计）。
**错位对照**：把目标打乱后跑同一套流程，R² 必须 ≈0 或为负，否则说明流程本身有泄漏。

用法：
    python scripts/probe_llm_distillability.py --priors priors/prompt_v3/glm-4v-flash
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
from src.prior.schema import AZIMUTHS, EM_NUM_KEYS, OBSTRUCTIONS  # noqa: E402

DATA_ROOT = ROOT / "data" / "radiomapseer"
# 物理上应当随 Tx 位置变化的字段（材料属性恒定是常识，不参与判定）
GEO_CONDITIONED = ("los_ratio", "reflection_ratio", "diffraction_ratio",
                   "scattering_ratio", "pathloss_min_db", "pathloss_max_db")


def build_dataset(priors_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    """返回 (特征矩阵, {字段: 目标}, 样本 id 列表)。几何在这里**重新精确计算**。"""
    feats: list[list[float]] = []
    targets: dict[str, list[float]] = {k: [] for k in EM_NUM_KEYS}
    sids: list[str] = []
    for f in sorted(priors_dir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        p = d.get("prior")
        if not p:
            continue
        m, t = int(d["map_id"]), int(d["tx_id"])
        mask = np.array(Image.open(DATA_ROOT / "buildings_complete" / f"{m}.png").convert("L"))
        ants = json.loads((DATA_ROOT / "antenna" / f"{m}.json").read_text(encoding="utf-8"))
        if t >= len(ants):
            continue
        tx = ants[t]
        geo = geometric_obstruction(mask, tx, radius_m=100.0, px_per_m=1.0)
        labs = [str(geo[a]["obstruction"]) for a in AZIMUTHS]
        dists = [min(float(geo[a]["nearest_building_m"]), 500.0) for a in AZIMUTHS]
        r50 = float(local_building_ratio(mask, tx, 50.0))
        r100 = float(local_building_ratio(mask, tx, 100.0))
        ty, txp = int(round(tx[1])), int(round(tx[0]))
        on_b = float(mask[ty, txp] > 127)
        finite = [d2 for d2 in dists if d2 < 400]
        # S1：少量标量
        row = [r50, r100, on_b, float(labs.count("blocked")), min(dists)]
        # S2：+ 8 方位距离
        row += dists
        # S3：+ 8 方位遮挡等级 one-hot
        for a_idx in range(len(AZIMUTHS)):
            for ob in OBSTRUCTIONS:
                row.append(1.0 if labs[a_idx] == ob else 0.0)
        feats.append(row)
        for k in EM_NUM_KEYS:
            targets[k].append(float(p.get(k, np.nan)))
        sids.append(d["sample_id"])
    X = np.asarray(feats, dtype=np.float64)
    Y = {k: np.asarray(v, dtype=np.float64) for k, v in targets.items()}
    return X, Y, sids


def cv_r2(X: np.ndarray, y: np.ndarray, model: str, seed: int = 0) -> float:
    """5 折交叉验证的 R²（用 cross_val_predict 汇总后统一算，比逐折平均更稳）。"""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if model == "ridge":
        est = make_pipeline(StandardScaler(),
                            RidgeCV(alphas=np.logspace(-3, 3, 13)))
    else:
        est = GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                        learning_rate=0.05, random_state=seed)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    pred = cross_val_predict(est, X, y, cv=kf)
    return float(r2_score(y, pred))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LLM 输出可否由给定几何复现（可解释方差）")
    ap.add_argument("--priors", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    X, Y, sids = build_dataset(Path(args.priors))
    n, nfeat = X.shape
    print(f"样本 {n}｜完整几何特征 {nfeat} 维（S1 5 / S2 13 / S3 45）\n")
    if n < 30:
        print(f"样本量 {n} 偏小，R² 的置信区间会很宽；建议用 ≥ 100 个样本。\n")

    sets = {"S1 少量标量": slice(0, 5),
            "S2 +8方位距离": slice(0, 13),
            "S3 完整几何": slice(0, 45)}
    rng = np.random.RandomState(0)

    print("=" * 92)
    print(f"{'LLM 字段':<20}{'目标std':>8} | " +
          " | ".join(f"{k:>16}" for k in sets) + " |  判定")
    print("=" * 92)
    rows = []
    for k in EM_NUM_KEYS:
        y = Y[k]
        ok = np.isfinite(y)
        yv, Xv = y[ok], X[ok]
        std = float(yv.std())
        r2s = {}
        for sname, sl in sets.items():
            r2s[sname] = cv_r2(Xv[:, sl], yv, "ridge")
        r2s["gbm_S3"] = cv_r2(Xv[:, :45], yv, "gbm")
        # 错位对照：打乱目标后 S3+Ridge 的 R² 应 ≈0 或负
        y_sh = rng.permutation(yv)
        r2_sh = cv_r2(Xv[:, :45], y_sh, "ridge")

        # 只对"应当随位置变化"的字段下判定；材料属性恒定是常识
        if k in GEO_CONDITIONED:
            best = max(r2s["S1 少量标量"], r2s["S2 +8方位距离"], r2s["S3 完整几何"], r2s["gbm_S3"])
            if std < 1e-6:
                verdict = "常量（无方差可解释）"
            elif best >= 0.80:
                verdict = f"**可由几何复现** (best R²={best:.2f}) → 再编码"
            elif best >= 0.50:
                verdict = f"部分可复现 (best R²={best:.2f}) → 可能有残余信息"
            else:
                # 关键：低 R² **不能**推断"有独立信息"。噪声同样无法由几何复现，
                # 所以必须与接地检验（probe_llm_grounding.py）联判：
                #   低 R² + 高 |ρ| → 真的随几何（只是含散点、非线性）
                #   低 R² + 低 |ρ| → 与几何无关 = 噪声；方差越大反而越糟
                #                     （把噪声注入条件向量，下游模型还得学会忽略它）
                verdict = (f"几何解释不了 (best R²={best:.2f}) → **须与接地检验联判**："
                           f"低 R² 与'噪声'同样相容")
        else:
            verdict = "材料属性（恒定属常识，不判定）"
        rows.append({"field": k, "std": std, "r2": r2s, "r2_shuffled": r2_sh,
                     "verdict": verdict})
        print(f"{k:<20}{std:>8.3f} | " +
              " | ".join(f"{r2s[s]:>16.2f}" for s in sets) +
              f" |  {verdict}")
        print(f"{'':<20}{'':>8} | GBM(S3) {r2s['gbm_S3']:>6.2f}"
              f"｜错位对照 {r2_sh:>6.2f}"
              f"{'  ← 正常' if r2_sh <= 0.05 else '  ← **异常，可能有泄漏**'}")

    print("\n" + "=" * 92)
    gated = [r for r in rows if r["field"] in GEO_CONDITIONED]
    high = [r for r in gated if max(r["r2"].values()) >= 0.80]
    low = [r for r in gated if max(r["r2"].values()) < 0.50]
    print(f"几何条件字段 {len(gated)} 个：可复现(R²≥0.80) {len(high)} 个"
          f"｜几何解释不了(R²<0.50) {len(low)} 个")
    print("\n结论（**必须与 probe_llm_grounding.py 联判，本测试单独不足以定论**）：")
    print("  低 R² 只说明'几何特征拟合不出来'，它同时与两件事相容：")
    print("    (i) LLM 提供了几何之外的真信息；")
    print("    (ii) LLM 在胡编 —— 噪声当然也无法由几何复现。")
    print("  区分办法：看接地检验的秩相关 ρ。")
    print("    低 R² + 高|ρ| → 真的随几何（含散点/非线性），是弱但真实的信息；")
    print("    低 R² + ρ≈0  → 与几何无关。若此时方差还很大，那不是信息而是**幻觉**：")
    print("                   它的数值在乱跳却不反映场景，下游模型只能学会忽略它。")
    if len(high) >= len(gated) * 0.75:
        print("\n  → LLM 输出基本可由几何复现：它是再编码器，LLM 的边际贡献很可能≈0。")
        print("     建议只训 3 组即可定论：")
        print("       prior_geometry_only / prior_v3_masked_em（几何真、电磁零）/ prior_real_v3")
    else:
        print(f"\n  → {len(sets)} 个特征集、两个模型都拟合不出这些字段，说明**不是**"
              f"'几何的简单函数'。")
        print("     但这**不等于**有信息。请对照 reports/llm_grounding_*.json 的 ρ 值逐字段判读。")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"n": n, "n_features": nfeat, "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
