"""同一批测试样本上的**配对比较**统计。

为什么必须配对（而不是独立样本的 Cohen's d）
------------------------------------------
两个模型跑的是**同一批 test maps、同一批 Tx**（`data/splits.json` 冻结，test 只评一次）。
逐样本的 RMSE 之间存在极强的正相关：建筑掩码复杂、Tx 靠近墙角的样本对**所有模型**都难。
把两组序列当成独立样本会严重高估方差、低估显著性 —— 上一版正是这样用了独立样本的
Cohen's d，把一个稳定的提升算成了"不显著"，于是报告里只挑了 12 个偏难样本说"75% 提升"，
而全量 26% 样本变差被隐去了。

配对比较只看**同一个样本上** a 与 b 的差 `d_i = a_i − b_i`，样本难度被差分抵消：
    * 效应量用 **Cohen's dz** = mean(d) / std(d)，而不是 pooled 的 d；
    * 显著性用 **Wilcoxon 符号秩**（非参数、不假设差值正态）+ **配对 t 检验**（stricter 报告惯例）；
    * 置信区间用 **bootstrap**（对差值重采样），因为差值的分布通常偏斜，正态近似的 CI 不可靠。

口径与可复现性
--------------
* bootstrap 固定 `RandomState(20260101)`、2000 次 —— 同一个输入永远得到同一个 CI，
  否则"CI 是否覆盖 0"会随运行变化，报告失去意义；
* 这里的 `n_a_better / n_b_better` 默认假设**指标越小越好**（RMSE/MAE）。若要比较
  SSIM 这种越大越好的指标，显式传 `lower_is_better=False`，不要靠调用者记在脑子里；
* 长度不一致、差值全为 0、样本量过小都属于**必须显式暴露**的情况：抛错或写进
  `note` 字段，绝不静默返回一个看起来很正常的数字（AGENTS.md 规则 5/6 的同一条精神）。

用法
----
    from src.eval.paired_stats import paired_compare
    r = paired_compare(real_rmse, masked_rmse, ("prior_real", "prior_masked"))
    print(r["mean_diff"], r["ci95_low"], r["ci95_high"], r["wilcoxon_p"])
"""
from __future__ import annotations

import sys
import warnings
from typing import Sequence

import numpy as np
from scipy import stats

try:  # Windows 控制台默认 GBK，统一按 UTF-8 输出，否则中文全是乱码
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

# bootstrap 的种子与次数写死在模块级：报告里的 CI 必须逐位可复现
BOOTSTRAP_SEED = 20260101
BOOTSTRAP_N = 2000
ALPHA = 0.05
# 样本量低于这个数时，非参数检验的功效很差，必须在 note 里说明
MIN_N_FOR_POWER = 8


def _f(x: float) -> float:
    """统一转成 python float（numpy 标量不能直接 json.dumps）。"""
    return float(x)


def paired_compare(
    a: Sequence[float],
    b: Sequence[float],
    names: tuple[str, str] = ("a", "b"),
    lower_is_better: bool = True,
) -> dict:
    """配对比较两组逐样本指标。返回差值 a−b 的统计量字典。

    参数
    ----
    a, b : 同一批样本上的两个逐样本指标序列，**顺序必须一一对应**
           （调用方负责按 sample_id 对齐；本函数不猜顺序）。
    names : 用于 note 与调用方打印的两个名字。
    lower_is_better : 指标是不是"越小越好"。RMSE/MAE=True（默认），SSIM=False。

    返回
    ----
    n, mean_a, mean_b, mean_diff, median_diff, std_diff,
    ci95_low, ci95_high（差值均值的 bootstrap 95% CI）,
    wilcoxon_stat, wilcoxon_p, ttest_stat, ttest_p, cohens_dz,
    n_a_better, n_b_better, n_tie, significant_at_005
    （+ 边界情况下的 "note"）。
    `significant_at_005` 以 **Wilcoxon** 为主判据（配对、非参数，最贴合"同一批样本"这个
    设计）；配对 t 检验的 p 同时给出，供审稿人交叉核对。两个 p 都不可信时（差值为 0、
    样本量 <2）按 1.0 处理并在 note 说明。
    """
    av = np.asarray(list(a), dtype=np.float64)
    bv = np.asarray(list(b), dtype=np.float64)
    if av.size != bv.size:
        raise ValueError(
            f"配对比较要求两条序列长度一致（同一批样本），"
            f"收到 {names[0]}={av.size} vs {names[1]}={bv.size}。"
            f"长度不等说明两侧评的不是同一批样本，任何比较都无意义。"
        )
    if av.size == 0:
        raise ValueError("配对比较收到空序列")
    if not (np.isfinite(av).all() and np.isfinite(bv).all()):
        raise ValueError(
            f"配对序列里存在 NaN/Inf（{names[0]} 非有限值 {int((~np.isfinite(av)).sum())} 个，"
            f"{names[1]} {int((~np.isfinite(bv)).sum())} 个）。"
            f"请先在数据源层决定如何处理缺失样本，不要在这里静默丢弃。"
        )

    n = int(av.size)
    diff = av - bv
    notes: list[str] = []

    mean_diff = _f(diff.mean())
    std_diff = _f(diff.std(ddof=1)) if n > 1 else 0.0
    median_diff = _f(np.median(diff))

    # ---- bootstrap CI：对差值重采样，得到"差值均值"的 95% 区间 ----
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    boot = np.empty(BOOTSTRAP_N, dtype=np.float64)
    for i in range(BOOTSTRAP_N):
        idx = rng.randint(0, n, size=n)
        boot[i] = diff[idx].mean()
    ci95_low = _f(np.percentile(boot, 100 * ALPHA / 2))
    ci95_high = _f(np.percentile(boot, 100 * (1 - ALPHA / 2)))

    # ---- 方向计数：默认"越小越好"，SSIM 这类要传 lower_is_better=False ----
    if lower_is_better:
        n_a_better = int((diff < 0).sum())
        n_b_better = int((diff > 0).sum())
    else:
        n_a_better = int((diff > 0).sum())
        n_b_better = int((diff < 0).sum())
    n_tie = int((diff == 0).sum())
    if n_tie:
        notes.append(f"{n_tie} 个样本两侧完全相等（已计入 n_tie，不参与 better 计数）")

    # ---- 配对检验 ----
    all_zero = bool(np.all(diff == 0))
    wilcoxon_stat, wilcoxon_p = 0.0, 1.0
    ttest_stat, ttest_p = 0.0, 1.0
    cohens_dz = 0.0

    if n < 2:
        notes.append("样本量 <2，配对检验无法进行（统计量按 0、p 按 1.0 记）")
    elif all_zero:
        # 全 0 差值时 wilcoxon 会直接抛 ValueError，ttest_rel 会返回 nan 并告警。
        # 这不是"无差异的强证据"，而是"检验失效"，所以必须显式写进 note。
        notes.append(
            "所有样本的差值都为 0：Wilcoxon/配对 t 检验在此退化（p 记为 1.0，"
            "不代表'无差异有证据'，只代表'无法检验'）"
        )
    else:
        try:
            # zero_method="wilcox"：丢弃 0 差值（Pratt 方法的默认行为反而会算错秩）
            w = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            wilcoxon_stat, wilcoxon_p = _f(w.statistic), _f(w.pvalue)
            if not np.isfinite(wilcoxon_p):
                wilcoxon_p = 1.0
                notes.append("Wilcoxon 返回非有限 p，按 1.0 记")
        except Exception as exc:  # noqa: BLE001
            wilcoxon_p = 1.0
            notes.append(f"Wilcoxon 检验失败（{type(exc).__name__}: {exc}），p 按 1.0 记")
        try:
            with warnings.catch_warnings():  # 差值方差为 0 时 scipy 会告警，我们已显式处理
                warnings.simplefilter("ignore", RuntimeWarning)
                t = stats.ttest_rel(av, bv)
            finite_t = bool(np.isfinite(t.statistic))
            ttest_stat = _f(t.statistic) if finite_t else 0.0
            ttest_p = _f(t.pvalue) if np.isfinite(t.pvalue) else 1.0
            if not finite_t:
                # 所有差值逐位相同时 t=±inf、p=0.0。把它当"极显著"是典型的数字陷阱
                # （n=3 的完全相同差值也会给出 p=0），所以这里按 1.0 记并说明，
                # 主判据仍是 Wilcoxon。
                ttest_p = 1.0
                notes.append(
                    "所有差值逐位相同（差值方差为 0）：配对 t 检验退化（t=±inf，"
                    "p 按 1.0 记），请以 Wilcoxon p 为准"
                )
        except Exception as exc:  # noqa: BLE001
            ttest_p = 1.0
            notes.append(f"配对 t 检验失败（{type(exc).__name__}: {exc}），p 按 1.0 记")
        if std_diff > 0:
            cohens_dz = _f(mean_diff / std_diff)
        else:
            cohens_dz = 0.0
            notes.append("差值标准差为 0：Cohen's dz 无定义，按 0.0 记")

    if n < MIN_N_FOR_POWER:
        notes.append("样本量偏小")

    out: dict = {
        "names": [names[0], names[1]],
        "n": n,
        "mean_a": _f(av.mean()),
        "mean_b": _f(bv.mean()),
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "std_diff": std_diff,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p,
        "ttest_stat": ttest_stat,
        "ttest_p": ttest_p,
        "cohens_dz": cohens_dz,
        "n_a_better": n_a_better,
        "n_b_better": n_b_better,
        "n_tie": n_tie,
        "significant_at_005": bool(wilcoxon_p < ALPHA),
        "significant_test": "wilcoxon",
        "lower_is_better": bool(lower_is_better),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "n_resamples": BOOTSTRAP_N,
                      "ci": 1.0 - ALPHA},
    }
    if notes:
        out["note"] = "；".join(notes)
    return out


def format_result(r: dict, unit: str = "dB") -> str:
    """把结果打印成一行中文摘要（供 __main__ 与 report.py 复用）。"""
    na, nb = r["names"]
    sig = "显著" if r["significant_at_005"] else "不显著"
    line = (
        f"{na} − {nb}：均值 {r['mean_diff']:+.3f} {unit}"
        f"（中位 {r['median_diff']:+.3f}，std {r['std_diff']:.3f}）"
        f"，95% CI [{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]"
        f"，dz={r['cohens_dz']:+.3f}，Wilcoxon p={r['wilcoxon_p']:.4g}"
        f"（配对 t p={r['ttest_p']:.4g}）→ {sig}（n={r['n']}，"
        f"{na} 更优 {r['n_a_better']} / {nb} 更优 {r['n_b_better']} / 平 {r['n_tie']}）"
    )
    if r.get("note"):
        line += f"\n  注意：{r['note']}"
    return line


def _selfcheck() -> int:
    """自检：造两组已知差异的数据，验证统计量方向与边界处理。"""
    rng = np.random.RandomState(7)
    n = 200
    base = rng.uniform(3.0, 12.0, size=n)          # 每个样本的"固有难度"
    noise = rng.normal(0, 0.5, size=n)
    b = base                                        # prior_masked
    a = base - 1.5 + noise                          # prior_real：平均好 1.5 dB
    r = paired_compare(a, b, ("real", "masked"))
    print("[自检 1] 已知真实差异 −1.5 dB（配对）：")
    print("  " + format_result(r).replace("\n", "\n  "))
    assert r["n"] == n
    assert r["mean_diff"] < -1.0, "配对差值应接近 −1.5"
    assert r["ci95_high"] < 0, "CI 应完全落在 0 以下"
    assert r["significant_at_005"], "这种信噪比下应显著"
    assert r["n_a_better"] > 150 and r["n_b_better"] < 50
    assert abs(r["cohens_dz"]) > 0.5, "配对 dz 应远大于独立样本 d（这是配对的意义）"

    # 同一个数据若错当成独立样本，独立样本 t 检验的 p 会大得多 —— 证明必须配对
    indep = stats.ttest_ind(a, b, equal_var=False)
    print(f"[自检 1] 同一数据的**非配对** Welch t 检验 p={indep.pvalue:.3g}"
          f"（配对 Wilcoxon p={r['wilcoxon_p']:.3g}）：非配对会浪费掉样本难度相关性")

    # 全 0 差值：必须走 note 分支而不是抛错
    z = paired_compare([1.0] * 10, [1.0] * 10, ("x", "y"))
    print("\n[自检 2] 全 0 差值：")
    print(f"  wilcoxon_p={z['wilcoxon_p']} ttest_p={z['ttest_p']} "
          f"significant={z['significant_at_005']} note={z.get('note')}")
    assert z["wilcoxon_p"] == 1.0 and z["ttest_p"] == 1.0
    assert "note" in z and z["n_tie"] == 10

    # 样本量偏小：必须有 note
    s = paired_compare([1.0, 2.0, 3.0], [1.5, 2.5, 3.5], ("a", "b"))
    print("\n[自检 3] 小样本：")
    print("  " + format_result(s).replace("\n", "\n  "))
    assert "样本量偏小" in s.get("note", "")
    assert s["ttest_p"] == 1.0, "差值方差为 0 时 t 检验退化，不能报 p=0（会像'极显著'）"

    # 长度不一致：必须抛 ValueError
    try:
        paired_compare([1.0, 2.0], [1.0], ("a", "b"))
    except ValueError as exc:
        print(f"\n[自检 4] 长度不一致已抛 ValueError：{exc}")
    else:
        raise AssertionError("长度不一致必须抛 ValueError")

    # 越大越好的指标方向不能搞反
    hi = paired_compare([0.9] * 20, [0.8] * 20, ("ssim_a", "ssim_b"), lower_is_better=False)
    print(f"\n[自检 5] SSIM 口径（越大越好）：a_better={hi['n_a_better']} "
          f"b_better={hi['n_b_better']} mean_diff={hi['mean_diff']:+.3f}")
    assert hi["n_a_better"] == 20 and hi["n_b_better"] == 0

    print("\n配对统计自检全部通过 [OK]")
    return 0


if __name__ == "__main__":
    sys.exit(_selfcheck())
