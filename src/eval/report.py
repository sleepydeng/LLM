"""把 `runs/*/metrics_*.json` 汇总成 markdown / JSON 报告 —— **报告里每个数字的唯一来源**。

为什么这个文件必须存在（AGENTS.md 规则 5）
----------------------------------------
上一版报告的数字是手抄的：正文写"第 25 epoch 领先"，图上第 21 epoch 就交叉了；
正文只报 12 个偏难样本的"75% 提升"，全量 26% 样本变差被隐去。手抄数字的错误无法被
自动发现，而"脚本从 runs/*/metrics_*.json 生成"可以被任何人重跑复现。所以这里的每一条
数字都是读文件算出来的，缺失一律写 **未记录 / 未跑**，**绝不补默认值、绝不编造**。

读哪些文件
--------
`src/eval/run_eval.py` 的产物是 `runs/<exp_id>/metrics_<split>.json`（例如
`metrics_test.json`），所以这里同时接受 `metrics.json` 与 `metrics_*.json`；
**同一个目录里只取一个**（优先精确的 `metrics.json`），其余会被列在"被忽略的文件"里
并打印警告 —— 同一次实验的两份 metrics 混进主表会把 n_seeds 算错。

三件必须做对的事
--------------
1. **多 seed**：一个 config_name 通常有 3 个 seed（结论要求 mean ± std）。
   所以先按 config_name 分组，再在 seed 维上取均值/标准差，而不是把 seed 当独立实验。
2. **配对比较**：`prior_real` 与 `prior_masked/shuffled` 跑的是同一批 test maps
   （test 只评一次、划分冻结），逐样本指标强相关，必须用 `paired_stats.paired_compare`
   做配对统计。多 seed 时先按 `sample_id` 对齐取 seed 平均，再配对 —— 口径写进报告。
3. **不选择性呈现**：第 4 节同时给出"最好 / 最差"样本和**变差样本的全量占比**，
   防止再出现"只报挑出来的子集"这种事。

退出码
-----
0  正常
2  `--runs` 下**没有任何** metrics.json / metrics_*.json
3  有 metrics 文件但解析失败（仍会输出报告，并在来源表里标红；绝不静默跳过文件）

CLI
---
    python -m src.eval.report --runs runs --out reports/summary.md \
        [--json reports/summary.json] [--baseline prior_masked]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

try:  # Windows 控制台默认 GBK，统一按 UTF-8 输出，否则中文报告全是乱码
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.paired_stats import BOOTSTRAP_N, BOOTSTRAP_SEED, paired_compare  # noqa: E402

# 消融阶梯有两套命名（v1 与 v3），用 `--ladder` 选择；缺失的配置明确写"未跑"。
# v1：prior_masked 把**整个** 64 维置零（含程序几何）。
# v3：路线 A 的分工是"空间来自程序、电磁来自 LLM"，所以容量对齐的对照必须
#     保留程序几何、只把 LLM 的 22 维置零 —— 即 prior_v3_masked_em。
#     两者混用会把"几何被置零"的损失算到 LLM 头上（AGENTS.md 的归因错误）。
LADDER_PROFILES = {
    "v1": {
        "order": ("prior_none", "prior_masked", "prior_real", "prior_shuffled"),
        "deltas": (("prior_real", "prior_masked"), ("prior_real", "prior_shuffled")),
        "primary": "prior_real",               # 主实验配置名
        "default_baseline": "prior_masked",    # 容量对齐的零信息对照
        "none_name": "prior_none",
        "shuffled_name": "prior_shuffled",
    },
    "v3": {
        "order": ("prior_none", "prior_geometry_only", "prior_v3_masked_em",
                  "prior_real_v3", "prior_real_v3_shuffled"),
        "deltas": (("prior_real_v3", "prior_v3_masked_em"),
                   ("prior_real_v3", "prior_real_v3_shuffled"),
                   ("prior_geometry_only", "prior_none")),
        "primary": "prior_real_v3",
        "default_baseline": "prior_v3_masked_em",
        "none_name": "prior_none",
        "shuffled_name": "prior_real_v3_shuffled",
        # 补充对照（阳性对照 + 容量天花板），也进配对表与参数量对齐检查：
        "extra_pairs": (("prior_real_v3_oracle", "prior_v3_masked_em"),
                        ("prior_none_big", "prior_none")),
    },
}

MISSING = "未记录"
NOT_RUN = "未跑"


# ------------------------------------------------------------------ 工具

def _fmt(v: Any, nd: int = 2) -> str:
    """数字格式化：非有限值/None 一律写"未记录"，不写 nan（那是伪造的可读性）。

    字符串直接原样返回 —— 这个函数被用在 `config_name` / `weights` 这类字段上过，
    早期版本会把它们误判成"未记录"，属于静默丢信息，所以显式支持字符串。
    """
    if v is None:
        return MISSING
    if isinstance(v, str):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return MISSING
    if not math.isfinite(f):
        return MISSING
    s = f"{f:.{nd}f}"
    if float(s) == 0.0 and f != 0.0:
        # 例如 −1.2e−05 用 .3f 会显示成 "-0.000"（看起来"完全没差别"），
        # 而 Wilcoxon 的 p 又可能显著 —— 这种显示会误导，所以改用科学计数法。
        return f"{f:.2e}"
    return s


def _mean_std(vals: Sequence[float], nd: int = 2) -> str:
    """mean ± std(ddof=1)。只有 1 个值时不写 ±0.00（那会假装方差已知）。"""
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not xs:
        return MISSING
    if len(xs) == 1:
        return f"{xs[0]:.{nd}f} ± —"
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return f"{mean:.{nd}f} ± {math.sqrt(var):.{nd}f}"


def _mean_std_nums(vals: Sequence[float]) -> dict:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not xs:
        return {"mean": None, "std": None, "n": 0}
    mean = sum(xs) / len(xs)
    std = 0.0
    if len(xs) > 1:
        std = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))
    return {"mean": mean, "std": (std if len(xs) > 1 else None), "n": len(xs)}


def _jsonable(obj: Any) -> Any:
    """把 numpy 标量 / NaN / Inf 清理成严格 JSON 能接受的形式（NaN→null）。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (int,)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if hasattr(obj, "item"):  # numpy 标量
        return _jsonable(obj.item())
    return str(obj)


def _md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------ 读取

def find_metric_files(runs_dir: Path) -> tuple[list[Path], list[dict]]:
    """找出所有 metrics 文件。返回 (采用的文件, 被忽略的文件记录)。

    同一个目录里同时存在 `metrics.json` 与 `metrics_test.json` 时只取前者：
    同一次实验的两份文件都进主表，会让"n_seeds"和 config 的均值/std 全部算错，
    而这种错误在表面上看不出任何异常（数字都"合理"），所以必须显式去重 + 打印。
    """
    chosen: dict[Path, Path] = {}
    ignored: list[dict] = []
    # 先扫精确名，再扫 metrics_*.json，保证精确名优先
    for pattern in ("metrics.json", "metrics_*.json"):
        for p in sorted(runs_dir.rglob(pattern)):
            d = p.parent
            if d in chosen:
                if chosen[d] != p:
                    ignored.append({"path": str(p), "used_instead": str(chosen[d]),
                                    "reason": "同一目录下已有被采用的 metrics 文件"})
                continue
            chosen[d] = p
    return [chosen[d] for d in sorted(chosen)], ignored


def load_runs(runs_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """扫描 runs 目录下的 metrics 文件，返回 (可用 run, 解析失败, 被忽略)。

    用 rglob 而不是 glob("*/metrics.json")：既支持 `runs/<exp>/metrics_test.json`，
    也支持 `runs/<config>/<seed>/metrics_test.json` 这种按 seed 建子目录的布局。

    **解析失败不静默跳过**：记进 bad 列表，调用方会把它写进报告并让退出码非 0。
    悄悄少统计一个 run 会让主表看起来"少了几个 config"却没人知道原因。
    """
    records: list[dict] = []
    bad: list[dict] = []
    files, ignored = find_metric_files(runs_dir)
    for mpath in files:
        try:
            raw = json.loads(mpath.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("metrics.json 顶层必须是对象")
            rec = _normalize_record(raw, mpath)
            records.append(rec)
        except Exception as exc:  # noqa: BLE001
            bad.append({"path": str(mpath), "error": f"{type(exc).__name__}: {exc}"})
    return records, bad, ignored


def _split_from_filename(name: str) -> str | None:
    """`metrics_test.json` → `test`（仅在 metrics.json 里没有 split 字段时的兜底）。"""
    m = re.fullmatch(r"metrics_([A-Za-z0-9]+)\.json", name)
    return m.group(1) if m else None


def _normalize_record(raw: dict, mpath: Path) -> dict:
    """把 metrics.json 归一化成一个内部记录（缺失字段一律 None，不填默认值）。"""
    meta = raw.get("meta") or {}
    agg = raw.get("aggregate") or {}
    exp_id = raw.get("exp_id") or mpath.parent.name
    config_name = raw.get("config_name")
    if not config_name:
        # 没有 config_name 时从 meta.prior_mode 反推，再不行从 exp_id 去掉 seed 后缀。
        # 这是"尽量给出可用分组"，但会在报告里标注来源，不假装它是原始字段。
        mode = meta.get("prior_mode")
        config_name = f"prior_{mode}" if mode else _strip_seed_suffix(exp_id)
    # split 优先用记录里的值；run_eval 会写 split，但手工产出的 metrics.json 可能没写。
    # 从文件名（metrics_test.json）兜底时必须标注"推导值"，否则读者会以为它是原始记录。
    split = raw.get("split")
    split_source = "recorded" if split else None
    if not split:
        split = _split_from_filename(mpath.name)
        split_source = f"derived_from_filename({mpath.name})" if split else None
    per_sample: dict[str, dict] = {}
    for row in raw.get("per_sample") or []:
        sid = row.get("sample_id")
        if sid is None:
            continue
        per_sample[str(sid)] = row
    hist_path = mpath.parent / "history.json"
    best_epoch = best_val = None
    if hist_path.is_file():
        try:
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
            best_epoch = hist.get("best_epoch")
            best_val = hist.get("best_val_rmse_db")
        except Exception:  # noqa: BLE001
            # history.json 只用于"best epoch"这类辅助信息，坏了不影响主表口径，
            # 但也不允许假装它有内容。
            best_epoch = best_val = None
    return {
        "path": str(mpath),
        "exp_id": exp_id,
        "config_name": str(config_name),
        "config_name_is_derived": bool(not raw.get("config_name")),
        "seed": raw.get("seed"),
        "split": split,
        "split_source": split_source,
        "weights": raw.get("weights"),
        "n_samples": raw.get("n_samples"),
        "aggregate": agg,
        "per_sample": per_sample,
        "meta": meta,
        "db_bounds": meta.get("db_bounds"),
        "n_params_total": meta.get("n_params_total"),
        "n_params_conditional": meta.get("n_params_conditional"),
        "prior_mode": meta.get("prior_mode"),
        "cot_mode": meta.get("cot_mode"),
        "splits_hash": meta.get("splits_hash"),
        "ckpt": meta.get("ckpt"),
        "history_path": str(hist_path) if hist_path.is_file() else None,
        "best_epoch": best_epoch,
        "best_val_rmse_db": best_val,
    }


def _strip_seed_suffix(exp_id: str) -> str:
    """`prior_real_seed0` → `prior_real`（仅在缺少 config_name 时的兜底）。"""
    parts = exp_id.split("_")
    if len(parts) >= 2 and parts[-1].lower().startswith("seed"):
        return "_".join(parts[:-1])
    return exp_id


def group_by_config(records: list[dict]) -> dict[str, list[dict]]:
    """按 config_name 分组；组内按 seed 升序（seed 缺失排最后）。"""
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["config_name"], []).append(r)
    for k in groups:
        groups[k].sort(key=lambda r: (r["seed"] is None, r["seed"] if r["seed"] is not None else 0,
                                      r["exp_id"]))
    return groups


def _metric_means(recs: Sequence[dict], key: str) -> list[float]:
    """取每个 run 的 aggregate.mean_<key>（缺失的 run 直接不参与，不做插补）。"""
    out = []
    for r in recs:
        v = (r["aggregate"] or {}).get(f"mean_{key}")
        if v is not None and math.isfinite(float(v)):
            out.append(float(v))
    return out


def _params_of(recs: Sequence[dict]) -> str:
    vals = [r["n_params_total"] for r in recs if r["n_params_total"] is not None]
    if not vals:
        return MISSING
    if len(set(int(v) for v in vals)) > 1:
        # 参数量在同 config 的 seed 间不一致 = 结构不一致，这本身就是必须暴露的异常
        return "不一致（" + ", ".join(str(int(v)) for v in vals) + "）"
    return f"{int(vals[0]):,}"


def _param_values(recs: Sequence[dict]) -> list[int]:
    """该 config 各 seed 记录的参数量（去重排序）。用于"参数量是否对齐"检查。"""
    return sorted({int(r["n_params_total"]) for r in recs if r["n_params_total"] is not None})


# ---------------------------------------------------------- 逐样本对齐与配对

def aligned_per_sample(recs: Sequence[dict], metric: str = "rmse_db"
                       ) -> tuple[list[str], list[float], dict]:
    """把一个 config 的多个 seed 对齐成"每个 sample_id 一个值"。

    **口径**：先按 sample_id 求各 seed 的算术平均，再用于配对比较。
    这样 paired_compare 里的 n 就是 test maps 的样本数（例如 800），
    而不是 800×3（那样会人为放大 n、让 p 值虚假变小）。
    只有**至少一个 seed 有该样本**才纳入；具体覆盖情况写进返回的 info。
    """
    acc: dict[str, list[float]] = {}
    per_seed_cov: dict[str, int] = {}
    for r in recs:
        sid_seed = str(r["seed"]) if r["seed"] is not None else r["exp_id"]
        got = 0
        for sid, row in r["per_sample"].items():
            v = row.get(metric)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fv):
                continue
            acc.setdefault(sid, []).append(fv)
            got += 1
        per_seed_cov[sid_seed] = got
    ids = sorted(acc.keys())
    vals = [sum(acc[s]) / len(acc[s]) for s in ids]
    info = {
        "metric": metric,
        "n_sample_ids": len(ids),
        "seed_coverage": per_seed_cov,
        "n_seeds": len(recs),
        "averaged_over_seeds": True,
    }
    return ids, vals, info


def compare_configs(recs_a: Sequence[dict], recs_b: Sequence[dict], name_a: str, name_b: str,
                    metric: str = "rmse_db", lower_is_better: bool = True) -> dict:
    """两个 config 的逐样本配对比较（先按 sample_id 对齐，再取交集）。"""
    ids_a, vals_a, info_a = aligned_per_sample(recs_a, metric)
    ids_b, vals_b, info_b = aligned_per_sample(recs_b, metric)
    map_a, map_b = dict(zip(ids_a, vals_a)), dict(zip(ids_b, vals_b))
    common = sorted(set(map_a) & set(map_b))
    if not common:
        return {
            "names": [name_a, name_b], "n": 0,
            "error": "两个 config 没有共同的 sample_id，无法配对比较",
            "alignment": {"a": info_a, "b": info_b},
        }
    res = paired_compare([map_a[s] for s in common], [map_b[s] for s in common],
                         (name_a, name_b), lower_is_better=lower_is_better)
    res["alignment"] = {
        "metric": metric,
        "a": info_a, "b": info_b,
        "n_common_sample_ids": len(common),
        "n_only_a": len(set(map_a) - set(map_b)),
        "n_only_b": len(set(map_b) - set(map_a)),
        "note": "多 seed 时先按 sample_id 对各 seed 取算术平均，再配对；"
                "只用两侧都存在的 sample_id。",
    }
    res["per_sample"] = {"sample_ids": common,
                         "a": [map_a[s] for s in common],
                         "b": [map_b[s] for s in common]}
    return res


def seed_level_delta(recs_a: Sequence[dict], recs_b: Sequence[dict], name_a: str, name_b: str,
                     metric: str = "mean_rmse_db") -> dict:
    """seed 级配对差值：同 seed 的两个 run 相减，再在 seed 维取 mean±std。

    与逐样本配对的分工：这一层回答"跨 seed 稳定吗"，逐样本配对回答"在固定测试集上
    差值是否显著"。两者都要报（结论要求 mean±std，而显著性只能从逐样本得到）。
    """
    by_seed_a = {r["seed"]: r for r in recs_a if r["seed"] is not None}
    by_seed_b = {r["seed"]: r for r in recs_b if r["seed"] is not None}
    seeds = sorted(set(by_seed_a) & set(by_seed_b))
    pairs = []
    for s in seeds:
        va = (by_seed_a[s]["aggregate"] or {}).get(metric)
        vb = (by_seed_b[s]["aggregate"] or {}).get(metric)
        if va is None or vb is None:
            continue
        pairs.append({"seed": s, "a": float(va), "b": float(vb),
                      "delta": float(va) - float(vb)})
    deltas = [p["delta"] for p in pairs]
    stat = _mean_std_nums(deltas)
    return {
        "names": [name_a, name_b], "metric": metric,
        "n_seeds_paired": len(pairs), "pairs": pairs,
        "mean_diff": stat["mean"], "std_diff": stat["std"],
        "seeds_paired": [p["seed"] for p in pairs],
        "note": ("只有两侧都有该 seed 的 run 才参与配对；n_seeds_paired=0 说明两个 config 的 "
                 "seed 集合没有交集（例如一个跑 seed0-2、另一个只跑 seed3），此时不能报差值。"),
    }


# ------------------------------------------------------------------ 报告

def build_summary(runs_dir: Path, baseline: str, ladder: str = "v1") -> dict:
    """读 runs → 生成结构化汇总（markdown 与 JSON 共用同一份数据，保证同源）。"""
    profile = LADDER_PROFILES.get(ladder, LADDER_PROFILES["v1"])
    ladder_order = profile["order"]
    ladder_deltas = profile["deltas"]
    primary = profile["primary"]
    none_name = profile["none_name"]
    shuffled_name = profile["shuffled_name"]
    if baseline == "prior_masked" and ladder == "v3":
        baseline = profile["default_baseline"]   # 默认基线随 ladder 档案切换

    records, bad, ignored = load_runs(runs_dir)
    groups = group_by_config(records)

    # ---- 主表：按 mean RMSE 升序 ----
    main_rows = []
    for name, recs in groups.items():
        rmse = _metric_means(recs, "rmse_db")
        row = {
            "config_name": name,
            "n_runs": len(recs),
            "n_seeds": len({r["seed"] for r in recs if r["seed"] is not None}),
            "exp_ids": [r["exp_id"] for r in recs],
            "seeds": [r["seed"] for r in recs],
            "rmse_db": _mean_std_nums(rmse),
            "mae_db": _mean_std_nums(_metric_means(recs, "mae_db")),
            "ssim": _mean_std_nums(_metric_means(recs, "ssim")),
            "rmse_db_nonsaturated": _mean_std_nums(_metric_means(recs, "rmse_db_nonsaturated")),
            "rmse_norm": _mean_std_nums(_metric_means(recs, "rmse_norm")),
            "n_params_total": _params_of(recs),
            "path": recs[0]["path"],
        }
        main_rows.append(row)
    main_rows.sort(key=lambda r: (r["rmse_db"]["mean"] is None, r["rmse_db"]["mean"]))

    # ---- 消融阶梯 ----
    ladder = []
    for name in ladder_order:
        recs = groups.get(name)
        if not recs:
            ladder.append({"config_name": name, "status": NOT_RUN})
            continue
        ladder.append({
            "config_name": name, "status": "已跑",
            "n_seeds": len({r["seed"] for r in recs if r["seed"] is not None}),
            "n_runs": len(recs),
            "rmse_db": _mean_std_nums(_metric_means(recs, "rmse_db")),
            "mae_db": _mean_std_nums(_metric_means(recs, "mae_db")),
            "ssim": _mean_std_nums(_metric_means(recs, "ssim")),
            "rmse_db_nonsaturated": _mean_std_nums(_metric_means(recs, "rmse_db_nonsaturated")),
            "n_params_total": _params_of(recs),
        })
    # 不在固定四档里的其它 config（例如 CoT 变体）也单独列出，不能因为不熟就丢掉
    extra_configs = [n for n in groups if n not in ladder_order]

    deltas = {}
    for a, b in ladder_deltas:
        key = f"{a} − {b}"
        if a not in groups or b not in groups:
            deltas[key] = {"status": NOT_RUN,
                           "missing": [n for n in (a, b) if n not in groups]}
            continue
        d = seed_level_delta(groups[a], groups[b], a, b)
        d["status"] = "已跑" if d["n_seeds_paired"] else "seed 不匹配，无法配对"
        deltas[key] = d

    # ---- 配对显著性（逐样本）----
    paired = {}
    pair_specs: list[tuple[str, str]] = []
    if baseline != primary:
        pair_specs.append((primary, baseline))
    pair_specs += [(primary, shuffled_name), (primary, none_name)]
    # AGENTS「配对对照清单」：任何 `*_shuffled` 配置都要和它的非打乱版本配对 ——
    # 这样回答的才是"起作用的是模块**内容**，还是那条条件通路/额外容量"。
    # 例：prior_real_cot vs prior_real_cot_shuffled（CoT 内容的净贡献）。
    # 只对存在的配置生成；缺的一律走"未跑"分支，绝不编造。
    for name in sorted(groups):
        if name.endswith("_shuffled") and len(name) > len("_shuffled"):
            base_name = name[: -len("_shuffled")]
            # 只有"非打乱版本"确实跑过才加这一对；否则会凭空多出一行
            # 「prior − prior_shuffled：未跑（缺 prior）」这种噪声
            if base_name in groups:
                pair_specs.append((base_name, name))
    # 配对对照清单里的第三行：`prior_real ↔ prior_real_cot`（加了一个 CoT 编码器）。
    # 这一对的参数量**本来就不同**（+286,880），所以它必须出现，并且由下面的
    # 参数量对齐检查明确标注"差值混入容量效应"，免得被算成 CoT 内容的功劳。
    for name in sorted(groups):
        if name.endswith("_cot") and len(name) > len("_cot"):
            base_name = name[: -len("_cot")]
            if base_name in groups:
                pair_specs.append((base_name, name))
    # 补充对照（阳性对照 oracle / 容量天花板 big）来自 profile，同样进配对表
    for a, b in profile.get("extra_pairs", ()):
        pair_specs.append((a, b))
    seen_pairs: set[tuple[str, str]] = set()
    pair_order: list[tuple[str, str]] = []
    for a, b in pair_specs:
        if (a, b) in seen_pairs:
            continue
        seen_pairs.add((a, b))
        pair_order.append((a, b))
        key = f"{a}_vs_{b}"
        if a not in groups or b not in groups:
            paired[key] = {"status": NOT_RUN, "names": [a, b],
                           "missing": [n for n in (a, b) if n not in groups]}
            continue
        paired[key] = compare_configs(groups[a], groups[b], a, b)
        paired[key]["status"] = "已跑"

    # ---- 参数量对齐检查（"差值能不能归因于内容"的前提）----
    # AGENTS：只报 `prior_real_cot − prior_real` 会把 28.7 万参数的容量效应算成 CoT 的功劳。
    # 所以每一组配对都要顺带报出两侧参数量，不一致就明确写"差值混入容量效应"。
    param_alignment = []
    for a, b in pair_order:
        if a not in groups or b not in groups:
            continue
        pa, pb = _param_values(groups[a]), _param_values(groups[b])
        known = len(pa) == 1 and len(pb) == 1
        aligned = bool(known and pa[0] == pb[0])
        param_alignment.append({
            "pair": f"{a} − {b}", "params_a": pa, "params_b": pb,
            "aligned": aligned,
            "delta_params_b_minus_a": (pb[0] - pa[0]) if known else None,
            "note": ("两侧结构逐位相同 → 差值可归因于内容" if aligned else
                     "两侧参数量不同或无记录 → 差值混入了容量效应，不能单独归因于内容"),
        })

    # ---- 失败样本与最坏情况（primary − baseline）----
    worst = {"status": NOT_RUN}
    diff_key = f"{primary} − {baseline}"
    if primary in groups and baseline in groups:
        cmp_res = compare_configs(groups[primary], groups[baseline],
                                  primary, baseline)
        if cmp_res.get("per_sample"):
            ps = cmp_res["per_sample"]
            rows = [{"sample_id": s, "real": a, "baseline": b, "delta": a - b}
                    for s, a, b in zip(ps["sample_ids"], ps["a"], ps["b"])]
            # delta > 0 表示 real 的 RMSE 更大 = real 变差（RMSE 越小越好）
            worse = [r for r in rows if r["delta"] > 1e-12]
            better = [r for r in rows if r["delta"] < -1e-12]
            tie = [r for r in rows if abs(r["delta"]) <= 1e-12]
            by_delta = sorted(rows, key=lambda r: r["delta"])
            # 逐 seed 的变差比例：seed 平均会抵消一部分噪声，只报平均口径会让人以为
            # 变差样本"比实际少"。这里把每个 seed 单独算一遍，暴露 seed 间的差异。
            per_seed = []
            seeds_a = {r["seed"]: r for r in groups[primary] if r["seed"] is not None}
            seeds_b = {r["seed"]: r for r in groups[baseline] if r["seed"] is not None}
            for s in sorted(set(seeds_a) & set(seeds_b)):
                ia, va, _ = aligned_per_sample([seeds_a[s]])
                ib, vb, _ = aligned_per_sample([seeds_b[s]])
                ma_, mb_ = dict(zip(ia, va)), dict(zip(ib, vb))
                common_s = sorted(set(ma_) & set(mb_))
                if not common_s:
                    continue
                nw = sum(1 for x in common_s if ma_[x] - mb_[x] > 1e-12)
                per_seed.append({"seed": s, "n": len(common_s), "n_worse": nw,
                                 "pct_worse": 100.0 * nw / len(common_s)})
            worst = {
                "status": "已跑",
                "baseline": baseline,
                "n_total": len(rows),
                "n_worse": len(worse),
                "n_better": len(better),
                "n_tie": len(tie),
                "pct_worse": (100.0 * len(worse) / len(rows)) if rows else None,
                "pct_better": (100.0 * len(better) / len(rows)) if rows else None,
                "worst5": by_delta[-5:][::-1],   # delta 最大 = real 变差最多
                "best5": by_delta[:5],           # delta 最小 = real 改善最多
                "per_seed": per_seed,
                "note": f"delta = {primary} − {baseline}（RMSE，越小越好）：delta>0 表示 {primary} 在该样本上变差。"
                        "本表覆盖**全部**配对样本，不采用任何子集筛选。"
                        "主口径为多 seed 平均后的逐样本序列（与第 3 节一致）；"
                        "单 seed 的口径见 per_seed，两者可能不同（seed 平均会抵消部分噪声）。",
            }
        else:
            worst = {"status": "无法配对", "reason": cmp_res.get("error")}

    # ---- 口径声明 ----
    weights = sorted({str(r["weights"]) for r in records if r["weights"] is not None})
    splits = sorted({str(r["split"]) for r in records if r["split"] is not None})
    db_bounds = sorted({tuple(r["db_bounds"]) for r in records if r["db_bounds"] is not None})
    hashes = sorted({str(r["splits_hash"]) for r in records if r["splits_hash"] is not None})
    derived = sorted({r["config_name"] for r in records if r["config_name_is_derived"]})
    split_derived = sorted({r["exp_id"] for r in records
                            if r["split"] and r["split_source"] != "recorded"})
    caveats = {
        "db_bounds_recorded": [list(map(float, b)) for b in db_bounds],
        "db_bounds_missing_in": [r["exp_id"] for r in records if r["db_bounds"] is None],
        "weights": weights,
        "weights_missing_in": [r["exp_id"] for r in records if r["weights"] is None],
        "splits": splits,
        "split_missing_in": [r["exp_id"] for r in records if not r["split"]],
        "split_derived_for": split_derived,
        "splits_hash": hashes,
        "config_name_derived_for": derived,
        "n_runs": len(records),
        "ignored_files": ignored,
    }

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": "src/eval/report.py",
        "runs_dir": str(runs_dir),
        "baseline": baseline,
        "ladder_profile": ladder,
        "primary": primary,
        "main_table": main_rows,
        "ladder": {"order": list(ladder_order), "rows": ladder,
                   "extra_configs": extra_configs, "deltas": deltas,
                   "deltas_order": list(ladder_deltas)},
        "paired": paired,
        "param_alignment": param_alignment,
        "worst_samples": worst,
        "caveats": caveats,
        "paired_protocol": {
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_n": BOOTSTRAP_N,
            "primary_test": "wilcoxon",
            "per_sample_aggregation": "同一 config 的多个 seed 先按 sample_id 取算术平均，"
                                      "再与被比较 config 取 sample_id 交集做配对比较",
        },
        "sources": [{"exp_id": r["exp_id"], "config_name": r["config_name"],
                     "seed": r["seed"], "split": r["split"],
                     "split_source": r["split_source"],
                     "weights": r["weights"], "n_samples": r["n_samples"],
                     "best_epoch": r["best_epoch"],
                     "metrics_json": r["path"],
                     "history_json": r["history_path"], "status": "已读"}
                    for r in records]
                   + [{"exp_id": Path(b["path"]).parent.name, "config_name": MISSING, "seed": None,
                       "split": _split_from_filename(Path(b["path"]).name),
                       "split_source": None, "weights": None, "n_samples": None,
                       "best_epoch": None, "metrics_json": b["path"], "history_json": None,
                       "status": f"解析失败：{b['error']}"} for b in bad],
        "errors": bad,
    }


def render_markdown(summary: dict) -> str:
    """把结构化汇总渲染成 markdown。**数字全部来自 summary，不在这里写死任何值。**"""
    L: list[str] = []
    add = L.append
    add("# 实验结果汇总（自动生成）\n")
    add("> **本文件由 `src/eval/report.py` 自动生成，禁止手改。**")
    add(f"> 生成时间：{summary['generated_at']}")
    add(f"> 数据来源目录：`{summary['runs_dir']}`（共 {summary['caveats']['n_runs']} 个 metrics 文件，"
        f"接受 `metrics.json` 与 `metrics_<split>.json`）")
    add(f"> 主对照基线：`{summary['baseline']}`")
    add("")
    add("所有数字（含 mean±std、配对检验、最差样本）均由脚本从 `runs/*/metrics_*.json` 计算，"
        "未记录 / 未跑的项一律显式标注，**没有任何手工填写或默认补齐的数字**。")
    add("")

    # ---------------- 1. 主表 ----------------
    add("## 1. 主表（按 mean RMSE 升序）\n")
    add("每个 `config_name` 一行；多个 seed 时对该指标先在 seed 维取 mean±std。")
    add("`± —` 表示只有一个 seed，方差无法估计（**不是** ±0.00）。\n")
    rows = []
    for r in summary["main_table"]:
        rows.append([
            f"`{r['config_name']}`", r["n_seeds"], _pair(r["rmse_db"]),
            _pair(r["mae_db"]), _pair(r["ssim"]), _pair(r["rmse_db_nonsaturated"]),
            r["n_params_total"],
        ])
    add(_md_table(["config_name", "n_seeds", "RMSE (dB) mean±std", "MAE (dB) mean±std",
                   "SSIM mean±std", "RMSE non-sat (dB) mean±std", "参数量"], rows))
    add("")
    if not rows:
        add("（无数据）\n")

    # ---------------- 2. 消融阶梯 ----------------
    add("## 2. 消融阶梯（None / Masked / Real / Shuffled）\n")
    ladder_rows = []
    for r in summary["ladder"]["rows"]:
        if r["status"] == NOT_RUN:
            ladder_rows.append([f"`{r['config_name']}`", "—", NOT_RUN, NOT_RUN, NOT_RUN,
                                NOT_RUN, NOT_RUN, NOT_RUN])
            continue
        ladder_rows.append([
            f"`{r['config_name']}`", r["n_seeds"], _pair(r["rmse_db"]), _pair(r["mae_db"]),
            _pair(r["ssim"]), _pair(r["rmse_db_nonsaturated"]), r["n_params_total"],
            r["status"],
        ])
    add(_md_table(["配置", "n_seeds", "RMSE (dB)", "MAE (dB)", "SSIM",
                   "RMSE non-sat (dB)", "参数量", "状态"], ladder_rows))
    add("")
    for a, b in summary["ladder"]["deltas_order"]:
        key = f"{a} − {b}"
        d = summary["ladder"]["deltas"].get(key, {})
        if d.get("status") == NOT_RUN:
            add(f"* **`{key}`：{NOT_RUN}**（缺少 {', '.join(d.get('missing', []))}，不编造数字）")
            continue
        if not d.get("n_seeds_paired"):
            add(f"* **`{key}`：无法配对**（{d.get('note', '')}）")
            continue
        ms = _mean_std_nums([p["delta"] for p in d["pairs"]])
        add(f"* **`{key}` = {_fmt(ms['mean'], 3)} ± "
            f"{_fmt(ms['std'], 3) if ms['std'] is not None else '—'} dB**"
            f"（seed 级配对，n_seeds={d['n_seeds_paired']}，seeds={d['seeds_paired']}；"
            f"负值 = real 的 RMSE 更小 = 更好）")
    add("")
    if summary["ladder"]["extra_configs"]:
        add(f"> 另有不在固定四档内的配置（如 CoT 变体），未纳入本表但已在主表第 1 节列出："
            f"{', '.join('`' + c + '`' for c in summary['ladder']['extra_configs'])}")
        add("")

    # ---------------- 3. 配对显著性 ----------------
    add("## 3. 配对显著性（同一批 test 样本逐样本配对）\n")
    proto = summary["paired_protocol"]
    add(f"口径：{proto['per_sample_aggregation']}。"
        f"非参数 Wilcoxon 符号秩为主判据（bootstrap {proto['bootstrap_n']} 次、"
        f"种子 {proto['bootstrap_seed']}，CI 固定可复现），配对 t 检验同时给出供交叉核对。"
        f"**不使用独立样本 Cohen's d**：同一批样本的难度相关性会被非配对检验浪费掉。\n")
    p_rows = []
    for key, res in summary["paired"].items():
        if res.get("status") == NOT_RUN:
            nm = res.get("names")
            label = f"{nm[0]} − {nm[1]}" if nm else key
            # 缺哪个配置必须写出来（"未跑"不等于"没差异"，更不允许留空让人误读）
            miss = "、".join(res.get("missing", []))
            p_rows.append([f"`{label}`", NOT_RUN + f"（缺 {miss}）", "—", "—", "—", "—",
                           "—", "—", "—"])
            continue
        if res.get("error"):
            p_rows.append([f"`{key}`", "无法配对", res["error"], "—", "—", "—", "—", "—", "—"])
            continue
        na, nb = res["names"]
        p_rows.append([
            f"`{na} − {nb}`", res["n"], _fmt(res["mean_diff"], 3),
            f"[{_fmt(res['ci95_low'], 3)}, {_fmt(res['ci95_high'], 3)}]",
            _fmt(res["cohens_dz"], 3), f"{res['wilcoxon_p']:.3g}",
            f"{res['ttest_p']:.3g}",
            f"{res['n_a_better']}/{res['n_b_better']}/{res['n_tie']}",
            ("**显著**" if res["significant_at_005"] else "不显著"),
        ])
    add(_md_table(["配对（a − b）", "n", "mean diff (dB)", "95% CI (bootstrap)", "Cohen's dz",
                   "Wilcoxon p", "配对 t p", "a优/b优/平", "p<0.05"], p_rows))
    add("")
    for key, res in summary["paired"].items():
        if res.get("note"):
            add(f"> `{key}` 注意：{res['note']}")
    if any(res.get("note") for res in summary["paired"].values()):
        add("")
    # 对齐明细要写清楚"用了多少个样本"，否则读者无法判断是否只用了子集
    for key, res in summary["paired"].items():
        al = res.get("alignment")
        if al:
            add(f"> `{key}` 对齐：a 侧 {al['a']['n_sample_ids']} 个 sample_id、"
                f"b 侧 {al['b']['n_sample_ids']} 个，共同 {al['n_common_sample_ids']} 个"
                f"（仅 a {al['n_only_a']} / 仅 b {al['n_only_b']}）；"
                f"指标口径 `{al['metric']}`，多 seed 先取 seed 平均。")
    add("")
    if summary.get("param_alignment"):
        add("### 参数量对齐检查（差值能否归因于「内容」的前提）\n")
        add("参数量不同就说明差值里混了容量效应，不能把提升单独算成先验/CoT 内容的功劳。\n")
        add(_md_table(["配对（a − b）", "参数量(a)", "参数量(b)", "是否逐位对齐", "含义"],
                      [[f"`{p['pair']}`",
                        ", ".join(f"{v:,}" for v in p["params_a"]) or MISSING,
                        ", ".join(f"{v:,}" for v in p["params_b"]) or MISSING,
                        "**是**" if p["aligned"] else "**否**",
                        p["note"]] for p in summary["param_alignment"]]))
        add("")

    # ---------------- 4. 失败样本与最坏情况 ----------------
    add("## 4. 失败样本与最坏情况（防止选择性呈现）\n")
    w = summary["worst_samples"]
    if w.get("status") != "已跑":
        add(f"本节无法生成：{w.get('reason', w.get('status'))}\n")
    else:
        add(f"口径：`delta = {summary['primary']} − {w['baseline']}` 的**逐样本** RMSE 差值，"
            f"覆盖全部 {w['n_total']} 个配对样本（无任何子集筛选）。"
            f"RMSE 越小越好，故 `delta > 0` 表示 {summary['primary']} 在该样本上**变差**。\n")
        add(f"* real **变差**的样本：**{w['n_worse']} / {w['n_total']}"
            f"（{_fmt(w['pct_worse'], 1)}%）**")
        add(f"* real 改善的样本：{w['n_better']} / {w['n_total']}"
            f"（{_fmt(w['pct_better'], 1)}%）")
        add(f"* 持平样本：{w['n_tie']}")
        add("")
        if w.get("per_seed"):
            add("逐 seed 口径（不做 seed 平均，防止平均把变差的样本数量「抹平」）：\n")
            add(_md_table(["seed", "n", "real 变差样本数", "占比"],
                          [[p["seed"], p["n"], p["n_worse"], f"{_fmt(p['pct_worse'], 1)}%"]
                           for p in w["per_seed"]]))
            add("")
        add("最差 5 个样本（real 相对基线变差最多）：\n")
        add(_md_table(["sample_id", f"real RMSE (dB)", f"{w['baseline']} RMSE (dB)",
                       "delta (dB)"],
                      [[f"`{r['sample_id']}`", _fmt(r["real"], 3), _fmt(r["baseline"], 3),
                        f"**{_fmt(r['delta'], 3)}**"] for r in w["worst5"]]))
        add("")
        add("最好 5 个样本（real 相对基线改善最多）：\n")
        add(_md_table(["sample_id", f"real RMSE (dB)", f"{w['baseline']} RMSE (dB)",
                       "delta (dB)"],
                      [[f"`{r['sample_id']}`", _fmt(r["real"], 3), _fmt(r["baseline"], 3),
                        _fmt(r["delta"], 3)] for r in w["best5"]]))
        add("")
        add(f"> 上一版的教训：只报 12 个偏难样本的 75% 提升、隐去全量变差比例。"
            f"本节把**变差比例（{_fmt(w['pct_worse'], 1)}%）**与最好/最差样本放在同一张表里，"
            f"读者可以同时看到代价与收益。")
        add("")

    # ---------------- 5. 口径声明 ----------------
    c = summary["caveats"]
    add("## 5. 口径声明\n")
    if c["db_bounds_recorded"]:
        for b in c["db_bounds_recorded"]:
            add(f"* **dB 区间**：`meta.db_bounds = [{_fmt(b[0], 1)}, {_fmt(b[1], 1)}]` "
                f"（来自 metrics 文件；本报告不替换、不推算）")
    else:
        add("* **dB 区间**：未记录（metrics.json 里没有 `meta.db_bounds`）。"
            "本文档不替它填 `src/data/normalize.py` 的默认值，请补齐后重跑本脚本。")
    if c["db_bounds_missing_in"]:
        add(f"  - 缺 `meta.db_bounds` 的 run：{', '.join('`' + e + '`' for e in c['db_bounds_missing_in'])}")
    add(f"* **权重口径（ema/raw）**："
        f"{', '.join('`' + str(x) + '`' for x in c['weights']) if c['weights'] else '未记录'}"
        f"（`metrics.json:weights`）。训练记 EMA 成绩、评测用原始权重是上一版的已知 bug，"
        f"两份口径**不得混用**。")
    if c["weights_missing_in"]:
        add(f"  - 缺 `weights` 的 run：{', '.join('`' + e + '`' for e in c['weights_missing_in'])}")
    add(f"* **split 口径**："
        f"{', '.join('`' + str(x) + '`' for x in c['splits']) if c['splits'] else '未记录'}"
        f"；test maps 只评一次（`data/splits.json` 冻结，val 只用于选模型/早停）。")
    if c.get("split_derived_for"):
        add(f"  - 以下 run 的 split 是**从文件名推导**的（metrics.json 里没有 `split` 字段），"
            f"属于推导值：{', '.join('`' + e + '`' for e in c['split_derived_for'])}")
    if c.get("split_missing_in"):
        add(f"  - 没有 split 信息（文件名也没有 `metrics_<split>` 形式）的 run："
            f"{', '.join('`' + e + '`' for e in c['split_missing_in'])}")
    add(f"* **splits_hash**："
        f"{', '.join('`' + h + '`' for h in c['splits_hash']) if c['splits_hash'] else '未记录'}"
        f"；不同 hash 的结果**不可直接比较**（划分不同）。")
    add("* **配对口径**：同一批 test maps 上逐样本配对；多 seed 先按 `sample_id` 取 seed 平均；"
        "显著性用 Wilcoxon 符号秩（非参数、配对），效应量用 Cohen's dz。")
    if c["config_name_derived_for"]:
        add(f"* **配置名来源**：以下 run 的 `metrics.json` 里没有 `config_name`，"
            f"报告按 `meta.prior_mode`/`exp_id` 推导："
            f"{', '.join('`' + x + '`' for x in c['config_name_derived_for'])}。"
            f"这属于推导值，补上原始字段后请重跑。")
    add("")

    # ---------------- 6. 数据来源 ----------------
    add("## 6. 数据来源（每个数字来自哪个文件）\n")
    add(_md_table(["exp_id", "config_name", "seed", "split", "weights", "n_samples",
                   "best_epoch", "metrics 文件", "status"],
                  [[f"`{s['exp_id']}`",
                    f"`{s['config_name'] if s['config_name'] else MISSING}`",
                    s["seed"] if s["seed"] is not None else MISSING,
                    (s["split"] if s["split"] else MISSING)
                    + ("（推导）" if s.get("split_source") not in (None, "recorded") else ""),
                    s["weights"] if s["weights"] is not None else MISSING,
                    s["n_samples"] if s["n_samples"] is not None else MISSING,
                    s["best_epoch"] if s["best_epoch"] is not None else MISSING,
                    f"`{s['metrics_json']}`", s["status"]] for s in summary["sources"]]))
    add("")
    if summary["caveats"].get("ignored_files"):
        add("### 被忽略的 metrics 文件（同一目录只取一个，避免重复计数）\n")
        for ig in summary["caveats"]["ignored_files"]:
            add(f"* `{ig['path']}` 被忽略：{ig['reason']}（实际采用 `{ig['used_instead']}`）")
        add("")
    if summary["errors"]:
        add("### 解析失败的 metrics.json（**必须修好再引用本报告**）\n")
        for e in summary["errors"]:
            add(f"* `{e['path']}`：{e['error']}")
        add("")
    return "\n".join(L)


def _pair(stat: dict) -> str:
    """把 _mean_std_nums 的结果渲染成 `mean ± std`。"""
    if stat.get("mean") is None:
        return MISSING
    if stat.get("std") is None:
        return f"{stat['mean']:.2f} ± —"
    return f"{stat['mean']:.2f} ± {stat['std']:.2f}"


# ------------------------------------------------------------------ CLI

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.eval.report",
        description="从 runs/*/metrics_*.json 自动生成汇总报告（禁止手抄数字）")
    ap.add_argument("--runs", default="runs", help="runs 目录（扫描其中的 metrics.json / "
                                                  "metrics_<split>.json）")
    ap.add_argument("--out", default="reports/summary.md", help="输出的 markdown 路径")
    ap.add_argument("--json", default=None, help="同时输出的结构化 JSON 路径（可选）")
    ap.add_argument("--baseline", default="prior_masked",
                    help="主对照配置名（默认 prior_masked；阶梯表固定给出 real−masked 与 "
                         "real−shuffled 两列）")
    ap.add_argument("--ladder", default="v1", choices=["v1", "v3"],
                    help="消融阶梯命名档案：v1（默认，prior_masked 整向量置零）或 "
                         "v3（prior_v3_masked_em 只把 LLM 电磁 22 维置零、保留程序几何）")
    args = ap.parse_args(argv)

    runs_dir = Path(args.runs)
    if not runs_dir.is_dir():
        print(f"[错误] runs 目录不存在：{runs_dir}", file=sys.stderr)
        return 2
    files, ignored = find_metric_files(runs_dir)
    if not files:
        print(f"[错误] 在 {runs_dir} 下没有找到任何 metrics.json / metrics_*.json。\n"
              f"       本报告只从 runs/<exp_id>/metrics_<split>.json 取数，"
              f"不会用任何默认值填充。\n"
              f"       请先跑 `python -m src.eval.run_eval --run-dir <dir> --split test` "
              f"把 metrics 落盘（AGENTS.md 规则 5）。", file=sys.stderr)
        return 2

    summary = build_summary(runs_dir, args.baseline, args.ladder)
    md = render_markdown(summary)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"[OK] markdown 已写出：{out_path}"
          f"（{len(summary['main_table'])} 个 config，{summary['caveats']['n_runs']} 个 run）")

    if args.json:
        jpath = Path(args.json)
        jpath.parent.mkdir(parents=True, exist_ok=True)
        jpath.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
                         encoding="utf-8")
        print(f"[OK] JSON 已写出：{jpath}")

    for ig in ignored:
        print(f"[警告] 忽略 {ig['path']}：{ig['reason']}（采用 {ig['used_instead']}）",
              file=sys.stderr)
    if summary["errors"]:
        print(f"[错误] {len(summary['errors'])} 个 metrics 文件解析失败：", file=sys.stderr)
        for e in summary["errors"]:
            print(f"       - {e['path']}: {e['error']}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
