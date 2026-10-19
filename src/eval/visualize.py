"""可视化：训练曲线、单样本三联图、checkpoint 预测渲染。

为什么 matplotlib 要**延迟导入**
-------------------------------
`tests/test_local.py`、`scripts/audit.py` 都不需要画图，而 matplotlib 在服务器上经常
装不上/装错后端。如果模块顶层 import matplotlib，那么"没有 matplotlib 的环境"
连 `from src.eval.visualize import ...` 都会失败，等于让一个可选依赖卡死整个评测链。
所以所有 matplotlib 的 import 都放进 `_mpl()` 里，只有真正要画图时才触发。

为什么本文件自带 `load_model_from_ckpt`（而不是复用 src/eval/checkpoint.py）
--------------------------------------------------------------------------
项目里**没有** `src/eval/checkpoint.py`，也不能新增，所以这里本地实现一份。
它必须与 `src/eval/run_eval.py:load_checkpoint` 的**口径逐条一致**，否则"可视化看到的
预测"和"报告里的数字"就来自两个不同的模型：
    * 结构：`config.model`（回退 `model_cfg`），`in_channels`/`prior_dim` 优先取
      checkpoint 顶层记录（train.py 会存），因为那是训练当场算出来的值；
    * 权重：`config.eval.weights`（train.py 会把"未启用 EMA 时回退 raw"的结果写进
      `state["eval_weights"]`，所以优先读它）→ ema 取 `ema_shadow`，raw 取 `model_state_dict`；
    * **EMA 缺失必须吵**：上一版的 bug 是"训练记 EMA 成绩、评测用原始权重"且没人发现。
      这里默认打印醒目警告并把实际口径记进返回值；要严格模式就传 `require_ema=True`
      （AGENTS.md 「缺 EMA **报错**而不是静默回退」正是这个语义，CLI 对应 `--require-ema`）。
    * `load_state_dict(..., strict=True)`：结构不一致就报错。绝不 strict=False 静默吞缺键
      （那会把"参数量逐位对齐"这个实验前提变成一个没人知道的谎言）。

CLI
---
    python -m src.eval.visualize curves  --runs runs --out reports/curves.png
    python -m src.eval.visualize predict --ckpt runs/<exp>/ckpt/best.pth \
        --data-root data/radiomapseer --splits data/splits.json \
        --prior-dir priors/<ver>/<llm> --out-dir reports/preds --n 8
    python -m src.eval.visualize selfcheck --out reports/_viz_selfcheck.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:  # Windows 控制台默认 GBK，统一按 UTF-8 输出，否则中文全是乱码
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.normalize import PATHLOSS_MAX, PATHLOSS_MIN, pathloss_norm_to_db  # noqa: E402

# 同一 config 的多个 seed：同色不同线型，保证"颜色=配置、线型=seed"这两个维度不混
LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)))
CONFIG_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
                 "#8c564b", "#17becf", "#e377c2", "#7f7f7f", "#bcbd22")
# 中文字体候选（缺失时自动退回英文标签，避免图上一堆方框）
CJK_FONTS = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
             "WenQuanYi Zen Hei", "PingFang SC", "Heiti SC", "Arial Unicode MS")


# ------------------------------------------------------------ matplotlib 延迟导入

def _mpl():
    """返回 (plt, use_chinese_labels)。所有 matplotlib 相关的东西都从这里拿。"""
    import matplotlib
    matplotlib.use("Agg")          # 无显示环境（服务器/CI）必须用 Agg，否则 savefig 会崩
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    cjk = [f for f in CJK_FONTS if f in available]
    if cjk:
        plt.rcParams["font.sans-serif"] = cjk + ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False   # 负号用 ASCII，避免中文字体缺字形
        return plt, True
    print("[提示] 系统里没找到中文字体（Microsoft YaHei/SimHei/Noto CJK 等），"
          "图内文字自动改用英文，避免出现方框。", file=sys.stderr)
    return plt, False


def _labels(zh: bool) -> dict[str, str]:
    if zh:
        return {"val_rmse": "验证集 RMSE (dB)", "epoch": "epoch", "curves_title":
                "验证集 RMSE 随 epoch 变化（best epoch 已标出）", "buildings": "建筑掩码 + Tx",
                "gt": "真值 pathloss", "pred": "预测 pathloss", "err": "绝对误差 |pred−gt|",
                "db": "pathloss (dB)", "db_err": "绝对误差 (dB)", "best": "best",
                "select_ema": "选模型口径=EMA", "select_raw": "选模型口径=raw",
                "ema_curve": "EMA 曲线",
                "rmse_raw": "RMSE（未裁剪原始预测口径，与 metrics.json 一致）",
                "rmse_disp": "RMSE（按显示用的裁剪图像计算，仅描述这张图）"}
    return {"val_rmse": "validation RMSE (dB)", "epoch": "epoch", "curves_title":
            "Validation RMSE vs epoch (best epoch marked)", "buildings": "building mask + Tx",
            "gt": "ground-truth pathloss", "pred": "predicted pathloss",
            "err": "absolute error |pred-gt|", "db": "pathloss (dB)",
            "db_err": "absolute error (dB)", "best": "best",
            "select_ema": "model selection = EMA", "select_raw": "model selection = raw",
            "ema_curve": "EMA curve",
            "rmse_raw": "RMSE (raw prediction, matches metrics.json)",
            "rmse_disp": "RMSE (from the displayed clipped image)"}


# ------------------------------------------------------------ 训练曲线

def collect_histories(runs_dir: str | Path) -> tuple[list[dict], list[str]]:
    """扫描 runs 下的 history.json。返回 (可用记录, 跳过/提示列表)。

    缺失/坏掉的 history.json **优雅跳过并说明**，不抛异常：曲线图是可选的辅助材料，
    不该阻断整个评测流程（但也不能假装画了 —— 跳过的原因会打印出来）。
    """
    runs_dir = Path(runs_dir)
    records: list[dict] = []
    skipped: list[str] = []
    files = sorted(runs_dir.rglob("history.json")) if runs_dir.is_dir() else []
    if not files:
        skipped.append(f"{runs_dir} 下没有任何 history.json")
        return records, skipped
    for hpath in files:
        try:
            raw = json.loads(hpath.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{hpath} 解析失败（{type(exc).__name__}: {exc}）")
            continue
        epochs = raw.get("epochs") or []
        if not epochs:
            skipped.append(f"{hpath} 里没有 epochs")
            continue
        xs, ys, ema, sel = [], [], [], []
        for e in epochs:
            ep = e.get("epoch")
            val = (e.get("val") or {}).get("mean_rmse_db")
            if ep is None or val is None or not math.isfinite(float(val)):
                continue
            xs.append(int(ep))
            ys.append(float(val))
            ev = e.get("ema_val_rmse_db")
            ema.append(float(ev) if ev is not None and math.isfinite(float(ev)) else None)
            sv = e.get("selected_val_rmse_db")
            sel.append(float(sv) if sv is not None and math.isfinite(float(sv)) else None)
        if not xs:
            skipped.append(f"{hpath} 的 epochs 里没有可用的 val.mean_rmse_db")
            continue
        config_name = raw.get("config_name")
        if not config_name:
            config_name = _strip_seed_suffix(hpath.parent.name)
            skipped.append(f"{hpath} 缺 config_name，按目录名推导为 {config_name}（推导值）")
        # train.py 用 eval_weights 决定"用哪条曲线选 best"：ema 时是 EMA 曲线。
        # 这一点必须带进图里，否则"best epoch 不在 val 曲线最低点"会被误读成代码有 bug。
        eval_weights = str(raw.get("eval_weights") or "ema").lower()
        rec = {
            "path": str(hpath),
            "config_name": str(config_name),
            "seed": raw.get("seed"),
            "eval_weights": eval_weights,
            "epochs": xs, "val_rmse_db": ys, "ema_val_rmse_db": ema,
            "selected_val_rmse_db": sel,
            "best_epoch_recorded": raw.get("best_epoch"),
            "best_val_rmse_db": raw.get("best_val_rmse_db"),
        }
        # 上一版的经典翻车点：正文说"第 25 epoch 领先"，图上第 21 epoch 就交叉。
        # 这里主动核对 history 里记录的 best_epoch 与**选模型那条曲线**的 argmin，
        # 不一致就报出来（不修改数据，只提示矛盾）。
        sel_curve = _selecting_curve(rec)
        if sel_curve:
            imin = int(np.argmin([v for v in sel_curve[1]]))
            rec["best_epoch_from_curve"] = sel_curve[2][imin]
            rec["selecting_curve_name"] = sel_curve[0]
            if (rec["best_epoch_recorded"] is not None
                    and int(rec["best_epoch_recorded"]) != sel_curve[2][imin]):
                skipped.append(
                    f"{hpath}: history.best_epoch={rec['best_epoch_recorded']} 与"
                    f"「{sel_curve[0]}」曲线上最低点 epoch={sel_curve[2][imin]} 不一致 —— "
                    f"正文若引用 epoch 号必须以此处为准核对")
        records.append(rec)
    return records, skipped


def _selecting_curve(rec: dict) -> tuple[str, list[float], list[int]] | None:
    """返回训练时**用来选 best** 的那条曲线 (名字, 值列表, epoch 列表,)。

    train.py：`selected = ema_rmse if eval_weights == "ema" else val_rmse`。
    所以 best epoch 理应落在这条曲线的最低点；打点必须打在这条线上，
    否则图上"best 不在最低点"看起来像 bug（上一版正文与图的矛盾就是这么来的）。
    """
    xs = list(rec["epochs"])
    if rec.get("eval_weights") == "ema":
        pairs = [(x, v) for x, v in zip(xs, rec["ema_val_rmse_db"]) if v is not None]
        if pairs:
            return ("EMA", [v for _x, v in pairs], [x for x, _v in pairs])
    pairs = [(x, v) for x, v in zip(xs, rec["val_rmse_db"]) if v is not None]
    if pairs:
        return ("raw val", [v for _x, v in pairs], [x for x, _v in pairs])
    return None


def _strip_seed_suffix(exp_id: str) -> str:
    parts = exp_id.split("_")
    if len(parts) >= 2 and parts[-1].lower().startswith("seed"):
        return "_".join(parts[:-1])
    return exp_id


def plot_training_curves(runs_dir: str | Path, out_path: str | Path,
                         show_ema: bool | None = None) -> str | None:
    """各 run 的 val RMSE(dB) vs epoch 多曲线图（颜色=config，线型=seed）。

    * 实线永远是 `val.mean_rmse_db`（raw 权重口径的验证曲线）；
    * `show_ema=None`（默认）时，**只要该 run 的选模型口径是 ema** 就额外画一条 EMA
      曲线，并把 best 标记打在 EMA 曲线上 —— 因为 best ckpt 确实是按 EMA 选的；
      传 True/False 可以强制打开/关闭。
    * 没有 history.json 时打印说明并返回 None（不抛异常、不生成空白图）。
    """
    records, skipped = collect_histories(runs_dir)
    for s in skipped:
        print(f"[提示] 训练曲线：{s}", file=sys.stderr)
    if not records:
        print(f"[跳过] {runs_dir} 下没有可画的 history.json，未生成曲线图。"
              f"（这属于正常情况：没跑过训练就没有 history）", file=sys.stderr)
        return None

    if show_ema is None:
        show_ema = any(r["eval_weights"] == "ema" and any(v is not None
                                                          for v in r["ema_val_rmse_db"])
                       for r in records)

    plt, zh = _mpl()
    L = _labels(zh)
    configs = sorted({r["config_name"] for r in records})
    color_of = {c: CONFIG_COLORS[i % len(CONFIG_COLORS)] for i, c in enumerate(configs)}
    # 同一 config 内按 seed 排序决定线型，保证多次运行得到同一张图
    seed_order: dict[str, list] = {}
    for r in records:
        seed_order.setdefault(r["config_name"], [])
        if r["seed"] not in seed_order[r["config_name"]]:
            seed_order[r["config_name"]].append(r["seed"])
    for c in seed_order:
        seed_order[c].sort(key=lambda s: (s is None, s if s is not None else 0))

    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=150)
    marked = 0
    for r in records:
        s = r["seed"]
        ls = LINESTYLES[seed_order[r["config_name"]].index(s) % len(LINESTYLES)]
        color = color_of[r["config_name"]]
        seed_txt = f"seed={s}" if s is not None else "seed=未记录"
        be = r["best_epoch_recorded"] if r["best_epoch_recorded"] is not None \
            else r.get("best_epoch_from_curve")
        sel_name = r.get("selecting_curve_name", "raw val")
        label = f"{r['config_name']} ({seed_txt}) {L['best']}@{be} [{sel_name}]"
        ax.plot(r["epochs"], r["val_rmse_db"], ls, color=color, linewidth=1.6, label=label)
        # EMA 曲线（选模型口径为 ema 时，best ckpt 就是按它选的）
        if show_ema and any(v is not None for v in r["ema_val_rmse_db"]):
            ex = [x for x, v in zip(r["epochs"], r["ema_val_rmse_db"]) if v is not None]
            ey = [v for v in r["ema_val_rmse_db"] if v is not None]
            ax.plot(ex, ey, ls, color=color, linewidth=1.0, alpha=0.5,
                    label=f"{r['config_name']} ({seed_txt}) {L['ema_curve']}")
        # 标出 best epoch：打在**选模型那条曲线**上，避免"图与结论不符"
        curve = _selecting_curve(r)
        if curve and be is not None and int(be) in curve[2]:
            k = curve[2].index(int(be))
            ax.plot([curve[2][k]], [curve[1][k]], marker="o", markersize=6,
                    color=color, zorder=5)
            marked += 1
    ax.set_xlabel(L["epoch"])
    ax.set_ylabel(L["val_rmse"])
    ax.set_title(L["curves_title"])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=1, loc="best")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 训练曲线已写出：{out}（{len(records)} 条曲线，{len(configs)} 个 config，"
          f"{marked} 个 best 标记；EMA 曲线={'开' if show_ema else '关'}）")
    return str(out)


# ------------------------------------------------------------ 单样本三联图

def _to_db(x: np.ndarray) -> np.ndarray:
    """归一化 pathloss → dB；形状统一压成 (H, W)。"""
    a = np.squeeze(np.asarray(x, dtype=np.float64))
    if a.ndim != 2:
        raise ValueError(f"pathloss 图必须是 (H,W) 或 (1,H,W)，收到形状 {np.shape(x)}")
    return pathloss_norm_to_db(a)


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    """建筑掩码 → 0/1。兼容 0/255 的 uint8 与已经是 0/1 的浮点。"""
    m = np.squeeze(np.asarray(mask))
    if m.ndim != 2:
        raise ValueError(f"建筑掩码必须是 (H,W)，收到形状 {np.shape(mask)}")
    m = m.astype(np.float64)
    if m.max() > 1.0:
        m = (m > 127.0).astype(np.float64)
    return m


def plot_sample_triplet(mask: np.ndarray, tx_xy: Sequence[float],
                        target_norm: np.ndarray, pred_norm: np.ndarray,
                        out_path: str | Path, title: str | None = None,
                        include_error: bool = True,
                        rmse_db: float | None = None) -> str:
    """三联图：建筑掩码+Tx | 真值 pathloss(dB) | 预测 pathloss(dB)（可选第 4 格误差）。

    * 真值/预测用**同一个颜色标尺**（vmin/vmax 取 dB 全局区间）：如果各自归一化，
      两张图看起来永远"差不多好"，这正是上一版图表骗过自己的方式。
    * 换算用 `src/data/normalize.pathloss_norm_to_db`，即全项目唯一的 dB 口径；
      colorbar 单位标注为 dB。
    * `rmse_db`：调用方（例如 render_predictions）传入的**权威** RMSE（未裁剪原始预测口径）。
      不传时本函数按"显示图像"自己算一个，并在标题里写明是哪个口径 ——
      同一张图上出现两个不同来源的 RMSE 正是"图与正文对不上"的起点，必须避免。
    """
    plt, zh = _mpl()
    L = _labels(zh)
    bld = _as_binary_mask(mask)
    gt = _to_db(target_norm)
    pr = _to_db(pred_norm)
    err = np.abs(pr - gt)
    tx = (float(tx_xy[0]), float(tx_xy[1]))

    ncol = 4 if include_error else 3
    fig, axes = plt.subplots(1, ncol, figsize=(4.1 * ncol, 4.4), dpi=150)
    axes = np.atleast_1d(axes)

    im0 = axes[0].imshow(bld, cmap="gray", vmin=0, vmax=1)
    axes[0].plot(tx[0], tx[1], marker="*", markersize=16,
                 color="#ff2d2d", markeredgecolor="black", markeredgewidth=0.6)
    axes[0].annotate("Tx", (tx[0], tx[1]), textcoords="offset points", xytext=(6, 6),
                     color="#ff2d2d", fontsize=9)
    axes[0].set_title(L["buildings"])
    # 二值掩码不加 colorbar：它只有 0/1 两个值，colorbar 只会占地方

    im1 = axes[1].imshow(gt, cmap="turbo", vmin=PATHLOSS_MIN, vmax=PATHLOSS_MAX)
    axes[1].set_title(L["gt"])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, label=L["db"])

    im2 = axes[2].imshow(pr, cmap="turbo", vmin=PATHLOSS_MIN, vmax=PATHLOSS_MAX)
    axes[2].set_title(L["pred"])
    fig.colorbar(im2, ax=axes[2], fraction=0.046, label=L["db"])

    if include_error:
        im3 = axes[3].imshow(err, cmap="magma", vmin=0.0)
        axes[3].set_title(L["err"])
        fig.colorbar(im3, ax=axes[3], fraction=0.046, label=L["db_err"])

    rmse_shown = float(np.sqrt(np.mean((pr - gt) ** 2)))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    if rmse_db is not None:
        sub = f"{L['rmse_raw']} = {float(rmse_db):.2f} dB"
    else:
        sub = f"{L['rmse_disp']} = {rmse_shown:.2f} dB"
    if title:
        sub = f"{title}   |   {sub}"
    fig.suptitle(sub, fontsize=11)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# ------------------------------------------------------------ checkpoint 载入

def _load_state(ckpt_path: str | Path) -> dict:
    """读 checkpoint（weights_only=False：里面存着 config 等非张量对象）。"""
    import torch

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{ckpt_path}")
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:  # 老版本 torch 没有 weights_only 参数
        state = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint 顶层必须是 dict，收到 {type(state).__name__}")
    return state


def _build_from_state(state: dict, device: Any = "cpu", in_channels: int | None = None,
                      prior_dim: int | None = None, require_ema: bool = False,
                      verbose: bool = True) -> tuple[Any, dict]:
    """从已载入的 checkpoint state 构造模型并载入权重。返回 (model, info)。"""
    from src.models.factory import build_model
    from src.prior.schema import ALL_GROUPS, vector_dim

    cfg = state.get("config") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    notes: list[str] = []

    # ---- 1. 模型结构：优先 state["config"]["model"]，其次 state["model_cfg"] ----
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        model_cfg = None
    if model_cfg is None:
        mc = state.get("model_cfg")
        if isinstance(mc, dict) and mc:
            model_cfg = mc
            notes.append("config.model 缺失，退回使用 state['model_cfg'] 构造模型")
    if model_cfg is None:
        raise ValueError(
            "checkpoint 里既没有 state['config']['model'] 也没有 state['model_cfg']，"
            "无法确定模型结构。请让训练脚本把 config 一并存进 checkpoint。")
    model_cfg = dict(model_cfg)

    # ---- 2. 输入通道 / 先验维度 ----
    # 优先级：调用方显式给 > checkpoint 顶层记录（train.py 训练当场算的，最可信）
    #         > model_info > config 推导
    info_state = state.get("model_info") or state.get("info") or {}
    if not isinstance(info_state, dict):
        info_state = {}
    if in_channels is None:
        if state.get("in_channels") is not None:
            in_channels = int(state["in_channels"])
        elif info_state.get("in_channels"):
            in_channels = int(info_state["in_channels"])
        else:
            data_cfg = cfg.get("data") or {}
            sectors = bool(data_cfg.get("sectors_channels", False))
            in_channels = int(data_cfg.get("in_channels", 4 + (8 if sectors else 0)))
            notes.append(f"in_channels={in_channels} 由 config.data 推导（非 checkpoint 记录）")
    if prior_dim is None:
        if state.get("prior_dim") is not None:
            prior_dim = int(state["prior_dim"])
        elif str(state.get("prior_mode", (cfg.get("prior") or {}).get("mode", "real"))) == "none":
            prior_dim = 0
        elif info_state.get("prior_dim"):
            prior_dim = int(info_state["prior_dim"])
        else:
            groups = tuple((cfg.get("prior") or {}).get("prior_groups") or ALL_GROUPS)
            prior_dim = vector_dim(groups)
            notes.append(f"prior_dim={prior_dim} 由 config.prior.prior_groups 推导")

    model, build_info = build_model(model_cfg, in_channels=int(in_channels),
                                    prior_dim=int(prior_dim))

    # ---- 3. 权重口径：ema 优先；缺失必须吵 ----
    # train.py 会把"未启用 EMA 时回退 raw"的结果写进 state["eval_weights"]，
    # 所以它比 config.eval.weights（用户声明的意图）更接近事实。
    declared = (state.get("eval_weights")
                or (cfg.get("eval") or {}).get("weights")
                or "ema")
    declared = str(declared).lower()
    raw_sd = state.get("model_state_dict") or state.get("model") or state.get("state_dict")
    if raw_sd is None:
        raise ValueError("checkpoint 里没有模型权重（model_state_dict / model 都缺失）")
    ema_sd = state.get("ema_shadow") or state.get("ema") or state.get("ema_state_dict")

    if declared == "raw":
        used, sd = "raw", raw_sd
        if ema_sd is not None:
            notes.append("checkpoint 声明 eval_weights=raw：即使有 ema_shadow 也按 raw 评测"
                         "（该口径必须与其它 run 一致）")
    elif ema_sd is not None:
        used, sd = "ema", ema_sd
    else:
        used, sd = "raw", raw_sd
        # 上一版的 bug 就在这里：分支悄悄走了 raw，报告却写"用的是 EMA"。
        if require_ema:
            raise RuntimeError(
                f"checkpoint 声明 eval_weights={declared}（要求 EMA），但里面没有 ema_shadow。"
                f"按 AGENTS.md「缺 EMA 报错而不是静默回退」，这里直接失败。"
                f"请改用 --weights raw 口径并统一声明，或重新训练（config train.ema=true）。")
        print("=" * 78, file=sys.stderr)
        print("[警告] checkpoint 声明 eval_weights=ema，但里面**没有** ema_shadow！",
              file=sys.stderr)
        print("       已退回原始权重（实际口径记为 raw）。这与其它 run 的 ema 口径不一致，",
              file=sys.stderr)
        print("       任何跨 run 比较都会混入权重口径差异 —— 请补上 EMA 权重或统一口径。",
              file=sys.stderr)
        print("       （严格模式：加 --require-ema 让它直接报错）", file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        notes.append("EMA 权重缺失，实际使用 raw 权重（与声明的 ema 不一致）")

    sd = {str(k)[7:] if str(k).startswith("module.") else str(k): v
          for k, v in dict(sd).items()}
    try:
        # strict=True：结构不一致就炸。strict=False 会静默吞掉缺键，
        # 而"参数量对齐 + 结构相同"是 real−masked 可解释的前提，不能被悄悄破坏。
        model.load_state_dict(sd, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"权重与模型结构不匹配（strict=True）：{exc}\n"
            f"提示：多半是 config.model 与训练时不一致（base_channels / cot / film.hidden），"
            f"或 in_channels/prior_dim 推导错了（当前 in_channels={in_channels}, "
            f"prior_dim={prior_dim}）。不要用 strict=False 掩盖，请修 config。") from exc

    model.to(device)
    model.eval()
    info = {
        "weights_used": used,                     # ★ 报告的权重口径以此为准
        "eval_weights_declared": declared,
        "in_channels": int(in_channels),
        "prior_dim": int(prior_dim),
        "model_cfg": model_cfg,
        "cot_mode": state.get("cot_mode") or build_info.get("cot_mode"),
        "cot_enabled": bool(state.get("cot_enabled", False)),
        "prior_mode": state.get("prior_mode"),
        "sectors_channels": bool(state.get("sectors_channels", False)),
        "n_params_total": int(build_info.get("n_params_total", 0)),
        "n_params_conditional": int(build_info.get("n_params_conditional", 0)),
        "epoch": state.get("epoch"),
        "notes": notes,
    }
    if verbose:
        print(f"[OK] 载入 checkpoint：{state.get('_path', '')}"
              f"\n     权重口径={used}（checkpoint 声明 {declared}）"
              f"{'，epoch=' + str(info['epoch']) if info['epoch'] is not None else ''}"
              f"\n     in_channels={in_channels}, prior_dim={prior_dim}, "
              f"参数量={info['n_params_total']:,}")
        for n in notes:
            print(f"     [注意] {n}", file=sys.stderr)
    return model, info


def load_model_from_ckpt(ckpt_path: str | Path, device: Any = "cpu",
                         in_channels: int | None = None,
                         prior_dim: int | None = None,
                         require_ema: bool = False) -> tuple[Any, dict]:
    """从 checkpoint 构造模型并载入权重。返回 (model, info)。

    本地实现（项目里没有 `src/eval/checkpoint.py`，也不允许新增）。
    info 里记录**实际使用的权重口径**（ema/raw）、参数量、以及与声明的差异，
    调用方必须把它写进图注/报告 —— 权重口径混用是上一版最隐蔽的 bug。
    """
    state = _load_state(ckpt_path)
    state["_path"] = str(ckpt_path)
    return _build_from_state(state, device=device, in_channels=in_channels,
                             prior_dim=prior_dim, require_ema=require_ema)


# ------------------------------------------------------------ 预测渲染

def _read_sample_list(list_file: str | Path) -> list[str]:
    """读取要渲染的 sample_id 列表。支持 .json（list 或 {"sample_ids": [...]}）与纯文本。"""
    p = Path(list_file)
    if not p.is_file():
        raise FileNotFoundError(f"--list-file 不存在：{p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        raw = json.loads(text)
        if isinstance(raw, dict):
            raw = raw.get("sample_ids") or raw.get("samples") or []
        return [str(x) for x in raw]
    ids = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line.split()[0])
    return ids


def render_predictions(ckpt_path: str | Path, data_root: str | Path | None = None,
                       splits_path: str | Path | None = None,
                       prior_dir: str | Path | None = None,
                       out_path_dir: str | Path = "reports/preds", split: str = "test",
                       list_file: str | Path | None = None, n: int = 8,
                       prior_mode: str = "real", device: str = "cpu",
                       require_ema: bool = False) -> dict:
    """从 checkpoint 载入模型，对 n 个样本画三联图，返回渲染汇总。

    数据集构造参数**照抄 `src/eval/run_eval.py`**（prior_groups / sectors_channels /
    shuffle_seed / freq_ghz / px_per_m / tx_encoding / return_cot_text 全部取自
    checkpoint 里记录的 state），否则"可视化用的输入"与"评测用的输入"会不是同一个东西。

    抽样口径：默认在 split 内**均匀间隔**取 n 个样本（而不是取前 n 个 —— 前 n 个往往
    同属一两张地图，会系统性偏向容易/困难的一侧）。传 `--list-file` 时用给定列表。
    渲染子集的逐样本 RMSE 会写进 index.json，并明确标注"这不是全量测试指标"，
    防止有人把挑出来的好看样本当成整体成绩（上一版的教训）。
    """
    import torch

    from src.data.dataset import RadioMapSeerDataset
    from src.eval import metrics as M

    state = _load_state(ckpt_path)
    state["_path"] = str(ckpt_path)
    model, info = _build_from_state(state, device=device, require_ema=require_ema)
    cfg = state.get("config") or {}
    dcfg = cfg.get("data") or {}
    pcfg = cfg.get("prior") or {}

    ckpt_prior_mode = state.get("prior_mode") or pcfg.get("mode")
    if ckpt_prior_mode and prior_mode != ckpt_prior_mode:
        print(f"[警告] 你要求用 prior_mode={prior_mode} 构造输入，但该 checkpoint 训练时用的是 "
              f"prior_mode={ckpt_prior_mode}。这会把模型从没见过的输入喂给它 —— "
              f"渲染出来的图不能当作该模型的成绩。", file=sys.stderr)

    data_root = data_root or (ROOT / str(dcfg.get("root", "data/radiomapseer")))
    splits_path = splits_path or (ROOT / str(dcfg.get("splits", "data/splits.json")))
    if prior_dir is None and pcfg.get("dir"):
        prior_dir = ROOT / str(pcfg["dir"])

    cot_enabled = info["cot_enabled"] or info.get("cot_mode") not in (None, "none")
    dataset = RadioMapSeerDataset(
        data_root=data_root, splits_path=splits_path, split=split,
        prior_mode=prior_mode, prior_dir=prior_dir,
        prior_groups=tuple(state.get("prior_groups")
                           or pcfg.get("prior_groups") or ("global_num", "global_cat", "sectors")),
        sectors_channels=info["sectors_channels"],
        shuffle_seed=int(pcfg.get("shuffle_seed", 12345)),
        freq_ghz=float(dcfg.get("freq_ghz", 0.9)),
        px_per_m=float(dcfg.get("px_per_m", 1.0)),
        tx_encoding=str(dcfg.get("tx_encoding", "constant")),
        return_cot_text=cot_enabled,
    )
    desc = dataset.describe()
    if desc["in_channels"] != info["in_channels"]:
        raise ValueError(
            f"数据集通道数 {desc['in_channels']} 与模型 in_channels={info['in_channels']} "
            f"不一致（检查 checkpoint 里的 sectors_channels）。强行前向只会得到无意义的结果。")
    if prior_mode in ("real", "shuffled") and desc["prior_dim"] != info["prior_dim"]:
        raise ValueError(
            f"dataset.prior_dim={desc['prior_dim']} 与模型 prior_dim={info['prior_dim']} 不一致，"
            f"FiLM 会吃错维度的向量。")

    # ---- 选样本 ----
    all_ids = [f"{m}_{t}" for m, t in dataset.samples]
    id_to_idx = {sid: i for i, sid in enumerate(all_ids)}
    if list_file:
        want = _read_sample_list(list_file)
        missing = [s for s in want if s not in id_to_idx]
        if missing:
            raise ValueError(f"--list-file 里的 sample_id 不在 {split} split 中（前 5 个）："
                             f"{missing[:5]}。请确认划分文件与 sample_id 是否一致。")
        indices = [id_to_idx[s] for s in want]
        pick_note = f"按 list_file 指定 {len(indices)} 个样本"
    else:
        total = len(dataset)
        k = max(1, min(int(n), total))
        # 均匀间隔：linspace 取整后去重，保证覆盖整个 split 而不是挤在一张地图上
        indices = sorted({int(round(v)) for v in np.linspace(0, total - 1, k)})[:k]
        pick_note = f"在 {split} split 的 {total} 个样本里均匀间隔取 {len(indices)} 个"

    from PIL import Image
    out_dir = Path(out_path_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for i in indices:
        s = dataset[i]
        with torch.no_grad():
            x = s["inputs"].unsqueeze(0).to(device)
            pv = s["prior_vec"].to(device)
            if pv.dim() == 1:
                pv = pv.unsqueeze(0)
            cot_text = [s.get("cot_text", "")] if cot_enabled else None
            # 与 train.py:forward_model 完全一致的调用方式（UNet 也接受这三个参数）
            pred = model(x, pv, cot_text)
        pred_np = pred.squeeze(0).detach().cpu().numpy()
        tgt_np = s["target"].numpy()
        # ★ 指标必须用**未裁剪**的原始预测算，与 run_eval.py / metrics.py 的口径逐位一致；
        #   裁剪只用于显示（图像需要落在 [0,1] 才能正确映射到 dB 色标）。
        #   否则图注上的 RMSE 会和 metrics_<split>.json 里的同一个样本对不上 ——
        #   那正是本项目最想消灭的"图与数字不一致"。
        m = M.sample_metrics(pred_np, tgt_np, s["sat_mask"].numpy())
        pred_disp = np.clip(pred_np, 0.0, 1.0)
        sid = str(s["sample_id"])
        map_id, tx_id = dataset.samples[i]
        mask = np.array(Image.open(Path(dataset.buildings_dir) / f"{map_id}.png").convert("L"))
        # 直接读 antenna json，而不是走 dataset 的私有缓存：可视化和 dataset 用的是同一份坐标
        tx_xy = json.loads(
            (Path(dataset.antenna_json_dir) / f"{map_id}.json").read_text(encoding="utf-8"))[tx_id]
        png = out_dir / f"{len(rendered):02d}_{sid}.png"
        # 标题里**不放** RMSE：RMSE 由 plot_sample_triplet 统一标注，并且必须是
        # metrics.json 同口径的那个值（否则一张图上会同时出现两个 RMSE）。
        title = (f"{sid}  weights={info['weights_used']}  "
                 f"prior={prior_mode}  split={split}")
        plot_sample_triplet(mask, tx_xy, tgt_np, pred_disp, png, title=title,
                            include_error=True, rmse_db=m["rmse_db"])
        rendered.append({
            "sample_id": sid, "map_id": int(map_id), "tx_id": int(tx_id),
            "png": str(png), "rmse_db": m["rmse_db"], "mae_db": m["mae_db"],
            "ssim": m["ssim"], "rmse_db_nonsaturated": m.get("rmse_db_nonsaturated"),
            "n_pixels": m.get("n_pixels"),
        })
        print(f"  [{len(rendered)}/{len(indices)}] {sid}  RMSE={m['rmse_db']:.2f} dB → {png}")

    index = {
        "ckpt": str(ckpt_path),
        "weights_used": info["weights_used"],
        "eval_weights_declared": info["eval_weights_declared"],
        "split": split, "prior_mode": prior_mode,
        "ckpt_trained_prior_mode": ckpt_prior_mode,
        "n_rendered": len(rendered),
        "n_samples_in_split": len(dataset),
        "selection": pick_note,
        "n_params_total": info["n_params_total"],
        "db_bounds": [PATHLOSS_MIN, PATHLOSS_MAX],
        "metric_protocol": "per-sample RMSE 用未裁剪的原始预测计算（与 run_eval.py / "
                           "metrics.py 同口径）；图像显示时才裁剪到 [0,1]。",
        "warning": "本文件只描述**被挑出来渲染的样本子集**，其 RMSE 不可当作全量测试指标；"
                   "全量口径数字必须来自 src/eval/report.py 生成的 reports/summary.md。",
        "samples": rendered,
    }
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"[OK] 渲染完成：{len(rendered)} 张 → {out_dir}"
          f"（索引 {out_dir / 'index.json'}，权重口径={info['weights_used']}）")
    print(f"     注意：{index['warning']}")
    return index


# ------------------------------------------------------------ 自检

def _selfcheck(out_path: str | Path) -> int:
    """不依赖数据与 checkpoint 的自检：造一张合成图，验证三联图能画出来。"""
    h = w = 64
    rng = np.random.RandomState(3)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:30, 10:50] = 255
    mask[40:60, 20:40] = 255
    tx = (16.0, 16.0)
    ys, xs = np.mgrid[0:h, 0:w]
    d = np.sqrt((xs - tx[0]) ** 2 + (ys - tx[1]) ** 2)
    target = np.clip(d / d.max(), 0, 1).astype(np.float64)
    pred = np.clip(target + rng.normal(0, 0.02, size=(h, w)), 0, 1)
    p = plot_sample_triplet(mask, tx, target, pred, out_path, title="selfcheck 合成样本")
    print(f"[OK] 三联图自检通过：{p}")
    return 0


# ------------------------------------------------------------ CLI

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.eval.visualize",
        description="评测可视化（matplotlib 延迟导入）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("curves", help="画各 run 的验证集 RMSE 曲线")
    c.add_argument("--runs", default="runs")
    c.add_argument("--out", default="reports/curves.png")
    c.add_argument("--show-ema", dest="show_ema", action="store_true", default=None,
                   help="强制额外画 EMA 验证曲线（默认：选模型口径为 ema 时自动画）")
    c.add_argument("--no-ema", dest="show_ema", action="store_false",
                   help="强制不画 EMA 曲线")

    p = sub.add_parser("predict", help="从 checkpoint 渲染预测三联图")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", default=None, help="默认取 checkpoint 里的 config.data.root")
    p.add_argument("--splits", default=None, help="默认取 checkpoint 里的 config.data.splits")
    p.add_argument("--prior-dir", default=None,
                   help="默认取 checkpoint 里的 config.prior.dir（prior_mode=real/shuffled 时必需）")
    p.add_argument("--out-dir", default="reports/preds")
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--list-file", default=None, help="指定 sample_id 列表（json 或纯文本）")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--prior-mode", default="real",
                   choices=("none", "masked", "real", "shuffled"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--require-ema", action="store_true",
                   help="缺 EMA 权重时直接报错（AGENTS 口径），而不是警告后退回 raw")

    s = sub.add_parser("selfcheck", help="用合成数据画一张三联图（验证 matplotlib 可用）")
    s.add_argument("--out", default="reports/_viz_selfcheck.png")

    args = ap.parse_args(argv)
    if args.cmd == "curves":
        # 缺 history.json 是"优雅跳过"，不算失败（退出码仍为 0）
        plot_training_curves(args.runs, args.out, show_ema=args.show_ema)
        return 0
    if args.cmd == "predict":
        render_predictions(args.ckpt, args.data_root, args.splits, args.prior_dir,
                           args.out_dir, split=args.split, list_file=args.list_file,
                           n=args.n, prior_mode=args.prior_mode, device=args.device,
                           require_ema=args.require_ema)
        return 0
    return _selfcheck(args.out)


if __name__ == "__main__":
    sys.exit(main())
