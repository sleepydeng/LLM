"""配置加载：base.yaml + 实验 yaml 的深合并，以及"把一切写进 runs/<exp_id>/"。

约定（AGENTS.md「目录与命名」）：
    * 一次实验 = 一个 `runs/<exp_id>/`，里面必须有 config.yaml / train.log / ckpt/ / metrics.json；
    * 所有路径写在 config 里，禁止硬编码绝对路径；
    * config 必须记录 seed、数据划分哈希、git commit —— 让该实验能独立复现。
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并：override 覆盖 base，dict 逐层合并，其它类型直接替换。"""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, _seen: set[str] | None = None) -> dict:
    """读取 yaml；若含 `_base_` 则先加载基座再深合并。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"配置不存在: {p}")
    _seen = _seen or set()
    key = str(p.resolve())
    if key in _seen:
        raise ValueError(f"配置 _base_ 出现循环引用: {key}")
    _seen.add(key)

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    base_ref = raw.pop("_base_", None)
    cfg: dict[str, Any]
    if base_ref:
        base_path = (p.parent / base_ref).resolve()
        cfg = deep_merge(load_config(base_path, _seen), raw)
    else:
        cfg = raw
    cfg["_config_path"] = str(p)
    cfg["_config_name"] = p.stem
    return cfg


def git_rev(root: str | Path) -> str | None:
    """当前 git commit（没有 git 就返回 None，不阻塞）。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root), capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _to_plain(obj: Any) -> Any:
    """把任意对象拍平成 YAML/JSON 一定能序列化的原生类型。

    为什么需要它：运行期元数据里很容易混进非原生类型 ——
    `torch.__version__` 其实是 `TorchVersion`（str 的子类，PyYAML 认不出）、
    指标里是 `np.float32`、参数量是 `np.int64`。这些都会让 `yaml.safe_dump`
    直接抛 `RepresenterError`，而且是在"训练已经跑完、准备落盘配置"的时刻才炸。
    所以在落盘边界统一转换，比在每个调用点各自 `float()` 更可靠。
    """
    import numpy as _np

    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, dict):
        return {_to_plain(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, _np.generic):          # np.float32 / np.int64 / np.bool_ ...
        return obj.item()
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, str):
        return str(obj)          # 关键：把 str 子类拍成普通 str
    return str(obj)


def freeze_run_dir(run_dir: str | Path, cfg: dict, extra: dict | None = None) -> Path:
    """把最终 config（含运行期补充信息）写进 runs/<exp_id>/config.yaml。

    extra 用来塞入 seed、splits 哈希、git rev、参数量、数据统计等。
    """
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    payload = _to_plain(copy.deepcopy(cfg))
    if extra:
        payload["_runtime"] = _to_plain(extra)
    (d / "config.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (d / "config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return d


def rel_or_abs(path: str | Path, root: str | Path) -> str:
    """尽量给出相对 root 的 posix 路径；不在 root 之下时退回绝对路径。

    为什么需要：训练/评测目录可能被放到项目之外（例如冒烟测试用临时目录），
    直接 `Path.relative_to(ROOT)` 会在"训练已经跑完"的时刻抛 ValueError。
    """
    p = Path(path)
    try:
        return p.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_path(cfg: dict, key_path: str, root: str | Path) -> Path:
    """从 config 取嵌套路径（点号分隔）并相对项目根解析。"""
    node: Any = cfg
    for part in key_path.split("."):
        node = node[part]
    p = Path(node)
    return p if p.is_absolute() else (Path(root) / p)


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}
