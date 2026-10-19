"""随机种子与 DataLoader 可复现性。

上一版三个训练脚本**都没有设任何种子**，`train_config.json` 里也没有 seed 字段，
所以 +0.25 dB 的"提升"无法与单次运行噪声区分（交接文档 §2 问题 2）。
这里把它变成训练入口的强制第一步（AGENTS.md 规则 2）。
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """固定所有随机源。

    deterministic=True 时打开 cudnn 确定性、关闭 benchmark，代价是慢一些；
    要求 bit 级复现时保持 True（交接文档 §3.3）。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch 2.x：某些算子需要显式允许确定性算法
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # noqa: BLE001
            pass
    else:
        torch.backends.cudnn.benchmark = True


def make_generator(seed: int) -> torch.Generator:
    """DataLoader 的 shuffle 也要用固定种子的 generator。"""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(seed: int):
    """DataLoader worker 的种子：`seed + worker_id`，避免各 worker 用同一条随机流。"""
    def _fn(worker_id: int) -> None:
        s = seed + worker_id
        np.random.seed(s)
        random.seed(s)
    return _fn


def describe_seed(seed: int, deterministic: bool) -> dict:
    """写进 config 的种子记录。

    注意 `torch.__version__` 是 `TorchVersion`（str 的子类），这里用 f-string 拍成普通 str，
    否则落盘时 PyYAML 会报 RepresenterError。
    """
    return {"seed": int(seed), "deterministic": bool(deterministic),
            "torch": f"{torch.__version__}"}
