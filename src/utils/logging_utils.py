"""训练日志：同时输出到 stdout 与 runs/<exp_id>/train.log。"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str | Path | None = None,
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        fp = Path(log_file)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(fp, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
