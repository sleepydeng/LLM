#!/usr/bin/env python3
"""便捷入口：`python scripts/audit.py ...` → `src.prior.audit.main()`。

把项目根加入 sys.path，这样脚本从任意工作目录都能直接跑。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prior.audit import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
