#!/usr/bin/env python3
"""把 PDF 抽成纯文本（过滤代理字符），供检索用。

为什么单独写个脚本：PDF 里常有数学字体的代理对字符（\\ud835 等），
直接 write 会抛 UnicodeEncodeError；而且从 PowerShell 传多行 Python 很容易被引号吃掉。
"""
from __future__ import annotations

import sys
from pathlib import Path

from pypdf import PdfReader


def clean(s: str) -> str:
    """去掉代理对与其它无法编码的字符，换成 '?'。"""
    return s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    reader = PdfReader(str(src))
    chunks: list[str] = []
    for i, page in enumerate(reader.pages):
        chunks.append(f"\n===== PAGE {i + 1} =====\n")
        try:
            chunks.append(clean(page.extract_text() or ""))
        except Exception as exc:  # noqa: BLE001
            chunks.append(f"[本页抽取失败: {type(exc).__name__}: {exc}]")
    text = "".join(chunks)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8", errors="replace")
    print(f"页数 {len(reader.pages)}｜写出 {len(text)} 字符 → {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
