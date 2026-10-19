"""CoT 文本编码器：让 `reasoning` 真正对模型可观测。

为什么必须有这个东西
-------------------
上一版项目名叫 CoT-RadioMap，但 `prior_to_vector_normalized()` 只读 11 个结构化字段，
`reasoning` 与 `nlos_description` 在全项目**零引用** —— CoT 对模型完全不可观测。
所以"CoT 有用"这件事在上一版根本没有被检验过（交接文档 §2 问题 1）。

设计取舍：为什么不用预训练句向量
-----------------------------
用 sentence-transformers / BGE 之类当然更强，但会引入 ① 需要联网下载的权重、
② 与训练数据无关的固定表示（不能针对本任务微调）、③ 额外的不确定性。
这里选**可训练的哈希词袋编码器**：无外部依赖、离线可用、确定性、参数量可控，
且可以诚实地报出它带来的收益或没有收益。

分词按中英文分别处理：`reasoning` 实测是中文长文本，如果直接用 `\\w+`，
一整句中文（无空格）会变成一个 token，信息几乎全丢。这里对 ASCII 取词、
对 CJK 取**单字 + 相邻双字**（双字提供少量局部上下文）。

这个编码器只服务一个可证伪的问题（H3）：
    带 CoT 的先验是否优于同一 schema 但不带 CoT 的？
所以它必须与结构化字段的注入方式可比（同样经 FiLM），且参数量要报出来。
"""
from __future__ import annotations

import hashlib
import re

import torch
import torch.nn as nn

# ASCII 词（含数字、百分号、小数点，便于抓住 "0.9GHz"、"12dB" 这类量）
_ASCII_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.%\-]*")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def tokenize(text: str) -> list[str]:
    """ASCII 取词 + CJK 取单字与相邻双字。"""
    if not text:
        return []
    toks: list[str] = [m.group(0).lower() for m in _ASCII_WORD.finditer(text)]
    cjk = _CJK.findall(text)
    toks.extend(cjk)
    toks.extend(cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1))
    return toks


def hash_token(token: str, buckets: int) -> int:
    """稳定哈希到桶。用 blake2b 而不是 Python 的 hash()：
    后者受 PYTHONHASHSEED 影响，会让"同一条先验"在不同进程里落到不同桶。"""
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % buckets


def encode_texts(texts: list[str], buckets: int, max_tokens: int = 512
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """一批文本 → (ids, offsets)，可直接喂 `nn.EmbeddingBag(mode='mean')`。"""
    ids: list[int] = []
    offsets: list[int] = []
    for t in texts:
        offsets.append(len(ids))
        toks = tokenize(t or "")[:max_tokens]
        if not toks:
            # 空文本：给一个固定的"空"桶，保证 EmbeddingBag 不会因 offsets 越界报错
            ids.append(hash_token("<empty>", buckets))
        else:
            ids.extend(hash_token(tok, buckets) for tok in toks)
    return (torch.tensor(ids, dtype=torch.long),
            torch.tensor(offsets, dtype=torch.long))


class HashingTextEncoder(nn.Module):
    """哈希词袋 + MLP → out_dim 维嵌入。"""

    def __init__(self, out_dim: int = 32, buckets: int = 8192, hidden: int = 256,
                 max_tokens: int = 512):
        super().__init__()
        self.out_dim = out_dim
        self.buckets = buckets
        self.max_tokens = max_tokens
        self.emb = nn.EmbeddingBag(buckets, hidden, mode="mean")
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _encode_cached(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        hit = self._cache.get(text)
        if hit is None:
            hit = encode_texts([text], self.buckets, self.max_tokens)
            if len(self._cache) < 20000:
                self._cache[text] = hit
        return hit

    def forward(self, texts: list[str] | str) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        ids_list, off_list = [], []
        base = 0
        for t in texts:
            i, o = self._encode_cached(t or "")
            ids_list.append(i)
            off_list.append(o + base)
            base += i.numel()
        ids = torch.cat(ids_list) if ids_list else torch.zeros(1, dtype=torch.long)
        offsets = torch.cat(off_list) if off_list else torch.zeros(1, dtype=torch.long)
        dev = next(self.parameters()).device
        ids, offsets = ids.to(dev), offsets.to(dev)
        h = self.emb(ids, offsets)
        return self.mlp(h)


if __name__ == "__main__":
    enc = HashingTextEncoder(out_dim=16, buckets=1024, hidden=32)
    demo = [
        "1. 场景分类：建筑密度中等，判定为urban。2. LOS估计：0.9GHz衍射能力强，约45%。",
        "1. 场景分类：密集城区。45度方向有连片建筑，最近距离17m。",
        "",
    ]
    out = enc(demo)
    print("输出形状:", tuple(out.shape), "| 参数量:", sum(p.numel() for p in enc.parameters()))
    print("空文本与正常文本是否可区分:", float((out[0] - out[2]).abs().mean()))
    print("tokenize 示例:", tokenize(demo[1])[:12])
