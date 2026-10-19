"""leave-maps-out 划分的读取、校验与哈希。

为什么按地图切（交接文档 §4.3）：
    上一版是样本级 `RandomState(42).permutation(4000)`，同一张地图的其它 Tx 大量
    出现在训练集里。对"建筑掩码 + Tx 位置 → pathloss 图"这个任务，模型可以记住该
    地图的建筑布局，测试时只是换一个 Tx 位置做插值 —— 5.11 dB 因此不代表任何泛化能力。

规则（AGENTS.md 规则 3）：
    * 划分单位只能是 map；任何按样本划分的代码都不允许存在；
    * `splits.json` 冻结后不许改，改动必须换文件名并记进 runs/*/config.yaml；
    * val 用于早停/选模型，test 只在最后评一次。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

SPLIT_NAMES = ("train", "val", "test")


class SplitsError(ValueError):
    """划分文件不合法。"""


def load_splits(path: str | Path) -> dict:
    """读取并校验 splits.json；不合法直接抛错（不静默容忍）。"""
    p = Path(path)
    if not p.is_file():
        raise SplitsError(f"划分文件不存在: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))

    required = {"split_unit", "n_tx_per_map", "train_maps", "val_maps", "test_maps"}
    missing = required - set(raw)
    if missing:
        raise SplitsError(f"{p} 缺少字段 {sorted(missing)}")

    if raw["split_unit"] != "map":
        raise SplitsError(
            f"split_unit 必须是 'map'，实际是 {raw['split_unit']!r}："
            f"按样本划分会造成空间泄漏（见交接文档 §4.3）"
        )

    for name in SPLIT_NAMES:
        key = f"{name}_maps"
        if not isinstance(raw[key], list) or not raw[key]:
            raise SplitsError(f"{key} 必须是非空列表")

    groups = {n: set(raw[f"{n}_maps"]) for n in SPLIT_NAMES}
    for a in SPLIT_NAMES:
        for b in SPLIT_NAMES:
            if a < b:
                overlap = groups[a] & groups[b]
                if overlap:
                    raise SplitsError(f"{a} 与 {b} 的地图重叠: {sorted(overlap)}")

    all_maps = set(raw.get("all_maps", []))
    if all_maps:
        union = set().union(*groups.values())
        if union != all_maps:
            raise SplitsError(
                f"三个 split 的并集 != all_maps；"
                f"多出 {sorted(union - all_maps)}，缺失 {sorted(all_maps - union)}"
            )

    # 数量自洽性：counts 必须等于 maps × n_tx_per_map
    n_tx = int(raw["n_tx_per_map"])
    counts = raw.get("counts", {})
    for name in SPLIT_NAMES:
        expect = len(raw[f"{name}_maps"]) * n_tx
        got = counts.get(f"{name}_samples")
        if got is not None and int(got) != expect:
            raise SplitsError(
                f"counts.{name}_samples = {got}，但 {len(raw[f'{name}_maps'])} maps × "
                f"{n_tx} tx = {expect}"
            )
    return raw


def splits_hash(path: str | Path) -> str:
    """划分文件的内容哈希（写进 config，保证实验可追溯到具体划分）。"""
    p = Path(path)
    canonical = json.dumps(json.loads(p.read_text(encoding="utf-8")),
                           sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def samples_for_split(
    splits: dict, split: str, tx_per_map: int | None = None
) -> list[tuple[int, int]]:
    """返回该 split 的 (map_id, tx_id) 列表，顺序确定（map 升序、tx 升序）。"""
    if split not in SPLIT_NAMES:
        raise SplitsError(f"split 必须是 {SPLIT_NAMES} 之一，收到 {split!r}")
    n_tx = int(tx_per_map if tx_per_map is not None else splits["n_tx_per_map"])
    out: list[tuple[int, int]] = []
    for map_id in sorted(splits[f"{split}_maps"]):
        for tx_id in range(n_tx):
            out.append((int(map_id), int(tx_id)))
    return out


def assert_disjoint_sample_lists(named: dict[str, Iterable[tuple[int, int]]]) -> None:
    """双保险：断言各 split 的样本列表两两不相交（不只信 map 级不相交）。"""
    seen: dict[tuple[int, int], str] = {}
    for name, samples in named.items():
        for s in samples:
            if s in seen:
                raise SplitsError(f"样本 {s} 同时出现在 {seen[s]} 与 {name}")
            seen[s] = name
