"""RadioMapSeer Dataset：通道构造、四种 prior 模式、独立 meta 返回。

与上一版的差别（都是踩过的坑）
------------------------------
1. **划分只从 `data/splits.json` 读**（leave-maps-out）。上一版在 dataset 里做
   `RandomState(42).permutation(4000)` 的样本级划分，造成同图不同 Tx 的空间泄漏。
2. **删掉"先验缓存未命中就规则生成并写回先验目录"**。上一版这么做会往先验目录里
   混入程序造的假先验，让"先验质量"这件事无法审计；现在缺文件直接抛错
   （audit 门槛本来就该在训练前拦住）。
3. **返回值统一**。上一版 dataset 返回 4 元组而训练脚本解包 3 元组，
   直接 `ValueError: too many values to unpack`，代码与产物脱节。
   现在统一返回 dict，字段名自解释。
4. **prior 向量按自己的物理量程归一化**。上一版 26 通道版本用的是
   `prior_to_vector()`（未归一化：permittivity≈7、pathloss 50–155）却和 [0,1] 的
   场景图拼在同一个输入里，量纲差两个数量级 —— 这正是 15.30/11.99 dB 那个
   "差 2–3 倍"落差的代码级证据。本实现只走 `prior_to_vector()` 的归一化版本。
5. **额外返回饱和像素掩码**，用于报告"排除噪声地板"的 RMSE。

四种 prior 模式（交接文档 §8.2 的消融阶梯）
------------------------------------------
    none      无先验、无 FiLM 分支          → 纯场景图能做到什么
    masked    有 FiLM 分支、先验全零        → "多了一个分支"本身值多少
    real      真先验                        → 主实验
    shuffled  真先验但在**样本间打乱**      → H2：起作用的是内容还是那条通路
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.normalize import MAP_SIZE, gain_to_pathloss_norm, saturation_mask
from src.data.sectors import obstruction_level, wedge_level_maps
from src.data.splits import load_splits, samples_for_split
from src.prior.schema import (
    ALL_GROUPS,
    AZIMUTHS,
    indices_for_groups,
    load_prior_file,
    prior_to_vector,
    vector_dim,
)

PriorMode = Literal["none", "masked", "real", "shuffled"]
TX_ENCODINGS = ("constant", "blob")


class RadioMapSeerDataset(Dataset):
    """一个 split 的 RadioMapSeer 数据集。"""

    def __init__(
        self,
        data_root: str | Path,
        splits_path: str | Path,
        split: Literal["train", "val", "test"],
        prior_mode: PriorMode = "real",
        prior_dir: str | Path | None = None,
        prior_groups: tuple[str, ...] | list[str] = ALL_GROUPS,
        sectors_channels: bool = False,
        shuffle_seed: int = 12345,
        shuffle_cot: bool = False,
        zero_groups: tuple[str, ...] | list[str] = (),
        freq_ghz: float = 0.9,
        px_per_m: float = 1.0,
        tx_encoding: str = "constant",
        return_cot_text: bool = False,
        allow_missing_priors: bool = False,
        blob_sigma_m: float = 6.0,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.prior_mode = prior_mode
        self.prior_groups = tuple(prior_groups)
        self.sectors_channels = bool(sectors_channels)
        self.shuffle_cot = bool(shuffle_cot)
        self.return_cot_text = bool(return_cot_text)
        self.freq_ghz = float(freq_ghz)
        self.px_per_m = float(px_per_m)
        self.tx_encoding = tx_encoding
        self.blob_sigma_m = float(blob_sigma_m)
        if tx_encoding not in TX_ENCODINGS:
            raise ValueError(f"tx_encoding 必须是 {TX_ENCODINGS}，收到 {tx_encoding!r}")

        self.buildings_dir = self.data_root / "buildings_complete"
        self.gain_dir = self.data_root / "gain" / "carsDPM"
        self.antenna_json_dir = self.data_root / "antenna"

        # sectors 楔形通道是从先验内容铺出来的，所以 none 模式下没有东西可铺。
        # 这里直接拦住，避免出现"配置说不用先验、输入里却带着先验"的隐性泄漏。
        if self.sectors_channels and prior_mode == "none":
            raise ValueError(
                "sectors_channels=True 需要 prior_mode 为 real / shuffled / masked；"
                "prior_mode=none 没有任何先验可铺成通道。"
            )

        splits = load_splits(splits_path)
        self.splits = splits
        self.samples: list[tuple[int, int]] = samples_for_split(splits, split)
        self._prior_dir: Path | None = Path(prior_dir) if prior_dir is not None else None

        # ---- 先验向量预加载 ----
        # none：没有条件分支（prior_dim=0）；masked/real/shuffled 的结构必须**逐位相同**，
        # 差别只在喂进去的是零向量、真向量还是被打乱的向量（交接文档 §8.3 参数量对齐）。
        self.prior_dim = 0 if prior_mode == "none" else vector_dim(self.prior_groups)
        self.prior_vectors: list[np.ndarray] | None = None
        self.cot_texts: list[str] = [""] * len(self.samples)
        self.prior_shuffle_index: np.ndarray | None = None
        self.cot_shuffle_index: np.ndarray | None = None

        if prior_mode in ("real", "shuffled"):
            if prior_dir is None:
                raise ValueError(f"prior_mode={prior_mode} 需要 prior_dir")
            self.prior_vectors = self._load_priors(Path(prior_dir), allow_missing_priors)
        elif prior_mode == "masked":
            self.prior_vectors = [
                np.zeros(self.prior_dim, dtype=np.float32) for _ in self.samples
            ]

        # ---- 错位排列（两个**互相独立**的映射，不能共用一个）----
        #   prior_mode=shuffled → 先验**向量**打乱（H2：内容有用还是通路有用）
        #   shuffle_cot=True    → 只把 CoT **文本**打乱（H3：CoT 内容有用还是容量有用）
        # 两者可以同时开，也可以只开一个。**绝不能让 shuffle_cot 顺带把向量也打乱**，
        # 否则 `real + shuffle_cot` 就变成了 `shuffled` 实验，对照失去意义。
        # 两个排列用不同种子偏移，避免一一对应。
        n = len(self.samples)
        if n > 1:
            if prior_mode == "shuffled":
                self.prior_shuffle_index = self._derangement(n, shuffle_seed)
            if self.shuffle_cot and self.return_cot_text:
                self.cot_shuffle_index = self._derangement(n, shuffle_seed + 1)

        # 缓存（必须在 __init__ 里初始化：放在方法定义之后会被吞进上一个方法体）
        self._mask_cache: dict[int, np.ndarray] = {}
        self._ant_cache: dict[int, list] = {}

        # ---- 容量对齐对照：把指定分组的维度置零（prior_dim 不变）----
        # 用途：`geometry + LLM 电磁` 与 `geometry + 电磁置零` 相比，prior_dim 完全相同，
        # 差值可干净地归因于 LLM 的电磁内容；否则就只能拿 dim=42 的纯几何配置去比，
        # 那里面混着容量效应。
        self.zero_groups = tuple(zero_groups)
        self.zero_index: list[int] = []
        if self.zero_groups:
            if self.prior_mode == "none":
                raise ValueError("zero_groups 对 prior_mode=none 没有意义（没有先验向量）")
            unknown = [g for g in self.zero_groups if g not in self.prior_groups]
            if unknown:
                raise ValueError(
                    f"zero_groups {unknown} 不在 prior_groups {list(self.prior_groups)} 里，"
                    f"无法置零（配置写错了）")
            self.zero_index = indices_for_groups(self.prior_groups, self.zero_groups)

    @staticmethod
    def _derangement(n: int, seed: int) -> np.ndarray:
        """构造一个没有不动点的随机排列（每个样本都拿到别人的东西）。

        为什么不用 `np.random.permutation` 直接当映射：它可能保留若干不动点
        （第 i 个样本拿到自己的先验/文本），那些样本会污染对照组的纯洁性。
        这里取随机顺序里每个元素的**前驱**，n>1 时一定没有不动点。
        """
        rng = np.random.RandomState(seed)
        order = rng.permutation(n)
        der = np.empty(n, dtype=np.int64)
        der[order] = np.roll(order, 1)
        return der

    # ---------- 先验加载 ----------

    def _load_priors(self, prior_dir: Path, allow_missing: bool) -> list[np.ndarray]:
        vecs: list[np.ndarray] = []
        missing: list[str] = []
        for (m, t) in self.samples:
            f = prior_dir / f"{m}_{t}.json"
            if not f.exists():
                missing.append(f"{m}_{t}")
                vecs.append(np.zeros(vector_dim(self.prior_groups), dtype=np.float32))
                continue
            prior, prov = load_prior_file(f)
            vecs.append(np.asarray(prior_to_vector(prior, self.prior_groups), dtype=np.float32))
            self.cot_texts[len(vecs) - 1] = prior.reasoning
        if missing and not allow_missing:
            raise FileNotFoundError(
                f"{len(missing)}/{len(self.samples)} 个样本在 {prior_dir} 里没有先验文件，"
                f"例如 {missing[:5]}。\n"
                f"请先跑 src/prior/generate.py 生成，并让 src/prior/audit.py 通过门槛；"
                f"**不要**用零向量静默兜底（那会让先验实验变成假的）。\n"
                f"确实要做缺失实验时显式传 allow_missing_priors=True。"
            )
        return vecs

    # ---------- 缓存读取 ----------

    def _mask(self, map_id: int) -> np.ndarray:
        if map_id not in self._mask_cache:
            p = self.buildings_dir / f"{map_id}.png"
            self._mask_cache[map_id] = np.array(Image.open(p).convert("L"))
        return self._mask_cache[map_id]

    def _antennas(self, map_id: int) -> list:
        if map_id not in self._ant_cache:
            p = self.antenna_json_dir / f"{map_id}.json"
            self._ant_cache[map_id] = json.loads(p.read_text(encoding="utf-8"))
        return self._ant_cache[map_id]

    # ---------- 通道构造 ----------

    def _tx_channels(self, tx_xy: tuple[float, float]) -> np.ndarray:
        """Tx 位置通道（2, H, W）。

        constant：上一版的做法，把归一化坐标广播成常量图（便于与 5.11 dB 的旧 baseline 对比）。
        blob：以 Tx 为中心的高斯斑，空间信息更强，作为可选改进项。
        """
        h = w = MAP_SIZE
        x0, y0 = float(tx_xy[0]), float(tx_xy[1])
        out = np.zeros((2, h, w), dtype=np.float32)
        if self.tx_encoding == "constant":
            out[0] = x0 / w
            out[1] = y0 / h
            return out
        ys, xs = np.mgrid[0:h, 0:w]
        d2 = (xs - x0) ** 2 + (ys - y0) ** 2
        blob = np.exp(-d2 / (2.0 * self.blob_sigma_m ** 2)).astype(np.float32)
        out[0] = blob * (x0 / w)
        out[1] = blob * (y0 / h)
        return out

    def _sector_channels(self, prior_vec_index: int) -> np.ndarray:
        """由先验的 sectors 生成 (8, H, W) 楔形通道。

        * `masked` 模式下必须返回**全零**通道：这一档的语义是"结构相同、先验信息为零"，
          如果楔形通道偷用了真先验，`masked` 就变成了半个 `real`，`real − masked` 失去意义。
        * `shuffled` 模式下必须用**被打乱后**那条先验的 sectors，否则打乱对照不干净。
        """
        if self.prior_mode == "masked":
            return np.zeros((len(AZIMUTHS), MAP_SIZE, MAP_SIZE), dtype=np.float32)
        idx = prior_vec_index
        if self.prior_shuffle_index is not None:
            idx = int(self.prior_shuffle_index[idx])
        m, t = self.samples[idx]
        f = self._prior_file(m, t)
        prior, _ = load_prior_file(f)
        levels = {s.azimuth_deg: obstruction_level(s.obstruction) for s in prior.sectors}
        return wedge_level_maps(levels, MAP_SIZE, MAP_SIZE, self._antennas(m)[t])

    def _prior_file(self, map_id: int, tx_id: int) -> Path:
        if self._prior_dir is None:
            raise RuntimeError("内部错误：prior_dir 未设置")
        return self._prior_dir / f"{map_id}_{tx_id}.json"

    # ---------- Dataset 接口 ----------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        map_id, tx_id = self.samples[index]
        tx_xy = self._antennas(map_id)[tx_id]

        # 场景图（1 通道，[0,1]）
        mask = self._mask(map_id)
        scene = (mask.astype(np.float32) / 255.0)[None, :, :]

        # Tx（2）+ 频率（1）
        tx_ch = self._tx_channels(tx_xy)
        freq_norm = min(self.freq_ghz / 28.0, 1.0)
        freq_ch = np.full((1, MAP_SIZE, MAP_SIZE), freq_norm, dtype=np.float32)
        chans = [scene, tx_ch, freq_ch]

        # 可选的 sectors 楔形通道
        if self.sectors_channels:
            chans.append(self._sector_channels(index))

        inputs = torch.from_numpy(np.concatenate(chans, axis=0))

        # 目标 + 饱和掩码
        gain_path = self.gain_dir / f"{map_id}_{tx_id}.png"
        gain_pixel = np.array(Image.open(gain_path).convert("L"))
        target = torch.from_numpy(gain_to_pathloss_norm(gain_pixel)[None, :, :])
        sat = torch.from_numpy(saturation_mask(gain_pixel)[None, :, :])

        # 先验向量（只有 shuffled 模式才走错位映射；shuffle_cot 不影响它）
        if self.prior_vectors is None:
            prior_vec = torch.zeros(1, dtype=torch.float32)   # 占位，模型不使用
        else:
            src = index if self.prior_shuffle_index is None else int(self.prior_shuffle_index[index])
            vec = self.prior_vectors[src].copy()
            if self.zero_index:
                vec[self.zero_index] = 0.0
            prior_vec = torch.from_numpy(vec)

        out: dict[str, Any] = {
            "inputs": inputs,
            "target": target,
            "sat_mask": sat,
            "prior_vec": prior_vec,
            "map_id": torch.tensor(map_id, dtype=torch.long),
            "tx_id": torch.tensor(tx_id, dtype=torch.long),
            "sample_id": f"{map_id}_{tx_id}",
        }
        if self.return_cot_text:
            ci = index
            if self.cot_shuffle_index is not None:
                ci = int(self.cot_shuffle_index[index])
            out["cot_text"] = self.cot_texts[ci]
        return out

    # ---------- 便捷信息 ----------

    def describe(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "n_samples": len(self.samples),
            "n_maps": len(set(m for m, _ in self.samples)),
            "prior_mode": self.prior_mode,
            "prior_dim": self.prior_dim,
            "prior_groups": list(self.prior_groups),
            "sectors_channels": self.sectors_channels,
            "in_channels": 4 + (8 if self.sectors_channels else 0),
            "tx_encoding": self.tx_encoding,
            "return_cot_text": self.return_cot_text,
            "shuffle_cot": self.shuffle_cot,
        }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """统一的 DataLoader 构造：固定 generator 与 worker 种子。"""
    from src.utils.seed import make_generator, worker_init_fn

    kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    if shuffle:
        kwargs["generator"] = make_generator(seed)
    if num_workers > 0:
        kwargs["worker_init_fn"] = worker_init_fn(seed)
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)
