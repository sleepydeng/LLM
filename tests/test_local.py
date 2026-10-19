#!/usr/bin/env python3
"""本地自检：不需要 GPU、不需要 API key，把能在本机验证的全部验证掉。

运行：
    python -m tests.test_local            # 全部
    python -m tests.test_local ssim       # 只跑名字里含 ssim 的项

覆盖：
  1. schema v2    —— 分组向量维度、量程归一化、非法值必须抛错、sectors 方位校验
  2. normalize    —— gain↔pathloss 映射、噪声地板、饱和掩码、dB 换算
  3. splits       —— learn-maps-out 校验、不相交、哈希稳定
  4. sectors      —— 方位约定与几何判定（用合成掩码验证方向不能搞反）
  5. prompt       —— 渲染尺寸、消息结构、**确认没有 scene_hint 泄漏**
  6. dataset      —— 四种 prior 模式、通道数、缺失先验必须抛错、shuffled 无不动点
  7. models       —— 参数量、FiLM identity 初始化、前向形状、参数量对齐
  8. losses/SSIM  —— torch 版 SSIM 与 numpy 版对拍、损失可反传
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

try:  # Windows 控制台默认 GBK，统一按 UTF-8 输出，否则中文报告全是乱码
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import normalize as N
from src.data.sectors import (
    angular_index_map,
    geometric_obstruction,
    local_building_ratio,
    obstruction_level,
    wedge_level_maps,
)
from src.data.splits import load_splits, samples_for_split, splits_hash
from src.prior import schema as S
from src.prior.prompt_v2 import (
    SYSTEM_PROMPT,
    build_messages,
    render_panels,
    render_prompt_image,
)

DATA_ROOT = ROOT / "data" / "radiomapseer"
PASS, FAIL = [], []


def check(name: str, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  [OK]   {name}")
    except Exception as exc:  # noqa: BLE001
        FAIL.append((name, exc))
        print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")


def make_prior(seed: int = 0, obstruction: str = "dense") -> dict:
    rng = np.random.RandomState(seed)
    return {
        "scene_type": S.SCENE_TYPES[seed % len(S.SCENE_TYPES)],
        "los_ratio": float(rng.uniform(0.1, 0.6)),
        "reflection_ratio": 0.25, "diffraction_ratio": 0.3, "scattering_ratio": 0.15,
        "dominant_material": S.MATERIALS[seed % len(S.MATERIALS)],
        "permittivity": float(rng.uniform(2, 10)),
        "penetration_loss_db": float(rng.uniform(5, 25)),
        "pathloss_min_db": float(rng.uniform(45, 75)),
        "pathloss_max_db": float(rng.uniform(120, 180)),
        "confidence": S.CONFIDENCES[seed % 3],
        "tx_local_building_ratio_50m": float(rng.uniform(0, 1)),
        "tx_local_building_ratio_100m": float(rng.uniform(0, 1)),
        "sectors": [
            {"azimuth_deg": a, "obstruction": S.OBSTRUCTIONS[(seed + i) % 4],
             "nearest_building_m": float((seed + i) * 7 % 200)}
            for i, a in enumerate(S.AZIMUTHS)
        ] if obstruction is None else [
            {"azimuth_deg": a, "obstruction": obstruction,
             "nearest_building_m": 12.0 + i}
            for i, a in enumerate(S.AZIMUTHS)
        ],
        "reasoning": f"自检用先验 seed={seed}：0.9GHz，Tx 周围建筑遮挡主要集中在 180 度方向。",
    }


# ---------------------------------------------------------------- 1. schema

def t_schema_dims():
    p = S.RadioPriorV2(**make_prior(1, "dense"))
    assert len(S.prior_to_vector(p, S.ALL_GROUPS)) == S.vector_dim(S.ALL_GROUPS) == 64
    assert len(S.prior_to_vector(p, (S.GROUP_GLOBAL_NUM,))) == 10
    assert len(S.prior_to_vector(p, (S.GROUP_GLOBAL_CAT,))) == 14
    assert len(S.prior_to_vector(p, (S.GROUP_SECTORS,))) == 40


def t_schema_range():
    p = S.RadioPriorV2(**make_prior(2))
    v = S.prior_to_vector(p)
    assert all(0.0 <= x <= 1.0 for x in v), f"向量越界: {[x for x in v if not 0 <= x <= 1]}"


def t_schema_rejects_bad():
    for bad, why in [
        ({"los_ratio": 1.5}, "越界"),
        ({"permittivity": 0.5}, "低于下限"),
        ({"pathloss_min_db": 200.0}, "min 不小于 max"),
        ({"sectors": []}, "sectors 数量不对"),
        ({"sectors": [{"azimuth_deg": a, "obstruction": "open",
                       "nearest_building_m": 1.0} for a in (0, 45, 90, 135, 180, 225, 270, 300)]},
         "方位角不合法"),
    ]:
        d = make_prior(3)
        d.update(bad)
        try:
            S.RadioPriorV2(**d)
        except Exception:  # noqa: BLE001
            continue
        raise AssertionError(f"非法输入没有被拦截（{why}）: {bad}")


def t_schema_extra_forbidden():
    d = make_prior(4)
    d["scene_hint"] = "urban"          # 上一版就是把这个喂给 LLM 的
    try:
        S.RadioPriorV2(**d)
    except Exception:  # noqa: BLE001
        return
    raise AssertionError("多余字段 scene_hint 没有被拦截")


def t_schema_record_roundtrip():
    rec = S.PriorRecord(sample_id="3_7", map_id=3, tx_id=7,
                        prior=S.RadioPriorV2(**make_prior(5)),
                        provenance={"model": "selfcheck"})
    again = S.PriorRecord(**json.loads(rec.to_json()))
    assert again.sample_id == "3_7" and again.prior.sectors[0].azimuth_deg == 0


# ------------------------------------------------------------- 2. normalize

def t_normalize_mapping():
    g = np.array([[255, 0], [51, 102]], dtype=np.uint8)  # 51/255 = 0.2 恰好在地板上
    pl = N.gain_to_pathloss_norm(g)
    assert abs(pl[0, 0] - 0.0) < 1e-6, f"gain=255 应映射到 pathloss=0，得到 {pl[0,0]}"
    assert abs(pl[0, 1] - 1.0) < 1e-6, f"gain=0（噪声地板下）应映射到 1，得到 {pl[0,1]}"
    assert abs(pl[1, 0] - 1.0) < 1e-6, "gain 恰好等于地板时也应饱和（pl 恰为 1.0）"
    # gain=102 → 0.4 → (0.4-0.2)/0.8 = 0.25 → pl = 0.75
    assert abs(pl[1, 1] - 0.75) < 1e-5, f"gain=102 应映射到 0.75，得到 {pl[1,1]}"


def t_normalize_roundtrip():
    # 只在噪声地板以上做往返：地板以下的 gain 被有意裁剪到 51，本来就不可还原
    g = np.random.randint(52, 256, size=(64, 64)).astype(np.uint8)
    back = N.pathloss_norm_to_gain_pixel(N.gain_to_pathloss_norm(g))
    assert np.abs(back.astype(int) - g.astype(int)).max() <= 1
    # 地板以下必须被裁剪到地板，而不是被"还原"成原值
    low = np.array([[0, 10, 51]], dtype=np.uint8)
    assert N.pathloss_norm_to_gain_pixel(N.gain_to_pathloss_norm(low)).max() == 51


def t_saturation_mask():
    g = np.array([[255, 0, 51, 52]], dtype=np.uint8)
    m = N.saturation_mask(g)
    # gain<=51 的全部 pl 都恰好等于 1.0，都属于饱和像素（边界那一列不能漏）
    assert m.tolist() == [[False, True, True, False]], m.tolist()
    pl = N.gain_to_pathloss_norm(g)
    assert np.array_equal(m, pl >= 1.0 - 1e-7), "饱和掩码必须与 pathloss 映射严格一致"


def t_db_span():
    assert N.db_span() == 160.0
    assert abs(N.norm_rmse_to_db(0.032) - 5.12) < 0.01


# ---------------------------------------------------------------- 3. splits

def t_splits_ok():
    sp = load_splits(ROOT / "data" / "splits.json")
    assert sp["split_unit"] == "map"
    a = set(sp["train_maps"]); b = set(sp["val_maps"]); c = set(sp["test_maps"])
    assert not (a & b) and not (a & c) and not (b & c)
    assert len(samples_for_split(sp, "train")) == sp["counts"]["train_samples"] == 2800
    assert len(samples_for_split(sp, "test")) == 800


def t_splits_hash_stable():
    h1 = splits_hash(ROOT / "data" / "splits.json")
    h2 = splits_hash(ROOT / "data" / "splits.json")
    assert h1 == h2 and len(h1) == 16


def t_splits_reject_sample_level():
    tmp = Path(tempfile.mkdtemp())
    try:
        bad = {"split_unit": "sample", "n_tx_per_map": 80,
               "train_maps": [0], "val_maps": [1], "test_maps": [2], "all_maps": [0, 1, 2]}
        f = tmp / "bad.json"
        f.write_text(json.dumps(bad), encoding="utf-8")
        try:
            load_splits(f)
        except Exception:  # noqa: BLE001
            return
        raise AssertionError("按样本划分的 splits.json 没有被拒绝")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------- 4. sectors

def t_angular_map_partition():
    idx, r, az = angular_index_map(64, 64, (32, 32))
    assert idx.shape == (64, 64) and r.shape == (64, 64)
    assert idx.min() >= 0 and idx.max() <= 7
    assert abs(az[32, 40] - 0.0) < 1e-6, "正右方应为 0 度"
    assert abs(az[20, 32] - 90.0) < 1e-6, "正上方应为 90 度（y 轴向下，故要取负）"
    assert abs(az[32, 20] - 180.0) < 1e-6
    assert abs(az[45, 32] - 270.0) < 1e-6


def t_geometric_obstruction_direction():
    """合成的"东侧有墙"掩码：0 度必须被判为有遮挡，180 度必须 open。

    这条测试专门防止方位约定搞反 —— 上一版没有 sectors 字段，
    但新加的字段如果方向搞反，一致率检查会全部失效且很难发现。
    """
    m = np.zeros((128, 128), dtype=np.uint8)
    m[40:90, 90:128] = 255            # 东侧（+x）一堵墙，距中心 26 px
    geo = geometric_obstruction(m, (64, 64), radius_m=60.0, px_per_m=1.0)
    assert geo[0]["obstruction"] in ("dense", "blocked"), geo[0]
    assert geo[180]["obstruction"] == "open", geo[180]
    assert geo[90]["obstruction"] == "open"
    assert geo[270]["obstruction"] == "open"
    assert abs(geo[0]["nearest_building_m"] - 26.0) < 2.0, geo[0]


def t_wedge_maps():
    levels = {a: obstruction_level("blocked" if a == 0 else "open") for a in S.AZIMUTHS}
    wm = wedge_level_maps(levels, 64, 64, (32, 32))
    assert wm.shape == (8, 64, 64)
    assert wm[0].max() == 1.0 and wm[1].max() == 0.0
    assert wm[0, 32, 50] == 1.0, "0 度扇区应覆盖正右方像素"


def t_local_building_ratio():
    m = np.zeros((128, 128), dtype=np.uint8)
    m[:64, :] = 255                   # 上半为建筑
    r = local_building_ratio(m, (64, 64), 20.0)
    assert 0.3 < r < 0.7, f"中心附近约一半应是建筑，得到 {r}"


# ---------------------------------------------------------------- 5. prompt

def t_prompt_render():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"缺少数据 {DATA_ROOT}")
    from PIL import Image
    mask = np.array(Image.open(DATA_ROOT / "buildings_complete" / "0.png").convert("L"))
    tx = json.loads((DATA_ROOT / "antenna" / "0.json").read_text(encoding="utf-8"))[0]
    img = render_prompt_image(mask, tx)
    assert img.size == (256 + 12 + 400, 400), img.size      # 左256 + 间隔 + 右400


def t_prompt_no_scene_hint():
    assert "scene_hint" not in SYSTEM_PROMPT
    assert "0.9 GHz" in SYSTEM_PROMPT or "0.9GHz" in SYSTEM_PROMPT
    # 必须明确禁止"更亮=更高"，因为掩码是二值的
    assert "NO height information" in SYSTEM_PROMPT
    assert "28" in SYSTEM_PROMPT      # 明确要求不要谈 28GHz


def t_prompt_local_panel_alignment():
    """局部面板必须逐像素等于全局掩码在 Tx 周围的裁剪，且方向不能翻转。

    这条测试是方位约定的"锚"：如果裁剪发生翻转或平移，LLM 在图上看到的
    0°/90° 方向就与 `src/data/sectors.py` 和 audit 算的不是同一个方向，
    sector 一致率检查会整体失效而且极难察觉。
    """
    from PIL import Image
    mask = np.array(Image.open(DATA_ROOT / "buildings_complete" / "0.png").convert("L"))
    tx = json.loads((DATA_ROOT / "antenna" / "0.json").read_text(encoding="utf-8"))[7]
    left_img, right_img = render_panels(mask, tx, zoom=1, guides=False, captions=False)
    left = np.array(left_img)[:, :, 0]
    right = np.array(right_img)[:, :, 0]

    g = np.where(mask > 127, 255, 0).astype(np.uint8)
    assert np.array_equal(left, g), "左侧全局面板必须就是二值掩码本身"

    pad = 100
    padded = np.pad(g, ((pad, pad), (pad, pad)))
    px, py = int(round(tx[0])) + pad, int(round(tx[1])) + pad
    exp = padded[py - pad:py + pad, px - pad:px + pad]

    assert right.shape == (200, 200), right.shape
    assert np.array_equal(right, exp), "局部面板与期望裁剪不一致（方向/对齐错了）"

    # 0°（东）方向在局部图里必须是"往右"，90°（北）必须是"往上"
    cx = cy = 100
    assert np.array_equal(right[cy, cx + 60:cx + 61], padded[py, px + 60:px + 61]), \
        "0° 方向应指向 +x"
    assert np.array_equal(right[cy - 60:cy - 59, cx], padded[py - 60:py - 59, px]), \
        "90° 方向应指向 −y（图上方）"
    # zoom=2 时像素翻倍，中心必须仍对齐 Tx
    _, right2_img = render_panels(mask, tx, zoom=2, guides=False, captions=False)
    right2 = np.array(right2_img)[:, :, 0]
    assert right2.shape == (400, 400)
    assert np.array_equal(right2[::2, ::2], right), "zoom=2 必须只是最近邻放大两倍"
    assert right2[200, 200] == exp[100, 100], "局部图中心必须是 Tx 所在像素"


def t_schema_rejects_height_fields():
    """任何 height 字段都必须被 schema 拒绝。

    背景（上一版的根因，值得完整记下来）：
      `pipeline/generate_synthetic_data.py` 生成合成数据时**真的有**一个建筑高度变量
      （`building_height` 米、每类场景 `min_height/max_height`、`tx_height` 15–35 m），
      并且把高度图用 `height / height.max() * 255` 转成灰度图 —— 所以在合成数据上
      "更亮的像素 = 更高的建筑"**是真的**。
      但换成真实的 RadioMapSeer 建筑掩码（严格二值 {0,255}，无高度通道）之后，
     这套 prompt 没有跟着改：`prompts/system_prompt.py:16` 仍然声明 Tx 有 z 高度，
     第 146-149 行还给了一张"密集城区 20-50m / 郊区 5-12m"的高度区间表，
      第 172 行甚至要求"结合场景类型提示和常识进行推理"高度。
      结果是 4000 条先验里 81.65% 写出了凭空的建筑高度。

    而 `RadioPrior` schema **从来没有** height 字段，所以这件事 schema 校验拦不住 ——
    它只活在 prompt 与 CoT 文本里。因此新 schema 用 `extra="forbid"` 把这条路彻底堵死：
    任何试图把高度塞回先验的改动都会立刻报错。
    """
    for field in ("height", "building_height", "building_height_m", "mean_height_m",
                  "tx_height", "tx_height_m", "tx_height_above_roof", "height_m",
                  "building_height_map", "z"):
        d = make_prior(1)
        d[field] = 20.0
        try:
            S.RadioPriorV2(**d)
        except Exception:  # noqa: BLE001
            continue
        raise AssertionError(f"高度类字段 {field!r} 没有被 schema 拒绝")


def t_prompt_has_no_height_instruction():
    """prompt 里不得出现任何"引导模型推理建筑高度"的内容。

    上一版 prompt 是**自相矛盾**的：第 159 行说"图像仅提供建筑的平面轮廓信息，
    不含建筑高度数据"，第 172 行却说"图像不含建筑高度信息，请结合场景类型提示和
    常识进行推理" —— 后半句正是 81.65% 高度幻觉的直接指令来源。
    这条测试把这个矛盾钉死：可以声明"没有高度"，但绝不能给出高度区间表、
    也不能要求模型去推测高度。
    """
    import re

    # 1) 不得出现中文高度词（用于描述建筑/天线本身）
    for w in ("建筑高度", "楼高", "楼层", "天线高度", "发射机高度"):
        assert w not in SYSTEM_PROMPT, f"prompt 里仍有高度词：{w}"

    # 2) 不得出现任何"高度区间"式的数字（例如 20-50m / 10-30 m）
    m = re.search(r"\d+\s*[-–~]\s*\d+\s*m\b", SYSTEM_PROMPT)
    assert m is None, f"prompt 里仍有高度区间样式的内容：{m.group(0) if m else ''}"

    # 3) 不得要求模型"结合常识推理"高度
    assert "结合场景类型提示" not in SYSTEM_PROMPT
    assert "常识进行推理" not in SYSTEM_PROMPT

    # 4) 必须**明确禁止**，而不只是沉默（沉默时模型会自己编）
    assert "NO height information" in SYSTEM_PROMPT
    assert "brighter pixels mean taller buildings" in SYSTEM_PROMPT, \
        "应显式点名这条错误说法并禁止它"
    assert "no meaningful height above ground" in SYSTEM_PROMPT

    # 5) 不得再声称 Tx 有 z 高度坐标
    assert "(x, y, z)" not in SYSTEM_PROMPT


def t_prompt_binarizes_mask():
    """渲染时必须把掩码二值化，使"更亮=更高"在图像层面就不成立。

    这是防止高度信息从输入侧渗回来的物理保证：即使有人误把一张灰度高度图
    当作掩码喂进来，LLM 看到的也只有 {0,255} 两个值，读不出高低。
    """
    from PIL import Image
    # 造一张"灰度高度图"当作输入
    grad = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    left, right = render_panels(grad, (128, 128), guides=False, captions=False)
    lv = np.unique(np.array(left)[:, :, 0])
    rv = np.unique(np.array(right)[:, :, 0])
    assert set(lv.tolist()) <= {0, 255}, f"全局面板未二值化：{lv[:6].tolist()}"
    assert set(rv.tolist()) <= {0, 255}, f"局部面板未二值化：{rv[:6].tolist()}"


def t_prompt_messages():
    from PIL import Image
    mask = np.array(Image.open(DATA_ROOT / "buildings_complete" / "0.png").convert("L"))
    tx = json.loads((DATA_ROOT / "antenna" / "0.json").read_text(encoding="utf-8"))[0]
    msgs = build_messages(mask, 0, 0, tx)
    assert len(msgs) == 2 and msgs[0]["role"] == "system"
    parts = msgs[1]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "scene_type" not in parts[1]["text"], "用户消息里不应出现任何答案字段"


# --------------------------------------------------------------- 6. dataset

def _make_tiny_split(tmp: Path, n_tx: int = 3) -> tuple[Path, Path]:
    """造一个 3 张地图的小划分 + 对应的先验文件，用于快速跑通 dataset/模型。"""
    data = tmp / "data"
    (data / "buildings_complete").mkdir(parents=True)
    (data / "gain" / "carsDPM").mkdir(parents=True)
    (data / "antenna").mkdir(parents=True)
    for m in (0, 1, 2):
        shutil.copy(DATA_ROOT / "buildings_complete" / f"{m}.png",
                    data / "buildings_complete" / f"{m}.png")
        ants = json.loads((DATA_ROOT / "antenna" / f"{m}.json").read_text(encoding="utf-8"))
        (data / "antenna" / f"{m}.json").write_text(json.dumps(ants[:n_tx]), encoding="utf-8")
        for t in range(n_tx):
            shutil.copy(DATA_ROOT / "gain" / "carsDPM" / f"{m}_{t}.png",
                        data / "gain" / "carsDPM" / f"{m}_{t}.png")
    splits = {
        "seed": 1, "split_unit": "map", "n_tx_per_map": n_tx, "all_maps": [0, 1, 2],
        "train_maps": [0], "val_maps": [1], "test_maps": [2],
        "counts": {"train_samples": n_tx, "val_samples": n_tx, "test_samples": n_tx},
    }
    sp = tmp / "splits.json"
    sp.write_text(json.dumps(splits), encoding="utf-8")

    priors = tmp / "priors"
    priors.mkdir()
    # train split 是 map 0 的全部 n_tx 个样本，先验必须齐（缺一个 dataset 就该抛错）
    for t in range(n_tx):
        rec = {
            "schema_version": "v2", "sample_id": f"0_{t}", "map_id": 0, "tx_id": t,
            "prior": make_prior(t + 1),
            "provenance": {"model": "selfcheck", "field_consistency": {"overall": 0.9}},
        }
        (priors / f"0_{t}.json").write_text(json.dumps(rec), encoding="utf-8")
    return sp, priors


def t_dataset_modes():
    from src.data.dataset import RadioMapSeerDataset
    tmp = Path(tempfile.mkdtemp())
    try:
        sp, priors = _make_tiny_split(tmp)
        data = tmp / "data"
        for mode, dim in (("none", 0), ("masked", 64), ("real", 64), ("shuffled", 64)):
            ds = RadioMapSeerDataset(data, sp, "train", prior_mode=mode, prior_dir=priors)
            s = ds[0]
            assert tuple(s["inputs"].shape) == (4, 256, 256), (mode, s["inputs"].shape)
            assert tuple(s["target"].shape) == (1, 256, 256)
            assert tuple(s["sat_mask"].shape) == (1, 256, 256)
            assert s["prior_vec"].numel() in (0, 1) or s["prior_vec"].numel() == dim
            assert 0.0 <= float(s["target"].min()) and float(s["target"].max()) <= 1.0
        # sectors 通道
        ds = RadioMapSeerDataset(data, sp, "train", prior_mode="real", prior_dir=priors,
                                 sectors_channels=True)
        assert tuple(ds[0]["inputs"].shape) == (12, 256, 256)
        assert ds.describe()["in_channels"] == 12
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_dataset_masked_is_zero():
    from src.data.dataset import RadioMapSeerDataset
    tmp = Path(tempfile.mkdtemp())
    try:
        sp, priors = _make_tiny_split(tmp)
        ds = RadioMapSeerDataset(tmp / "data", sp, "train", prior_mode="masked",
                                 prior_dir=priors)
        assert float(ds[0]["prior_vec"].abs().max()) == 0.0
        assert ds.prior_dim == 64, "masked 必须与 real 同维度，才能参数量对齐"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_dataset_shuffled_no_fixed_point():
    from src.data.dataset import RadioMapSeerDataset
    tmp = Path(tempfile.mkdtemp())
    try:
        sp, priors = _make_tiny_split(tmp, n_tx=80)
        data = tmp / "data"
        ds_real = RadioMapSeerDataset(data, sp, "train", prior_mode="real", prior_dir=priors)
        ds_sh = RadioMapSeerDataset(data, sp, "train", prior_mode="shuffled", prior_dir=priors)
        idx = ds_sh.prior_shuffle_index
        assert idx is not None and len(idx) == len(ds_sh)
        assert all(int(idx[i]) != i for i in range(len(idx))), "打乱必须没有不动点"
        assert len(set(idx.tolist())) == len(idx), "打乱必须是排列"
        # 至少要有相当比例的样本先验被换掉了
        diff = sum(1 for i in range(len(ds_sh))
                   if not np.allclose(ds_real[i]["prior_vec"].numpy(),
                                      ds_sh[i]["prior_vec"].numpy()))
        assert diff > len(ds_sh) * 0.9, f"只有 {diff}/{len(ds_sh)} 个样本的先验被换掉"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_dataset_shuffle_cot_isolation():
    """`shuffle_cot` 只能打乱 CoT 文本，绝不能顺带把先验向量也打乱。

    这条测试来自一个真实 bug：初版把"打乱向量"和"打乱 CoT"共用了一个错位映射，
    于是 `mode=real + shuffle_cot=true` 实际上变成了 `mode=shuffled` 实验 ——
    H3 的容量对照会被悄悄换成一个不同的对照，而且从指标上看不出任何异常。
    """
    from src.data.dataset import RadioMapSeerDataset
    tmp = Path(tempfile.mkdtemp())
    try:
        sp, priors = _make_tiny_split(tmp, n_tx=8)
        data = tmp / "data"
        common = dict(prior_mode="real", prior_dir=priors, return_cot_text=True)
        ds_a = RadioMapSeerDataset(data, sp, "train", shuffle_cot=False, **common)
        ds_b = RadioMapSeerDataset(data, sp, "train", shuffle_cot=True, **common)

        # 1) 向量完全不受 shuffle_cot 影响
        assert ds_b.prior_shuffle_index is None, "shuffle_cot 不应触发向量错位映射"
        for i in range(len(ds_a)):
            assert np.array_equal(ds_a[i]["prior_vec"].numpy(), ds_b[i]["prior_vec"].numpy()), \
                f"shuffle_cot 改变了第 {i} 个样本的先验向量"

        # 2) CoT 文本确实被打乱，且没有不动点、是排列
        ci = ds_b.cot_shuffle_index
        assert ci is not None and all(int(ci[i]) != i for i in range(len(ci)))
        assert len(set(ci.tolist())) == len(ci)
        diff = sum(1 for i in range(len(ds_b))
                   if ds_a[i]["cot_text"] != ds_b[i]["cot_text"])
        assert diff > len(ds_b) * 0.9, f"只有 {diff}/{len(ds_b)} 条 CoT 被换掉"
        # 方向不能写反：ci 是"样本 i 去取谁的文本"，即 ds_b[i] == ds_a[ci[i]]。
        # 错位映射不是对合映射（前驱 ≠ 后继），反过来断言会失败。
        assert all(ds_b[i]["cot_text"] == ds_a[int(ci[i])]["cot_text"]
                   for i in range(len(ds_b))), "打乱后的 CoT 必须正好来自错位后的样本"
        # 文本集合必须守恒（打乱不是丢数据）
        assert sorted(ds_b[i]["cot_text"] for i in range(len(ds_b))) == \
            sorted(ds_a[i]["cot_text"] for i in range(len(ds_a)))

        # 3) 两种打乱可以同时开，且用的是**不同**的映射
        ds_c = RadioMapSeerDataset(data, sp, "train", prior_mode="shuffled",
                                  prior_dir=priors, return_cot_text=True, shuffle_cot=True)
        assert ds_c.prior_shuffle_index is not None and ds_c.cot_shuffle_index is not None
        assert not np.array_equal(ds_c.prior_shuffle_index, ds_c.cot_shuffle_index), \
            "向量错位与 CoT 错位不应是同一个排列"

        # 4) 没开 return_cot_text 时不该白白构造 CoT 错位映射
        ds_d = RadioMapSeerDataset(data, sp, "train", prior_dir=priors,
                                  return_cot_text=False, shuffle_cot=True)
        assert ds_d.cot_shuffle_index is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_dataset_sector_channels_modes():
    """sectors 楔形通道在三种模式下的语义必须与向量一致。

    * masked   → 楔形通道必须全零（否则 masked 变成半个 real，real − masked 失效）
    * real     → 楔形通道来自自己的先验
    * shuffled → 楔形通道来自**别人**的先验
    * none     → 直接报错（没有先验可铺）
    """
    from src.data.dataset import RadioMapSeerDataset
    tmp = Path(tempfile.mkdtemp())
    try:
        sp, priors = _make_tiny_split(tmp, n_tx=8)
        data = tmp / "data"
        ds_m = RadioMapSeerDataset(data, sp, "train", prior_mode="masked", prior_dir=priors,
                                   sectors_channels=True)
        s = ds_m[0]
        assert tuple(s["inputs"].shape) == (12, 256, 256)
        assert float(s["inputs"][4:].abs().max()) == 0.0, "masked 的楔形通道必须全零"

        ds_r = RadioMapSeerDataset(data, sp, "train", prior_mode="real", prior_dir=priors,
                                   sectors_channels=True)
        nonzero = [i for i in range(len(ds_r)) if float(ds_r[i]["inputs"][4:].abs().max()) > 0]
        assert nonzero, "real 模式下楔形通道不该全零"

        ds_s = RadioMapSeerDataset(data, sp, "train", prior_mode="shuffled", prior_dir=priors,
                                   sectors_channels=True)
        diff = sum(1 for i in range(len(ds_s))
                   if not np.array_equal(ds_r[i]["inputs"][4:].numpy(),
                                         ds_s[i]["inputs"][4:].numpy()))
        assert diff > len(ds_s) * 0.9, f"shuffled 的楔形通道只有 {diff} 个与 real 不同"

        try:
            RadioMapSeerDataset(data, sp, "train", prior_mode="none",
                                sectors_channels=True)
        except ValueError:
            pass
        else:
            raise AssertionError("prior_mode=none + sectors_channels 应当报错")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_dataset_missing_prior_raises():
    from src.data.dataset import RadioMapSeerDataset
    tmp = Path(tempfile.mkdtemp())
    try:
        sp, priors = _make_tiny_split(tmp)
        (priors / "0_1.json").unlink()
        try:
            RadioMapSeerDataset(tmp / "data", sp, "train", prior_mode="real", prior_dir=priors)
        except FileNotFoundError:
            return
        raise AssertionError("缺先验文件时应当抛 FileNotFoundError，而不是静默补零")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_dataset_split_isolation():
    """val/test 的样本必须完全不与 train 重合（leave-maps-out）。"""
    from src.data.dataset import RadioMapSeerDataset
    tmp = Path(tempfile.mkdtemp())
    try:
        sp, priors = _make_tiny_split(tmp)
        data = tmp / "data"
        tr = set(RadioMapSeerDataset(data, sp, "train", "masked").samples)
        va = set(RadioMapSeerDataset(data, sp, "val", "masked").samples)
        te = set(RadioMapSeerDataset(data, sp, "test", "masked").samples)
        assert not (tr & va) and not (tr & te) and not (va & te)
        assert len(tr | va | te) == 9
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- 7. models

def t_model_params_and_identity():
    import torch
    from src.models.factory import build_model, identity_check

    base_cfg = {"name": "unet_film", "base_channels": 32, "dropout": 0.15,
                "film": {"hidden": [128, 256], "identity_init": True},
                "cot": {"enabled": False}}
    m, info = build_model(base_cfg, in_channels=4, prior_dim=64)
    assert info["n_params_total"] > 0
    ic = identity_check(m)
    # 容差 1e-2：最后一层权重是 1e-4 量级的小随机值（为让梯度能回传，
    # 见 film.apply_film_identity_init 的注释），所以 γ 与 1 的偏差是 1e-4 量级而非 0。
    # 精确到 1e-5 的"逐位恒等"是**做不到**的，硬要它成立就等于把梯度路径掐断。
    assert ic is not None and ic["ok"] == 1.0, f"FiLM identity 初始化失败: {ic}"
    assert ic["max_abs_gamma_minus_1"] < 1e-2, f"γ 偏离 1 太多: {ic}"

    x = torch.randn(2, 4, 64, 64)
    pv = torch.rand(2, 64)
    with torch.no_grad():
        y = m(x, pv)
    assert tuple(y.shape) == (2, 1, 64, 64)


def t_model_param_matching():
    """masked 与 real 必须参数量逐位相同 —— 这是 real−masked 可解释的前提。"""
    from src.models.factory import build_model
    cfg = {"name": "unet_film", "base_channels": 32, "dropout": 0.15,
           "cot": {"enabled": False}}
    m1, i1 = build_model(cfg, 4, 64)
    m2, i2 = build_model(cfg, 4, 64)
    assert i1["n_params_total"] == i2["n_params_total"]
    mu, iu = build_model({"name": "unet", "base_channels": 32}, 4, 0)
    assert iu["n_params_total"] < i1["n_params_total"], "FiLM 分支应带来额外参数"
    assert i1["n_params_conditional"] > 0


def t_model_rejects_dropping_prior():
    from src.models.factory import build_model
    try:
        build_model({"name": "unet"}, in_channels=4, prior_dim=64)
    except ValueError:
        return
    raise AssertionError("unet + prior_dim>0 应当报错，避免悄悄把先验丢掉")


def t_model_cot_modes():
    import torch
    from src.models.factory import build_model
    texts = ["0.9GHz 场景，180 度方向有连片建筑。", "开阔区域，直射为主。"]
    for mode, prior_dim in (("cot_only", 64), ("both", 64)):
        cfg = {"name": "unet_film", "base_channels": 32,
               "cot": {"enabled": True, "mode": mode, "out_dim": 16, "buckets": 512,
                       "hidden": 32}}
        m, info = build_model(cfg, 4, prior_dim)
        assert info["cot_mode"] == mode and info["cot_dim"] == 16
        with torch.no_grad():
            y = m(torch.randn(2, 4, 64, 64), torch.rand(2, prior_dim), texts)
        assert tuple(y.shape) == (2, 1, 64, 64)
    # cot_only 时结构化字段不应进入模型，因此 prior_dim>0 也不该报错
    try:
        build_model({"name": "unet_film", "cot": {"enabled": True, "mode": "cot_only"}}, 4, 64)
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"cot_only 不应因为 prior_dim>0 而报错: {exc}")


def t_model_film_identity_end_to_end():
    """identity 初始化必须让"换先验"在初始化时刻对输出**几乎无影响**。

    把"修好 FiLM 初始化"从一句说法变成可验证的不变量。同时量化旧版问题：
    不做 identity 初始化时，仅把先验向量从全零换成随机，初始化输出就会变化
    约 6.8 dB（实测），这正是上一版 epoch 1 FiLM 118 dB vs 消融 76 dB 的机制。

    ⚠️ 这里**不再断言逐位相等**：`apply_film_identity_init()` 的最后一层权重必须是
    1e-4 量级的**小随机值**而不是全零 —— 全零会让该层梯度恒为 0、先验永久失效
    （实测 `real_v3` 与 `masked_em` 跑 5 个 epoch 后 84 个张量仍逐位相同）。
    所以 identity 现在是"到 1e-4 量级"，用 dB 口径设一个远小于噪声的容差。
    """
    import torch
    from src.data.normalize import PATHLOSS_MAX, PATHLOSS_MIN
    from src.models.film import UNetFiLM
    from src.utils.seed import set_seed

    set_seed(0, deterministic=True)
    x = torch.randn(2, 4, 64, 64)
    span = PATHLOSS_MAX - PATHLOSS_MIN

    def build(identity: bool):
        set_seed(0, deterministic=True)
        return UNetFiLM(in_channels=4, base_channels=32, prior_dim=64,
                        film_identity_init=identity, allow_no_identity_init=True).eval()

    m_ok = build(True)
    with torch.no_grad():
        y_zero = m_ok(x, torch.zeros(2, 64))
        y_rand = m_ok(x, torch.rand(2, 64))
        y_rand2 = m_ok(x, torch.rand(2, 64))
    d_ok = float((y_zero - y_rand).abs().mean()) * span
    # 1e-3 dB 量级：远小于任何有意义的差异，也远小于"旧版 6.8 dB"
    assert d_ok < 0.01, (
        f"identity 初始化下先验对初始输出的影响应当可忽略（实测 {d_ok:.4f} dB）")
    assert float((y_rand - y_rand2).abs().max()) > 0.0, (
        "不同先验给出逐位相同的输出 → 先验根本没接到前向上（这本身就是 bug）")

    m_bad = build(False)
    with torch.no_grad():
        d_bad = float((m_bad(x, torch.rand(2, 64)) - m_bad(x, torch.zeros(2, 64))).abs().mean())
    assert d_bad * span > 1.0, (
        f"旧版式初始化理应让先验显著改变初始输出，实测只有 {d_bad * span:.2f} dB；"
        f"若这条不再成立，说明 identity 修复的对照实验失去意义"
    )
    # 关键对照：identity 版本的影响必须比旧版式小好几个数量级，
    # 否则"identity 初始化"只是名义上的
    assert d_ok * 100 < d_bad * span, (
        f"identity 版影响 {d_ok:.4f} dB 与旧版式 {d_bad * span:.2f} dB 差距不足 100×")


def t_configs_all_load():
    """每个 config 都要能加载、能据此构造出模型。

    为什么值得单独测：11 个 config 的 `prior_groups` / `sectors_channels` / `cot` 组合
    很容易配错（例如把 sectors 同时送进 FiLM 和楔形通道、或 `prior_none` 忘了关掉
    FiLM 分支、或 `unet` 配了非零 prior_dim）。这类错误如果不在开工前拦住，
    会在训练跑了一段时间之后才以 `RuntimeError` 或"结果无法解释"的形式暴露。
    """
    from src.models.factory import build_model
    from src.prior.schema import vector_dim
    from src.utils.config import load_config

    cfg_dir = ROOT / "configs"
    files = sorted(cfg_dir.glob("*.yaml"))
    assert len(files) >= 10, f"configs 数量异常: {[f.name for f in files]}"
    rows = []
    for f in files:
        if f.name == "base.yaml":
            continue
        cfg = load_config(f)
        pcfg, mcfg = cfg["prior"], cfg["model"]
        groups = tuple(pcfg.get("prior_groups", ["global_num", "global_cat", "sectors"]))
        p_dim = 0 if pcfg["mode"] == "none" else vector_dim(groups)
        in_ch = 4 + (8 if pcfg.get("sectors_channels", False) else 0)
        model, info = build_model(dict(mcfg), in_channels=in_ch, prior_dim=p_dim)
        # 语义检查：none 必须是纯 UNet；其余必须是带条件分支且 prior_dim 与配置一致
        if pcfg["mode"] == "none":
            assert info["name"] == "unet" and info["prior_dim"] == 0, f.name
        else:
            assert info["name"] == "unet_film", f.name
            assert info["prior_dim"] == p_dim, f.name
            assert info["n_params_conditional"] > 0, f.name
        if pcfg.get("sectors_channels", False):
            assert "sectors" not in groups, (
                f"{f.name}: sectors 同时进了 FiLM 向量和楔形通道，"
                f"两个实验组会互相污染，请二选一")
        cot = mcfg.get("cot") or {}
        if cot.get("enabled", False):
            assert info["cot_mode"] in ("cot_only", "both"), f.name
            assert info["cot_dim"] > 0, f.name
        rows.append((f.name, info["name"], in_ch, info["prior_dim"],
                     info["cot_mode"], info["n_params_total"]))
    print("\n      配置一览：")
    print(f"      {'config':<30}{'model':<11}{'in_ch':>6}{'prior_dim':>10}"
          f"{'cot':>10}{'params':>13}")
    for r in rows:
        print(f"      {r[0]:<30}{r[1]:<11}{r[2]:>6}{r[3]:>10}{r[4]:>10}{r[5]:>13,}")


def t_cot_encoder_deterministic():
    from src.models.cot_encoder import HashingTextEncoder, hash_token, tokenize
    assert hash_token("建筑", 1024) == hash_token("建筑", 1024)
    toks = tokenize("0.9GHz 场景")
    assert "0.9ghz" in toks and any(t == "场" for t in toks)
    e1 = HashingTextEncoder(out_dim=8, buckets=256, hidden=16)
    e2 = HashingTextEncoder(out_dim=8, buckets=256, hidden=16)
    e2.load_state_dict(e1.state_dict())
    a = e1(["同一段推理文本"]).detach()
    b = e2(["同一段推理文本"]).detach()
    assert float((a - b).abs().max()) == 0.0, "同一文本必须得到同一编码"


# ------------------------------------------------------- 8. losses & SSIM

def t_film_prior_actually_conditions():
    """**先验必须真的影响前向**，且梯度必须能回传到整个 FiLM 分支。

    为什么必须单独立这一条（本项目实测踩出来的最贵的一个坑）：`apply_film_identity_init()`
    初版把最后一层 `film_mlp` 的**权重与 bias 一起置零**，只把 γ 的 bias 设成 1。
    结果 film_mlp 的输出恒为 [1, 0, …]、与输入无关；更致命的是最后一层权重的梯度
    ∝（上游梯度 × 该层输入），而该层输入恒为 0 ⇒ 梯度**永远是 0**、权重永久冻结。

    这个 bug **不报错**：训练照跑、loss 照降、val 照算，只是 `prior_real_v3` 与
    `prior_v3_masked_em` 会**永远逐位相同**（实测 5 个 epoch 后 84 个张量 max|Δ|=0），
    整轮实验失去意义。所以必须在自检里直接断言"换一个先验 → 输出必须变"。

    这里只用合成张量，不需要数据盘；真实数据上的同款检查见 `tests/test_typology.py`
    的 `prior.actually_conditions_model`（有数据盘时才会跑）。
    """
    import torch
    from src.models.film import UNetFiLM

    torch.manual_seed(0)
    m = UNetFiLM(in_channels=4, base_channels=8, prior_dim=16, dropout=0.0,
                 film_identity_init=True)
    m.eval()

    # (1) identity 性质：初始 γ ≈ 1、β ≈ 0（容许 1e-2 量级的小扰动，
    #     因为最后一层权重是小随机值而不是 0 —— 这是让梯度能回传的代价）
    chk = m.check_identity(atol=1e-2)
    assert chk["ok"] == 1.0, f"identity 初始化没生效：{chk}"

    # (2) 最后一层的权重**不得**全为 0（全 0 就意味着梯度永久为 0）
    last = m.film_mlp[-1]
    assert float(last.weight.abs().max()) > 0.0, (
        "最后一层 FiLM 权重全为 0：梯度 ∝ 该层输入，而它在 identity 初始化下恒为 0，"
        "于是 FiLM 分支永远收不到梯度、先验完全不起作用")

    # (3) 端到端：换一个先验向量，输出必须变化
    x = torch.randn(2, 4, 32, 32)
    p1 = torch.zeros(2, 16)
    p2 = torch.rand(2, 16)
    with torch.no_grad():
        y1 = m(x, p1)
        y2 = m(x, p2)
    diff = float((y1 - y2).abs().max())
    assert diff > 1e-6, (
        f"换了先验向量但输出完全没变（max|Δ|={diff:.3e}）：先验没有接到前向上")

    # (4) 梯度必须真的流到 FiLM 的每一层（含最后一层的 weight）
    m.train()
    out = m(x, p2)
    out.sum().backward()
    for name, p in m.film_mlp.named_parameters():
        assert p.grad is not None, f"{name} 没有梯度"
        assert torch.isfinite(p.grad).all(), f"{name} 梯度非有限"
        assert float(p.grad.abs().max()) > 0.0, (
            f"{name} 的梯度恒为 0 —— 该层永远不会被训练（identity 初始化的经典陷阱）")


def t_ssim_crosscheck():
    """torch 版 SSIM 与评测用 numpy 版必须一致，否则训练与报告的 SSIM 是两个东西。"""
    import torch
    from src.eval.metrics import ssim_numpy
    from src.train.losses import ssim_torch
    rng = np.random.RandomState(0)
    for shape in [(1, 1, 64, 64), (2, 1, 48, 48)]:
        a = rng.rand(*shape).astype(np.float32)
        b = rng.rand(*shape).astype(np.float32)
        tv = float(ssim_torch(torch.from_numpy(a), torch.from_numpy(b)))
        nv = float(np.mean([ssim_numpy(a[i, 0], b[i, 0]) for i in range(shape[0])]))
        assert abs(tv - nv) < 1e-5, f"torch={tv} numpy={nv} 差异过大"
    # 完全相同输入应给出 SSIM=1
    a = rng.rand(1, 1, 32, 32).astype(np.float32)
    assert abs(float(ssim_torch(torch.from_numpy(a), torch.from_numpy(a))) - 1.0) < 1e-5
    assert abs(ssim_numpy(a[0, 0], a[0, 0]) - 1.0) < 1e-6


def t_loss_backward():
    import torch
    from src.train.losses import masked_loss
    pred = torch.rand(2, 1, 64, 64, requires_grad=True)
    tgt = torch.rand(2, 1, 64, 64)
    sat = torch.zeros(2, 1, 64, 64, dtype=torch.bool)
    for cfg in ({"smooth_l1": 1.0, "grad": 0.5, "ssim": 0.3},
                {"smooth_l1": 1.0, "grad": 0.0, "ssim": 0.0},
                {"smooth_l1": 1.0, "grad": 0.5, "ssim": 0.3, "exclude_saturated": True}):
        parts = masked_loss(pred, tgt, sat, cfg)
        assert torch.isfinite(parts["total"]), cfg
        parts["total"].backward(retain_graph=True)
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def t_loss_fp32_under_half_inputs():
    """AMP 下 pred 可能是 fp16，损失必须内部升到 fp32。

    这是只会在 GPU 上暴露的风险：混合精度训练时 SSIM 里的多次卷积与除法在 fp16 下
    会掉精度甚至产生噪声梯度，若直接拿它反传，优化方向本身就不可信。
    本机（CPU）也能验证这件事：喂 fp16 输入，检查损失是 fp32 且与 fp32 计算一致。
    """
    import torch
    from src.train.losses import total_loss

    cfg = {"smooth_l1": 1.0, "grad": 0.5, "ssim": 0.3}
    rng = np.random.RandomState(7)
    a = rng.rand(2, 1, 48, 48).astype(np.float32)
    b = rng.rand(2, 1, 48, 48).astype(np.float32)

    ref = total_loss(torch.from_numpy(a), torch.from_numpy(b), cfg)

    a_h = torch.from_numpy(a).half().requires_grad_(True)
    b_h = torch.from_numpy(b).half()
    parts = total_loss(a_h, b_h, cfg)
    assert parts["total"].dtype == torch.float32, \
        f"损失应为 fp32，实际 {parts['total'].dtype}"
    assert torch.isfinite(parts["total"]), "fp16 输入下损失出现非有限值"
    # 与 fp32 参考的差异应只是 fp16 输入的量化误差量级
    assert abs(float(parts["total"]) - float(ref["total"])) < 5e-3, \
        f"fp16 输入下损失与 fp32 参考偏差过大: {float(parts['total'])} vs {float(ref['total'])}"
    parts["total"].backward()
    assert a_h.grad is not None and torch.isfinite(a_h.grad).all()


def t_v3_groups_and_merge():
    """v3 的分组下标与"几何 + LLM 电磁"合并。

    v3 的核心是分工：程序给精确几何、LLM 只做电磁推理。这条测试保证
    (a) 分组下标正确（容量对齐对照要靠它把电磁维度单独置零）；
    (b) 合并后 sectors / 局部占比确实来自**程序**，而不是 LLM 输出。
    """
    from src.prior.schema import (
        ALL_GROUPS,
        EM_GROUPS,
        GEOMETRY_GROUPS,
        RadioPriorEM,
        group_index_map,
        indices_for_groups,
        merge_em_with_geometry,
        vector_dim,
    )
    m = group_index_map(ALL_GROUPS)
    assert [m["em_num"][0], m["em_num"][-1]] == [0, 7]
    assert [m["local_ratio"][0], m["local_ratio"][-1]] == [8, 9]
    assert [m["global_cat"][0], m["global_cat"][-1]] == [10, 23]
    assert [m["sectors"][0], m["sectors"][-1]] == [24, 63]
    assert len(indices_for_groups(ALL_GROUPS, EM_GROUPS)) == 22
    assert len(indices_for_groups(ALL_GROUPS, GEOMETRY_GROUPS)) == 42
    # 与旧分组必须产生同一顺序的同一向量（历史结果可比）
    assert vector_dim((S.GROUP_GLOBAL_NUM, S.GROUP_GLOBAL_CAT, S.GROUP_SECTORS)) == 64
    assert vector_dim(ALL_GROUPS) == 64

    em = RadioPriorEM(scene_type="urban", los_ratio=0.4, reflection_ratio=0.25,
                      diffraction_ratio=0.25, scattering_ratio=0.1,
                      dominant_material="concrete", permittivity=6.0,
                      penetration_loss_db=12.0, pathloss_min_db=60.0,
                      pathloss_max_db=150.0, confidence="medium", reasoning="自检")
    geo = {"sectors": [{"azimuth_deg": a, "obstruction": "blocked",
                        "nearest_building_m": 0.0} for a in S.AZIMUTHS],
           "tx_local_building_ratio_50m": 0.31, "tx_local_building_ratio_100m": 0.29}
    prior = merge_em_with_geometry(em, geo)
    assert isinstance(prior, S.RadioPriorV2)
    assert all(s.obstruction == "blocked" for s in prior.sectors), "sectors 必须来自程序"
    assert abs(prior.tx_local_building_ratio_100m - 0.29) < 1e-9
    assert abs(prior.los_ratio - 0.4) < 1e-9, "电磁字段应来自 LLM"
    # 合并后向量必须能用，且电磁部分与几何部分落在正确的下标上
    v = S.prior_to_vector(prior, ALL_GROUPS)
    assert len(v) == 64
    assert abs(v[0] - 0.4) < 1e-9, "第 0 维应是 los_ratio"


def t_v3_facts_are_exact():
    """`compute_geometry_facts` 的几何必须与 `geometric_obstruction` 完全一致。

    这是 v3 的立身之本：空间信息由程序精确给出，不该有任何"估计"成分。
    """
    from src.prior.prompt_v3 import compute_geometry_facts
    m = np.zeros((128, 128), dtype=np.uint8)
    m[40:90, 90:128] = 255              # 东侧一堵墙
    tx = (64, 64)
    f = compute_geometry_facts(m, tx, radius_m=60.0, px_per_m=1.0)
    geo = geometric_obstruction(m, tx, radius_m=60.0, px_per_m=1.0)
    for s in f["sectors"]:
        az = s["azimuth_deg"]
        assert s["obstruction"] == str(geo[az]["obstruction"]), f"{az} 度标签不一致"
        assert abs(s["nearest_building_m"] - round(float(geo[az]["nearest_building_m"]), 1)) < 1e-6
    assert f["tx_on_building"] is False
    assert 0.0 <= f["tx_local_building_ratio_50m"] <= 1.0
    assert "azimuths by obstruction class" in f["facts_text"]
    # Tx 落在建筑上时必须被识别出来（实测 30.3% 的 Tx 是这种）
    f2 = compute_geometry_facts(m, (100, 60), radius_m=60.0, px_per_m=1.0)
    assert f2["tx_on_building"] is True
    assert "lies ON a building" in f2["facts_text"]


def t_v3_prompt_no_height_instruction():
    """v3 的 system prompt 同样不得引导模型推理建筑高度。"""
    import re
    from src.prior.prompt_v3 import GEOMETRY_FACTS_SYSTEM as SYS
    for w in ("建筑高度", "楼高", "楼层", "天线高度"):
        assert w not in SYS
    assert re.search(r"\d+\s*[-–~]\s*\d+\s*m\b", SYS) is None
    assert "NO height information" in SYS
    assert "brighter pixels mean taller buildings" in SYS
    assert "no meaningful height above ground" in SYS
    assert "0.9 GHz" in SYS


def t_dataset_zero_groups():
    """`zero_groups` 必须**只**把指定分组置零，几何部分原样保留。

    这是容量对齐对照的基础：prior_dim 不变（64），只让电磁维度归零，
    差值才能干净地归因于 LLM 的电磁内容。
    """
    from src.data.dataset import RadioMapSeerDataset
    from src.prior.schema import ALL_GROUPS, EM_GROUPS, GEOMETRY_GROUPS, indices_for_groups
    tmp = Path(tempfile.mkdtemp())
    try:
        sp, priors = _make_tiny_split(tmp, n_tx=6)
        data = tmp / "data"
        em_idx = indices_for_groups(ALL_GROUPS, EM_GROUPS)
        geo_idx = indices_for_groups(ALL_GROUPS, GEOMETRY_GROUPS)
        ds_a = RadioMapSeerDataset(data, sp, "train", prior_mode="real", prior_dir=priors)
        ds_b = RadioMapSeerDataset(data, sp, "train", prior_mode="real", prior_dir=priors,
                                  zero_groups=list(EM_GROUPS))
        assert ds_b.prior_dim == ds_a.prior_dim == 64, "置零不能改变 prior_dim（否则容量不同）"
        for i in range(len(ds_a)):
            va = ds_a[i]["prior_vec"].numpy()
            vb = ds_b[i]["prior_vec"].numpy()
            assert np.allclose(vb[em_idx], 0.0), "指定的电磁维度必须全零"
            assert np.allclose(vb[geo_idx], va[geo_idx]), "几何维度不能被改动"
        # 配置写错时必须报错，而不是静默什么都不做。
        # 注意：默认 prior_groups 就含 sectors，所以必须显式传一个**不含**它的
        # prior_groups，才能真正构造出"未知分组"的情形。
        try:
            RadioMapSeerDataset(data, sp, "train", prior_mode="real", prior_dir=priors,
                                prior_groups=("em_num", "global_cat"),
                                zero_groups=["sectors"])
        except ValueError:
            return
        raise AssertionError("zero_groups 含不在 prior_groups 里的分组时应报错")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_metrics_sample():
    from src.eval import metrics as M
    rng = np.random.RandomState(1)
    t = rng.rand(1, 64, 64).astype(np.float32)
    p = t.copy()
    m = M.sample_metrics(p, t, np.zeros((1, 64, 64), dtype=bool))
    assert abs(m["rmse_db"]) < 1e-9 and abs(m["ssim"] - 1.0) < 1e-6
    assert abs(m["mae_db"]) < 1e-9, "MAE 绝不能像上一版那样恒为 0（这里是因为完全一致）"
    p2 = np.clip(t + 0.05, 0, 1)
    m2 = M.sample_metrics(p2, t, np.zeros((1, 64, 64), dtype=bool))
    assert m2["mae_db"] > 0 and m2["rmse_db"] > 0
    # 饱和像素被排除后 RMSE 应不同（这里构造成有饱和区）
    sat = np.zeros((1, 64, 64), dtype=bool)
    sat[0, :32, :] = True
    p3 = p2.copy()
    p3[0, :32, :] = 1.0
    m3 = M.sample_metrics(p3, t, sat)
    assert m3["rmse_db_nonsaturated"] < m3["rmse_db"], "排除饱和像素后 RMSE 应更小"


TESTS = [
    ("schema.dims", t_schema_dims),
    ("schema.range", t_schema_range),
    ("schema.rejects_bad", t_schema_rejects_bad),
    ("schema.extra_forbidden", t_schema_extra_forbidden),
    ("schema.record_roundtrip", t_schema_record_roundtrip),
    ("normalize.mapping", t_normalize_mapping),
    ("normalize.roundtrip", t_normalize_roundtrip),
    ("normalize.saturation", t_saturation_mask),
    ("normalize.db_span", t_db_span),
    ("splits.ok", t_splits_ok),
    ("splits.hash", t_splits_hash_stable),
    ("splits.reject_sample_level", t_splits_reject_sample_level),
    ("sectors.angular_map", t_angular_map_partition),
    ("sectors.geometric_direction", t_geometric_obstruction_direction),
    ("sectors.wedge_maps", t_wedge_maps),
    ("sectors.local_ratio", t_local_building_ratio),
    ("prompt.render", t_prompt_render),
    ("prompt.local_panel_alignment", t_prompt_local_panel_alignment),
    ("prompt.no_scene_hint", t_prompt_no_scene_hint),
    ("prompt.no_height_instruction", t_prompt_has_no_height_instruction),
    ("prompt.binarizes_mask", t_prompt_binarizes_mask),
    ("schema.rejects_height_fields", t_schema_rejects_height_fields),
    ("prompt.messages", t_prompt_messages),
    ("dataset.modes", t_dataset_modes),
    ("dataset.masked_zero", t_dataset_masked_is_zero),
    ("dataset.shuffled_no_fixed_point", t_dataset_shuffled_no_fixed_point),
    ("dataset.shuffle_cot_isolation", t_dataset_shuffle_cot_isolation),
    ("dataset.sector_channels_modes", t_dataset_sector_channels_modes),
    ("dataset.missing_prior_raises", t_dataset_missing_prior_raises),
    ("dataset.split_isolation", t_dataset_split_isolation),
    ("models.params_and_identity", t_model_params_and_identity),
    ("models.film_identity_end_to_end", t_model_film_identity_end_to_end),
    ("models.film_prior_actually_conditions", t_film_prior_actually_conditions),
    ("models.param_matching", t_model_param_matching),
    ("models.rejects_dropping_prior", t_model_rejects_dropping_prior),
    ("models.cot_modes", t_model_cot_modes),
    ("v3.groups_and_merge", t_v3_groups_and_merge),
    ("v3.facts_are_exact", t_v3_facts_are_exact),
    ("v3.prompt_no_height_instruction", t_v3_prompt_no_height_instruction),
    ("dataset.zero_groups", t_dataset_zero_groups),
    ("configs.all_load_and_build", t_configs_all_load),
    ("models.cot_encoder_deterministic", t_cot_encoder_deterministic),
    ("losses.ssim_crosscheck", t_ssim_crosscheck),
    ("losses.backward", t_loss_backward),
    ("losses.fp32_under_half_inputs", t_loss_fp32_under_half_inputs),
    ("metrics.sample", t_metrics_sample),
]


def main(argv: list[str]) -> int:
    filt = argv[0] if argv else ""
    sel = [(n, f) for n, f in TESTS if filt in n]
    print(f"本地自检：{len(sel)} 项（过滤器 {filt!r}）\n")
    for name, fn in sel:
        check(name, fn)
    print(f"\n通过 {len(PASS)} / {len(sel)}")
    if FAIL:
        print("失败项：")
        for n, e in FAIL:
            print(f"  - {n}: {type(e).__name__}: {e}")
        return 1
    print("全部通过 [OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
