"""批量调用多模态 LLM 生成先验：多轮采样 + 共识聚合 + 断点续传。

为什么是"多轮采样 + 共识"而不是单次 temperature=0（交接文档 §5.5 第 3 条）
------------------------------------------------------------------------
上一版用 `temperature=0` 追求"确定性输出"，实测反而因为 5 个固定 few-shot 被无限
复读而**坍缩**：4000 条只有 55 个唯一向量，`reasoning` 里 475 条整段复读 28 GHz 模板。
而且 temperature=0 并没有真的带来确定性（同一张图的两个 Tx，一个走固定六步模板、
另一个是自由叙述）。

v2 改成：`temperature=0.7` 采样 `n` 次 →
    * 数值字段取中位数；
    * 类别字段（scene_type / dominant_material / confidence）取众数；
    * sectors 逐方位取 obstruction 众数、nearest_building_m 中位数；
    * 把"多次采样的一致程度"记进 provenance，它同时是一个天然的**不确定性度量**。

硬规则（AGENTS.md 规则 6）
--------------------------
解析失败 / 校验失败 **绝不**填默认值写进先验目录，而是记到
`<prior_dir>/_failed/<sample>.json`（含原始回复），并在 audit 报告里体现。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import ValidationError

from src.data.sectors import geometric_obstruction, local_building_ratio
from src.prior.prompt_v2 import build_messages
from src.prior.prompt_v3 import (
    build_geometry_facts_messages,
    compute_geometry_facts,
)
from src.prior.schema import (
    AZIMUTHS,
    EM_NUM_KEYS,
    NUM_RANGES,
    SCHEMA_VERSION,
    RadioPriorEM,
    RadioPriorV2,
    make_sample_id,
    merge_em_with_geometry,
)

PROVIDERS: dict[str, dict[str, str]] = {
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4v-flash",
        "api_key_env": "ZHIPUAI_API_KEY",
    },
    "openai_compatible": {
        "base_url": "",
        "model": "",
        "api_key_env": "LLM_API_KEY",
    },
}

# 同一个 key 在不同地方有不同叫法：官方 SDK 文档用 ZHIPUAI_API_KEY，
# 上一版的脚本里用的是 ZHIPU_API_KEY。两个都认，避免"key 明明设了却说不存在"。
API_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "ZHIPUAI_API_KEY": ("ZHIPUAI_API_KEY", "ZHIPU_API_KEY", "GLM_API_KEY"),
    "LLM_API_KEY": ("LLM_API_KEY", "OPENAI_API_KEY"),
    "OPENAI_API_KEY": ("OPENAI_API_KEY", "LLM_API_KEY"),
}

NUM_KEYS = (
    "los_ratio", "reflection_ratio", "diffraction_ratio", "scattering_ratio",
    "permittivity", "penetration_loss_db", "pathloss_min_db", "pathloss_max_db",
    "tx_local_building_ratio_50m", "tx_local_building_ratio_100m",
)
CAT_KEYS = ("scene_type", "dominant_material", "confidence")

_PRINT_LOCK = threading.Lock()


def log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, flush=True)


# ===== JSON 抽取 =====


def extract_json(text: str) -> dict:
    """从模型回复里抽出第一个完整的 JSON 对象。

    模型可能带 ```json 围栏或前后废话，这里做鲁棒抽取；抽不到就抛错，
    由调用方记入 _failed/（不静默兜底）。
    """
    if text is None:
        raise ValueError("空回复")
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```")[1] if len(s.split("```")) > 1 else s
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    start = s.find("{")
    if start < 0:
        raise ValueError(f"回复里没有 JSON 对象：{text[:200]!r}")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start:i + 1])
    raise ValueError(f"JSON 括号不闭合：{text[:200]!r}")


# ===== 共识聚合 =====


def _median(xs: list[float]) -> float:
    return float(np.median(np.asarray(xs, dtype=np.float64)))


def _mode(xs: list[str]) -> str:
    return Counter(xs).most_common(1)[0][0]


def _mode_share(xs: list[str]) -> float:
    return Counter(xs).most_common(1)[0][1] / len(xs)


def derive_confidence(consistency_overall: float) -> str:
    """把多轮采样的一致性**推导**成 confidence，而不是采信 LLM 的自报值。

    实测理由：`confidence` 是上一版最死的字段（97.9% 都是 medium），本批 24 个样本
    也是 100% medium —— 自报置信度没有逐样本信息量。而多轮采样的一致性天然就是
    一个不确定性度量（交接文档 §5.5 第 3 条也是这么建议的），并且它可被核对。
    LLM 自报的原始值仍然记进 provenance，便于对比。
    """
    if consistency_overall >= 0.90:
        return "high"
    if consistency_overall >= 0.70:
        return "medium"
    return "low"


def aggregate(priors: list[RadioPriorV2]) -> tuple[dict[str, Any], dict[str, Any]]:
    """多轮采样 → (聚合后的 prior dict, 一致性统计)。

    一致性定义：数值字段落在中位数 ±10% 量程内算一致；类别字段与众数相同算一致；
    sectors 的 obstruction 与该方位众数相同算一致。整体一致性为各项均值。
    """
    n = len(priors)
    agg: dict[str, Any] = {}
    consistency: dict[str, float] = {}

    for k in NUM_KEYS:
        vals = [float(getattr(p, k)) for p in priors]
        # 用**均值**而不是中位数：实测 3 样本取中位数会把连续字段量化掉
        # （同一个 prompt、不取共识时 los_ratio 有 7 个取值，取 3 样本中位数后只剩 2 个），
        # 而向量需要这些字段携带逐样本信息。
        mean = float(np.mean(vals))
        lo, hi = NUM_RANGES[k]
        tol = 0.10 * (hi - lo)          # 量程的 10% 作为"实质一致"的容差
        agg[k] = mean
        consistency[k] = float(np.mean([abs(v - mean) <= tol for v in vals]))

    for k in CAT_KEYS:
        vals = [str(getattr(p, k)) for p in priors]
        agg[k] = _mode(vals)
        consistency[k] = _mode_share(vals)

    agg, extra = _append_sectors(priors, agg, consistency)
    agg["reasoning"] = extra["reasoning"]
    consistency.update(extra["consistency"])
    return agg, consistency


def _append_sectors(priors, agg: dict, consistency: dict) -> tuple[dict, dict]:
    """按方位聚合 sectors，并选一条 reasoning。v3 的 prior 同样含 sectors（程序精确给定）。"""
    by_az = [{s.azimuth_deg: s for s in p.sectors} for p in priors]
    sectors: list[dict[str, Any]] = []
    ob_hits, nb_hits = [], []
    for az in AZIMUTHS:
        obs = [str(d[az].obstruction) for d in by_az]
        nbs = [float(d[az].nearest_building_m) for d in by_az]
        ob_mode = _mode(obs)
        nb_med = _median(nbs)
        sectors.append({"azimuth_deg": az, "obstruction": ob_mode,
                        "nearest_building_m": nb_med})
        ob_hits.append(_mode_share(obs))
        nb_hits.append(float(np.mean([abs(v - nb_med) <= 50.0 for v in nbs])))
    agg["sectors"] = sectors
    consistency["sector_obstruction"] = float(np.mean(ob_hits))
    consistency["sector_nearest_building_m"] = float(np.mean(nb_hits))

    # reasoning：取数值字段最接近聚合结果的那一条（确定性选择，可复现）
    ref = np.array([agg[k] for k in NUM_KEYS], dtype=np.float64)
    dists = [float(np.abs(np.array([float(getattr(p, k)) for k in NUM_KEYS]) - ref).mean())
             for p in priors]
    consistency["reasoning_pick_distance"] = float(min(dists))
    return agg, {"reasoning": priors[int(np.argmin(dists))].reasoning,
                 "consistency": consistency}


def aggregate_em(priors: list["RadioPriorEM"]) -> tuple[dict[str, Any], dict[str, Any]]:
    """v3：只聚合 LLM 的电磁字段（sectors 由程序给定，无需聚合）。"""
    agg: dict[str, Any] = {}
    consistency: dict[str, float] = {}
    for k in EM_NUM_KEYS:
        vals = [float(getattr(p, k)) for p in priors]
        mean = float(np.mean(vals))     # 均值而非中位数，理由同 aggregate()
        lo, hi = NUM_RANGES[k]
        tol = 0.10 * (hi - lo)
        agg[k] = mean
        consistency[k] = float(np.mean([abs(v - mean) <= tol for v in vals]))
    llm_reported_confidence = [str(getattr(p, "confidence")) for p in priors]
    for k in CAT_KEYS:
        if k == "confidence":
            continue                # confidence 由一致性推导，不用自报值
        vals = [str(getattr(p, k)) for p in priors]
        agg[k] = _mode(vals)
        consistency[k] = _mode_share(vals)
    # 先算整体一致性，再据此推导 confidence
    consistency["overall"] = float(np.mean(
        [v for k, v in consistency.items()
         if k not in ("n_samples", "reasoning_pick_distance")]))
    agg["confidence"] = derive_confidence(consistency["overall"])
    consistency["llm_reported_confidence_mode"] = _mode(llm_reported_confidence)
    consistency["llm_reported_confidence_unique"] = float(len(set(llm_reported_confidence)))
    ref = np.array([agg[k] for k in EM_NUM_KEYS], dtype=np.float64)
    dists = [float(np.abs(np.array([float(getattr(p, k)) for k in EM_NUM_KEYS]) - ref).mean())
             for p in priors]
    agg["reasoning"] = priors[int(np.argmin(dists))].reasoning
    consistency["reasoning_pick_distance"] = float(min(dists))
    consistency["n_samples"] = len(priors)
    return agg, consistency


# ===== 单样本生成 =====


class PriorGenerator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        preset = PROVIDERS[args.provider]
        self.base_url = args.base_url or os.environ.get("LLM_BASE_URL") or preset["base_url"]
        self.model = args.model or os.environ.get("LLM_MODEL") or preset["model"]
        key_env = args.api_key_env or preset["api_key_env"]
        self.api_key = ""
        self.api_key_source = None
        for name in API_KEY_ALIASES.get(key_env, (key_env,)):
            if os.environ.get(name):
                self.api_key = os.environ[name]
                self.api_key_source = name
                break
        self._client = None
        if not self.base_url or not self.model:
            raise SystemExit("必须提供 --base-url 与 --model（或设置 LLM_BASE_URL / LLM_MODEL）")

        data = Path(args.data)
        self.buildings = data / "buildings_complete"
        self.antennas = data / "antenna"
        self._mask_cache: dict[int, np.ndarray] = {}
        self._tx_cache: dict[int, list] = {}

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI  # 延迟导入：本地 selfcheck 不需要装 openai
            if not self.api_key:
                key_env = self.args.api_key_env or PROVIDERS[self.args.provider]["api_key_env"]
                names = API_KEY_ALIASES.get(key_env, (key_env,))
                raise SystemExit(
                    f"未找到 API key。请设置环境变量之一：{' 或 '.join(names)}\n"
                    f"例如：export {names[0]}=你的key"
                )
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                  timeout=self.args.timeout)
        return self._client

    def mask(self, map_id: int) -> np.ndarray:
        if map_id not in self._mask_cache:
            p = self.buildings / f"{map_id}.png"
            self._mask_cache[map_id] = np.array(Image.open(p).convert("L"))
        return self._mask_cache[map_id]

    def tx_list(self, map_id: int) -> list:
        if map_id not in self._tx_cache:
            p = self.antennas / f"{map_id}.json"
            self._tx_cache[map_id] = json.loads(p.read_text(encoding="utf-8"))
        return self._tx_cache[map_id]

    def _call_once(self, messages: list[dict]) -> str:
        """单次调用 + 指数退避重试。"""
        last = None
        for attempt in range(self.args.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.args.temperature,
                    "max_tokens": self.args.max_tokens,
                }
                if self.args.response_format_json:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                last = exc
                sleep = min(2.0 ** attempt, 30.0)
                log(f"    [retry {attempt + 1}/{self.args.max_retries}] {type(exc).__name__}: "
                    f"{str(exc)[:160]} → sleep {sleep:.0f}s")
                time.sleep(sleep)
        raise RuntimeError(f"调用失败 {self.args.max_retries} 次: {last!r}")

    def generate_sample(self, map_id: int, tx_id: int) -> dict[str, Any]:
        """生成一个样本的先验记录（不含落盘）。失败时抛出，由调用方记录。

        两种设计：
          v2_image    —— 让 VLM 从图上读 8 方位遮挡（**实测不可行**，见 reports/vlm_perception_probe.md，
                         保留它是为了将来换更强 VLM 时做对比）
          v3_geometry —— 程序精确测量几何，LLM 只做电磁推理
        """
        if self.args.design == "v3_geometry":
            return self._generate_v3(map_id, tx_id)
        return self._generate_v2(map_id, tx_id)

    # ---------------- v3：程序测量 + LLM 电磁推理 ----------------

    def _generate_v3(self, map_id: int, tx_id: int) -> dict[str, Any]:
        mask = self.mask(map_id)
        tx_list = self.tx_list(map_id)
        if tx_id >= len(tx_list):
            raise IndexError(f"map {map_id} 只有 {len(tx_list)} 个 Tx，请求 tx_id={tx_id}")
        tx_xy = tx_list[tx_id]

        # 1) 程序侧精确测量（不经过 LLM）
        facts = compute_geometry_facts(mask, tx_xy, radius_m=self.args.local_radius_m,
                                       px_per_m=self.args.px_per_m)
        messages = build_geometry_facts_messages(
            mask, map_id, tx_id, tx_xy, freq_ghz=self.args.freq_ghz,
            px_per_m=self.args.px_per_m, radius_m=self.args.local_radius_m,
            zoom=self.args.local_zoom, include_image=self.args.v3_include_image,
            facts=facts)

        # 2) LLM 只产出电磁字段；同样多轮采样取共识
        raw_texts: list[str] = []
        parsed: list[RadioPriorEM] = []
        errors: list[str] = []
        for k in range(self.args.samples_per_call):
            try:
                text = self._call_once(messages)
                raw_texts.append(text)
                try:
                    parsed.append(RadioPriorEM(**extract_json(text)))
                except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"sample {k}: {type(exc).__name__}: {str(exc)[:300]}")
                    if self.args.feedback_retry:
                        fb = messages + [
                            {"role": "assistant", "content": text[:2000]},
                            {"role": "user",
                             "content": "That JSON was rejected by the schema validator with:\n"
                                        f"{str(exc)[:600]}\n"
                                        "Reply with a corrected JSON object only, same semantics, "
                                        "fixing the offending fields."},
                        ]
                        try:
                            t2 = self._call_once(fb)
                            raw_texts.append(t2)
                            parsed.append(RadioPriorEM(**extract_json(t2)))
                            errors[-1] += " | 反馈重试成功"
                        except Exception as exc2:  # noqa: BLE001
                            errors.append(f"sample {k} feedback: {type(exc2).__name__}: "
                                          f"{str(exc2)[:300]}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"sample {k}: {type(exc).__name__}: {str(exc)[:300]}")

        if len(parsed) < self.args.min_ok:
            raise RuntimeError(f"有效采样 {len(parsed)} < min_ok {self.args.min_ok}；错误：{errors}")

        agg_em, consistency = aggregate_em(parsed)
        prior = merge_em_with_geometry(agg_em, facts)

        rec = {
            "schema_version": f"{SCHEMA_VERSION}+geomfacts",
            "sample_id": make_sample_id(map_id, tx_id),
            "map_id": int(map_id), "tx_id": int(tx_id),
            "prior": prior.model_dump(),
            "provenance": {
                "design": "v3_geometry",
                # 记清每个字段的来源 —— 这是 prior_geometry_only 对照能成立的前提
                "field_sources": {"em": "llm", "sectors": "program", "local_ratios": "program"},
                "model": self.model, "base_url": self.base_url,
                "prompt_version": "prompt_v3", "temperature": self.args.temperature,
                "samples_per_call": self.args.samples_per_call, "n_parsed": len(parsed),
                "n_errors": len(errors), "errors": errors[:5],
                "field_consistency": consistency,
                "geometry_facts": {"tx_on_building": facts["tx_on_building"],
                                   "facts_text": facts["facts_text"]},
                "v3_include_image": bool(self.args.v3_include_image),
                "tx_xy": [float(tx_xy[0]), float(tx_xy[1])],
                "freq_ghz": self.args.freq_ghz, "px_per_m": self.args.px_per_m,
                "prompt_sha1": hashlib.sha1(
                    (messages[1]["content"][-1]["text"]).encode("utf-8")).hexdigest()[:16],
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        }
        return rec

    # ---------------- v2：让 VLM 读图（实测不可行，保留作对比）----------------

    def _generate_v2(self, map_id: int, tx_id: int) -> dict[str, Any]:
        mask = self.mask(map_id)
        tx_list = self.tx_list(map_id)
        if tx_id >= len(tx_list):
            raise IndexError(f"map {map_id} 只有 {len(tx_list)} 个 Tx，请求 tx_id={tx_id}")
        tx_xy = tx_list[tx_id]
        messages = build_messages(mask, map_id, tx_id, tx_xy,
                                  freq_ghz=self.args.freq_ghz,
                                  px_per_m=self.args.px_per_m,
                                  radius_m=self.args.local_radius_m,
                                  zoom=self.args.local_zoom)

        raw_texts: list[str] = []
        parsed: list[RadioPriorV2] = []
        errors: list[str] = []
        for k in range(self.args.samples_per_call):
            try:
                text = self._call_once(messages)
                raw_texts.append(text)
                try:
                    parsed.append(RadioPriorV2(**extract_json(text)))
                except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"sample {k}: {type(exc).__name__}: {str(exc)[:400]}")
                    if self.args.feedback_retry:
                        # 带错误反馈重试一次：把校验错误告诉模型让它改，而不是替它填默认值
                        fb = messages + [
                            {"role": "assistant", "content": text[:2000]},
                            {"role": "user",
                             "content": "That JSON was rejected by the schema validator with:\n"
                                        f"{str(exc)[:800]}\n"
                                        "Reply with a corrected JSON object only, keeping the "
                                        "same semantic content but fixing the offending fields."},
                        ]
                        try:
                            text2 = self._call_once(fb)
                            raw_texts.append(text2)
                            parsed.append(RadioPriorV2(**extract_json(text2)))
                            errors[-1] += " | 反馈重试成功"
                        except Exception as exc2:  # noqa: BLE001
                            errors.append(f"sample {k} feedback: {type(exc2).__name__}: "
                                          f"{str(exc2)[:400]}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"sample {k}: {type(exc).__name__}: {str(exc)[:400]}")

        if len(parsed) < self.args.min_ok:
            raise RuntimeError(f"有效采样 {len(parsed)} < min_ok {self.args.min_ok}；错误：{errors}")

        agg, consistency = aggregate(parsed)

        # 程序侧交叉核对（**不喂给 LLM**，只在生成后用于审计"LLM 有没有真的看图"）
        geo = geometric_obstruction(mask, tx_xy, radius_m=self.args.local_radius_m,
                                    px_per_m=self.args.px_per_m)
        agree = [agg_s["obstruction"] == geo[agg_s["azimuth_deg"]]["obstruction"]
                 for agg_s in agg["sectors"]]
        program_check = {
            "tx_local_building_ratio_50m": local_building_ratio(mask, tx_xy, 50.0, self.args.px_per_m),
            "tx_local_building_ratio_100m": local_building_ratio(mask, tx_xy, 100.0, self.args.px_per_m),
            "sector_obstruction_agreement": float(np.mean(agree)),
            "geometric_obstruction": {str(az): geo[az] for az in AZIMUTHS},
        }

        rec = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": make_sample_id(map_id, tx_id),
            "map_id": int(map_id),
            "tx_id": int(tx_id),
            "prior": agg,
            "provenance": {
                "model": self.model,
                "base_url": self.base_url,
                "prompt_version": self.args.prompt_version,
                "temperature": self.args.temperature,
                "samples_per_call": self.args.samples_per_call,
                "n_parsed": len(parsed),
                "n_errors": len(errors),
                "errors": errors[:5],
                "field_consistency": consistency,
                "program_check": program_check,
                "tx_xy": [float(tx_xy[0]), float(tx_xy[1])],
                "freq_ghz": self.args.freq_ghz,
                "px_per_m": self.args.px_per_m,
                # 送进模型的图与文本的哈希：以后要查"某条先验到底对应哪张图"时用
                "prompt_sha1": hashlib.sha1(
                    (messages[1]["content"][0]["image_url"]["url"]
                     + messages[1]["content"][1]["text"]).encode("utf-8")
                ).hexdigest()[:16],
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        }
        return rec


# ===== 主流程 =====


def load_tasks(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.samples_file:
        lines = Path(args.samples_file).read_text(encoding="utf-8").split()
        tasks = []
        for ln in lines:
            m, t = ln.split("_")
            tasks.append((int(m), int(t)))
        return tasks
    if args.splits and args.split:
        from src.data.splits import load_splits, samples_for_split
        sp = load_splits(args.splits)
        return samples_for_split(sp, args.split)
    if args.map_range:
        a, b = args.map_range.split(":")
        return [(int(m), t) for m in range(int(a), int(b)) for t in range(args.tx_per_map)]
    raise SystemExit("需要 --samples-file 或 (--splits + --split) 或 --map-range")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="批量生成 LLM 先验（多轮采样 + 共识）")
    ap.add_argument("--data", default="data/radiomapseer")
    ap.add_argument("--priors", required=True, help="先验输出目录，例如 priors/prompt_v2/glm-4v-flash")
    ap.add_argument("--splits", default=None)
    ap.add_argument("--split", default=None, choices=["train", "val", "test"])
    ap.add_argument("--samples-file", default=None, help="每行一个 map_tx 的清单")
    ap.add_argument("--map-range", default=None, help="例如 0:5（不含右端）")
    ap.add_argument("--tx-per-map", type=int, default=80)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个（0=全部），用于试生成")
    ap.add_argument("--design", default="v3_geometry",
                    choices=["v2_image", "v3_geometry"],
                    help="v3_geometry（默认）：程序精确测量几何 + LLM 只做电磁推理；"
                         "v2_image：让 VLM 从图里读 8 方位遮挡（实测不可行，保留作对比）")
    ap.add_argument("--v3-include-image", dest="v3_include_image",
                    action="store_true", default=True,
                    help="v3 是否仍把图发给 LLM（默认发；--v3-no-image 可做消融）")
    ap.add_argument("--v3-no-image", dest="v3_include_image", action="store_false")
    ap.add_argument("--provider", default="zhipu", choices=sorted(PROVIDERS))
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--samples-per-call", type=int, default=3, help="每个样本采样几次做共识")
    ap.add_argument("--min-ok", type=int, default=2, help="至少几条有效采样才算成功")
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="单次回复上限。注意 glm-4v-flash 的限制是 [1,1024]，"
                         "设大了会被 API 直接拒（code 1210）")
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--freq-ghz", type=float, default=0.9)
    ap.add_argument("--px-per-m", type=float, default=1.0)
    ap.add_argument("--local-radius-m", type=float, default=100.0)
    ap.add_argument("--local-zoom", type=int, default=2)
    ap.add_argument("--prompt-version", default="prompt_v2")
    ap.add_argument("--response-format-json", action="store_true", default=True)
    ap.add_argument("--no-response-format-json", dest="response_format_json", action="store_false")
    ap.add_argument("--feedback-retry", action="store_true", default=True)
    ap.add_argument("--no-feedback-retry", dest="feedback_retry", action="store_false")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--dry-run", action="store_true", help="只渲染图片与任务清单，不调 API")
    args = ap.parse_args(argv)

    tasks = load_tasks(args)
    if args.limit:
        tasks = tasks[: args.limit]

    out_dir = Path(args.priors)
    fail_dir = out_dir / "_failed"
    out_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for m, t in tasks:
        f = out_dir / f"{make_sample_id(m, t)}.json"
        if args.resume and f.exists():
            try:
                json.loads(f.read_text(encoding="utf-8"))
                continue
            except Exception:  # noqa: BLE001  坏文件重跑
                pass
        todo.append((m, t))

    log(f"任务 {len(tasks)} 个，已存在 {len(tasks) - len(todo)} 个，待生成 {len(todo)} 个")
    log(f"输出目录: {out_dir}")
    log(f"模型: {args.model or '(preset)'} | 采样 {args.samples_per_call} 次/样本 | "
        f"temperature={args.temperature} | 并发 {args.workers}")

    if args.dry_run:
        log(f"[dry-run] 不调用 API。前 3 个待生成样本: {todo[:3]}")
        return 0

    gen = PriorGenerator(args)
    log(f"base_url={gen.base_url} model={gen.model} "
        f"api_key={'已设置（来自 ' + str(gen.api_key_source) + '）' if gen.api_key else '未设置'}")

    log_path = out_dir / "gen_log.jsonl"
    stats = {"ok": 0, "fail": 0, "skip": len(tasks) - len(todo)}
    t0 = time.time()

    def work(mt: tuple[int, int]) -> tuple[int, int, bool, str]:
        m, t = mt
        try:
            rec = gen.generate_sample(m, t)
            f = out_dir / f"{rec['sample_id']}.json"
            tmp = f.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(f)          # 原子落盘：中断也不会留下半个文件
            return m, t, True, ""
        except Exception as exc:  # noqa: BLE001
            payload = {"sample_id": make_sample_id(m, t), "map_id": m, "tx_id": t,
                       "error": f"{type(exc).__name__}: {str(exc)[:2000]}",
                       "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            (fail_dir / f"{make_sample_id(m, t)}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return m, t, False, str(exc)[:200]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(work, mt): mt for mt in todo}
        done = 0
        with log_path.open("a", encoding="utf-8") as lg:
            for fut in as_completed(futs):
                m, t, ok, err = fut.result()
                done += 1
                stats["ok" if ok else "fail"] += 1
                lg.write(json.dumps({"sample_id": make_sample_id(m, t), "ok": ok,
                                     "error": err, "at": time.time()}, ensure_ascii=False) + "\n")
                lg.flush()
                if done % 10 == 0 or not ok or done == len(todo):
                    el = time.time() - t0
                    rate = done / el if el > 0 else 0.0
                    remain = (len(todo) - done) / rate if rate > 0 else float("nan")
                    log(f"  [{done}/{len(todo)}] ok={stats['ok']} fail={stats['fail']} "
                        f"| {rate:.2f} 样本/s | 预计剩余 {remain / 60:.1f} min")

    log(f"\n完成：成功 {stats['ok']}，失败 {stats['fail']}，跳过（已存在）{stats['skip']}")
    log(f"耗时 {(time.time() - t0) / 60:.1f} min；失败样本见 {fail_dir}")
    if stats["fail"]:
        log("[注意] 存在失败样本。audit 会把它算作缺失；不要手工补默认值。")
    return 0 if stats["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
