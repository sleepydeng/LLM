#!/usr/bin/env python3
"""报告链路端到端自检：train.py → run_eval.py → report.py → visualize.py。

为什么单独一个测试
-----------------
`tests/test_pipeline.py` 验证到"训练 + 评测 + metrics 契约"为止；
`report.py` / `visualize.py` 之前只用**手工造的** metrics.json 测过。
两者之间是**跨模块的字段契约**（metrics.json 的键名、history.json 的结构、
checkpoint 里要存什么）。这类契约一旦对不上，代价是"等 12 次训练全部跑完、
准备出报告时才发现表是空的" —— 这是本项目最不该发生的一类浪费。

所以这里用**真实产出**把四个脚本串起来跑一遍：
    4 个核心配置（smoke 2 epoch）→ run_eval(test) → report.py → visualize curves/predict

评测刻意用 `--weights raw`：冒烟 2 epoch 时 EMA 还停在初始化附近，
masked/real/shuffled 三者会给出完全相同的指标，配对检验退化。
raw 权重会因首步梯度不同而分叉，才能得到非退化的配对比较。

运行：python -m tests.test_report_integration
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from src.utils.config import deep_merge, load_config  # noqa: E402
from tests.test_pipeline import CONFIGS, N_TX, build_tiny_workspace  # noqa: E402


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="radiomap_report_"))
    failures: list[str] = []
    try:
        ws = build_tiny_workspace(tmp)
        runs_dir = tmp / "runs"
        out_md = tmp / "summary.md"
        out_json = tmp / "summary.json"
        curves = tmp / "curves.png"
        preds_dir = tmp / "preds"

        base = load_config(ROOT / "configs" / "base.yaml")
        base.pop("_base_", None)

        from src.eval.report import main as report_main
        from src.eval.run_eval import main as run_eval_main
        from src.eval.visualize import main as viz_main
        from src.train.train import main as train_main

        # ---------- 1) 训练 + 评测 ----------
        for name, override in CONFIGS:
            cfg = deep_merge(base, override)
            cfg["runs_dir"] = str(runs_dir)
            cfg["data"] = deep_merge(cfg["data"], {
                "root": ws["data_root"], "splits": ws["splits"], "num_workers": 0})
            cfg["prior"] = deep_merge(cfg["prior"], {"dir": ws["priors"]})
            cfg["train"] = deep_merge(cfg["train"], {"epochs": 2, "batch_size": N_TX})
            for k in ("_config_path", "_config_name"):
                cfg.pop(k, None)
            cfg_path = tmp / f"{name}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                                encoding="utf-8")

            if train_main(["--config", str(cfg_path), "--seed", "0", "--smoke",
                           "--device", "cpu"]) != 0:
                failures.append(f"{name}: 训练失败")
                continue
            run_dir = runs_dir / f"{name}_seed0"
            if run_eval_main(["--run-dir", str(run_dir), "--split", "test",
                              "--weights", "raw", "--device", "cpu",
                              "--batch-size", str(N_TX)]) != 0:
                failures.append(f"{name}: 评测失败")

        # ---------- 2) 报告 ----------
        print("\n" + "=" * 70 + "\nreport.py\n" + "=" * 70)
        rc = report_main(["--runs", str(runs_dir), "--out", str(out_md),
                          "--json", str(out_json)])
        if rc != 0:
            failures.append(f"report.py 退出码 {rc}")
        if not out_md.is_file() or not out_json.is_file():
            failures.append("report.py 没有产出 markdown/JSON")
        else:
            md = out_md.read_text(encoding="utf-8")
            js = json.loads(out_json.read_text(encoding="utf-8"))

            # 四个配置都必须出现在主表里
            for name, _ in CONFIGS:
                if name not in md:
                    failures.append(f"报告里没有配置 {name}")

            # 两个关键差值必须真的算出来（而不是"未跑"占位）。
            # 注意报告用的是**完整配置名**（`prior_real − prior_masked`），
            # 且单 seed 时方差无法估计，会输出 `± —` 而不是 `± 0.00` ——
            # 这是刻意的防误读设计，断言必须允许它。
            for label in ("prior_real − prior_masked", "prior_real − prior_shuffled"):
                if label not in md:
                    failures.append(f"报告里缺少差值行 '{label}'")
            m = re.search(
                r"`prior_real − prior_masked`\s*=\s*([-+0-9.]+)\s*±\s*(—|[0-9.]+)\s*dB", md)
            if not m:
                failures.append("报告里没有算出 prior_real − prior_masked 的数值")
            # 单 seed 时必须明确说明"方差无法估计"，不能悄悄写 ±0.00
            if "± —" in md and "方差无法估计" not in md:
                failures.append("单 seed 时报告用了 ± — 却没有解释它，容易被误读成 ±0.00")

            # 必须包含失败样本/最坏情况那一节，且表里**真的有数据行**
            for kw in ("失败样本", "变差", "口径声明", "数据来源", "参数量对齐"):
                if kw not in md:
                    failures.append(f"报告缺少必需章节关键词：{kw}")
            if not re.search(r"\|\s*`\d+_\d+`\s*\|", md):
                failures.append("失败样本/最坏情况表里没有任何逐样本数据行")

            # JSON 结构化汇总必须含四个配置与关键字段
            body = json.dumps(js, ensure_ascii=False)
            for name, _ in CONFIGS:
                if name not in body:
                    failures.append(f"JSON 汇总里没有配置 {name}")
            for key in ("main_table", "ladder", "paired", "worst_samples", "caveats"):
                if key not in js:
                    failures.append(f"JSON 汇总缺少字段 {key}")
            print(f"  报告 {len(md)} 字符；real−masked = "
                  f"{(m.group(1) + ' ± ' + m.group(2) + ' dB') if m else '未解析到'}")

        # ---------- 3) 曲线 ----------
        print("\n" + "=" * 70 + "\nvisualize curves\n" + "=" * 70)
        if viz_main(["curves", "--runs", str(runs_dir), "--out", str(curves)]) != 0:
            failures.append("visualize curves 退出码非 0")
        if not curves.is_file() or curves.stat().st_size < 1000:
            failures.append("训练曲线图没有生成或过小")

        # ---------- 4) 预测三联图（用 train.py 真实产出的 ckpt）----------
        print("\n" + "=" * 70 + "\nvisualize predict\n" + "=" * 70)
        ckpt = runs_dir / "prior_real_seed0" / "ckpt" / "best.pth"
        if viz_main(["predict", "--ckpt", str(ckpt), "--out-dir", str(preds_dir),
                     "--split", "test", "--n", "2", "--prior-mode", "real",
                     "--device", "cpu"]) != 0:
            failures.append("visualize predict 退出码非 0")
        pngs = sorted(preds_dir.glob("*.png")) if preds_dir.is_dir() else []
        if not pngs:
            failures.append("predict 没有产出任何图")
        elif not any(p.stat().st_size > 1000 for p in pngs):
            failures.append("predict 产出的图过小，可能是空白图")
        print(f"  产出 {len(pngs)} 张预测图")

        # ---------- 3) 追加第二个 seed，验证 mean±std 路径 ----------
        # 上面每个配置只有 1 个 seed，报告会输出 `± —`（方差无法估计）。
        # 这里给 prior_real 补一个 seed，检查报告真的会变成 mean±std，
        # 并且**只对**有多个 seed 的那个配置给 std。
        print("\n" + "=" * 70 + "\n追加 prior_real 的第二个 seed\n" + "=" * 70)
        cfg = deep_merge(base, dict(CONFIGS)["prior_real"])
        cfg["runs_dir"] = str(runs_dir)
        cfg["data"] = deep_merge(cfg["data"], {
            "root": ws["data_root"], "splits": ws["splits"], "num_workers": 0})
        cfg["prior"] = deep_merge(cfg["prior"], {"dir": ws["priors"]})
        cfg["train"] = deep_merge(cfg["train"], {"epochs": 2, "batch_size": N_TX})
        for k in ("_config_path", "_config_name"):
            cfg.pop(k, None)
        # 配置文件名决定 exp_id 前缀（`<config名>_seed<seed>`），所以必须叫
        # `prior_real.yaml`、只换目录 —— 初版命名为 prior_real_seed1.yaml，
        # 结果 exp_id 变成 prior_real_seed1_seed1，run 落到了别的目录里，
        # 而我又没检查 run_eval 的返回码，于是"第二个 seed"被静默丢掉了。
        seed1_dir = tmp / "seed1"
        seed1_dir.mkdir(exist_ok=True)
        cfg2 = seed1_dir / "prior_real.yaml"
        cfg2.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
        if train_main(["--config", str(cfg2), "--seed", "1", "--smoke",
                       "--device", "cpu"]) != 0:
            failures.append("prior_real seed=1 训练失败")
        rc = run_eval_main(["--run-dir", str(runs_dir / "prior_real_seed1"), "--split", "test",
                            "--weights", "raw", "--device", "cpu", "--batch-size", str(N_TX)])
        if rc != 0:
            failures.append(f"prior_real seed=1 评测失败（退出码 {rc}）")
        if not (runs_dir / "prior_real_seed1" / "metrics_test.json").is_file():
            failures.append("prior_real seed=1 没有产出 metrics_test.json")

        out_md2 = tmp / "summary2.md"
        if report_main(["--runs", str(runs_dir), "--out", str(out_md2)]) != 0:
            failures.append("第二次 report.py 退出码非 0")
        else:
            md2 = out_md2.read_text(encoding="utf-8")
            # 只在**主表**区段内解析 n_seeds。配置名在报告里出现多次
            # （主表 / 阶梯表 / 参数量对齐表 / 数据来源表），
            # 初版直接对全文 finditer，把来源表里的 seed 列当成了 n_seeds。
            main_block = md2.split("## 2.")[0]
            rows = {m.group(1): m.group(2).strip()
                    for m in re.finditer(r"\|\s*`(prior_\w+)`\s*\|\s*(\d+)\s*\|", main_block)}
            if rows.get("prior_real") != "2":
                failures.append(
                    f"prior_real 补了第二个 seed 后 n_seeds 应为 2，实际 {rows.get('prior_real')!r}")
            if rows.get("prior_masked") != "1":
                failures.append(
                    f"prior_masked 只有 1 个 seed，n_seeds 应为 1，实际 {rows.get('prior_masked')!r}")
            # prior_real 那行现在必须给出真实的 ± 数值（不再是 —）
            line_real = next((l for l in md2.splitlines()
                              if l.startswith("| `prior_real`")), "")
            if not re.search(r"\d+\.\d+\s*±\s*\d+\.\d+", line_real):
                failures.append(f"prior_real 有 2 个 seed 时应给出 mean±std，实际行：{line_real}")
            line_masked = next((l for l in md2.splitlines()
                                if l.startswith("| `prior_masked`")), "")
            if "± —" not in line_masked:
                failures.append(f"单 seed 的 prior_masked 应显示 ± —，实际行：{line_masked}")
            print(f"  prior_real（2 seed）：{line_real.strip()[:90]}")
            print(f"  prior_masked（1 seed）：{line_masked.strip()[:90]}")

        # ---------- 4) 只跑了一部分配置时，缺失的必须写"未跑" ----------
        print("\n" + "=" * 70 + "\n部分配置缺失（模拟消融还没跑）\n" + "=" * 70)
        partial = tmp / "partial_runs"
        partial.mkdir()
        for seed in (0, 1):
            src_dir = runs_dir / f"prior_real_seed{seed}"
            dst_dir = partial / f"prior_real_seed{seed}"
            dst_dir.mkdir()
            shutil.copy(src_dir / "metrics_test.json", dst_dir / "metrics_test.json")
        out_md3 = tmp / "partial.md"
        if report_main(["--runs", str(partial), "--out", str(out_md3)]) != 0:
            failures.append("部分配置时 report.py 退出码非 0")
        else:
            md3 = out_md3.read_text(encoding="utf-8")
            if "未跑" not in md3:
                failures.append("部分配置缺失时报告应显式标注「未跑」，实际没有")
            for missing in ("prior_none", "prior_masked", "prior_shuffled"):
                line = next((l for l in md3.splitlines()
                             if l.startswith(f"| `{missing}`")), "")
                if "未跑" not in line:
                    failures.append(f"{missing} 未跑时应标「未跑」，实际行：{line}")
            # 绝不能给未跑的配置编造数值
            for missing in ("prior_none", "prior_masked", "prior_shuffled"):
                line = next((l for l in md3.splitlines()
                             if l.startswith(f"| `{missing}`")), "")
                if re.search(r"\d+\.\d+\s*±", line):
                    failures.append(f"{missing} 未跑却给出了数值（编造数字）：{line}")
            print("  部分配置时的阶梯表：")
            for line in md3.splitlines():
                if line.startswith("| `prior_"):
                    print("   ", line.strip()[:100])

        # ---------- 5) 缺 metrics 时要给出清晰错误 ----------
        empty = tmp / "empty_runs"
        empty.mkdir()
        rc = report_main(["--runs", str(empty), "--out", str(tmp / "x.md")])
        if rc == 0:
            failures.append("runs 目录为空时 report.py 应当返回非 0 退出码")
        print(f"\n空目录时 report.py 退出码 = {rc}（应为非 0）")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 70)
    if failures:
        print("报告链路端到端自检失败：")
        for f in failures:
            print("  -", f)
        return 1
    print("报告链路端到端自检通过 [OK]："
          "train.py → run_eval.py → report.py → visualize(curves/predict) 契约全部对上")
    return 0


if __name__ == "__main__":
    sys.exit(main())
