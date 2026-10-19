# llm-prior-radiomap

> **LLM-produced electromagnetic priors for radio map generation — a falsifiable 3-layer criterion, and a negative result.**
>
> 判断"LLM 产出的电磁先验到底有没有信息"：一套可证伪的三层判据 + 三仪器类型学，以及一次带正确对照的负面实验。

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 核心结论（TL;DR）

在 glm-4v-flash × prompt_v3 × RadioMapSeer 上，把 LLM 输出的 22 维电磁先验注入 FiLM 条件生成模型：

| 配置 | RMSE (dB) mean±std | 含义 |
|---|---|---|
| `prior_none`（无先验） | 8.64 ± 0.08 | 绝对地板 |
| `prior_geometry_only`（程序几何） | 8.38 ± 0.08 | 几何先验有用 |
| `prior_v3_masked_em`（电磁置零） | 8.40 ± 0.08 | 容量对齐的零信息对照 |
| `prior_real_v3`（真 LLM 电磁） | 8.43 ± 0.12 | 主实验 |
| `prior_real_v3_shuffled`（内容打乱） | 8.38 ± 0.08 | 内容 vs 通路 |

**一句话：程序精确测量的几何先验显著有用（−0.267 dB），但 LLM 的电磁先验不仅无益、反而轻微有害**——比"置零"差 +0.029 dB（p=0.015）、比"随机打乱"差 +0.045 dB（p<0.001）。

阳性对照与天花板证明这不是"通路坏了"也不是"任务饱和"：

| 对照 | 差值 | 结论 |
|---|---|---|
| `oracle − masked_em`（真值统计量灌入 EM 维） | −0.302 dB (p≈1e-17) | EM 维度**能**携带真信号，判据有区分度 |
| `prior_none_big − prior_none`（4× 容量） | −0.833 dB (p≈1e-112) | 8.4 dB **不是**任务地板，还有空间 |

完整数字见 [reports/remote_20261018/reports/summary.md](reports/remote_20261018/reports/summary.md)（由 `report.py` 自动生成，禁止手抄）。

---

## 这是什么

让多模态 LLM 对无线电场景做**电磁推理**（结构化 prior），注入 U-Net + FiLM 生成 pathloss map。
本项目把"先验到底有没有信息"当作**第一个**要解决的问题，而不是最后一个。

核心主张：**在论证"LLM 先验有用"之前，必须先论证"LLM 先验有信息"**，而后者需要多个互相不蕴含的测量来三角定位——单个指标永远不够（低 R² 同时与"有信息"和"纯噪声"相容）。

## 目录

- [方法：三层判据 + 三仪器](#方法三层判据--三仪器)
- [实验与结果](#实验与结果)
- [快速开始（复现）](#快速开始复现)
- [目录结构](#目录结构)
- [测试](#测试)
- [文档](#文档)

---

## 方法：三层判据 + 三仪器

### 三层判据（层层递进、互不蕴含）

| 层 | 问题 | 通过意味着 | **不**意味着 |
|---|---|---|---|
| **L0 非退化** | 先验取值真的随样本变化吗 | 它至少是个变量 | 与场景有关 |
| **L1 接地** | 它的变化与场景真值有关吗 | 反映场景 | 下游用得上 |
| **L2 增量** | 注入后在留出地图上优于同容量对照吗 | 有下游效用 | 效用来自"内容"而非容量 |

### 三仪器（L1 的三角定位）

| 仪器 | 问的问题 | 来源 |
|---|---|---|
| 1 可重复性 | 同一输入多轮采样取值稳不稳 | `provenance.field_consistency`（生成时已记录） |
| 2 接地 | 与给定几何事实的 Spearman ρ（预指定配对 + 错位对照 + 置换检验） | `provenance.geometry_facts` |
| 3 可复现性 | 几何特征 5 折 CV 能拟合出它多少（R²） | 同上 |
| 3b 冗余复核 | 只用**单个**几何量能否拟合出它 | 同上 |

另有两条**自校准**诊断（不依赖人工阈值）：**地图层内 ρ**（区分"随 Tx 变" vs "地图级常量"）、**量程归一化散度**（`std÷量程 ≤ 0.05` 才判恒定——因为一致率的容差是 ±10% 量程）。

`scripts/probe_l1_typology.py` 零 API / 零 GPU 即可跑出完整三仪器表。

---

## 实验与结果

### 实验设置

- **数据**：RadioMapSeer（50 maps / 4000 Tx），leave-maps-out（35/5/10），test 只评一次。
- **先验**：prompt_v3——程序用 numpy 精确测量几何（8 方位遮挡剖面 + Tx 局部建筑占比）作为**事实**喂给 LLM，LLM 只输出 8 个电磁数值 + 2 个类别字段；多轮采样取共识。共 4000 条。
- **模型**：U-Net（base=32）+ FiLM（identity 初始化），主实验与对照参数量**逐位相同**（6,608,161）。
- **训练**：50 epoch × 3 seed；`train.amp=false`（见下方"两个坑"）。

### L0 / L1：先验非退化，但 22 维里只有 1 个接地

- **L0 全过**：唯一向量占比 0.9998（上一版同设计 0.0138）、同图内唯一占比 0.9997、数值字段最少唯一值 20、方位一致率 1.0、自校准增益 0.494。
- **L1**：三仪器判定，`los_ratio` 是唯一真接地字段（ρ=+0.73、5/5 地图同号），但它与程序已喂的几何量高度单调相关（单驱动量样本内 R²=0.76），只是"几何的重新编码"；`diffraction_ratio`、`pathloss_max_db` 是**方向反了的"自信地错"**；其余是常数/噪声/幻觉。详见 [reports/llm_l1_typology.md](reports/llm_l1_typology.md)。

### L2：LLM 电磁先验无用且有害

见顶部 TL;DR 表。逐样本看，`real_v3` 相对 `masked_em` 有 **52.6% 的样本变差**。

---

## 快速开始（复现）

完整复现手册见 [REPRODUCE.md](REPRODUCE.md)，这里只列主流程：

```bash
pip install -r requirements.txt

# 1) 数据与划分
python scripts/verify_data.py --data data/radiomapseer
python scripts/make_splits.py --data data/radiomapseer --out data/splits.json

# 2) L1 三仪器类型学（零 API / 零 GPU，先看清哪些字段有信息）
python -m scripts.probe_l1_typology --priors priors/prompt_v3/glm-4v-flash

# 3) 生成先验（需要智谱 API key）
export ZHIPUAI_API_KEY=...
python -m src.prior.generate --data data/radiomapseer \
    --priors priors/prompt_v3/glm-4v-flash --split train --workers 8 \
    --temperature 0.7 --samples-per-call 3

# 4) 训练 + 评测 + 报告（阶梯 × 3 seed，可并行）
SEEDS="0 1 2" PAR=2 DETACH=1 bash scripts/run_v3_full.sh

# 5) 阳性对照 + 容量天花板
PY=/root/miniconda3/bin/python DETACH=1 bash scripts/run_positive_controls.sh
```

> 关键配置：`configs/base.yaml` 里 `train.amp` **必须为 false**（见下）；正式训练在 Linux + NVIDIA GPU 上跑（[RUNBOOK_AutoDL.md](RUNBOOK_AutoDL.md)）。

---

## 目录结构

```
llm-prior-radiomap/
├── configs/           # base + v3 消融阶梯 + oracle/天花板
├── data/splits.json   # leave-maps-out 划分（冻结）
├── priors/            # LLM 产出（按 prompt 版本 + 模型分目录，不进 git）
├── reports/           # audit / L1 类型学 / 最终 summary（自动生成）
├── scripts/           # 数据、先验、训练阶梯、probe_* 分析、复现脚本
├── src/
│   ├── prior/         # schema、prompt_v2/v3、generate（多轮共识）、audit（门槛）
│   ├── data/          # normalize（dB 口径唯一来源）、splits、sectors、dataset
│   ├── models/        # unet、film（identity init）、cot_encoder、factory
│   ├── train/         # train.py、losses.py
│   └── eval/          # metrics、paired_stats、run_eval、report、visualize
└── tests/             # test_local / test_gate / test_typology / test_pipeline
```

---

## 测试

```bash
python -m tests.test_local       # 46 项：schema/几何/数据集/模型/SSIM 对拍/损失/配置
python -m tests.test_gate        # audit 门槛区分度
python -m tests.test_typology    # L1 类型学判据（含"先验必须影响前向"）
python -m tests.test_pipeline    # 端到端：训练→ckpt→评测→metrics 契约
```

---

## 两个会把结果全做假的坑（已修复并有回归自检）

| 坑 | 症状 | 修复 |
|---|---|---|
| **AMP + SSIM 梯度 NaN** | GradScaler 静默跳过更新，模型停在随机初始化（val RMSE 钉在 212.9 dB） | `train.amp=false` |
| **FiLM identity 初始化把末层权重置零** | 先验从不影响前向、梯度恒 0 → 主实验与对照数学恒等 | 末层权重改 1e-4 小随机值 |

两者都不报错、指标照出、结论全错。`test_local: film_prior_actually_conditions` 与
`test_typology: prior.actually_conditions_model` 直接断言"换先验必须改变输出、FiLM 每层
梯度非零"，把这两条锁死。

---

## 文档

- [PAPER.md](PAPER.md) — 论文正文（摘要 / 方法 / 结果 / 讨论 / 结论）。
- [REPRODUCE.md](REPRODUCE.md) — 从零复现全部数字的精确步骤。
- [AGENTS.md](AGENTS.md) — 项目约定、配对对照清单、历次实测教训。
- [RUNBOOK_AutoDL.md](RUNBOOK_AutoDL.md) — 远程 GPU 机器操作手册。

## 边界（诚实声明）

当前结论限于 **glm-4v-flash × prompt_v3 × RadioMapSeer**。换更强的 VLM、或换"让 LLM 真正
看图"的设计（而非把几何直接喂给它），结论可能不同——这正是同一套判据下一步的应用场景
（跨模型迁移）。

## License

[MIT](LICENSE)
