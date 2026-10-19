# llm-prior-radiomap

**让多模态 LLM 输出对场景的电磁推理（结构化 prior），作为条件注入 U-Net 生成 pathloss map。**

这是上一版 `cot-radiomap` 的重做版。上一版跑通了 pipeline（800 样本验证集 RMSE 5.11 → 4.86 dB），
但**证据不支持核心论点**：4000 条先验只有 55 个唯一向量、`scene_type` 只有 2 种取值、
CoT 文本全项目零引用、无随机种子、样本级划分造成空间泄漏。完整复盘见
[`../CoT-RadioMap_实验分析报告.md`](../CoT-RadioMap_实验分析报告.md)，
设计说明与红线见 [`../新项目交接_LLM-Prior-RadioMap.md`](../新项目交接_LLM-Prior-RadioMap.md)。

本项目把"**先验的信息量**"当成第一个要解决的问题，而不是最后一个。

---

## 与上一版的四处关键改动

| 改动 | 上一版 | 本项目 |
|---|---|---|
| **输入信息** | 只给 LLM 一张全局建筑掩码 + Tx 坐标 | 给"全局掩码 + Tx 中心 100 m 局部放大图 + 8 条方位线"，并明确"掩码是二值的、无高度信息" |
| **先验结构** | 22 维，19 维基本是常量 | 64 维 = 全局数值 10 + 全局类别 14 + **sectors 8 方位遮挡剖面 40**（真正随 Tx 变化） |
| **生成方式** | 固定 5 个 few-shot + `temperature=0`（输出坍缩） | 无固定示例；`temperature=0.7` 采样 3 次取共识，共识度即不确定性度量 |
| **准入门槛** | 无 | `scripts/audit.py` 三条硬门槛不过就**禁止进入训练** |

另外修掉的两个旧 bug：**FiLM identity 初始化**（γ=1、β=0）与**参数量对齐 baseline**
（`prior_masked` 结构与 `prior_real` 逐位相同，只把先验置零，所以 `real − masked` 才可归因于先验信息）。

---

## 快速开始

```bash
pip install -r requirements.txt

# 1) 数据校验（50 maps / 4000 gain / 50 antenna，不通过不许开工）
python scripts/verify_data.py --data data/radiomapseer

# 2) 冻结 leave-maps-out 划分（35 train / 5 val / 10 test maps）
python scripts/make_splits.py --data data/radiomapseer --out data/splits.json

# 2.5) L1 三仪器类型学（**零 API / 零 GPU**：先看清哪几个字段真有信息，再决定训什么）
python -m scripts.probe_l1_typology --priors priors/prompt_v3/glm-4v-flash
#   → reports/llm_l1_typology.{md,json}：三仪器表 + 逐字段判读 + 条件向量该放什么
#   只读 provenance，不需要 data/radiomapseer/；--validate-against 可与旧接地检验对拍

# 3) 先验：几何可行性检查 → 试生成 → 体检 → 全量 → 再体检
export ZHIPUAI_API_KEY=...
python scripts/pilot_preflight.py --maps 0:4          # 0 成本：看门槛有没有区分度
python -m src.prior.generate --priors priors/prompt_v2/glm-4v-flash --map-range 0:4 --workers 6
python scripts/audit.py --priors priors/prompt_v2/glm-4v-flash --data data/radiomapseer \
       --out reports/audit_pilot.json          # 退出码 1 = 没过门槛，改 prompt 重跑

# 4) 训练（四配置 × 3 seed）
python -m src.train.train --config configs/prior_real.yaml --seed 0

# 5) 评测（test maps 只评一次；权重口径必须显式）
python -m src.eval.run_eval --run-dir runs/prior_real_seed0 --split test --weights ema

# 6) 自动报告（禁止手抄数字）
python -m src.eval.report --runs runs --out reports/summary.md
```

---

## 目录结构

```
llm-prior-radiomap/
├── AGENTS.md                  # 项目约定与"已知的坑"（含实测出的门槛经验）
├── RUNBOOK_AutoDL.md          # 远程 GPU 机器上的逐步操作手册
├── configs/                   # base + 四配置阶梯 + CoT/注入方式消融
├── data/splits.json           # leave-maps-out 划分（冻结）
├── priors/                    # LLM 产出，按 prompt 版本 + 模型分目录
│   └── <version>/<llm>/       # 每条先验含 prior 内容 + provenance（来源、共识度）
├── reports/                   # audit 报告与自动生成的汇总
│   └── llm_l1_typology.{md,json}  # L1 三仪器类型学（自动生成，见下）
├── scripts/                   # verify_data / make_splits / audit / probe_* / shell 流程
├── src/
│   ├── prior/                 # schema v2、prompt_v2、generate（多轮共识）、audit（门槛）
│   ├── data/                  # normalize（dB 口径唯一来源）、splits、sectors 几何、dataset
│   ├── models/                # unet、film（identity init）、cot_encoder、factory
│   ├── train/                 # train.py（单入口）、losses.py
│   ├── eval/                  # metrics、paired_stats、run_eval、report、visualize
│   └── utils/                 # seed、config、logging
└── tests/
    ├── test_local.py          # 33 项本地自检（不需要 GPU / API key）
    └── test_pipeline.py       # 端到端：四配置 训练→ckpt→评测→metrics 契约
```

---

## L1 三仪器类型学（先验到底有没有信息）

`audit.py` 回答 **L0**（先验是不是退化的），但它不回答 **L1**（先验的取值是不是
**与场景有关**）。L1 又**不能靠单个指标回答**：低 R² 同时与"有信息"和"纯噪声"相容，
高一致率同时与"学到真规律"和"输出模板"相容。所以用三个**互相不蕴含**的仪器三角定位
（`scripts/probe_l1_typology.py`，零 API / 零 GPU，只读 `provenance`）：

| 仪器 | 问的问题 | 数据来源 |
|---|---|---|
| 1 可重复性 | 同一输入多轮采样，取值稳不稳？ | `provenance.field_consistency`（生成时已记录） |
| 2 接地 | 它的变化跟**给定几何事实**有关吗？（预指定配对 + 错位对照 + 置换检验） | `provenance.geometry_facts` 重新解析 |
| 3 可复现性 | 几何特征能不能**拟合出**它？（5 折 CV，Ridge/GBM） | 同上 |
| 3b 冗余复核 | 只用**单个**几何量能不能拟合出它？ | 同上 |

另有两条**自校准**诊断（不依赖人工阈值）：

- **地图层内 ρ**：把样本按地图分层重算。若总体 ρ 显著而地图内 ρ 反复变号，
  那段总体相关来自**地图之间的几何差异**，字段对"同图不同 Tx"没有分辨力 ——
  这正是上一版"地图级常量"失败模式的检测器；
- **逐图唯一值数**：同图 80 个 Tx 里该字段有几个不同取值。

结论由脚本自动写进 `reports/llm_l1_typology.md`（当前覆盖 371 条 prompt_v3 先验，
**数字以那份报告为准，不要在文档里手抄**）。读它时注意两点：

1. 一致率 1.0 **不等于**确定：容差是 ±10% 量程，对 `pathloss_max_db` 就是 ±19 dB；
2. `sectors` 与 `tx_local_building_ratio_*` 是**程序**测量，不在 LLM 类型学表里。

判据本身有自检：`python -m tests.test_typology` 覆盖几何文本解析、**方位角对齐**
（sectors 数组顺序被打乱时驱动量必须不变）、六个判据分支、小样本保护，
以及对拍机制**能失败**（把参考结果改坏必须报不一致）。此外还有一条反向自检：
若真的把方位与距离错配，驱动量**必须**变 —— 否则上面那条"对齐自检"没有功效。

---

## 消融阶梯（主表必须包含全部四档）

| 配置 | 条件分支 | 先验输入 | 回答什么问题 |
|---|---|---|---|
| `prior_none` | 无 | 无 | 纯场景图能做到什么 |
| `prior_masked` | 有（参数量逐位对齐） | 全零向量 | "多了一条条件通路"本身值多少 |
| `prior_real` | 有 | 真先验 | 主实验 |
| `prior_shuffled` | 有 | 真先验但样本间打乱 | 起作用的是先验**内容**还是那条通路（H2） |

**报告 `real − masked` 与 `real − shuffled`，而不只是 `real − none`。**

补充消融：`prior_global_only`（去掉 sectors）、`prior_sectors_only`（只用 sectors）、
`prior_sectors_channels`（sectors 改走 8 个楔形输入通道）、
`prior_real_cot` / `prior_real_cot_shuffled`（CoT 进模型，后者是**容量对照**）、
`prior_real_cot_only` / `prior_cot_only_shuffled`（只用 CoT）。

### 为什么每个"加东西"的实验都有配对的打乱对照

这是本项目最容易做错的地方，也是上一版留下的最深刻教训。任何"新模块带来提升"的结论，
都必须先扣掉**多出来的参数量**这个平凡解释：

| 对照 | 结构 | 差别 | 差值含义 |
|---|---|---|---|
| `prior_masked` vs `prior_real` | 逐位相同 | 零向量 vs 真向量 | 先验**内容**的贡献 |
| `prior_real_cot_shuffled` vs `prior_real_cot` | 逐位相同 | 打乱的 CoT vs 真 CoT | CoT**内容**的贡献 |
| `prior_real` vs `prior_real_cot` | 多一个 CoT 编码器 | 无 CoT vs 有 CoT | CoT 内容 **+** 额外容量 |

实测参数量：`prior_real` 6,608,161；`prior_real_cot` 与 `prior_real_cot_shuffled`
都是 **6,895,041**（CoT 编码器 +286,880，与 FiLM 分支的 288,064 同量级）。
所以只报 `real_cot − real` 会把 28.7 万参数的容量效应算成 CoT 的功劳。

## 三个必须被数据回答的假设

- **H1（有效性）**：真先验在**留出地图**上优于"无先验"的同容量 baseline。
- **H2（正确性）**：把先验在样本间打乱会显著变差 → 起作用的是先验内容。
- **H3（推理的作用）**：带 CoT 的先验优于不带 CoT 的同一 schema。

> 上一版只回答了 H1 的一半（+0.25 dB），且没做 H2，所以无法排除"多了一个小分支"这个平凡解释。

---

## 评测口径（写进论文，不要靠读者猜）

- 反归一化：`pl_db = pl_norm × (200 − 40) + 40`。**换区间 dB 数就变**，必须写明。
- 噪声地板 `gain_norm = max(gain_norm, 0.2)` 会把大量低增益像素钉在 200 dB，
  这批像素真值不是真实路损却主导 RMSE → 除主指标外**额外报 `mean_rmse_db_nonsaturated`**。
- 统计：同一批 test maps 上的**配对**比较（Wilcoxon signed-rank / 配对 t），报差值均值与 95% CI。
  **不要**用独立样本的 Cohen's d（上一版就是这里用错）。
- 选模型看 val，报结果看 test。`test_maps` 只在最后评一次。
- 报告必须包含**失败样本与最坏情况**（变差样本数/占比），不允许只报胜率。

---

## 本地自检状态

不需要 GPU 的自检里，`test_local` / `test_gate` / `test_typology` 三项**也不需要数据盘**
（`test_typology` 更是只读 `priors/` 里的 provenance）；`test_pipeline` /
`test_report_integration` 需要 torch 但不需 GPU。本地实跑结果：

| 自检 | 内容 | 结果 |
|---|---|---|
| `python -m tests.test_local` | schema 分组向量、方位几何方向、prompt 局部裁剪逐像素对齐、**高度类字段必须被 schema 拒绝**、**prompt 不得含高度区间或"推测高度"指令**、**渲染强制二值化**、四种 prior 模式、**shuffle_cot 只打乱文本不动向量**、缺失先验必须抛错、shuffled 无不动点、FiLM identity、参数量对齐、11 个 config 全部可加载可建模、**torch/numpy 两版 SSIM 对拍**、**fp16 输入下损失必须升到 fp32**、损失反传、指标 | **41 / 41 通过** |
| `python -m tests.test_pipeline` | 四配置真实命令行：训练 → best ckpt → 按 EMA 口径载入 → test 评测 → `metrics_test.json` 契约 | **通过** |
| `python -m tests.test_report_integration` | **报告链路**：`train.py → run_eval.py → report.py → visualize(curves/predict)` 用真实产出跑通，断言四配置都在主表、`prior_real − prior_masked` 与 `− prior_shuffled` 真的算出数值、失败样本表有逐样本数据行、空目录时给出非 0 退出码 | **通过** |
| `python -m tests.test_gate` | **门槛区分度**：造 5 份"只犯一种病"的先验集，确认每条硬门槛都因为正确的原因通过/失败，且阈值可达 | **通过** |
| `python -m tests.test_typology` | **L1 类型学判据**：几何文本解析（残缺必须抛错）、**sectors 数组顺序被打乱后驱动量不变**（反向自检：真错配时必须变）、六个判据分支逐个触发、小样本保护、`--validate-against` 一致时退出 0 / 被改坏时退出 3、报告渲染不崩 | **全部通过**（项数以脚本自报为准） |

其中几条是**专门用来防止重蹈覆辙**的：

- `prompt.local_panel_alignment` 逐像素校验局部裁剪图方向没有翻转（否则 sector 一致率检查会整体失效且极难察觉）；
- `losses.ssim_crosscheck` 保证训练优化的 SSIM 与报告里写的 SSIM 是同一个量；
- `dataset.missing_prior_raises` 保证缺先验时抛错而不是静默补零；
- `schema.rejects_height_fields` / `prompt.no_height_instruction` / `prompt.binarizes_mask`
  三条把"高度"这件事从数据、指令、图像三个入口全部堵死（见下）；
- `tests.test_gate` 保证 audit 门槛**不会把一份正确的 prompt 永远挡在门外** ——
  这是本项目踩过的第二个门槛设计错误；
- `tests.test_typology` 保证 L1 判据**不会静默贴错标签**：几何解析错位、
  扇形数组顺序错位、判据分支写错、小样本硬下结论，这四种都会让它立刻失败
  （前两种正是"不报错但结论全错"的类型）。

### 关于"高度"：它不是幻觉，是代码里真的有一个高度变量

上一版 4000 条先验里 **81.65%** 写出了凭空的建筑高度（"建筑高度 10-30m"），
另有 **11.87%（475 条）** 复读 28 GHz 模板。追下去发现根因不在 LLM：

| 位置 | 内容 | 问题 |
|---|---|---|
| `pipeline/generate_synthetic_data.py` | `building_height` 数组（米）、`tx_height` 15–35 m、遮挡判据 `building_height > tx_height*0.8`；**高度图用 `height/height.max()*255` 转灰度图** | 这是"更亮=更高"的出处 —— 在**合成数据**上它是真的 |
| `prompts/system_prompt.py:16` | "发射机(Tx)位置：(x, y, z) 坐标，z 为高度（米）" | 告诉模型存在高度 |
| `prompts/system_prompt.py:146-149` | "密集城区…建筑高度20-50m / 郊区5-12m" 等 | 递给模型一张高度区间表 |
| 同上 `:159` vs `:172` | 一边说"不含建筑高度数据"，一边说"请结合场景类型提示和常识进行推理" | **自相矛盾** |
| `prompts/few_shot_demos.py:19,35` | Demo1 = "建筑高度20-40m…Tx高度25m…28GHz" | 那 475 条复读的原文 |

而旧的 `RadioPrior` schema **从来没有** height 字段 —— 高度只活在 prompt 与 CoT 文本里，
所以 schema 校验永远拦不住它。合成数据管线被换成 RadioMapSeer 二值掩码时，这套 prompt 没跟着改。

新项目从三个入口堵死：**schema**（`extra="forbid"`，任何 height 字段直接报错）、
**prompt**（明确禁止 + 不再给高度区间 + 明确"2-D 数据集、Tx 无高度"）、
**渲染**（掩码强制二值化，即使误喂灰度高度图，模型也只能看到 {0,255}）。

额外交叉验证：本项目 `prior_none`（U-Net, base=32）参数量 **6,320,097**，与上一版消融模型的
6,320,097 **完全一致**，说明主干架构复现无误，5.11 dB 那个数字可比。

## 尚未完成（需要在 GPU / API 上做）

**已完成（本机、零 API）**：L1 三仪器类型学 —— `reports/llm_l1_typology.{md,json}`，
覆盖 `prior.provenance.field_consistency` 与新写的 `scripts/probe_l1_typology.py`，
本地自检 `test_local` 45/45、`test_gate` 通过。

1. **先验只覆盖 5 张地图**（maps 0–4，其中 4 只生成到 Tx 51）。要跑训练必须
   先把 train/val/test 三个 split 的先验补齐到 4000 条量级并全部通过 audit 硬门槛
   （`prior.dir` 缺文件会**抛错**，这是有意的）；
2. 按类型学结论重排条件向量后再训练：主表至少要有
   `prior_real_v3` / `prior_v3_masked_em`（容量对齐）/ `prior_geometry_only` /
   `prior_real_v3_shuffled` 四档，3 seed，报 mean ± std；
3. test maps 上评测一次，用 `src/eval/report.py` 自动生成报告（禁止手抄）；
4. 下游天花板估计（同一张图其它 Tx 或更大容量/更多 epoch）——
   没有它就无法把"增量≈0"归因于先验无用而不是任务饱和；
5. 跨模型迁移：换一个不同厂商的 LLM 重跑同一批样本，看 L1 类型学是否一致。
