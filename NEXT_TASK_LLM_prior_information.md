# 下一个任务：判断 LLM 先验到底有没有信息（以及怎么判断）

> 交接自上一个会话（2026-10-13）。
> 读这份文件之前，**先读** `AGENTS.md`（项目规矩与已知的坑）与 `README.md`（代码结构与自检状态）。
> 完整背景（上一版实验为什么失败）见上一级目录的 `../新项目交接_LLM-Prior-RadioMap.md`
> 与 `../CoT-RadioMap_实验分析报告.md`。

---

## 0. 一句话任务

**不要再做"再挤一个 +0.X dB"的实验。** 要做的是：

> 建立一套**可证伪的判据**，用来回答"LLM 产出的电磁先验到底有没有信息"，并用它给出一个具体的、
> 可复现的结论（正面或负面都算成功）。方法本身（三层判据 + 三仪器类型学）是主要贡献。

用户已明确：**2601.18242 的 method 精读不用做**，新颖性核查到此为止
（ControlRadio 经全文核对**不占**这条路线：其全文 "LLM" 出现 0 次，文字是人写的目标规格、
编码器是 CLIP；其消融是"去掉文本分支"，不是内容对照）。

---

## 1. 为什么这个问题值得做（已核实的事实，**不要重测**）

### 1.1 三层判据，层层递进且互不蕴含

| 层 | 问题 | 通过意味着 | **不**意味着 |
|---|---|---|---|
| **L0 非退化** | 先验取值真的随样本变化吗（还是模板/常量）？ | 它至少是个变量 | 与场景有关 |
| **L1 接地** | 它的变化与**场景真值**有关吗（还是噪声）？ | 反映场景 | 下游用得上 |
| **L2 增量** | 条件注入后在**留出地图**上优于同容量对照吗？ | 有下游效用 | 效用来自"内容"而非容量/通路 |

现有文献基本只报 L2 且常无对照 → 不可证伪。**空隙就在这里。**

### 1.2 已实测的关键数字（直接引用，不要重跑）

**上一版 4000 条旧先验（反面基线，`reports/audit_legacy_v2.json`）**

```
唯一向量 55/4000｜unique_ratio 0.0138｜同 map 内唯一占比 0.1043
confidence top1 0.979｜penetration_loss_db 熵 0.65｜pathloss_min_db 唯一值 3
毫米波幻觉 475/4000 = 11.87%｜建筑高度幻觉 3266/4000 = 81.65%
audit 硬门槛：4 条 FAIL（unique_ratio / intra_map / numeric_unique_min / numeric_entropy_min）
```

**视觉模型感知能力（`reports/vlm_perception_probe.md`，人造图对照已证明图像管道正常）**

| 模型 | 左右半边谁更白 | 哪个方位建筑最近 | 白占比估计 MAE |
|---|---|---|---|
| glm-4v-flash | 2/8 = 0.250（随机 0.5） | — | 0.181 |
| glm-4v | 4/8 = 0.500 | — | 0.106 |
| glm-4v-plus | 2/8 = 0.250 | — | 0.258 |
| 三模型合计 | — | **0/6 = 0.000**（随机 0.125） | — |

**关键：换 5 种输入图（双面板 / 去字幕 / 只有局部 / 纯掩码裁剪 / 只有全局）答案完全相同** →
模型不是在读图，是在给一个位置偏置的默认词。

**`prompt_v3`（程序给精确几何、LLM 只做电磁推理）240 样本：7 条硬门槛全部 PASS**
（`reports/audit_v3_240.json`）—— 这是本项目第一次有先验集过门槛。

**接地检验（240 样本，Spearman 标准误 0.065）（`reports/llm_grounding_240.json`）**

| 配对（预先指定的物理假设） | 真实 ρ | 错位对照 ρ | 判读 |
|---|---|---|---|
| `los_ratio` ↔ 最近建筑距离 | **+0.719** | +0.061 | ✅ 强、方向对 |
| `los_ratio` ↔ 被挡方位数 | **−0.656** | −0.034 | ✅ |
| `reflection_ratio` ↔ 最近建筑距离 | −0.333 | −0.021 | ✅ |
| `diffraction_ratio` ↔ 被挡方位数 | −0.265 | −0.005 | ⚠️ 与预期**反号** |
| `scattering_ratio` ↔ 100m 建筑占比 | +0.112 | +0.101 | ❌ 无信号 |
| `pathloss_max_db` ↔ 被挡方位数 | −0.080 | −0.042 | ❌ 无信号 |

方向正确 **6/10**，显著强于错位对照 **5/10**。

**可解释方差（371 条先验，5 折 CV）（`reports/llm_distillability.json`）**

| LLM 字段 | 目标 std | 几何特征能解释 R²（最好） | 错位对照 R² |
|---|---|---|---|
| `los_ratio` | 0.071 | 0.43 | −0.00 |
| `reflection_ratio` | 0.100 | 0.26 | −0.02 |
| `diffraction_ratio` | 0.094 | 0.05 | −0.01 |
| `scattering_ratio` | 0.061 | 0.07 | −0.01 |
| `pathloss_min_db` | **7.58** | 0.03 | +0.01 |
| `pathloss_max_db` | **17.33** | 0.05 | −0.02 |
| `permittivity` | 0.577 | 0.06 | −0.02 |
| `penetration_loss_db` | 0.992 | −0.01 | −0.02 |

**没有任何字段能由几何复现（最高 0.43）。**

### 1.3 **L1 的关键二义性 —— 这是整个任务的智力核心**

**低 R² 同时与"有信息"和"纯噪声"相容。** 噪声当然也无法由几何复现。
所以**单个指标永远不够**，必须多仪器三角定位。

### 1.4 L1 三仪器类型学（**建议作为方法论主贡献**）

三个正交仪器（前两个已实现，第三个待补）：

1. **可重复性**：同一样本多轮采样的字段一致率（已在 `provenance.field_consistency` 里，**逐字段**）
2. **接地**：与几何的秩相关 ρ，对照"同图另一个 Tx"的错位 → `scripts/probe_llm_grounding.py`
3. **可复现性**：几何特征能解释的方差 R²，含错位对照 → `scripts/probe_llm_distillability.py`

| 可重复性 | 接地 ρ | 可复现 R² | 类型 | 对条件向量的意义 |
|---|---|---|---|---|
| 高 | 高 | 高 | 真实但**冗余** | LLM 在做特征工程，直接喂输入更好 |
| 高 | 高 | 低 | 真实且有**残余信息** | ✅ 唯一支持继续的类型 |
| 高 | ≈0 | 低 | **自信地错**（系统性幻觉） | 有害 |
| 低 | ≈0 | 低 | **噪声** | 有害 |
| 高 | ≈0 | 低，**方差大** | **幻觉** | 比恒定更糟：下游得额外学会忽略它 |
| 高 | — | — | 方差≈0 | **合法常数**（材料属性） | 无区分度，不该指望它带信息 |

用现有数据填表：`pathloss_min/max` 是第 5 行典型（std 7.6/17.3 dB、ρ≈0、R²≈0.03）；
`los_ratio`/`reflection_ratio` 落在第 2 行 —— **它们是唯一值得往下走的字段**。

---

## 2. 要做的事（按顺序）

### 第 0 步：补上 L1 的第三个仪器（**零 API 成本，先做这个**）

从 `priors/prompt_v3/glm-4v-flash/*.json` 的 `provenance.field_consistency` 里
**逐字段**提取多轮采样一致率（现在只取了 `overall`），与 ρ、R² 一起填成完整的三仪器表。
产出：`reports/llm_l1_typology.md`（或 json）。

**判读规则见 §1.4 的表。** 这一步不需要任何 API 调用，几分钟就能完成，
而且它直接决定"哪几个字段该留在条件向量里"。

### 第 1 步：L2 的最小三组训练（**不是 12 组**）

只跑这 3 组，机器已通、先验已就绪：

| 配置 | prior_dim | 参数量（实测） | 作用 |
|---|---|---|---|
| `configs/prior_geometry_only.yaml` | 42 | 6,605,345 | 纯几何基线（无 LLM 输出） |
| `configs/prior_v3_masked_em.yaml` | 64 | **6,608,161** | **容量对齐的零信息对照**（几何真、电磁置零） |
| `configs/prior_real_v3.yaml` | 64 | **6,608,161** | LLM 电磁先验 |

后两者参数量**逐位相同** → `real_v3 − masked_em` 才是 LLM 电磁内容的净贡献。

**要检验的预言（上个会话给出的，可证伪）**：

> 既然电磁那 22 维里大部分是噪声/幻觉，**`masked_em` 很可能与 `real_v3` 持平甚至更好**。

再加 3 个 seed（论文级结论必须 3 seed 报 mean±std）。

```bash
python -m src.train.train --config configs/prior_geometry_only.yaml --seed 0
python -m src.train.train --config configs/prior_v3_masked_em.yaml --seed 0
python -m src.train.train --config configs/prior_real_v3.yaml --seed 0
python -m src.eval.run_eval --run-dir runs/prior_real_v3_seed0 --split test --weights ema
python -m src.eval.report --runs runs --out reports/summary.md --json reports/summary.json
```

**若要更强的对照，可再加**（成本低）：
- `configs/prior_real_v3_shuffled.yaml`（内容打乱，检验"是内容还是通路"）

### 第 2 步：下游天花板（否则不能把"增量≈0"归因于先验无用）

估目标本身的不可约误差：例如用**同一张图的其它 Tx** 或**上界模型**（更多容量/更多 epoch）
估计可达的 RMSE 下界。若任务本身就只能到 4.5 dB 而三组都在 4.5–4.6，那"增量≈0"
是**任务饱和**，不是先验无用。**这一步不做，第 1 步的结论不成立。**

### 第 3 步：跨模型迁移（**决定论文力度**）

换**一个不同厂商的 LLM** 重跑同一批样本（371 条，免费模型约 15 分钟），
看 L1 的三仪器结论是否一致。

- 一致 → 结论是"关于 LLM 这一类方法"（力度大得多）
- 反转 → 结论只是"关于 glm-4v-flash"（必须在标题/摘要里限定）

代码已支持：`src/prior/generate.py` 是 OpenAI 兼容协议，`--design v3_geometry`，
`--base-url/--model/--api-key-env` 可切换（见 §4 的 PROVIDERS 与 API_KEY_ALIASES）。

### 第 4 步：写结论（三种结果都能写，不要预设方向）

| L2 结果 | 结论 | 可发表性 |
|---|---|---|
| `real_v3` 明显优于 `masked_em` | 弱接地字段（LOS/反射）确有下游可用信号 | 正面结果，再补 shuffled 与 CoT 消融 |
| `real_v3` ≈ `masked_em` | 把 LLM 的电磁数值喂进去**没有正贡献** | **负面结果**，干净 |
| `real_v3` < `masked_em` | 那些噪声/幻觉数值**有害** | **更强的负面结果** |

---

## 3. 资料在哪里（**确切路径**）

### 3.1 本项目的代码与产物

根目录：`C:\Users\Administrator\Desktop\new LLM\llm-prior-radiomap\`

| 内容 | 路径 |
|---|---|
| 项目规矩与已知的坑（**必读**） | `AGENTS.md` |
| 代码结构、自检状态、与上一版的差异 | `README.md` |
| 远程 GPU 机器逐步手册 | `RUNBOOK_AutoDL.md` |
| 数据划分（leave-maps-out，冻结） | `data/splits.json` |
| RadioMapSeer 数据（junction 到旧仓库） | `data/radiomapseer/`（50 maps / 4000 gain / 50 antenna，1 px = 1 m） |
| **prompt_v3 已生成的 371 条先验**（0 失败，240 条试生成通过门槛 + 131 条全量续传产物） | `priors/prompt_v3/glm-4v-flash/` |
| 旧先验体检报告（反面基线） | `reports/audit_legacy_v2.json` |
| 240 样本 v3 门槛报告（**7 条全 PASS**） | `reports/audit_v3_240.json` |
| 接地检验结果 | `reports/llm_grounding_240.json`、`reports/llm_grounding.json` |
| 可解释方差结果 | `reports/llm_distillability.json` |
| VLM 能力边界报告（含全部数字与判读） | `reports/vlm_perception_probe.md` |
| 试生成前的几何可行性检查结果 | `reports/pilot_preflight.json` |
| ControlRadio 全文纯文本（已核对：无 LLM） | `reports/_controlradio_text.txt`（原 PDF：`C:\Users\Administrator\Desktop\2608.09357v1.pdf`） |
| prompt_v2 送给 LLM 的合成图样例 | `reports/prompt_v2_example.png` |

### 3.2 关键脚本

| 脚本 | 作用 |
|---|---|
| `scripts/audit.py` → `src/prior/audit.py` | **L0 门槛**：7 条硬门槛 + 10 条建议级。阈值集中在一个 `GATES_FAIL`/`GATES_WARN` 里 |
| `scripts/probe_llm_grounding.py` | **L1 仪器 2**：预先指定的物理配对 + 同图错位对照 + 统计功效提示 |
| `scripts/probe_llm_distillability.py` | **L1 仪器 3**：几何特征 → LLM 输出的 5 折 CV R²（三档特征集 × Ridge/GBM + 错位对照） |
| `scripts/probe_vlm_perception.py` | VLM 感知边界（sanity / perceive / facts 三个子命令） |
| `scripts/pilot_preflight.py` | 生成前的几何可行性（偷懒基线、门槛余量、Tx 落点） |
| `src/prior/generate.py` | 先验生成（多轮采样共识、断点续传、失败进 `_failed/`）。`--design v3_geometry` 为当前设计 |
| `src/prior/prompt_v3.py` | **v3 提示词**：程序给精确几何事实，LLM 只做电磁推理；`compute_geometry_facts()` 是精确测量 |
| `src/prior/prompt_v2.py` | v2 提示词（让 VLM 读 8 方位，**实测不可行**，保留作对比） |
| `scripts/run_experiments.sh` | 批量训练 + test 评测 + 自动报告 |
| `scripts/remote_verify.sh` | 远程一键自检（数据 + 4 条自检链路） |
| `scripts/upload_to_autodl.ps1` | 打包上传（**注意：本文件必须存为 UTF-8 with BOM**，PS 5.1 否则按 GBK 解码会报一堆无关语法错） |

### 3.3 自检（**改代码后必须跑，全绿才算没弄坏**）

```bash
python -m tests.test_local              # 45 项：schema/几何/数据集/模型/SSIM 对拍/fp16 损失/v3 分组/高度不变量
python -m tests.test_pipeline           # 端到端：四配置 训练→ckpt→按 EMA 口径评测→metrics 契约
python -m tests.test_report_integration # 报告链路：train→run_eval→report→visualize 契约
python -m tests.test_gate               # 门槛区分度：7 个"只犯一种病"的先验集，逐条验证门槛因正确原因成败
```

`test_gate` 特别重要：它证明门槛**既可达又有区分度**。动门槛阈值前先跑它。

### 3.4 环境

| 项 | 值 |
|---|---|
| 本机（Windows）不能训练 | 无 CUDA、无本地 GPU。本机只做不需要 GPU 的验证 |
| 本地 venv（**有 torch CPU / sklearn / pypdf**） | `C:\Users\Administrator\Desktop\new LLM\.venv-local\Scripts\python.exe` |
| 远程 GPU 机器 | SSH 别名 **`radiomap`** → `connect.bjb2.seetacloud.com:21187`，root |
| 远程 GPU | **RTX 5090 / 32 GB**（Blackwell sm_120） |
| 远程 Python | **`/root/miniconda3/bin/python`**（不在非交互式 SSH 的 PATH 里，`python`/`python3` 都找不到） |
| 远程 torch | **2.8.0+cu128**，`arch_list` 含 `sm_120`，CUDA matmul/conv2d 已验证 |
| 远程已装 | numpy / pillow / pyyaml / matplotlib / scipy / pydantic / openai |
| 远程**尚未上传**代码与数据 | 用 `scripts/upload_to_autodl.ps1 -What both`（code 0.12 MB，data 64.84 MB） |
| **AutoDL 每次关机再开都会换 host 与 port** | 重启后需更新 `~/.ssh/config` 里的 `radiomap` 段 |

### 3.5 需要向用户索取的东西

- **`ZHIPUAI_API_KEY`**（智谱，glm-4v-flash 免费）—— **故意没有写进本文件**，
  以免密钥随代码目录被复制/上传。程序接受 `ZHIPUAI_API_KEY` / `ZHIPU_API_KEY` / `GLM_API_KEY` 任一。
- 若要做第 3 步跨模型迁移：另一个厂商的 API key（Qwen-VL / GPT-4o 等）。
  注意 `glm-4v` 系列**只有 glm-4v-flash / glm-4v / glm-4v-plus 可用**，
  `glm-4.1v-thinking-flash` 返回的内容无法解析出数值。

---

## 4. 必须避开的坑（**都是实测踩出来的，别重犯**）

### 4.1 最根本的一条

> **任何单个指标都能同时被"好机制"和"坏机制"解释。** 必须用**两个正交的测量**三角定位。

上个会话里这条被验证了 **8 次**，每次都是"某个数字的朴素判读是错的"：

| # | 朴素判读 | 实际 |
|---|---|---|
| 1 | `fields_varying_within_map ≥ 3` 通过 → 先验健康 | 旧先验 **9/13 通过**却彻底坍缩（9 个字段的变化是纯噪声） |
| 2 | `max_top1_share < 0.4` 当硬门槛 | 会把"数据本来不均衡"误判成"LLM 偷懒"，**让正确的 prompt 永远进不了训练** |
| 3 | `sector_consistency ≥ 0.70` 足够 | 实测偷懒基线就有 0.593，余量仅 0.107 → 必须加**自校准**的 `sector_consistency_gain`（本 Tx − 同图另一 Tx）；偷懒策略增益**恒为 0** |
| 4 | 数值字段必须都有多个取值 | 材料属性（ε_r、穿透损耗）在单一材质城市上**本该恒定** |
| 5 | LLM 的 `scene_type` 恒定 = 坍缩 | 它是**地图级**属性；maps 0–2 在地图级全是 urban（0.253/0.332/0.291）→ 恒定是**对的** |
| 6 | 低 R² = 有独立信息 | **噪声同样无法由几何复现**。必须与接地 ρ 联判 |
| 7 | 消融用 `use_prior=False` 就是"零先验对照" | 旧代码是**把 FiLM 分支整个拿掉**，参数量差 **282,688** → 混着容量效应 |
| 8 | 8 个样本没通过 = 门槛不可达 | 先做区分度自检（`test_gate`）才知道是**阈值有问题**还是**先验有问题** |

### 4.2 工程上的坑

- **`glm-4v-flash` 的 `max_tokens` 上限是 1024**，设大了被 API 直接拒（code 1210）。
- **`torch.__version__` 是 str 子类**，`yaml.safe_dump` 会抛 `RepresenterError` → 落盘前统一拍平（`src/utils/config.py:_to_plain`）。
- **不要在 PowerShell 5.1 里对含中文的 UTF-8 文件做 `Get-Content`/`Set-Content` 往返**：
  它按 GBK 解码，会把中文**连后面的 ASCII（包括 heredoc 的 `EOF`）**一起改坏。
  上个会话就这么毁掉过一个 `.sh`。用编辑工具，`.ps1` 存为 UTF-8 with BOM。
- **不要在 PowerShell 里用 here-string 传多行 Python**：引号会被吃掉。写成 `.py` 文件再跑。
- **远程探测用管道**：`$script | ssh radiomap "bash -s"`，比把脚本当参数传安全。
- `export.arxiv.org` 可达；`arxiv.org` / `ar5iv` / `openreview` / GitHub raw 在本机 DNS 解析失败。
- 缺先验文件时 `RadioMapSeerDataset` 会**抛错**（这是有意的，见规则 6）；不要加 `allow_missing_priors` 绕过。

### 4.3 项目红线（摘自 `AGENTS.md`，不可协商）

1. 先验生成后**必须先过 `scripts/audit.py` 硬门槛**，不过不许进训练；
2. 每次训练必须 `set_seed()`，seed 写进 `runs/<exp_id>/config.yaml`；**论文级结论必须 3 seed 报 mean±std**；
3. 划分只能用 `data/splits.json`（leave-maps-out）；**禁止用 val 选模型又用 val 报结果**；test 只评一次；
4. 主表必须报告 **`real − masked`** 与 **`real − shuffled`**，而不是只报 `real − none`；
5. 报告所有数字必须由 `src/eval/report.py` 自动生成，**禁止手抄**（上一版正文与图表对不上就是这么来的）；
6. LLM 解析/校验失败**不得静默填默认值**：记入 `_failed/` 并在 audit 里体现；
7. 不在 CPU / Mac MPS 上做正式训练。

### 4.4 上一版为什么失败（**背景，避免重演**）

不是"prompt 没调好"，而是**把"模型能看懂图"当成了已知条件**，然后在"模型确实看懂了图"的
假设上做了一整套条件注入实验 —— 而那个假设从没人验证过，实测下来也不成立。

分层原因：

1. **先验没信息**：55 个唯一向量 / 4000（0.0138）；同 map 内 0.104；`confidence` 97.9% 是 medium。
2. **为什么会没信息**（五个根因）：
   - prompt 要求模型做它做不到的事（VLM 无法对掩码做定量空间聚合，实测 4 个模型）；
   - 固定 5 个 few-shot + `temperature=0` → 输出坍缩到 Demo 1（reasoning top1 占 11.9%，475 条逐字复读 28GHz 文本）；
   - 程序把自己的答案（`scene_hint`）塞进 prompt；
   - **代码里真的有一个高度变量**：合成数据管线 `generate_synthetic_data.py` 有 `building_height`（米）与
     `tx_height`，并把高度图用 `height/height.max()*255` 转灰度图 —— 在**合成数据**上"更亮=更高"是真的；
     换成 RadioMapSeer 二值掩码后 prompt 没跟着改（`system_prompt.py:16` 声称 Tx 有 z 高度、`:146-149`
     给了高度区间表、`:172` 要求"结合常识推理"高度）→ **81.65%** 先验写出凭空的建筑高度；
     而旧 schema **从来没有** height 字段，所以校验拦不住；
   - 项目叫 CoT 但 `reasoning` 零引用。
3. **就算有信息也证明不了**：无种子；baseline 参数量差 282,688；样本级划分造成同图泄漏；
   EMA 分支永远走不到（训练记 EMA、评测用原始权重）；MAE 恒为 0；手抄数字。
4. **26 通道版本喂的是未归一化的先验**（`model/dataset.py:141` 用 `prior_to_vector()`，
   permittivity≈7、pathloss 50–155），却和 [0,1] 的场景图拼在同一输入 → 量纲差两个数量级，
   这才是 15.30/11.99 dB 那 2–3 倍落差的代码级原因。

**`sector_consistency_gain` 这类"自校准"判据是防重演的关键工具**：它对任何偷懒策略恒为 0，
不依赖任何人工阈值标定。

---

## 5. 我可以预先声明的证伪条件

如果出现下面任一条，说明**判据设计有问题**，而不是"LLM 没用"：

1. **L0 通过、L1 全灭、L2 却显著更好** → 存在 L0/L1 测不到的通道，L1 仪器不完备；
2. **`real_v3` 显著优于 `masked_em`，而 `los/reflection` 的 ρ 与 R² 都很低** → 增益来自未识别的字段；
3. **换一个 LLM 后 L1 结论整体反转** → 结论只是模型特性，不能作为"LLM 先验"的一般结论。

---

## 6. 建议的第一步

**先做 §2 第 0 步**（从现成 provenance 里提取逐字段可重复性，零 API 成本，几分钟），
把 §1.4 的三仪器表填满。它会直接告诉你"哪几个字段值得进条件向量"，
**再决定**是否投第 1 步的 3 组训练。

不要跳过这一步直接训 12 组 —— 你已经知道输入侧图像通道已死、输出侧 8 个数值字段里只有 2 个弱接地。
