# 项目：LLM 物理先验引导的 Radio Map 生成

> 完整背景见 `../新项目交接_LLM-Prior-RadioMap.md` 与 `../CoT-RadioMap_实验分析报告.md`。
> 本文件是 AI 协作者与人都必须遵守的项目约定。

## 目标

让多模态 LLM（当前：智谱 `glm-4v-flash`）输出对场景的**电磁推理**（结构化 prior），
作为条件注入专用生成模型（U-Net + FiLM），提升 pathloss map 生成的精度与跨场景泛化。
LLM 只用 prompt engineering（方案 A），**不训练 LLM**。

## 不可协商的规则

1. **先验生成后必须先跑 `python scripts/audit.py`，未通过硬门槛不得进入训练。**
   理由：上一版 4000 条先验只有 55 个唯一向量（unique_ratio 0.0138），实验因此无效。
2. **每个训练入口的第一个调用必须是 `set_seed()`**，seed 写进 `runs/<exp_id>/config.yaml`。
   结论必须来自 **3 个 seed**，报 mean ± std。
3. **数据划分只能用 `data/splits.json`（leave-maps-out）**，禁止按样本随机划分。
   **禁止用 val 选模型后又用同一个 val 报结果**；`test_maps` 只在最后评一次。
4. 主表必须包含 `prior_none / prior_masked / prior_real / prior_shuffled` 四种配置，
   并报告 **`real − masked`** 与 **`real − shuffled`** 两个差值，而不是只报 `real − none`。
5. 报告里的所有数字必须由 `src/eval/report.py` 从 `runs/*/metrics_*.json` **自动生成**，
   **禁止手抄**。图表数据与正文必须同源。
6. LLM 输出解析/校验失败**不得静默填默认值**：记入 `priors/<...>/_failed/` 并在 audit 中体现。
7. **不在 CPU / Mac MPS 上做正式训练**；正式结果一律来自 Linux + NVIDIA GPU。
   本地只做"不需要 GPU 的验证"（`tests/test_local.py`、`tests/test_pipeline.py`）。

## 目录与命名

- 一次实验 = 一个 `runs/<exp_id>/`，内含 `config.yaml` / `config.json` / `train.log` /
  `ckpt/best.pth` / `history.json` / `metrics_<split>.json` / `eval.log`
- prior 按 `priors/<prompt_version>/<llm_name>/` 存放，**禁止原地覆盖**
- 所有路径写在 config 里，禁止硬编码绝对路径
- 大文件（数据、ckpt）放数据盘，不进 git
- `exp_id` 默认 = `<config名>_seed<seed>`

## 环境

Linux + CUDA，数据放 `/root/autodl-tmp`。
数据下载后先跑 `python scripts/verify_data.py --data <path>`，不通过不许开工。
详细步骤见 `RUNBOOK_AutoDL.md`。

## 已知的坑（上一版踩过，别再踩）

| 坑 | 上一版的症状 | 本项目的规避 |
|---|---|---|
| 固定 few-shot + `temperature=0` | 输出坍缩成模板，4000 条只有 55 个唯一向量 | 去掉固定示例；`temperature=0.7` 采样 3 次取共识（`src/prior/generate.py`） |
| 把程序算好的答案塞进 prompt | `scene_type` 变成复读，LLM 没真的看图 | prompt 里**没有** `scene_hint`；程序统计只在生成后用于核对（sector 一致率） |
| 只给全局建筑掩码 | LLM 无法推断"这个 Tx 周围"，先验退化成 per-map 常量 | 额外给 Tx 中心 100 m 局部放大图 + 8 条方位线（`src/prior/prompt_v2.py`） |
| 说"brighter pixels 表示更高建筑" | 475 条先验写"28GHz、Tx 高度 25m"这一事实错误 | prompt 里明确"掩码是二值的、无高度信息"，并把幻觉做成自动回归检查 |
| CoT 文本不进模型 | 项目叫 CoT，但 `reasoning` 全项目零引用 | `src/models/cot_encoder.py`：要么真正注入（`prior_real_cot`），要么在文档里明确声明不注入 |
| 不设随机种子 | 0.25 dB 的"提升"无法排除噪声 | 强制 `set_seed()` + 3 seed |
| baseline 与主模型参数量不同 | FiLM 多 28.3 万参数，无法排除容量效应 | `prior_masked`：结构逐位相同、先验置零（`real − masked` 才可归因） |
| 样本级划分 | 同图不同 Tx 泄漏，5.11 dB 不代表泛化 | leave-maps-out，35/5/10 maps，冻结进 `data/splits.json` |
| 评测脚本 EMA 分支永远走不到 | 训练记 EMA 成绩、评测用原始权重 | `--weights {ema,raw}` 必须显式；缺 EMA **报错**而不是静默回退 |
| FiLM 无 identity 初始化 | 训练前期 FiLM 明显劣于 baseline（epoch 1：118 vs 76 dB） | γ 的 bias 初值 1、β 初值 0；有新自检 `check_identity()` 验证 |
| 手抄报告数字 | 正文说"第 25 epoch 领先"，图上第 21 epoch 交叉 | 报告由 `src/eval/report.py` 自动生成 |
| 噪声地板把低增益钉在 200 dB | RMSE 被这批像素主导 | 主指标之外额外报 `mean_rmse_db_nonsaturated` |
| 未归一化的 prior 拼进输入通道 | permittivity≈7、pathloss≈155 与 [0,1] 的场景图混在一起，15.30/11.99 dB 差 2–3 倍 | 只用归一化向量（`src/prior/schema.py:prior_to_vector`），每个维度按自己的量程缩放 |
| 程序造的假先验写回先验目录 | 让"先验质量"事后无法审计 | 缺先验文件直接抛错；合成数据只允许存在于临时目录 |
| 报告选择性呈现 | 只报 12 个偏难样本的 75%，隐去全量 26% 样本变差 | `report.py` 必须输出失败样本与"变差样本占比" |
| 新增模块却不扣掉它的参数量 | 无法区分"内容有用"与"多了容量" | 每个"加东西"的实验都要有配对的打乱对照（见下） |

## 配对对照清单（任何"新模块带来提升"的说法都要先过这一关）

| 对照 | 结构关系 | 差值含义 |
|---|---|---|
| `prior_masked` ↔ `prior_real` | 逐位相同 | 先验**内容**的贡献 |
| `prior_real_cot_shuffled` ↔ `prior_real_cot` | 逐位相同 | CoT**内容**的贡献 |
| `prior_real` ↔ `prior_real_cot` | 多一个 CoT 编码器（+286,880 参数） | CoT 内容 **+** 额外容量 |

只报 `prior_real_cot − prior_real` 会把 28.7 万参数的容量效应误算成 CoT 的功劳。
`prior_real_cot` 与 `prior_real_cot_shuffled` 的参数量必须**逐位相同**（实测都是 6,895,041），
这条由 `tests/test_local.py:configs.all_load_and_build` 与端到端自检共同保证。

同理，`prior.shuffle_cot` **只允许打乱 CoT 文本，绝不能顺带打乱先验向量**
（否则 `real + shuffle_cot` 会悄悄变成 `shuffled` 实验）。这条有专门的自检：
`tests/test_local.py:dataset.shuffle_cot_isolation`。

## 关于 audit 门槛的三条实测教训（都是本项目踩出来的）

**教训 1：`fields_varying_within_map` 不足以判定健康。**
它统计"同一地图内会随 Tx 变化的字段数"。用上一版 4000 条先验实跑是 **9/13 通过**，
而同一批先验是彻底坍缩的（唯一向量 55、unique_ratio 0.0138、同 map 内唯一占比 0.104）。
原因是那 9 个字段的变化是**纯噪声**（同一张掩码、同一频率下 `permittivity` 从 4.0 跳到 6.0）。
所以它只作为**建议级**。

**教训 2：`max_top1_share` 不能当硬门槛，因为它把"LLM 偷懒"和"数据本来就不均衡"混为一谈。**
初版把它设为硬门槛（任一单维 top1 < 0.40）。`tests/test_gate.py` 立刻证明这个设计有问题：
一份**健康的**先验集（sectors 与真实几何一致率 0.925、unique_ratio 1.000）被判 FAIL。
更要命的是真实场景 —— 50 张城市地图真实就是 60% urban / 40% dense_urban 时，
`scene_type` 的 top1 = 0.60 > 0.40，一份完全正确的 prompt 会被**永远挡在训练门外**。
所以硬门槛改成按字段类型分工，`max_top1_share` 降级为建议级。

**现在的硬门槛（`src/prior/audit.py:GATES_FAIL`，阈值就在那一处）**：

| 硬门槛 | 阈值 | 判定什么 | 上一版实测 |
|---|---|---|---|
| `unique_ratio` | ≥ 0.50 | 向量整体是否有逐样本信息 | 0.0138 **FAIL** |
| `intra_map_unique_ratio_mean` | ≥ 0.50 | 同一地图内不同 Tx 是否真的不同 | 0.1043 **FAIL** |
| `numeric_unique_min` | ≥ 5 | 每个**数值**字段是否真的在变（恒定=坍缩） | 3（`pathloss_min_db`）**FAIL** |
| `numeric_entropy_min` | ≥ 1.0 bits | 每个数值字段的熵 | 0.65（`penetration_loss_db`）**FAIL** |
| `categorical_min_unique` | ≥ 2 | 每个**类别**字段是否退化成单一常量 | 2 **pass** |
| `sector_consistency_strict` | ≥ 0.70 | LLM 是否真的按方位线看图 | 无该字段 |
| `sector_consistency_gain` | ≥ 0.10 | **自校准**：本 Tx 一致率 − 同图另一 Tx 一致率 | 无该字段 |

> 数值字段与类别字段的标准不同是有意的：数值恒定只有一个解释（坍缩）；
> 类别字段取值不均衡可能只是数据的性质，**不是**错误。

**为什么 `sector_consistency` 要有两条门槛（第三条实测教训）。**
`scripts/pilot_preflight.py` 实测 maps 0–3 的 2560 个扇区判定：

```
blocked 59.3%｜dense 20.0%｜sparse 20.0%｜open 仅 0.7%
「永远回答 blocked」的一致率 = 0.593      ← 离门槛 0.70 只差 0.107
Tx 直接落在建筑像素内 = 97/320 = 30.3%
8 方位里最近建筑 ≤10 m 的 Tx = 87.8%
```

也就是说绝对一致率的偷懒下限**取决于几何真值的标签分布**（而那个分布又取决于我拍的
代理阈值 10 m / 30%），换一批建筑更密的地图，偷懒基线就会越过 0.70。
所以加了自校准判据：把 LLM 的 sector 与**本 Tx** 几何比、再与**同一地图另一个 Tx**
几何比。实测区分度（`tests/test_gate.py`）：

| 集合 | 本 Tx | 错位对照 | 增益 |
|---|---|---|---|
| 健康（sectors 取自真实几何） | 0.925 | 0.488 | **+0.438** |
| 恒答 blocked（偷懒） | 0.655 | 0.655 | **+0.000** |
| sectors 随机 | 0.257 | 0.227 | +0.030 |
| map 内恒定 | 0.538 | 0.538 | +0.000 |

**任何偷懒策略的增益都精确为 0**，所以即使绝对门槛在某批数据上失效，这条仍然拦得住。
跑试生成前先跑 `python scripts/pilot_preflight.py --maps 0:4`，它会直接打印
本批地图的偷懒下限、余量与可用区分空间。

**这套阈值是在两批数据上标定的，不要凭感觉改**：
(a) 上一版真实的 4000 条先验 —— 硬门槛必须拦住它；
(b) `tests/test_gate.py` 构造的健康集 —— 硬门槛必须放它过。
该测试还构造了 6 种"只犯一种病"的先验集（map 内恒定 / 类别字段恒定 /
sectors 与几何随机 / 单个数值字段恒定 / **恒答 blocked 的偷懒策略**），
逐条确认**每条门槛都因为正确的原因失败**。
跑 `python -m tests.test_gate`，全部通过说明阈值既可达又有区分度。

## 当前实测基线（旧先验，反面参照）

```
样本 4000｜唯一向量 55｜unique_ratio 0.0138｜同 map 内唯一占比 0.1043
数值字段：唯一值最少 3（门槛 >= 5）｜熵最低 0.65（门槛 >= 1.0）
  唯一值不足的字段：['pathloss_min_db']
  熵过低的字段：['penetration_loss_db', 'pathloss_min_db']
类别字段：取值数最少 2（门槛 >= 2）
内容污点：毫米波幻觉 475/4000 = 0.1187｜高度幻觉 3266/4000 = 0.8165
[FAIL][硬] unique_ratio = 0.0138
[FAIL][硬] intra_map_unique_ratio_mean = 0.1043
[FAIL][硬] numeric_unique_min = 3
[FAIL][硬] numeric_entropy_min = 0.65
```

**新生成的先验必须全面优于这份基线**（完整报告见 `reports/audit_legacy_v2.json`）。

另外一条实测出来的事实：`height` 类幻觉（在二值掩码上"看出"建筑高度）比交接文档
记录的问题**更广** —— 文档只记了 475 条 28GHz 模板复读（12%），
而"建筑高度 10-30m"这类无根据的描述占 **81.65%**。新 prompt 必须把这条压下去。

## 命令速查

```bash
# 0) 数据与划分
python scripts/verify_data.py --data data/radiomapseer
python scripts/make_splits.py --data data/radiomapseer --out data/splits.json

# 1) 先验：试生成 → 体检（硬门槛）→ 全量 → 再体检
export ZHIPUAI_API_KEY=...
python -m src.prior.generate --priors priors/prompt_v2/glm-4v-flash \
    --map-range 0:4 --workers 6
python scripts/audit.py --priors priors/prompt_v2/glm-4v-flash \
    --data data/radiomapseer --out reports/audit_pilot.json
# 退出码 1 = 没过门槛，改 prompt 重跑，不要进训练

# 2) 训练（单配置单种子）
python -m src.train.train --config configs/prior_real.yaml --seed 0

# 3) 评测（test maps 只评一次，显式权重口径）
python -m src.eval.run_eval --run-dir runs/prior_real_seed0 --split test --weights ema

# 4) 自动报告
python -m src.eval.report --runs runs --out reports/summary.md

# 本地自检（不需要 GPU / API key）
python -m tests.test_local              # 41 项：schema/几何/数据集/模型/SSIM 对拍/损失/配置/高度不变量
python -m tests.test_pipeline           # 端到端：四配置 训练→ckpt→评测→metrics 契约
python -m tests.test_report_integration # 报告链路：train→run_eval→report→visualize 契约
python -m tests.test_gate               # 门槛区分度：健康集必过、每种病被对应门槛抓住
python -m tests.test_typology           # L1 类型学：几何解析/方位对齐/判据分支/小样本保护/对拍机制

# L1 类型学（零 API / 零 GPU，只读 provenance）
python -m scripts.probe_l1_typology --priors priors/prompt_v3/glm-4v-flash
```

## 关于 L1（"先验有没有信息"）的六条实测教训

门槛（audit）只回答 **L0 非退化**。要回答 **L1 接地**，必须用多个正交测量三角定位，
否则一定会误判。以下都是本项目实测踩出来的：

**教训 1：低 R² 与"有信息"和"纯噪声"都相容。**
几何拟合不出某个字段（R² 低），既可能是 LLM 提供了几何之外的信息，也可能纯粹是
在胡编 —— 噪声当然也拟合不出来。必须与接地 ρ 联判。

**教训 2：一致率高**不**等于确定，因为容差是 ±10% 量程。**
`field_consistency` 是"各轮取值落在样本均值 ±10% 量程内"的比例。对
`pathloss_max_db`（量程 190 dB）那个容差是 ±19 dB —— 一致率 1.0 完全可以掩盖
±19 dB 的抖动。所以一致性必须与"散度 ÷ 量程"一起看：散度/量程 ≤ 0.05 才算实质恒定。

**教训 3：总体 ρ 可能全部来自"地图之间"的差异，必须在地图层内复查。**
把样本按 map 分层、在地图内部重算 ρ 与逐图唯一值数：若总体 ρ 显著而地图内 ρ
反复变号，那段总体相关就来自地图之间的几何分布差异，该字段只是**地图级**信息，
对"同图不同 Tx"没有分辨力 —— 这正是上一版"地图级常量"的失败模式。
实测有用例：一个字段总体 ρ 显著、但逐图 ρ 只有 2/5 张图与物理预期同号。
**报 L1 结论时必须同时报逐图同号数，否则会把地图间差异当成字段的信息。**

**教训 4：接地强 ≠ 增量强；真正的对照是"它所编码的那个几何量"。**
若某字段与条件向量里**已有的**几何量（`min_dist` / `sectors`）高度单调相关，
那么"LLM 字段 vs 零向量"的差值会把"几何量的重新编码"误算成 LLM 的贡献。
所以除了 `masked_em`（零向量、参数量逐位对齐），还需要一个
「只保留接地字段」的配置作为**新配置**报出来 —— 不许静默删字段后宣称先验有效。

**教训 5：样本量太小时"方差"本身不可靠，结论会随 n 翻转。**
实测同一个 `los_ratio`：在 24 条子集上归一化散度低于 0.05 被判**常数**，
在 371 条全量上远高于 0.05 被判**真实且有残余信息**。差别只来自样本量。
所以样本数低于 `LOW_N` 时，报告与每一行字段都必须显式标注"不可作为定论"。

**教训 6：常量化 ≠ 坏，有散度也 ≠ 有信息。**
材料属性（`permittivity`、`penetration_loss_db`）在单一材质城市上**本该**恒定：
恒定不是错，只是下游不能指望它带信息。反过来，某字段散度不小、但与几何的关系
方向**反了**（如"被挡方位越多、最大路损反而越小"）—— 这种"自信地错"比恒定更糟，
因为下游还得额外学会忽略它。所以类型学里"常数"与"幻觉"必须分开标注。

## 关于 AMP 的一条硬教训（直接决定训练有没有在学）

**`train.amp` 必须保持 `false`。** 混合精度下 SSIM 损失（`ssim_torch` 里多次卷积与
除法）的梯度会变成 NaN/Inf，`GradScaler` 检测到后**静默跳过整步更新**，模型永远停在
随机初始化，而且不报任何错——日志里的 `val RMSE` 钉在 212.9 dB 附近、`train total`
几乎不动。这是本会话实测出来的：

| 设置 | 3 epoch 后 val RMSE | SSIM | 判读 |
|---|---|---|---|
| amp: true  | 212.899 dB | −0.026 | 模型没在学（随机输出 ≈ 212.9 dB） |
| amp: false | 11.796 dB  | 0.870 | 正常学习 |

`losses.py` 里的注释早就警告过这个风险，但没在 config 层挡住。现在
`configs/base.yaml` 已把 `amp` 改为 `false` 并注释原因。**谁都不许改回 true**；
如果一定要用 AMP，必须先给 SSIM 项单独用 fp32 前向（或加 NaN 梯度守卫），并跑
"amp 开/关 val RMSE 必须一致"这条自检确认梯度没被吃掉。

> 结论数字一律由 `scripts/probe_l1_typology.py` 自动写进
> `reports/llm_l1_typology.md`，**不要在文档里手抄**（规则 5）。
> `--limit-maps 0:3 --validate-against reports/llm_grounding_240.json` 可对拍历史结果
> （`a:b` 与 `--map-range` 一样是左闭右开，`0:3` = maps 0/1/2 = 240 个样本）。

## 背景文献（备查）

- **LLM4PG**, arXiv:2511.02423 — 反面案例：把 LLM 当 sequence backbone 做数值回归
- **WiCo-PG**, arXiv:2511.15030 — 同组否定 LLM4PG 的后续工作，应作为 baseline
- **RadioSim Agent**, arXiv:2511.05912 — LLM 作 orchestrator，demo paper，已被占位
- **Quan et al.**, IEEE Networking Letters 2025 — LLM agent 做 radio map 规划
- **RadioMapSeer / RadioUNet** — 数据集与基线
- **3GPP TR 38.901**；**ITU-R P.2109** — 先验 prompt 里的领域知识来源
