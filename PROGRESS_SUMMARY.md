# 项目进度总结（截至 2026-10-17）

> 本文档是人工写的进度快照。数字一律以自动生成产物为准：
> `reports/llm_l1_typology.md`、`reports/summary.md`、`runs/*/metrics_test.json`。

---

## 0. 一句话

项目目标是回答"LLM 产出的电磁先验到底有没有信息"，而不是"再挤 +0.X dB"。
我们已经把这个目标拆成一套可证伪的判据（L0/L1/L2 + 三仪器），并跑通了
**第一次完整的、有有效对照的**下游实验。结果是一个干净的负面结果，但还差
seed / 模型 / 天花板 / 阳性对照才算完整。

---

## 1. 做了什么（按时间线）

### 1.1 方法层：L1 三仪器类型学（零 API / 零 GPU）

- 新增 `scripts/probe_l1_typology.py`，产出 `reports/llm_l1_typology.{md,json}`。
- 三个正交仪器：**可重复性**（多轮采样一致率，来自 provenance）、
  **接地**（字段↔几何的 Spearman ρ + 错位对照 + 置换检验）、
  **可复现性**（几何特征 → 字段的 5 折 CV R²，Ridge/GBM）。
- 加了两条自校准诊断：**地图层内 ρ**（区分"随 Tx 变" vs "地图级常量"）、
  **量程归一化散度**（揭穿"一致率 1.0 ≠ 确定"，因为容差是 ±10% 量程）。
- 自检 `tests/test_typology.py`：几何解析、方位对齐、六个判据分支、小样本保护、
  对拍机制必须能失败。本地实跑全绿。

**结论**（371 条 prompt_v3 先验，全部数字见自动报告）：

| 类型 | 字段 |
|---|---|
| ✅ 真实且有残余信息（唯一支持继续） | `los_ratio`（ρ=+0.73，5/5 地图内同号；但仪器 3b 显示它 ≈ `min_dist` 的重新编码，R² 样本内 0.76） |
| 🌀 接地不稳（地图间混淆） | `reflection_ratio`（总体 ρ 显著但逐图只有 2/5 同号） |
| ⚠️ 自信地错 / 幻觉 | `diffraction_ratio`、`scattering_ratio`、`pathloss_max_db` |
| 常数（无区分度） | `permittivity`、`penetration_loss_db`、`pathloss_min_db`、`scene_type` |
| 噪声 | `dominant_material` |

### 1.2 数据层：先验全量生成 + 门槛

- 用 `ZHIPUAI_API_KEY` 在远程 GPU 机器生成 **4000 条** prompt_v3 先验（0 失败）。
- audit 硬门槛 **7 条全过**（unique_ratio 0.9998 vs 上一版 0.0138；intra_map 0.9997
  vs 上一版 0.1043 等）。期间把 `scene_type_follows_geometry` 这条门槛从
  "LLM 必须给出 ≥2 类"修正为三分支：分档处**没有天然空隙**（整图建筑占比是 0.242→0.435
  的平滑连续分布）时不拿它否决 LLM。这是一个有据可依的判据修正，不是放宽到能过。

### 1.3 训练层：抓出三个会让结果作假的 bug

| # | bug | 症状 | 修复 | 防线 |
|---|---|---|---|---|
| 1 | **AMP + SSIM 梯度 NaN** | `GradScaler` 静默跳过所有更新，模型停在随机初始化（val RMSE 钉在 212.9 dB） | `train.amp=false`（fp32） | 记录进 AGENTS.md「AMP 硬教训」 |
| 2 | **FiLM identity 初始化把末层权重置零** | 先验从不影响前向，且梯度恒 0 → `real_v3` 与 `masked_em` 数学恒等（84 个张量逐位相同） | 末层权重改 1e-4 量级小随机值（bias 保持 γ=1/β=0） | `test_local: film_prior_actually_conditions` + `test_typology: prior.actually_conditions_model` |
| 3 | **warmup 太长被误以为"没学"** | 诊断时发现 lr 太低，真实原因是上面的 AMP bug | 保持原调度 | 无（是误报） |

> 修复顺序：先 AMP（这是"模型不学"的真凶），再 FiLM（这是"两臂恒等"的真凶）。
> 两个都是"不报错、指标照出、结论全错"的静默 bug，最危险的一类。

### 1.4 下游实验：核心对照（seed 0）

跑通了 `real_v3`（真 LLM 电磁）vs `masked_em`（几何真、电磁置零），两者**参数量逐位
相同（6,608,161）**，唯一差别是 LLM 的 22 维电磁是真值还是零。

**test 口径结果（EMA 权重，800 样本，自动生成 `reports/summary.md`）：**

| 配置 | RMSE (dB) | MAE (dB) | SSIM | RMSE 非饱和 (dB) |
|---|---|---|---|---|
| `prior_real_v3` | 8.312 | 4.769 | 0.912 | 11.728 |
| `prior_v3_masked_em` | 8.312 | 4.769 | 0.912 | 11.714 |

**`real_v3 − masked_em` ≈ 0.000 dB** → LLM 电磁数值对下游**没有正贡献**。
同时验证了修复有效：两臂 FiLM 权重**真的分岔**（max|Δ|=0.16），证明先验确实进了模型，
差值 0 是"内容没用"而不是"先验没进去"。

---

## 2. 结论：目前有把握的是什么

1. **负面结果成立（在限定范围内）**：在 glm-4v-flash × prompt_v3 × RadioMapSeer 上，
   LLM 的电磁先验喂进下游无正贡献。这个结论是**可复现、有正确对照**的。
2. **机制上说得通**：22 维里只有 1 个弱接地字段，其余是噪声/幻觉/常数，所以
   `real` 换成 `masked` 当然不变。
3. **方法有价值**：三仪器判据是可迁移的工具，这是比"负面结果"更值得写的东西。

---

## 3. 还没做完的（缺什么）

| 缺口 | 为什么重要 | 估计成本 |
|---|---|---|
| 只跑了 seed 0 | 0.000 dB 没有方差估计 | 补 seed 1/2，~3.5 h，可并行 |
| 只跑了 `real_v3` / `masked_em` 两组 | 缺 `none` / `geometry_only` / `shuffled` 三档，无法排容量效应与通路效应 | ~1 h |
| 单模型 | 结论可能只是 glm-4v-flash 的特性 | 换厂商重跑 L1 |
| 无天花板 | 不能排除"任务本身饱和在 8.3 dB" | 更大容量 / 更多 epoch / 同图其它 Tx |
| 无阳性对照 | 若判据在"已知有信号"的先验上也测不出增量，判据本身没区分度 | 用程序构造的真先验验证 |

**判断**：把这些补齐之前，这还是一个"方法论 + 一次干净应用"的骨架。
最有价值的下一步是**阳性对照 + 天花板**——它们决定这套判据到底成不成立，比继续堆 seed
更能说明问题。

---

## 4. 环境与可复现状态

- 远程机器：AutoDL，最新地址 `ssh -p 20466 root@connect.bjb1.seetacloud.com`（RTX 4090 / 49 GB）。
- 代码、4000 条先验、splits、已跑的 runs 全部在 `/root/autodl-tmp/work/llm-prior-radiomap/`。
- 关键修复（AMP=false、FiLM 初始化）已写回本机仓库 `configs/base.yaml`、`src/models/film.py`、`AGENTS.md`。
- 恢复命令：`SEEDS="0 1 2" DETACH=1 bash scripts/run_v3_ladder.sh`（自动跳过已完成）。
