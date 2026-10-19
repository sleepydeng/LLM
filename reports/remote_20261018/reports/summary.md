# 实验结果汇总（自动生成）

> **本文件由 `src/eval/report.py` 自动生成，禁止手改。**
> 生成时间：2026-10-18T15:18:38+08:00
> 数据来源目录：`runs`（共 17 个 metrics 文件，接受 `metrics.json` 与 `metrics_<split>.json`）
> 主对照基线：`prior_v3_masked_em`

所有数字（含 mean±std、配对检验、最差样本）均由脚本从 `runs/*/metrics_*.json` 计算，未记录 / 未跑的项一律显式标注，**没有任何手工填写或默认补齐的数字**。

## 1. 主表（按 mean RMSE 升序）

每个 `config_name` 一行；多个 seed 时对该指标先在 seed 维取 mean±std。
`± —` 表示只有一个 seed，方差无法估计（**不是** ±0.00）。

| config_name | n_seeds | RMSE (dB) mean±std | MAE (dB) mean±std | SSIM mean±std | RMSE non-sat (dB) mean±std | 参数量 |
|---|---|---|---|---|---|---|
| `prior_none_big` | 1 | 7.81 ± — | 4.49 ± — | 0.92 ± — | 10.99 ± — | 25,264,065 |
| `prior_real_v3_oracle` | 1 | 8.10 ± — | 4.62 ± — | 0.91 ± — | 11.33 ± — | 6,608,161 |
| `prior_geometry_only` | 3 | 8.38 ± 0.08 | 4.81 ± 0.05 | 0.91 ± 0.00 | 11.81 ± 0.12 | 6,605,345 |
| `prior_real_v3_shuffled` | 3 | 8.38 ± 0.08 | 4.89 ± 0.08 | 0.91 ± 0.00 | 11.79 ± 0.15 | 6,608,161 |
| `prior_v3_masked_em` | 3 | 8.40 ± 0.08 | 4.86 ± 0.10 | 0.91 ± 0.00 | 11.85 ± 0.13 | 6,608,161 |
| `prior_real_v3` | 3 | 8.43 ± 0.12 | 4.89 ± 0.12 | 0.91 ± 0.00 | 11.87 ± 0.17 | 6,608,161 |
| `prior_none` | 3 | 8.64 ± 0.08 | 5.10 ± 0.08 | 0.91 ± 0.00 | 12.21 ± 0.14 | 6,320,097 |

## 2. 消融阶梯（None / Masked / Real / Shuffled）

| 配置 | n_seeds | RMSE (dB) | MAE (dB) | SSIM | RMSE non-sat (dB) | 参数量 | 状态 |
|---|---|---|---|---|---|---|---|
| `prior_none` | 3 | 8.64 ± 0.08 | 5.10 ± 0.08 | 0.91 ± 0.00 | 12.21 ± 0.14 | 6,320,097 | 已跑 |
| `prior_geometry_only` | 3 | 8.38 ± 0.08 | 4.81 ± 0.05 | 0.91 ± 0.00 | 11.81 ± 0.12 | 6,605,345 | 已跑 |
| `prior_v3_masked_em` | 3 | 8.40 ± 0.08 | 4.86 ± 0.10 | 0.91 ± 0.00 | 11.85 ± 0.13 | 6,608,161 | 已跑 |
| `prior_real_v3` | 3 | 8.43 ± 0.12 | 4.89 ± 0.12 | 0.91 ± 0.00 | 11.87 ± 0.17 | 6,608,161 | 已跑 |
| `prior_real_v3_shuffled` | 3 | 8.38 ± 0.08 | 4.89 ± 0.08 | 0.91 ± 0.00 | 11.79 ± 0.15 | 6,608,161 | 已跑 |

* **`prior_real_v3 − prior_v3_masked_em` = 0.029 ± 0.044 dB**（seed 级配对，n_seeds=3，seeds=[0, 1, 2]；负值 = real 的 RMSE 更小 = 更好）
* **`prior_real_v3 − prior_real_v3_shuffled` = 0.045 ± 0.120 dB**（seed 级配对，n_seeds=3，seeds=[0, 1, 2]；负值 = real 的 RMSE 更小 = 更好）
* **`prior_geometry_only − prior_none` = -0.267 ± 0.026 dB**（seed 级配对，n_seeds=3，seeds=[0, 1, 2]；负值 = real 的 RMSE 更小 = 更好）

> 另有不在固定四档内的配置（如 CoT 变体），未纳入本表但已在主表第 1 节列出：`prior_none_big`, `prior_real_v3_oracle`

## 3. 配对显著性（同一批 test 样本逐样本配对）

口径：同一 config 的多个 seed 先按 sample_id 取算术平均，再与被比较 config 取 sample_id 交集做配对比较。非参数 Wilcoxon 符号秩为主判据（bootstrap 2000 次、种子 20260101，CI 固定可复现），配对 t 检验同时给出供交叉核对。**不使用独立样本 Cohen's d**：同一批样本的难度相关性会被非配对检验浪费掉。

| 配对（a − b） | n | mean diff (dB) | 95% CI (bootstrap) | Cohen's dz | Wilcoxon p | 配对 t p | a优/b优/平 | p<0.05 |
|---|---|---|---|---|---|---|---|---|
| `prior_real_v3 − prior_v3_masked_em` | 800 | 0.029 | [0.004, 0.053] | 0.084 | 0.015 | 0.0184 | 379/421/0 | **显著** |
| `prior_real_v3 − prior_real_v3_shuffled` | 800 | 0.045 | [0.021, 0.071] | 0.124 | 0.000165 | 0.00047 | 352/448/0 | **显著** |
| `prior_real_v3 − prior_none` | 800 | -0.215 | [-0.247, -0.183] | -0.461 | 3.18e-32 | 2.16e-35 | 540/260/0 | **显著** |
| `prior_real_v3_oracle − prior_v3_masked_em` | 800 | -0.302 | [-0.368, -0.234] | -0.310 | 5.88e-18 | 1.17e-17 | 511/289/0 | **显著** |
| `prior_none_big − prior_none` | 800 | -0.833 | [-0.878, -0.782] | -1.206 | 1.8e-112 | 4.05e-158 | 718/82/0 | **显著** |

> `prior_real_v3_vs_prior_v3_masked_em` 对齐：a 侧 800 个 sample_id、b 侧 800 个，共同 800 个（仅 a 0 / 仅 b 0）；指标口径 `rmse_db`，多 seed 先取 seed 平均。
> `prior_real_v3_vs_prior_real_v3_shuffled` 对齐：a 侧 800 个 sample_id、b 侧 800 个，共同 800 个（仅 a 0 / 仅 b 0）；指标口径 `rmse_db`，多 seed 先取 seed 平均。
> `prior_real_v3_vs_prior_none` 对齐：a 侧 800 个 sample_id、b 侧 800 个，共同 800 个（仅 a 0 / 仅 b 0）；指标口径 `rmse_db`，多 seed 先取 seed 平均。
> `prior_real_v3_oracle_vs_prior_v3_masked_em` 对齐：a 侧 800 个 sample_id、b 侧 800 个，共同 800 个（仅 a 0 / 仅 b 0）；指标口径 `rmse_db`，多 seed 先取 seed 平均。
> `prior_none_big_vs_prior_none` 对齐：a 侧 800 个 sample_id、b 侧 800 个，共同 800 个（仅 a 0 / 仅 b 0）；指标口径 `rmse_db`，多 seed 先取 seed 平均。

### 参数量对齐检查（差值能否归因于「内容」的前提）

参数量不同就说明差值里混了容量效应，不能把提升单独算成先验/CoT 内容的功劳。

| 配对（a − b） | 参数量(a) | 参数量(b) | 是否逐位对齐 | 含义 |
|---|---|---|---|---|
| `prior_real_v3 − prior_v3_masked_em` | 6,608,161 | 6,608,161 | **是** | 两侧结构逐位相同 → 差值可归因于内容 |
| `prior_real_v3 − prior_real_v3_shuffled` | 6,608,161 | 6,608,161 | **是** | 两侧结构逐位相同 → 差值可归因于内容 |
| `prior_real_v3 − prior_none` | 6,608,161 | 6,320,097 | **否** | 两侧参数量不同或无记录 → 差值混入了容量效应，不能单独归因于内容 |
| `prior_real_v3_oracle − prior_v3_masked_em` | 6,608,161 | 6,608,161 | **是** | 两侧结构逐位相同 → 差值可归因于内容 |
| `prior_none_big − prior_none` | 25,264,065 | 6,320,097 | **否** | 两侧参数量不同或无记录 → 差值混入了容量效应，不能单独归因于内容 |

## 4. 失败样本与最坏情况（防止选择性呈现）

口径：`delta = prior_real_v3 − prior_v3_masked_em` 的**逐样本** RMSE 差值，覆盖全部 800 个配对样本（无任何子集筛选）。RMSE 越小越好，故 `delta > 0` 表示 prior_real_v3 在该样本上**变差**。

* real **变差**的样本：**421 / 800（52.6%）**
* real 改善的样本：379 / 800（47.4%）
* 持平样本：0

逐 seed 口径（不做 seed 平均，防止平均把变差的样本数量「抹平」）：

| seed | n | real 变差样本数 | 占比 |
|---|---|---|---|
| 0 | 800 | 381 | 47.6% |
| 1 | 800 | 404 | 50.5% |
| 2 | 800 | 438 | 54.8% |

最差 5 个样本（real 相对基线变差最多）：

| sample_id | real RMSE (dB) | prior_v3_masked_em RMSE (dB) | delta (dB) |
|---|---|---|---|
| `1_21` | 12.986 | 11.548 | **1.439** |
| `39_48` | 9.315 | 8.127 | **1.188** |
| `47_47` | 8.998 | 7.873 | **1.126** |
| `47_61` | 8.444 | 7.498 | **0.946** |
| `47_40` | 11.094 | 10.153 | **0.941** |

最好 5 个样本（real 相对基线改善最多）：

| sample_id | real RMSE (dB) | prior_v3_masked_em RMSE (dB) | delta (dB) |
|---|---|---|---|
| `1_35` | 8.484 | 9.478 | -0.994 |
| `47_71` | 10.232 | 11.198 | -0.966 |
| `41_70` | 10.378 | 11.331 | -0.953 |
| `11_74` | 8.819 | 9.767 | -0.948 |
| `21_52` | 6.460 | 7.379 | -0.919 |

> 上一版的教训：只报 12 个偏难样本的 75% 提升、隐去全量变差比例。本节把**变差比例（52.6%）**与最好/最差样本放在同一张表里，读者可以同时看到代价与收益。

## 5. 口径声明

* **dB 区间**：`meta.db_bounds = [40.0, 200.0]` （来自 metrics 文件；本报告不替换、不推算）
* **权重口径（ema/raw）**：`ema`（`metrics.json:weights`）。训练记 EMA 成绩、评测用原始权重是上一版的已知 bug，两份口径**不得混用**。
* **split 口径**：`test`；test maps 只评一次（`data/splits.json` 冻结，val 只用于选模型/早停）。
* **splits_hash**：`89f81856f103ce5e`；不同 hash 的结果**不可直接比较**（划分不同）。
* **配对口径**：同一批 test maps 上逐样本配对；多 seed 先按 `sample_id` 取 seed 平均；显著性用 Wilcoxon 符号秩（非参数、配对），效应量用 Cohen's dz。

## 6. 数据来源（每个数字来自哪个文件）

| exp_id | config_name | seed | split | weights | n_samples | best_epoch | metrics 文件 | status |
|---|---|---|---|---|---|---|---|---|
| `prior_geometry_only_seed0` | `prior_geometry_only` | 0 | test | ema | 800 | 38 | `runs/prior_geometry_only_seed0/metrics_test.json` | 已读 |
| `prior_geometry_only_seed1` | `prior_geometry_only` | 1 | test | ema | 800 | 39 | `runs/prior_geometry_only_seed1/metrics_test.json` | 已读 |
| `prior_geometry_only_seed2` | `prior_geometry_only` | 2 | test | ema | 800 | 38 | `runs/prior_geometry_only_seed2/metrics_test.json` | 已读 |
| `prior_none_big_seed0` | `prior_none_big` | 0 | test | ema | 800 | 29 | `runs/prior_none_big_seed0/metrics_test.json` | 已读 |
| `prior_none_seed0` | `prior_none` | 0 | test | ema | 800 | 36 | `runs/prior_none_seed0/metrics_test.json` | 已读 |
| `prior_none_seed1` | `prior_none` | 1 | test | ema | 800 | 36 | `runs/prior_none_seed1/metrics_test.json` | 已读 |
| `prior_none_seed2` | `prior_none` | 2 | test | ema | 800 | 43 | `runs/prior_none_seed2/metrics_test.json` | 已读 |
| `prior_real_v3_oracle_seed0` | `prior_real_v3_oracle` | 0 | test | ema | 800 | 39 | `runs/prior_real_v3_oracle_seed0/metrics_test.json` | 已读 |
| `prior_real_v3_seed0` | `prior_real_v3` | 0 | test | ema | 800 | 39 | `runs/prior_real_v3_seed0/metrics_test.json` | 已读 |
| `prior_real_v3_seed1` | `prior_real_v3` | 1 | test | ema | 800 | 38 | `runs/prior_real_v3_seed1/metrics_test.json` | 已读 |
| `prior_real_v3_seed2` | `prior_real_v3` | 2 | test | ema | 800 | 36 | `runs/prior_real_v3_seed2/metrics_test.json` | 已读 |
| `prior_real_v3_shuffled_seed0` | `prior_real_v3_shuffled` | 0 | test | ema | 800 | 39 | `runs/prior_real_v3_shuffled_seed0/metrics_test.json` | 已读 |
| `prior_real_v3_shuffled_seed1` | `prior_real_v3_shuffled` | 1 | test | ema | 800 | 36 | `runs/prior_real_v3_shuffled_seed1/metrics_test.json` | 已读 |
| `prior_real_v3_shuffled_seed2` | `prior_real_v3_shuffled` | 2 | test | ema | 800 | 36 | `runs/prior_real_v3_shuffled_seed2/metrics_test.json` | 已读 |
| `prior_v3_masked_em_seed0` | `prior_v3_masked_em` | 0 | test | ema | 800 | 38 | `runs/prior_v3_masked_em_seed0/metrics_test.json` | 已读 |
| `prior_v3_masked_em_seed1` | `prior_v3_masked_em` | 1 | test | ema | 800 | 41 | `runs/prior_v3_masked_em_seed1/metrics_test.json` | 已读 |
| `prior_v3_masked_em_seed2` | `prior_v3_masked_em` | 2 | test | ema | 800 | 36 | `runs/prior_v3_masked_em_seed2/metrics_test.json` | 已读 |
