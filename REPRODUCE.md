# 复现手册：LLM 先验有没有信息（路线 A / prompt_v3）

> 本文档给出**从零复现全部论文级数字**的精确步骤与命令。所有数字由
> `src/eval/report.py` 自动生成，禁止手抄（AGENTS.md 规则 5）。

## 0. 结论一句话（先看这个）

在 glm-4v-flash × prompt_v3 × RadioMapSeer 上，把 LLM 产出的 22 维电磁先验
喂进 FiLM 条件生成模型，**对 test 集 RMSE 没有正贡献**（`real_v3 ≈ masked_em`）。
这条负面结果由三个正交判据（L0 门槛 / L1 三仪器 / L2 消融）三角定位，
并附阳性对照（oracle）与容量天花板，证明判据本身有区分度。

## 1. 环境

- 远程 GPU：AutoDL，`ssh -p <port> root@connect.<region>.seetacloud.com`，
  项目在 `/root/autodl-tmp/work/llm-prior-radiomap`。
- Python：`/root/miniconda3/bin/python`（非交互式 SSH 里 `python` 不在 PATH）。
- 依赖：torch 2.8.0+cu128、sklearn、pydantic、openai、Pillow、pyyaml、scipy。
- **`configs/base.yaml` 里 `train.amp` 必须为 `false`**（AMP+SSIM 会让梯度 NaN、
  GradScaler 静默跳过更新，模型停在随机初始化 —— 见 AGENTS.md「AMP 硬教训」）。

## 2. 数据与划分

数据 50 maps / 4000 gain / 4000 Tx，划分冻结在 `data/splits.json`
（leave-maps-out，35 train / 5 val / 10 test）。

```bash
cd /root/autodl-tmp/work/llm-prior-radiomap
/root/miniconda3/bin/python scripts/verify_data.py --data data/radiomapseer
```

## 3. 先验（LLM 生成 + oracle 阳性对照）

```bash
# LLM 先验（prompt_v3：程序给几何事实，LLM 只做电磁推理）
export ZHIPUAI_API_KEY=...
for S in train val test; do
  /root/miniconda3/bin/python -m src.prior.generate \
    --data data/radiomapseer --priors priors/prompt_v3/glm-4v-flash \
    --splits data/splits.json --split $S --workers 8 --temperature 0.7 \
    --samples-per-call 3
done

# 硬门槛体检（退出码 1 = 不许进训练）
/root/miniconda3/bin/python scripts/audit.py \
  --priors priors/prompt_v3/glm-4v-flash --data data/radiomapseer \
  --px-per-m 1.0 --out reports/audit.json

# oracle 阳性对照：把 EM 22 维换成真值 pathloss 统计量
/root/miniconda3/bin/python scripts/make_oracle_priors.py
```

L1 三仪器类型学（零 GPU，回答"哪些字段有信息"）：

```bash
/root/miniconda3/bin/python scripts/probe_l1_typology.py \
  --priors priors/prompt_v3/glm-4v-flash
# → reports/llm_l1_typology.md
```

## 4. 训练阶梯（5 配置 × 3 seed，并行 2）

```bash
SEEDS="0 1 2" PAR=2 DETACH=1 bash scripts/run_v3_full.sh
# 阶梯：prior_none / prior_geometry_only / prior_v3_masked_em /
#       prior_real_v3 / prior_real_v3_shuffled
# 完成后自动 test 评测 + report.py --ladder v3 报告。
```

## 5. 阳性对照 + 容量天花板（阶梯后串行）

```bash
PY=/root/miniconda3/bin/python DETACH=1 bash scripts/run_positive_controls.sh
# prior_real_v3_oracle（oracle 先验，1 seed）+ prior_none_big（base_channels=64，1 seed）
# 之后重出报告（含 oracle/big 的配对检验与参数量对齐检查）。
```

## 6. 报告（数字唯一来源）

```bash
/root/miniconda3/bin/python -m src.eval.report \
  --runs runs --out reports/summary.md --json reports/summary.json --ladder v3
```

关键差值（自动生成）：
- `prior_real_v3 − prior_v3_masked_em`：LLM 电磁**内容**净贡献（参数量逐位相同 6,608,161）；
- `prior_real_v3 − prior_real_v3_shuffled`：内容 vs 通路；
- `prior_geometry_only − prior_none`：程序几何先验（阳性对照 1）；
- `prior_real_v3_oracle − prior_v3_masked_em`：oracle 阳性对照 2（EM 22 维能否携带真信号）；
- `prior_none_big − prior_none`：容量天花板。

## 7. 自检（改代码后必跑）

```bash
python -m tests.test_local      # 46 项（含 FiLM 先验必须影响前向、AMP 检查）
python -m tests.test_gate       # audit 门槛区分度
python -m tests.test_typology   # L1 类型学判据（几何解析/方位对齐/判据分支/小样本保护）
# oracle 先验的 schema 有效性由 make_oracle_priors.py 自身保证：
# 训练时 RadioMapSeerDataset 会用量程/方位校验兜底，越界会抛错（规则 6）
```

## 8. 关键修复记录（不修这两处，结果全假）

1. **AMP**：`train.amp=false`，否则 SSIM 梯度 NaN → 模型不学。
2. **FiLM identity 初始化**：末层权重不能置零（要 1e-4 量级小随机值），否则
   先验从不影响前向、`real_v3` 与 `masked_em` 数学恒等。
   自检 `test_local: film_prior_actually_conditions` 锁死这条。
