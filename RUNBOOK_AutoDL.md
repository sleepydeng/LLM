# AutoDL（Linux + NVIDIA GPU）运行手册

> 本机（Windows）不能训练：没有 CUDA、没有系统 Python、代码只在本地做"不需要 GPU 的验证"。
> 正式结果一律来自这里描述的远程机器（AGENTS 规则 7）。

---

## 0. 你需要准备的东西

| 项 | 说明 |
|---|---|
| AutoDL 实例 | 单卡 24 GB（3090 / 4090 / A5000）足够：模型 6.6 M 参数、输入 256×256 |
| 镜像 | PyTorch 2.x + CUDA 12.x + Python 3.10 的官方镜像 |
| 智谱 API key | `ZHIPUAI_API_KEY`，用于 `glm-4v-flash`（免费模型，4000 样本 × 3 次采样 = 12000 次调用） |
| 数据 | 本机已有 65.8 MB（8103 张图），打包上传即可，**不需要重新下载** |

AutoDL 的 SSH 登录命令每次开机都会变（host + port 都会换），以控制台显示的为准。

---

## 1. 把工作目录放到数据盘

AutoDL 系统盘通常只有 30–50 GB，训练产物和 ckpt 都会变大，所以代码与数据都放 `/root/autodl-tmp`：

```bash
mkdir -p /root/autodl-tmp/work
ln -sfn /root/autodl-tmp/work /root/work
cd /root/work
```

## 2. 在 Windows 上打包并上传

仓库里提供了打包脚本（本机执行，不是在远程机）：

```powershell
# 只打代码（不含数据、不含 venv、不含 runs）
powershell -File scripts\upload_to_autodl.ps1 -Target distill-lab -What code

# 单独打数据（65.8 MB，压完会更小）
powershell -File scripts\upload_to_autodl.ps1 -Target distill-lab -What data
```

脚本会用 `tar` 打包（Windows 10+ 自带 bsdtar）再 `scp` 上传，并在远程校验文件数。
`-Target` 是 `~/.ssh/config` 里的主机别名，也支持 `user@host -Port 12345` 形式。

上传结束后远程应长这样：

```
/root/work/llm-prior-radiomap/
├── data/radiomapseer/{buildings_complete,gain/carsDPM,antenna}/
├── data/splits.json
├── configs/ src/ scripts/ tests/
```

## 3. 安装依赖

```bash
cd /root/work/llm-prior-radiomap
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`cuda.is_available()` 必须是 `True`，否则后面的训练会退化成 CPU（脚本会打警告但不会拦你）。

## 4. 开工前的三道校验（都不通过就不许往下走）

```bash
# 4.1 数据完整性：应为 建筑掩码 50 / gain 4000 / antenna 50 → 校验通过
python scripts/verify_data.py --data data/radiomapseer

# 4.2 划分：应为 train 35 maps(2800) / val 5(400) / test 10(800)
python -c "
from src.data.splits import load_splits
s = load_splits('data/splits.json'); print(s['counts']); print('test maps', s['test_maps'])"

# 4.3 本地自检（不需要 GPU，但值得在真机上跑一遍确认环境没坏）
python -m tests.test_local        # 期望 33/33 通过
python -m tests.test_pipeline     # 期望 端到端自检通过
```

如果 4.3 失败，先解决环境问题再看先验；不要把"环境没装好"误判成"代码有问题"。

## 5. 生成先验（当前唯一的关键路径）

```bash
export ZHIPUAI_API_KEY=你的key
tmux new -s priors          # 长任务别用前台终端
bash scripts/gen_priors.sh
```

脚本会依次做：

1. 试生成 4 张地图（320 样本 × 3 次采样）；
2. 跑 `scripts/audit.py` 硬门槛 —— **不通过就直接退出，并打印该改 prompt 的哪一部分**；
3. 通过后才生成全量 4000 个样本（按 train/val/test 三个 split）；
4. 再体检一次，写入 `reports/audit.json`，并列出 `_failed/` 里的失败样本。

**门槛不通过时不要做任何"绕过"**：不要手工把失败样本填成默认值（AGENTS 规则 6），
不要跳过 audit 直接训练。整轮实验的价值都建立在这一步上。

判断标准（必须全面优于上一版的基线）。**硬门槛**（不过就退出码 1，禁止进训练）：

| 硬门槛 | 阈值 | 判定什么 | 上一版实测 |
|---|---|---|---|
| `unique_ratio` | ≥ 0.50 | 向量整体是否有逐样本信息 | 0.0138 |
| `intra_map_unique_ratio_mean` | ≥ 0.50 | 同一地图内不同 Tx 是否真的不同 | 0.1043 |
| `numeric_unique_min` | ≥ 5 | 每个数值字段是否真的在变 | 3（`pathloss_min_db`） |
| `numeric_entropy_min` | ≥ 1.0 bits | 每个数值字段的熵 | 0.65（`penetration_loss_db`） |
| `categorical_min_unique` | ≥ 2 | 类别字段是否退化成单一常量 | 2（通过） |
| `sector_consistency_strict` | ≥ 0.70 | LLM 是否真的按方位线看图 | 无该字段（随机约 0.25） |
| `sector_consistency_gain` | ≥ 0.10 | **自校准**：本 Tx 一致率 − 同图另一 Tx 一致率 | 无该字段 |

> `sector_consistency` 之所以有两条：绝对一致率的偷懒下限取决于几何真值的标签分布。
> 实测 maps 0–3 上 `blocked` 占 59.3%、`open` 仅 0.7%，于是"永远回答 blocked"
> 就能拿 0.593，离 0.70 只差 0.107 —— 换一批建筑更密的地图就会越线。
> 而**任何偷懒策略的自校准增益都精确为 0**，所以这条更稳。
> 生成前先跑 `python scripts/pilot_preflight.py --maps 0:4` 看本批地图的余量。

建议级（不阻断，但要在报告里如实说明）：`max_top1_share`、`confidence_top1_share`、
`scene_type_classes`、`reasoning_repeat_share`、`content_contamination_share`、
`mmwave_hallucination_share`、`height_hallucination_share`、`field_consistency_mean`、
`fields_varying_within_map`。

> `max_top1_share` 之所以只是建议级：它会把"LLM 偷懒"和"数据本来就不均衡"混为一谈。
> 50 张城市地图真实是 60% urban / 40% dense_urban 时，一份正确的 prompt 也会 top1 = 0.60。
> 详见 `AGENTS.md`「关于 audit 门槛的两条实测教训」。

每样本耗时参考：320 样本 × 3 次采样、6 并发，约十几分钟。

生成成本参考：4000 样本 × 3 次调用 = 12000 次请求，`glm-4v-flash` 免费。
时间主要取决于并发与限流，可按 `WORKERS=8` 调整。

## 6. 训练（四配置 × 3 seed = 12 次）

```bash
tmux new -s train
bash scripts/run_experiments.sh --core
```

每个 run 产出 `runs/<config>_seed<seed>/`：
`config.yaml` / `config.json` / `train.log` / `ckpt/best.pth` / `history.json` / `metrics_test.json`。

机时估算：模型 6.6 M、数据 2800 训练样本、256×256、batch 16、50 epoch。
单卡 24 GB 上**几分钟一个 epoch 以内**（上一版在 Mac MPS 上约 5 小时/次，GPU 会快一个量级）。
按 12 次算，先跑 1 个 run 测出单次耗时，再乘 12 决定是否要分两批。

先只跑一个 run 验证链路也是允许的，但**不要用单种子结果做结论**：

```bash
bash scripts/run_experiments.sh --core     # 中途可以 Ctrl-C，重跑会自动跳过已完成的
```

## 7. 补充消融（注入方式 / CoT / sectors）

```bash
bash scripts/run_experiments.sh --ablations
```

**注意 CoT 必须成对跑**：`prior_real_cot` 与 `prior_real_cot_shuffled` 的结构逐位相同
（参数量都是 6,895,041）。只跑前者会把 CoT 编码器带来的 28.7 万参数容量效应
误算成"CoT 推理有用"。三个数的关系是：

```
prior_real_cot − prior_real_cot_shuffled  →  CoT 内容的净贡献
prior_real_cot − prior_real               →  CoT 内容 + 额外容量
```

同理 `prior_real_cot_only` 要配 `prior_cot_only_shuffled`。

## 8. 报告

```bash
python -m src.eval.report --runs runs --out reports/summary.md --json reports/summary.json
python -m src.eval.visualize curves --runs runs --out reports/curves.png

# 单个样本的可视化（mask+Tx | 真值 | 预测 | 误差）
python -m src.eval.visualize predict \
    --ckpt runs/prior_real_seed0/ckpt/best.pth \
    --out-dir reports/preds --split test --n 8
```

报告里的每一张表都由 `runs/*/metrics_test.json` 自动生成。**如果某个数字不在
metrics 文件里，就不要把它写进报告** —— 上一版的正文与图表对不上就是这么来的。

## 9. 下载结果回本机

```bash
# 在远程
tar czf /root/autodl-tmp/results.tgz runs reports priors/*/*/audit*.json
```

```powershell
# 在本机
scp distill-lab:/root/autodl-tmp/results.tgz .
```

ckpt 很大，如果只关心数字，可以不打包 `ckpt/`。

---

## 排错表

| 症状 | 原因与处理 |
|---|---|
| `torch.cuda.is_available()` 为 False | 镜像或驱动不对；重开实例选 PyTorch + CUDA 镜像 |
| `verify_data.py` 报文件数不符 | 上传不完整；重跑上传脚本（脚本会校验文件数） |
| `dataset` 抛 `FileNotFoundError: N/2800 个样本…没有先验文件` | 先验没生成全。**不要**加 `allow_missing_priors`，直接重跑 `gen_priors.sh`（断点续传） |
| 先验生成大量 `_failed/` | 看 `error` 字段：限流就降 `--workers`；schema 校验失败说明模型没按格式回，先看是哪个字段越界 |
| `audit.py` 退出码 1 | 正常行为：门槛没过。按输出的提示改 prompt，换新版本目录重跑 |
| 训练时 val RMSE 前几个 epoch 高达上百 dB | 正常（随机初始化 + EMA 还没热身）。**注意**：`eval_weights=ema` 时前期 EMA 会很差，best ckpt 会落在后期；这正是上一版"epoch 1 FiLM 118 dB"的来源，不是方法缺陷 |
| `run_eval` 报 `没有 ema_shadow，但请求了 --weights ema` | 有意的硬失败（防止上一版那种静默回退）。改用 `--weights raw` 或重训（`train.ema=true`） |
| OOM | 降 `train.batch_size`（16 → 8），或把 `--batch-size` 传给 run_eval |
| 要求 bit 级复现 | 保持 `--deterministic`（默认开）；会慢一些，且 `cudnn.benchmark` 会被关掉 |
