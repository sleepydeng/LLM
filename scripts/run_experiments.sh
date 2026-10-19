#!/usr/bin/env bash
# 批量训练 + test 评测 + 自动报告。
#
# 为什么脚本里写死了 3 个 seed（AGENTS 规则 2）：
#   上一版没有任何随机种子，+0.25 dB 的"提升"与单次运行波动同量级，无法排除噪声。
#   结论必须来自 3 个 seed 的 mean ± std，所以这里把 seed 循环固化，
#   不给"先跑一个看看"留下悄悄出一个单种子结论的机会。
#
# 用法：
#   bash scripts/run_experiments.sh --core        # 四配置 × 3 seed（主表）
#   bash scripts/run_experiments.sh --ablations   # 注入方式 / CoT / sectors 消融
#   bash scripts/run_experiments.sh --all
set -euo pipefail

cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1 2}"
PY="${PY:-python}"
EVAL_WEIGHTS="${EVAL_WEIGHTS:-ema}"      # ema | raw，必须显式
RESUME="${RESUME:-1}"                    # 1 = 已有 metrics 就跳过

CORE_CONFIGS=(
  configs/prior_none.yaml
  configs/prior_masked.yaml
  configs/prior_real.yaml
  configs/prior_shuffled.yaml
)
ABLATION_CONFIGS=(
  configs/prior_global_only.yaml
  configs/prior_sectors_only.yaml
  configs/prior_sectors_channels.yaml
  configs/prior_real_cot.yaml
  configs/prior_real_cot_shuffled.yaml     # CoT 的容量对照，H3 必须成对跑
  configs/prior_real_cot_only.yaml
  configs/prior_cot_only_shuffled.yaml
)

MODE="core"
for a in "$@"; do
  case "$a" in
    --core)      MODE="core" ;;
    --ablations) MODE="ablations" ;;
    --all)       MODE="all" ;;
    *) echo "未知参数 $a"; exit 2 ;;
  esac
done

CONFIGS=()
if [[ "$MODE" == "core" || "$MODE" == "all" ]]; then CONFIGS+=("${CORE_CONFIGS[@]}"); fi
if [[ "$MODE" == "ablations" || "$MODE" == "all" ]]; then CONFIGS+=("${ABLATION_CONFIGS[@]}"); fi

echo "模式 $MODE｜配置 ${#CONFIGS[@]} 个｜seed: $SEEDS｜评测权重口径 $EVAL_WEIGHTS"
echo

FAILED=()
for CFG in "${CONFIGS[@]}"; do
  NAME="$(basename "$CFG" .yaml)"
  for SEED in $SEEDS; do
    RUN_DIR="runs/${NAME}_seed${SEED}"
    echo "=============================================================="
    echo " $NAME  seed=$SEED  → $RUN_DIR"
    echo "=============================================================="

    if [[ "$RESUME" == "1" && -f "$RUN_DIR/metrics_test.json" ]]; then
      echo "[跳过] 已有 $RUN_DIR/metrics_test.json"
      continue
    fi

    if [[ ! -f "$RUN_DIR/ckpt/best.pth" ]]; then
      if ! $PY -m src.train.train --config "$CFG" --seed "$SEED"; then
        echo "[失败] 训练 $NAME seed=$SEED"
        FAILED+=("train:$NAME:$SEED")
        continue
      fi
    else
      echo "[跳过训练] 已有 $RUN_DIR/ckpt/best.pth"
    fi

    # test maps 只在这里评一次，且权重口径显式指定
    if ! $PY -m src.eval.run_eval --run-dir "$RUN_DIR" --split test \
            --weights "$EVAL_WEIGHTS" --out "$RUN_DIR/metrics_test.json"; then
      echo "[失败] 评测 $NAME seed=$SEED"
      FAILED+=("eval:$NAME:$SEED")
    fi
  done
  echo
done

echo "=============================================================="
echo " 自动生成报告（禁止手抄数字，见 AGENTS 规则 5）"
echo "=============================================================="
$PY -m src.eval.report --runs runs --out reports/summary.md \
    --json reports/summary.json || echo "[警告] 报告生成失败，请看上面的报错"

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "有 ${#FAILED[@]} 个任务失败："
  printf '  - %s\n' "${FAILED[@]}"
  echo "重跑同一条命令即可续跑（已完成的会被跳过）。"
  exit 1
fi
echo "全部完成。报告：reports/summary.md"
echo "训练曲线：python -m src.eval.visualize curves --runs runs --out reports/curves.png"
