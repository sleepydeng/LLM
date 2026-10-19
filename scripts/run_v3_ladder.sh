#!/usr/bin/env bash
# 路线 A（prompt_v3）训练阶梯：四档对照 × N 个 seed → test 评测一次 → 自动报告。
#
# 为什么不直接用 run_experiments.sh --core
# ----------------------------------------
# `--core` 跑的是 **v1/v2** 四档（configs/prior_none|masked|real|shuffled.yaml），
# 其中 `prior_masked` 把**整个 64 维向量**置零。路线 A 的分工是"空间信息来自程序、
# 电磁信息来自 LLM"，所以容量对齐的对照必须**保留程序测出的几何**、只把 LLM 的
# 22 维（em_num 8 + global_cat 14）置零 —— 那就是 `prior_v3_masked_em`。
# 两者混用会把"几何被置零"的损失算到 LLM 头上，正是本项目反复在防的归因错误。
#
# 本阶梯：
#   prior_none            无先验，也无条件分支            ← 纯场景图的天花板
#   prior_geometry_only   42 维，只有程序几何             ← 不含任何 LLM 输出
#   prior_v3_masked_em    64 维，几何真、电磁置零         ← 容量逐位对齐的零信息对照
#   prior_real_v3         64 维，几何真、电磁真           ← 主实验（要检验的论点）
#   prior_real_v3_shuffled 64 维，内容样本间打乱          ← H2：起作用的是内容还是通路
#
# 关键差值：
#   real_v3 − masked_em       LLM 电磁**内容**的净贡献（参数量逐位相同）
#   real_v3 − geometry_only   几何 + 电磁 vs 只有几何（**参数量不同**，含容量效应）
#   real_v3 − real_v3_shuffled 内容 vs 通路
#
# 用法：
#   SEEDS="0" bash scripts/run_v3_ladder.sh            # 单 seed 先测链路与耗时
#   SEEDS="0 1 2" bash scripts/run_v3_ladder.sh        # 正式（3 seed，报 mean±std）
#   ONLY=real_v3 SEEDS="0" bash scripts/run_v3_ladder.sh   # 只跑名字里含该串的配置
#   DETACH=1 SEEDS="0 1 2" bash scripts/run_v3_ladder.sh   # 挂后台跑（SSH 断连不影响）
set -u

cd "$(dirname "$0")/.."

PY="${PY:-python}"
SEEDS="${SEEDS:-0 1 2}"
EVAL_WEIGHTS="${EVAL_WEIGHTS:-ema}"     # ema | raw，必须显式（AGENTS：禁止静默回退）
ONLY="${ONLY:-}"
DETACH="${DETACH:-0}"

# 训练是小时级任务：SSH 一断，前台进程会被 SIGHUP 杀掉。所以支持 DETACH=1
# 自我重启为后台任务，日志落在 reports/train_v3_ladder.log。
if [[ "$DETACH" == "1" && "${_V3_DETACHED:-0}" != "1" ]]; then
  mkdir -p reports
  LOG="reports/train_v3_ladder.log"
  echo "[detach] 后台启动，日志 $LOG"
  _V3_DETACHED=1 nohup env _V3_DETACHED=1 SEEDS="$SEEDS" ONLY="$ONLY" \
      EVAL_WEIGHTS="$EVAL_WEIGHTS" PY="$PY" bash "$0" > "$LOG" 2>&1 &
  echo "[detach] pid $!  →  tail -f $LOG"
  exit 0
fi

CONFIGS=(
  configs/prior_none.yaml
  configs/prior_geometry_only.yaml
  configs/prior_v3_masked_em.yaml
  configs/prior_real_v3.yaml
  configs/prior_real_v3_shuffled.yaml
)

SELECTED=()
for CFG in "${CONFIGS[@]}"; do
  NAME="$(basename "$CFG" .yaml)"
  if [[ -n "$ONLY" && "$NAME" != *"$ONLY"* ]]; then
    continue
  fi
  SELECTED+=("$CFG")
done
if [[ ${#SELECTED[@]} -eq 0 ]]; then
  echo "[错误] ONLY='$ONLY' 没有匹配到任何配置。可选："
  for CFG in "${CONFIGS[@]}"; do echo "  - $(basename "$CFG" .yaml)"; done
  exit 2
fi

echo "profile: seeds='$SEEDS' | eval weights=$EVAL_WEIGHTS | 选中 ${#SELECTED[@]} 个配置"
echo "start $(date -Is)"

FAILED=()
for CFG in "${SELECTED[@]}"; do
  NAME="$(basename "$CFG" .yaml)"
  for SEED in $SEEDS; do
    RUN_DIR="runs/${NAME}_seed${SEED}"
    echo "=============================================================="
    echo " $NAME  seed=$SEED  → $RUN_DIR   $(date -Is)"
    echo "=============================================================="

    if [[ -f "$RUN_DIR/metrics_test.json" ]]; then
      echo "[跳过] 已有 $RUN_DIR/metrics_test.json"
      continue
    fi

    if [[ -f "$RUN_DIR/ckpt/best.pth" ]]; then
      echo "[跳过训练] 已有 $RUN_DIR/ckpt/best.pth"
    else
      T0=$(date +%s)
      if ! $PY -m src.train.train --config "$CFG" --seed "$SEED"; then
        echo "[失败] 训练 $NAME seed=$SEED"
        FAILED+=("train:$NAME:$SEED")
        continue
      fi
      echo "[耗时] 训练 $(( $(date +%s) - T0 ))s"
    fi

    # test maps 只在这里评一次；权重口径显式
    if ! $PY -m src.eval.run_eval --run-dir "$RUN_DIR" --split test \
            --weights "$EVAL_WEIGHTS" --out "$RUN_DIR/metrics_test.json"; then
      echo "[失败] 评测 $NAME seed=$SEED"
      FAILED+=("eval:$NAME:$SEED")
    fi
  done
done

echo "=============================================================="
echo " 自动生成报告（禁止手抄数字，AGENTS 规则 5）"
echo "=============================================================="
$PY -m src.eval.report --runs runs --out reports/summary.md --json reports/summary.json \
  || echo "[警告] 报告生成失败，请看上面的报错"

echo "done $(date -Is)"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "有 ${#FAILED[@]} 个任务失败："
  printf '  - %s\n' "${FAILED[@]}"
  echo "重跑同一条命令即可续跑（已完成的会被跳过）。"
  exit 1
fi
echo "全部完成。报告：reports/summary.md"
