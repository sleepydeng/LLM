#!/usr/bin/env bash
# 路线 A（prompt_v3）全阶梯训练：5 配置 × 3 seed，**并行**训练 → 统一 test 评测 → 自动报告。
#
# 与 run_v3_ladder.sh 的区别：那个是串行、边训边评。这个是先并行训完所有缺的 run，
# 再统一评测 + 报告。并行度 PAR（默认 2，单卡 49GB 下每个 run 约 15GB，3 个就贴上限，
# 2 个安全）。已完成的 run（有 metrics_test.json）自动跳过，可安全重跑续传。
#
# 阶梯（见 run_v3_ladder.sh 的完整注释）：
#   prior_none            无先验无条件分支            ← 绝对地板
#   prior_geometry_only   42 维程序几何               ← 阳性对照：条件通路有没有用
#   prior_v3_masked_em    64 维 几何真/电磁零         ← 容量逐位对齐的零信息对照
#   prior_real_v3         64 维 几何真/电磁真         ← 主实验
#   prior_real_v3_shuffled 64 维 内容打乱             ← H2 内容 vs 通路
#
# 用法：
#   SEEDS="0 1 2" PAR=2 DETACH=1 bash scripts/run_v3_full.sh
set -u
cd "$(dirname "$0")/.."

PY="${PY:-python}"
# 远程 AutoDL 的非交互式 SSH 里 `python` 不在 PATH（只有 /root/miniconda3/bin/python）。
# 自愈：如果 `python` 找不到、但 miniconda 在，就回退到它。避免"脚本能跑但每个任务
# 都秒退 rc=127"这种静默失败。
if ! command -v "$PY" >/dev/null 2>&1 && [ -x /root/miniconda3/bin/python ]; then
  PY=/root/miniconda3/bin/python
fi
SEEDS="${SEEDS:-0 1 2}"
EVAL_WEIGHTS="${EVAL_WEIGHTS:-ema}"
PAR="${PAR:-2}"
DETACH="${DETACH:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ "$DETACH" == "1" && "${_V3F_DETACHED:-0}" != "1" ]]; then
  mkdir -p reports
  LOG="reports/train_v3_full.log"
  echo "[detach] 后台启动，日志 $LOG"
  _V3F_DETACHED=1 nohup env _V3F_DETACHED=1 SEEDS="$SEEDS" EVAL_WEIGHTS="$EVAL_WEIGHTS" \
      PAR="$PAR" PY="$PY" bash "$0" > "$LOG" 2>&1 &
  echo "[detach] pid $!"
  exit 0
fi

CONFIGS=(
  configs/prior_none.yaml
  configs/prior_geometry_only.yaml
  configs/prior_v3_masked_em.yaml
  configs/prior_real_v3.yaml
  configs/prior_real_v3_shuffled.yaml
)

log() { echo "[$(date -Is)] $*"; }

log "profile: seeds='$SEEDS' | PAR=$PAR | eval_weights=$EVAL_WEIGHTS"

# ---- Phase 0：列出待训练任务 ----
JOBS=()
for CFG in "${CONFIGS[@]}"; do
  NAME="$(basename "$CFG" .yaml)"
  for SEED in $SEEDS; do
    RUN_DIR="runs/${NAME}_seed${SEED}"
    if [[ -f "$RUN_DIR/metrics_test.json" ]]; then
      log "[skip 已评] $NAME seed=$SEED"
      continue
    fi
    if [[ -f "$RUN_DIR/ckpt/best.pth" ]]; then
      log "[skip 已训] $NAME seed=$SEED (待评测)"
      JOBS+=("EVAL:$NAME:$SEED")
    else
      JOBS+=("TRAIN:$NAME:$SEED")
    fi
  done
done

# ---- Phase 1：并行训练 ----
log "待训练 ${#JOBS[@]} 项，并行度 $PAR"
running=0
for JOB in "${JOBS[@]}"; do
  KIND="${JOB%%:*}"; REST="${JOB#*:}"; NAME="${REST%%:*}"; SEED="${REST#*:}"
  [[ "$KIND" != "TRAIN" ]] && continue
  CFG="configs/${NAME}.yaml"
  RUN_DIR="runs/${NAME}_seed${SEED}"
  (
    $PY -m src.train.train --config "$CFG" --seed "$SEED" \
      > "reports/train_${NAME}_seed${SEED}.log" 2>&1
    echo "TRAIN_DONE ${NAME} seed=${SEED} rc=$?" >> reports/train_v3_full.status
  ) &
  running=$((running+1))
  log "[launch] train $NAME seed=$SEED (并发 $running)"
  if [[ $running -ge $PAR ]]; then
    wait -n || true
    running=$((running-1))
  fi
done
wait
log "训练阶段完成"

# ---- Phase 2：统一评测 ----
for CFG in "${CONFIGS[@]}"; do
  NAME="$(basename "$CFG" .yaml)"
  for SEED in $SEEDS; do
    RUN_DIR="runs/${NAME}_seed${SEED}"
    [[ -f "$RUN_DIR/metrics_test.json" ]] && continue
    if [[ -f "$RUN_DIR/ckpt/best.pth" ]]; then
      $PY -m src.eval.run_eval --run-dir "$RUN_DIR" --split test --weights "$EVAL_WEIGHTS" \
          --out "$RUN_DIR/metrics_test.json" >> "reports/eval_${NAME}_seed${SEED}.log" 2>&1 \
        && log "[eval ok] $NAME seed=$SEED" || log "[eval FAIL] $NAME seed=$SEED"
    else
      log "[无 ckpt] $NAME seed=$SEED —— 训练阶段可能失败了"
    fi
  done
done

# ---- Phase 3：报告 ----
log "生成报告"
$PY -m src.eval.report --runs runs --out reports/summary.md --json reports/summary.json \
    --ladder v3 || log "[警告] 报告生成失败"

log "done $(date -Is)"
