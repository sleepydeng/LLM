#!/usr/bin/env bash
# 阶梯训练完成后运行的两项补充（阳性对照 oracle + 容量天花板），然后重出报告。
#
# 为什么不能和阶梯并行：prior_none_big（base_channels=64）单卡约 40GB 显存，
# 不能与任何 15GB 的训练共存；所以这里**串行**跑 oracle → big。
#
# 用法：等 run_v3_full.sh 的阶梯训练全部结束后，手动或由上层 driver 调用：
#   PY=/root/miniconda3/bin/python DETACH=1 bash scripts/run_positive_controls.sh
set -u
cd "$(dirname "$0")/.."

PY="${PY:-python}"
if ! command -v "$PY" >/dev/null 2>&1 && [ -x /root/miniconda3/bin/python ]; then
  PY=/root/miniconda3/bin/python
fi
EVAL_WEIGHTS="${EVAL_WEIGHTS:-ema}"
DETACH="${DETACH:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ "$DETACH" == "1" && "${_PC_DETACHED:-0}" != "1" ]]; then
  mkdir -p reports
  LOG="reports/positive_controls.log"
  echo "[detach] 后台启动，日志 $LOG"
  _PC_DETACHED=1 nohup env _PC_DETACHED=1 EVAL_WEIGHTS="$EVAL_WEIGHTS" PY="$PY" \
      bash "$0" > "$LOG" 2>&1 &
  echo "[detach] pid $!"
  exit 0
fi

log() { echo "[$(date -Is)] $*"; }

train_and_eval() {
  local cfg="$1" seed="$2"
  local name run_dir
  name="$(basename "$cfg" .yaml)"
  run_dir="runs/${name}_seed${seed}"
  if [[ -f "$run_dir/metrics_test.json" ]]; then
    log "[skip 已评] $name seed=$seed"
    return 0
  fi
  if [[ ! -f "$run_dir/ckpt/best.pth" ]]; then
    log "[train] $name seed=$seed"
    $PY -m src.train.train --config "$cfg" --seed "$seed" \
      > "reports/train_${name}_seed${seed}.log" 2>&1 \
      || { log "[FAIL train] $name"; return 1; }
  else
    log "[skip 已训] $name seed=$seed"
  fi
  $PY -m src.eval.run_eval --run-dir "$run_dir" --split test --weights "$EVAL_WEIGHTS" \
      --out "$run_dir/metrics_test.json" >> "reports/eval_${name}_seed${seed}.log" 2>&1 \
    && log "[eval ok] $name" || log "[eval FAIL] $name"
}

log "=== 阳性对照 oracle（真值统计量灌入 EM 22 维）==="
train_and_eval configs/prior_real_v3_oracle.yaml 0

log "=== 容量天花板（base_channels=64 纯场景图）==="
train_and_eval configs/prior_none_big.yaml 0

log "=== 重出报告（含 oracle 与 big）==="
$PY -m src.eval.report --runs runs --out reports/summary.md --json reports/summary.json \
    --ladder v3 || log "[警告] 报告生成失败"

log "=== 关键对照速览 ==="
$PY - <<'PY'
import json, pathlib
def rmse(d):
    p = pathlib.Path(f"runs/{d}/metrics_test.json")
    if not p.exists(): return None
    m = json.loads(p.read_text())
    a = m.get("aggregate", {})
    return a.get("mean_rmse_db"), a.get("mean_rmse_db_nonsaturated")
for d in ("prior_v3_masked_em_seed0", "prior_real_v3_seed0",
          "prior_real_v3_oracle_seed0", "prior_none_big_seed0", "prior_none_seed0"):
    r = rmse(d)
    print(f"  {d:28s} RMSE={r[0]:.3f} dB  non-sat={r[1]:.3f}" if r else f"  {d}: (未跑)")
PY
log "done $(date -Is)"
