#!/usr/bin/env bash
# 先验生成全流程：几何可行性检查 → 试生成 → 硬门槛体检 → 全量 → 再体检。
#
# 为什么必须先试生成再体检（AGENTS 规则 1）：
#   上一版 4000 条先验花了整轮 API 调用，事后才发现只有 55 个唯一向量、实验无效。
#   代价不是 API 费用，而是**整轮实验报废**。所以在任何全量生成之前，
#   先用 4 张地图（320 个样本）把门槛跑一遍，不达标就改 prompt 重来。
#
# 注意：本文件是 UTF-8。**不要**用 Windows PowerShell 5.1 的
# Get-Content/Set-Content 做读写往返 —— 它默认按 GBK 解码，会把中文以及
# 紧随其后的 ASCII（连 heredoc 的 EOF 终止符）一起改坏。要改用编辑工具。
#
# 用法：
#   export ZHIPUAI_API_KEY=xxxx
#   bash scripts/gen_priors.sh                 # 完整流程
#   bash scripts/gen_priors.sh --pilot-only    # 只跑可行性检查 + 试生成 + 体检
#   bash scripts/gen_priors.sh --full-only     # 只跑全量（试生成已过门槛后）
set -euo pipefail

cd "$(dirname "$0")/.."

PRIORS_DIR="${PRIORS_DIR:-priors/prompt_v2/glm-4v-flash}"
DATA_DIR="${DATA_DIR:-data/radiomapseer}"
SPLITS="${SPLITS:-data/splits.json}"
WORKERS="${WORKERS:-6}"
PILOT_MAP_RANGE="${PILOT_MAP_RANGE:-0:4}"     # 4 张地图 × 80 Tx = 320 个样本
PY="${PY:-python}"

MODE="all"
for a in "$@"; do
  case "$a" in
    --pilot-only) MODE="pilot" ;;
    --full-only)  MODE="full" ;;
    *) echo "未知参数 $a"; exit 2 ;;
  esac
done

if [[ -z "${ZHIPUAI_API_KEY:-}" ]]; then
  echo "[错误] 未设置 ZHIPUAI_API_KEY。示例：export ZHIPUAI_API_KEY=xxxx"
  exit 2
fi

echo "=============================================================="
echo " 0) 数据校验（不通过不许开工）"
echo "=============================================================="
$PY scripts/verify_data.py --data "$DATA_DIR"

if [[ "$MODE" == "all" || "$MODE" == "pilot" ]]; then
  echo
  echo "=============================================================="
  echo " 1) 几何可行性检查（不调 API，0 成本）"
  echo "=============================================================="
  # 这一步回答两个问题：本批地图的标签分布下"偷懒策略"能拿多少一致率、
  # 门槛还剩多少余量；以及有多少 Tx 落在建筑轮廓内（那些样本 8 个方位
  # 大概率全 blocked，逐方位读图本来就没有区分空间）。
  $PY scripts/pilot_preflight.py --data "$DATA_DIR" \
      --maps "$PILOT_MAP_RANGE" --out reports/pilot_preflight.json || true

  echo
  echo "=============================================================="
  echo " 2) 试生成：maps $PILOT_MAP_RANGE（约 320 样本 × 3 次采样）"
  echo "=============================================================="
  $PY -m src.prior.generate \
      --data "$DATA_DIR" \
      --priors "$PRIORS_DIR" \
      --map-range "$PILOT_MAP_RANGE" \
      --workers "$WORKERS" \
      --temperature 0.7 \
      --samples-per-call 3

  echo
  echo "=============================================================="
  echo " 3) 硬门槛体检（退出码 1 = 未通过，禁止进入训练）"
  echo "=============================================================="
  set +e
  $PY scripts/audit.py \
      --priors "$PRIORS_DIR" \
      --data "$DATA_DIR" \
      --px-per-m 1.0 \
      --out reports/audit_pilot.json
  AUDIT_RC=$?
  set -e

  if [[ $AUDIT_RC -ne 0 ]]; then
    cat <<'EOF'

[停止] 试生成未通过硬门槛。
不要继续生成全量、更不要进训练。按下面的顺序改，然后重跑本脚本：

  1. 看 reports/audit_pilot.json 里 gate.results 哪几条 severity=fail 且 pass=false：
     * unique_ratio 低        → LLM 输出仍然坍缩。检查 scene_hint 是否又被塞回 prompt、
                                 few-shot 是否又变成固定示例、temperature 是否被改回 0
     * intra_map 低           → 同一张地图内不同 Tx 的先验没变化，说明局部裁剪图没起作用；
                                 确认 prompt 里说清了 1px=1m 与方位线含义
     * numeric_unique_min 低  → 某个数值维度几乎恒定（报告里会直接列出字段名）。
                                 通常要给该字段更明确的判断依据，或改它的量程
     * numeric_entropy_min 低 → 同上，该字段取值过于集中
     * categorical_min_unique→ 某个类别字段（scene_type / dominant_material / confidence）
                                 退化成单一常量；常见于 confidence 被多轮共识抹平
     * sector_consistency_strict 低 → LLM 没在按方位线读图；可放大局部图、加粗方位线、
                                 或把 nearest_building_m 的问法改得更具体
     * sector_consistency_gain 低   → 更硬的信号：LLM 的方位判定与本 Tx 几何的吻合度
                                 并不比与**同图另一个 Tx** 几何的吻合度更高，
                                 也就是它其实没在看这个 Tx。任何偷懒策略此值恒为 0。
                                 此时先确认本地裁剪图里 Tx 位置与方位线是否清晰可辨
  2. severity=warn 的条目**不阻断**，但要在报告里如实说明（例如数据本身类别不均衡）
  3. 改完先用更小的范围（例如 --map-range 0:1）快速试，再放大到 0:4
  4. 每次改动都要在 reports/ 下留一份 audit 报告作为对照

建议：把 prompt 版本号一起改（例如 priors/prompt_v3/glm-4v-flash），不要原地覆盖。
EOF
    exit 1
  fi
  echo "[OK] 试生成通过硬门槛。"
fi

if [[ "$MODE" == "all" || "$MODE" == "full" ]]; then
  echo
  echo "=============================================================="
  echo " 4) 全量生成：train / val / test 三个 split 合起来 4000 个样本"
  echo "=============================================================="
  for SPLIT in train val test; do
    echo "---- split=$SPLIT ----"
    $PY -m src.prior.generate \
        --data "$DATA_DIR" \
        --priors "$PRIORS_DIR" \
        --splits "$SPLITS" --split "$SPLIT" \
        --workers "$WORKERS" \
        --temperature 0.7 \
        --samples-per-call 3
  done

  echo
  echo "=============================================================="
  echo " 5) 全量体检（必须比 reports/audit_legacy_v2.json 的那份基线全面更好）"
  echo "=============================================================="
  $PY scripts/audit.py \
      --priors "$PRIORS_DIR" \
      --data "$DATA_DIR" \
      --px-per-m 1.0 \
      --out reports/audit.json

  echo
  echo "---- 失败样本（不许手工补默认值，见 AGENTS 规则 6）----"
  if compgen -G "$PRIORS_DIR/_failed/*.json" > /dev/null; then
    echo "存在失败样本，数量：$(ls -1 "$PRIORS_DIR/_failed"/*.json | wc -l)"
    echo "先看它们的 error 字段，判断是限流/超时还是 schema 校验失败："
    for f in $(ls -1 "$PRIORS_DIR/_failed"/*.json | head -3); do
      echo "  --- $f"
      $PY -c "import json,sys;d=json.load(open(sys.argv[1],encoding='utf-8'));print('   ',d.get('error','')[:200])" "$f"
    done
    echo "重跑同一条命令即可断点续传（已存在的会被跳过）。"
  else
    echo "没有失败样本。"
  fi
fi

echo
echo "全部完成。下一步：训练"
echo "  bash scripts/run_experiments.sh --core     # 四配置 × 3 seed"
