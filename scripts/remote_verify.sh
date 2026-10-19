#!/usr/bin/env bash
# 远程环境自检：在 GPU 机器上解包之后跑一次，确认"文件齐、环境对、自检过"。
#
# 为什么单独做成脚本：上传脚本里用字符串拼 ssh 命令 + 引号转义非常脆弱
# （跨越 PowerShell → ssh → bash 三层引号），出错时报的还是"语法错误"这种
# 与真实原因无关的信息。放到一个文件里，就只有 bash 一层引号要管。
#
# 用法（在项目根）：
#   bash scripts/remote_verify.sh
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
echo "项目根: $(pwd)"
FAIL=0

check_file() {
  if [ -e "$1" ]; then echo "  [OK]   $1"; else echo "  [缺失] $1"; FAIL=1; fi
}

echo
echo "---- 关键文件 ----"
for f in requirements.txt AGENTS.md RUNBOOK_AutoDL.md \
         data/splits.json configs/base.yaml \
         configs/prior_none.yaml configs/prior_masked.yaml \
         configs/prior_real.yaml configs/prior_shuffled.yaml \
         src/prior/schema.py src/prior/prompt_v2.py src/prior/generate.py src/prior/audit.py \
         src/data/dataset.py src/data/normalize.py src/data/sectors.py src/data/splits.py \
         src/models/unet.py src/models/film.py src/models/cot_encoder.py src/models/factory.py \
         src/train/train.py src/train/losses.py \
         src/eval/metrics.py src/eval/run_eval.py \
         scripts/verify_data.py scripts/make_splits.py scripts/audit.py \
         scripts/remote_verify.sh scripts/gen_priors.sh scripts/run_experiments.sh \
         scripts/pilot_preflight.py \
         tests/test_local.py tests/test_pipeline.py tests/test_gate.py \
         tests/test_report_integration.py; do
  check_file "$f"
done

echo
echo "---- 数据 ----"
if [ -d data/radiomapseer ]; then
  b=$(ls -1 data/radiomapseer/buildings_complete/*.png 2>/dev/null | wc -l)
  g=$(ls -1 data/radiomapseer/gain/carsDPM/*.png 2>/dev/null | wc -l)
  a=$(ls -1 data/radiomapseer/antenna/*.json 2>/dev/null | wc -l)
  echo "  建筑掩码 $b / gain $g / antenna $a   （期望 50 / 4000 / 50）"
  if [ "$b" -ne 50 ] || [ "$g" -ne 4000 ] || [ "$a" -ne 50 ]; then
    echo "  [警告] 数量与期望不符，先跑 python scripts/verify_data.py --data data/radiomapseer"
    FAIL=1
  fi
else
  echo "  [缺失] data/radiomapseer（这次可能没传数据）"
  FAIL=1
fi

echo
echo "---- 环境 ----"
python - <<'PY' || FAIL=1
import sys
print("  python", sys.version.split()[0])
try:
    import torch
    print("  torch", torch.__version__, "| cuda 可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  GPU:", torch.cuda.get_device_name(0))
    else:
        print("  [警告] CUDA 不可用：正式训练会退化成 CPU（AGENTS 规则 7）")
except Exception as e:
    print("  [错误] torch 不可用:", e); sys.exit(1)
for m in ("numpy", "scipy", "PIL", "yaml", "pydantic"):
    try:
        __import__(m)
    except Exception as e:
        print(f"  [错误] 缺少依赖 {m}: {e}"); sys.exit(1)
print("  依赖: numpy/scipy/PIL/yaml/pydantic 均可用")
PY

echo
echo "---- 代码自检（不需要 GPU / API key）----"
python -m tests.test_local             || { echo "  [失败] tests.test_local"; FAIL=1; }
python -m tests.test_pipeline          || { echo "  [失败] tests.test_pipeline"; FAIL=1; }
python -m tests.test_gate              || { echo "  [失败] tests.test_gate"; FAIL=1; }
python -m tests.test_report_integration || { echo "  [失败] tests.test_report_integration"; FAIL=1; }

echo
if [ "$FAIL" -eq 0 ]; then
  echo "[OK] 远程环境自检全部通过。下一步："
  echo "  export ZHIPUAI_API_KEY=...   # 然后 bash scripts/gen_priors.sh"
else
  echo "[FAIL] 有检查项未通过，先解决再开工。"
fi
exit "$FAIL"
