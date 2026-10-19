#!/bin/bash
# 远程训练进度速查（在远端执行；由 scripts/check_remote_progress.ps1 送过来）
# 只读，不改动任何东西。
cd /root/autodl-tmp/work/llm-prior-radiomap || exit 1
echo "=== remote time: $(date -Is) ==="
echo
echo "--- audit / priors ---"
grep -q "audit PASSED" reports/core_pair.log 2>/dev/null && echo "  audit: PASSED"
grep -o "hard_pass=[A-Za-z]*" reports/audit_run2.log 2>/dev/null | tail -1 | sed "s/^/  /"
echo "  priors: $(ls priors/prompt_v3/glm-4v-flash/*.json 2>/dev/null | wc -l) files, $(ls priors/prompt_v3/glm-4v-flash/_failed/*.json 2>/dev/null | wc -l) failed"
echo
echo "--- core pair: epochs done / speed / ETA ---"
for f in reports/train_real_v3_seed0.log reports/train_masked_em_seed0.log; do
  name=$(basename "$f" .log)
  n=$(grep -c "epoch " "$f" 2>/dev/null)
  if [ "${n:-0}" -gt 1 ]; then
    t_first=$(grep "epoch " "$f" | head -1 | cut -d" " -f1-2)
    t_last=$(grep "epoch " "$f" | tail -1 | cut -d" " -f1-2)
    e1=$(date -d "$t_first" +%s 2>/dev/null)
    e2=$(date -d "$t_last" +%s 2>/dev/null)
    if [ -n "$e1" ] && [ -n "$e2" ]; then
      # 真实速度 = 首末 epoch 的时间差 / 间隔数（用 now-last 会被暂停/重启污染）
      sppe=$(( (e2 - e1) / (n - 1) ))
      eta=$(( (50 - n) * sppe / 60 ))
      printf "  %-26s %2d/50 ep   %3ds/ep   ~%3d min left\n" "$name" "$n" "$sppe" "$eta"
    else
      printf "  %-26s %2d/50 ep\n" "$name" "$n"
    fi
    grep "epoch " "$f" | tail -1 | sed "s/^/      /"
  else
    printf "  %-26s (%s epoch lines)\n" "$name" "${n:-0}"
  fi
done
echo
echo "--- results / artifacts ---"
for d in prior_real_v3_seed0 prior_v3_masked_em_seed0; do
  printf "  %-26s ckpt=%s metrics_test=%s\n" "$d" \
    "$([ -f runs/$d/ckpt/best.pth ] && echo YES || echo no)" \
    "$([ -f runs/$d/metrics_test.json ] && echo YES || echo no)"
done
echo "  reports/summary.md: $([ -f reports/summary.md ] && echo YES || echo no)"
echo
echo "--- gpu / procs ---"
echo "  trainers running: $(pgrep -c -f src.train.train 2>/dev/null || echo 0)"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed "s/^/  gpu: /"
echo "--- runs on disk ---"
ls -1 runs/ 2>/dev/null | sed "s/^/  /"
