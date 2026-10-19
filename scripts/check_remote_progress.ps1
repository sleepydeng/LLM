# 远程训练进度速查（本机 PowerShell）
#   powershell -ExecutionPolicy Bypass -File scripts\check_remote_progress.ps1
#
# 实现说明（都是踩过的坑）：
#   * 远端脚本放在同目录的 remote_progress.sh 里，本文件**只负责把它的字节原样送过去**。
#     曾在 .ps1 里内联 here-string，结果行尾 CRLF 让远端每行多一个 \r，脚本静默不执行。
#   * 用 [IO.File]::ReadAllBytes + UTF8.GetString 读，避免 PowerShell 按 GBK 误解码。
#   * **不要**给 ssh 加 -n：带上它远端把 stdin 当 /dev/null，什么都不执行。
$ErrorActionPreference = "Continue"
$target = "root@connect.bjb2.seetacloud.com"
$port   = 11448
$here   = Split-Path -Parent $MyInvocation.MyCommand.Path
$shPath = Join-Path $here "remote_progress.sh"

if (-not (Test-Path $shPath)) { Write-Host "missing $shPath" -ForegroundColor Red; exit 2 }

Write-Host "Querying remote progress: $target (port $port)" -ForegroundColor Cyan
Write-Host ""

$script = [System.Text.Encoding]::UTF8.GetString([System.IO.File]::ReadAllBytes($shPath))
$script | ssh -p $port -o StrictHostKeyChecking=no $target "bash -s"

Write-Host ""
Write-Host "(remote clock is UTC+8)" -ForegroundColor DarkGray
