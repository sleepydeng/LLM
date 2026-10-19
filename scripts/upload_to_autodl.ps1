<#
.SYNOPSIS
    把 llm-prior-radiomap 的代码与数据打包上传到远程 GPU 机器（AutoDL）。

.NOTES
    编码要求：本文件必须保存为 **UTF-8 with BOM**。
    Windows PowerShell 5.1 在文件没有 BOM 时会按系统 ANSI（中文机器上是 GBK）解码，
    于是所有中文注释与字符串都会变成乱码，进而报出一堆"缺少右 }""字符串缺少终止符"
    这类与真实原因完全无关的语法错误。若手工编辑过本文件，请重新另存为 UTF-8 with BOM：
        $p = "scripts\upload_to_autodl.ps1"
        Set-Content -LiteralPath $p -Value (Get-Content -LiteralPath $p -Raw -Encoding UTF8) -Encoding UTF8

.DESCRIPTION
    本机（Windows）不能训练，代码与数据要送到 Linux + NVIDIA 机器上跑。
    这里用 Windows 自带的 tar（bsdtar）打包再 scp，并在远程解包后**校验文件数**，
    避免出现"上传不完整 → verify_data 失败 → 误判成代码问题"这种反复。

    注意：本仓库的 data\radiomapseer 很可能是指向旧仓库的 junction（目录联接）。
    tar 处理 junction 的行为不可靠，所以脚本会把 junction 解析成真实路径后再打包。

.EXAMPLE
    # 只打代码（不含数据、不含 venv、不含 runs）
    powershell -File scripts\upload_to_autodl.ps1 -Target distill-lab -What code

    # 只打数据（65.8 MB）
    powershell -File scripts\upload_to_autodl.ps1 -Target distill-lab -What data

    # 两者都传，并指定远程目录
    powershell -File scripts\upload_to_autodl.ps1 -Target root@connect.weste.seetacloud.com `
        -Port 15697 -What both -RemoteDir /root/autodl-tmp/work

    # 只打包不上传（验证打包逻辑 / 看看包多大）
    powershell -File scripts\upload_to_autodl.ps1 -What both -DryRun
#>
[CmdletBinding()]
param(
    [string]$Target = "distill-lab",
    [int]$Port = 0,
    [string]$IdentityFile = "",
    [ValidateSet("code", "data", "both")]
    [string]$What = "code",
    [string]$RemoteDir = "/root/autodl-tmp/work",
    [string]$ProjectName = "llm-prior-radiomap",
    [switch]$DryRun,
    [switch]$SkipVerify
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$StageDir = Join-Path $env:TEMP ("radiomap_upload_" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force -Path $StageDir | Out-Null

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn2($msg) { Write-Host "  [!] $msg" -ForegroundColor Yellow }

$sshOpts = @("-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30")
if ($Port -gt 0) { $sshOpts += @("-p", "$Port") }
if ($IdentityFile) { $sshOpts += @("-i", $IdentityFile) }
$scpOpts = @("-o", "StrictHostKeyChecking=accept-new")
if ($Port -gt 0) { $scpOpts += @("-P", "$Port") }
if ($IdentityFile) { $scpOpts += @("-i", $IdentityFile) }

function Invoke-Remote([string]$cmd) {
    & ssh @sshOpts $Target $cmd
    if ($LASTEXITCODE -ne 0) { throw "远程命令失败（退出码 $LASTEXITCODE）：$cmd" }
}

# ---------------------------------------------------------------- 打包代码
function New-CodePackage {
    Write-Step "打包代码"
    $tarPath = Join-Path $StageDir "code.tgz"
    # 明确的排除清单：数据、虚拟环境、历史运行产物、缓存
    $excludes = @(
        "--exclude=./data/radiomapseer",
        "--exclude=./.venv-local",
        "--exclude=./.venv",
        "--exclude=./runs",
        "--exclude=./__pycache__",
        "--exclude=*/__pycache__",
        "--exclude=./priors",
        "--exclude=./reports/*.png",
        "--exclude=*.pyc",
        "--exclude=./.git"
    )
    Push-Location $ProjectRoot
    try {
        & tar -czf $tarPath @excludes .
        if ($LASTEXITCODE -ne 0) { throw "tar 打包代码失败" }
    } finally { Pop-Location }
    $mb = [math]::Round((Get-Item $tarPath).Length / 1MB, 2)
    Write-Ok "code.tgz 生成完毕（$mb MB）"
    return $tarPath
}

# ------------------------------------------------------------------ 打包数据
function New-DataPackage {
    Write-Step "打包数据（RadioMapSeer）"
    $src = Join-Path $ProjectRoot "data\radiomapseer"
    if (-not (Test-Path -LiteralPath $src)) { throw "找不到数据目录：$src" }

    # junction → 解析成真实路径，避免 tar 递归行为不确定
    $item = Get-Item -LiteralPath $src -Force
    $real = $src
    if ($item.LinkType -and $item.Target) {
        $real = @($item.Target)[0]
        Write-Warn2 "data\radiomapseer 是 $($item.LinkType)，已解析为：$real"
    }
    if (-not (Test-Path -LiteralPath $real)) { throw "解析后的数据路径不存在：$real" }

    $tarPath = Join-Path $StageDir "data.tgz"
    $parent = Split-Path -Parent $real
    $leaf = Split-Path -Leaf $real
    Push-Location $parent
    try {
        # 包内路径写成 data/radiomapseer，解到项目根即可对齐
        & tar -czf $tarPath --exclude="*/.DS_Store" --exclude="*/__MACOSX" `
            -C $parent $leaf
        if ($LASTEXITCODE -ne 0) { throw "tar 打包数据失败" }
    } finally { Pop-Location }

    # 把包内顶层目录名规范成 radiomapseer（已经是了），并统计文件数
    $n = (Get-ChildItem -LiteralPath $real -Recurse -File | Measure-Object).Count
    $mb = [math]::Round((Get-Item $tarPath).Length / 1MB, 2)
    Write-Ok "data.tgz 生成完毕（$mb MB，原始 $n 个文件）"
    return @{ Path = $tarPath; Files = $n; Leaf = $leaf }
}

function Send-Package($localPath, $remoteName, $remoteSubDir) {
    Write-Step "上传 $remoteName → ${Target}:$RemoteDir"
    if ($DryRun) { Write-Warn2 "DryRun：跳过 scp"; return }
    Invoke-Remote "mkdir -p '$RemoteDir/$remoteSubDir'"
    & scp @scpOpts $localPath "${Target}:$RemoteDir/$remoteSubDir/$remoteName"
    if ($LASTEXITCODE -ne 0) { throw "scp 失败：$remoteName" }
    Write-Ok "上传完成 $remoteName"
}

$codeInfo = $null
$dataInfo = $null
if ($What -eq "code" -or $What -eq "both") { $codeInfo = New-CodePackage }
if ($What -eq "data" -or $What -eq "both") { $dataInfo = New-DataPackage }

if ($codeInfo) { Send-Package $codeInfo "code.tgz" "." }

if ($dataInfo) {
    # 数据包解到 <RemoteDir>/<ProjectName>/data/ 下
    Send-Package $dataInfo.Path "data.tgz" "."
}

if ($DryRun) {
    Write-Host "`nDryRun 结束。暂存目录：$StageDir" -ForegroundColor Yellow
    Get-ChildItem $StageDir | Select-Object Name, @{n = "MB"; e = { [math]::Round($_.Length / 1MB, 2) } } | Format-Table -AutoSize
    return
}

# ---------------------------------------------------------------- 远程解包
Write-Step "远程解包"
Invoke-Remote "mkdir -p '$RemoteDir/$ProjectName'"
if ($codeInfo) {
    Invoke-Remote "tar xzf '$RemoteDir/code.tgz' -C '$RemoteDir/$ProjectName' && rm -f '$RemoteDir/code.tgz'"
    Write-Ok "代码已解到 $RemoteDir/$ProjectName"
}
if ($dataInfo) {
    Invoke-Remote "mkdir -p '$RemoteDir/$ProjectName/data' && tar xzf '$RemoteDir/data.tgz' -C '$RemoteDir/$ProjectName/data' && rm -f '$RemoteDir/data.tgz'"
    Write-Ok "数据已解到 $RemoteDir/$ProjectName/data"
}

# ---------------------------------------------------------------- 远程校验
if (-not $SkipVerify) {
    Write-Step "远程校验（文件数 / 环境 / 代码自检）"
    # 校验逻辑放在仓库里的 scripts/remote_verify.sh：避免跨 PowerShell→ssh→bash
    # 三层引号拼字符串，那类错误报出来的是"语法错误"，与真实原因无关。
    # 这里直接调 ssh 而不复用 Invoke-Remote：自检失败是可接受的结果，不该抛异常中断脚本。
    & ssh @sshOpts $Target "cd '$RemoteDir/$ProjectName' && bash scripts/remote_verify.sh"
    if ($LASTEXITCODE -ne 0) {
        Write-Warn2 "远程自检未全部通过（见上面的输出）。如果这次没传数据，属于预期。"
    } else {
        Write-Ok "远程自检全部通过"
    }
}

Write-Host @"

下一步（在远程机器上执行）：
  cd $RemoteDir/$ProjectName
  pip install -r requirements.txt
  bash scripts/remote_verify.sh
  export ZHIPUAI_API_KEY=...
  bash scripts/gen_priors.sh

详细步骤见 RUNBOOK_AutoDL.md
"@ -ForegroundColor Green

Write-Host "本地暂存目录（可删）：$StageDir" -ForegroundColor DarkGray

