# run_scheduler.ps1 — launcher for the Kalshi WC bot scheduler (Windows Task Scheduler).
#
# Starts the persistent APScheduler loop in LIVE-ORDER mode. Whether real money is at
# risk is still governed by .env: orders only hit prod when KALSHI_ENV=prod AND
# KALSHI_ALLOW_PROD_ORDERS=1 (L8); otherwise they go to the demo account. The stop-loss
# still halts betting via the risk check inside signal generation, even unattended.
#
# Append-logs to data/logs/scheduler.log so an unattended run leaves a trail. Designed to
# be the -File target of a scheduled task triggered at logon with restart-on-failure.

$ErrorActionPreference = "Stop"

# Repo root is the parent of this script's folder (…\scripts\run_scheduler.ps1 -> repo).
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "venv python not found at $python — create it with: python -m venv .venv"
}

$logDir = Join-Path $root "data\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "scheduler.log"

"$(Get-Date -Format o)  starting scheduler (live-orders)" | Out-File -FilePath $logFile -Append -Encoding utf8
& $python -m scheduler.jobs --live-orders *>> $logFile
