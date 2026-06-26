# run_scheduler.ps1 — launcher for the Kalshi WC bot (Windows Task Scheduler).
#
# Runs ONE full cycle in LIVE-ORDER mode (refresh -> signals/place orders -> settle ->
# bankroll) and exits. Windows Task Scheduler is the clock: trigger this daily at midnight
# for one bet-placement pass per day. (To run the continuous in-process loop instead,
# drop the --once flag below.)
#
# Whether real money is at risk is still governed by .env: orders only hit prod when
# KALSHI_ENV=prod AND KALSHI_ALLOW_PROD_ORDERS=1 (L8); otherwise they go to the demo
# account. The stop-loss still halts betting via the risk check in signal generation.
#
# Append-logs to data/logs/scheduler.log so an unattended run leaves a trail.

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

"$(Get-Date -Format o)  starting daily cycle (once, live-orders)" | Out-File -FilePath $logFile -Append -Encoding utf8
& $python -m scheduler.jobs --once --live-orders *>> $logFile
