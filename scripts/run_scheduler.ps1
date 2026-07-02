# run_scheduler.ps1 — launcher for the Kalshi WC bot (Windows Task Scheduler).
#
# Runs the CONTINUOUS in-process loop in DRY-RUN mode (2026-07-01, Devon-approved): signals
# are generated and logged to the DB but NO orders are placed while the market-anchored
# model rework lands and validates. To resume live orders, add "--live-orders" back to the
# ArgumentList below and restart the task. APScheduler fires the jobs on their cadences
# (signals every 3h, settle every 2h, sync/bankroll every 30m) until the process is stopped. Registered to start at boot (-AtStartup) and auto-restart on crash,
# so it comes back by itself after a reboot. The host must NOT sleep (a sleeping PC freezes
# the loop) — keep sleep disabled. (For one pass per run instead, add --once below and use
# a timed trigger.)
#
# Whether real money is at risk is still governed by .env: orders only hit prod when
# KALSHI_ENV=prod AND KALSHI_ALLOW_PROD_ORDERS=1 (L8); otherwise they go to the demo
# account. The stop-loss still halts betting via the risk check in signal generation, and
# the no-re-bet guard means a held market is never topped up across cycles.
#
# Append-logs to data/logs/scheduler.log so an unattended run leaves a trail.

$ErrorActionPreference = "Stop"

# Repo root — use $PSScriptRoot (set by powershell.exe -File) with a hardcoded fallback
# so Task Scheduler's S4U context can never produce an empty root.
if ($PSScriptRoot) {
    $root = Split-Path -Parent $PSScriptRoot
} else {
    Write-Error "PSScriptRoot is empty. Run this script via: powershell.exe -File path\to\scripts\run_scheduler.ps1"
    exit 1
}
Set-Location $root

# Write log FIRST — every run leaves a trace even if python resolution fails later.
$logDir = Join-Path $root "data\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "scheduler.log"
"$(Get-Date -Format o)  run_scheduler.ps1 started (root=$root)" | Out-File -FilePath $logFile -Append -Encoding utf8

# Resolve python: prefer the local venv (always present), then the py launcher fallback.
# PS 5.1-compatible — no ?. null-conditional operator.
$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $venvPy) {
    $python = $venvPy
} else {
    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        $python = $pyCmd.Source
    } else {
        "$(Get-Date -Format o)  ERROR: python not found at $venvPy and py.exe not on PATH" | Out-File -FilePath $logFile -Append -Encoding utf8
        exit 1
    }
}
"$(Get-Date -Format o)  python=$python" | Out-File -FilePath $logFile -Append -Encoding utf8

"$(Get-Date -Format o)  starting persistent loop (DRY-RUN — no orders placed)" | Out-File -FilePath $logFile -Append -Encoding utf8

# Start-Process redirects at the OS level — captures output even if Python dies before
# writing a single byte (unlike the PowerShell pipeline which requires bytes to flow).
$stdoutLog = Join-Path $logDir "scheduler_stdout.log"
$stderrLog = Join-Path $logDir "scheduler_stderr.log"
$proc = Start-Process -FilePath $python `
    -ArgumentList "-m", "scheduler.jobs" `
    -WorkingDirectory $root `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError  $stderrLog `
    -PassThru -NoNewWindow
$proc.WaitForExit()
$code = $proc.ExitCode

# Merge stdout + stderr into the main log for a single trail.
if (Test-Path $stdoutLog) { Get-Content $stdoutLog | Out-File -FilePath $logFile -Append -Encoding utf8 }
if (Test-Path $stderrLog) { Get-Content $stderrLog | Out-File -FilePath $logFile -Append -Encoding utf8 }
"$(Get-Date -Format o)  scheduler.jobs exited (code $code)" | Out-File -FilePath $logFile -Append -Encoding utf8
