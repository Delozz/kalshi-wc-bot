# restart_scheduler.ps1 — manual restart for the Kalshi WC bot scheduler.
#
# MUST be run in an ELEVATED PowerShell: the KalshiWCBotLoop task runs under an elevated S4U
# principal, so its python children can only be stopped from an elevated shell.
#
# Why a kill step is needed: Stop-ScheduledTask terminates the task's launcher (powershell.exe)
# but NOT the python loop it spawned — the loop reparents and keeps running, holding the
# single-instance lock and the .env it started with, which would block (or duplicate) a clean
# start. So the cycle is:  stop task -> kill orphaned venv python -> start task -> verify.
#
# Targets ONLY python running from this repo's .venv (by executable path), so any other Python
# process on the machine is never touched.

$ErrorActionPreference = "Stop"

# Repo root = parent of this script's folder (hardcoded fallback for odd host contexts).
if ($PSScriptRoot) {
    $Bot = Split-Path -Parent $PSScriptRoot
} else {
    $Bot = "C:\Users\mario\OneDrive\Documents\World Cup Kalshi Bot"
}
$TaskName = "KalshiWCBotLoop"
$StderrLog = Join-Path $Bot "data\logs\scheduler_stderr.log"

# Refuse to run unelevated — the kill step would fail with Access Denied and leave the orphan
# alive, producing a blocked or duplicate start.
$isAdmin = (
    [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
if (-not $isAdmin) {
    Write-Warning "Not elevated. Re-run in an ADMIN PowerShell, or the orphan kill will fail."
    return
}

function Get-BotPython {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.ExecutablePath -like "$Bot\.venv\*" }
}

# 1. Stop the task (kills the launcher powershell, not its python children).
Write-Host "Stopping task $TaskName ..."
Stop-ScheduledTask -TaskName $TaskName | Out-Null

# 2. Kill the orphaned loop — only this bot's venv python.
$orphans = Get-BotPython
if ($orphans) {
    Write-Host ("Killing orphaned loop python: " + (($orphans.ProcessId) -join ", "))
    $orphans | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
} else {
    Write-Host "No orphaned loop python found."
}
Start-Sleep -Seconds 2

# 3. Start fresh — loads the current .env (incl. STOP_LOSS_THRESHOLD).
Write-Host "Starting task ..."
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 7

# 4. Verify.
$state = (Get-ScheduledTask -TaskName $TaskName).State
$py = Get-BotPython
Write-Host ""
Write-Host "task state : $state"
Write-Host ("python procs: " + (($py.ProcessId) -join ", ") + "  (expect 2 = single instance)")
Write-Host "--- last log lines (expect 'Scheduler started', env=prod) ---"
if (Test-Path $StderrLog) { Get-Content $StderrLog -Tail 6 } else { Write-Host "(no stderr log yet)" }
