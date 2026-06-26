# status.ps1 — one-shot health check for the Kalshi WC bot loop.
#
# Prints the scheduled task's state (Running = healthy for the persistent loop), its last
# run time/result and next run, then tails the recent scheduler log. Read-only; safe to
# run any time. Usage:  .\scripts\status.ps1   (optionally -Lines 50 to show more log)

param(
    [int]$Lines = 20,
    [string]$TaskName = "KalshiWCBotLoop"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "=== Scheduled task: $TaskName ===" -ForegroundColor Cyan
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -eq $task) {
    Write-Host "Task not registered. Register it (elevated) before it can run." -ForegroundColor Yellow
}
else {
    $info = $task | Get-ScheduledTaskInfo
    # 0 = last run OK; 267009 (0x41301) = currently running (the healthy steady state).
    $result = switch ($info.LastTaskResult) {
        0 { "0 (OK)" }
        267009 { "267009 (running)" }
        default { "$($info.LastTaskResult)" }
    }
    [PSCustomObject]@{
        State        = $task.State
        LastRunTime  = $info.LastRunTime
        LastResult   = $result
        NextRunTime  = $info.NextRunTime
        MissedRuns   = $info.NumberOfMissedRuns
    } | Format-List

    # Confirm the python worker is actually alive (the task can show Running while the
    # child process has died and is mid-restart).
    $py = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*scheduler.jobs*" }
    if ($py) {
        Write-Host ("python worker: ALIVE (PID {0})" -f ($py.ProcessId -join ", ")) -ForegroundColor Green
    }
    else {
        Write-Host "python worker: not found (idle between restarts, or stopped)" -ForegroundColor Yellow
    }
}

Write-Host "`n=== scheduler.log (last $Lines lines) ===" -ForegroundColor Cyan
$log = Join-Path $root "data\logs\scheduler.log"
if (Test-Path $log) {
    Get-Content $log -Tail $Lines
}
else {
    Write-Host "No log yet at $log (the loop hasn't written one)." -ForegroundColor Yellow
}
