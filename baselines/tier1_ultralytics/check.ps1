# Stage 1 overnight orchestrator progress check.
# Usage:
#   cd C:\twb\oil_leak\Paper3\paper3-10
#   .\baselines\tier1_ultralytics\check.ps1

$ErrorActionPreference = "Stop"
$root   = "C:\twb\oil_leak\Paper3\paper3-10"
$status = Join-Path $root "results_v2\baselines\tier1\STATUS.json"
$report = Join-Path $root "results_v2\baselines\tier1\FINAL_REPORT.md"
$csv    = Join-Path $root "results_v2\paper_materials\tables\tier1_comparison.csv"

if (Test-Path $report) {
    Write-Host "===== Training COMPLETE =====" -ForegroundColor Green
    Get-Content $report
    Write-Host ""
    if (Test-Path $csv) {
        Write-Host "===== tier1_comparison.csv =====" -ForegroundColor Cyan
        Get-Content $csv
    } else {
        Write-Host "(tier1_comparison.csv not generated)" -ForegroundColor Yellow
    }
    exit 0
}

if (Test-Path $status) {
    $s = Get-Content $status -Raw | ConvertFrom-Json
    Write-Host "===== Training IN PROGRESS =====" -ForegroundColor Yellow
    Write-Host "Started:    $($s.started_at)"
    Write-Host "Phase:      $($s.phase)"
    Write-Host "Current:    $($s.current)"
    Write-Host "Elapsed:    $($s.elapsed_h) h"
    Write-Host "Budget:     $($s.total_timeout_h) h"
    Write-Host ""
    Write-Host "Completed:  $($s.completed.Count) baseline(s)"
    foreach ($c in $s.completed) {
        $mAP = if ($c.mAP50 -ne $null) { $c.mAP50 } else { "N/A" }
        $dur = if ($c.duration_h -ne $null) { "$($c.duration_h)h" } else { "?" }
        Write-Host ("  - {0,-10} status={1,-12} mAP50={2,-8} dur={3}" -f $c.model, $c.status, $mAP, $dur)
    }
    Write-Host ""
    if ($s.pending) {
        Write-Host "Pending:    $($s.pending -join ', ')"
    }
    Write-Host ""

    $pid_ = $s.pid
    $alive = $false
    if ($pid_) {
        try {
            $proc = Get-Process -Id $pid_ -ErrorAction Stop
            $alive = $true
            Write-Host ("PID {0} alive (process: {1})" -f $pid_, $proc.ProcessName) -ForegroundColor Green
        } catch {
            Write-Host ("PID {0} NOT running" -f $pid_) -ForegroundColor Red
            Write-Host "(Orchestrator may have crashed; check results_v2\baselines\tier1\orchestrator.log)" -ForegroundColor Red
        }
    }

    # Tail orchestrator.log
    $log = Join-Path $root "results_v2\baselines\tier1\orchestrator.log"
    if (Test-Path $log) {
        Write-Host ""
        Write-Host "===== Last 15 log lines =====" -ForegroundColor Cyan
        Get-Content $log -Tail 15
    }
    exit 0
}

Write-Host "STATUS.json not found at $status" -ForegroundColor Red
Write-Host "Orchestrator was never launched, or was killed before writing status." -ForegroundColor Red
$launch = Join-Path $root "results_v2\baselines\tier1\LAUNCH_FAIL.md"
if (Test-Path $launch) {
    Write-Host ""
    Write-Host "===== LAUNCH_FAIL.md =====" -ForegroundColor Red
    Get-Content $launch
}
exit 1
