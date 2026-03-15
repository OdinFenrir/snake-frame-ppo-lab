param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [switch]$Quick,
    [switch]$CopySummary
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Text)
    Write-Host ""
    Write-Host ("=" * 88) -ForegroundColor DarkGray
    Write-Host ("  " + $Text) -ForegroundColor Cyan
    Write-Host ("=" * 88) -ForegroundColor DarkGray
}

function Write-Status {
    param(
        [string]$Label,
        [string]$Status
    )
    $color = "White"
    if ($Status -eq "PASS") { $color = "Green" }
    elseif ($Status -eq "FAIL") { $color = "Red" }
    elseif ($Status -eq "SKIP") { $color = "Yellow" }
    Write-Host ("[{0}] {1}" -f $Status, $Label) -ForegroundColor $color
}

function Run-Step {
    param(
        [string]$Name,
        [string]$Command,
        [string]$LogFile
    )
    Write-Section $Name
    Write-Host ("Command: " + $Command) -ForegroundColor DarkCyan
    Write-Host ("Log:     " + $LogFile) -ForegroundColor DarkGray

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & cmd /c $Command 2>&1 | Tee-Object -FilePath $LogFile | Out-Host
    $exitCode = $LASTEXITCODE
    $sw.Stop()

    $status = if ($exitCode -eq 0) { "PASS" } else { "FAIL" }
    Write-Status -Label ("{0} ({1:N2}s)" -f $Name, $sw.Elapsed.TotalSeconds) -Status $status
    return [PSCustomObject]@{
        Name = $Name
        Status = $status
        Seconds = [math]::Round($sw.Elapsed.TotalSeconds, 2)
        ExitCode = $exitCode
        LogFile = $LogFile
        Command = $Command
    }
}

if (-not (Test-Path $Python)) {
    throw "Python interpreter not found at $Python"
}

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$runDir = Join-Path $root "artifacts\test_dashboard\latest"
$latestDir = Join-Path $root "artifacts\test_dashboard\latest"
if (Test-Path $latestDir) {
    Remove-Item -Recurse -Force $latestDir
}
$null = New-Item -ItemType Directory -Path $latestDir -Force

$steps = @(
    @{
        Name = "Lint (ruff)"
        Command = "$Python -m ruff check snake_frame tests main.py scripts\validate_determinism.py"
        Log = "01_lint.log"
    },
    @{
        Name = "Tests (pytest -m not render)"
        Command = "$Python -m pytest -q -m ""not render"""
        Log = "02_pytest_not_render.log"
    },
    @{
        Name = "Render Regression (pytest -m render)"
        Command = "$Python -m pytest -q -m render"
        Log = "03_pytest_render.log"
    },
    @{
        Name = "E2E Smoke + Perf Budgets (Median of 3)"
        Command = "$Python scripts\smoke_gate_median.py --runs 3 --train-steps 2048 --game-steps 300 --max-frame-p95-ms 40 --max-frame-avg-ms 34 --max-frame-jitter-ms 8 --max-inference-p95-ms 12 --min-training-steps-per-sec 250 --metrics-out artifacts/smoke_metrics.json"
        Log = "04_smoke_perf.log"
    },
    @{
        Name = "Deterministic Drift Check"
        Command = "$Python scripts\validate_determinism.py --baseline tests\baselines\deterministic_windows.json --seed 1337 --train-steps 2048 --game-steps 300"
        Log = "05_deterministic.log"
    }
)

if ($Quick) {
    $steps = $steps | Where-Object { $_.Name -in @("Lint (ruff)", "Tests (pytest -m not render)") }
    Write-Host "Quick mode enabled: running lint + core tests only." -ForegroundColor Yellow
}

$results = @()
foreach ($step in $steps) {
    $logFile = Join-Path $runDir $step.Log
    $result = Run-Step -Name $step.Name -Command $step.Command -LogFile $logFile
    $results += $result
}

$failCount = ($results | Where-Object { $_.Status -eq "FAIL" }).Count
$totalSeconds = ($results | Measure-Object -Property Seconds -Sum).Sum

$summaryLines = @()
$summaryLines += "# Test Dashboard Summary"
$summaryLines += ""
$summaryLines += ("- Timestamp: " + (Get-Date).ToString("u"))
$summaryLines += ("- Root: " + $root)
$summaryLines += ("- Python: " + $Python)
$summaryLines += ("- Mode: " + ($(if ($Quick) { "quick" } else { "full" })))
$summaryLines += ("- Total Duration (sum of steps): " + ([math]::Round($totalSeconds, 2)) + "s")
$summaryLines += ("- Overall: " + ($(if ($failCount -eq 0) { "PASS" } else { "FAIL" })))
$summaryLines += ""
$summaryLines += "| Step | Status | Seconds | Log |"
$summaryLines += "|---|---:|---:|---|"
foreach ($r in $results) {
    $summaryLines += ("| {0} | {1} | {2} | `{3}` |" -f $r.Name, $r.Status, $r.Seconds, $r.LogFile)
}

$summaryPath = Join-Path $latestDir "summary.md"
$summaryLines -join [Environment]::NewLine | Set-Content -Path $summaryPath -Encoding UTF8

$jsonPath = Join-Path $latestDir "summary.json"
$results | ConvertTo-Json -Depth 3 | Set-Content -Path $jsonPath -Encoding UTF8

# Cleanup legacy timestamped folders from older dashboard versions.
$dashboardRoot = Join-Path $root "artifacts\test_dashboard"
if (Test-Path $dashboardRoot) {
    Get-ChildItem -Path $dashboardRoot -Directory | Where-Object { $_.Name -ne "latest" } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Section "Final Summary"
foreach ($r in $results) {
    Write-Status -Label ("{0} ({1:N2}s)" -f $r.Name, $r.Seconds) -Status $r.Status
}
Write-Host ""
Write-Host ("Summary (markdown): " + $summaryPath) -ForegroundColor Cyan
Write-Host ("Summary (json):     " + $jsonPath) -ForegroundColor Cyan
Write-Host ("Latest alias dir:   " + $latestDir) -ForegroundColor Cyan

if ($CopySummary) {
    Get-Content -Raw $summaryPath | Set-Clipboard
    Write-Host "Copied summary.md content to clipboard." -ForegroundColor Green
}

if ($failCount -gt 0) {
    exit 1
}
exit 0
