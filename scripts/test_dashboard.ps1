param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [switch]$Quick,
    [switch]$CopySummary,
    [switch]$SkipControllerGate,
    [string]$ControllerGateBaselineFull = "",
    [string]$ControllerGateCandidateFull = "artifacts\live_eval\suites\latest_suite.json",
    [string]$ControllerGateBaselineWorst = "",
    [string]$ControllerGateCandidateWorst = "",
    [string]$ControllerGateBaselineTraceDir = "",
    [string]$ControllerGateCandidateTraceDir = "",
    [double]$ControllerGateMinFullDeltaGain = 0.0,
    [double]$ControllerGateMinWorstDeltaGain = 0.0,
    [double]$ControllerGateMaxInterventionRate = 8.0,
    [double]$ControllerGateMaxInterventionRateIncrease = 1.5,
    [int]$ControllerGateTopN = 10,
    [bool]$ControllerGateRequireWorstImprovement = $true
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

    if (Test-Path $LogFile) { Remove-Item -Force $LogFile }
    $errFile = "$LogFile.stderr"
    if (Test-Path $errFile) { Remove-Item -Force $errFile }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $proc = Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c $Command" `
        -NoNewWindow `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $LogFile `
        -RedirectStandardError $errFile
    $exitCode = $proc.ExitCode
    if (Test-Path $errFile) {
        Get-Content -Path $errFile | Add-Content -Path $LogFile
        Remove-Item -Force $errFile
    }
    if (Test-Path $LogFile) {
        Get-Content -Path $LogFile | Out-Host
    }
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

function Find-PreviousSuitePath {
    param(
        [string]$Root,
        [string]$CandidatePath
    )
    $suiteRoot = Join-Path $Root "artifacts\live_eval\suites"
    if (-not (Test-Path $suiteRoot)) {
        return ""
    }
    $candidateFull = ""
    if ($CandidatePath) {
        try {
            $candidateFull = [System.IO.Path]::GetFullPath((Join-Path $Root $CandidatePath))
        } catch {
            $candidateFull = ""
        }
    }
    $suiteFiles = Get-ChildItem -Path $suiteRoot -File -Filter "suite_*.json" | Sort-Object LastWriteTime -Descending
    foreach ($f in $suiteFiles) {
        if (-not $candidateFull) {
            return $f.FullName
        }
        if ($f.FullName -ne $candidateFull) {
            return $f.FullName
        }
    }
    return ""
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
} elseif (-not $SkipControllerGate) {
    $baselineFull = $ControllerGateBaselineFull
    if (-not $baselineFull) {
        $baselineFull = Find-PreviousSuitePath -Root $root -CandidatePath $ControllerGateCandidateFull
    }

    $gatePieces = @()
    $gatePieces += "$Python scripts\controller_candidate_gate.py"
    if ($baselineFull) {
        $gatePieces += "--baseline-full `"$baselineFull`""
    }
    $gatePieces += "--candidate-full `"$ControllerGateCandidateFull`""
    $gatePieces += "--top-n $ControllerGateTopN"
    $gatePieces += "--min-full-delta-gain $ControllerGateMinFullDeltaGain"
    $gatePieces += "--min-worst-delta-gain $ControllerGateMinWorstDeltaGain"
    $gatePieces += "--max-intervention-rate $ControllerGateMaxInterventionRate"
    $gatePieces += "--max-intervention-rate-increase $ControllerGateMaxInterventionRateIncrease"
    if ($ControllerGateRequireWorstImprovement) {
        $gatePieces += "--require-worst-improvement"
    }
    if ($ControllerGateBaselineWorst) {
        $gatePieces += "--baseline-worst `"$ControllerGateBaselineWorst`""
    }
    if ($ControllerGateCandidateWorst) {
        $gatePieces += "--candidate-worst `"$ControllerGateCandidateWorst`""
    }
    if ($ControllerGateBaselineTraceDir) {
        $gatePieces += "--baseline-trace-dir `"$ControllerGateBaselineTraceDir`""
    }
    if ($ControllerGateCandidateTraceDir) {
        $gatePieces += "--candidate-trace-dir `"$ControllerGateCandidateTraceDir`""
    }
    $gatePieces += "--out `"artifacts\live_eval\controller_candidate_gate.json`""
    $gatePieces += "--enforce"

    $steps += @{
        Name = "Controller Candidate Gate"
        Command = ($gatePieces -join " ")
        Log = "06_controller_candidate_gate.log"
    }
}

$results = @()
foreach ($step in $steps) {
    $logFile = Join-Path $runDir $step.Log
    $result = Run-Step -Name $step.Name -Command $step.Command -LogFile $logFile
    $results += $result
}

$failCount = @($results | Where-Object { $_.Status -eq "FAIL" }).Count
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

$controllerGate = $results | Where-Object { $_.Name -eq "Controller Candidate Gate" } | Select-Object -First 1
if ($null -ne $controllerGate) {
    if ($controllerGate.Status -eq "PASS") {
        Write-Host "CONTROLLER CANDIDATE: PASS" -ForegroundColor Green
    } else {
        $gateReason = ""
        if (Test-Path $controllerGate.LogFile) {
            $gateReason = (Get-Content -Path $controllerGate.LogFile | Select-Object -Last 1)
        }
        if (-not $gateReason) {
            $gateReason = "see log for failed checks"
        }
        Write-Host ("CONTROLLER CANDIDATE: REJECT - " + $gateReason) -ForegroundColor Red
    }
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
