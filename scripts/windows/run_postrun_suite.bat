@echo off
setlocal

set "ROOT=%~dp0..\..\"
cd /d "%ROOT%"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "SCRIPT=%ROOT%scripts\post_run_suite.py"
set "ARTIFACT_DIR=state\ppo\baseline"

if not "%~1"=="" set "ARTIFACT_DIR=%~1"

if not exist "%PYTHON%" (
  echo Python not found at "%PYTHON%".
  exit /b 1
)
if not exist "%SCRIPT%" (
  echo Script not found at "%SCRIPT%".
  exit /b 1
)

echo Collecting post-run diagnostics...
"%PYTHON%" "%SCRIPT%" --artifact-dir "%ARTIFACT_DIR%" --artifacts-root artifacts --out-dir artifacts\share --print-summary
if errorlevel 1 exit /b 1

echo.
echo Done. Share:
echo   artifacts\share\diagnostics_bundle.json
echo   artifacts\share\diagnostics_bundle.md

endlocal
