@echo off
setlocal

set "ROOT=%~dp0..\..\"
cd /d "%ROOT%"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "SCRIPT_REPORT=%ROOT%scripts\agent_performance\build_agent_performance_report.py"
set "SCRIPT_VISUALS=%ROOT%scripts\agent_performance\build_agent_performance_visuals.py"
set "SCRIPT_HUB=%ROOT%scripts\reporting\build_reports_hub.py"
set "DASH_HTML=%ROOT%artifacts\agent_performance\agent_performance_dashboard_latest.html"
set "HUB_TXT=%ROOT%artifacts\reports\reports_hub_latest.txt"

if not exist "%PYTHON%" (
  echo Python not found at "%PYTHON%".
  exit /b 1
)
if not exist "%SCRIPT_REPORT%" (
  echo Script not found at "%SCRIPT_REPORT%".
  exit /b 1
)
if not exist "%SCRIPT_VISUALS%" (
  echo Script not found at "%SCRIPT_VISUALS%".
  exit /b 1
)
if not exist "%SCRIPT_HUB%" (
  echo Script not found at "%SCRIPT_HUB%".
  exit /b 1
)

echo Building agent-performance report...
"%PYTHON%" "%SCRIPT_REPORT%" --out-dir artifacts\agent_performance --tag latest %*
if errorlevel 1 exit /b 1
echo Building agent-performance dashboard...
"%PYTHON%" "%SCRIPT_VISUALS%" --in-dir artifacts\agent_performance --out-dir artifacts\agent_performance --tag latest
if errorlevel 1 exit /b 1
echo Building reports hub...
"%PYTHON%" "%SCRIPT_HUB%" --artifacts-root artifacts --out-dir artifacts\reports
if errorlevel 1 exit /b 1

echo.
echo Done. Outputs:
echo   artifacts\agent_performance\agent_performance_latest.json
echo   artifacts\agent_performance\agent_performance_latest.md
echo   artifacts\agent_performance\agent_performance_rows_latest.csv
echo   artifacts\agent_performance\agent_performance_dashboard_latest.html
echo   artifacts\reports\reports_hub_latest.md
echo   artifacts\reports\reports_hub_latest.txt

if exist "%DASH_HTML%" (
  start "" "%DASH_HTML%"
)
if exist "%HUB_TXT%" (
  start "" notepad.exe "%HUB_TXT%"
)

endlocal
