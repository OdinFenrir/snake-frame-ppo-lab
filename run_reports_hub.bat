@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=.venv\Scripts\python.exe"
set "SCRIPT=scripts\reporting\build_reports_hub.py"
set "HUB_MD=artifacts\reports\reports_hub_latest.md"
set "HUB_TXT=artifacts\reports\reports_hub_latest.txt"
set "DASH_HTML=artifacts\training_input\training_input_dashboard_latest.html"

if not exist "%PYTHON%" (
  echo Python not found at "%PYTHON%".
  goto :end
)
if not exist "%SCRIPT%" (
  echo Script not found at "%SCRIPT%".
  goto :end
)

echo Building reports hub...
"%PYTHON%" "%SCRIPT%" --artifacts-root artifacts --out-dir artifacts\reports
if errorlevel 1 (
  echo Failed to build reports hub.
  goto :end
)

echo.
echo Done:
echo   %DASH_HTML%
echo   %HUB_MD%
echo   %HUB_TXT%

if exist "%DASH_HTML%" (
  start "" "%DASH_HTML%"
)

:end
if /I "%~1"=="--no-pause" goto :eof
pause
endlocal
