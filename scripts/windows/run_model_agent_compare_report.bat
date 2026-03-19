@echo off
setlocal EnableDelayedExpansion

set "ROOT=%~dp0..\..\"
cd /d "%ROOT%"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "SCRIPT_REPORT=%ROOT%scripts\phase3_compare\build_model_agent_compare_report.py"
set "SCRIPT_VISUALS=%ROOT%scripts\phase3_compare\build_model_agent_compare_visuals.py"
set "SCRIPT_HUB=%ROOT%scripts\reporting\build_reports_hub.py"
set "DASH_HTML=%ROOT%artifacts\phase3_compare\model_agent_compare_dashboard_latest.html"
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

set "LEFT_EXP=%~1"
set "RIGHT_EXP=%~2"

if "%LEFT_EXP%"=="" goto :pick_experiments
if "%RIGHT_EXP%"=="" goto :pick_experiments
goto :validate_inputs

:pick_experiments
echo.
echo Available experiment folders:
set "EXP_COUNT=0"
for /d %%D in ("%ROOT%state\ppo\*") do (
  set /a EXP_COUNT+=1
  set "EXP_!EXP_COUNT!=%%~nxD"
  echo   [!EXP_COUNT!] %%~nxD
)
if %EXP_COUNT% LEQ 0 (
  echo No experiment folders found under state\ppo.
  exit /b 1
)
echo.
if "%LEFT_EXP%"=="" (
  set /p LEFT_IDX=Pick LEFT experiment number: 
  for %%I in (%LEFT_IDX%) do set "LEFT_EXP=!EXP_%%I!"
)
if "%RIGHT_EXP%"=="" (
  set /p RIGHT_IDX=Pick RIGHT experiment number: 
  for %%I in (%RIGHT_IDX%) do set "RIGHT_EXP=!EXP_%%I!"
)

:validate_inputs
if "%LEFT_EXP%"=="" (
  echo Left experiment is invalid or empty.
  exit /b 1
)
if "%RIGHT_EXP%"=="" (
  echo Right experiment is invalid or empty.
  exit /b 1
)
if not exist "%ROOT%state\ppo\%LEFT_EXP%" (
  echo Left experiment not found: %LEFT_EXP%
  exit /b 1
)
if not exist "%ROOT%state\ppo\%RIGHT_EXP%" (
  echo Right experiment not found: %RIGHT_EXP%
  exit /b 1
)

echo Building model+agent compare report...
"%PYTHON%" "%SCRIPT_REPORT%" --left-exp "%LEFT_EXP%" --right-exp "%RIGHT_EXP%" --out-dir artifacts\phase3_compare --tag latest
if errorlevel 1 exit /b 1
echo Building model+agent compare dashboard...
"%PYTHON%" "%SCRIPT_VISUALS%" --in-dir artifacts\phase3_compare --out-dir artifacts\phase3_compare --tag latest
if errorlevel 1 exit /b 1
echo Building reports hub...
"%PYTHON%" "%SCRIPT_HUB%" --artifacts-root artifacts --out-dir artifacts\reports
if errorlevel 1 exit /b 1

echo.
echo Done. Outputs:
echo   artifacts\phase3_compare\model_agent_compare_latest.json
echo   artifacts\phase3_compare\model_agent_compare_latest.md
echo   artifacts\phase3_compare\model_agent_compare_rows_latest.csv
echo   artifacts\phase3_compare\model_agent_compare_dashboard_latest.html
echo   artifacts\reports\reports_hub_latest.md
echo   artifacts\reports\reports_hub_latest.txt

if exist "%DASH_HTML%" (
  start "" "%DASH_HTML%"
)
if exist "%HUB_TXT%" (
  start "" notepad.exe "%HUB_TXT%"
)

endlocal
