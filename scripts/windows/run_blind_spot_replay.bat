@echo off
setlocal

set "ROOT=%~dp0..\..\"
cd /d "%ROOT%"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "SCRIPT_JSON=%ROOT%scripts\blind_spot_replay.py"
set "SCRIPT_HTML=%ROOT%scripts\blind_spot_replay_view.py"
set "TRACE_ROOT=%ROOT%artifacts\live_eval\focused_traces"
set "OUT_JSON=%ROOT%artifacts\live_eval\blind_spot_replay_latest.json"
set "OUT_HTML=%ROOT%artifacts\live_eval\blind_spot_replay_latest.html"

if not exist "%PYTHON%" (
  echo Python not found at "%PYTHON%".
  exit /b 1
)
if not exist "%SCRIPT_JSON%" (
  echo Script not found at "%SCRIPT_JSON%".
  exit /b 1
)
if not exist "%SCRIPT_HTML%" (
  echo Script not found at "%SCRIPT_HTML%".
  exit /b 1
)

echo Building blind-spot replay JSON...
"%PYTHON%" "%SCRIPT_JSON%" --trace-root "%TRACE_ROOT%" --latest-only --min-confidence 0.7 --max-steps-to-death 10 --replay-window 30 --max-spots 50 --out "%OUT_JSON%"
if errorlevel 1 exit /b 1

echo Building blind-spot replay HTML...
"%PYTHON%" "%SCRIPT_HTML%" --input "%OUT_JSON%" --out "%OUT_HTML%"
if errorlevel 1 exit /b 1

if exist "%OUT_HTML%" (
  echo Opening "%OUT_HTML%" ...
  start "" "%OUT_HTML%"
) else (
  echo Expected output not found: "%OUT_HTML%"
  exit /b 1
)

endlocal
