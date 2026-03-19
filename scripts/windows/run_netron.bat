@echo off
setlocal

set "ROOT=%~dp0..\..\"
cd /d "%ROOT%"
set "NETRON=%ROOT%.venv\Scripts\netron.exe"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "EXPORTER=%ROOT%scripts\export_policy_trace.py"
set "MODEL=%ROOT%state\ppo\baseline\best_score_model.zip"
set "MODEL_TO_OPEN=%MODEL%"
set "TRACE_OUT=%ROOT%artifacts\netron\policy_trace.pt"

if not "%~1"=="" set "MODEL=%~1"
set "MODEL_TO_OPEN=%MODEL%"
if "%~1"=="" if not exist "%MODEL%" set "MODEL=%ROOT%state\ppo\baseline\best_model.zip"
if "%~1"=="" set "MODEL_TO_OPEN=%MODEL%"

if not exist "%NETRON%" (
  echo Netron not found at "%NETRON%".
  echo Install with: .\.venv\Scripts\python -m pip install netron
  exit /b 1
)

if not exist "%MODEL%" (
  echo Model file not found: "%MODEL%"
  echo Usage: run_netron.bat [model_path]
  exit /b 1
)

if /I "%~x1"==".zip" goto EXPORT
if "%~1"=="" goto EXPORT
goto OPEN

:EXPORT
if not exist "%PYTHON%" (
  echo Python not found at "%PYTHON%".
  exit /b 1
)
if not exist "%EXPORTER%" (
  echo Exporter not found at "%EXPORTER%".
  exit /b 1
)
echo Exporting policy graph for Netron...
"%PYTHON%" "%EXPORTER%" --model "%MODEL%" --out "%TRACE_OUT%"
if errorlevel 1 exit /b 1
set "MODEL_TO_OPEN=%TRACE_OUT%"

:OPEN
"%NETRON%" "%MODEL_TO_OPEN%" -b

endlocal
