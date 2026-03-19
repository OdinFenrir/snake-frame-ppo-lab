@echo off
setlocal

set "ROOT=%~dp0..\..\"
cd /d "%ROOT%"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "SCRIPT=%ROOT%scripts\view_policy_3d.py"
set "MODEL=%ROOT%state\ppo\baseline\best_score_model.zip"
set "OUT=%ROOT%artifacts\share\policy_3d_latest.html"

if not "%~1"=="" set "MODEL=%~1"
if "%~1"=="" if not exist "%MODEL%" set "MODEL=%ROOT%state\ppo\baseline\best_model.zip"

if not exist "%PYTHON%" (
  echo Python not found at "%PYTHON%".
  exit /b 1
)
if not exist "%SCRIPT%" (
  echo Script not found at "%SCRIPT%".
  exit /b 1
)
if not exist "%MODEL%" (
  echo Model file not found: "%MODEL%"
  echo Usage: run_policy_3d.bat [model_path]
  exit /b 1
)

echo Building 3D view from "%MODEL%" ...
"%PYTHON%" "%SCRIPT%" --model "%MODEL%" --episodes 8 --max-steps 800 --max-points 4000 --out "%OUT%"
if errorlevel 1 exit /b 1

if exist "%OUT%" (
  echo Opening "%OUT%" ...
  start "" "%OUT%"
) else (
  echo Expected output not found: "%OUT%"
  exit /b 1
)

endlocal
