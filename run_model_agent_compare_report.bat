@echo off
setlocal
cd /d "%~dp0"
call scripts\windows\run_model_agent_compare_report.bat %*
endlocal
