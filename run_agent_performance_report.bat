@echo off
setlocal
cd /d "%~dp0"
call scripts\windows\run_agent_performance_report.bat %*
endlocal
