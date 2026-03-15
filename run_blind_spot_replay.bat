@echo off
setlocal
cd /d "%~dp0"
call scripts\windows\run_blind_spot_replay.bat %*
endlocal
