@echo off
cd /d "%~dp0"
if not exist "python\python.exe" (
  echo The python folder is missing.
  echo Unzip the whole results folder, then run this file from inside it.
  pause
  exit /b 1
)
echo Starting the results GUI at http://127.0.0.1:8055
echo Leave this window open while you use it. Close the window to stop the GUI.
start "" cmd /c "ping -n 4 127.0.0.1 >nul & start http://127.0.0.1:8055/"
"python\python.exe" "view_results.py"
if errorlevel 1 pause
pause
