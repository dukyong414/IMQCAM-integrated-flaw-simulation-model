@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set "PY=%LOCALAPPDATA%\NASA-porosity\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0.venv\Scripts\python.exe"
REM No arguments: launch Dash GUI and open the browser
if "%~1"=="" (
  echo Starting Integrated Porosity GUI...
  echo Browser will open at http://127.0.0.1:8050
  echo Keep this window open while using the site.
  echo Close this window to stop the server.
  echo.
  "%PY%" -m integrated_cli gui
  echo.
  echo GUI stopped.
  pause
  goto :eof
)
"%PY%" -m integrated_cli %*
