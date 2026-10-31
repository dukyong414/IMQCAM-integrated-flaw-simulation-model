@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set MPLBACKEND=Agg
set PYVISTA_OFF_SCREEN=true
set VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN=1

REM Prefer a per-machine venv outside OneDrive so it is not synced between PCs.
set "PY=%LOCALAPPDATA%\NASA-porosity\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo ERROR: Project venv not found:
  echo   %PY%
  echo.
  echo Create it with:
  echo   py -3 -m venv "%LOCALAPPDATA%\NASA-porosity\.venv"
  echo   "%LOCALAPPDATA%\NASA-porosity\.venv\Scripts\python.exe" -m pip install -r requirements_integrated.txt
  echo   "%LOCALAPPDATA%\NASA-porosity\.venv\Scripts\python.exe" -m pip install -e .\Raptor
  echo.
  pause
  exit /b 1
)

echo Starting the porosity GUI at http://127.0.0.1:8050
echo Leave this window open while you use it. Close the window to stop the GUI.
start "" cmd /c "ping -n 4 127.0.0.1 >nul & start http://127.0.0.1:8050/"
"%PY%" -m integrated_cli gui --no-browser
if errorlevel 1 (
  echo.
  echo GUI failed to start. If you see "No Python at ...Miniconda...", the
  echo .venv was copied from another PC and must be recreated on this machine.
)
echo.
echo GUI stopped.
pause
