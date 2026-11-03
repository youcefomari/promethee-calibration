@echo off
REM Re-runs the two timing experiments of the paper on this computer (about 10-15 minutes).
REM Double-click this file, or run it from a terminal opened in the "code" folder.
REM Close other heavy programs and keep the laptop plugged in during the run.
cd /d "%~dp0"
if not exist .venv (
    python -m venv .venv || (echo Python 3.10 or newer is required: https://www.python.org/downloads/ & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
if not exist ..\results\revision mkdir ..\results\revision
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt || (echo Package installation failed & pause & exit /b 1)
python -m pytest tests -q -x || (echo Tests failed, please send me the output above & pause & exit /b 1)
python -m experiments.run_timing --workers 1 > ..\results\revision\timing_log.txt 2>&1
python -m experiments.run_scaling > ..\results\revision\scaling_log.txt 2>&1
echo.
echo Done. Results are in results\revision (timing.csv, scaling_*.csv, hardware_*.txt, *_log.txt).
pause
