@echo off
cd /d "%~dp0"
echo Starting daily update...
REM Adjust "python" to your specific python path if necessary (e.g. C:\Users\YourName\Anaconda3\python.exe)
python run_daily_update.py
echo Update finished.
pause
