@echo off
echo Installing Mean Reversion Strategy Signal Service...
echo.

REM Create Windows Task Scheduler task to run at 3:45 PM on weekdays
schtasks /create /tn "MeanReversionStrategy" /tr "python C:\Users\edwar\mean_reversion_strategy\run_signals.py" /sc weekly /d MON,TUE,WED,THU,FRI /st 15:45 /f

echo.
echo Task scheduled! The service will run automatically at 3:45 PM ET on weekdays.
echo.
echo To check status: schtasks /query /tn "MeanReversionStrategy"
echo To run manually: schtasks /run /tn "MeanReversionStrategy"
echo To delete: schtasks /delete /tn "MeanReversionStrategy" /f
echo.
pause
