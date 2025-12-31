@echo off
REM ============================================
REM  BTC Trading Bot - Quick Start (Windows)
REM ============================================

echo.
echo ============================================
echo   BTC TRADING BOT - BYBIT DEMO
echo ============================================
echo.

REM Check if venv exists
if not exist "btc_trading_env\Scripts\activate.bat" (
    echo [!] Virtual environment not found. Creating...
    python -m venv btc_trading_env
    call btc_trading_env\Scripts\activate.bat
    pip install -r requirements.txt
    echo.
    echo [OK] Environment created and packages installed!
) else (
    call btc_trading_env\Scripts\activate.bat
)

REM Set API keys (CHANGE THESE!)
set BYBIT_API_KEY=ZVtFkU98lwei7fhuo4
set BYBIT_API_SECRET=f4ruVkXNP5vIkVkqc0iuj76ze0DbQqe3pI6y

REM Check if keys are set
if "%BYBIT_API_KEY%"=="YOUR_API_KEY_HERE" (
    echo.
    echo [ERROR] Please edit start_bot.bat and set your API keys!
    echo         BYBIT_API_KEY=your_key
    echo         BYBIT_API_SECRET=your_secret
    echo.
    pause
    exit /b 1
)

echo [*] Starting bot...
echo.
python live_trading_bot.py

pause
