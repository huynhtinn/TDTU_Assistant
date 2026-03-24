@echo off
REM Script chạy nhanh giao diện web TDTU AI Assistant

echo ========================================
echo    TDTU AI Assistant - Web Interface
echo ========================================
echo.

cd /d "%~dp0"

echo [1/2] Kiem tra Streamlit...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [!] Streamlit chua duoc cai dat!
    echo [+] Dang cai dat Streamlit...
    pip install streamlit
)

echo [2/2] Khoi dong giao dien web...
echo.
echo Giao dien se mo tai: http://localhost:8501
echo Nhan Ctrl+C de dung.
echo.

streamlit run src\app\app.py

pause
