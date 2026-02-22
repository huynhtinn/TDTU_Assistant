#!/bin/bash
# Script to run TDTU AI Assistant Web Interface on Linux/Mac

echo "========================================"
echo "   TDTU AI Assistant - Web Interface"
echo "========================================"
echo ""

cd "$(dirname "$0")"

echo "[1/2] Checking Streamlit installation..."
python -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[!] Streamlit not installed!"
    echo "[+] Installing Streamlit..."
    pip install streamlit
fi

echo "[2/2] Starting web interface..."
echo ""
echo "Interface will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop."
echo ""

streamlit run src/app/app.py
