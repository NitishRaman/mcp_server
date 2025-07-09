@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: Move to the script directory
cd /d %~dp0

:: === Check if Python is installed ===
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not added to PATH.
    echo Please install Python 3.8+ and try again.
    pause
    exit /b
)

:: === Check if port 8000 is already in use ===
netstat -ano | findstr :8000 >nul
if %errorlevel%==0 (
    echo ❌ Port 8000 is already in use. Please free it and re-run this script.
    pause
    exit /b
)

:: === Optional: Create and activate virtual environment ===
if not exist .venv (
    echo 🧪 Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

:: === First-time setup: install dependencies ===
set SETUP_MARKER=first_time_setup_done.txt

if not exist %SETUP_MARKER% (
    echo 📦 Upgrading pip safely...
    python -m pip install --upgrade pip

    echo 📦 Installing required packages from requirements.txt...
    python -m pip install -r requirements.txt --no-cache-dir

    echo Setup complete > %SETUP_MARKER%
) else (
    echo ✅ Requirements already installed. Skipping setup...
)

:: === Start ChromaDB in background ===
echo 🚀 Starting ChromaDB on port 8000...
start /min cmd /c "chroma run --path .\mcp_server\files\chroma --host localhost --port 8000"

:: === Pull Ollama model in background ===
echo ⬇️ Pulling Ollama model (llama3)...
start /min cmd /c "ollama pull llama3"

:: === Start FastAPI backend in background ===
echo 🛰️ Starting FastAPI server...
start /min cmd /c "uvicorn app_server:app --reload"

:: === Start Streamlit frontend in background ===
echo 🌐 Starting Streamlit UI...
start /min cmd /c "streamlit run home.py"

echo ✅ All systems launched in background.
endlocal
pause
