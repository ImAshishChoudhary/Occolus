@echo off
chcp 65001 >nul

echo 🧬 Welcome to OccolusAI Setup!
echo ================================

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11+
    pause
    exit /b 1
) else (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    echo [SUCCESS] Python %PYTHON_VERSION% found
)

:: Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Please install Node.js 18+
    pause
    exit /b 1
) else (
    for /f %%i in ('node --version') do set NODE_VERSION=%%i
    echo [SUCCESS] Node.js %NODE_VERSION% found
)

:: Check if npm is installed
npm --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm not found. Please install npm
    pause
    exit /b 1
) else (
    for /f %%i in ('npm --version') do set NPM_VERSION=%%i
    echo [SUCCESS] npm %NPM_VERSION% found
)

:: Check if Git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found. Please install Git
    pause
    exit /b 1
) else (
    for /f "tokens=3" %%i in ('git --version') do set GIT_VERSION=%%i
    echo [SUCCESS] Git %GIT_VERSION% found
)

echo.
echo [INFO] Setting up backend...

:: Backend setup
cd server

:: Create virtual environment
echo [INFO] Creating Python virtual environment...
python -m venv venv

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

:: Install Python dependencies
echo [INFO] Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

:: Create .env file if it doesn't exist
if not exist .env (
    echo [INFO] Creating .env file...
    (
        echo # Google Gemini API Key for AI-powered insights
        echo # Get your API key from: https://makersuite.google.com/app/apikey
        echo GEMINI_API_KEY=your_gemini_api_key_here
        echo.
        echo # Server Configuration
        echo HOST=localhost
        echo PORT=8000
        echo DEBUG=True
        echo.
        echo # CORS Settings
        echo ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
        echo.
        echo # Model Configuration
        echo MODEL_PATH=drug_target_model.pth
        echo DRUG_DB_PATH=drug_db.csv
    ) > .env
    echo [WARNING] Please edit server/.env and add your GEMINI_API_KEY
)

:: Deactivate virtual environment
deactivate

cd ..

echo.
echo [INFO] Setting up frontend...

:: Frontend setup
cd client

:: Install Node.js dependencies
echo [INFO] Installing Node.js dependencies...
npm install

:: Create .env.local file if it doesn't exist
if not exist .env.local (
    echo [INFO] Creating .env.local file...
    (
        echo # API Configuration
        echo # Backend server URL
        echo NEXT_PUBLIC_API_URL=http://localhost:8000
        echo.
        echo # Development Configuration
        echo NODE_ENV=development
    ) > .env.local
)

cd ..

echo.
echo [SUCCESS] Setup completed successfully!
echo.
echo 📋 Next Steps:
echo ==============
echo 1. Edit server/.env and add your GEMINI_API_KEY
echo 2. Start the backend: cd server ^&^& venv\Scripts\activate ^&^& uvicorn main:app --reload
echo 3. Start the frontend: cd client ^&^& npm run dev
echo 4. Open http://localhost:3000 in your browser
echo.
echo 🔗 Useful Links:
echo ================
echo • Frontend: http://localhost:3000
echo • Backend API: http://localhost:8000
echo • API Documentation: http://localhost:8000/docs
echo • Gemini API Key: https://makersuite.google.com/app/apikey
echo.
echo [SUCCESS] Happy drug discovering! 🧬
pause