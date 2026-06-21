@echo off
echo ================================
echo  RAG Eval Suite - First Time Setup
echo ================================

echo.
echo [1/5] Copying .env file...
if not exist .env copy .env.example .env

echo.
echo [2/5] Creating Python virtual environment...
cd backend
python -m venv venv

echo.
echo [3/5] Installing Python dependencies...
call venv\Scripts\activate
pip install -r requirements.txt

echo.
echo [4/5] Installing frontend dependencies...
cd ..\frontend
npm install

echo.
echo [5/5] Seeding demo data...
cd ..
call backend\venv\Scripts\activate
python -m scripts.seed_data

echo.
echo ================================
echo  Setup complete!
echo  Now run: start.bat
echo ================================
pause