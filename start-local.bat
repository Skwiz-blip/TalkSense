@echo off
REM ============================================================
REM LiveYDream - Setup & Start Script (Windows CMD)
REM ============================================================

echo.
echo 🚀 LiveYDream Local Setup with Supabase
echo ========================================

REM ── 1. Check .env ────────────────────────────────────────────
if not exist .env (
    echo ❌ .env file not found. Creating from .env.example...
    copy .env.example .env
    echo ⚠️  Please edit .env and add your Supabase password!
    echo    DB_PASSWORD=your_password_here
    pause
    exit /b 1
)

REM ── 2. Start Docker services ─────────────────────────────────
echo.
echo 🐳 Starting Docker services (Redis, MinIO, MLflow, Bull Board)...
docker compose -f docker-compose.local.yml up -d

echo.
echo ⏳ Waiting for services to be healthy...
timeout /t 5 /nobreak > nul

REM ── 3. Check services ────────────────────────────────────────
echo.
echo ✅ Services status:
docker compose -f docker-compose.local.yml ps

REM ── 4. Install Node dependencies ─────────────────────────────
echo.
if exist package.json (
    echo 📦 Installing Node.js dependencies...
    call npm install
)

REM ── 5. Summary ───────────────────────────────────────────────
echo.
echo ========================================
echo ✅ Setup complete!
echo.
echo 📋 Next steps:
echo    1. Add your Supabase password to .env
echo       DB_PASSWORD=your_supabase_password
echo.
echo    2. Run the SQL schema on Supabase:
echo       Go to: https://supabase.com/dashboard/project/ljbkjafykyiqmjtwlwpn/sql
echo       Paste the contents of: 001_supabase_schema.sql
echo.
echo    3. Start the backend:
echo       npm run start:dev
echo.
echo 🌐 Services:
echo    • Backend:     http://localhost:3000
echo    • Bull Board:  http://localhost:3001
echo    • MLflow:      http://localhost:5000
echo    • MinIO:       http://localhost:9001 (minioadmin/minioadmin)
echo.
pause
