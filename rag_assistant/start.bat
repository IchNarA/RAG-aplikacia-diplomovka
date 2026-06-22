@echo off
SETLOCAL
cls
echo ======================================================
echo           RAG STUDY ASSISTANT - START
echo ======================================================
echo.

:: 1. Kontrola Ollamy
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Ollama nebezi. Prosim, spusti ju.
    pause
    exit /b
)

:: 2. Príprava priečinkov
if not exist "data\notebooks" mkdir "data\notebooks"
if not exist "data\models_cache" mkdir "data\models_cache"

:: 3. Docker Build a Run
echo [*] Spustam Docker kontajner
docker-compose up -d --build

:: 4. Automatický Pull LLM modelu
echo [*] Kontrolujem a stahujem potrebny LLM model...
docker exec rag_study_assistant python -c "import requests; from config import LLM_MODEL; print(f'Target model: {LLM_MODEL}'); r = requests.post('http://host.docker.internal:11434/api/pull', json={'name': LLM_MODEL}, timeout=None); print('Model ready.')"

echo.
echo === APLIKACIA BEZI ===
echo Adresa: http://localhost:8501
echo.
start http://localhost:8501
pause