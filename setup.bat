@echo off
setlocal EnableDelayedExpansion
title Audio Transcript - Installation

echo.
echo  ╔══════════════════════════════════════════╗
echo  ║   Audio Transcript — Installation        ║
echo  ╚══════════════════════════════════════════╝
echo.

REM ── Vérification Python ─────────────────────
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  ERREUR : Python non trouvé.
    echo  Installez Python 3.11+ depuis https://python.org
    pause & exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  Python %PYVER% détecté

REM ── Environnement virtuel ────────────────────
if not exist .venv (
    echo  Création de l'environnement virtuel .venv...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo  ERREUR : impossible de créer le venv.
        pause & exit /b 1
    )
) else (
    echo  Environnement virtuel existant — réutilisé.
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet

echo.
echo  [1/4] Installation de PyTorch (CUDA 12.4)...
echo        ^(premier lancement : ~2-4 GB à télécharger^)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet
if %errorlevel% neq 0 (
    echo  ERREUR lors de l'installation de PyTorch.
    pause & exit /b 1
)
echo        OK

echo.
echo  [2/4] Installation de faster-whisper...
pip install faster-whisper --quiet
echo        OK

echo.
echo  [3/4] Installation de pyannote.audio (diarisation)...
pip install pyannote.audio --quiet
echo        OK

echo.
echo  [4/4] Installation des dependances complementaires...
pip install anthropic python-dotenv librosa groq --quiet
echo        OK

REM ── Fichier .env ─────────────────────────────
if not exist .env (
    copy .env.example .env >nul
    echo.
    echo  Fichier .env créé à partir de .env.example.
    echo  *** Editez .env pour renseigner vos clés API ***
) else (
    echo.
    echo  Fichier .env existant conservé.
)

echo.
echo  ╔══════════════════════════════════════════╗
echo  ║   Installation terminée !                ║
echo  ╚══════════════════════════════════════════╝
echo.
echo  Utilisation :
echo    transcribe.bat votre_audio.mp3
echo    transcribe.bat --help
echo.
pause
