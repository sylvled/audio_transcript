#!/usr/bin/env bash
# Audio Transcript — Script d'installation (bash / Git Bash)
set -e

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Audio Transcript — Installation        ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Vérification Python ──────────────────────
if ! command -v python &>/dev/null; then
    echo "ERREUR : Python non trouvé. Installez Python 3.11+ depuis https://python.org"
    exit 1
fi
echo "Python : $(python --version)"

# ── Environnement virtuel ────────────────────
if [ ! -d ".venv" ]; then
    echo "Création de l'environnement virtuel .venv…"
    python -m venv .venv
else
    echo "Environnement virtuel existant — réutilisé."
fi

# Activation (Windows Git Bash)
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

python -m pip install --upgrade pip --quiet

echo ""
echo "[1/4] Installation de PyTorch (CUDA 12.4)…"
echo "      (premier lancement : ~2-4 GB à télécharger)"
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet
echo "      OK"

echo ""
echo "[2/4] Installation de faster-whisper…"
pip install faster-whisper --quiet
echo "      OK"

echo ""
echo "[3/4] Installation de pyannote.audio (diarisation)…"
pip install pyannote.audio --quiet
echo "      OK"

echo ""
echo "[4/4] Installation des dépendances complémentaires…"
pip install anthropic python-dotenv librosa groq --quiet
echo "      OK"

# ── Fichier .env ─────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "Fichier .env créé. *** Éditez-le pour renseigner vos clés API ***"
else
    echo ""
    echo "Fichier .env existant conservé."
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Installation terminée !                ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "Utilisation :"
echo "  ./transcribe.sh votre_audio.mp3"
echo "  ./transcribe.sh --help"
echo ""
