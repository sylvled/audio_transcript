#!/usr/bin/env bash
# Lanceur bash pour Audio Transcript

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Environnement non installé. Lancez ./setup.sh d'abord."
    exit 1
fi

if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate   # Windows Git Bash
else
    source .venv/bin/activate       # Linux/macOS
fi

python transcribe.py "$@"
