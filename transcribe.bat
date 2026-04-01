@echo off
setlocal EnableDelayedExpansion

REM -- Detection des flags d'aide ------------------------------------------
set "ARG=%~1"
if /i "%ARG%"=="-h"      goto :help
if /i "%ARG%"=="--h"     goto :help
if /i "%ARG%"=="-help"   goto :help
if /i "%ARG%"=="--help"  goto :help
if /i "%ARG%"=="/?"      goto :help
if /i "%ARG%"=="/h"      goto :help

REM -- Verification de l'environnement ------------------------------------
if not exist "%~dp0.venv" (
    echo.
    echo  ERREUR : environnement virtuel non trouve.
    echo  Lancez setup.bat pour installer les dependances.
    echo.
    pause
    exit /b 1
)

REM -- Lancement ----------------------------------------------------------
call "%~dp0.venv\Scripts\activate.bat"
python "%~dp0transcribe.py" %*
exit /b %errorlevel%


REM ========================================================================
:help
echo.
echo  ================================================================
echo   Audio Transcript  --  Aide et utilisation
echo  ================================================================
echo.
echo  USAGE :
echo    transcribe.bat  ^<fichier_audio^>  [options]
echo.
echo  ================================================================
echo   EXEMPLES RAPIDES
echo  ================================================================
echo.
echo  Cas le plus courant (tout automatique, Groq par defaut) :
echo    transcribe.bat reunion.mp3
echo.
echo  Totalement autonome -- 100%% local, aucune donnee envoyee en ligne :
echo    transcribe.bat reunion.mp3 --llm ollama
echo.
echo  Forcer la langue pour eviter l'auto-detection :
echo    transcribe.bat interview.wav --language fr
echo.
echo  Nombre de locuteurs connu -- ameliore la diarisation :
echo    transcribe.bat conf.mp4 --max-speakers 3
echo.
echo  Fichier de sortie personnalise + export metadonnees JSON :
echo    transcribe.bat conf.mp4 --output compte-rendu.txt --json
echo.
echo  Modele plus rapide (vitesse x2, qualite quasi identique) :
echo    transcribe.bat podcast.mp3 --model large-v3-turbo
echo.
echo  Meilleure qualite LLM (Claude au lieu de Groq, payant) :
echo    transcribe.bat audio.mp3 --llm claude
echo.
echo  Transcription brute sans identification des locuteurs :
echo    transcribe.bat audio.m4a --no-diarize
echo.
echo  Transcription brute sans aucun traitement LLM (le plus rapide) :
echo    transcribe.bat audio.mp3 --llm none
echo.
echo  Controle precis du nombre de locuteurs (ex: 2 a 4 personnes) :
echo    transcribe.bat rec.wav --min-speakers 2 --max-speakers 4
echo.
echo  ================================================================
echo   OPTIONS
echo  ================================================================
echo.
echo    --model, -m ^<nom^>      Modele Whisper de transcription
echo                            (defaut : large-v3, voir liste ci-dessous)
echo    --language, -l ^<code^>  Code langue ISO 639-1 : fr, en, es, de...
echo                            (auto-detection si omis -- generalement fiable)
echo    --output, -o ^<fichier^>  Chemin du fichier de sortie .txt
echo                            (defaut : meme nom que le fichier audio)
echo    --llm ^<backend^>         Moteur LLM pour identification + correction :
echo                              auto   = Groq si configure, sinon Ollama,
echo                                       sinon Claude  [DEFAUT]
echo                              groq   = Groq API, gratuit, cloud
echo                              ollama = Mistral local, 100%% prive
echo                              claude = Anthropic Claude, payant, qualite max
echo                              none   = pas de traitement LLM (brut Whisper)
echo    --min-speakers ^<N^>      Nombre minimum de locuteurs (hint pyannote)
echo    --max-speakers ^<N^>      Nombre maximum de locuteurs (hint pyannote)
echo                            -^> Toujours specifier si connu, ameliore
echo                               nettement la separation des voix
echo    --no-diarize            Desactiver diarisation et analyse timbre
echo                            (utile pour monologue ou dictee simple)
echo    --json                  Exporter aussi un fichier .json avec :
echo                            langue, duree, profils vocaux, mapping noms
echo    -h, --help              Afficher cette aide
echo.
echo  ================================================================
echo   MODES D'UTILISATION
echo  ================================================================
echo.
echo  Mode GRATUIT + CLOUD (defaut recommande) :
echo    -^> Transcription et diarisation 100%% locales sur GPU
echo    -^> Identification + correction via Groq (gratuit, texte seulement)
echo    -^> Commande : transcribe.bat audio.mp3
echo.
echo  Mode 100%% LOCAL (prive, aucune donnee externe) :
echo    -^> Tout tourne sur votre machine (GPU RTX)
echo    -^> Aucune connexion internet requise apres installation
echo    -^> Commande : transcribe.bat audio.mp3 --llm ollama
echo.
echo  Mode QUALITE MAXIMALE (payant, Claude) :
echo    -^> Meilleure identification des noms et correction des erreurs
echo    -^> Recommande pour reunions importantes ou audio difficile
echo    -^> Commande : transcribe.bat audio.mp3 --llm claude
echo.
echo  Mode RAPIDE (transcription seule, sans LLM) :
echo    -^> Whisper + pyannote uniquement, pas de polish
echo    -^> Resultat brut mais tres rapide
echo    -^> Commande : transcribe.bat audio.mp3 --llm none
echo.
echo  ================================================================
echo   MODELES WHISPER (option --model)
echo  ================================================================
echo.
echo    large-v3          Qualite maximale (~3 GB VRAM)      [DEFAUT]
echo    large-v3-turbo    Quasi-identique, 2x plus rapide (~1.6 GB)
echo    large-v2          Tres bon, ancienne generation
echo    medium            Bon compromis taille/qualite (~1.5 GB)
echo    small             Rapide, qualite correcte (~500 MB)
echo    base / tiny       Tests uniquement
echo.
echo  ================================================================
echo   PIPELINE (ordre d'execution)
echo  ================================================================
echo.
echo    1. Transcription    Whisper large-v3 sur GPU  -^>  texte + timestamps
echo    2. Diarisation      pyannote.audio 3.1        -^>  qui parle quand ?
echo    3. Timbre vocal     librosa                   -^>  pitch, debit, timbre
echo    4. Identification   LLM : croise contexte + acoustique -^> vrais noms
echo       + Polish         LLM : corrige homophones, ponctuation
echo.
echo  ================================================================
echo   FORMATS AUDIO SUPPORTES
echo  ================================================================
echo.
echo    Audio : mp3  wav  m4a  flac  ogg  opus  aac  wma  amr
echo    Video : mp4  mkv  mov  avi  webm  (piste audio extraite)
echo    Telephonie : amr  3gp  (WhatsApp, enregistrements mobiles)
echo.
echo  ================================================================
echo   CONFIGURATION (fichier .env)
echo  ================================================================
echo.
echo    GROQ_API_KEY        Cle Groq (gratuit)  -^>  console.groq.com
echo    ANTHROPIC_API_KEY   Cle Claude (payant) -^>  console.anthropic.com
echo    HF_TOKEN            Token HuggingFace   -^>  huggingface.co/settings/tokens
echo    LLM_BACKEND         Backend par defaut  (auto/groq/ollama/claude/none)
echo    GROQ_MODEL          Modele Groq         (defaut: llama-3.1-8b-instant)
echo    OLLAMA_MODEL        Modele Ollama       (defaut: mistral)
echo    CLAUDE_MODEL        Modele Claude       (defaut: claude-sonnet-4-5)
echo    WHISPER_MODEL       Modele Whisper      (defaut: large-v3)
echo    GROQ_CHUNK_TOKENS   Tokens max/chunk    (defaut: 4000)
echo.
echo  ================================================================
echo   FICHIERS DE SORTIE
echo  ================================================================
echo.
echo    ^<audio^>.txt    Transcription finale avec locuteurs nommes et timestamps
echo    ^<audio^>.json   Metadonnees completes (avec --json) :
echo                   langue, duree, profils vocaux, mapping des noms
echo.
goto :eof
