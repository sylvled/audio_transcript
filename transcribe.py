#!/usr/bin/env python3
"""
Audio Transcript -- Transcription avec identification des locuteurs
Pipeline : faster-whisper (GPU) + pyannote.audio + LLM (Groq / Ollama / Claude)

Usage :
    python transcribe.py reunion.mp3
    python transcribe.py interview.wav --language fr --max-speakers 3
    python transcribe.py podcast.m4a --llm ollama
    python transcribe.py audio.mp3 --llm none
"""

import os
import sys
import json
import logging
import warnings
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional

# -- Suppression des warnings verbeux de PyTorch / pyannote / torchcodec --
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "3")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("pytorch").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Constantes LLM
# =============================================================================

GROQ_DEFAULT_MODEL   = "llama-3.1-8b-instant"   # 30k TPM, 128k ctx, gratuit
GROQ_CHUNK_TOKENS    = 1500                       # tokens max par chunk (free tier : 6k/req, ~2.4x ratio fr)
OLLAMA_DEFAULT_MODEL = "mistral"
OLLAMA_BASE_URL      = "http://localhost:11434"


# =============================================================================
# Utilitaires generaux
# =============================================================================

def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def get_speaker_for_segment(diarization, start: float, end: float) -> str:
    durations: dict[str, float] = defaultdict(float)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        overlap_start = max(turn.start, start)
        overlap_end   = min(turn.end, end)
        if overlap_start < overlap_end:
            durations[speaker] += overlap_end - overlap_start
    if not durations:
        return "SPEAKER_UNKNOWN"
    return max(durations, key=durations.get)


def normalize_speakers(segments: list[dict]) -> list[dict]:
    mapping: dict[str, str] = {}
    counter = 1
    for seg in segments:
        raw = seg.get("speaker", "SPEAKER_UNKNOWN")
        if raw not in mapping:
            mapping[raw] = f"Intervenant {counter}"
            counter += 1
    for seg in segments:
        seg["speaker"] = mapping.get(seg.get("speaker", "SPEAKER_UNKNOWN"), "Intervenant ?")
    return segments


# =============================================================================
# Formatage de la transcription
# =============================================================================

def format_with_speakers(segments: list[dict]) -> str:
    if not segments:
        return ""
    lines = []
    cur_speaker = segments[0]["speaker"]
    cur_start   = segments[0]["start"]
    cur_end     = segments[0]["end"]
    cur_texts   = [segments[0]["text"]]

    def flush(speaker, start, end, texts):
        ts = f"[{format_time(start)} -> {format_time(end)}]"
        lines.append(f"\n{speaker} {ts}")
        lines.append(" ".join(t for t in texts if t))

    for seg in segments[1:]:
        if seg["speaker"] == cur_speaker:
            cur_end = seg["end"]
            cur_texts.append(seg["text"])
        else:
            flush(cur_speaker, cur_start, cur_end, cur_texts)
            cur_speaker = seg["speaker"]
            cur_start   = seg["start"]
            cur_end     = seg["end"]
            cur_texts   = [seg["text"]]
    flush(cur_speaker, cur_start, cur_end, cur_texts)
    return "\n".join(lines).strip()


def format_simple(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        ts = f"[{format_time(seg['start'])} -> {format_time(seg['end'])}]"
        lines.append(f"{ts} {seg['text']}")
    return "\n".join(lines)


# =============================================================================
# Analyse du timbre vocal (librosa)
# =============================================================================

def extract_speaker_profiles(
    audio_array,
    segments: list[dict],
    sample_rate: int = 16000,
) -> dict[str, dict]:
    """
    Profil acoustique par locuteur : pitch, timbre, debit.
    Transmis au LLM pour croiser avec les indices contextuels.
    """
    try:
        import numpy as np
        import librosa
    except ImportError:
        return {}

    speaker_segs: dict[str, list[dict]] = defaultdict(list)
    for seg in segments:
        if "speaker" in seg:
            speaker_segs[seg["speaker"]].append(seg)

    profiles: dict[str, dict] = {}

    for speaker, segs in speaker_segs.items():
        try:
            chunks, total_words, total_duration = [], 0, 0.0
            for seg in segs:
                s = int(seg["start"] * sample_rate)
                e = int(seg["end"]   * sample_rate)
                chunk = audio_array[s:e]
                if len(chunk) > sample_rate * 0.1:
                    chunks.append(chunk)
                total_words    += len(seg.get("words", []))
                total_duration += max(0.0, seg["end"] - seg["start"])
            if not chunks:
                continue

            audio_spk = np.concatenate(chunks).astype(np.float32)

            f0, voiced_flag, _ = librosa.pyin(
                audio_spk,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C6"),
                sr=sample_rate,
            )
            f0_valid = f0[voiced_flag & ~np.isnan(f0)]
            pitch_med = float(np.median(f0_valid)) if len(f0_valid) > 20 else 0.0
            pitch_std = float(np.std(f0_valid))    if len(f0_valid) > 20 else 0.0

            centroid = float(np.mean(
                librosa.feature.spectral_centroid(y=audio_spk, sr=sample_rate)
            ))
            wpm = (total_words / total_duration * 60) if total_duration > 0 else 0.0

            if   pitch_med > 190: pitch_desc = f"voix feminine tres probable (F0={pitch_med:.0f} Hz)"
            elif pitch_med > 155: pitch_desc = f"voix feminine probable (F0={pitch_med:.0f} Hz)"
            elif pitch_med > 110: pitch_desc = f"voix masculine, registre moyen (F0={pitch_med:.0f} Hz)"
            elif pitch_med >   0: pitch_desc = f"voix masculine grave (F0={pitch_med:.0f} Hz)"
            else:                 pitch_desc = "registre vocal non determine"

            if   pitch_std > 55: mood_desc = "intonation tres expressive"
            elif pitch_std > 28: mood_desc = "intonation naturelle"
            else:                mood_desc = "intonation posee"

            if   centroid > 2200: timbre_desc = "timbre clair et brillant"
            elif centroid > 1300: timbre_desc = "timbre equilibre"
            else:                 timbre_desc = "timbre grave et chaud"

            if   wpm > 190: rate_desc = f"debit rapide ({wpm:.0f} mots/min)"
            elif wpm > 130: rate_desc = f"debit normal ({wpm:.0f} mots/min)"
            elif wpm >   0: rate_desc = f"debit lent ({wpm:.0f} mots/min)"
            else:           rate_desc = "debit non determine"

            profiles[speaker] = {
                "pitch_median_hz":    round(pitch_med, 1),
                "pitch_variation_hz": round(pitch_std, 1),
                "spectral_centroid":  round(centroid, 0),
                "wpm":                round(wpm, 0),
                "description": f"{pitch_desc}; {mood_desc}; {timbre_desc}; {rate_desc}",
            }
        except Exception as exc:
            profiles[speaker] = {"description": f"analyse impossible ({exc})"}

    return profiles


def format_profiles_for_prompt(profiles: dict[str, dict]) -> str:
    if not profiles:
        return ""
    lines = ["\nPROFILS VOCAUX ACOUSTIQUES :"]
    for speaker, p in sorted(profiles.items()):
        lines.append(f"  {speaker}: {p.get('description', 'non disponible')}")
    return "\n".join(lines)


# =============================================================================
# Moteur LLM multi-backend
# =============================================================================

def estimate_tokens(text: str) -> int:
    """Estimation grossiere : ~1.3 tokens/mot + marge."""
    return int(len(text.split()) * 1.3) + 100


def _ollama_available() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def select_backend(transcript: str, forced: str) -> str:
    """
    Selectionne automatiquement le backend LLM si forced='auto'.
    Priorite : Groq (gratuit) -> Ollama (local) -> Claude (payant)
    """
    if forced != "auto":
        return forced

    tokens = estimate_tokens(transcript)
    has_groq   = bool(os.getenv("GROQ_API_KEY"))
    has_claude = bool(os.getenv("ANTHROPIC_API_KEY"))

    print(f"      Tokens estimes : ~{tokens:,}")

    if has_groq:
        print(f"      -> Backend selectionne : GROQ (chunking automatique si necessaire)")
        return "groq"
    if _ollama_available():
        print(f"      -> Backend selectionne : OLLAMA local")
        return "ollama"
    if has_claude:
        print(f"      -> Backend selectionne : CLAUDE")
        return "claude"

    print("      [!] Aucun backend LLM disponible -> traitement LLM ignore")
    print("          Configurez GROQ_API_KEY dans .env ou lancez Ollama en local.")
    return "none"


def chunk_transcript(transcript: str, max_tokens: int) -> list[str]:
    """Decoupe en chunks sans couper un tour de parole."""
    lines = transcript.split("\n")
    chunks, current, current_tok = [], [], 0
    for line in lines:
        lt = estimate_tokens(line) + 1
        if current_tok + lt > max_tokens and current:
            chunks.append("\n".join(current))
            current, current_tok = [line], lt
        else:
            current.append(line)
            current_tok += lt
    if current:
        chunks.append("\n".join(current))
    return chunks if chunks else [transcript]


def _build_id_prompt(sample: str, language: str, voice_profiles: dict) -> str:
    profiles_section = format_profiles_for_prompt(voice_profiles)
    return f"""Analyse ce debut de transcription et identifie les vrais noms des locuteurs.
{profiles_section}

Indices a rechercher :
- Presentations : "Je m'appelle...", "Bonjour je suis...", "Mon nom est..."
- Apostrophes directes : "Et toi Pierre...", "Qu'en penses-tu Marie ?"
- Auto-references, titres ou fonctions mentionnes
- Corrobore avec les profils vocaux si disponibles (voix feminine -> confirme un prenom feminin)

Regles strictes :
- Identification CERTAINE uniquement -> donne le vrai nom
- Identification incertaine -> conserve "Intervenant N"
- Ne jamais inventer un nom
- Langue : {language}

Reponds UNIQUEMENT avec ce JSON (sans texte avant ni apres) :
{{"speakers": {{"Intervenant 1": "<nom reel ou Intervenant 1>", "Intervenant 2": "<nom reel ou Intervenant 2>"}}}}

ECHANTILLON :
{sample}"""


def _build_polish_prompt(chunk: str, language: str) -> str:
    return f"""Corrige cette transcription audio (produite par Whisper).

INSTRUCTIONS :
- Corriger les erreurs de reconnaissance vocale (homophones, noms propres)
- Ameliorer la ponctuation et la segmentation des phrases
- Supprimer les repetitions accidentelles ("le le", "que que"...)
- Conserver les timestamps EXACTEMENT tels quels [MM:SS.ss -> MM:SS.ss]
- Conserver les noms des locuteurs EXACTEMENT tels quels
- Ne PAS resumer, couper ou paraphraser
- Langue : {language}

Reponds UNIQUEMENT avec ce JSON (sans texte avant ni apres) :
{{"transcript": "<texte corrige>"}}

TRANSCRIPTION :
{chunk}"""


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


class GroqChunkTooLargeError(Exception):
    """Levee quand le chunk depasse la limite de tokens Groq (413)."""
    pass


def _call_groq(prompt: str, model: str, max_retries: int = 3) -> str:
    import time
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("Package 'groq' absent. Lancez : pip install groq")
    client   = Groq(api_key=os.environ["GROQ_API_KEY"])
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            return resp.choices[0].message.content
        except Exception as exc:
            err = str(exc)
            # Erreur 413 : chunk trop grand pour le tier -- pas la peine de reessayer
            if "413" in err:
                raise GroqChunkTooLargeError(err) from exc
            last_exc = exc
            if attempt < max_retries:
                wait = 10 * attempt
                print(f"\n      [!] Groq tentative {attempt}/{max_retries} ({exc}). Attente {wait}s...")
                time.sleep(wait)
    raise last_exc


def _call_ollama(prompt: str, model: str, timeout: int = 300) -> str:
    import urllib.request
    payload = json.dumps({
        "model":   model,
        "messages": [{"role": "user", "content": prompt}],
        "stream":  False,
        "format":  "json",
        "options": {"temperature": 0.1},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def _call_claude(prompt: str, model: str, max_retries: int = 3) -> str:
    import time
    import anthropic
    client   = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=16000,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = 5 * attempt
                print(f"\n      [!] Claude tentative {attempt}/{max_retries} ({exc}). Attente {wait}s...")
                time.sleep(wait)
    raise last_exc


def _call_llm(prompt: str, backend: str,
              groq_model: str, ollama_model: str, claude_model: str) -> str:
    if backend == "groq":
        return _call_groq(prompt, groq_model)
    elif backend == "ollama":
        return _call_ollama(prompt, ollama_model)
    elif backend == "claude":
        return _call_claude(prompt, claude_model)
    raise ValueError(f"Backend inconnu : {backend}")


def llm_process(
    transcript: str,
    language: str,
    has_speakers: bool,
    voice_profiles: dict[str, dict],
    backend: str = "auto",
) -> tuple[str, dict[str, str]]:
    """
    Traitement LLM en deux passes :
      1. Identification des locuteurs sur un echantillon
      2. Correction/polish en chunks (contourne les limites de tokens)

    Retourne (transcript_ameliore, mapping_locuteurs).
    """
    if backend == "none":
        return transcript, {}

    groq_model   = os.getenv("GROQ_MODEL",        GROQ_DEFAULT_MODEL)
    ollama_model = os.getenv("OLLAMA_MODEL",       OLLAMA_DEFAULT_MODEL)
    claude_model = os.getenv("CLAUDE_MODEL",       "claude-sonnet-4-5")
    chunk_tokens = int(os.getenv("GROQ_CHUNK_TOKENS", str(GROQ_CHUNK_TOKENS)))

    effective = select_backend(transcript, backend)
    if effective == "none":
        return transcript, {}

    model_name = {"groq": groq_model, "ollama": ollama_model, "claude": claude_model}.get(effective, "?")
    print(f"    Backend : {effective.upper()} ({model_name})")

    speaker_map: dict[str, str] = {}

    # -- Passe 1 : Identification des locuteurs (sur un echantillon) --
    if has_speakers:
        try:
            print("    Passe 1/2 : identification des locuteurs...")
            chunks  = chunk_transcript(transcript, min(chunk_tokens, 3000))
            sample  = chunks[0]
            raw     = _call_llm(_build_id_prompt(sample, language, voice_profiles),
                                effective, groq_model, ollama_model, claude_model)
            data    = _parse_json(raw)
            speaker_map = data.get("speakers", {})

            identified = {k: v for k, v in speaker_map.items() if k != v}
            if identified:
                print("    Locuteurs identifies :")
                for anon, name in identified.items():
                    print(f"      {anon}  ->  {name}")
                for anon, name in identified.items():
                    transcript = transcript.replace(anon, name)
            else:
                print("    Aucun nom identifie dans le contenu (labels conserves)")
        except Exception as exc:
            print(f"    [!] Identification echouee : {exc}")

    # -- Passe 2 : Correction/polish en chunks (taille adaptive si 413) --
    try:
        print("    Passe 2/2 : correction et polish...")
        adaptive_tokens = chunk_tokens
        chunks = chunk_transcript(transcript, adaptive_tokens)
        corrected: list[str] = []
        idx = 0
        while idx < len(chunks):
            chunk = chunks[idx]
            print(f"\r      Chunk {idx+1}/{len(chunks)}...", end="", flush=True)
            try:
                raw  = _call_llm(_build_polish_prompt(chunk, language),
                                 effective, groq_model, ollama_model, claude_model)
                data = _parse_json(raw)
                corrected.append(data.get("transcript", chunk))
                idx += 1
            except GroqChunkTooLargeError:
                # Chunk encore trop grand meme apres reduction initiale :
                # on reduit de moitie et on re-decoupe la queue restante
                adaptive_tokens = max(400, adaptive_tokens // 2)
                print(f"\n      [!] Chunk trop grand (413) -> re-decoupage : {adaptive_tokens} tok/chunk")
                remaining = "\n".join(chunks[idx:])
                chunks = chunks[:idx] + chunk_transcript(remaining, adaptive_tokens)
                # Ne pas incrementer idx -- on relance sur le meme idx avec le chunk plus petit
        print(f"\r      {len(corrected)} chunks traites.          ")
        transcript = "\n".join(corrected)
    except Exception as exc:
        print(f"\n    [!] Correction echouee : {exc}")

    return transcript, speaker_map


# =============================================================================
# Pipeline principal
# =============================================================================

def transcribe(
    audio_path: str,
    model_size: str = "large-v3",
    language: Optional[str] = None,
    hf_token: Optional[str] = None,
    llm_backend: str = "auto",
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> tuple[str, dict]:
    """Retourne (transcript_text, metadata_dict)."""
    import torch
    from faster_whisper import WhisperModel

    device       = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"\n{'='*52}")
    print(f"Fichier  : {audio_path}")
    print(f"GPU      : {torch.cuda.get_device_name(0)}" if device == "cuda"
          else "GPU      : aucun CUDA -- utilisation CPU")

    # -------------------------------------------------------------------------
    # Etape 1 : Transcription Whisper
    # -------------------------------------------------------------------------
    print(f"\n[1/4] Chargement du modele Whisper '{model_size}'...")
    model = WhisperModel(
        model_size, device=device, compute_type=compute_type,
        download_root=str(Path.home() / ".cache" / "whisper"),
    )

    print("[2/4] Transcription...")
    segments_gen, info = model.transcribe(
        audio_path, language=language, beam_size=5,
        word_timestamps=True, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200},
        condition_on_previous_text=True,
    )

    segments: list[dict] = []
    print(f"      Langue detectee : {info.language}  (confiance : {info.language_probability:.0%})")
    print(f"      Duree audio     : {format_time(info.duration)}")

    last_pct = -1
    for seg in segments_gen:
        segments.append({
            "start": seg.start, "end": seg.end, "text": seg.text.strip(),
            "words": [{"word": w.word, "start": w.start, "end": w.end}
                      for w in (seg.words or [])],
        })
        pct = min(int(seg.end / info.duration * 100) if info.duration > 0 else 0, 99)
        if pct != last_pct:
            print(f"\r      Progression : {pct:3d}%  [{format_time(seg.end)} / {format_time(info.duration)}]",
                  end="", flush=True)
            last_pct = pct
    print(f"\r      Progression : 100%  [{format_time(info.duration)} / {format_time(info.duration)}]"
          f"  -- {len(segments)} segments          ")

    metadata = {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration_seconds": round(info.duration, 2),
        "model": model_size, "device": device,
    }

    # -------------------------------------------------------------------------
    # Etape 2 : Diarisation (pyannote)
    # -------------------------------------------------------------------------
    diarization = None
    audio_array = None

    if hf_token:
        try:
            from pyannote.audio import Pipeline
            from faster_whisper.audio import decode_audio

            print("[3/4] Diarisation des locuteurs (pyannote.audio 3.1)...")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", token=hf_token)
            if device == "cuda":
                pipeline = pipeline.to(torch.device("cuda"))

            audio_array = decode_audio(audio_path, sampling_rate=16000)
            audio_input = {"waveform": torch.from_numpy(audio_array).unsqueeze(0),
                           "sample_rate": 16000}

            diarize_kwargs: dict = {}
            if min_speakers: diarize_kwargs["min_speakers"] = min_speakers
            if max_speakers: diarize_kwargs["max_speakers"] = max_speakers
            diarization = pipeline(audio_input, **diarize_kwargs)

            unique_speakers = {spk for _, _, spk in diarization.itertracks(yield_label=True)}
            print(f"      {len(unique_speakers)} locuteur(s) detecte(s)")
            metadata["num_speakers"] = len(unique_speakers)

        except ImportError:
            print("      [!] pyannote.audio non installe -- diarisation ignoree")
        except Exception as exc:
            msg = str(exc)
            if "403" in msg or "gated" in msg or "restricted" in msg or "authorized" in msg:
                print("      [!] Acces refuse (erreur 403). Acceptez les CGU sur HuggingFace :")
                print("          -> https://huggingface.co/pyannote/speaker-diarization-3.1")
                print("          -> https://huggingface.co/pyannote/pyannote/speaker-diarization-community-1")
                print("          -> https://huggingface.co/pyannote/segmentation-3.0")
            else:
                print(f"      [!] Erreur diarisation : {exc}")
    else:
        print("[3/4] Diarisation ignoree (HF_TOKEN non configure)")

    # -------------------------------------------------------------------------
    # Etape 3 : Fusion + analyse du timbre vocal
    # -------------------------------------------------------------------------
    if diarization is not None:
        for seg in segments:
            seg["speaker"] = get_speaker_for_segment(diarization, seg["start"], seg["end"])
        segments = normalize_speakers(segments)
        transcript = format_with_speakers(segments)
    else:
        transcript = format_simple(segments)

    voice_profiles: dict[str, dict] = {}
    if diarization is not None and audio_array is not None:
        try:
            import librosa as _
            print("      Analyse du timbre vocal...")
            voice_profiles = extract_speaker_profiles(audio_array, segments, sample_rate=16000)
            for spk, p in sorted(voice_profiles.items()):
                print(f"        {spk}: {p.get('description', '?')}")
        except ImportError:
            print("      [!] librosa absent -- analyse timbre ignoree (pip install librosa)")
    metadata["voice_profiles"] = voice_profiles

    # -------------------------------------------------------------------------
    # Etape 4 : Identification des locuteurs + polish LLM
    # -------------------------------------------------------------------------
    speaker_map: dict[str, str] = {}
    if llm_backend != "none":
        print("[4/4] Identification des locuteurs et amelioration (LLM)...")
        transcript, speaker_map = llm_process(
            transcript, info.language, diarization is not None,
            voice_profiles, backend=llm_backend,
        )
    else:
        print("[4/4] Traitement LLM desactive (--llm none)")

    metadata["speaker_map"]  = speaker_map
    metadata["llm_backend"]  = llm_backend
    return transcript, metadata


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Transcription audio avec identification des locuteurs par nom",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXEMPLES :
  transcribe.bat reunion.mp3
  transcribe.bat interview.wav --language fr --llm groq
  transcribe.bat conf.mp4 --max-speakers 3 --output cr.txt
  transcribe.bat podcast.mp3 --model large-v3-turbo --json
  transcribe.bat audio.m4a --llm ollama
  transcribe.bat long.mp3 --llm claude
  transcribe.bat audio.mp3 --llm none

PIPELINE :
  1. Transcription    faster-whisper (Whisper large-v3, GPU RTX)
  2. Diarisation      pyannote.audio 3.1  -> qui parle quand ?
  3. Timbre vocal     librosa -> pitch, timbre, debit par locuteur
  4. Identification   LLM : croise contexte + acoustique pour nommer les voix
     + Polish         LLM : corrige les erreurs de reconnaissance vocale

BACKENDS LLM (option --llm) :
  auto    Selection automatique : Groq si dispo, sinon Ollama, sinon Claude
  groq    Groq API (gratuit, llama-3.1-8b-instant par defaut, chunking auto)
  ollama  LLM local via Ollama (necessite ollama.ai installe + modele pull)
  claude  Anthropic Claude (payant, meilleure qualite)
  none    Pas de traitement LLM

MODELES WHISPER (option --model) :
  large-v3         Qualite maximale, ~3 GB    [defaut]
  large-v3-turbo   Quasi-identique, 2x rapide, ~1.6 GB
  large-v2 / medium / small / base / tiny

CONFIGURATION (.env) :
  GROQ_API_KEY        Cle Groq   ->  console.groq.com  (gratuit)
  ANTHROPIC_API_KEY   Cle Claude ->  console.anthropic.com
  HF_TOKEN            HuggingFace -> huggingface.co/settings/tokens
  GROQ_MODEL          Modele Groq   (defaut: llama-3.1-8b-instant)
  OLLAMA_MODEL        Modele Ollama (defaut: mistral)
  CLAUDE_MODEL        Modele Claude (defaut: claude-sonnet-4-5)
  WHISPER_MODEL       Modele Whisper par defaut
  GROQ_CHUNK_TOKENS   Tokens max par chunk Groq (defaut: 4000)
""",
    )

    parser.add_argument("audio",
        help="Fichier audio (mp3, wav, m4a, flac, ogg, mp4, mkv...)")
    parser.add_argument("--model", "-m",
        default=os.getenv("WHISPER_MODEL", "large-v3"),
        choices=["tiny","base","small","medium","large-v2","large-v3","large-v3-turbo","distil-large-v3"],
        help="Modele Whisper (defaut : WHISPER_MODEL dans .env, sinon large-v3)")
    parser.add_argument("--language", "-l", default=None,
        help="Code ISO 639-1 (ex: fr, en). Auto-detection si omis.")
    parser.add_argument("--output", "-o", default=None,
        help="Fichier .txt de sortie (defaut : meme nom que l'audio)")
    parser.add_argument("--llm",
        default=os.getenv("LLM_BACKEND", "auto"),
        choices=["auto", "groq", "ollama", "claude", "none"],
        help="Backend LLM (defaut: auto). Voir la section BACKENDS ci-dessus.")
    parser.add_argument("--min-speakers", type=int, default=None, metavar="N",
        help="Nombre minimum de locuteurs (hint pyannote)")
    parser.add_argument("--max-speakers", type=int, default=None, metavar="N",
        help="Nombre maximum de locuteurs (hint pyannote)")
    parser.add_argument("--no-diarize", action="store_true",
        help="Desactiver diarisation + analyse timbre")
    # Alias legacy
    parser.add_argument("--no-claude", action="store_true",
        help=argparse.SUPPRESS)   # conserve pour compat, equivalent a --llm none
    parser.add_argument("--json", action="store_true",
        help="Sauvegarder aussi les metadonnees en .json")

    args = parser.parse_args()

    # Legacy compat
    if args.no_claude:
        args.llm = "none"

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Erreur : fichier introuvable -> {audio_path}", file=sys.stderr)
        sys.exit(1)

    hf_token = None
    if not args.no_diarize:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("[!] HF_TOKEN absent -> diarisation desactivee")
            print("    Ajoutez HF_TOKEN=hf_... dans .env")

    transcript, metadata = transcribe(
        str(audio_path),
        model_size=args.model,
        language=args.language,
        hf_token=hf_token,
        llm_backend=args.llm,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    output_path = Path(args.output) if args.output else audio_path.with_suffix(".txt")
    output_path.write_text(transcript, encoding="utf-8")
    print(f"\n[OK] Transcription  -> {output_path}")

    if args.json:
        json_path = output_path.with_suffix(".json")
        json_path.write_text(
            json.dumps({"metadata": metadata, "transcript": transcript},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"[OK] Metadonnees    -> {json_path}")

    dur = metadata["duration_seconds"]
    h, rem = divmod(int(dur), 3600)
    m, s   = divmod(rem, 60)
    dur_str = f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"
    speaker_map = metadata.get("speaker_map", {})
    identified  = {k: v for k, v in speaker_map.items() if k != v}

    print(f"\nLangue : {metadata['language']}  |  Duree : {dur_str}  |  "
          f"Locuteurs : {metadata.get('num_speakers', 'N/A')}  |  "
          f"LLM : {metadata.get('llm_backend','?')}")
    if identified:
        print("Nommes : " + "  ".join(f"{k} -> {v}" for k, v in identified.items()))

    preview = transcript.split("\n")[:20]
    print(f"\n{'='*52}\nAPERCU :")
    print("\n".join(preview))
    if len(transcript.split("\n")) > 20:
        print("[...]")


if __name__ == "__main__":
    main()
