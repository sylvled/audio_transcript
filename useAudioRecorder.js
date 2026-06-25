/**
 * Hook d'enregistrement audio robuste pour mobile
 *
 * Fonctionnalités :
 * - Wake Lock : empeche le telephone de se mettre en veille
 * - Audio silencieux en boucle : empeche le navigateur Android de suspendre
 *   le processus quand l'onglet passe en arriere-plan (technique NoSleep)
 * - visibilitychange : reprend l'AudioContext et le MediaRecorder si
 *   interrompus lors d'un retour au premier plan
 * - Chunks de 30s : chaque chunk est envoye immediatement
 * - IndexedDB : stockage local des chunks en attente pour retry
 * - Reprise automatique apres erreur reseau
 * - 3h max d'enregistrement
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { directionService } from '../services/directionService';

const CHUNK_DURATION_MS = 30_000;
const MAX_DURATION_MS = 3 * 60 * 60 * 1000;
const DB_NAME = 'direction_audio';
const DB_STORE = 'pending_chunks';

// ── IndexedDB helpers ──────────────────────────────────────────────────────

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(DB_STORE)) {
        db.createObjectStore(DB_STORE, { keyPath: 'key' });
      }
    };
    req.onsuccess = (e) => resolve(e.target.result);
    req.onerror = (e) => reject(e.target.error);
  });
}

async function saveChunkLocally(sessionId, chunkIndex, blob) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(DB_STORE, 'readwrite');
    tx.objectStore(DB_STORE).put({
      key: `${sessionId}_${chunkIndex}`,
      sessionId,
      chunkIndex,
      blob,
      savedAt: Date.now(),
    });
    tx.oncomplete = () => resolve();
    tx.onerror = (e) => reject(e.target.error);
  });
}

async function removeChunkLocally(sessionId, chunkIndex) {
  const db = await openDb();
  return new Promise((resolve) => {
    const tx = db.transaction(DB_STORE, 'readwrite');
    tx.objectStore(DB_STORE).delete(`${sessionId}_${chunkIndex}`);
    tx.oncomplete = () => resolve();
  });
}

async function getPendingChunks(sessionId) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(DB_STORE, 'readonly');
    const req = tx.objectStore(DB_STORE).getAll();
    req.onsuccess = (e) => {
      const all = e.target.result.filter((c) => c.sessionId === sessionId);
      all.sort((a, b) => a.chunkIndex - b.chunkIndex);
      resolve(all);
    };
    req.onerror = (e) => reject(e.target.error);
  });
}

// ── Choisir le meilleur codec audio ───────────────────────────────────────

function getBestMimeType() {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4',
  ];
  for (const mime of candidates) {
    if (MediaRecorder.isTypeSupported(mime)) return mime;
  }
  return '';
}

// ── Audio silencieux (NoSleep) — garde le navigateur actif en arriere-plan ─
// Quand un element <audio> joue, Android Chrome ne suspend pas le processus.

function createSilentAudio() {
  const sampleRate = 8000;
  const numSamples = sampleRate; // 1 seconde
  const buf = new ArrayBuffer(44 + numSamples);
  const v = new DataView(buf);
  const str = (off, s) => s.split('').forEach((c, i) => v.setUint8(off + i, c.charCodeAt(0)));
  str(0, 'RIFF'); v.setUint32(4, 36 + numSamples, true);
  str(8, 'WAVE'); str(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate, true);
  v.setUint16(32, 1, true); v.setUint16(34, 8, true);
  str(36, 'data'); v.setUint32(40, numSamples, true);
  // Les donnees restent a zero = silence
  const url = URL.createObjectURL(new Blob([buf], { type: 'audio/wav' }));
  const audio = new Audio(url);
  audio.loop = true;
  audio.volume = 0.001; // inaudible
  return { audio, url };
}

// ── Hook ───────────────────────────────────────────────────────────────────

export function useAudioRecorder({ sessionId, onChunkUploaded, onError, onLog, deviceId }) {
  const [state, setState] = useState('idle'); // idle | recording | paused | uploading | done | error
  const [elapsed, setElapsed] = useState(0);
  const [chunksSent, setChunksSent] = useState(0);
  const [uploadQueue, setUploadQueue] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);

  const mediaRecorderRef  = useRef(null);
  const streamRef         = useRef(null);
  const wakeLockRef       = useRef(null);
  const chunkIndexRef     = useRef(0);
  const elapsedTimerRef   = useRef(null);
  const startTimeRef      = useRef(null);
  const uploadingRef      = useRef(false);
  const pendingBlobsRef   = useRef([]);
  const uploadPromisesRef = useRef([]);
  const audioContextRef   = useRef(null);
  const analyserRef       = useRef(null);
  const animFrameRef      = useRef(null);
  const silentAudioRef    = useRef(null); // element <audio> NoSleep
  const stateRef          = useRef('idle'); // miroir sans closure perimee

  const log = useCallback((msg) => { if (onLog) onLog(msg); }, [onLog]);

  // Maintenir stateRef synchronise
  const setStateAndRef = useCallback((s) => {
    stateRef.current = s;
    setState(s);
  }, []);

  // ── Wake Lock ────────────────────────────────────────────────────────────

  const acquireWakeLock = useCallback(async () => {
    if (!('wakeLock' in navigator)) return;
    if (wakeLockRef.current && !wakeLockRef.current.released) return;
    try {
      wakeLockRef.current = await navigator.wakeLock.request('screen');
      wakeLockRef.current.addEventListener('release', () => {
        // Re-acquisition uniquement si on enregistre ET que la page est visible
        if (stateRef.current === 'recording' && document.visibilityState === 'visible') {
          acquireWakeLock();
        }
      });
    } catch (e) {
      log('Wake Lock non disponible : ' + e.message);
    }
  }, [log]);

  const releaseWakeLock = useCallback(() => {
    if (wakeLockRef.current) {
      wakeLockRef.current.release().catch(() => {});
      wakeLockRef.current = null;
    }
  }, []);

  // ── Audio silencieux (NoSleep) ────────────────────────────────────────────

  const startNoSleep = useCallback(() => {
    if (silentAudioRef.current) return;
    try {
      const { audio, url } = createSilentAudio();
      silentAudioRef.current = { audio, url };
      audio.play().catch(() => {});
    } catch (e) {}
  }, []);

  const stopNoSleep = useCallback(() => {
    if (!silentAudioRef.current) return;
    const { audio, url } = silentAudioRef.current;
    audio.pause();
    URL.revokeObjectURL(url);
    silentAudioRef.current = null;
  }, []);

  // ── Upload d'un chunk avec retry ─────────────────────────────────────────

  const uploadChunkFn = useCallback(async (blob, index, retries) => {
    if (retries === undefined) retries = 0;
    try {
      await saveChunkLocally(sessionId, index, blob);
      setUploadQueue((q) => q + 1);
      await directionService.uploadChunk(sessionId, index, blob);
      await removeChunkLocally(sessionId, index);
      setUploadQueue((q) => Math.max(0, q - 1));
      setChunksSent((n) => n + 1);
      if (onChunkUploaded) onChunkUploaded(index);
    } catch (err) {
      if (retries < 8) {
        const delay = Math.min(2000 * Math.pow(1.5, retries), 60000);
        log(`Chunk ${index} : erreur, retry dans ${Math.round(delay / 1000)}s`);
        await new Promise((r) => setTimeout(r, delay));
        return uploadChunkFn(blob, index, retries + 1);
      }
      log(`Chunk ${index} : echec definitif — sera rejoue au prochain envoi`);
    }
  }, [sessionId, onChunkUploaded, log]);

  // ── Retry des chunks en attente (depuis IndexedDB) ───────────────────────

  const retryPending = useCallback(async () => {
    if (!sessionId || uploadingRef.current) return;
    const pending = await getPendingChunks(sessionId);
    if (!pending.length) return;
    uploadingRef.current = true;
    log(`Retry de ${pending.length} chunk(s) en attente…`);
    for (const c of pending) {
      await uploadChunkFn(c.blob, c.chunkIndex);
    }
    uploadingRef.current = false;
  }, [sessionId, uploadChunkFn, log]);

  // ── Construction du MediaRecorder ─────────────────────────────────────────

  const buildMediaRecorder = useCallback((stream) => {
    const mime = getBestMimeType();
    const mr = new MediaRecorder(stream, mime ? { mimeType: mime } : {});
    mr.ondataavailable = (e) => {
      if (!e.data || e.data.size === 0) return;
      const idx = chunkIndexRef.current++;
      pendingBlobsRef.current.push({ blob: e.data, index: idx });
      const p = uploadChunkFn(e.data, idx);
      uploadPromisesRef.current.push(p);
    };
    mr.onerror = (e) => {
      log('Erreur MediaRecorder : ' + (e.error?.message || 'inconnue'));
    };
    return mr;
  }, [uploadChunkFn, log]);

  // ── Redemarrage du MediaRecorder (apres retour au premier plan) ───────────

  const restartMediaRecorder = useCallback(() => {
    const stream = streamRef.current;
    if (!stream || !stream.active) {
      log('Flux audio perdu — impossible de reprendre automatiquement');
      return;
    }
    log('Reprise de l\'enregistrement apres retour au premier plan…');
    const mr = buildMediaRecorder(stream);
    mediaRecorderRef.current = mr;
    mr.start(CHUNK_DURATION_MS);
  }, [buildMediaRecorder, log]);

  // ── Gestion visibilitychange ──────────────────────────────────────────────

  useEffect(() => {
    const handleVisibility = async () => {
      if (document.visibilityState === 'hidden') {
        if (stateRef.current === 'recording') {
          log('Page en arriere-plan — audio silencieux maintient l\'enregistrement actif');
        }
        return;
      }

      // Page revenue au premier plan
      if (stateRef.current !== 'recording') return;

      // Reprendre l'AudioContext si suspendu
      if (audioContextRef.current?.state === 'suspended') {
        try { await audioContextRef.current.resume(); } catch (e) {}
      }

      // Re-acquerir le Wake Lock
      await acquireWakeLock();

      // Relancer le MediaRecorder s'il s'est arrete
      const mr = mediaRecorderRef.current;
      if (mr && mr.state === 'inactive') {
        restartMediaRecorder();
      }
    };

    document.addEventListener('visibilitychange', handleVisibility);
    return () => document.removeEventListener('visibilitychange', handleVisibility);
  }, [acquireWakeLock, restartMediaRecorder, log]);

  // ── Demarrer l'enregistrement ────────────────────────────────────────────

  const start = useCallback(async () => {
    if (!sessionId) { if (onError) onError('Session non creee'); return; }
    try {
      const audioConstraints = { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 };
      if (deviceId) audioConstraints.deviceId = { exact: deviceId };
      const stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
      streamRef.current = stream;

      // VU metre
      try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        audioContextRef.current = audioCtx;
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        analyserRef.current = analyser;
        audioCtx.createMediaStreamSource(stream).connect(analyser);
        const tick = () => {
          if (!analyserRef.current) return;
          const buf = new Uint8Array(analyserRef.current.frequencyBinCount);
          analyserRef.current.getByteFrequencyData(buf);
          const avg = buf.reduce((a, b) => a + b, 0) / buf.length;
          setAudioLevel(Math.min(1, avg / 55));
          animFrameRef.current = requestAnimationFrame(tick);
        };
        animFrameRef.current = requestAnimationFrame(tick);
      } catch (_) {}

      // Lancer l'audio silencieux AVANT Wake Lock (maximise la compatibilite mobile)
      startNoSleep();
      await acquireWakeLock();

      const mr = buildMediaRecorder(stream);
      mediaRecorderRef.current = mr;
      mr.start(CHUNK_DURATION_MS);
      startTimeRef.current = Date.now();
      setStateAndRef('recording');

      elapsedTimerRef.current = setInterval(() => {
        const el = Date.now() - startTimeRef.current;
        setElapsed(el);
        if (el >= MAX_DURATION_MS) {
          log('Duree maximale (3h) atteinte — arret automatique');
          stop();
        }
      }, 1000);

      log('Enregistrement demarre');
    } catch (err) {
      if (onError) onError('Impossible d\'acceder au microphone : ' + err.message);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, acquireWakeLock, startNoSleep, buildMediaRecorder, setStateAndRef, log, onError, deviceId]);

  // ── Arret de l'enregistrement ────────────────────────────────────────────

  const stop = useCallback(async () => {
    clearInterval(elapsedTimerRef.current);
    cancelAnimationFrame(animFrameRef.current);
    animFrameRef.current = null;
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    setAudioLevel(0);
    releaseWakeLock();
    stopNoSleep();

    const mr = mediaRecorderRef.current;
    if (mr && mr.state !== 'inactive') {
      await new Promise((resolve) => {
        mr.onstop = resolve;
        mr.stop();
      });
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    setStateAndRef('uploading');
    log('Arret de l\'enregistrement — envoi des derniers chunks…');
    if (uploadPromisesRef.current.length > 0) {
      await Promise.allSettled(uploadPromisesRef.current);
      uploadPromisesRef.current = [];
    }
    await retryPending();
    setStateAndRef('done');
    log('Tous les chunks envoyes');
  }, [releaseWakeLock, stopNoSleep, retryPending, setStateAndRef, log]);

  // Cleanup
  useEffect(() => {
    return () => {
      clearInterval(elapsedTimerRef.current);
      releaseWakeLock();
      stopNoSleep();
    };
  }, [releaseWakeLock, stopNoSleep]);

  const formatElapsed = (ms) => {
    const s = Math.floor(ms / 1000);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    if (h > 0) return `${h}h${String(m).padStart(2, '0')}m${String(sec).padStart(2, '0')}s`;
    return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
  };

  return {
    state,
    elapsed,
    elapsedStr: formatElapsed(elapsed),
    chunksSent,
    uploadQueue,
    audioLevel,
    start,
    stop,
    retryPending,
  };
}
