"""
Kaggle / NeMo inference: load FLAC without TorchCodec.

Runtime evidence: torchaudio>=2.x may use TorchCodec for FLAC; if FFmpeg libs
or ABI do not match, every torchaudio.load() prints a multi-page traceback and
can freeze the notebook. soundfile uses libsndfile and avoids that path.

Paste `safe_load_audio` from this file into run_inference.py (replace the old one).
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np
import torch
import torchaudio

# Optional debug NDJSON (same session as your notebook debug)
DEBUG_LOG_PATH = os.environ.get("DEBUG_LOG_PATH", "/kaggle/working/debug-8c96a7.log")
DEBUG_SESSION_ID = os.environ.get("DEBUG_SESSION_ID", "8c96a7")

_backend_logged = False
_fail_logged = False


def _dbg_audio(hypothesis_id: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": "audio-loader",
        "hypothesisId": hypothesis_id,
        "location": "safe_load_audio_soundfile_first.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def safe_load_audio(
    path: str,
    run_id: str = "audio",
    target_sr: int = 16000,
) -> torch.Tensor:
    """
    Load mono float32 waveform at target_sr. FLAC/WAV prefer soundfile;
    other extensions try torchaudio, then soundfile.
    """
    global _backend_logged, _fail_logged
    path = os.fspath(path).strip()
    ext = os.path.splitext(path)[1].lower()
    t0 = time.time()

    try:
        import soundfile as sf
    except ImportError:
        sf = None  # type: ignore

    def _from_soundfile() -> tuple[torch.Tensor, int]:
        if sf is None:
            raise RuntimeError("soundfile not installed")
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        # [frames, channels] -> [C, T]
        wav = torch.from_numpy(np.ascontiguousarray(data.T))
        return wav, int(sr)

    def _from_torchaudio() -> tuple[torch.Tensor, int]:
        wav, sr = torchaudio.load(path)
        return wav, int(sr)

    err_parts: list[str] = []
    wav: Optional[torch.Tensor] = None
    sr: Optional[int] = None
    backend = "none"

    # Prefer soundfile for FLAC (and WAV) — avoids TorchCodec stack on many images.
    use_sf_first = ext in (".flac", ".wav") or os.environ.get(
        "AUDIO_LOAD_SOUNDFILE_FIRST", "1"
    ).strip().lower() in ("1", "true", "yes")

    if use_sf_first and sf is not None:
        try:
            wav, sr = _from_soundfile()
            backend = "soundfile"
        except Exception as e:
            err_parts.append(f"soundfile:{type(e).__name__}:{str(e)[:120]}")

    if wav is None:
        try:
            wav, sr = _from_torchaudio()
            backend = "torchaudio"
        except Exception as e:
            err_parts.append(f"torchaudio:{type(e).__name__}:{str(e)[:120]}")
            if sf is not None and backend == "none":
                try:
                    wav, sr = _from_soundfile()
                    backend = "soundfile_fallback"
                except Exception as e2:
                    err_parts.append(f"soundfile_fallback:{type(e2).__name__}:{str(e2)[:120]}")

    if wav is None or sr is None:
        if not _fail_logged:
            _fail_logged = True
            # region agent log
            _dbg_audio(
                "H7",
                "audio load failed all backends",
                {"path_tail": path[-80:], "errors": err_parts},
            )
            # endregion
        print(f"  ⚠ Could not load {os.path.basename(path)}: {' | '.join(err_parts)}")
        return torch.zeros(target_sr, dtype=torch.float32)

    if not _backend_logged and backend.startswith("soundfile"):
        _backend_logged = True
        # region agent log
        _dbg_audio(
            "H7",
            "audio load backend selected",
            {"backend": backend, "ext": ext},
        )
        # endregion

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.squeeze(0).float()
    if wav.numel() == 0:
        return torch.zeros(target_sr, dtype=torch.float32)
    out = wav[: target_sr * 30]

    dt = time.time() - t0
    if dt > 1.5:
        # region agent log
        _dbg_audio(
            "H3",
            "slow audio load",
            {"path_tail": path[-80:], "load_s": round(dt, 3), "backend": backend},
        )
        # endregion

    return out
