# -*- coding: utf-8 -*-
"""
Parakeet TDT 1.1B + TPA (two parallel FFN adapters per layer) — Kaggle one-cell trainer.

Paste this entire file into ONE Kaggle notebook cell.
For the original 0.6B chained-bottleneck adapter pipeline, use
``kaggle_nemo_parakeet_tdt_0.6b_single_cell.py`` instead.

  Part A — pip install (clean env + NeMo)
  Part B — write training script to /kaggle/working
  Part C — subprocess.run (fresh Python interpreter)

Pipeline:
  - Model: nvidia/parakeet-tdt-1.1b (default; override MODEL_ID)
  - Stage 1: NUM_EPOCHS (default 6) — TPA encoder adapters + joint + partial decoder; encoder frozen
  - Stage 2: STAGE2_EPOCHS (default 1) — full encoder fine-tune; clip 0.5, warmup 50 steps
  - Pretrained weights: ASRModel.from_pretrained(MODEL_ID) from Hugging Face Hub (MODEL_ID env)
  - TPA encoder adapters: two parallel residual adapters per layer (FFN1 + FFN2 hooks on NeMo ConformerFeedForward)
  - Optional ENCODER_DIM / ADAPTER_DIM env (else infer hidden size from encoder; ADAPTER_DIM defaults to encoder dim)
  - Merged dataset: Real child speech + childrenized LibriSpeech (manifest.jsonl + wav tree; UPPERCASE text normalized later)
  - Fallback: classic LIBRISPEECH_DIR walk if childrenized root/manifest missing
  - 50% of validation data used for training (due to child data shortage)
"""

import os
import subprocess
import sys

# =============================================================================
# ENVIRONMENT / PATH CONFIGURATION
# Set these BEFORE running the cell.  Every os.environ line below is passed
# through to the training subprocess automatically.
# =============================================================================
# -- Hugging Face (gated model download) --
# os.environ["HF_TOKEN"] = "<your-hf-read-token>"

# -- Child speech manifests --
# os.environ["CHILD_TRAIN_MANIFEST"] = "/kaggle/input/datasets/akarshks/noise/train_manifest.jsonl"
# os.environ["CHILD_VAL_MANIFEST"]   = "/kaggle/input/datasets/akarshks/noise/val_manifest.jsonl"

# -- Childrenized LibriSpeech (from childrenize_librispeech.py) --
# os.environ["LIBRISPEECH_CHILDRENIZED_ROOT"] = (
#     "/kaggle/input/datasets/megamu2/librispeech-childrenized-5000/child_synthetic_librispeech"
# )
# os.environ["LIBRISPEECH_CHILDRENIZED_MANIFEST"] = (
#     "/kaggle/input/datasets/megamu2/librispeech-childrenized-5000/child_synthetic_librispeech/manifest.jsonl"
# )

# -- OR pre-built manifest (from prepare_librispeech_manifest.py; takes priority) --
# os.environ["LIBRISPEECH_MANIFEST"] = "/kaggle/working/librispeech_normalised.jsonl"

# -- OR raw LibriSpeech tree (fallback) --
# os.environ["LIBRISPEECH_DIR"] = "/kaggle/input/LIBRISPEECH/LibriSpeech/"

# -- Classroom noise directories --
# os.environ["CLASSROOM_NOISE_DIR_1"] = "/kaggle/input/datasets/akarshks/noise/noise_part_1"
# os.environ["CLASSROOM_NOISE_DIR_2"] = "/kaggle/input/datasets/akarshkumarshukla/asr-data/noise_part_0"

# -- Training hyperparams (all optional, shown with defaults) --
# os.environ["MODEL_ID"]       = "nvidia/parakeet-tdt-1.1b"
# os.environ["ENCODER_DIM"]    = "1024"   # optional; normally inferred from first Conformer FFN
# os.environ["ADAPTER_DIM"]    = "1024"   # optional; defaults to encoder hidden size
# os.environ["BATCH_SIZE"]     = "64"
# os.environ["NUM_EPOCHS"]     = "6"
# os.environ["STAGE2_EPOCHS"]  = "1"
# os.environ["MANIFEST_SEED"]  = "42"
# os.environ["USE_EXISTING_MANIFESTS"] = "0"

# =============================================================================
# PART A: INSTALLATION (version cleaning + pip installs)
# =============================================================================


def _pip(*args):
    subprocess.check_call(
        [sys.executable, "-m", "pip", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


# Step 1: Clean conflicting packages
print("Step 1: Cleaning...")
for _ in range(2):  # Run twice to ensure clean
    subprocess.run(
        [
            sys.executable, "-m", "pip", "uninstall", "-y",
            "numpy", "scipy", "nemo_toolkit", "lightning", "pytorch-lightning",
            "datasets", "diffusers", "gradio", "peft", "sentence-transformers",
            "transformers", "huggingface_hub", "torch", "torchaudio", "torchvision",
            "numba",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

# Step 2: numpy + scipy
print("Step 2: numpy + scipy...")
_pip("install", "--no-cache-dir", "numpy>=2.1,<2.3", "scipy>=1.14,<1.16")

# Step 3: PyTorch (CUDA 12.6)
print("Step 3: PyTorch (CUDA 12.6 wheels)...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--no-cache-dir",
    "--index-url", "https://download.pytorch.org/whl/cu126",
    "torch>=2.9.0", "torchaudio",
], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

# Step 4: Dependencies
print("Step 4: Dependencies...")
_pip("install", "--no-cache-dir",
    "transformers>=4.57.6,<4.58",
    "huggingface_hub>=0.30.0",
    "lightning>=2.2.0",
    "omegaconf>=2.3.0",
    "hydra-core>=1.3.2",
    "soundfile>=0.12.0",
    "librosa>=0.10.0",
    "sentencepiece>=0.2.0",
    "datasets>=2.18.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.4.0",
    "loguru>=0.7.0",
    "jiwer>=3.0.0",
    "tqdm>=4.60.0",
    "webdataset>=0.2.80",
    "braceexpand>=0.1.7",
    "editdistance>=0.6.0",
    "whisper-normalizer",
)

# Step 5: NeMo
print("Step 5: NeMo...")
_pip("install", "--no-cache-dir", "nemo_toolkit[asr]>=2.5.0")

# Step 6: Re-pin numpy (NeMo may override)
print("Step 6: Re-pin numpy...")
_pip("install", "--no-cache-dir", "--force-reinstall",
     "numpy>=2.1,<2.3", "scipy>=1.14,<1.16")

print("Installation complete.\n")

# =============================================================================
# PART B: WRITE SELF-CONTAINED TRAINING SCRIPT
# =============================================================================
TRAIN_SCRIPT = "/kaggle/working/train_nemo_adapter_1.1b_tpa.py"
train_code = r'''# -*- coding: utf-8 -*-
"""NeMo Parakeet-TDT ~1.1B TPA (two parallel FFN adapters per layer) training (subprocess worker).

Stage 1: TPA encoder adapters (+ joint / partial decoder), encoder backbone frozen.
Stage 2: full encoder (1 epoch default), tighter clip, minimal warmup.

TPA approximation: forward hooks on each layer's first/second ConformerFeedForward; hook output =
FFN(h) + ParallelResidualAdapter(h), so adapter runs parallel to the FFN trunk (shared LN input h).

Libri merge: default LIBRISPEECH_CHILDRENIZED_ROOT + manifest.jsonl from childrenize_librispeech.py
(relative audio_filepath + UPPERCASE text). Override with LIBRISPEECH_MANIFEST env; if missing, LIBRISPEECH_DIR + .trans.txt walk.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import wave
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf, open_dict
from whisper_normalizer.english import EnglishTextNormalizer

# -----------------------------------------------------------------------------
# Config (env overrides)
# -----------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "nvidia/parakeet-tdt-1.1b").strip()
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "6"))
# Cap max utterance length for training: very large values + batch_size + DataLoader workers
# caused SIGKILL (OOM) on Kaggle (logs: max_duration_sec=200, batch_size=64, num_workers=2).
_MAX_DURATION_REQUESTED = float(os.environ.get("MAX_DURATION_SEC", "20.0"))
_MAX_DURATION_CAP = float(os.environ.get("MAX_DURATION_CAP", "40.0"))
MAX_DURATION_SEC = min(_MAX_DURATION_REQUESTED, _MAX_DURATION_CAP)
if MAX_DURATION_SEC < _MAX_DURATION_REQUESTED:
    print(
        f"[warn] MAX_DURATION_SEC requested {_MAX_DURATION_REQUESTED} capped to {MAX_DURATION_SEC} "
        f"(MAX_DURATION_CAP={_MAX_DURATION_CAP}) to avoid DataLoader worker OOM.",
        flush=True,
    )
NORMALIZE_TEXT = True
MANIFEST_SEED = int(os.environ.get("MANIFEST_SEED", "42"))
TRAIN_DEBUG_STEPS = int(os.environ.get("TRAIN_DEBUG_STEPS", "0"))

LR_ADAPTERS = 5e-4
LR_JOINT = 1e-4
LR_DECODER = 5e-5
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.15
MIN_LR = 1e-6
GRADIENT_CLIP_VAL = 1.0
PRECISION = "bf16-mixed"
SAVE_EPOCHS = [1, 2, 4]

# Stage 2
STAGE2_EPOCHS = int(os.environ.get("STAGE2_EPOCHS", "1"))
STAGE2_ENCODER_LR = float(os.environ.get("STAGE2_ENCODER_LR", "3e-5"))
STAGE2_ADAPTER_LR = float(os.environ.get("STAGE2_ADAPTER_LR", "3e-5"))
STAGE2_JOINT_LR = float(os.environ.get("STAGE2_JOINT_LR", "1e-5"))
STAGE2_DECODER_LR = float(os.environ.get("STAGE2_DECODER_LR", "5e-6"))
STAGE2_GRADIENT_CLIP = float(os.environ.get("STAGE2_GRADIENT_CLIP", "0.5"))
STAGE2_WARMUP_STEPS = int(os.environ.get("STAGE2_WARMUP_STEPS", "50"))
STAGE2_MIN_LR = float(os.environ.get("STAGE2_MIN_LR", "1e-7"))
STAGE2_UNIFORM_LR = os.environ.get("STAGE2_UNIFORM_LR", "0").strip() in ("1", "true", "True", "yes")
STAGE2_UNIFORM_LR_VALUE = float(os.environ.get("STAGE2_UNIFORM_LR_VALUE", "1e-5"))

WAVEFORM_PITCH_PROB_CHILD = float(os.environ.get("WAVEFORM_PITCH_PROB_CHILD", "0.2"))
WAVEFORM_PITCH_PROB_OTHER = float(os.environ.get("WAVEFORM_PITCH_PROB_OTHER", "0.5"))

WORK_ROOT = "/kaggle/working/nemo_adapter_1.1b_tpa"
MANIFEST_DIR = os.path.join(WORK_ROOT, "manifests")
CKPT_DIR = os.path.join(WORK_ROOT, "checkpoints")
TB_DIR = os.path.join(WORK_ROOT, "tensorboard_logs")
COMBINED_MANIFEST = os.path.join(MANIFEST_DIR, "train_manifest.jsonl")
LIBRI_MONO_CACHE = os.path.join(WORK_ROOT, "libri_mono_cache")

USE_EXISTING_MANIFESTS = os.environ.get("USE_EXISTING_MANIFESTS", "0").strip() in ("1", "true", "True", "yes")

CHILD_TRAIN_MANIFEST = os.environ.get(
    "CHILD_TRAIN_MANIFEST",
    "/kaggle/input/CHILD_DATASET/train_manifest.jsonl",
)
CHILD_VAL_MANIFEST = os.environ.get(
    "CHILD_VAL_MANIFEST",
    "/kaggle/input/CHILD_DATASET/val_manifest.jsonl",
)
LIBRISPEECH_DIR = os.environ.get(
    "LIBRISPEECH_DIR",
    "/kaggle/input/LIBRISPEECH/LibriSpeech/",
)
LIBRISPEECH_CHILDRENIZED_ROOT = os.environ.get(
    "LIBRISPEECH_CHILDRENIZED_ROOT",
    "/kaggle/input/datasets/megamu2/librispeech-childrenized-5000/child_synthetic_librispeech",
).strip()
LIBRISPEECH_CHILDRENIZED_MANIFEST = os.environ.get(
    "LIBRISPEECH_CHILDRENIZED_MANIFEST",
    "",
).strip()
if not LIBRISPEECH_CHILDRENIZED_MANIFEST:
    LIBRISPEECH_CHILDRENIZED_MANIFEST = os.path.join(
        LIBRISPEECH_CHILDRENIZED_ROOT, "manifest.jsonl"
    )
# Pre-built manifest from prepare_librispeech_manifest.py (absolute paths, normalised text).
# Takes priority over LIBRISPEECH_CHILDRENIZED_* when set.
LIBRISPEECH_MANIFEST = os.environ.get("LIBRISPEECH_MANIFEST", "").strip()

CLASSROOM_NOISE_DIR_1 = os.environ.get("CLASSROOM_NOISE_DIR_1", "/kaggle/input/classroom-noise-1/")
CLASSROOM_NOISE_DIR_2 = os.environ.get("CLASSROOM_NOISE_DIR_2", "/kaggle/input/classroom-noise-2/")

CLASSROOM_NOISE_DIRS = [CLASSROOM_NOISE_DIR_1, CLASSROOM_NOISE_DIR_2]

# -----------------------------------------------------------------------------
# Paths + helpers
# -----------------------------------------------------------------------------
os.makedirs(MANIFEST_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)
os.makedirs(LIBRI_MONO_CACHE, exist_ok=True)

_english_norm = EnglishTextNormalizer()


def _whisper_normalizer(text: str) -> str:
    return _english_norm(text)


def collect_noise_files(directories: List[str]) -> List[str]:
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    files: List[str] = []
    for d in directories:
        if not d or not os.path.isdir(d):
            continue
        for root, _, filenames in os.walk(d):
            for f in filenames:
                if os.path.splitext(f)[1].lower() in exts:
                    files.append(os.path.join(root, f))
    return files


CLASSROOM_NOISE_FILES = collect_noise_files(CLASSROOM_NOISE_DIRS)


# #region agent log
def _agent_debug_log(location: str, message: str, data: Dict[str, Any], hypothesis_id: str = "") -> None:
    import time

    payload = {
        "sessionId": "1cff8e",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    line = json.dumps(payload, default=str) + "\n"
    _paths = []
    _e = (os.environ.get("DEBUG_LOG_PATH") or "").strip()
    if _e:
        _paths.append(_e)
    _paths.append("/kaggle/working/debug-1cff8e.log")
    _paths.append(os.path.join(os.getcwd(), "debug-1cff8e.log"))
    for _p in _paths:
        try:
            _d = os.path.dirname(_p)
            if _d and not os.path.isdir(_d):
                continue
            with open(_p, "a", encoding="utf-8") as _fp:
                _fp.write(line)
            break
        except Exception:
            continue


# #endregion


def convert_to_mono_if_needed(audio_path: str, output_path: Optional[str] = None) -> str:
    """Convert stereo audio to mono. Returns path to mono file (never writes under read-only input)."""
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        if output_path is not None:
            out_path = output_path
        else:
            ext = os.path.splitext(audio_path)[1] or ".wav"
            key = hashlib.sha256(os.path.abspath(audio_path).encode("utf-8")).hexdigest()[:32] + ext
            out_path = os.path.join(LIBRI_MONO_CACHE, key)
        torchaudio.save(out_path, waveform, sample_rate)
        return out_path
    return audio_path


def _info_num_channels(info: Any) -> int:
    for attr in ("num_channels", "channels", "num_channel"):
        if hasattr(info, attr):
            v = getattr(info, attr)
            if v is not None:
                return int(v)
    return 1


def _audio_channels_duration_header(audio_path: str) -> Optional[Tuple[int, float]]:
    """Channel count + duration from container header only (no decode). soundfile first, then stdlib wave."""
    try:
        import soundfile as sf

        inf = sf.info(audio_path)
        return int(inf.channels), float(inf.duration)
    except Exception:
        pass
    if os.path.splitext(audio_path)[1].lower() != ".wav":
        return None
    try:
        with wave.open(audio_path, "rb") as w:
            nc = w.getnchannels()
            sr = float(w.getframerate())
            nf = w.getnframes()
            dur = (nf / sr) if sr > 0 else 0.0
            return int(nc), float(dur)
    except Exception:
        return None


def ensure_mono_audio_path(audio_path: str, duration_fallback: float) -> Tuple[str, float]:
    """
    Point to mono audio for NeMo collate. Header probe (soundfile/wave) avoids torchaudio.info on
    typical mono files; decode + downmix only when channels > 1 or header read fails.
    """
    if not audio_path or not os.path.isfile(audio_path):
        return audio_path, duration_fallback
    hdr = _audio_channels_duration_header(audio_path)
    if hdr is not None:
        nc, dur_hdr = hdr
        if nc <= 1:
            dur = dur_hdr if dur_hdr > 0 else duration_fallback
            return audio_path, dur if dur > 0 else duration_fallback
        out = convert_to_mono_if_needed(audio_path)
        h2 = _audio_channels_duration_header(out)
        if h2 is not None and h2[1] > 0:
            return out, h2[1]
        try:
            info2 = torchaudio.info(out)
            sr2 = float(info2.sample_rate)
            dur2 = float(info2.num_frames) / sr2 if sr2 > 0 else duration_fallback
            return out, dur2 if dur2 > 0 else duration_fallback
        except Exception:
            return out, duration_fallback
    try:
        info = torchaudio.info(audio_path)
        sr = float(info.sample_rate)
        nc = _info_num_channels(info)
        if nc <= 1:
            dur = float(info.num_frames) / sr if sr > 0 else duration_fallback
            return audio_path, dur if dur > 0 else duration_fallback
        out = convert_to_mono_if_needed(audio_path)
        info2 = torchaudio.info(out)
        sr2 = float(info2.sample_rate)
        dur2 = float(info2.num_frames) / sr2 if sr2 > 0 else duration_fallback
        return out, dur2 if dur2 > 0 else duration_fallback
    except Exception:
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] <= 1:
                dur = float(waveform.shape[-1]) / float(sr)
                return audio_path, dur if dur > 0 else duration_fallback
            out = convert_to_mono_if_needed(audio_path)
            info2 = torchaudio.info(out)
            sr2 = float(info2.sample_rate)
            dur2 = float(info2.num_frames) / sr2 if sr2 > 0 else duration_fallback
            return out, dur2 if dur2 > 0 else duration_fallback
        except Exception:
            return audio_path, duration_fallback


def apply_mono_to_manifest_record(rec: Dict[str, Any]) -> None:
    ap = rec.get("audio_filepath") or rec.get("audio_file")
    if not ap:
        return
    ap = os.path.normpath(os.path.expanduser(str(ap)))
    if not os.path.isfile(ap):
        return
    try:
        dur_fb = float(rec.get("duration", 0) or 0)
    except (TypeError, ValueError):
        dur_fb = 0.0
    hdr = _audio_channels_duration_header(ap)
    if hdr is not None and hdr[0] <= 1:
        dur = hdr[1] if hdr[1] > 0 else dur_fb
        rec["duration"] = dur if dur > 0 else dur_fb
        return
    ap2, dur = ensure_mono_audio_path(ap, dur_fb)
    rec["audio_filepath"] = ap2
    if "audio_file" in rec:
        rec["audio_file"] = ap2
    rec["duration"] = dur


def _manifest_mono_rewrite_needed(manifest_path: str) -> bool:
    """True if any non-Libri row is stereo or lacks a fast header read (must run full mono pass)."""
    # #region agent log
    _ag = {
        "rows_parsed": 0,
        "libri_tagged": 0,
        "non_libri_no_source_key": 0,
        "non_libri_no_audio_path": 0,
        "non_libri_not_file": 0,
        "non_libri_mono_hdr_ok": 0,
    }
    # #endregion
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # #region agent log
            _ag["rows_parsed"] += 1
            # #endregion
            if rec.get("source") == "librispeech":
                # #region agent log
                _ag["libri_tagged"] += 1
                # #endregion
                continue
            # #region agent log
            if "source" not in rec:
                _ag["non_libri_no_source_key"] += 1
            # #endregion
            ap = rec.get("audio_filepath") or rec.get("audio_file")
            if not ap:
                # #region agent log
                _ag["non_libri_no_audio_path"] += 1
                # #endregion
                continue
            ap = os.path.normpath(os.path.expanduser(str(ap)))
            if not os.path.isfile(ap):
                # #region agent log
                _ag["non_libri_not_file"] += 1
                # #endregion
                continue
            hdr = _audio_channels_duration_header(ap)
            if hdr is None or hdr[0] > 1:
                # #region agent log
                _agent_debug_log(
                    "_manifest_mono_rewrite_needed:early",
                    "rewrite required",
                    {
                        **_ag,
                        "reason": "hdr_none" if hdr is None else "stereo",
                        "basename": os.path.basename(ap),
                        "ext": os.path.splitext(ap)[1].lower(),
                    },
                    "H2",
                )
                # #endregion
                return True
            # #region agent log
            _ag["non_libri_mono_hdr_ok"] += 1
            # #endregion
    # #region agent log
    _agent_debug_log(
        "_manifest_mono_rewrite_needed:done",
        "rewrite not required",
        {**_ag, "rewrite_needed": False},
        "H1",
    )
    # #endregion
    return False


def rewrite_manifest_apply_mono(manifest_path: str) -> None:
    """Rewrite JSONL when needed; skip copy if header scan shows all non-Libri clips are mono."""
    if not manifest_path or not os.path.isfile(manifest_path):
        return
    if not _manifest_mono_rewrite_needed(manifest_path):
        print(
            "Manifest mono pass: skipped full rewrite (header scan: all non-Libri clips mono).",
            flush=True,
        )
        # #region agent log
        _agent_debug_log(
            "rewrite_manifest_apply_mono:skip",
            "skipped full rewrite",
            {"manifest_path": manifest_path},
            "H2",
        )
        # #endregion
        return
    tmp = manifest_path + ".mono_tmp"
    n = 0
    n_stereo = 0
    n_libri_unchanged = 0
    with open(manifest_path, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("source") == "librispeech":
                n_libri_unchanged += 1
            else:
                ap0 = rec.get("audio_filepath") or rec.get("audio_file")
                apply_mono_to_manifest_record(rec)
                ap1 = rec.get("audio_filepath") or rec.get("audio_file")
                if ap0 != ap1:
                    n_stereo += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, manifest_path)
    print(
        f"Manifest mono pass: {n} rows ({n_libri_unchanged} LibriSpeech skipped, already mono); "
        f"{n_stereo} non-Libri paths rewritten to cached mono (stereo→mono).",
        flush=True,
    )
    # #region agent log
    _agent_debug_log(
        "rewrite_manifest_apply_mono:full",
        "full rewrite done",
        {"n": n, "n_libri_unchanged": n_libri_unchanged, "n_stereo": n_stereo},
        "H2",
    )
    # #endregion


def prepare_childrenized_audio(
    abs_path: str, manifest_duration: float
) -> Tuple[Optional[str], float]:
    """
    Childrenized WAVs are often written with soundfile/pyworld; torchaudio can fail on some builds.
    Fall back to soundfile + optional PCM_16 re-write to LIBRI_MONO_CACHE for stereo or picky decoders.
    """
    md = float(manifest_duration or 0.0)
    try:
        ap = convert_to_mono_if_needed(abs_path)
        info = torchaudio.info(ap)
        dur = info.num_frames / float(info.sample_rate)
        return ap, dur if dur > 0 else md
    except Exception as e_first:
        first_err = str(e_first)
    try:
        import soundfile as sf

        data, sr = sf.read(abs_path, always_2d=True, dtype="float32")
        if data.shape[1] > 1:
            mono = np.mean(data, axis=1).astype(np.float32, copy=False)
        else:
            mono = np.ascontiguousarray(data[:, 0])
        dur = float(len(mono)) / float(sr)
        if dur < 0.01 and md > 0.1:
            dur = md
        ext = os.path.splitext(abs_path)[1] or ".wav"
        key = hashlib.sha256(os.path.abspath(abs_path).encode("utf-8")).hexdigest()[:32] + ext
        out_path = os.path.join(LIBRI_MONO_CACHE, key)
        sf.write(out_path, mono, int(sr), subtype="PCM_16")
        return out_path, dur
    except Exception:
        pass
    if md > 0.1:
        if not getattr(prepare_childrenized_audio, "_warned_manifest_fallback", False):
            print(
                "[childrenized] torchaudio+soundfile failed for some files; "
                f"will use manifest duration where needed (example torchaudio error: {first_err[:200]})"
            )
            prepare_childrenized_audio._warned_manifest_fallback = True
        return abs_path, md
    print(f"  [childrenized] unreadable audio, skipping: {abs_path!r} ({first_err[:200]})")
    return None, 0.0


def create_librispeech_manifest(librispeech_dir: str) -> List[Dict[str, Any]]:
    """Walk LibriSpeech, parse .trans.txt, mono-convert, duration via torchaudio.info."""
    records: List[Dict[str, Any]] = []
    if not librispeech_dir or not os.path.isdir(librispeech_dir):
        print(f"LibriSpeech dir missing or not a directory: {librispeech_dir}")
        return records

    for root, _, files in os.walk(librispeech_dir):
        for fname in files:
            if not fname.endswith(".trans.txt"):
                continue
            trans_path = os.path.join(root, fname)
            with open(trans_path, "r", encoding="utf-8") as tf:
                for line in tf:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) != 2:
                        continue
                    file_id, text = parts
                    wav_path = os.path.join(root, f"{file_id}.wav")
                    if not os.path.exists(wav_path):
                        wav_path = os.path.join(root, f"{file_id}.flac")
                        if not os.path.exists(wav_path):
                            continue
                    wav_path = convert_to_mono_if_needed(wav_path)
                    try:
                        info = torchaudio.info(wav_path)
                        duration = info.num_frames / float(info.sample_rate)
                    except Exception:
                        continue
                    records.append({"audio_filepath": wav_path, "duration": duration, "text": text})
    return records


def load_childrenized_librispeech_manifest(
    audio_root: str, manifest_jsonl: str
) -> List[Dict[str, Any]]:
    """
    Load manifest.jsonl from childrenize_librispeech.py.

    *audio_filepath* entries are relative to *audio_root* (e.g. 103/1240/103-1240-0000.wav).
    *text* is often UPPERCASE; build_combined_manifest applies Whisper normalizer (or .lower()).
    """
    records: List[Dict[str, Any]] = []
    if not audio_root or not os.path.isdir(audio_root):
        print(f"Childrenized LibriSpeech root missing or not a directory: {audio_root}")
        return records
    if not manifest_jsonl or not os.path.isfile(manifest_jsonl):
        print(f"Childrenized manifest not found: {manifest_jsonl}")
        return records

    n_missing_audio = 0
    n_audio_err = 0
    with open(manifest_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rel = row.get("audio_filepath") or row.get("audio_path") or ""
            rel = str(rel).strip().replace("/", os.sep)
            if not rel:
                continue
            abs_path = (
                rel
                if os.path.isabs(rel)
                else os.path.normpath(os.path.join(audio_root, rel))
            )
            if not os.path.isfile(abs_path):
                n_missing_audio += 1
                continue
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            try:
                md = float(row.get("duration") or row.get("audio_duration_sec") or 0.0)
            except (TypeError, ValueError):
                md = 0.0
            ap, duration = prepare_childrenized_audio(abs_path, md)
            if ap is None or duration <= 0:
                n_audio_err += 1
                continue
            records.append({"audio_filepath": ap, "duration": duration, "text": text})

    print(
        f"Childrenized LibriSpeech: manifest {manifest_jsonl} + root {audio_root} -> "
        f"{len(records)} rows ({n_missing_audio} missing audio, {n_audio_err} audio read errors)"
    )
    return records


def build_combined_manifest() -> str:
    random.seed(MANIFEST_SEED)
    records: List[Dict[str, Any]] = []

    # Child train 100%
    if not os.path.isfile(CHILD_TRAIN_MANIFEST):
        raise FileNotFoundError(f"CHILD_TRAIN_MANIFEST not found: {CHILD_TRAIN_MANIFEST}")
    with open(CHILD_TRAIN_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                rec["source"] = "real_child"
                records.append(rec)
    n_child_train = len(records)
    print(f"Child train samples: {n_child_train}")

    # Child val 50%
    if not os.path.isfile(CHILD_VAL_MANIFEST):
        raise FileNotFoundError(f"CHILD_VAL_MANIFEST not found: {CHILD_VAL_MANIFEST}")
    val_records: List[Dict[str, Any]] = []
    with open(CHILD_VAL_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_records.append(json.loads(line))
    random.shuffle(val_records)
    val_sample_size = len(val_records) // 2
    for rec in val_records[:val_sample_size]:
        rec["source"] = "real_child"
        records.append(rec)
    print(f"Child val samples (50%): {val_sample_size}")

    child_total = len(records)
    print(f"Total child samples: {child_total}")

    if LIBRISPEECH_MANIFEST and os.path.isfile(LIBRISPEECH_MANIFEST):
        libri_records = []
        with open(LIBRISPEECH_MANIFEST, "r", encoding="utf-8") as _lf:
            for _line in _lf:
                _line = _line.strip()
                if _line:
                    try:
                        libri_records.append(json.loads(_line))
                    except json.JSONDecodeError:
                        pass
        print(f"LibriSpeech: loaded {len(libri_records)} from pre-built manifest {LIBRISPEECH_MANIFEST}")
    elif os.path.isdir(LIBRISPEECH_CHILDRENIZED_ROOT) and os.path.isfile(
        LIBRISPEECH_CHILDRENIZED_MANIFEST
    ):
        libri_records = load_childrenized_librispeech_manifest(
            LIBRISPEECH_CHILDRENIZED_ROOT, LIBRISPEECH_CHILDRENIZED_MANIFEST
        )
    else:
        print(
            f"[warn] Childrenized Libri not found (root={LIBRISPEECH_CHILDRENIZED_ROOT!r}, "
            f"manifest={LIBRISPEECH_CHILDRENIZED_MANIFEST!r}); falling back to LIBRISPEECH_DIR."
        )
        libri_records = create_librispeech_manifest(LIBRISPEECH_DIR)
    for rec in libri_records:
        rec["source"] = "librispeech"
        records.append(rec)
    print(f"LibriSpeech samples: {len(libri_records)}")

    random.shuffle(records)
    print(f"Total combined samples (shuffled): {len(records)}")
    print(f"  - Real child: {child_total}")
    print(f"  - LibriSpeech: {len(libri_records)}")

    print(
        "Real_child mono check: header-only (soundfile/wave) for speed; torchaudio only if needed...",
        flush=True,
    )
    for rec in records:
        if rec.get("source") != "librispeech":
            apply_mono_to_manifest_record(rec)

    for rec in records:
        if NORMALIZE_TEXT:
            rec["text"] = _whisper_normalizer(str(rec.get("text", "")))
        else:
            rec["text"] = str(rec.get("text", "")).lower()

    # #region agent log
    _pre_n = len(records)
    _bad_dur = sum(
        1
        for r in records
        if not (0.1 < float(r.get("duration", 0) or 0) <= MAX_DURATION_SEC)
    )
    _by_src = {}
    for r in records:
        s = r.get("source", "__missing__")
        _by_src[s] = _by_src.get(s, 0) + 1
    _agent_debug_log(
        "build_combined_manifest:pre_filter",
        "before duration filter",
        {
            "n": _pre_n,
            "n_fail_duration": _bad_dur,
            "max_duration_sec": MAX_DURATION_SEC,
            "counts_by_source": _by_src,
        },
        "H3",
    )
    # #endregion

    records = [r for r in records if 0.1 < float(r["duration"]) <= MAX_DURATION_SEC]

    df = pd.DataFrame(records)
    df.to_json(COMBINED_MANIFEST, orient="records", lines=True)
    print(f"Saved combined manifest: {COMBINED_MANIFEST} (rows after duration filter: {len(df)})")
    # #region agent log
    _agent_debug_log(
        "build_combined_manifest:post_filter",
        "after duration filter",
        {"n_after": len(df), "combined_manifest": COMBINED_MANIFEST},
        "H3",
    )
    # #endregion
    return COMBINED_MANIFEST


# -----------------------------------------------------------------------------
# NeMo imports (after env ready)
# -----------------------------------------------------------------------------
from nemo.collections.asr.models import ASRModel, EncDecRNNTBPEModel
from nemo.core.classes import adapter_mixins


# --- MAJOR CHANGE: TPA-style parallel residual adapters (replaces chained bottleneck design) ---


def _looks_like_nemo_conformer_feedforward(mod: nn.Module) -> bool:
    lin1 = getattr(mod, "linear1", None)
    lin2 = getattr(mod, "linear2", None)
    return isinstance(lin1, nn.Linear) and isinstance(lin2, nn.Linear)


def get_encoder_layers(encoder: nn.Module) -> List[nn.Module]:
    if hasattr(encoder, "layers") and isinstance(encoder.layers, nn.ModuleList):
        return list(encoder.layers)
    inner = getattr(encoder, "encoder", None)
    if inner is not None and hasattr(inner, "layers") and isinstance(inner.layers, nn.ModuleList):
        return list(inner.layers)
    raise RuntimeError(
        "Could not locate Conformer encoder layers on model.encoder "
        "(expected .layers ModuleList or .encoder.layers)."
    )


def find_ffn_modules_in_conformer_layer(layer: nn.Module, layer_idx: int) -> Tuple[nn.Module, nn.Module]:
    """
    Return (FFN1, FFN2) submodule instances for NeMo ConformerLayer-style blocks.

    NeMo's standard ConformerLayer uses feed_forward1 / feed_forward2 (ConformerFeedForward).
    Some variants expose ffn1 / ffn2. If explicit names are missing, we scan direct children
    for modules with linear1+linear2 (ConformerFeedForward-shaped) and require exactly two.
    """
    reasons: List[str] = []

    if hasattr(layer, "feed_forward1") and hasattr(layer, "feed_forward2"):
        f1, f2 = layer.feed_forward1, layer.feed_forward2
        if _looks_like_nemo_conformer_feedforward(f1) and _looks_like_nemo_conformer_feedforward(f2):
            return f1, f2
        reasons.append(
            f"feed_forward1/2 present but not ConformerFeedForward-shaped: "
            f"{type(f1).__name__}, {type(f2).__name__}"
        )
    else:
        reasons.append("no feed_forward1 + feed_forward2 attributes")

    if hasattr(layer, "ffn1") and hasattr(layer, "ffn2"):
        f1, f2 = layer.ffn1, layer.ffn2
        if _looks_like_nemo_conformer_feedforward(f1) and _looks_like_nemo_conformer_feedforward(f2):
            return f1, f2
        reasons.append(
            f"ffn1/2 present but not ConformerFeedForward-shaped: {type(f1).__name__}, {type(f2).__name__}"
        )
    else:
        reasons.append("no ffn1 + ffn2 attributes")

    ff_like: List[Tuple[str, nn.Module]] = []
    for name, child in layer.named_children():
        if _looks_like_nemo_conformer_feedforward(child):
            ff_like.append((name, child))

    if len(ff_like) >= 2:
        ff_like.sort(key=lambda t: t[0])
        names = [n for n, _ in ff_like]
        msg = (
            f"layer {layer_idx}: using two ConformerFeedForward-like direct children "
            f"(sorted by name): {names[:2]}"
        )
        print(f"[tpa] {msg}", flush=True)
        return ff_like[0][1], ff_like[1][1]

    raise RuntimeError(
        "TPA FFN discovery failed for encoder layer "
        f"{layer_idx} ({type(layer).__name__}). "
        "Need two Conformer-style feed-forward blocks (linear1+linear2). "
        f"Diagnostics: {'; '.join(reasons)}; "
        f"scanned {len(ff_like)} FFN-like direct children."
    )


def infer_encoder_hidden_dim(layer0: nn.Module) -> Tuple[int, str]:
    ff1, _ff2 = find_ffn_modules_in_conformer_layer(layer0, 0)
    lin1 = getattr(ff1, "linear1", None)
    if isinstance(lin1, nn.Linear) and lin1.in_features > 0:
        return int(lin1.in_features), f"inferred from {type(ff1).__name__}.linear1.in_features"
    raise RuntimeError(
        f"Cannot infer encoder hidden size from layer 0 FFN ({type(ff1).__name__}); "
        "set ENCODER_DIM in the environment."
    )


def resolve_encoder_and_adapter_dims(encoder: nn.Module) -> Tuple[int, int, str, str]:
    layers = get_encoder_layers(encoder)
    ed_env = os.environ.get("ENCODER_DIM", "").strip()
    if ed_env:
        encoder_dim = int(ed_env)
        enc_src = "ENCODER_DIM env"
    else:
        encoder_dim, enc_src = infer_encoder_hidden_dim(layers[0])

    ad_env = os.environ.get("ADAPTER_DIM", "").strip()
    if ad_env:
        adapter_dim = int(ad_env)
        ad_src = "ADAPTER_DIM env"
    else:
        adapter_dim = encoder_dim
        ad_src = "default (= encoder hidden size)"

    return encoder_dim, adapter_dim, enc_src, ad_src


class ParallelResidualAdapter(nn.Module):
    """Bottleneck-free residual adapter: LN -> down -> ReLU -> up -> dropout; up init zeros (near-identity)."""

    def __init__(self, in_features: int, adapter_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, adapter_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(adapter_dim, in_features)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h)


def log_trainable_param_counts(model: "ParakeetAdapterModel", stage_label: str) -> None:
    def _trainable_in(mod: Optional[nn.Module]) -> int:
        if mod is None:
            return 0
        return sum(p.numel() for p in mod.parameters() if p.requires_grad)

    ad1 = getattr(model, "encoder_ffn1_adapters", None)
    ad2 = getattr(model, "encoder_ffn2_adapters", None)
    n_ad = _trainable_in(ad1) + _trainable_in(ad2)
    n_enc = _trainable_in(model.encoder)
    n_joint = _trainable_in(getattr(model, "joint", None))
    n_dec = _trainable_in(getattr(model, "decoder", None))
    print(
        f"[trainable:{stage_label}] adapters={n_ad:,} encoder={n_enc:,} "
        f"joint={n_joint:,} decoder={n_dec:,}",
        flush=True,
    )


# -----------------------------------------------------------------------------
# Waveform augmentation
# -----------------------------------------------------------------------------
class WaveformAugmentor(nn.Module):
    def __init__(
        self,
        speed_min: float = 0.90,
        speed_max: float = 1.10,
        pitch_min: float = -1.0,
        pitch_max: float = 3.0,
        sample_rate: int = 16000,
        speed_prob: float = 0.5,
        pitch_prob_child: float = 0.2,
        pitch_prob_other: float = 0.5,
        classroom_noise_prob: float = 0.5,
        classroom_snr_min: float = 5.0,
        classroom_snr_max: float = 10.0,
        noise_file_paths: Optional[List[str]] = None,
        gain_prob: float = 0.3,
        gain_db_min: float = -6.0,
        gain_db_max: float = 6.0,
    ):
        super().__init__()
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.sample_rate = sample_rate
        self.speed_prob = speed_prob
        self.pitch_prob_child = pitch_prob_child
        self.pitch_prob_other = pitch_prob_other
        self.classroom_noise_prob = classroom_noise_prob
        self.classroom_snr_min = classroom_snr_min
        self.classroom_snr_max = classroom_snr_max
        self.noise_file_paths = list(noise_file_paths or [])
        self.gain_prob = gain_prob
        self.gain_db_min = gain_db_min
        self.gain_db_max = gain_db_max
        self.register_buffer("_dummy", torch.zeros(1))

    def _apply_speed(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() != 1:
            wav = wav.view(-1)
        rate = random.uniform(self.speed_min, self.speed_max)
        effects = [["speed", str(rate)], ["rate", str(self.sample_rate)]]
        try:
            aug, _ = torchaudio.sox_effects.apply_effects_tensor(wav.unsqueeze(0), self.sample_rate, effects)
            return aug.squeeze(0)
        except Exception:
            return wav

    def _apply_pitch(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() != 1:
            wav = wav.view(-1)
        n_steps = random.uniform(self.pitch_min, self.pitch_max)
        try:
            return torchaudio.functional.pitch_shift(wav, self.sample_rate, n_steps)
        except Exception:
            return wav

    def _apply_gain(self, wav: torch.Tensor) -> torch.Tensor:
        db = random.uniform(self.gain_db_min, self.gain_db_max)
        g = 10.0 ** (db / 20.0)
        return wav * g

    def _mix_noise(self, wav: torch.Tensor) -> torch.Tensor:
        if not self.noise_file_paths:
            return wav
        npath = random.choice(self.noise_file_paths)
        try:
            noise, sr = torchaudio.load(npath)
            if noise.shape[0] > 1:
                noise = noise.mean(dim=0, keepdim=True)
            noise = noise.squeeze(0)
            if sr != self.sample_rate:
                noise = torchaudio.functional.resample(noise, sr, self.sample_rate)
            if noise.numel() < wav.numel():
                reps = int(math.ceil(wav.numel() / max(noise.numel(), 1)))
                noise = noise.repeat(reps)[: wav.numel()]
            else:
                start = random.randint(0, noise.numel() - wav.numel())
                noise = noise[start : start + wav.numel()]
            snr = random.uniform(self.classroom_snr_min, self.classroom_snr_max)
            p_sig = wav.pow(2).mean().clamp_min(1e-8)
            p_n = noise.pow(2).mean().clamp_min(1e-8)
            target_noise = p_sig / (10.0 ** (snr / 10.0))
            scale = torch.sqrt(target_noise / p_n)
            mixed = wav + noise * scale
            max_abs = mixed.abs().max().clamp_min(1e-8)
            if max_abs > 0.99:
                mixed = mixed / max_abs * 0.99
            return mixed
        except Exception:
            return wav

    def forward(
        self,
        input_signal: torch.Tensor,
        input_signal_length: torch.Tensor,
        is_real_child: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return input_signal, input_signal_length
        if input_signal.dim() == 2:
            b, t = input_signal.shape
            out_list = []
            new_lens = []
            for i in range(b):
                w = input_signal[i, : int(input_signal_length[i].item())].clone()
                if random.random() < self.speed_prob:
                    w = self._apply_speed(w)
                if is_real_child is not None and bool(is_real_child[i].item()):
                    p_pitch = self.pitch_prob_child
                else:
                    p_pitch = self.pitch_prob_other
                if random.random() < p_pitch:
                    w = self._apply_pitch(w)
                if random.random() < self.classroom_noise_prob:
                    w = self._mix_noise(w)
                if random.random() < self.gain_prob:
                    w = self._apply_gain(w)
                new_lens.append(w.numel())
                out_list.append(w)
            max_len = max(new_lens) if new_lens else t
            padded = input_signal.new_zeros(b, max_len)
            for i, w in enumerate(out_list):
                padded[i, : w.numel()] = w
            return padded, torch.tensor(new_lens, device=input_signal.device, dtype=torch.long)
        return input_signal, input_signal_length


# -----------------------------------------------------------------------------
# Manifest -> sample_id flags (NeMo return_sample_id=True)
# -----------------------------------------------------------------------------
def build_audio_filepath_source_map(manifest_jsonl: str) -> Dict[str, int]:
    mp: Dict[str, int] = {}
    if not manifest_jsonl or not os.path.isfile(manifest_jsonl):
        return mp
    with open(manifest_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ap = rec.get("audio_filepath") or rec.get("audio_file")
            if not ap:
                continue
            ap = os.path.normpath(os.path.expanduser(str(ap)))
            src = rec.get("source", None)
            if src is None:
                flag = 1
            else:
                flag = 1 if src == "real_child" else 0
            mp[ap] = flag
            mp[os.path.abspath(ap)] = flag
            mp["basename:" + os.path.basename(ap)] = flag
    return mp


def _lookup_child_flag(audio_file: str, mp: Dict[str, int]) -> int:
    if not audio_file:
        return 0
    ap = os.path.normpath(os.path.expanduser(str(audio_file)))
    if ap in mp:
        return mp[ap]
    aab = os.path.abspath(ap)
    if aab in mp:
        return mp[aab]
    return mp.get("basename:" + os.path.basename(ap), 0)


def unwrap_to_manifest_dataset(ds: Any) -> Any:
    seen = set()
    cur = ds
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "manifest_processor"):
            return cur
        nxt = getattr(cur, "dataset", None)
        if nxt is cur:
            break
        cur = nxt
    raise RuntimeError("Could not find a dataset with manifest_processor on the training DataLoader.")


def attach_child_sample_flags(model: Any, manifest_jsonl: str) -> None:
    pmap = build_audio_filepath_source_map(manifest_jsonl)
    base = unwrap_to_manifest_dataset(model.train_dataloader().dataset)
    n = len(base)
    arr = np.zeros(n, dtype=np.int64)
    for i in range(n):
        sample = base.manifest_processor.collection[i]
        path = getattr(sample, "audio_file", None) or getattr(sample, "audio_filepath", "")
        arr[i] = _lookup_child_flag(str(path), pmap)
    model._child_by_sample_idx = arr
    print(f"Child flags for training: {int(arr.sum())} / {n} samples matched as real_child.")
    # #region agent log
    _sample_paths = []
    for i in range(min(3, n)):
        s = base.manifest_processor.collection[i]
        _sample_paths.append(
            str(getattr(s, "audio_file", None) or getattr(s, "audio_filepath", "") or "")
        )
    _keys_sample = list(pmap.keys())[:5]
    _agent_debug_log(
        "attach_child_sample_flags",
        "child flag attachment",
        {
            "pmap_size": len(pmap),
            "n_dataset": n,
            "n_child_matched": int(arr.sum()),
            "sample_neMo_paths": _sample_paths,
            "sample_map_keys": _keys_sample,
        },
        "H4",
    )
    # #endregion


# -----------------------------------------------------------------------------
# Model subclass (TPA hooks on FFN1 / FFN2 per encoder layer)
# -----------------------------------------------------------------------------
class ParakeetAdapterModel(EncDecRNNTBPEModel):
    def init_tpa_adapter_stack(self, encoder_dim: int, adapter_dim: int, num_layers: int) -> None:
        drop = float(os.environ.get("ADAPTER_DROPOUT", "0.1"))
        self.encoder_ffn1_adapters = nn.ModuleList(
            [ParallelResidualAdapter(encoder_dim, adapter_dim, dropout=drop) for _ in range(num_layers)]
        )
        self.encoder_ffn2_adapters = nn.ModuleList(
            [ParallelResidualAdapter(encoder_dim, adapter_dim, dropout=drop) for _ in range(num_layers)]
        )
        self._adapter_hooks: List[Any] = []
        self.waveform_augmentor: Optional[WaveformAugmentor] = None

    def training_step(self, batch, batch_idx=0):
        is_child_t = None
        if isinstance(batch, (list, tuple)) and len(batch) == 5:
            sig, sig_len, tr, tr_len, sample_id = batch
            batch = (sig, sig_len, tr, tr_len)
            idx = getattr(self, "_child_by_sample_idx", None)
            if idx is not None:
                s = sample_id.view(-1).long().cpu().numpy()
                flags = idx[s]
                is_child_t = torch.as_tensor(flags, device=sig.device, dtype=torch.bool)
        self._batch_is_real_child = is_child_t
        try:
            return super().training_step(batch, batch_idx)
        finally:
            self._batch_is_real_child = None

    def register_tpa_encoder_adapters(self) -> None:
        for h in self._adapter_hooks:
            h.remove()
        self._adapter_hooks = []
        layers = get_encoder_layers(self.encoder)
        n = len(layers)
        if n != len(self.encoder_ffn1_adapters) or n != len(self.encoder_ffn2_adapters):
            raise RuntimeError(
                f"TPA adapter count mismatch: encoder has {n} layers but "
                f"ffn1={len(self.encoder_ffn1_adapters)} ffn2={len(self.encoder_ffn2_adapters)} adapters."
            )

        for idx, layer in enumerate(layers):
            ffn1_mod, ffn2_mod = find_ffn_modules_in_conformer_layer(layer, idx)
            print(
                f"[tpa] layer {idx}: FFN1={type(ffn1_mod).__name__} "
                f"({type(ffn1_mod).__module__}) "
                f"FFN2={type(ffn2_mod).__name__} ({type(ffn2_mod).__module__})",
                flush=True,
            )
            ad1 = self.encoder_ffn1_adapters[idx]
            ad2 = self.encoder_ffn2_adapters[idx]

            # NeMo ConformerFeedForward.forward(h) -> out. Parent applies 0.5 * dropout(out) to residual.
            # We add adapter(h) to out so both see the same normalized sublayer input h (parallel path).
            def _hook_ffn1(mod, inp, out, adapter=ad1):
                h = inp[0]
                return out + adapter(h)

            def _hook_ffn2(mod, inp, out, adapter=ad2):
                h = inp[0]
                return out + adapter(h)

            self._adapter_hooks.append(ffn1_mod.register_forward_hook(_hook_ffn1))
            self._adapter_hooks.append(ffn2_mod.register_forward_hook(_hook_ffn2))

    def forward(self, *args, **kwargs):
        input_signal = kwargs.get("input_signal", None)
        input_signal_length = kwargs.get("input_signal_length", None)
        processed_signal = kwargs.get("processed_signal", None)
        processed_signal_length = kwargs.get("processed_signal_length", None)

        if input_signal is None and len(args) > 0:
            input_signal = args[0]
        if input_signal_length is None and len(args) > 1:
            input_signal_length = args[1]

        kwargs = dict(kwargs)

        if processed_signal is None and input_signal is not None:
            is_child = kwargs.pop("is_real_child_batch", None)
            if is_child is None:
                is_child = getattr(self, "_batch_is_real_child", None)
            if self.training and self.waveform_augmentor is not None:
                input_signal, input_signal_length = self.waveform_augmentor(
                    input_signal, input_signal_length, is_real_child=is_child
                )
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length
            )
            kwargs["processed_signal"] = processed_signal
            kwargs["processed_signal_length"] = processed_signal_length
            kwargs.pop("input_signal", None)
            kwargs.pop("input_signal_length", None)
            rest_args = args[2:] if len(args) > 2 else ()
            return super().forward(*rest_args, **kwargs)

        return super().forward(*args, **kwargs)

    def save_adapters(self, path: str) -> None:
        payload = {
            "format": "tpa_ffn_hooks_v1",
            "encoder_ffn1_adapters": self.encoder_ffn1_adapters.state_dict(),
            "encoder_ffn2_adapters": self.encoder_ffn2_adapters.state_dict(),
        }
        torch.save(payload, path)

    def load_adapters(self, path: str) -> None:
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(path, map_location="cpu")
        if not isinstance(data, dict):
            raise ValueError("Adapter checkpoint must be a dict.")
        if "encoder_ffn1_adapters" not in data or "encoder_ffn2_adapters" not in data:
            raise ValueError(
                "Unsupported adapter checkpoint: expected keys encoder_ffn1_adapters and "
                "encoder_ffn2_adapters (TPA format). Old chained-adapter checkpoints are not compatible."
            )
        self.encoder_ffn1_adapters.load_state_dict(data["encoder_ffn1_adapters"])
        self.encoder_ffn2_adapters.load_state_dict(data["encoder_ffn2_adapters"])


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
class SaveSelectedEpochs(Callback):
    def __init__(self, directory: str, save_epochs: List[int]):
        self.directory = directory
        self.save_epochs = list(save_epochs)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: ParakeetAdapterModel) -> None:
        epoch = trainer.current_epoch
        if epoch in self.save_epochs:
            pl_module.save_to(os.path.join(self.directory, f"model_epoch{epoch}.nemo"))
            pl_module.save_adapters(os.path.join(self.directory, f"adapter_epoch{epoch}.pt"))


class ReinforceDecoderJointTrainMode(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if hasattr(pl_module, "decoder"):
            pl_module.decoder.train()
        if hasattr(pl_module, "joint"):
            pl_module.joint.train()
        if hasattr(pl_module, "waveform_augmentor") and pl_module.waveform_augmentor is not None:
            pl_module.waveform_augmentor.train()
        if hasattr(pl_module, "spec_augmentation") and pl_module.spec_augmentation is not None:
            pl_module.spec_augmentation.train()


class SaveFinalCheckpoint(Callback):
    def __init__(self, directory: str):
        self.directory = directory

    def on_fit_end(self, trainer: Trainer, pl_module: ParakeetAdapterModel) -> None:
        pl_module.save_to(os.path.join(self.directory, "model_final.nemo"))
        pl_module.save_adapters(os.path.join(self.directory, "adapter_final.pt"))


# -----------------------------------------------------------------------------
# Freeze / trainable params
# -----------------------------------------------------------------------------
def _set_requires_grad(module: Optional[nn.Module], value: bool) -> None:
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = value


def _decoder_embedding_module(decoder: nn.Module) -> Optional[nn.Module]:
    for name in ("embedding", "embed", "prediction_embedding", "word_embedding"):
        if hasattr(decoder, name):
            m = getattr(decoder, name)
            if isinstance(m, nn.Module):
                return m
    return None


def _find_prediction_lstm(decoder: nn.Module) -> Optional[nn.LSTM]:
    lstms = [m for m in decoder.modules() if isinstance(m, nn.LSTM)]
    return lstms[-1] if lstms else None


def apply_freeze_stage1_adapters_only(model: ParakeetAdapterModel) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.encoder_ffn1_adapters.parameters():
        p.requires_grad = True
    for p in model.encoder_ffn2_adapters.parameters():
        p.requires_grad = True
    if hasattr(model, "joint") and model.joint is not None:
        for p in model.joint.parameters():
            p.requires_grad = True
    emb = _decoder_embedding_module(model.decoder)
    if emb is not None:
        for p in emb.parameters():
            p.requires_grad = True
    lstm = _find_prediction_lstm(model.decoder)
    if lstm is not None:
        n = lstm.num_layers
        last = n - 1
        for name, p in lstm.named_parameters():
            if f"_l{last}" in name or f"l{last}" in name:
                p.requires_grad = True


def apply_freeze_stage2_full_encoder(model: ParakeetAdapterModel) -> None:
    for p in model.parameters():
        p.requires_grad = False
    _set_requires_grad(model.preprocessor, False)
    for p in model.encoder.parameters():
        p.requires_grad = True
    for p in model.encoder_ffn1_adapters.parameters():
        p.requires_grad = True
    for p in model.encoder_ffn2_adapters.parameters():
        p.requires_grad = True
    if hasattr(model, "joint") and model.joint is not None:
        for p in model.joint.parameters():
            p.requires_grad = True
    emb = _decoder_embedding_module(model.decoder)
    if emb is not None:
        for p in emb.parameters():
            p.requires_grad = True
    lstm = _find_prediction_lstm(model.decoder)
    if lstm is not None:
        n = lstm.num_layers
        last = n - 1
        for name, p in lstm.named_parameters():
            if f"_l{last}" in name or f"l{last}" in name:
                p.requires_grad = True
    model.encoder.train()


def build_param_groups(model: ParakeetAdapterModel) -> List[Dict[str, Any]]:
    adapter_params: List[torch.nn.Parameter] = []
    adapter_params.extend(list(model.encoder_ffn1_adapters.parameters()))
    adapter_params.extend(list(model.encoder_ffn2_adapters.parameters()))
    joint_params: List[torch.nn.Parameter] = []
    if hasattr(model, "joint") and model.joint is not None:
        joint_params.extend(list(model.joint.parameters()))
    dec_params_high: List[torch.nn.Parameter] = []
    dec_params_low: List[torch.nn.Parameter] = []
    emb = _decoder_embedding_module(model.decoder)
    if emb is not None:
        dec_params_high.extend(list(emb.parameters()))
    lstm = _find_prediction_lstm(model.decoder)
    if lstm is not None:
        n = lstm.num_layers
        last = n - 1
        for name, p in lstm.named_parameters():
            if f"_l{last}" in name or f"l{last}" in name:
                dec_params_low.append(p)
    return [
        {"params": adapter_params, "lr": LR_ADAPTERS},
        {"params": joint_params, "lr": LR_JOINT},
        {"params": dec_params_high, "lr": LR_DECODER},
        {"params": dec_params_low, "lr": LR_DECODER},
    ]


def attach_configure_optimizers_stage1(model: ParakeetAdapterModel) -> None:
    def configure_optimizers():
        groups = build_param_groups(model)
        groups = [g for g in groups if len(g["params"]) > 0]
        opt = torch.optim.AdamW(groups, betas=BETAS, weight_decay=WEIGHT_DECAY)
        tr = getattr(model, "trainer", None)
        if tr is None or getattr(tr, "estimated_stepping_batches", None) in (None, float("inf")):
            try:
                dl = model.train_dataloader()
                steps_per_epoch = max(1, len(dl))
                total_steps = max(1, steps_per_epoch * int(NUM_EPOCHS))
            except Exception:
                total_steps = 1000
        else:
            total_steps = int(tr.estimated_stepping_batches)
        warmup_steps = int(max(1, round(WARMUP_RATIO * max(1, total_steps))))
        min_lr = MIN_LR

        def lr_lambda(step: int):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            floor = min_lr / LR_ADAPTERS
            return floor + (1.0 - floor) * cos

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }

    model.configure_optimizers = configure_optimizers


def build_param_groups_stage2(model: ParakeetAdapterModel) -> List[Dict[str, Any]]:
    enc = list(model.encoder.parameters())
    adapter = list(model.encoder_ffn1_adapters.parameters()) + list(model.encoder_ffn2_adapters.parameters())
    joint: List[torch.nn.Parameter] = []
    if hasattr(model, "joint") and model.joint is not None:
        joint = list(model.joint.parameters())
    dec_high: List[torch.nn.Parameter] = []
    dec_low: List[torch.nn.Parameter] = []
    emb = _decoder_embedding_module(model.decoder)
    if emb is not None:
        dec_high.extend(list(emb.parameters()))
    lstm = _find_prediction_lstm(model.decoder)
    if lstm is not None:
        n = lstm.num_layers
        last = n - 1
        for name, p in lstm.named_parameters():
            if f"_l{last}" in name or f"l{last}" in name:
                dec_low.append(p)
    if STAGE2_UNIFORM_LR:
        all_p = enc + adapter + joint + dec_high + dec_low
        return [{"params": all_p, "lr": STAGE2_UNIFORM_LR_VALUE}]
    return [
        {"params": enc, "lr": STAGE2_ENCODER_LR},
        {"params": adapter, "lr": STAGE2_ADAPTER_LR},
        {"params": joint, "lr": STAGE2_JOINT_LR},
        {"params": dec_high, "lr": STAGE2_DECODER_LR},
        {"params": dec_low, "lr": STAGE2_DECODER_LR},
    ]


def attach_configure_optimizers_stage2(model: ParakeetAdapterModel) -> None:
    def configure_optimizers():
        groups = build_param_groups_stage2(model)
        groups = [g for g in groups if len(g["params"]) > 0]
        opt = torch.optim.AdamW(groups, betas=BETAS, weight_decay=WEIGHT_DECAY)
        try:
            dl = model.train_dataloader()
            steps_per_epoch = max(1, len(dl))
            total_steps = max(1, steps_per_epoch * int(STAGE2_EPOCHS))
        except Exception:
            total_steps = 1000
        warmup_steps = max(1, int(STAGE2_WARMUP_STEPS))
        ref_lr = STAGE2_UNIFORM_LR_VALUE if STAGE2_UNIFORM_LR else STAGE2_ENCODER_LR
        min_lr = STAGE2_MIN_LR

        def lr_lambda(step: int):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            floor = min_lr / ref_lr
            return floor + (1.0 - floor) * cos

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }

    model.configure_optimizers = configure_optimizers


# -----------------------------------------------------------------------------
# SpecAugment config
# -----------------------------------------------------------------------------
spec_augment_cfg = {
    "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
    "freq_masks": 2,
    "freq_width": 27,
    "time_masks": 10,
    "time_width": 0.05,
}


def _encoder_target_key(model_cfg: Any) -> str:
    enc = model_cfg.encoder
    if "_target_" in enc:
        return "_target_"
    if "target" in enc:
        return "target"
    raise AttributeError("model_cfg.encoder has neither _target_ nor target")


def _apply_cuda_graph_disable(model: ASRModel) -> None:
    with open_dict(model.cfg):
        if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
            model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    if hasattr(model, "change_decoding_strategy"):
        model.change_decoding_strategy(model.cfg.decoding)


def _load_base_asr_model() -> EncDecRNNTBPEModel:
    """Load pretrained ASR from Hub: config → adapter patch → weights → CUDA graph off."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    map_loc = torch.device(dev)

    print(f"[model] (1/4) Loading config: ASRModel.from_pretrained({MODEL_ID!r}, return_config=True)", flush=True)
    model_cfg = ASRModel.from_pretrained(MODEL_ID, return_config=True)

    print("[model] (2/4) Patching encoder for adapter-aware class", flush=True)
    enc_key = _encoder_target_key(model_cfg)
    with open_dict(model_cfg):
        adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder[enc_key])
        if adapter_metadata is not None:
            model_cfg.encoder[enc_key] = adapter_metadata.adapter_class_path
            print(f"[model]     encoder {enc_key} -> {adapter_metadata.adapter_class_path}", flush=True)
        else:
            print("[model]     (no registered adapter metadata; encoder target unchanged)", flush=True)

    print(
        f"[model] (3/4) Loading weights: ASRModel.from_pretrained({MODEL_ID!r}, "
        "override_config_path=..., trainer=None)",
        flush=True,
    )
    model = ASRModel.from_pretrained(
        MODEL_ID,
        override_config_path=model_cfg,
        trainer=None,
        map_location=map_loc,
    )

    print("[model] (4/4) Disabling CUDA graph decoder (greedy decoding)", flush=True)
    _apply_cuda_graph_disable(model)
    print(f"[model] Done: {MODEL_ID!r} on {map_loc}.", flush=True)
    return model


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # #region agent log
    _agent_debug_log(
        "main:manifest_branch",
        "manifest entry",
        {
            "USE_EXISTING_MANIFESTS": USE_EXISTING_MANIFESTS,
            "COMBINED_MANIFEST": COMBINED_MANIFEST,
            "combined_exists": os.path.isfile(COMBINED_MANIFEST),
            "cwd": os.getcwd(),
        },
        "H5",
    )
    # #endregion

    if USE_EXISTING_MANIFESTS and os.path.isfile(COMBINED_MANIFEST):
        manifest_path = COMBINED_MANIFEST
        print(f"Using existing manifest: {manifest_path}")
        rewrite_manifest_apply_mono(manifest_path)
    else:
        manifest_path = build_combined_manifest()

    print(f"[config] MODEL_ID (training worker): {MODEL_ID!r}", flush=True)
    model = _load_base_asr_model()
    enc_dim, ad_dim, enc_src, ad_src = resolve_encoder_and_adapter_dims(model.encoder)
    layers = get_encoder_layers(model.encoder)
    print(f"[tpa] encoder_hidden_dim={enc_dim} ({enc_src})", flush=True)
    print(f"[tpa] adapter_hidden_dim={ad_dim} ({ad_src})", flush=True)
    print(f"[tpa] num_encoder_layers={len(layers)}", flush=True)
    print(f"[tpa] total_parallel_adapter_modules={2 * len(layers)} (FFN1 + FFN2 per layer)", flush=True)

    model.__class__ = ParakeetAdapterModel
    model.init_tpa_adapter_stack(enc_dim, ad_dim, len(layers))

    model.preprocessor.eval()
    _set_requires_grad(model.preprocessor, False)
    model.encoder.train()

    from hydra.utils import instantiate
    model.spec_augmentation = instantiate(OmegaConf.create(spec_augment_cfg))
    model.waveform_augmentor = WaveformAugmentor(
        noise_file_paths=CLASSROOM_NOISE_FILES,
        pitch_prob_child=WAVEFORM_PITCH_PROB_CHILD,
        pitch_prob_other=WAVEFORM_PITCH_PROB_OTHER,
    )

    model.register_tpa_encoder_adapters()
    apply_freeze_stage1_adapters_only(model)
    log_trainable_param_counts(model, "stage1_after_freeze")
    attach_configure_optimizers_stage1(model)

    train_cfg = OmegaConf.create({
        "manifest_filepath": manifest_path,
        "sample_rate": 16000,
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "use_start_end_token": False,
        "trim_silence": False,
        "max_duration": MAX_DURATION_SEC,
        "min_duration": 0.11,
        "is_tarred": False,
        "return_sample_id": True,
    })
    model.setup_training_data(train_cfg)
    attach_child_sample_flags(model, manifest_path)

    if TRAIN_DEBUG_STEPS > 0:
        orig_training_step = model.training_step

        def debug_training_step(batch, batch_idx=None):
            out = orig_training_step(batch, batch_idx)
            if batch_idx is not None and batch_idx < TRAIN_DEBUG_STEPS:
                print(f"[debug] batch {batch_idx} loss step ok")
            return out

        model.training_step = debug_training_step

    tb1 = TensorBoardLogger(save_dir=TB_DIR, name="stage1_tpa_ffn_adapters")
    callbacks_stage1 = [
        SaveSelectedEpochs(CKPT_DIR, SAVE_EPOCHS),
        ReinforceDecoderJointTrainMode(),
    ]

    trainer_stage1 = Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="gpu",
        devices=1,
        precision=PRECISION,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=callbacks_stage1,
        logger=tb1,
        enable_checkpointing=False,
        log_every_n_steps=25,
    )

    print(
        f"Stage 1: {NUM_EPOCHS} epochs — TPA FFN adapters + joint + partial decoder "
        f"(encoder frozen), clip={GRADIENT_CLIP_VAL}"
    )
    model.train()
    trainer_stage1.fit(model)

    model.save_to(os.path.join(CKPT_DIR, "model_stage1_end.nemo"))
    model.save_adapters(os.path.join(CKPT_DIR, "adapter_stage1_end.pt"))
    print(f"Stage 1 checkpoints: model_stage1_end.nemo, adapter_stage1_end.pt")

    if STAGE2_EPOCHS <= 0:
        model.save_to(os.path.join(CKPT_DIR, "model_final.nemo"))
        model.save_adapters(os.path.join(CKPT_DIR, "adapter_final.pt"))
        print("STAGE2_EPOCHS=0: skipped Stage 2; wrote model_final / adapter_final from Stage 1.")
        print("Training finished.")
        return

    print(
        f"Stage 2: {STAGE2_EPOCHS} epoch(s) — full encoder; "
        f"encoder_lr={STAGE2_ENCODER_LR} (uniform={STAGE2_UNIFORM_LR}), "
        f"clip={STAGE2_GRADIENT_CLIP}, warmup_steps={STAGE2_WARMUP_STEPS}"
    )
    apply_freeze_stage2_full_encoder(model)
    log_trainable_param_counts(model, "stage2_after_freeze")
    attach_configure_optimizers_stage2(model)

    tb2 = TensorBoardLogger(save_dir=TB_DIR, name="stage2_encoder")
    callbacks_stage2 = [
        ReinforceDecoderJointTrainMode(),
        SaveFinalCheckpoint(CKPT_DIR),
    ]
    trainer_stage2 = Trainer(
        max_epochs=STAGE2_EPOCHS,
        accelerator="gpu",
        devices=1,
        precision=PRECISION,
        gradient_clip_val=STAGE2_GRADIENT_CLIP,
        callbacks=callbacks_stage2,
        logger=tb2,
        enable_checkpointing=False,
        log_every_n_steps=25,
    )
    model.train()
    trainer_stage2.fit(model)
    print("Training finished (Stage 1 + Stage 2).")


if __name__ == "__main__":
    main()
'''
with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
    f.write(train_code)

# =============================================================================
# PART C: FRESH SUBPROCESS (avoids stale notebook imports)
# =============================================================================
def _bootstrap_hf_token_notebook() -> None:
    """Copy Kaggle secret into os.environ so the training subprocess inherits it."""
    if (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip():
        return
    try:
        from kaggle_secrets import UserSecretsClient

        sec = UserSecretsClient()
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "huggingface_token"):
            try:
                t = sec.get_secret(name)
            except Exception:
                continue
            if t and str(t).strip():
                t = str(t).strip()
                os.environ["HF_TOKEN"] = t
                os.environ["HUGGING_FACE_HUB_TOKEN"] = t
                print(f"[notebook] HF token loaded from Kaggle secret {name!r} for subprocess.", flush=True)
                return
    except Exception:
        pass


_bootstrap_hf_token_notebook()

result = subprocess.run(
    [sys.executable, TRAIN_SCRIPT],
    cwd="/kaggle/working",
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
)
if result.returncode != 0:
    raise RuntimeError(f"Training failed with exit code {result.returncode}")
