# -*- coding: utf-8 -*-
"""
Paste this entire file into ONE Kaggle notebook cell.

  Part A — pip install (clean env + NeMo)
  Part B — write training script to /kaggle/working
  Part C — subprocess.run (fresh Python interpreter)

Data: merge CHILD_TRAIN_MANIFEST + LIBRISPEECH_MANIFEST (both JSONL; Libri must be pre-built).
Optional COMBINED_TRAIN_MANIFEST: skip merge if that file already exists.
USE_EXISTING_MANIFESTS=1 reuses /kaggle/working/.../train_manifest.jsonl from a prior run.

Short runs / resume: set NUM_EPOCHS (default 3 in notebook env). To start from a local checkpoint instead of HuggingFace,
set INIT_NEMO_PATH to your .nemo file. If the .nemo is base-only, optionally set ADAPTER_PT_PATH to adapter_epoch*.pt.
NEMO_RESTORE_STRICT=0 forces strict=False on restore (use if strict load fails).

Parakeet-TDT 1.1B stage 2 (embedded in this file): set TDT_11B_STAGE2_ONLY=1, INIT_NEMO_PATH to your stage-1
model.nemo, STAGE2_EPOCHS (e.g. 3). Training data matches the 0.6B worker: same DATA_WORK_ROOT (default
/kaggle/working/nemo_adapter_0.6b/manifests), USE_EXISTING_MANIFESTS, COMBINED_TRAIN_MANIFEST, CHILD_*,
LIBRISPEECH_MANIFEST, and merge logic. Optional TRAIN_MANIFEST overrides with a direct JSONL path.
"""
import os
import subprocess
import sys

# --- Notebook overrides (subprocess inherits these) ---
os.environ["MODEL_ID"] = "nvidia/parakeet-tdt-0.6b-v2"
os.environ["BATCH_SIZE"] = "40"
os.environ["NUM_EPOCHS"] = "3"
os.environ["STAGE2_EPOCHS"] = "1"
# --- Parakeet 1.1B: stage-2-only (encoder + adapters; worker embedded in Part B below) ---
# os.environ["TDT_11B_STAGE2_ONLY"] = "1"
# os.environ["INIT_NEMO_PATH"] = "/kaggle/input/your-dataset/model_final.nemo"
# os.environ["STAGE2_EPOCHS"] = "3"
# os.environ["USE_EXISTING_MANIFESTS"] = "1"   # same as 0.6B — reuse merged JSONL under DATA_WORK_ROOT
# os.environ["DATA_WORK_ROOT"] = "/kaggle/working/nemo_adapter_0.6b"
# os.environ["TRAIN_MANIFEST"] = ""  # optional: force one JSONL; leave unset to use 0.6B manifest pipeline
# os.environ["NEMO_RESTORE_STRICT"] = "0"
# 0.6B resume (when TDT_11B_STAGE2_ONLY is off): INIT_NEMO_PATH / ADAPTER_PT_PATH as before
# os.environ["ADAPTER_PT_PATH"] = "/kaggle/input/your-dataset/adapter_epoch2.pt"
os.environ["MANIFEST_SEED"] = "42"
os.environ["USE_EXISTING_MANIFESTS"] = "0"
os.environ["SKIP_MONO_CONVERSION"] = "1"
os.environ["SKIP_CHILD_VAL_MIX"] = "0"

os.environ["CHILD_TRAIN_MANIFEST"] = "/kaggle/input/datasets/akarshkumarshukla/child-mani/clean_train.jsonl"
os.environ["CHILD_VAL_MANIFEST"] = "/kaggle/input/datasets/akarshkumarshukla/child-mani/clean_val.jsonl"
os.environ["LIBRISPEECH_MANIFEST"] = "/kaggle/input/datasets/akarshkumarshukla/merge-libri/merged_clean_libri.jsonl"
os.environ["CLASSROOM_NOISE_DIR_1"] = "/kaggle/input/datasets/akarshks/noise/noise_part_1"
os.environ["CLASSROOM_NOISE_DIR_2"] = "/kaggle/input/datasets/akarshkumarshukla/asr-data/noise_part_0"
# os.environ["HF_TOKEN"] = "<your-hf-read-token>"
# os.environ["COMBINED_TRAIN_MANIFEST"] = "/path/to/ready_merged_train.jsonl"  # optional: skip merge

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

# Step 3: PyTorch (CUDA 12.6) -- exact matching versions
print("Step 3: PyTorch (CUDA 12.6 wheels)...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--no-cache-dir",
    "--index-url", "https://download.pytorch.org/whl/cu126",
    "torch==2.8.0",
    "torchaudio==2.8.0",
    "torchvision==0.23.0",
], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
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

# Step 5b: Re-pin PyTorch stack after NeMo
print("Step 5b: Re-pin torch/torchaudio after NeMo...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--no-cache-dir", "--force-reinstall",
    "--index-url", "https://download.pytorch.org/whl/cu126",
    "torch==2.8.0",
    "torchaudio==2.8.0",
    "torchvision==0.23.0",
], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

# Step 4: Dependencies

# # Step 5: NeMo
# print("Step 5: NeMo...")
# _pip("install", "--no-cache-dir", "nemo_toolkit[asr]>=2.5.0")

# Step 6: Re-pin numpy (NeMo may override)
print("Step 6: Re-pin numpy...")
_pip("install", "--no-cache-dir", "--force-reinstall",
     "numpy>=2.1,<2.3", "scipy>=1.14,<1.16")

print("Installation complete.\n")


# =============================================================================
# PART B: WRITE SELF-CONTAINED TRAINING SCRIPT
# =============================================================================
TRAIN_CODE_11B_STAGE2 = r'''# -*- coding: utf-8 -*-
"""
Parakeet-TDT 1.1B — stage 2 only (encoder + adapters trainable).

Expects a full .nemo from stage-1 training (NeMo native adapters replaced with
ChainedLinearAdapter, same as train_nemo_adapter_1.1b / Kaggle 1.1B cell).

Env:
  INIT_NEMO_PATH     — required path to stage-1 model.nemo
  STAGE2_EPOCHS      — default 3
  DATA_WORK_ROOT     — default /kaggle/working/nemo_adapter_0.6b (same manifest dir as 0.6B)
  Same manifest env as 0.6B: USE_EXISTING_MANIFESTS, COMBINED_TRAIN_MANIFEST, CHILD_*, LIBRISPEECH_MANIFEST, …
  TRAIN_MANIFEST       — optional override JSONL (highest priority if set and file exists)
  BATCH_SIZE, NUM_WORKERS (default NUM_WORKERS=0 like 0.6B)
  STAGE2_ENCODER_LR, STAGE2_ADAPTER_LR, STAGE2_JOINT_LR, STAGE2_DECODER_LR
  STAGE2_GRADIENT_CLIP, STAGE2_WARMUP_STEPS, STAGE2_MIN_LR
  STAGE2_UNIFORM_LR, STAGE2_UNIFORM_LR_VALUE
  NEMO_RESTORE_STRICT — 0 for strict=False
  SAVE_EPOCHS        — comma-separated Lightning epoch indices (0-based) to save mid-run
  CLASSROOM_NOISE_DIR_1, CLASSROOM_NOISE_DIR_2 — optional noise folders
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import types
import warnings
import wave
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=Warning, module="numba")
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf, open_dict
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil, LinearAdapter


# -----------------------------------------------------------------------------
# Must match stage-1 ChainedLinearAdapter (1024 -> 128 -> 1024, GELU post-norm)
# -----------------------------------------------------------------------------
class AdapterChainState:
    def __init__(self):
        self.prev_bottleneck = None
        self.layer_counter = 0

    def reset(self):
        self.prev_bottleneck = None
        self.layer_counter = 0

    def update(self, bottleneck):
        self.prev_bottleneck = bottleneck
        self.layer_counter += 1


class ChainedLinearAdapter(nn.Module, AdapterModuleUtil):
    def __init__(
        self,
        in_features,
        dim,
        activation="gelu",
        norm_position="post",
        dropout=0.1,
        is_first=False,
        chain_state_ref=None,
        adapter_strategy=None,
    ):
        nn.Module.__init__(self)
        assert norm_position == "post", "ChainedLinearAdapter only implements post-norm (NeMo parity)"
        self.down = nn.Linear(in_features, dim)
        self.up = nn.Linear(dim, in_features)
        self.norm = nn.LayerNorm(in_features)
        self.dropout_layer = nn.Dropout(dropout)
        self.is_first = is_first
        self.chain_state_ref = chain_state_ref
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "swish":
            self.act = nn.SiLU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.GELU()
        if not is_first:
            self.chain_proj = nn.Linear(dim, dim, bias=False)
            nn.init.orthogonal_(self.chain_proj.weight)
            with torch.no_grad():
                self.chain_proj.weight.mul_(0.9)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.setup_adapter_strategy(adapter_strategy)

    def forward(self, x):
        prev_bottleneck = None
        if self.chain_state_ref is not None and not self.is_first:
            prev_bottleneck = self.chain_state_ref.prev_bottleneck
        h = self.down(x)
        if prev_bottleneck is not None:
            if prev_bottleneck.shape[1] != h.shape[1]:
                prev_bottleneck = F.interpolate(
                    prev_bottleneck.transpose(1, 2),
                    size=h.shape[1],
                    mode="nearest",
                ).transpose(1, 2)
            h = h + self.chain_proj(prev_bottleneck)
        bottleneck = self.act(h)
        if self.chain_state_ref is not None:
            self.chain_state_ref.update(bottleneck)
        h_up = self.up(bottleneck)
        h_norm = self.norm(h_up)
        h_drop = self.dropout_layer(h_norm)
        return h_drop


LinearAdapter.register(ChainedLinearAdapter)


# -----------------------------------------------------------------------------
# Waveform augmentation (batched; matches stage-1 Kaggle script)
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
        pitch_prob: float = 0.0,
        classroom_noise_prob: float = 0.0,
        classroom_snr_min: float = 5.0,
        classroom_snr_max: float = 10.0,
        noise_file_paths: list | None = None,
        gain_prob: float = 0.2,
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
        self.pitch_prob = pitch_prob
        self.classroom_noise_prob = classroom_noise_prob
        self.classroom_snr_min = classroom_snr_min
        self.classroom_snr_max = classroom_snr_max
        self.noise_file_paths = list(noise_file_paths or [])
        self.gain_prob = gain_prob
        self.gain_db_min = gain_db_min
        self.gain_db_max = gain_db_max

    @staticmethod
    def _mix_at_snr(speech: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
        eps = 1e-8
        p_s = speech.pow(2).mean().clamp_min(eps)
        p_n = noise.pow(2).mean().clamp_min(eps)
        snr_lin = 10 ** (snr_db / 10.0)
        alpha = torch.sqrt(p_s / (snr_lin * p_n + eps))
        return speech + alpha * noise

    def _load_noise_segment(self, num_samples: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        paths = self.noise_file_paths
        if not paths:
            return None
        path = paths[random.randint(0, len(paths) - 1)]
        try:
            wav, sr = torchaudio.load(path)
        except Exception:
            return None
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        n = int(wav.shape[0])
        if n < 1:
            return None
        if n < num_samples:
            reps = (num_samples + n - 1) // n
            wav = wav.repeat(reps)[:num_samples]
        else:
            start = random.randint(0, n - num_samples)
            wav = wav[start : start + num_samples]
        return wav.to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def forward(
        self, audio_signal: torch.Tensor, signal_length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return audio_signal, signal_length

        B, T = audio_signal.shape

        if torch.rand(1).item() < self.speed_prob:
            speed_factor = self.speed_min + torch.rand(1).item() * (self.speed_max - self.speed_min)
            new_T = max(1, int(T / speed_factor))
            audio_signal = F.interpolate(
                audio_signal.unsqueeze(1),
                size=new_T,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            signal_length = (signal_length.float() / speed_factor).long()
            signal_length = signal_length.clamp(min=1, max=new_T)

        if torch.rand(1).item() < self.pitch_prob:
            n_steps = self.pitch_min + torch.rand(1).item() * (self.pitch_max - self.pitch_min)
            try:
                audio_signal = torchaudio.functional.pitch_shift(audio_signal, self.sample_rate, n_steps)
            except Exception:
                pass

        if self.noise_file_paths and torch.rand(1).item() < self.classroom_noise_prob:
            snr_lo, snr_hi = self.classroom_snr_min, self.classroom_snr_max
            for b in range(B):
                L = int(min(signal_length[b].item(), audio_signal.shape[1]))
                if L < 1:
                    continue
                noise_seg = self._load_noise_segment(L, audio_signal.device, audio_signal.dtype)
                if noise_seg is None:
                    continue
                snr = snr_lo + torch.rand(1).item() * (snr_hi - snr_lo)
                seg = audio_signal[b, :L].float()
                mixed = self._mix_at_snr(seg, noise_seg, snr)
                audio_signal[b, :L] = mixed.to(dtype=audio_signal.dtype)

        if torch.rand(1).item() < self.gain_prob:
            gain_db = self.gain_db_min + torch.rand(1).item() * (self.gain_db_max - self.gain_db_min)
            gain_linear = 10 ** (gain_db / 20)
            audio_signal = audio_signal * gain_linear

        return audio_signal, signal_length


# -----------------------------------------------------------------------------
# Init checkpoint + stage-2 schedule
# -----------------------------------------------------------------------------
INIT_NEMO_PATH = os.environ.get("INIT_NEMO_PATH", "").strip()
NEMO_RESTORE_STRICT = os.environ.get("NEMO_RESTORE_STRICT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
STAGE2_EPOCHS = int(os.environ.get("STAGE2_EPOCHS", "3"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
_MAX_DURATION_REQUESTED = float(os.environ.get("MAX_DURATION_SEC", "20.0"))
_MAX_DURATION_CAP = float(os.environ.get("MAX_DURATION_CAP", "40.0"))
MAX_DURATION_SEC = min(_MAX_DURATION_REQUESTED, _MAX_DURATION_CAP)
if MAX_DURATION_SEC < _MAX_DURATION_REQUESTED:
    print(
        f"[stage2][warn] MAX_DURATION_SEC requested {_MAX_DURATION_REQUESTED} capped to {MAX_DURATION_SEC} "
        f"(MAX_DURATION_CAP={_MAX_DURATION_CAP})",
        flush=True,
    )

# -----------------------------------------------------------------------------
# Manifests — same paths / merge as 0.6B worker (DATA_WORK_ROOT default)
# -----------------------------------------------------------------------------
DATA_WORK_ROOT = os.environ.get("DATA_WORK_ROOT", "/kaggle/working/nemo_adapter_0.6b").strip()
MANIFEST_DIR = os.path.join(DATA_WORK_ROOT, "manifests")
COMBINED_MANIFEST = os.path.join(MANIFEST_DIR, "train_manifest.jsonl")
LIBRI_MONO_CACHE = os.path.join(DATA_WORK_ROOT, "libri_mono_cache")

USE_EXISTING_MANIFESTS = os.environ.get("USE_EXISTING_MANIFESTS", "0").strip() in ("1", "true", "True", "yes")
COMBINED_TRAIN_MANIFEST = os.environ.get("COMBINED_TRAIN_MANIFEST", "").strip()
SKIP_MONO_CONVERSION = os.environ.get("SKIP_MONO_CONVERSION", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
SKIP_CHILD_VAL_MIX = os.environ.get("SKIP_CHILD_VAL_MIX", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
CHILD_TRAIN_MANIFEST = os.environ.get(
    "CHILD_TRAIN_MANIFEST",
    "/kaggle/input/CHILD_DATASET/train_manifest.jsonl",
)
CHILD_VAL_MANIFEST = os.environ.get(
    "CHILD_VAL_MANIFEST",
    "/kaggle/input/CHILD_DATASET/val_manifest.jsonl",
)
LIBRISPEECH_MANIFEST = os.environ.get("LIBRISPEECH_MANIFEST", "").strip()
MANIFEST_SEED = int(os.environ.get("MANIFEST_SEED", "42"))
NORMALIZE_TEXT = True
# If set, use this JSONL directly (skips merge / COMBINED_MANIFEST reuse below).
TRAIN_MANIFEST_OVERRIDE = os.environ.get("TRAIN_MANIFEST", "").strip()

CLASSROOM_NOISE_DIR_1 = os.environ.get("CLASSROOM_NOISE_DIR_1", "/kaggle/input/classroom-noise-1/")
CLASSROOM_NOISE_DIR_2 = os.environ.get("CLASSROOM_NOISE_DIR_2", "/kaggle/input/classroom-noise-2/")
CLASSROOM_NOISE_DIRS = [CLASSROOM_NOISE_DIR_1, CLASSROOM_NOISE_DIR_2]

STAGE2_ENCODER_LR = float(os.environ.get("STAGE2_ENCODER_LR", "3e-5"))
STAGE2_ADAPTER_LR = float(os.environ.get("STAGE2_ADAPTER_LR", "3e-5"))
STAGE2_JOINT_LR = float(os.environ.get("STAGE2_JOINT_LR", "1e-5"))
STAGE2_DECODER_LR = float(os.environ.get("STAGE2_DECODER_LR", "5e-6"))
STAGE2_GRADIENT_CLIP = float(os.environ.get("STAGE2_GRADIENT_CLIP", "0.5"))
STAGE2_WARMUP_STEPS = int(os.environ.get("STAGE2_WARMUP_STEPS", "50"))
STAGE2_MIN_LR = float(os.environ.get("STAGE2_MIN_LR", "1e-7"))
STAGE2_UNIFORM_LR = os.environ.get("STAGE2_UNIFORM_LR", "0").strip().lower() in ("1", "true", "yes")
STAGE2_UNIFORM_LR_VALUE = float(os.environ.get("STAGE2_UNIFORM_LR_VALUE", "1e-5"))
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.01
LOG_EVERY_N_STEPS = max(1, int(os.environ.get("LOG_EVERY_N_STEPS", "200")))

_save_epochs_env = os.environ.get("SAVE_EPOCHS", "").strip()
if _save_epochs_env:
    SAVE_EPOCHS = sorted({int(x.strip()) for x in _save_epochs_env.split(",") if x.strip()})
else:
    if STAGE2_EPOCHS <= 4:
        SAVE_EPOCHS = list(range(STAGE2_EPOCHS))
    else:
        SAVE_EPOCHS = [e for e in (1, 2, 4) if e < STAGE2_EPOCHS]
        if STAGE2_EPOCHS > 0 and (STAGE2_EPOCHS - 1) not in SAVE_EPOCHS:
            SAVE_EPOCHS.append(STAGE2_EPOCHS - 1)
        SAVE_EPOCHS = sorted(set(SAVE_EPOCHS))

WORK_ROOT = os.environ.get("STAGE2_WORK_ROOT", "/kaggle/working/nemo_adapter_1.1b_stage2")
CKPT_DIR = os.path.join(WORK_ROOT, "checkpoints")
os.makedirs(MANIFEST_DIR, exist_ok=True)
os.makedirs(LIBRI_MONO_CACHE, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

if SKIP_MONO_CONVERSION:
    print(
        "[stage2] SKIP_MONO_CONVERSION=1 — no manifest mono rewrite; NeMo channel_selector='average'.",
        flush=True,
    )
else:
    print(
        "[stage2] SKIP_MONO_CONVERSION=0 — legacy mono rewrite may run on manifests.",
        flush=True,
    )

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


def _probe_duration_seconds(audio_path: str, duration_fallback: float) -> float:
    """Audio duration without torchaudio (avoids torchcodec when torchaudio.load/info use it)."""
    h = _audio_channels_duration_header(audio_path)
    if h is not None and h[1] > 0:
        return float(h[1])
    try:
        import soundfile as sf

        d = float(sf.info(audio_path).duration)
        if d > 0:
            return d
    except Exception:
        pass
    return float(duration_fallback) if duration_fallback > 0 else 0.0


def convert_to_mono_if_needed(audio_path: str, output_path: Optional[str] = None) -> str:
    """Legacy SKIP_MONO_CONVERSION=0: downmix to mono. Prefers soundfile/librosa; torchaudio last (torchcodec can break on Kaggle)."""
    ext = os.path.splitext(audio_path)[1] or ".wav"
    if output_path is not None:
        out_path = output_path
    else:
        key = hashlib.sha256(os.path.abspath(audio_path).encode("utf-8")).hexdigest()[:32] + ext
        out_path = os.path.join(LIBRI_MONO_CACHE, key)

    def _write_mono_pcm16(mono: np.ndarray, sr: int, dest: str) -> str:
        import soundfile as sf

        sf.write(dest, mono, int(sr), subtype="PCM_16")
        return dest

    try:
        import soundfile as sf

        inf = sf.info(audio_path)
        if inf.channels <= 1:
            return audio_path
        data, sr = sf.read(audio_path, always_2d=True, dtype="float32")
        if data.shape[1] > 1:
            mono = np.mean(data, axis=1).astype(np.float32, copy=False)
        else:
            mono = np.ascontiguousarray(data[:, 0])
        return _write_mono_pcm16(mono, int(sr), out_path)
    except Exception:
        pass

    try:
        import librosa

        y, sr = librosa.load(audio_path, sr=None, mono=True)
        mono = np.asarray(y, dtype=np.float32)
        return _write_mono_pcm16(mono, int(sr), out_path)
    except Exception:
        pass

    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] <= 1:
            return audio_path
        waveform = waveform.mean(dim=0, keepdim=True)
        torchaudio.save(out_path, waveform, int(sample_rate))
        return out_path
    except Exception as e:
        raise RuntimeError(
            f"convert_to_mono_if_needed: failed for {audio_path!r} "
            f"(soundfile, librosa, and torchaudio all failed: {e})"
        ) from e


def ensure_mono_audio_path(audio_path: str, duration_fallback: float) -> Tuple[str, float]:
    """
    Legacy fallback (SKIP_MONO_CONVERSION=0): point to mono audio for collate. Uses header/soundfile
    first; stereo downmix via convert_to_mono_if_needed (soundfile/librosa, not torchaudio by default).
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
        dur2 = _probe_duration_seconds(out, duration_fallback)
        return out, dur2 if dur2 > 0 else duration_fallback
    try:
        import soundfile as sf

        inf = sf.info(audio_path)
        nc = int(inf.channels)
        if nc <= 1:
            d = float(inf.duration)
            return audio_path, d if d > 0 else duration_fallback
        out = convert_to_mono_if_needed(audio_path)
        dur2 = _probe_duration_seconds(out, duration_fallback)
        return out, dur2 if dur2 > 0 else duration_fallback
    except Exception:
        pass
    try:
        info = torchaudio.info(audio_path)
        sr = float(info.sample_rate)
        nc = _info_num_channels(info)
        if nc <= 1:
            dur = float(info.num_frames) / sr if sr > 0 else duration_fallback
            return audio_path, dur if dur > 0 else duration_fallback
        out = convert_to_mono_if_needed(audio_path)
        dur2 = _probe_duration_seconds(out, duration_fallback)
        return out, dur2 if dur2 > 0 else duration_fallback
    except Exception:
        return audio_path, duration_fallback


def apply_mono_to_manifest_record(rec: Dict[str, Any]) -> None:
    """Legacy path (SKIP_MONO_CONVERSION=0): rewrite record to cached mono path when stereo."""
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
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("source") == "librispeech":
                continue
            ap = rec.get("audio_filepath") or rec.get("audio_file")
            if not ap:
                continue
            ap = os.path.normpath(os.path.expanduser(str(ap)))
            if not os.path.isfile(ap):
                continue
            hdr = _audio_channels_duration_header(ap)
            if hdr is None or hdr[0] > 1:
                return True
    return False


def rewrite_manifest_apply_mono(manifest_path: str) -> None:
    """Rewrite JSONL when needed; skip copy if header scan shows all non-Libri clips are mono."""
    if not manifest_path or not os.path.isfile(manifest_path):
        return
    if SKIP_MONO_CONVERSION:
        print(
            "Manifest mono pass: skipped (SKIP_MONO_CONVERSION=1); training uses NeMo channel_selector='average'.",
            flush=True,
        )
        return
    if not _manifest_mono_rewrite_needed(manifest_path):
        print(
            "Manifest mono pass: skipped full rewrite (header scan: all non-Libri clips mono).",
            flush=True,
        )
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

    val_sample_size = 0
    if SKIP_CHILD_VAL_MIX:
        print(
            "[manifest] SKIP_CHILD_VAL_MIX=1 — child split uses CHILD_TRAIN_MANIFEST only (no val rows).",
            flush=True,
        )
    else:
        # Child val: add 50% of validation manifest into training (legacy data-shortage recipe).
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
        print(f"Child val samples (50% of val manifest): {val_sample_size}")

    child_total = len(records)
    print(f"Total child samples: {child_total}")

    if not LIBRISPEECH_MANIFEST or not os.path.isfile(LIBRISPEECH_MANIFEST):
        raise FileNotFoundError(
            "LIBRISPEECH_MANIFEST must be set to an existing JSONL (pre-built Libri / merged synthetic). "
            "Tree walk and childrenized bundles were removed."
        )
    libri_records = []
    with open(LIBRISPEECH_MANIFEST, "r", encoding="utf-8") as _lf:
        for _line in _lf:
            _line = _line.strip()
            if _line:
                try:
                    libri_records.append(json.loads(_line))
                except json.JSONDecodeError:
                    pass
    print(f"LibriSpeech: loaded {len(libri_records)} from {LIBRISPEECH_MANIFEST}")
    for rec in libri_records:
        rec["source"] = "librispeech"
        records.append(rec)
    print(f"LibriSpeech samples: {len(libri_records)}")

    random.shuffle(records)
    print(f"Total combined samples (shuffled): {len(records)}")
    print(f"  - Real child: {child_total}")
    print(f"  - LibriSpeech: {len(libri_records)}")

    if SKIP_MONO_CONVERSION:
        print(
            "[manifest] SKIP_MONO_CONVERSION=1 — skipping real_child mono rewrite; NeMo averages channels at load.",
            flush=True,
        )
    else:
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

    records = [r for r in records if 0.1 < float(r["duration"]) <= MAX_DURATION_SEC]

    df = pd.DataFrame(records)
    df.to_json(COMBINED_MANIFEST, orient="records", lines=True)
    print(f"Saved combined manifest: {COMBINED_MANIFEST} (rows after duration filter: {len(df)})")
    return COMBINED_MANIFEST


def resolve_training_manifest_path() -> str:
    """Same order as 0.6B: TRAIN_MANIFEST override → COMBINED_TRAIN_MANIFEST → USE_EXISTING → merge build."""
    if TRAIN_MANIFEST_OVERRIDE and os.path.isfile(TRAIN_MANIFEST_OVERRIDE):
        p = TRAIN_MANIFEST_OVERRIDE
        print(f"[stage2] Using TRAIN_MANIFEST override: {p}", flush=True)
        rewrite_manifest_apply_mono(p)
        return p
    if COMBINED_TRAIN_MANIFEST and os.path.isfile(COMBINED_TRAIN_MANIFEST):
        p = COMBINED_TRAIN_MANIFEST
        print(f"[stage2] Using COMBINED_TRAIN_MANIFEST (no merge): {p}", flush=True)
        rewrite_manifest_apply_mono(p)
        return p
    if USE_EXISTING_MANIFESTS and os.path.isfile(COMBINED_MANIFEST):
        p = COMBINED_MANIFEST
        print(f"[stage2] Using existing working manifest: {p}", flush=True)
        rewrite_manifest_apply_mono(p)
        return p
    p = build_combined_manifest()
    print(f"[stage2] Built combined manifest (same recipe as 0.6B): {p}", flush=True)
    return p


def _ensure_train_dataloader_channel_averaging(model: Any) -> None:
    """
    Match legacy working scripts: non-Lhotse BPE dataset with channel_selector='average'.
    If merged cfg dropped these fields, patch model.cfg.train_ds and rebuild _train_dl once.
    """
    if not hasattr(model.cfg, "train_ds"):
        print("[dataloader][warn] model.cfg.train_ds missing; cannot set use_lhotse/channel_selector.", flush=True)
        return
    td = model.cfg.train_ds
    ul = OmegaConf.select(model.cfg, "train_ds.use_lhotse")
    cs = OmegaConf.select(model.cfg, "train_ds.channel_selector")
    patched = False
    if ul is not False or cs != "average":
        with open_dict(td):
            td.use_lhotse = False
            td.channel_selector = "average"
        patched = True
        model._train_dl = model._setup_dataloader_from_config(config=model.cfg.train_ds)
    eff_ul = OmegaConf.select(model.cfg, "train_ds.use_lhotse")
    eff_cs = OmegaConf.select(model.cfg, "train_ds.channel_selector")
    print(
        f"[dataloader] Effective train_ds: use_lhotse={eff_ul!r}, channel_selector={eff_cs!r}"
        + (" — rebuilt training DataLoader after cfg patch" if patched else ""),
        flush=True,
    )
    print(
        "[dataloader] Primary multichannel handling: NeMo load-time averaging (use_lhotse=False, channel_selector='average'); "
        "no stereo→mono manifest rewrite required for normal runs.",
        flush=True,
    )
    print(
        "[dataloader] Manual mono conversion (torchaudio / manifest rewrite) is optional legacy fallback — set SKIP_MONO_CONVERSION=0.",
        flush=True,
    )


spec_augment_cfg = {
    "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
    "freq_masks": 2,
    "freq_width": 27,
    "time_masks": 10,
    "time_width": 0.05,
}


def update_model_cfg(orig_cfg, new_cfg):
    with open_dict(orig_cfg):
        for k, v in new_cfg.items():
            orig_cfg[k] = v
    return orig_cfg


def _apply_cuda_graph_disable(model: ASRModel) -> None:
    with open_dict(model.cfg):
        if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
            model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    if hasattr(model, "change_decoding_strategy"):
        model.change_decoding_strategy(model.cfg.decoding)


def _reattach_chain_state(model: ASRModel) -> int:
    cs = AdapterChainState()
    model.adapter_chain_state = cs
    n = 0
    for m in model.modules():
        if isinstance(m, ChainedLinearAdapter):
            m.chain_state_ref = cs
            n += 1
    return n


def _decoder_embedding_module(decoder: nn.Module) -> nn.Module | None:
    for name in ("embedding", "embed", "prediction_embedding", "word_embedding"):
        if hasattr(decoder, name):
            m = getattr(decoder, name)
            if isinstance(m, nn.Module):
                return m
    return None


def _find_prediction_lstm(decoder: nn.Module) -> nn.LSTM | None:
    lstms = [m for m in decoder.modules() if isinstance(m, nn.LSTM)]
    return lstms[-1] if lstms else None


def apply_freeze_stage2(model: ASRModel) -> None:
    """Encoder (base + adapters) + joint + decoder embedding + last LSTM layer."""
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "preprocessor") and model.preprocessor is not None:
        for p in model.preprocessor.parameters():
            p.requires_grad = False
    for p in model.encoder.parameters():
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
        last_i = lstm.num_layers - 1
        for name, p in lstm.named_parameters():
            if f"_l{last_i}" in name:
                p.requires_grad = True
    model.encoder.train()
    if hasattr(model, "decoder"):
        model.decoder.train()
    if hasattr(model, "joint") and model.joint is not None:
        model.joint.train()


def build_param_groups_stage2(model: ASRModel) -> list[dict]:
    enc_base: list[torch.nn.Parameter] = []
    enc_adapter: list[torch.nn.Parameter] = []
    joint: list[torch.nn.Parameter] = []
    dec: list[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nl = name.lower()
        if nl.startswith("encoder."):
            if "adapter" in nl or "asr_children" in nl:
                enc_adapter.append(p)
            else:
                enc_base.append(p)
        elif "joint" in nl:
            joint.append(p)
        else:
            dec.append(p)

    if STAGE2_UNIFORM_LR:
        all_p = enc_base + enc_adapter + joint + dec
        return [{"params": all_p, "lr": STAGE2_UNIFORM_LR_VALUE}]
    return [
        {"params": enc_base, "lr": STAGE2_ENCODER_LR},
        {"params": enc_adapter, "lr": STAGE2_ADAPTER_LR},
        {"params": joint, "lr": STAGE2_JOINT_LR},
        {"params": dec, "lr": STAGE2_DECODER_LR},
    ]


def attach_configure_optimizers_stage2(model: ASRModel) -> None:
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
        model._optimizer = opt
        model._scheduler = sched
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }

    model.configure_optimizers = configure_optimizers


class ReinforceDecoderJointTrainMode(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if hasattr(pl_module, "decoder"):
            pl_module.decoder.train()
        if hasattr(pl_module, "joint"):
            pl_module.joint.train()
        wa = getattr(pl_module, "waveform_augmentor", None)
        if wa is not None:
            wa.train()
        sa = getattr(pl_module, "spec_augmentation", None)
        if sa is not None:
            sa.train()


class SaveSelectedEpochs(Callback):
    def __init__(self, directory: str, save_epochs: list):
        super().__init__()
        self.directory = directory
        self.save_epochs = list(save_epochs)

    def on_train_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        epoch = trainer.current_epoch
        if epoch not in self.save_epochs:
            return
        nemo_path = os.path.join(self.directory, f"model_stage2_epoch{epoch}.nemo")
        adapter_path = os.path.join(self.directory, f"adapter_stage2_epoch{epoch}.pt")
        pl_module.save_to(nemo_path)
        try:
            pl_module.save_adapters(adapter_path)
        except Exception as e:
            print(f"[SaveSelectedEpochs] save_adapters failed: {e}", flush=True)
        print(f"[SaveSelectedEpochs] epoch {epoch} -> {nemo_path}", flush=True)


class SaveFinalCheckpoint(Callback):
    def __init__(self, directory: str):
        super().__init__()
        self.directory = directory

    def on_fit_end(self, trainer, pl_module):
        pl_module.save_to(os.path.join(self.directory, "model_stage2_final.nemo"))
        try:
            pl_module.save_adapters(os.path.join(self.directory, "adapter_stage2_final.pt"))
        except Exception as e:
            print(f"[SaveFinalCheckpoint] save_adapters failed: {e}", flush=True)


def _patch_training_forward(model: ASRModel) -> None:
    _original_forward = model.forward.__func__

    def _augmented_forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        if hasattr(self, "adapter_chain_state"):
            self.adapter_chain_state.reset()
        if self.training and input_signal is not None and input_signal_length is not None:
            wa = getattr(self, "waveform_augmentor", None)
            if wa is not None:
                input_signal, input_signal_length = wa(input_signal, input_signal_length)
        return _original_forward(
            self,
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
        )

    model.forward = types.MethodType(_augmented_forward, model)


def main() -> None:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not INIT_NEMO_PATH:
        raise SystemExit("Set INIT_NEMO_PATH to your stage-1 model.nemo")
    if not os.path.isfile(INIT_NEMO_PATH):
        raise SystemExit(f"INIT_NEMO_PATH not found: {INIT_NEMO_PATH}")

    manifest_path = resolve_training_manifest_path()

    print(f"[stage2] INIT_NEMO_PATH={INIT_NEMO_PATH}", flush=True)
    print(f"[stage2] manifest_path={manifest_path}", flush=True)
    print(f"[stage2] STAGE2_EPOCHS={STAGE2_EPOCHS} SAVE_EPOCHS={SAVE_EPOCHS}", flush=True)
    print(f"[stage2] Classroom noise clips: {len(CLASSROOM_NOISE_FILES)}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = ASRModel.restore_from(INIT_NEMO_PATH, map_location=dev, strict=NEMO_RESTORE_STRICT)
    except TypeError:
        model = ASRModel.restore_from(INIT_NEMO_PATH, map_location=dev)
    except Exception as e:
        if NEMO_RESTORE_STRICT:
            print(f"[stage2] strict restore failed ({e}); retrying strict=False", flush=True)
            model = ASRModel.restore_from(INIT_NEMO_PATH, map_location=dev, strict=False)
        else:
            raise

    _apply_cuda_graph_disable(model)
    n_ad = _reattach_chain_state(model)
    print(f"[stage2] Re-attached chain state to {n_ad} ChainedLinearAdapter module(s)", flush=True)

    if not CLASSROOM_NOISE_FILES:
        print("[stage2] WARNING: no classroom noise files; noise mixing disabled.", flush=True)

    if getattr(model, "spec_augmentation", None) is None:
        model.spec_augmentation = model.from_config_dict(OmegaConf.create(spec_augment_cfg))
    if not hasattr(model, "waveform_augmentor") or model.waveform_augmentor is None:
        model.add_module(
            "waveform_augmentor",
            WaveformAugmentor(
                noise_file_paths=[],
                pitch_prob=0.0,
                classroom_noise_prob=0.0,
                gain_prob=0.2,
            ),
        )
    _patch_training_forward(model)

    train_ds_patch = {
        "manifest_filepath": manifest_path,
        "sample_rate": 16000,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "use_lhotse": False,
        "channel_selector": "average",
        "is_tarred": False,
        "shuffle": True,
        "use_start_end_token": False,
        "trim_silence": False,
        "min_duration": 0.11,
        "return_sample_id": False,
    }
    cfg_td = model.cfg.train_ds
    update_model_cfg(cfg_td, train_ds_patch)
    with open_dict(cfg_td):
        cfg_td.max_duration = MAX_DURATION_SEC
    model.setup_training_data(cfg_td)
    _ensure_train_dataloader_channel_averaging(model)

    apply_freeze_stage2(model)
    attach_configure_optimizers_stage2(model)

    if hasattr(model, "waveform_augmentor"):
        model.waveform_augmentor.train()
    if getattr(model, "spec_augmentation", None) is not None:
        model.spec_augmentation.train()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[stage2] Total params: {total:,}  trainable: {trainable:,}", flush=True)

    trainer = pl.Trainer(
        max_epochs=STAGE2_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed",
        gradient_clip_val=STAGE2_GRADIENT_CLIP,
        callbacks=[
            ReinforceDecoderJointTrainMode(),
            SaveSelectedEpochs(CKPT_DIR, SAVE_EPOCHS),
            SaveFinalCheckpoint(CKPT_DIR),
        ],
        logger=False,
        enable_checkpointing=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        log_every_n_steps=LOG_EVERY_N_STEPS,
    )

    print(
        f"[stage2] Training: encoder+adapters+joint+partial decoder | "
        f"clip={STAGE2_GRADIENT_CLIP} warmup_steps={STAGE2_WARMUP_STEPS}",
        flush=True,
    )
    model.train()
    trainer.fit(model)
    print(f"[stage2] Done. Outputs under {CKPT_DIR}", flush=True)


if __name__ == "__main__":
    main()
'''

train_code = r'''# -*- coding: utf-8 -*-
"""NeMo Parakeet-TDT 0.6B chained adapter training (subprocess worker).

Stage 1: adapters (+ joint / partial decoder), encoder frozen.
Stage 2: full encoder (1 epoch default), tighter clip, minimal warmup.

Train manifest: merge CHILD_TRAIN_MANIFEST + LIBRISPEECH_MANIFEST (pre-built JSONL only).
Optional COMBINED_TRAIN_MANIFEST skips merge. No LibriSpeech tree walk or childrenized bundle.

Training: use_lhotse=False, channel_selector='average'. SKIP_MONO_CONVERSION=0 enables legacy mono rewrite.
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
import torch.nn.functional as F
import torchaudio
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf, open_dict
from whisper_normalizer.english import EnglishTextNormalizer

# -----------------------------------------------------------------------------
# Config (env overrides)
# -----------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "nvidia/parakeet-tdt-0.6b-v2").strip()
# Layer adapters use encoder output dim inferred at runtime (e.g. Parakeet-TDT 0.6B → 1024, not 512).
BOTTLENECK_DIM = 256
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))
INIT_NEMO_PATH = os.environ.get("INIT_NEMO_PATH", "").strip()
ADAPTER_PT_PATH = os.environ.get("ADAPTER_PT_PATH", "").strip()
NEMO_RESTORE_STRICT = os.environ.get("NEMO_RESTORE_STRICT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
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
# NeMo logs per-batch WER samples when (step+1) % log_every_n_steps == 0; lower = very chatty (default 200).
LOG_EVERY_N_STEPS = max(1, int(os.environ.get("LOG_EVERY_N_STEPS", "200")))

LR_ADAPTERS = 5e-4
LR_JOINT = 1e-4
LR_DECODER = 5e-5
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.15
MIN_LR = 1e-6
GRADIENT_CLIP_VAL = 1.0
PRECISION = "bf16-mixed"
_save_epochs_env = os.environ.get("SAVE_EPOCHS", "").strip()
if _save_epochs_env:
    SAVE_EPOCHS = sorted({int(x.strip()) for x in _save_epochs_env.split(",") if x.strip()})
else:
    if NUM_EPOCHS <= 4:
        SAVE_EPOCHS = list(range(NUM_EPOCHS))
    else:
        SAVE_EPOCHS = [e for e in (1, 2, 4) if e < NUM_EPOCHS]
        if NUM_EPOCHS > 0 and (NUM_EPOCHS - 1) not in SAVE_EPOCHS:
            SAVE_EPOCHS.append(NUM_EPOCHS - 1)
        SAVE_EPOCHS = sorted(set(SAVE_EPOCHS))

print(
    f"[config] NUM_EPOCHS={NUM_EPOCHS} STAGE2_EPOCHS={STAGE2_EPOCHS} "
    f"SAVE_EPOCHS(PL 0-based epoch end)={SAVE_EPOCHS} "
    f"INIT_NEMO_PATH={INIT_NEMO_PATH or '(unset → HuggingFace)'}",
    flush=True,
)

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

WAVEFORM_PITCH_PROB_CHILD = float(os.environ.get("WAVEFORM_PITCH_PROB_CHILD", "0"))
WAVEFORM_PITCH_PROB_OTHER = float(os.environ.get("WAVEFORM_PITCH_PROB_OTHER", "0"))

WORK_ROOT = "/kaggle/working/nemo_adapter_0.6b"
MANIFEST_DIR = os.path.join(WORK_ROOT, "manifests")
CKPT_DIR = os.path.join(WORK_ROOT, "checkpoints")
TB_DIR = os.path.join(WORK_ROOT, "tensorboard_logs")
COMBINED_MANIFEST = os.path.join(MANIFEST_DIR, "train_manifest.jsonl")
LIBRI_MONO_CACHE = os.path.join(WORK_ROOT, "libri_mono_cache")

USE_EXISTING_MANIFESTS = os.environ.get("USE_EXISTING_MANIFESTS", "0").strip() in ("1", "true", "True", "yes")
# If set to an existing JSONL, training uses it directly (no merge with child/Libri in this run).
COMBINED_TRAIN_MANIFEST = os.environ.get("COMBINED_TRAIN_MANIFEST", "").strip()
# When True: no manifest mono rewrite, no real_child mono pass in merge, no convert_to_mono in legacy mono paths.
SKIP_MONO_CONVERSION = os.environ.get("SKIP_MONO_CONVERSION", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# When True: child data comes only from CHILD_TRAIN_MANIFEST (no val rows added).
SKIP_CHILD_VAL_MIX = os.environ.get("SKIP_CHILD_VAL_MIX", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

CHILD_TRAIN_MANIFEST = os.environ.get(
    "CHILD_TRAIN_MANIFEST",
    "/kaggle/input/CHILD_DATASET/train_manifest.jsonl",
)
CHILD_VAL_MANIFEST = os.environ.get(
    "CHILD_VAL_MANIFEST",
    "/kaggle/input/CHILD_DATASET/val_manifest.jsonl",
)
# Pre-built Libri (or merged synthetic) JSONL — required when building the combined train manifest.
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

if SKIP_MONO_CONVERSION:
    print(
        "[config] SKIP_MONO_CONVERSION=1 — no manifest/torchaudio stereo→mono rewrite; "
        "multichannel audio is handled at train time via NeMo (use_lhotse=False, channel_selector='average').",
        flush=True,
    )
else:
    print(
        "[config] SKIP_MONO_CONVERSION=0 — legacy fallback: may rewrite manifests / convert stereo with torchaudio "
        "where enabled below.",
        flush=True,
    )

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


def _probe_duration_seconds(audio_path: str, duration_fallback: float) -> float:
    """Audio duration without torchaudio (avoids torchcodec when torchaudio.load/info use it)."""
    h = _audio_channels_duration_header(audio_path)
    if h is not None and h[1] > 0:
        return float(h[1])
    try:
        import soundfile as sf

        d = float(sf.info(audio_path).duration)
        if d > 0:
            return d
    except Exception:
        pass
    return float(duration_fallback) if duration_fallback > 0 else 0.0


def convert_to_mono_if_needed(audio_path: str, output_path: Optional[str] = None) -> str:
    """Legacy SKIP_MONO_CONVERSION=0: downmix to mono. Prefers soundfile/librosa; torchaudio last (torchcodec can break on Kaggle)."""
    ext = os.path.splitext(audio_path)[1] or ".wav"
    if output_path is not None:
        out_path = output_path
    else:
        key = hashlib.sha256(os.path.abspath(audio_path).encode("utf-8")).hexdigest()[:32] + ext
        out_path = os.path.join(LIBRI_MONO_CACHE, key)

    def _write_mono_pcm16(mono: np.ndarray, sr: int, dest: str) -> str:
        import soundfile as sf

        sf.write(dest, mono, int(sr), subtype="PCM_16")
        return dest

    try:
        import soundfile as sf

        inf = sf.info(audio_path)
        if inf.channels <= 1:
            return audio_path
        data, sr = sf.read(audio_path, always_2d=True, dtype="float32")
        if data.shape[1] > 1:
            mono = np.mean(data, axis=1).astype(np.float32, copy=False)
        else:
            mono = np.ascontiguousarray(data[:, 0])
        return _write_mono_pcm16(mono, int(sr), out_path)
    except Exception:
        pass

    try:
        import librosa

        y, sr = librosa.load(audio_path, sr=None, mono=True)
        mono = np.asarray(y, dtype=np.float32)
        return _write_mono_pcm16(mono, int(sr), out_path)
    except Exception:
        pass

    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] <= 1:
            return audio_path
        waveform = waveform.mean(dim=0, keepdim=True)
        torchaudio.save(out_path, waveform, int(sample_rate))
        return out_path
    except Exception as e:
        raise RuntimeError(
            f"convert_to_mono_if_needed: failed for {audio_path!r} "
            f"(soundfile, librosa, and torchaudio all failed: {e})"
        ) from e


def ensure_mono_audio_path(audio_path: str, duration_fallback: float) -> Tuple[str, float]:
    """
    Legacy fallback (SKIP_MONO_CONVERSION=0): point to mono audio for collate. Uses header/soundfile
    first; stereo downmix via convert_to_mono_if_needed (soundfile/librosa, not torchaudio by default).
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
        dur2 = _probe_duration_seconds(out, duration_fallback)
        return out, dur2 if dur2 > 0 else duration_fallback
    try:
        import soundfile as sf

        inf = sf.info(audio_path)
        nc = int(inf.channels)
        if nc <= 1:
            d = float(inf.duration)
            return audio_path, d if d > 0 else duration_fallback
        out = convert_to_mono_if_needed(audio_path)
        dur2 = _probe_duration_seconds(out, duration_fallback)
        return out, dur2 if dur2 > 0 else duration_fallback
    except Exception:
        pass
    try:
        info = torchaudio.info(audio_path)
        sr = float(info.sample_rate)
        nc = _info_num_channels(info)
        if nc <= 1:
            dur = float(info.num_frames) / sr if sr > 0 else duration_fallback
            return audio_path, dur if dur > 0 else duration_fallback
        out = convert_to_mono_if_needed(audio_path)
        dur2 = _probe_duration_seconds(out, duration_fallback)
        return out, dur2 if dur2 > 0 else duration_fallback
    except Exception:
        return audio_path, duration_fallback


def apply_mono_to_manifest_record(rec: Dict[str, Any]) -> None:
    """Legacy path (SKIP_MONO_CONVERSION=0): rewrite record to cached mono path when stereo."""
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
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("source") == "librispeech":
                continue
            ap = rec.get("audio_filepath") or rec.get("audio_file")
            if not ap:
                continue
            ap = os.path.normpath(os.path.expanduser(str(ap)))
            if not os.path.isfile(ap):
                continue
            hdr = _audio_channels_duration_header(ap)
            if hdr is None or hdr[0] > 1:
                return True
    return False


def rewrite_manifest_apply_mono(manifest_path: str) -> None:
    """Rewrite JSONL when needed; skip copy if header scan shows all non-Libri clips are mono."""
    if not manifest_path or not os.path.isfile(manifest_path):
        return
    if SKIP_MONO_CONVERSION:
        print(
            "Manifest mono pass: skipped (SKIP_MONO_CONVERSION=1); training uses NeMo channel_selector='average'.",
            flush=True,
        )
        return
    if not _manifest_mono_rewrite_needed(manifest_path):
        print(
            "Manifest mono pass: skipped full rewrite (header scan: all non-Libri clips mono).",
            flush=True,
        )
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

    val_sample_size = 0
    if SKIP_CHILD_VAL_MIX:
        print(
            "[manifest] SKIP_CHILD_VAL_MIX=1 — child split uses CHILD_TRAIN_MANIFEST only (no val rows).",
            flush=True,
        )
    else:
        # Child val: add 50% of validation manifest into training (legacy data-shortage recipe).
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
        print(f"Child val samples (50% of val manifest): {val_sample_size}")

    child_total = len(records)
    print(f"Total child samples: {child_total}")

    if not LIBRISPEECH_MANIFEST or not os.path.isfile(LIBRISPEECH_MANIFEST):
        raise FileNotFoundError(
            "LIBRISPEECH_MANIFEST must be set to an existing JSONL (pre-built Libri / merged synthetic). "
            "Tree walk and childrenized bundles were removed."
        )
    libri_records = []
    with open(LIBRISPEECH_MANIFEST, "r", encoding="utf-8") as _lf:
        for _line in _lf:
            _line = _line.strip()
            if _line:
                try:
                    libri_records.append(json.loads(_line))
                except json.JSONDecodeError:
                    pass
    print(f"LibriSpeech: loaded {len(libri_records)} from {LIBRISPEECH_MANIFEST}")
    for rec in libri_records:
        rec["source"] = "librispeech"
        records.append(rec)
    print(f"LibriSpeech samples: {len(libri_records)}")

    random.shuffle(records)
    print(f"Total combined samples (shuffled): {len(records)}")
    print(f"  - Real child: {child_total}")
    print(f"  - LibriSpeech: {len(libri_records)}")

    if SKIP_MONO_CONVERSION:
        print(
            "[manifest] SKIP_MONO_CONVERSION=1 — skipping real_child mono rewrite; NeMo averages channels at load.",
            flush=True,
        )
    else:
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

    records = [r for r in records if 0.1 < float(r["duration"]) <= MAX_DURATION_SEC]

    df = pd.DataFrame(records)
    df.to_json(COMBINED_MANIFEST, orient="records", lines=True)
    print(f"Saved combined manifest: {COMBINED_MANIFEST} (rows after duration filter: {len(df)})")
    return COMBINED_MANIFEST


# -----------------------------------------------------------------------------
# NeMo imports (after env ready)
# -----------------------------------------------------------------------------
from nemo.collections.asr.models import ASRModel, EncDecRNNTBPEModel
from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil
from nemo.core.classes import adapter_mixins


def infer_encoder_hidden_dim(model: Any) -> int:
    """Return encoder output last dimension via a short dummy forward (preprocessor → encoder)."""
    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        for key in (
            "encoder.d_model",
            "encoder.defaults.d_model",
            "model.encoder.d_model",
        ):
            d = OmegaConf.select(cfg, key)
            if d is not None:
                return int(d)
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        sig_len = 32000
        sig = torch.zeros(1, sig_len, device=device, dtype=dtype)
        sig_lens = torch.tensor([sig_len], device=device, dtype=torch.long)
        with torch.no_grad():
            processed, pl = model.preprocessor(input_signal=sig, length=sig_lens)
            try:
                enc_out, _ = model.encoder(audio_signal=processed, length=pl)
            except TypeError:
                enc_out, _ = model.encoder(input_signal=processed, length=pl)
        if enc_out.dim() < 2:
            raise RuntimeError(f"unexpected encoder output shape {tuple(enc_out.shape)}")
        dim = int(enc_out.shape[-1])
        if dim < 1:
            raise RuntimeError(f"invalid encoder hidden dim {dim}")
        return dim
    finally:
        model.train(was_training)


class AdapterChainState:
    """Manages bottleneck chain state across encoder layers."""

    def __init__(self) -> None:
        self.prev_bottleneck: Optional[torch.Tensor] = None
        self.layer_counter: int = 0

    def reset(self) -> None:
        self.prev_bottleneck = None
        self.layer_counter = 0

    def update(self, bottleneck: torch.Tensor) -> None:
        self.prev_bottleneck = bottleneck
        self.layer_counter += 1


class ChainedLinearAdapter(nn.Module, AdapterModuleUtil):
    def __init__(
        self,
        in_features: int = 512,
        dim: int = 256,
        activation: str = "relu",
        norm_position: str = "pre",
        dropout: float = 0.1,
        is_first: bool = False,
        chain_state_ref: Optional[AdapterChainState] = None,
        adapter_strategy: Any = None,
    ):
        nn.Module.__init__(self)
        self.norm = nn.LayerNorm(in_features)
        self.down = nn.Linear(in_features, dim)
        self.act = nn.ReLU()
        self.up = nn.Linear(dim, in_features)
        self.dropout_layer = nn.Dropout(dropout)
        self.is_first = is_first
        self.chain_state_ref = chain_state_ref
        if not is_first:
            self.chain_proj = nn.Linear(dim, dim, bias=False)
            nn.init.zeros_(self.chain_proj.weight)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.setup_adapter_strategy(adapter_strategy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.down(h)
        if not self.is_first and self.chain_state_ref is not None:
            prev = self.chain_state_ref.prev_bottleneck
            if prev is not None:
                if prev.shape[1] != h.shape[1]:
                    prev = F.interpolate(
                        prev.transpose(1, 2),
                        size=h.shape[1],
                        mode="nearest",
                    ).transpose(1, 2)
                h = h + self.chain_proj(prev)
        bottleneck = self.act(h)
        if self.chain_state_ref is not None:
            self.chain_state_ref.update(bottleneck)
        h = self.up(bottleneck)
        h = self.dropout_layer(h)
        return h


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
        pitch_prob_child: float = 0.0,
        pitch_prob_other: float = 0.0,
        classroom_noise_prob: float = 0.0,
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


# -----------------------------------------------------------------------------
# Model subclass
# -----------------------------------------------------------------------------
def _get_encoder_layers(encoder: nn.Module) -> List[nn.Module]:
    if hasattr(encoder, "layers") and isinstance(encoder.layers, nn.ModuleList):
        return list(encoder.layers)
    if hasattr(encoder, "encoder"):
        inner = encoder.encoder
        if hasattr(inner, "layers") and isinstance(inner.layers, nn.ModuleList):
            return list(inner.layers)
    raise RuntimeError("Could not locate Conformer encoder layers on model.encoder")


class ParakeetAdapterModel(EncDecRNNTBPEModel):
    def init_adapter_stack(self) -> None:
        self.adapter_chain_state = AdapterChainState()
        self.pre_encoder_adapter = None
        self._encoder_hidden_dim = infer_encoder_hidden_dim(self)
        self.layer_adapters = nn.ModuleList()
        self._adapter_hooks = []
        self.waveform_augmentor: Optional[WaveformAugmentor] = None
        print(
            f"[adapter] Pre-encoder adapter disabled. "
            f"Inferred encoder hidden dim={self._encoder_hidden_dim} (BOTTLENECK_DIM={BOTTLENECK_DIM}); "
            "layer adapters use this in_features.",
            flush=True,
        )

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

    def register_encoder_adapters(self, reuse_existing: bool = False) -> None:
        for h in getattr(self, "_adapter_hooks", []) or []:
            try:
                h.remove()
            except Exception:
                pass
        self._adapter_hooks = []
        layers = _get_encoder_layers(self.encoder)
        n_layers = len(layers)

        if reuse_existing:
            la = getattr(self, "layer_adapters", None)
            if isinstance(la, nn.ModuleList) and len(la) == n_layers:
                if all(isinstance(m, ChainedLinearAdapter) for m in la):
                    self.layer_adapters = la.to(self.device)
                    for idx, layer in enumerate(layers):
                        adapter = self.layer_adapters[idx]

                        def _hook(mod, inp, out, ad=adapter):
                            if isinstance(out, tuple):
                                x, lens = out
                                return (x + ad(x), lens)
                            return out + ad(out)

                        self._adapter_hooks.append(layer.register_forward_hook(_hook))
                    print(
                        f"[adapter] Re-attached hooks to {n_layers} existing ChainedLinearAdapter modules",
                        flush=True,
                    )
                    return
            print(
                "[adapter] reuse_existing=True but layer_adapters missing or wrong length/type; rebuilding.",
                flush=True,
            )

        self.layer_adapters = nn.ModuleList()
        for _ in layers:
            self.layer_adapters.append(
                ChainedLinearAdapter(
                    in_features=self._encoder_hidden_dim,
                    dim=BOTTLENECK_DIM,
                    is_first=False,
                    chain_state_ref=self.adapter_chain_state,
                )
            )
        self.layer_adapters = self.layer_adapters.to(self.device)

        for idx, layer in enumerate(layers):
            adapter = self.layer_adapters[idx]

            def _hook(mod, inp, out, ad=adapter):
                if isinstance(out, tuple):
                    x, lens = out
                    return (x + ad(x), lens)
                return out + ad(out)

            self._adapter_hooks.append(layer.register_forward_hook(_hook))

    def forward(self, *args, **kwargs):
        if hasattr(self, "adapter_chain_state"):
            self.adapter_chain_state.reset()

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
            if self.pre_encoder_adapter is not None:
                processed_signal = processed_signal + self.pre_encoder_adapter(processed_signal)
            kwargs["processed_signal"] = processed_signal
            kwargs["processed_signal_length"] = processed_signal_length
            kwargs.pop("input_signal", None)
            kwargs.pop("input_signal_length", None)
            rest_args = args[2:] if len(args) > 2 else ()
            return super().forward(*rest_args, **kwargs)

        return super().forward(*args, **kwargs)

    def save_adapters(self, path: str) -> None:
        payload: Dict[str, Any] = {"layer_adapters": self.layer_adapters.state_dict()}
        if self.pre_encoder_adapter is not None:
            payload["pre_encoder"] = self.pre_encoder_adapter.state_dict()
        torch.save(payload, path)

    def load_adapters(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        if self.pre_encoder_adapter is not None and "pre_encoder" in data:
            self.pre_encoder_adapter.load_state_dict(data["pre_encoder"])
        self.layer_adapters.load_state_dict(data["layer_adapters"])


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
    if model.pre_encoder_adapter is not None:
        for p in model.pre_encoder_adapter.parameters():
            p.requires_grad = True
    for p in model.layer_adapters.parameters():
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
    if model.pre_encoder_adapter is not None:
        for p in model.pre_encoder_adapter.parameters():
            p.requires_grad = True
    for p in model.layer_adapters.parameters():
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
    if model.pre_encoder_adapter is not None:
        adapter_params.extend(list(model.pre_encoder_adapter.parameters()))
    adapter_params.extend(list(model.layer_adapters.parameters()))
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

        # LambdaLR applies an initial step; linear warmup uses lr_lambda(0)==0 so param lrs start at 0 until steps advance.
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        # NeMo RNNT training_step reads self._optimizer.param_groups[0]['lr']; ModelPT sets this in setup_optimization.
        model._optimizer = opt
        model._scheduler = sched
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }

    model.configure_optimizers = configure_optimizers


def build_param_groups_stage2(model: ParakeetAdapterModel) -> List[Dict[str, Any]]:
    enc = list(model.encoder.parameters())
    adapter: List[torch.nn.Parameter] = list(model.layer_adapters.parameters())
    if model.pre_encoder_adapter is not None:
        adapter = list(model.pre_encoder_adapter.parameters()) + adapter
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
        model._optimizer = opt
        model._scheduler = sched
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
    """Load ASR from INIT_NEMO_PATH (.nemo) or HuggingFace MODEL_ID; disable CUDA graph decoder."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    map_loc = torch.device(dev)

    if INIT_NEMO_PATH:
        if not os.path.isfile(INIT_NEMO_PATH):
            raise FileNotFoundError(f"INIT_NEMO_PATH is set but file not found: {INIT_NEMO_PATH}")
        print(
            f"[model] Loading from local .nemo (strict={NEMO_RESTORE_STRICT}): {INIT_NEMO_PATH}",
            flush=True,
        )
        try:
            model = ASRModel.restore_from(
                INIT_NEMO_PATH,
                map_location=map_loc,
                strict=NEMO_RESTORE_STRICT,
            )
        except Exception as e:
            if NEMO_RESTORE_STRICT:
                print(
                    f"[model] strict restore failed ({e}); retrying with strict=False "
                    "(adapter keys may be dropped if architecture mismatch).",
                    flush=True,
                )
                model = ASRModel.restore_from(
                    INIT_NEMO_PATH,
                    map_location=map_loc,
                    strict=False,
                )
            else:
                raise
        _apply_cuda_graph_disable(model)
        print(f"[model] Done: restore_from local .nemo on {map_loc}.", flush=True)
        return model

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


def _ensure_train_dataloader_channel_averaging(model: Any) -> None:
    """
    Match legacy working scripts: non-Lhotse BPE dataset with channel_selector='average'.
    If merged cfg dropped these fields, patch model.cfg.train_ds and rebuild _train_dl once.
    """
    if not hasattr(model.cfg, "train_ds"):
        print("[dataloader][warn] model.cfg.train_ds missing; cannot set use_lhotse/channel_selector.", flush=True)
        return
    td = model.cfg.train_ds
    ul = OmegaConf.select(model.cfg, "train_ds.use_lhotse")
    cs = OmegaConf.select(model.cfg, "train_ds.channel_selector")
    patched = False
    if ul is not False or cs != "average":
        with open_dict(td):
            td.use_lhotse = False
            td.channel_selector = "average"
        patched = True
        model._train_dl = model._setup_dataloader_from_config(config=model.cfg.train_ds)
    eff_ul = OmegaConf.select(model.cfg, "train_ds.use_lhotse")
    eff_cs = OmegaConf.select(model.cfg, "train_ds.channel_selector")
    print(
        f"[dataloader] Effective train_ds: use_lhotse={eff_ul!r}, channel_selector={eff_cs!r}"
        + (" — rebuilt training DataLoader after cfg patch" if patched else ""),
        flush=True,
    )
    print(
        "[dataloader] Primary multichannel handling: NeMo load-time averaging (use_lhotse=False, channel_selector='average'); "
        "no stereo→mono manifest rewrite required for normal runs.",
        flush=True,
    )
    print(
        "[dataloader] Manual mono conversion (torchaudio / manifest rewrite) is optional legacy fallback — set SKIP_MONO_CONVERSION=0.",
        flush=True,
    )


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if COMBINED_TRAIN_MANIFEST and os.path.isfile(COMBINED_TRAIN_MANIFEST):
        manifest_path = COMBINED_TRAIN_MANIFEST
        print(f"Using COMBINED_TRAIN_MANIFEST (no merge): {manifest_path}", flush=True)
        rewrite_manifest_apply_mono(manifest_path)
    elif USE_EXISTING_MANIFESTS and os.path.isfile(COMBINED_MANIFEST):
        manifest_path = COMBINED_MANIFEST
        print(f"Using existing working manifest: {manifest_path}", flush=True)
        rewrite_manifest_apply_mono(manifest_path)
    else:
        manifest_path = build_combined_manifest()

    model = _load_base_asr_model()
    model.__class__ = ParakeetAdapterModel

    layers_n = len(_get_encoder_layers(model.encoder))
    la = getattr(model, "layer_adapters", None)
    reuse_adapters = (
        isinstance(la, nn.ModuleList)
        and len(la) == layers_n
        and all(isinstance(m, ChainedLinearAdapter) for m in la)
    )

    if reuse_adapters:
        if not hasattr(model, "adapter_chain_state") or model.adapter_chain_state is None:
            model.adapter_chain_state = AdapterChainState()
        if not hasattr(model, "_encoder_hidden_dim"):
            model._encoder_hidden_dim = infer_encoder_hidden_dim(model)
        if not hasattr(model, "pre_encoder_adapter"):
            model.pre_encoder_adapter = None
        model.register_encoder_adapters(reuse_existing=True)
        print(
            "[adapter] Resuming from checkpoint: kept existing layer_adapters; encoder hooks re-attached.",
            flush=True,
        )
    else:
        model.init_adapter_stack()
        model.register_encoder_adapters()
        if ADAPTER_PT_PATH:
            if os.path.isfile(ADAPTER_PT_PATH):
                model.load_adapters(ADAPTER_PT_PATH)
                print(f"[adapter] Loaded adapter weights from {ADAPTER_PT_PATH}", flush=True)
            else:
                print(f"[adapter] ADAPTER_PT_PATH set but file not found: {ADAPTER_PT_PATH}", flush=True)

    model.preprocessor.eval()
    _set_requires_grad(model.preprocessor, False)
    model.encoder.train()

    from hydra.utils import instantiate
    model.spec_augmentation = instantiate(OmegaConf.create(spec_augment_cfg))
    model.waveform_augmentor = WaveformAugmentor(
        noise_file_paths=[],
        pitch_prob_child=WAVEFORM_PITCH_PROB_CHILD,
        pitch_prob_other=WAVEFORM_PITCH_PROB_OTHER,
        classroom_noise_prob=0.0,
    )

    apply_freeze_stage1_adapters_only(model)
    attach_configure_optimizers_stage1(model)

    print(
        "[dataloader] Building training DataLoader with use_lhotse=False, channel_selector='average' "
        "(same strategy as legacy NeMo scripts; multichannel Libri / child WAVs averaged at load).",
        flush=True,
    )
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
        "return_sample_id": False,
        "use_lhotse": False,
        "channel_selector": "average",
    })
    model.setup_training_data(train_cfg)
    _ensure_train_dataloader_channel_averaging(model)
    # attach_child_sample_flags(model, manifest_path)

    tb1 = TensorBoardLogger(save_dir=TB_DIR, name="stage1_adapters")
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
        log_every_n_steps=LOG_EVERY_N_STEPS,
    )

    print(
        f"Stage 1: {NUM_EPOCHS} epochs — adapters + joint + partial decoder "
        f"(encoder frozen), clip={GRADIENT_CLIP_VAL}; log_every_n_steps={LOG_EVERY_N_STEPS}",
        flush=True,
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
        log_every_n_steps=LOG_EVERY_N_STEPS,
    )
    model.train()
    trainer_stage2.fit(model)
    print("Training finished (Stage 1 + Stage 2).")


if __name__ == "__main__":
    main()
'''
USE_11B_STAGE2 = os.environ.get("TDT_11B_STAGE2_ONLY", "0").strip().lower() in ("1", "true", "yes", "on")
TRAIN_SCRIPT = (
    "/kaggle/working/train_nemo_adapter_1_1b_stage2.py"
    if USE_11B_STAGE2
    else "/kaggle/working/train_nemo_adapter_0.6b.py"
)
_payload = TRAIN_CODE_11B_STAGE2 if USE_11B_STAGE2 else train_code
os.makedirs("/kaggle/working", exist_ok=True)
with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
    f.write(_payload)
print(
    f"[notebook] Wrote {'1.1B stage-2' if USE_11B_STAGE2 else '0.6B'} training script -> {TRAIN_SCRIPT}",
    flush=True,
)

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
