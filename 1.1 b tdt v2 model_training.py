# -*- coding: utf-8 -*-
"""
Paste this entire file into ONE Kaggle notebook cell (or run locally).

  Part A — pip install (clean env + NeMo)
  Part B — write training script under WORK_DIR
  Part C — subprocess.run (fresh Python interpreter)

Pipeline (0.2307): Parakeet TDT 1.1B + linear bottleneck adapter, waveform aug,
SpecAugment, 5 epochs, cosine LR, saves only model_epoch3.nemo + model_final.nemo.

Optional env: WORK_DIR (default /kaggle/working), BATCH_SIZE, LEARNING_RATE,
TRAIN_MANIFEST, USE_EXISTING_MANIFESTS, CLASSROOM_NOISE_DIRS, MODEL_OUTPUT_DIR
"""
#  this is with balanced augmentationss


import os
import subprocess
import sys


def _pip(*args):
    cmd = [sys.executable, "-m", "pip", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("\n[pip] command failed:")
        print(" ".join(cmd))
        if proc.stdout:
            print("\n[pip stdout]\n" + proc.stdout[-4000:])
        if proc.stderr:
            print("\n[pip stderr]\n" + proc.stderr[-4000:])
        raise RuntimeError(f"pip failed with exit code {proc.returncode}")


def _install_numpy_scipy_with_fallback():
    # Preferred pins (kept aligned with notebook format).
    try:
        _pip(
            "install",
            "--no-cache-dir",
            "--prefer-binary",
            "numpy>=2.1,<2.3",
            "scipy>=1.14,<1.16",
        )
        return
    except Exception as e:
        print(f"[warn] pinned numpy/scipy install failed: {e}")

    # Fallback for transient index issues / wheel availability on current Python build.
    _pip(
        "install",
        "--no-cache-dir",
        "--prefer-binary",
        "numpy>=2.1",
        "scipy",
    )


# =============================================================================
# PART A: INSTALLATION
# =============================================================================
print("Step 1: Cleaning...")
for _ in range(2):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "numpy",
            "scipy",
            "nemo_toolkit",
            "lightning",
            "pytorch-lightning",
            "datasets",
            "diffusers",
            "gradio",
            "peft",
            "sentence-transformers",
            "transformers",
            "huggingface_hub",
            "torch",
            "torchaudio",
            "torchvision",
            "numba",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

print("Step 2: numpy + scipy...")
_pip("install", "--no-cache-dir", "--upgrade", "pip", "setuptools", "wheel")
_install_numpy_scipy_with_fallback()

print("Step 3: PyTorch (CUDA 12.6 wheels)...")
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--index-url",
        "https://download.pytorch.org/whl/cu126",
        "torch>=2.9.0",
        "torchaudio",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
)

print("Step 4: Dependencies...")
_pip(
    "install",
    "--no-cache-dir",
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

print("Step 5: NeMo...")
_pip("install", "--no-cache-dir", "nemo_toolkit[asr]>=2.5.0")

print("Step 6: Re-pin numpy...")
_pip(
    "install",
    "--no-cache-dir",
    "--force-reinstall",
    "numpy>=2.1,<2.3",
    "scipy>=1.14,<1.16",
)

print("Installation complete.\n")

# =============================================================================
# PART B: WRITE SELF-CONTAINED TRAINING SCRIPT
# =============================================================================
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")
os.makedirs(WORK_DIR, exist_ok=True)
TRAIN_SCRIPT = os.path.join(WORK_DIR, "train_parakeet_2307_adapter.py")

train_code = r'''
from __future__ import annotations

import json
import logging
import os
import random
import types
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List

warnings.filterwarnings("ignore", category=Warning, module="numba")
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

import torch.nn.functional as F
import torchaudio
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from nemo.collections.asr.models import ASRModel
from nemo.core.classes import adapter_mixins
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf, open_dict

# ==========================================
# Waveform Augmentation (speed + pitch + classroom noise SNR + gain)
# ==========================================
class WaveformAugmentor(nn.Module):
    """On-the-fly waveform augmentation before the mel preprocessor."""

    def __init__(
        self,
        speed_min: float = 0.90,
        speed_max: float = 1.10,
        pitch_min: float = -1.0,
        pitch_max: float = 2.0,
        sample_rate: int = 16000,
        speed_prob: float = 0.5,
        pitch_prob: float = 0.5,
        classroom_noise_prob: float = 0.5,
        classroom_snr_min: float = 5.0,
        classroom_snr_max: float = 10.0,
        noise_file_paths: list | None = None,
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
            speed_factor = (
                self.speed_min
                + torch.rand(1).item() * (self.speed_max - self.speed_min)
            )
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
            n_steps = (
                self.pitch_min
                + torch.rand(1).item() * (self.pitch_max - self.pitch_min)
            )
            try:
                audio_signal = torchaudio.functional.pitch_shift(
                    audio_signal, self.sample_rate, n_steps
                )
            except Exception:
                pass

        if (
            self.noise_file_paths
            and torch.rand(1).item() < self.classroom_noise_prob
        ):
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


def _collect_noise_paths_from_dirs(dir_list: list[str]) -> list[str]:
    exts = (".wav", ".flac", ".mp3", ".ogg")
    out: list[str] = []
    for d in dir_list:
        d = d.strip()
        if not d or not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(exts):
                    out.append(os.path.join(root, f))
    return out


# ==========================================
# Save only specific epochs — .nemo only (no adapter .pt)
# ==========================================
class SaveSelectedEpochs(Callback):
    """Save full .nemo only for listed 0-indexed epochs (one file per save)."""

    def __init__(self, directory: str, save_epochs: list[int]):
        super().__init__()
        self.directory = directory
        self.save_epochs = save_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        epoch = trainer.current_epoch
        if epoch not in self.save_epochs:
            return
        os.makedirs(self.directory, exist_ok=True)
        # 1-based epoch in filename (e.g. epoch index 2 -> model_epoch3.nemo)
        human_ep = epoch + 1
        nemo_path = os.path.join(self.directory, f"model_epoch{human_ep}.nemo")
        pl_module.save_to(nemo_path)
        print(f"[SaveSelectedEpochs] saved {nemo_path}", flush=True)


# ==========================================
# Configuration
# ==========================================
MODEL_ID = "nvidia/parakeet-tdt-1.1b"

ASR_DATA_DIR = "/kaggle/input/datasets/akarshkumarshukla/asr-data"
ASR_JSONL = os.path.join(ASR_DATA_DIR, "train_word_transcripts.jsonl")
TALKBANK_DIR = "/kaggle/input/datasets/akarshkumarshukla/talk-bank-data/audio"
TALKBANK_JSON = (
    "/kaggle/input/datasets/akarshkumarshukla/talkbank-transcript-full/"
    "train_word_transcripts.jsonl"
)

SAVE_DIR = "/kaggle/working/nemo_adapter_1.1b"
MANIFEST_DIR = os.path.join(SAVE_DIR, "manifests")
# Only these two .nemo files are written here (epoch 3 mid-train + final); nothing else.
MODEL_OUTPUT_DIR = os.environ.get(
    "MODEL_OUTPUT_DIR",
    os.path.join(SAVE_DIR, "saved_models_only"),
)

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = 2
NUM_EPOCHS = 5
MAX_DURATION_SEC = 200.0
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-4"))

# 0-indexed: end of epoch 3 -> current_epoch == 2; final saved after fit (epoch 5)
SAVE_EPOCHS = [2]

os.makedirs(MANIFEST_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

print(f"torch:  {torch.__version__}")
print(f"CUDA:   {torch.version.cuda}")
print(f"numpy:  {np.__version__}")

# ==========================================
# Manifest Handling
# ==========================================
USE_EXISTING_MANIFESTS = True
TRAIN_MANIFEST = os.environ.get(
    "TRAIN_MANIFEST",
    "/kaggle/input/datasets/akarshks/train-meta/train_manifest.jsonl",
)

if USE_EXISTING_MANIFESTS:
    if not os.path.isfile(TRAIN_MANIFEST):
        raise FileNotFoundError(
            f"Train manifest not found: {TRAIN_MANIFEST}. "
            "Attach the dataset or set TRAIN_MANIFEST / USE_EXISTING_MANIFESTS=0."
        )
    train_path = TRAIN_MANIFEST
    print("Using existing manifest:")
    print("  Train:", train_path)
else:

    def _index_directory(search_dir):
        local = {}
        if not os.path.isdir(search_dir):
            return local
        for root, _, files in os.walk(search_dir):
            for f in files:
                if f.endswith(".flac"):
                    local[f] = os.path.join(root, f)
        return local

    print("Indexing audio files...")
    audio_index = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = [
            ex.submit(_index_directory, ASR_DATA_DIR),
            ex.submit(_index_directory, TALKBANK_DIR),
        ]
        for fut in futures:
            audio_index.update(fut.result())
    print(f"Indexed {len(audio_index):,} files")

    def _find_audio(path):
        return audio_index.get(os.path.basename(path), None)

    print("Building manifests from JSONL...")
    records: List[dict] = []
    jsonl_files = [ASR_JSONL]
    if os.path.exists(TALKBANK_JSON):
        jsonl_files.append(TALKBANK_JSON)

    for filepath in jsonl_files:
        if not os.path.exists(filepath):
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("orthographic_text", "").strip().lower()
                    dur = float(data.get("audio_duration_sec", 0.0))
                    path = _find_audio(data.get("audio_path", ""))
                    if path and text and (0.1 < dur <= MAX_DURATION_SEC):
                        records.append(
                            {"audio_filepath": path, "duration": dur, "text": text}
                        )
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue

    df = pd.DataFrame(records)
    print(f"Total: {len(df):,} utterances")
    if len(df) == 0:
        raise RuntimeError(
            "No training rows after JSONL + audio index. Check ASR_DATA_DIR / TALKBANK paths."
        )

    train_path = os.path.join(MANIFEST_DIR, "train_manifest.jsonl")
    df.to_json(train_path, orient="records", lines=True)
    print(f"Train: {len(df):,} utterances (all data, no val split)")


# ==========================================
# Helper
# ==========================================
def update_model_cfg(orig_cfg, new_cfg):
    with open_dict(orig_cfg):
        for k, v in new_cfg.items():
            orig_cfg[k] = v
    return orig_cfg


def _encoder_target_key(model_cfg):
    enc = model_cfg.encoder
    if "_target_" in enc:
        return "_target_"
    if "target" in enc:
        return "target"
    raise AttributeError("model_cfg.encoder has neither _target_ nor target")


# ==========================================
# OmegaConf Config
# ==========================================
cfg = OmegaConf.create(
    {
        "model": {
            "pretrained_model": MODEL_ID,
            "log_prediction": False,
            "spec_augment": {
                "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
                "freq_masks": 2,
                "freq_width": 27,
                "time_masks": 10,
                "time_width": 0.05,
            },
            "adapter": {
                "adapter_name": "asr_children_adapter",
                "adapter_module_name": "encoder",
                "adapter_type": "linear",
                "linear": {
                    "_target_": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
                    "in_features": 1024,
                    "dim": 128,
                    "activation": "gelu",
                    "norm_position": "post",
                    "dropout": 0.1,
                },
            },
            "train_ds": {
                "manifest_filepath": train_path,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "pin_memory": False,
                "use_lhotse": False,
                "channel_selector": "average",
                "is_tarred": False,
                "shuffle": True,
            },
            "optim": {
                "name": "adamw",
                "lr": LEARNING_RATE,
                "betas": [0.9, 0.999],
                "weight_decay": 0.01,
                "sched": {
                    "name": "CosineAnnealing",
                    "warmup_ratio": 0.15,
                    "min_lr": 1e-6,
                },
            },
        },
        "trainer": {
            "devices": 1,
            "accelerator": "gpu",
            "precision": "bf16-mixed",
            "max_epochs": NUM_EPOCHS,
            "enable_progress_bar": True,
            "log_every_n_steps": 2000,
            "gradient_clip_val": 1.0,
        },
        "exp_manager": {
            "exp_dir": SAVE_DIR,
            "name": "ParakeetAdapter1.1B",
            "create_tensorboard_logger": False,
            "create_checkpoint_callback": False,
            "resume_if_exists": False,
            "resume_ignore_no_checkpoint": True,
        },
    }
)

# ==========================================
# Trainer
# ==========================================
print("Initializing Trainer...")
trainer = pl.Trainer(
    devices=1,
    accelerator="gpu",
    precision="bf16-mixed",
    max_epochs=NUM_EPOCHS,
    limit_val_batches=0,
    num_sanity_val_steps=0,
    enable_progress_bar=True,
    log_every_n_steps=2000,
    gradient_clip_val=1.0,
    logger=False,
    enable_checkpointing=False,
)
exp_log_dir = exp_manager(trainer, cfg.exp_manager)
print(f"Experiment dir (minimal logs): {exp_log_dir}")
print(f"Model outputs (.nemo only): {MODEL_OUTPUT_DIR}")
print("Validation: DISABLED (training only)")

trainer.callbacks.append(SaveSelectedEpochs(MODEL_OUTPUT_DIR, save_epochs=SAVE_EPOCHS))
print(
    f"Will save model_epoch{SAVE_EPOCHS[0]+1}.nemo mid-run + model_final.nemo -> {MODEL_OUTPUT_DIR}"
)

# ==========================================
# Load Pretrained Model
# ==========================================
print(f"Loading {MODEL_ID}...")
model_cfg = ASRModel.from_pretrained(MODEL_ID, return_config=True)
enc_key = _encoder_target_key(model_cfg)

with open_dict(model_cfg):
    adapter_metadata = adapter_mixins.get_registered_adapter(
        model_cfg.encoder[enc_key]
    )
    if adapter_metadata is not None:
        model_cfg.encoder[enc_key] = adapter_metadata.adapter_class_path
        print(f"Patched encoder to: {adapter_metadata.adapter_class_path}")

model = ASRModel.from_pretrained(
    MODEL_ID, override_config_path=model_cfg, trainer=trainer
)

with open_dict(model.cfg):
    if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
        model.cfg.decoding.greedy.use_cuda_graph_decoder = False
if hasattr(model, "change_decoding_strategy"):
    model.change_decoding_strategy(model.cfg.decoding)

print("Model loaded")

# ==========================================
# Waveform Augmentation
# ==========================================
_noise_dirs: list[str] = []
_env_noise = True
if _env_noise:


      _noise_dirs.extend(
        [
            "/kaggle/input/datasets/akarshkumarshukla/noise-1",
            "/kaggle/input/datasets/akarshkumarshukla/asr-data/noise_part_0",
        ]
    )
else:
    _noise_dirs.extend(
        [
            "/kaggle/input/datasets/akarshkumarshukla/noise-1",
            "/kaggle/input/datasets/akarshkumarshukla/asr-data/noise_part_0",
        ]
    )
_noise_paths = _collect_noise_paths_from_dirs(_noise_dirs)
print(
    f"Attaching waveform aug: speed 0.90–1.10, pitch -1..+2 semitones, "
    f"classroom noise SNR 5–10 dB (prob=0.5), {len(_noise_paths)} noise clips"
)
waveform_aug = WaveformAugmentor(
    speed_min=0.90,
    speed_max=1.10,
    pitch_min=-1.0,
    pitch_max=2.0,
    sample_rate=16000,
    speed_prob=0.5,
    pitch_prob=0.5,
    classroom_noise_prob=0.5,
    classroom_snr_min=5.0,
    classroom_snr_max=10.0,
    noise_file_paths=_noise_paths if _noise_paths else None,
    gain_prob=0.3,
    gain_db_min=-6.0,
    gain_db_max=6.0,
)
model.add_module("waveform_augmentor", waveform_aug)

_original_forward = model.forward.__func__


def _augmented_forward(
    self,
    input_signal=None,
    input_signal_length=None,
    processed_signal=None,
    processed_signal_length=None,
):
    if self.training and input_signal is not None and input_signal_length is not None:
        input_signal, input_signal_length = self.waveform_augmentor(
            input_signal, input_signal_length
        )
    return _original_forward(
        self,
        input_signal=input_signal,
        input_signal_length=input_signal_length,
        processed_signal=processed_signal,
        processed_signal_length=processed_signal_length,
    )


model.forward = types.MethodType(_augmented_forward, model)
print("Waveform augmentation attached to model forward (no noise clips if list empty)")

if "spec_augment" in cfg.model:
    model.spec_augmentation = model.from_config_dict(cfg.model.spec_augment)
    print("SpecAugment enabled (freq_masks=2, time_masks=10)")

# ==========================================
# Data
# ==========================================
print("Setting up training data...")
cfg.model.train_ds = update_model_cfg(model.cfg.train_ds, cfg.model.train_ds)
model.setup_training_data(cfg.model.train_ds)

# ==========================================
# Adapter
# ==========================================
print("Adding bottleneck adapter (1024 -> 128 -> 1024, activation=gelu)...")
adapter_name = "encoder:asr_children_adapter"
adapter_type_cfg = cfg.model.adapter.linear
model.add_adapter(name=adapter_name, cfg=adapter_type_cfg)
model.set_enabled_adapters(enabled=False)
model.set_enabled_adapters(name=adapter_name, enabled=True)
model.freeze()
model.unfreeze_enabled_adapters()

model.setup_optimization(cfg.model.optim)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params:     {total:,}")
print(f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)")

# ==========================================
# Train
# ==========================================
print("=" * 60)
print("STARTING TRAINING (no validation)")
print(f"  Model:      {MODEL_ID}")
print(f"  Adapter:    1024 -> 128 -> 1024 (GELU + LayerNorm + dropout=0.1)")
print(f"  Epochs:     {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  LR:         {LEARNING_RATE} (AdamW + cosine, warmup_ratio=0.15, min_lr=1e-6)")
print(f"  Saves:      Only {MODEL_OUTPUT_DIR}/model_epoch3.nemo + model_final.nemo")
print(f"  Augmentation:")
print(f"    - Speed: 0.90–1.10x (prob=0.5)")
print(f"    - Pitch: -1 to +2 semitones (prob=0.5)")
print(f"    - Noise: SNR 5–10 dB (prob=0.5) if noise clips found")
print(f"    - Gain: +/-6 dB (prob=0.3)")
print(f"    - SpecAugment: freq_masks=2 w=27, time_masks=10 w=0.05")
print("=" * 60)

trainer.fit(model)

# ==========================================
# Save Final (end of epoch 5)
# ==========================================
final_path = os.path.join(MODEL_OUTPUT_DIR, "model_final.nemo")
print(f"Saving final model -> {final_path}")
model.save_to(final_path)
print("TRAINING COMPLETE!")
print(f"\nOnly model files in {MODEL_OUTPUT_DIR}:")
print("  - model_epoch3.nemo")
print("  - model_final.nemo")

'''

with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
    f.write(train_code)

print(f"Training script written to {TRAIN_SCRIPT}\n")
print(
    "Optional env (set BEFORE this cell; child inherits):\n"
    "  USE_EXISTING_MANIFESTS — 1 use pre-built manifest (default 1)\n"
    "  TRAIN_MANIFEST — override train manifest path\n"
    "  BATCH_SIZE — default 64\n"
    "  LEARNING_RATE — default 5e-4\n"
    "  CLASSROOM_NOISE_DIRS — comma-separated noise clip folders\n"
    "  MODEL_OUTPUT_DIR — where .nemo files are saved\n"
    "  WORK_DIR — where train script is written (default /kaggle/working)\n"
)

# =============================================================================
# PART C: FRESH SUBPROCESS
# =============================================================================
print("=" * 60)
print("Launching training in fresh subprocess")
print("=" * 60)

_compile = subprocess.run(
    [sys.executable, "-m", "py_compile", TRAIN_SCRIPT],
    cwd=WORK_DIR,
    capture_output=True,
    text=True,
)
if _compile.returncode != 0:
    print("Syntax error in generated training script (py_compile failed):\n")
    print(_compile.stderr or _compile.stdout)
    raise RuntimeError(
        "train script failed py_compile — fix the generator or paste errors above."
    )

result = subprocess.run(
    [sys.executable, TRAIN_SCRIPT],
    cwd=WORK_DIR,
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
    stdout=None,
    stderr=subprocess.STDOUT,
)

if result.returncode != 0:
    print(
        f"\nTraining subprocess exited with code {result.returncode}.\n"
        "Scroll up for the Python traceback from the child process.\n"
    )
    raise RuntimeError(f"Training failed with exit code {result.returncode}")

print("\nTraining completed successfully.")
print("Check MODEL_OUTPUT_DIR (default under nemo_adapter_1.1b/saved_models_only) for .nemo files.")

# =============================================================================
# OPTIONAL Part D — adapter pipeline verify
# =============================================================================
VERIFY_SCRIPT = os.environ.get(
    "VERIFY_SCRIPT",
    "/kaggle/working/verify_nemo_adapter_pipeline.py",
)
if os.environ.get("RUN_ADAPTER_VERIFY", "").strip().lower() in ("1", "true", "yes"):
    print("\n" + "=" * 60)
    print("Part D: Launching verify_nemo_adapter_pipeline.py in fresh subprocess")
    print("=" * 60)
    if not os.path.isfile(VERIFY_SCRIPT):
        print(
            f"SKIP: VERIFY_SCRIPT not found: {VERIFY_SCRIPT}\n"
            "Upload verify_nemo_adapter_pipeline.py to /kaggle/working/ or set VERIFY_SCRIPT."
        )
    else:
        _vout = os.environ.get("VERIFY_OUT", "/kaggle/working/adapter_verify_report")
        _vmanifest = os.environ.get("TRAIN_MANIFEST", "").strip()
        _vquick = os.environ.get("VERIFY_QUICK", "").strip().lower() in ("1", "true", "yes")
        _vcmd = [
            sys.executable,
            VERIFY_SCRIPT,
            "--out",
            _vout,
        ]
        if _vmanifest:
            _vcmd.extend(["--manifest", _vmanifest])
        if _vquick:
            _vcmd.append("--quick")
        _vc = subprocess.run(
            _vcmd,
            cwd=WORK_DIR,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=None,
            stderr=subprocess.STDOUT,
        )
        if _vc.returncode != 0:
            raise RuntimeError(
                f"Verification subprocess exited with code {_vc.returncode}. "
                f"See log above; report under {_vout}/VERIFY_REPORT.md if partial."
            )
        print(f"Verification OK. Report: {_vout}/VERIFY_REPORT.md")
