# spec augmentations-"spec_augment": {
                "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
                "freq_masks": 2,
                "freq_width": 27,
                "time_masks": 10,
                "time_width": 0.05,
            },
# speed_and_pitch_aug-waveform_aug = WaveformAugmentor(
    speed_min=0.95,
    speed_max=1.05,
    pitch_min=-1.0,
    pitch_max=1.0,
    sample_rate=16000,
    speed_prob=0.5,
    pitch_prob=0.5,
)
# thts it for this model-epoch-2 14.58 per
#  for epoch 3

from __future__ import annotations

import json
import logging
import os
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
import torch.nn.functional as F
import torchaudio
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from nemo.collections.asr.models import ASRModel
from nemo.core.classes import adapter_mixins
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf, open_dict

# ==========================================
# Waveform Augmentation (speed + pitch)
# ==========================================
class WaveformAugmentor(nn.Module):
    """On-the-fly waveform augmentation applied before the mel preprocessor.

    Speed perturbation uses linear interpolation to stretch/compress the signal.
    Pitch shifting uses torchaudio's phase-vocoder-based pitch_shift.
    Both are applied per-batch with independent random draws.
    """

    def __init__(
        self,
        speed_min: float = 0.95,
        speed_max: float = 1.05,
        pitch_min: float = -1.0,
        pitch_max: float = 1.0,
        sample_rate: int = 16000,
        speed_prob: float = 0.5,
        pitch_prob: float = 0.5,
    ):
        super().__init__()
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.sample_rate = sample_rate
        self.speed_prob = speed_prob
        self.pitch_prob = pitch_prob

    @torch.no_grad()
    def forward(
        self, audio_signal: torch.Tensor, signal_length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return audio_signal, signal_length

        B, T = audio_signal.shape

        # --- Speed perturbation (per-batch, changes duration) ---
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

        # --- Pitch perturbation (preserves duration) ---
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
                pass  # graceful fallback: skip pitch shift on error

        return audio_signal, signal_length


# ==========================================
# Save after every epoch
# ==========================================
class SaveEveryEpoch(Callback):
    """Save adapter weights + full .nemo checkpoint at the end of each epoch."""

    def __init__(self, directory: str):
        super().__init__()
        self.directory = directory

    def on_train_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        epoch = trainer.current_epoch
        nemo_path = os.path.join(self.directory, f"model_epoch{epoch}.nemo")
        adapter_path = os.path.join(self.directory, f"adapter_epoch{epoch}.pt")
        pl_module.save_to(nemo_path)
        pl_module.save_adapters(adapter_path)
        print(
            f"[SaveEveryEpoch] epoch {epoch} -> {nemo_path}, {adapter_path}",
            flush=True,
        )


# ==========================================
# Configuration
# ==========================================
MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"

ASR_DATA_DIR = "/kaggle/input/datasets/akarshkumarshukla/asr-data"
ASR_JSONL = os.path.join(ASR_DATA_DIR, "train_word_transcripts.jsonl")
TALKBANK_DIR = "/kaggle/input/datasets/akarshkumarshukla/talk-bank-data/audio"
TALKBANK_JSON = (
    "/kaggle/input/datasets/akarshkumarshukla/talkbank-transcript-full/"
    "train_word_transcripts.jsonl"
)

SAVE_DIR = "/kaggle/working/nemo_adapter_06b_v2"
MANIFEST_DIR = os.path.join(SAVE_DIR, "manifests")

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = 2
NUM_EPOCHS = 4
MAX_DURATION_SEC = 2000.0
LEARNING_RATE = 0.001

os.makedirs(MANIFEST_DIR, exist_ok=True)

print(f"torch:  {torch.__version__}")
print(f"CUDA:   {torch.version.cuda}")
print(f"numpy:  {np.__version__}")

# ==========================================
# Manifest Handling
# ==========================================
USE_EXISTING_MANIFESTS = os.environ.get(
    "USE_EXISTING_MANIFESTS", "0"
).strip().lower() in ("1", "true", "yes")
TRAIN_MANIFEST = os.environ.get(
    "TRAIN_MANIFEST",
    "/kaggle/input/datasets/akarshks/tr-manifest/train_manifest.jsonl",
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
                    "activation": "swish",
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
                    "warmup_ratio": 0.1,
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
            "name": "ParakeetAdapter06bV2",
            "create_tensorboard_logger": True,
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
ckpt_dir = os.path.join(exp_log_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
print(f"Logs: {exp_log_dir}")
print("Validation: DISABLED (training only)")

trainer.callbacks.append(SaveEveryEpoch(ckpt_dir))
print(f"Saving .nemo + adapter after every epoch to {ckpt_dir}")

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
# Waveform Augmentation (speed ±5%, pitch ±1 semitone)
# ==========================================
print("Attaching waveform augmentation (speed ±5%, pitch ±1 semitone)...")
waveform_aug = WaveformAugmentor(
    speed_min=0.95,
    speed_max=1.05,
    pitch_min=-1.0,
    pitch_max=1.0,
    sample_rate=16000,
    speed_prob=0.5,
    pitch_prob=0.5,
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
print("Waveform augmentation attached to model forward")

# SpecAugment (operates on mel spectrograms, standard NeMo)
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
# Adapter: Bottleneck 1024 → 128 → 1024 (Swish nonlinearity)
# ==========================================
print("Adding bottleneck adapter (1024 -> 128 -> 1024, activation=swish)...")
adapter_name = "encoder:asr_children_adapter"
adapter_type_cfg = cfg.model.adapter.linear
model.add_adapter(name=adapter_name, cfg=adapter_type_cfg)
model.set_enabled_adapters(enabled=False)
model.set_enabled_adapters(name=adapter_name, enabled=True)
model.freeze()
model.unfreeze_enabled_adapters()

# Optimizer (must see final requires_grad mask)
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
print(f"  Adapter:    1024 -> 128 -> 1024 (swish + LayerNorm + dropout=0.1)")
print(f"  Epochs:     {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  LR:         {LEARNING_RATE}")
print(f"  Augmentation:")
print(f"    - Speed perturbation: +/-5% (prob=0.5)")
print(f"    - Pitch perturbation: +/-1 semitone (prob=0.5)")
print(f"    - SpecAugment: freq_masks=2 w=27, time_masks=10 w=0.05")
print("=" * 60)

trainer.fit(model)

# ==========================================
# Save
# ==========================================
print("Saving...")
model.save_adapters(os.path.join(ckpt_dir, "adapter_final.pt"))
model.save_to(os.path.join(ckpt_dir, "model_final.nemo"))
print(f"Saved adapter_final.pt and model_final.nemo under {ckpt_dir}")
print("TRAINING COMPLETE!")
