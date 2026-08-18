"""
Kaggle / local Stage 1: NeMo parakeet-tdt-1.1b + linear encoder adapters + learnable layer fusion.

Place this file next to nemo_layer_fusion.py (same directory) or add that directory to PYTHONPATH.

Expected layout on Kaggle:
  /kaggle/working/train_nemo_layer_fusion_stage1.py
  /kaggle/working/nemo_layer_fusion.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=Warning, module="numba")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import logging

logging.getLogger("nemo_logger").setLevel(logging.ERROR)

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from sklearn.model_selection import train_test_split

from nemo.collections.asr.models import ASRModel
from nemo.utils.exp_manager import exp_manager
from nemo.core.classes import adapter_mixins
from omegaconf import OmegaConf, open_dict

from nemo_layer_fusion import attach_layer_fusion_to_model

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

SAVE_DIR = "/kaggle/working/nemo_layer_fusion_stage1"
MANIFEST_DIR = os.path.join(SAVE_DIR, "manifests")

BATCH_SIZE = 64
NUM_WORKERS = 2
NUM_EPOCHS = 3
MAX_DURATION_SEC = 20.0
VAL_SPLIT = 0.2
LEARNING_RATE = 0.001
VAL_CHECK_INTERVAL = 0.5

FUSION_INIT = "last_layer"  # "last_layer" | "uniform" | "edges"

os.makedirs(MANIFEST_DIR, exist_ok=True)

print(f"torch: {torch.__version__}")
print(f"CUDA:  {torch.version.cuda}")
print(f"numpy: {np.__version__}")

# ==========================================
# Audio indexing
# ==========================================
print("Indexing audio files...")
audio_index = {}


def _index_directory(search_dir):
    local = {}
    if not os.path.isdir(search_dir):
        return local
    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.endswith(".flac"):
                local[f] = os.path.join(root, f)
    return local


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


# ==========================================
# Manifests
# ==========================================
print("Building manifests...")
records = []
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

train_df, val_df = train_test_split(df, test_size=VAL_SPLIT, random_state=42)
train_path = os.path.join(MANIFEST_DIR, "train_manifest.jsonl")
val_path = os.path.join(MANIFEST_DIR, "val_manifest.jsonl")
train_df.to_json(train_path, orient="records", lines=True)
val_df.to_json(val_path, orient="records", lines=True)
print(f"Train: {len(train_df):,}, Val: {len(val_df):,}")


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
# OmegaConf
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
                    "dim": 64,
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
            "validation_ds": {
                "manifest_filepath": val_path,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "pin_memory": False,
                "use_lhotse": False,
                "channel_selector": "average",
                "is_tarred": False,
                "shuffle": False,
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
            "val_check_interval": VAL_CHECK_INTERVAL,
            "enable_progress_bar": True,
            "log_every_n_steps": 2000,
            "gradient_clip_val": 1.0,
        },
        "exp_manager": {
            "exp_dir": SAVE_DIR,
            "name": "ParakeetAdapterLayerFusion",
            "create_tensorboard_logger": True,
            "create_checkpoint_callback": False,
            "resume_if_exists": False,
            "resume_ignore_no_checkpoint": True,
        },
    }
)


def _parse_nemo_save_steps() -> list[int]:
    raw = os.environ.get("NEMO_SAVE_AT_STEPS", "6000").strip()
    if not raw:
        return []
    out: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if p.isdigit():
            out.append(int(p))
    return sorted(set(out))


class SaveFullNemoAtSteps(Callback):
    def __init__(self, directory: str, steps: list[int]):
        super().__init__()
        self.directory = directory
        self.targets = sorted(set(int(s) for s in steps))
        self._saved: set[int] = set()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        gs = int(trainer.global_step)
        for t in self.targets:
            if gs == t and t not in self._saved:
                self._saved.add(t)
                path = os.path.join(self.directory, f"model_stage1_step{t}.nemo")
                pl_module.save_to(path)
                print(f"[SaveFullNemoAtSteps] saved {path} (global_step={gs})", flush=True)


# ==========================================
# Trainer
# ==========================================
print("Initializing Trainer...")
trainer = pl.Trainer(
    devices=1,
    accelerator="gpu",
    precision="bf16-mixed",
    max_epochs=NUM_EPOCHS,
    val_check_interval=VAL_CHECK_INTERVAL,
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
print(f"Checkpoints dir: {ckpt_dir}")

# ==========================================
# Load model + adapter-enabled encoder
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
# Layer fusion (replaces forward encoder output)
# ==========================================
print("Attaching learnable layer fusion...")
attach_layer_fusion_to_model(model, init_strategy=FUSION_INIT, module_name="layer_fusion")
print(f"Fusion over {model.layer_fusion.num_layers} encoder layers")

# ==========================================
# SpecAugment
# ==========================================
if "spec_augment" in cfg.model:
    model.spec_augmentation = model.from_config_dict(cfg.model.spec_augment)
    print("SpecAugment enabled")

# ==========================================
# Data
# ==========================================
print("Setting up data...")
cfg.model.train_ds = update_model_cfg(model.cfg.train_ds, cfg.model.train_ds)
model.setup_training_data(cfg.model.train_ds)
cfg.model.validation_ds = update_model_cfg(
    model.cfg.validation_ds, cfg.model.validation_ds
)
model.setup_multiple_validation_data(cfg.model.validation_ds)

# ==========================================
# Adapters + fusion + decoder/joint trainability (optimizer after final mask)
# ==========================================
print("Adding adapter...")
adapter_name = "encoder:asr_children_adapter"
adapter_type_cfg = cfg.model.adapter.linear
model.add_adapter(name=adapter_name, cfg=adapter_type_cfg)
model.set_enabled_adapters(enabled=False)
model.set_enabled_adapters(name=adapter_name, enabled=True)
model.freeze()
model.unfreeze_enabled_adapters()

for p in model.layer_fusion.parameters():
    p.requires_grad = True

TRAIN_DECODER_JOINT = os.environ.get("TRAIN_DECODER_JOINT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
if TRAIN_DECODER_JOINT:
    if hasattr(model, "decoder"):
        for p in model.decoder.parameters():
            p.requires_grad = True
    if hasattr(model, "joint"):
        for p in model.joint.parameters():
            p.requires_grad = True
    print("Decoder + joint: trainable (TRAIN_DECODER_JOINT=0 to freeze)")

model.setup_optimization(cfg.model.optim)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,}, Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")

imp = model.layer_fusion.importance().cpu().numpy()
print("Initial fusion softmax:", imp)

_save_steps = _parse_nemo_save_steps()
if _save_steps:
    trainer.callbacks.append(SaveFullNemoAtSteps(ckpt_dir, _save_steps))
    print(
        f"Full-model .nemo snapshots at global_steps {_save_steps} "
        "(override with env NEMO_SAVE_AT_STEPS)"
    )

# ==========================================
# Train
# ==========================================
print("=" * 60)
print(
    "STAGE 1: encoder frozen + adapters + layer fusion"
    + (" + decoder + joint" if TRAIN_DECODER_JOINT else "")
)
print("=" * 60)
trainer.fit(model)

# ==========================================
# Save
# ==========================================
print("Saving...")
model.save_adapters(os.path.join(ckpt_dir, "adapter_final.pt"))
torch.save(
    {"raw_weights": model.layer_fusion.raw_weights.detach().cpu()},
    os.path.join(ckpt_dir, "layer_fusion_weights.pt"),
)
model.save_to(os.path.join(ckpt_dir, "model_stage1_fusion.nemo"))
print(f"Saved .nemo and adapters under {ckpt_dir}")
imp = model.layer_fusion.importance().cpu().numpy()
print("Final fusion softmax:", imp)
print("STAGE 1 COMPLETE")
