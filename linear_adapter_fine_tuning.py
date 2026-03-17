# ==========================================
# CELL 1: INSTALL + TRAIN (SINGLE CELL)
# ==========================================
# This does EVERYTHING in one cell:
#   1. Installs packages
#   2. Writes training script to disk
#   3. Runs training in a SUBPROCESS (fresh Python = no stale imports)
#  the first attempt that worked
import subprocess
import sys
import os

# ======================
# PART A: INSTALLATION
# ======================
def _pip(*args):
    subprocess.check_call(
        [sys.executable, "-m", "pip", *args],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )

print("🧹 Step 1: Cleaning...")
for _ in range(2):
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y",
         "numpy", "scipy", "nemo_toolkit", "lightning",
         "pytorch-lightning", "datasets", "diffusers", "gradio",
         "peft", "sentence-transformers", "transformers",
         "huggingface_hub", "torch", "torchaudio", "torchvision", "numba"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

print("🔧 Step 2: numpy + scipy...")
_pip("install", "--no-cache-dir", "numpy>=2.1,<2.3", "scipy>=1.14,<1.16")

print("⚡ Step 3: PyTorch...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--no-cache-dir",
    "--index-url", "https://download.pytorch.org/whl/cu126",
    "torch>=2.9.0", "torchaudio"
], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

print("📦 Step 4: Dependencies...")
_pip("install", "--no-cache-dir",
     "transformers>=4.57.6,<4.58", "huggingface_hub>=0.30.0",
     "lightning>=2.2.0", "omegaconf>=2.3.0", "hydra-core>=1.3.2",
     "soundfile>=0.12.0", "librosa>=0.10.0", "sentencepiece>=0.2.0",
     "datasets>=2.18.0", "pandas>=2.0.0", "scikit-learn>=1.4.0",
     "loguru>=0.7.0", "jiwer>=3.0.0", "tqdm>=4.60.0",
     "webdataset>=0.2.80", "braceexpand>=0.1.7", "editdistance>=0.6.0")

print("🔥 Step 5: NeMo...")
_pip("install", "--no-cache-dir", "nemo_toolkit[asr]>=2.5.0")

print("🔒 Step 6: Re-pin numpy...")
_pip("install", "--no-cache-dir", "--force-reinstall",
     "numpy>=2.1,<2.3", "scipy>=1.14,<1.16")

print("✅ Installation complete!\n")

# ======================
# PART B: WRITE TRAINING SCRIPT TO DISK
# ======================
TRAIN_SCRIPT = "/kaggle/working/train_nemo.py"

train_code = '''
import warnings
import logging
warnings.filterwarnings("ignore", category=Warning, module="numba")
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import torch
import lightning.pytorch as pl
from sklearn.model_selection import train_test_split

from nemo.collections.asr.models import ASRModel
from nemo.utils.exp_manager import exp_manager
from nemo.core.classes import adapter_mixins
from omegaconf import OmegaConf, open_dict

# ==========================================
# Configuration
# ==========================================
MODEL_ID = "nvidia/parakeet-tdt-1.1b"

ASR_DATA_DIR  = "/kaggle/input/datasets/akarshkumarshukla/asr-data"
ASR_JSONL     = os.path.join(ASR_DATA_DIR, "train_word_transcripts.jsonl")
TALKBANK_DIR  = "/kaggle/input/datasets/akarshkumarshukla/talk-bank-data/audio"
TALKBANK_JSON = "/kaggle/input/datasets/akarshkumarshukla/talkbank-transcript-full/train_word_transcripts.jsonl"

SAVE_DIR     = "/kaggle/working/nemo_adapter_run"
MANIFEST_DIR = os.path.join(SAVE_DIR, "manifests")

BATCH_SIZE        = 64
NUM_WORKERS       = 2
NUM_EPOCHS        = 3
MAX_DURATION_SEC  = 20.0
VAL_SPLIT         = 0.2
LEARNING_RATE     = 0.001
VAL_CHECK_INTERVAL = 0.5

os.makedirs(MANIFEST_DIR, exist_ok=True)

print(f"torch:  {torch.__version__}")
print(f"CUDA:   {torch.version.cuda}")

import numpy as np
print(f"numpy:  {np.__version__}")

# ==========================================
# Audio Indexing
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
        ex.submit(_index_directory, TALKBANK_DIR)
    ]
    for fut in futures:
        audio_index.update(fut.result())

print(f"Indexed {len(audio_index):,} files")

def _find_audio(path):
    return audio_index.get(os.path.basename(path), None)

# ==========================================
# Build Manifests
# ==========================================
print("Building manifests...")
records = []
for filepath in [ASR_JSONL, TALKBANK_JSON]:
    if not os.path.exists(filepath):
        continue
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                text = data.get("orthographic_text", "").strip().lower()
                dur = float(data.get("audio_duration_sec", 0.0))
                path = _find_audio(data.get("audio_path", ""))
                if path and text and (0.1 < dur <= MAX_DURATION_SEC):
                    records.append({"audio_filepath": path, "duration": dur, "text": text})
            except:
                continue

df = pd.DataFrame(records)
print(f"Total: {len(df):,} utterances")

train_df, val_df = train_test_split(df, test_size=VAL_SPLIT, random_state=42)
train_path = os.path.join(MANIFEST_DIR, "train_manifest.jsonl")
val_path = os.path.join(MANIFEST_DIR, "val_manifest.jsonl")
train_df.to_json(train_path, orient="records", lines=True)
val_df.to_json(val_path, orient="records", lines=True)
print(f"Train: {len(train_df):,}, Val: {len(val_df):,}")

# ==========================================
# Helper
# ==========================================
def update_model_cfg(orig_cfg, new_cfg):
    with open_dict(orig_cfg):
        for k, v in new_cfg.items():
            orig_cfg[k] = v
    return orig_cfg

# ==========================================
# Config
# ==========================================
cfg = OmegaConf.create({
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
                "dropout": 0.1
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
                "min_lr": 1e-6
            }
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
        "name": "ParakeetAdapter",
        "create_tensorboard_logger": True,
        "create_checkpoint_callback": True,
        "checkpoint_callback_params": {
            "monitor": "val_wer",
            "mode": "min",
            "save_top_k": 2,
            "filename": "parakeet-{epoch:02d}-{val_wer:.4f}",
        },
        "resume_if_exists": False,
        "resume_ignore_no_checkpoint": True,
    },
})

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
print(f"Logs: {exp_log_dir}")

# ==========================================
# Load Model with Adapter Support
# ==========================================
print(f"Loading {MODEL_ID}...")
model_cfg = ASRModel.from_pretrained(MODEL_ID, return_config=True)

with open_dict(model_cfg):
    adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder._target_)
    if adapter_metadata is not None:
        model_cfg.encoder._target_ = adapter_metadata.adapter_class_path
        print(f"Patched encoder to: {adapter_metadata.adapter_class_path}")

model = ASRModel.from_pretrained(MODEL_ID, override_config_path=model_cfg, trainer=trainer)

with open_dict(model.cfg):
    if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
        model.cfg.decoding.greedy.use_cuda_graph_decoder = False
if hasattr(model, "change_decoding_strategy"):
    model.change_decoding_strategy(model.cfg.decoding)

print("Model loaded")

# SpecAugment
if "spec_augment" in cfg.model:
    model.spec_augmentation = model.from_config_dict(cfg.model.spec_augment)
    print("SpecAugment enabled")

# ==========================================
# Data
# ==========================================
print("Setting up data...")
cfg.model.train_ds = update_model_cfg(model.cfg.train_ds, cfg.model.train_ds)
model.setup_training_data(cfg.model.train_ds)
cfg.model.validation_ds = update_model_cfg(model.cfg.validation_ds, cfg.model.validation_ds)
model.setup_multiple_validation_data(cfg.model.validation_ds)

# ==========================================
# Optimizer
# ==========================================
model.setup_optimization(cfg.model.optim)

# ==========================================
# Adapter
# ==========================================
print("Adding adapter...")
adapter_name = "encoder:asr_children_adapter"
adapter_type_cfg = cfg.model.adapter.linear
model.add_adapter(name=adapter_name, cfg=adapter_type_cfg)
model.set_enabled_adapters(enabled=False)
model.set_enabled_adapters(name=adapter_name, enabled=True)
model.freeze()
model.unfreeze_enabled_adapters()

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,}, Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

# ==========================================
# Train
# ==========================================
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)
trainer.fit(model)

# ==========================================
# Save
# ==========================================
print("Saving...")
ckpt_dir = os.path.join(exp_log_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
model.save_adapters(os.path.join(ckpt_dir, "adapter_final.pt"))
model.save_to(os.path.join(ckpt_dir, "model_final.nemo"))
print("TRAINING COMPLETE!")
'''

with open(TRAIN_SCRIPT, "w") as f:
    f.write(train_code)

print(f"📝 Training script written to {TRAIN_SCRIPT}\n")

# ======================
# PART C: RUN IN SUBPROCESS (FRESH PYTHON)
# ======================
print("=" * 60)
print("🚀 LAUNCHING TRAINING IN FRESH SUBPROCESS")
print("=" * 60)
print("(This bypasses the stale numpy problem)\n")

# Run training in a completely fresh Python process
result = subprocess.run(
    [sys.executable, TRAIN_SCRIPT],
    cwd="/kaggle/working",
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
    # Stream output live to notebook
    stdout=None,  # Inherit stdout (prints to notebook)
    stderr=subprocess.STDOUT,  # Merge stderr into stdout
)

if result.returncode != 0:
    print(f"\n❌ Training failed with exit code {result.returncode}")
else:
    print(f"\n✅ Training completed successfully!")
    print("📁 Check /kaggle/working/nemo_adapter_run/ for outputs")
