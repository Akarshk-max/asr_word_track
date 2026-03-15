!pip install -q "numpy<2.0.0" nemo_toolkit[asr] lightning loguru omegaconf scikit-learn pandas
#  THIS VERISON IS NOT WORKING CORRECTLY WITH KAGGLE ENVOIRMENT , PACKAGE VERSIONS ALSO ARENT COMPATIBLE WITH SUBMISSON STRUCTURE
"""
======================================================
Kaggle Fine-Tuning: Nvidia Parakeet-TDT 0.6B (NeMo)
======================================================
This script adapts NVIDIA's pretrained Parakeet TDT 0.6B model using NeMo.
Instead of fine-tuning all model weights, this injects lightweight linear 
adapters into the Conformer encoder and trains ONLY the adapter weights.
This is identical to the official "On Top of Pasketti" reference implementation.

PREREQUISITES (run in a cell before this script):

"""

import os
import json
import logging
from pathlib import Path

# Core ML & audio stack
import pandas as pd
import torch
import lightning.pytorch as pl
from sklearn.model_selection import train_test_split

# NeMo imports
from nemo.collections.asr.models import ASRModel
from nemo.utils.exp_manager import exp_manager
from nemo.core import adapter_mixins
from nemo.utils.trainer_utils import resolve_trainer_cfg
from omegaconf import OmegaConf, open_dict

from concurrent.futures import ThreadPoolExecutor, as_completed
# ==========================================
# 1. Configuration
# ==========================================
MODEL_ID ="nvidia/parakeet-tdt-1.1b"

# --- Kaggle dataset paths ---
ASR_DATA_DIR  = "/kaggle/input/datasets/akarshkumarshukla/asr-data"
ASR_JSONL     = os.path.join(ASR_DATA_DIR, "train_word_transcripts.jsonl")

TALKBANK_DIR  = "/kaggle/input/datasets/akarshkumarshukla/talk-bank-data/audio"
TALKBANK_JSON = "/kaggle/input/datasets/akarshkumarshukla/talkbank-transcript-full/train_word_transcripts.jsonl"

SAVE_DIR      = "/kaggle/working/nemo_adapter_run"
MANIFEST_DIR  = os.path.join(SAVE_DIR, "manifests")

# Global Hyperparameters

NUM_WORKERS           = min(8, os.cpu_count() or 4)
MAX_DURATION_SEC      = 25.0
VAL_SPLIT             = 0.2
LEARNING_RATE         = 0.001

if not os.path.exists(MANIFEST_DIR):
    os.makedirs(MANIFEST_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ==========================================
# 2. Audio Indexing & Data Preparation
# ==========================================
print("Indexing audio files (parallel)...")
audio_index = {}

def _index_directory(search_dir):
    """Walk one directory and return {filename: fullpath} for all .flac files."""
    local = {}
    if not os.path.isdir(search_dir):
        print(f"  Warning: {search_dir} not found, skipping.")
        return local
    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.endswith(".flac"):
                local[f] = os.path.join(root, f)
    return local

with ThreadPoolExecutor(max_workers=2) as ex:
    futures = {ex.submit(_index_directory, d): d for d in [ASR_DATA_DIR, TALKBANK_DIR]}
    for fut in as_completed(futures):
        audio_index.update(fut.result())

print(f"Indexed {len(audio_index)} total audio files")


def _find_audio(audio_path_field):
    return audio_index.get(os.path.basename(audio_path_field), None)


def build_nemo_manifests():
    """
    Load TalkBank and ASR Data JSONL files.
    Format them EXACTLY as NeMo requires.
    Remove clips > 25 seconds.
    Split into train/val and write to disk.
    """
    print("Building datasets...")
    records = []

    for filepath in [ASR_JSONL, TALKBANK_JSON]:
        if not os.path.exists(filepath):
            print(f"  Skipping: {filepath} (Not found)")
            continue
        
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                
                text = data.get("orthographic_text", "").strip().lower()
                dur  = float(data.get("audio_duration_sec", 0.0))
                path = _find_audio(data.get("audio_path", ""))

                if path and text and (dur > 0) and (dur <= MAX_DURATION_SEC):
                    records.append({
                        "audio_filepath": path,
                        "duration": dur,
                        "text": text
                    })

    df = pd.DataFrame(records)
    print(f"Total valid utterances (<= {MAX_DURATION_SEC}s): {len(df)}")

    # Train / Val split
    train_df, val_df = train_test_split(df, test_size=VAL_SPLIT, random_state=42)

    train_path = os.path.join(MANIFEST_DIR, "train_manifest.jsonl")
    val_path   = os.path.join(MANIFEST_DIR, "val_manifest.jsonl")

    train_df.to_json(train_path, orient="records", lines=True)
    val_df.to_json(val_path, orient="records", lines=True)

    print(f"Train manifest saved ({len(train_df)} rows): {train_path}")
    print(f"Val manifest saved ({len(val_df)} rows): {val_path}")

    return train_path, val_path

BATCH_SIZE=50
def update_model_cfg(orig_cfg, new_cfg):
    """Deep merge configurations."""
    with open_dict(orig_cfg):
        for k, v in new_cfg.items():
            orig_cfg[k] = v
    return orig_cfg
def update_model_config_to_support_adapter(model_cfg, current_cfg):
    """Swap the encoder target class to its adapter-compatible variant."""
    with open_dict(model_cfg):
        model_cfg.log_prediction = current_cfg.model.get('log_prediction', False)

        adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder._target_)
        if adapter_metadata is not None:
            model_cfg.encoder._target_ = adapter_metadata.adapter_class_path

def main():
    # 1. Build manifests
    train_manifest_path, val_manifest_path = build_nemo_manifests()

    # 2. Build configuration
    cfg = OmegaConf.create({
        "model": {
            "pretrained_model": MODEL_ID,
            "adapter": {
                "adapter_name": "asr_children_orthographic",
                "adapter_module_name": "encoder",
                "adapter_type": "linear",
                "linear": {   # Changed from 'linear' to 'encoder', based on GPT insight
                    "_target_": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
                    "in_features": 1024,
                    "dim": 32,
                    "norm_position": "post",
                    "dropout": 0.1
                },
            },
            "train_ds": {
                "manifest_filepath": str(train_manifest_path),
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "use_lhotse": False,
                "channel_selector": "average",
                "is_tarred": False, 
                "pin_memory": True,
            },
            "validation_ds": {
                "manifest_filepath": str(val_manifest_path),
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "use_lhotse": False,
                "channel_selector": "average",
                "is_tarred": False, 
                "pin_memory": True,
            },
            "optim": {
                "name": "adamw",
                "lr": LEARNING_RATE,
                "weight_decay": 0.0,
                "sched": {
                    "name": "CosineAnnealing",
                    "warmup_ratio": 0.1,
                    "min_lr": 1e-5
                }
            },
        },
        "trainer": {
            "devices": 1 if torch.cuda.is_available() else 0,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "precision": "bf16-mixed" if torch.cuda.is_available() else 32,
            "strategy": "auto",
            "max_epochs": 1, 
            "val_check_interval": 1.0,  # Evaluate at end of epoch
            "enable_progress_bar": True,
        },
        "exp_manager": {
            "exp_dir": SAVE_DIR,
            "name": "ParakeetAdapter",
            "create_tensorboard_logger": True,
            "create_checkpoint_callback": True,
            "checkpoint_callback_params": {
                "monitor": "val_wer",
                "mode": "min",
                "save_top_k": 3,
            }
        },
    })

    # 3. Setup Trainer
    print("Initializing PyTorch Lightning Trainer...")
    trainer_kwargs = resolve_trainer_cfg(cfg.trainer)
    trainer_kwargs["logger"] = False  # Prevent NeMo exp_manager from complaining about duplicate loggers
    trainer_kwargs["enable_checkpointing"] = False  # Prevent NeMo exp_manager from complaining about duplicate ModelCheckpoints
    trainer = pl.Trainer(**trainer_kwargs)
    exp_log_dir = exp_manager(trainer, cfg.get("exp_manager", None))

    # 4. Load Model and setup adapters
    print(f"Loading Base Model Config: {cfg.model.pretrained_model}...")
    
    # Load config first
    model_cfg = ASRModel.from_pretrained(cfg.model.pretrained_model, return_config=True)

    # Patch config to support adapters
    update_model_config_to_support_adapter(model_cfg, cfg)

    # Now load the model with the patched architecture
    print("Initializing patched ASRModel...")
    model = ASRModel.from_pretrained(
        cfg.model.pretrained_model,
        override_config_path=model_cfg,
        trainer=trainer,
    )

    # Disable cuda graph decoding for compatibility
    with open_dict(model.cfg):
        if "decoding" in model.cfg and hasattr(model.cfg.decoding, "greedy"):
            model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    
    # Refresh decoding strategy just in case
    # Required for some versions of Nemo to acknowledge the cuda graph config
    if hasattr(model, 'change_decoding_strategy'):
        model.change_decoding_strategy(model.cfg.decoding)

    # 5. Setup Data
    print("Setting up data loaders...")
    model.setup_training_data(update_model_cfg(model.cfg.train_ds, cfg.model.train_ds))
    model.setup_multiple_validation_data(update_model_cfg(model.cfg.validation_ds, cfg.model.validation_ds))
    
    # 6. Setup Optimizer
    model.setup_optimization(cfg.model.optim)
    
    # 7. Add & Enable Adapter
    print("Configuring Adapters...")
    adapter_cfg = cfg.model.adapter

    adapter_name = adapter_cfg.adapter_name
    adapter_type = adapter_cfg.adapter_type
    adapter_module_name = adapter_cfg.adapter_module_name
    adapter_type_cfg = adapter_cfg[adapter_type]

    if adapter_module_name is not None and ":" not in adapter_name:
        adapter_name = f"{adapter_module_name}:{adapter_name}"

    model.add_adapter(adapter_name, cfg=adapter_type_cfg)
    
    # Enable and isolate training
    model.set_enabled_adapters(enabled=False)
    model.set_enabled_adapters(adapter_name, enabled=True)
    model.freeze()  # Freeze ALL base model weights
    model.unfreeze_enabled_adapters()  # Open adapter weights for training

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # 8. Train
    print("="*60)
    print("STARTING ADAPTER TRAINING")
    print("="*60)
    trainer.fit(model)

    # 9. Save standalone adapter checkpoint
    print("Training complete! Saving adapter weights...")
    state_path = exp_log_dir if exp_log_dir is not None else os.getcwd()
    ckpt_path = os.path.join(state_path, "checkpoints")
    if os.path.exists(ckpt_path):
        state_path = ckpt_path
    
    adapter_save_loc = os.path.join(state_path, "asr_children_adapter.ckpt")
    model.save_adapters(adapter_save_loc)
    print(f"Saved standalone NeMo adapter to: {adapter_save_loc}")


if __name__ == "__main__":
    main()
