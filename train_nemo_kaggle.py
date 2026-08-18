# ==========================================
# CELL 1: KAGGLE ENVIRONMENT SETUP
# ==========================================
# ⚠️ RUN THIS FIRST, THEN RESTART KERNEL
# ⚠️ DO NOT IMPORT ANYTHING BEFORE THIS CELL

"""
This setup script:
1. Fixes numpy/scipy version mismatch (your friend's fix)
2. Installs NeMo 2.5.0 with correct dependencies
3. Prevents all known Kaggle conflicts
"""

import subprocess
import sys
from pathlib import Path

def run_cmd(cmd):
    """Execute shell command safely"""
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def pip_install(*packages):
    """Install packages via pip"""
    run_cmd([sys.executable, "-m", "pip", "install", "--no-cache-dir", *packages])

def pip_uninstall(*packages):
    """Uninstall packages via pip"""
    run_cmd([sys.executable, "-m", "pip", "uninstall", "-y", *packages])

print("🧹 STEP 1/4: Removing conflicting packages...")
pip_uninstall(
    "datasets", "diffusers", "gradio", "peft", 
    "sentence-transformers", "transformers", 
    "huggingface_hub", "numpy", "pandas", 
    "numba", "scipy", "nemo_toolkit", 
    "pytorch-lightning", "lightning"
)

print("\n🔧 STEP 2/4: Installing clean numpy + scipy...")
# CRITICAL: numpy 2.x + scipy 1.14+ (your friend's fix)
pip_install(
    "numpy>=2.1,<2.3",
    "scipy>=1.14,<1.16"
)

print("\n⚡ STEP 3/4: Installing PyTorch 2.9 stack...")
# Must use PyTorch index for CUDA 12.6
run_cmd([
    sys.executable, "-m", "pip", "install",
    "--index-url", "https://download.pytorch.org/whl/cu126",
    "torch==2.9.0",
    "torchaudio==2.9.0",
    "torchvision==0.19.0"
])

print("\n📦 STEP 4/4: Installing NeMo 2.5 + dependencies...")
pip_install(
    # HuggingFace stack
    "transformers==4.57.6",
    "huggingface_hub==0.34.0",
    "datasets>=2.18.0",
    
    # Lightning (must be 2.4.0 for NeMo 2.5)
    "lightning==2.4.0",
    
    # Config system
    "omegaconf==2.3.0",
    "hydra-core==1.3.2",
    
    # Audio processing
    "soundfile==0.13.1",
    "librosa==0.10.2.post1",
    "sentencepiece==0.2.0",
    
    # ML utilities
    "scikit-learn==1.5.1",
    "jiwer>=3.0.0",
    "whisper-normalizer>=0.1.0",
    "loguru==0.7.2",
    "tqdm==4.67.1",
    
    # NeMo LAST (with --no-deps to prevent version conflicts)
    "--no-deps", "nemo_toolkit==2.5.0"
)

# Re-pin numpy/scipy in case NeMo tried to overwrite
print("\n🔒 Final step: Re-pinning numpy + scipy...")
pip_install(
    "--force-reinstall",
    "numpy>=2.1,<2.3",
    "scipy>=1.14,<1.16"
)

print("\n" + "="*60)
print("✅ INSTALLATION COMPLETE!")
print("="*60)
print("\n⚠️  CRITICAL: RESTART KERNEL NOW")
print("⚠️  Then run verification cell")
print("="*60)
# ==========================================
# VERIFICATION CELL
# Run this AFTER restarting kernel
# ==========================================

import sys
print(f"Python: {sys.version}\n")

# Check critical versions
packages = {
    "numpy": "2.1-2.3",
    "scipy": "1.14-1.16",
    "torch": "2.9.0",
    "torchaudio": "2.9.0",
    "transformers": "4.57.6",
    "lightning": "2.4.0",
    "nemo": "2.5.0",
    "omegaconf": "2.3.0",
}

print("📦 Package Versions:")
print("-" * 60)

all_correct = True
for pkg, expected in packages.items():
    try:
        mod = __import__(pkg.replace("-", "_"))
        version = getattr(mod, "__version__", "unknown")
        
        # Check if version matches expected
        if expected in version or version.startswith(expected.split("-")[0]):
            print(f"✅ {pkg:20s} {version:15s} (expected: {expected})")
        else:
            print(f"⚠️  {pkg:20s} {version:15s} (expected: {expected})")
            all_correct = False
    except ImportError:
        print(f"❌ {pkg:20s} NOT INSTALLED")
        all_correct = False

print("-" * 60)

# Test critical imports
print("\n🧪 Testing NeMo Imports...")
try:
    from nemo.collections.asr.models import ASRModel
    from nemo.utils.exp_manager import exp_manager
    from nemo.core.classes import adapter_mixins
    print("✅ All NeMo imports successful!")
except Exception as e:
    print(f"❌ NeMo import failed: {e}")
    all_correct = False

# Test CUDA
print("\n🔥 CUDA Status:")
import torch
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA: {torch.version.cuda}")
    test = torch.zeros(1).cuda()
    print(f"✅ CUDA test passed: {test.device}")
    del test
    torch.cuda.empty_cache()
else:
    print("⚠️  No GPU available")

print("\n" + "="*60)
if all_correct:
    print("🎉 ALL SYSTEMS GO!")
    print("✅ Ready for NeMo 2.5 training")
else:
    print("⚠️  Some packages have issues")
    print("Try reinstalling or check versions")
print("="*60)
# ==========================================
# NEMO 2.5 ADAPTER TRAINING SCRIPT
# ==========================================

import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import torch
import lightning.pytorch as pl
from sklearn.model_selection import train_test_split

# NeMo imports
from nemo.collections.asr.models import ASRModel
from nemo.utils.exp_manager import exp_manager
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

# Hyperparameters (Kaggle-optimized)
BATCH_SIZE        = 8   # Safe for most Kaggle GPUs
NUM_WORKERS       = 2   # Reduced from 8 (safer)
NUM_EPOCHS        = 3   # Start small
MAX_DURATION_SEC  = 20.0  # Reduced from 25
VAL_SPLIT         = 0.2
LEARNING_RATE     = 0.001
VAL_CHECK_INTERVAL = 0.5  # Check twice per epoch

os.makedirs(MANIFEST_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ==========================================
# Audio Indexing
# ==========================================
print("📁 Indexing audio files...")
audio_index = {}

def _index_directory(search_dir):
    local = {}
    if not os.path.isdir(search_dir):
        print(f"  ⚠️ {search_dir} not found")
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

print(f"✅ Indexed {len(audio_index):,} files\n")

def _find_audio(audio_path_field):
    return audio_index.get(os.path.basename(audio_path_field), None)

# ==========================================
# Build Manifests
# ==========================================
def build_nemo_manifests():
    print("📝 Building NeMo manifests...")
    records = []

    for filepath in [ASR_JSONL, TALKBANK_JSON]:
        if not os.path.exists(filepath):
            print(f"  ⚠️ Skipping {filepath}")
            continue
        
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("orthographic_text", "").strip().lower()
                    dur  = float(data.get("audio_duration_sec", 0.0))
                    path = _find_audio(data.get("audio_path", ""))

                    if path and text and (0.1 < dur <= MAX_DURATION_SEC):
                        records.append({
                            "audio_filepath": path,
                            "duration": dur,
                            "text": text
                        })
                except:
                    continue

    df = pd.DataFrame(records)
    print(f"  Total: {len(df):,} utterances\n")

    train_df, val_df = train_test_split(df, test_size=VAL_SPLIT, random_state=42)

    train_path = os.path.join(MANIFEST_DIR, "train_manifest.jsonl")
    val_path   = os.path.join(MANIFEST_DIR, "val_manifest.jsonl")

    train_df.to_json(train_path, orient="records", lines=True)
    val_df.to_json(val_path, orient="records", lines=True)

    print(f"  ✅ Train: {len(train_df):,} → {train_path}")
    print(f"  ✅ Val:   {len(val_df):,} → {val_path}\n")

    return train_path, val_path

# ==========================================
# Helper Functions
# ==========================================
def update_model_cfg(orig_cfg, new_cfg):
    """Deep merge configs"""
    with open_dict(orig_cfg):
        for k, v in new_cfg.items():
            orig_cfg[k] = v
    return orig_cfg

# ==========================================
# Main Training
# ==========================================
def main():
    # 1. Build manifests
    train_manifest, val_manifest = build_nemo_manifests()

    # 2. Build config
    cfg = OmegaConf.create({
        "model": {
            "pretrained_model": MODEL_ID,
            "log_prediction": False,
            "adapter": {
                "adapter_name": "asr_children_adapter",
                "adapter_module_name": "encoder",
                "adapter_type": "linear",
                "linear": {
                    "_target_": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
                    "in_features": 1024,
                    "dim": 64,  # Adapter dimension
                    "norm_position": "post",
                    "dropout": 0.1
                },
            },
            "train_ds": {
                "manifest_filepath": train_manifest,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "pin_memory": False,  # ← CRITICAL FIX
                "use_lhotse": False,
                "channel_selector": "average",
                "is_tarred": False,
                "shuffle": True,
            },
            "validation_ds": {
                "manifest_filepath": val_manifest,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "pin_memory": False,  # ← CRITICAL FIX
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
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "precision": "bf16-mixed" if torch.cuda.is_available() else 32,
            "max_epochs": NUM_EPOCHS,
            "val_check_interval": VAL_CHECK_INTERVAL,
            "enable_progress_bar": True,
            "log_every_n_steps": 100,
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

    # 3. Setup Trainer
    print("⚙️ Initializing Trainer...")
    trainer = pl.Trainer(
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
    )
    
    exp_log_dir = exp_manager(trainer, cfg.exp_manager)
    print(f"  ✅ Logs: {exp_log_dir}\n")

    # 4. Load Model
    print(f"📥 Loading {MODEL_ID}...")
    model = ASRModel.from_pretrained(MODEL_ID, trainer=trainer)
    
    # Disable CUDA graph decoder
    with open_dict(model.cfg):
        if hasattr(model.cfg, "decoding"):
            if hasattr(model.cfg.decoding, "greedy"):
                model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    
    if hasattr(model, 'change_decoding_strategy'):
        model.change_decoding_strategy(model.cfg.decoding)
    
    print("  ✅ Model loaded\n")

    # 5. Setup Data
    print("📊 Setting up data...")
    cfg.model.train_ds = update_model_cfg(model.cfg.train_ds, cfg.model.train_ds)
    model.setup_training_data(cfg.model.train_ds)
    
    cfg.model.validation_ds = update_model_cfg(model.cfg.validation_ds, cfg.model.validation_ds)
    model.setup_multiple_validation_data(cfg.model.validation_ds)
    print("  ✅ Data ready\n")

    # 6. Setup Optimizer
    print("⚙️ Setting up optimizer...")
    model.setup_optimization(cfg.model.optim)
    print("  ✅ Optimizer ready\n")

    # 7. Add Adapter
    print("🔧 Adding adapter...")
    adapter_cfg = cfg.model.adapter
    adapter_name = f"{adapter_cfg.adapter_module_name}:{adapter_cfg.adapter_name}"
    adapter_type_cfg = adapter_cfg[adapter_cfg.adapter_type]

    model.add_adapter(name=adapter_name, cfg=adapter_type_cfg)
    model.set_enabled_adapters(enabled=False)
    model.set_enabled_adapters(name=adapter_name, enabled=True)
    
    model.freeze()
    model.unfreeze_enabled_adapters()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total:     {total:,}")
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    print("  ✅ Adapter configured\n")

    # 8. Train
    print("="*60)
    print("🚀 STARTING TRAINING")
    print("="*60)
    
    try:
        trainer.fit(model)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

    # 9. Save
    print("\n💾 Saving adapter...")
    adapter_path = os.path.join(exp_log_dir, "checkpoints", "adapter_final.pt")
    model.save_adapters(adapter_path)
    print(f"  ✅ Saved: {adapter_path}")
    
    print("\n🎉 Training complete!")

if __name__ == "__main__":
    main()