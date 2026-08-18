"""
Stage 2: Load Stage 1 ``model_stage1_fusion.nemo``, re-bind layer fusion forward, partially unfreeze encoder.

Run after Stage 1 on the same machine (or copy manifests + .nemo). Adjust paths below.
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
import warnings

warnings.filterwarnings("ignore", category=Warning, module="numba")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import logging

logging.getLogger("nemo_logger").setLevel(logging.ERROR)

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.utils.exp_manager import exp_manager

from nemo_layer_fusion import attach_layer_fusion_to_model


def _state_dict_from_nemo(nemo_path: str) -> dict:
    with tarfile.open(nemo_path, "r:*") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            if not (
                m.name.endswith(".ckpt")
                or m.name.endswith(".pth")
                or "model_weights" in m.name.replace("\\", "/")
            ):
                continue
            try:
                with tar.extractfile(m) as f:
                    blob = f.read()
                obj = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                return obj["state_dict"]
            if any(
                isinstance(k, str)
                and (
                    k.startswith("encoder.")
                    or k.startswith("decoder.")
                    or k.startswith("layer_fusion.")
                )
                for k in obj
            ):
                return obj
    raise RuntimeError(f"No state_dict in {nemo_path}")


def _load_layer_fusion_weights_from_nemo(model: torch.nn.Module, nemo_path: str) -> None:
    fusion = getattr(model, "layer_fusion", None)
    if fusion is None:
        print("[warn] No layer_fusion — skip fusion weight load.", flush=True)
        return
    sd = _state_dict_from_nemo(nemo_path)
    prefix = "layer_fusion."
    lf = {
        k[len(prefix) :]: v
        for k, v in sd.items()
        if isinstance(k, str) and k.startswith(prefix)
    }
    if not lf:
        print("[warn] No layer_fusion.* in Stage 1 .nemo", flush=True)
        return
    dev = next(fusion.parameters()).device
    lf = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in lf.items()}
    fusion.load_state_dict(lf, strict=True)
    print(f"Loaded fusion submodule ({len(lf)} tensor(s)) from Stage 1 archive.", flush=True)


# ==========================================
# Paths (edit for your Kaggle session)
# ==========================================
# Prefer env STAGE1_NEMO; else search under the Stage 1 experiment dir.
STAGE1_NEMO_EXPLICIT = os.environ.get(
    "STAGE1_NEMO",
    "/kaggle/working/nemo_layer_fusion_stage1/model_stage1_fusion.nemo",
)

MANIFEST_DIR = os.environ.get(
    "MANIFEST_DIR",
    "/kaggle/working/nemo_layer_fusion_stage1/manifests",
)
train_path = os.path.join(MANIFEST_DIR, "train_manifest.jsonl")
val_path = os.path.join(MANIFEST_DIR, "val_manifest.jsonl")

SAVE_DIR = "/kaggle/working/nemo_layer_fusion_stage2"

BATCH_SIZE = 32
NUM_WORKERS = 2
NUM_EPOCHS = 2
VAL_CHECK_INTERVAL = 0.5
# Single LR for all trainable params in Stage 2 (tune down if WER degrades)
LEARNING_RATE = 5e-5
# Unfreeze only the last K Conformer blocks (subsampling / front stays frozen)
UNFREEZE_LAST_K = 6

ADAPTER_NAME = "encoder:asr_children_adapter"


def _resolve_stage1_nemo() -> str:
    if os.path.isfile(STAGE1_NEMO_EXPLICIT):
        return os.path.abspath(STAGE1_NEMO_EXPLICIT)
    import glob

    matches = glob.glob(
        "/kaggle/working/nemo_layer_fusion_stage1/**/model_stage1_fusion.nemo",
        recursive=True,
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return max(matches, key=os.path.getmtime)
    raise FileNotFoundError(
        "Could not find model_stage1_fusion.nemo. Set env STAGE1_NEMO to the full path."
    )


def update_model_cfg(orig_cfg, new_cfg):
    with open_dict(orig_cfg):
        for k, v in new_cfg.items():
            orig_cfg[k] = v
    return orig_cfg


nemo_path = _resolve_stage1_nemo()
print(f"Restoring from: {nemo_path}")

cfg = OmegaConf.create(
    {
        "model": {
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
                    "min_lr": 1e-7,
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
            "log_every_n_steps": 200,
            "gradient_clip_val": 1.0,
        },
        "exp_manager": {
            "exp_dir": SAVE_DIR,
            "name": "ParakeetLayerFusionStage2",
            "create_tensorboard_logger": True,
            "create_checkpoint_callback": True,
            "checkpoint_callback_params": {
                "monitor": "val_wer",
                "mode": "min",
                "save_top_k": 2,
                "filename": "parakeet-s2-{epoch:02d}-{val_wer:.4f}",
            },
            "resume_if_exists": False,
            "resume_ignore_no_checkpoint": True,
        },
    }
)

trainer = pl.Trainer(
    devices=1,
    accelerator="gpu",
    precision="bf16-mixed",
    max_epochs=NUM_EPOCHS,
    val_check_interval=VAL_CHECK_INTERVAL,
    enable_progress_bar=True,
    log_every_n_steps=200,
    gradient_clip_val=1.0,
    logger=False,
    enable_checkpointing=False,
)
exp_log_dir = exp_manager(trainer, cfg.exp_manager)
print(f"Logs: {exp_log_dir}")

try:
    model = ASRModel.restore_from(nemo_path, trainer=trainer, strict=False)
except TypeError:
    model = ASRModel.restore_from(nemo_path, trainer=trainer)

with open_dict(model.cfg):
    if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
        model.cfg.decoding.greedy.use_cuda_graph_decoder = False
if hasattr(model, "change_decoding_strategy"):
    model.change_decoding_strategy(model.cfg.decoding)

print("Re-binding layer fusion forward; loading layer_fusion.* from .nemo...")
attach_layer_fusion_to_model(model, init_strategy="last_layer", module_name="layer_fusion")
_load_layer_fusion_weights_from_nemo(model, nemo_path)

# --- selective unfreeze: last K conformer layers + adapters + fusion ---
model.freeze()
n_layers = len(model.encoder.layers)
if UNFREEZE_LAST_K > n_layers:
    raise ValueError(f"UNFREEZE_LAST_K={UNFREEZE_LAST_K} > encoder layers {n_layers}")
first_trainable = n_layers - UNFREEZE_LAST_K

for i, layer in enumerate(model.encoder.layers):
    requires = i >= first_trainable
    for p in layer.parameters():
        p.requires_grad = requires

if getattr(model, "is_adapter_available", None) and model.is_adapter_available():
    model.set_enabled_adapters(enabled=False)
    model.set_enabled_adapters(name=ADAPTER_NAME, enabled=True)
    model.unfreeze_enabled_adapters()
else:
    print("Warning: no adapters detected on checkpoint; training encoder + fusion only.")

for p in model.layer_fusion.parameters():
    p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Stage 2 trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
print("Fusion importance:", model.layer_fusion.importance().cpu().numpy())

cfg.model.train_ds = update_model_cfg(model.cfg.train_ds, cfg.model.train_ds)
model.setup_training_data(cfg.model.train_ds)
cfg.model.validation_ds = update_model_cfg(
    model.cfg.validation_ds, cfg.model.validation_ds
)
model.setup_multiple_validation_data(cfg.model.validation_ds)

model.setup_optimization(cfg.model.optim)

print("=" * 60)
print("STAGE 2: partial encoder unfreeze + adapters + fusion")
print("=" * 60)
trainer.fit(model)

ckpt_dir = os.path.join(exp_log_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
model.save_adapters(os.path.join(ckpt_dir, "adapter_stage2.pt"))
torch.save(
    {"raw_weights": model.layer_fusion.raw_weights.detach().cpu()},
    os.path.join(ckpt_dir, "layer_fusion_weights_stage2.pt"),
)
model.save_to(os.path.join(ckpt_dir, "model_stage2_fusion.nemo"))
print(f"Saved Stage 2 artifacts to {ckpt_dir}")
print("STAGE 2 COMPLETE")
