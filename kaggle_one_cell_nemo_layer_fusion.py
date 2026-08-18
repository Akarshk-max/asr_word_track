# -*- coding: utf-8 -*-
import os; os.environ["USE_EXISTING_MANIFESTS"]="1"
"""
Paste this entire file into ONE Kaggle notebook cell (or run locally).

  Part A — pip install (clean env + NeMo)
  Part B — write ONE training script to /kaggle/working (fusion code inlined, no import clash)
  Part C — subprocess.run(fresh Python interpreter)

Outputs: /kaggle/working/nemo_layer_fusion_stage1/ ...
"""

import os
import subprocess
import sys


def _pip(*args):
    subprocess.check_call(
        [sys.executable, "-m", "pip", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
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
_pip("install", "--no-cache-dir", "numpy>=2.1,<2.3", "scipy>=1.14,<1.16")

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
# PART B: WRITE SINGLE SELF-CONTAINED TRAINING SCRIPT (no nemo_layer_fusion import)
# =============================================================================
TRAIN_SCRIPT = "/kaggle/working/train_nemo_layer_fusion_stage1_standalone.py"

train_code = r'''from __future__ import annotations

import json
import logging
import os
import types
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Literal

warnings.filterwarnings("ignore", category=Warning, module="numba")
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from sklearn.model_selection import train_test_split

from nemo.collections.asr.models import ASRModel
from nemo.core.classes import adapter_mixins
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf, open_dict

InitStrategy = Literal["last_layer", "uniform", "edges"]


class LearnableLayerFusion(nn.Module):
    def __init__(self, num_layers: int, init_strategy: InitStrategy = "last_layer"):
        super().__init__()
        self.num_layers = num_layers
        self.raw_weights = nn.Parameter(torch.zeros(num_layers))
        if init_strategy == "last_layer":
            self.raw_weights.data[-1] = 5.0
        elif init_strategy == "uniform":
            self.raw_weights.data.zero_()
        elif init_strategy == "edges":
            for i in range(num_layers):
                if i < 4 or i >= num_layers - 4:
                    self.raw_weights.data[i] = 2.0
        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")

    def forward(self, layer_tensors: List[torch.Tensor]) -> torch.Tensor:
        if len(layer_tensors) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layer tensors, got {len(layer_tensors)}"
            )
        w = torch.softmax(self.raw_weights, dim=0)
        stacked = torch.stack(layer_tensors, dim=0)
        return (stacked * w.view(-1, 1, 1, 1)).sum(dim=0)

    @torch.no_grad()
    def importance(self) -> torch.Tensor:
        return torch.softmax(self.raw_weights, dim=0)


def _layer_tensor_from_module_output(out):
    return out[0] if isinstance(out, tuple) else out


def _encoder_out_proj_module(encoder: nn.Module):
    for mod in (encoder, getattr(encoder, "encoder", None)):
        if mod is None:
            continue
        op = getattr(mod, "out_proj", None)
        if isinstance(op, nn.Module):
            return op
    return None


def _conformer_hook_to_fusion_bdt(encoder: nn.Module, hook_out: torch.Tensor, encoded_ref: torch.Tensor):
    """Match NeMo interctc: layer (B,T,d_in) -> optional out_proj -> (B,d_out,T)."""
    t = hook_out
    if t.dim() != 3:
        return t
    op = _encoder_out_proj_module(encoder)
    target_feats = encoded_ref.shape[1]
    if op is not None:
        inf = int(getattr(op, "in_features", 0) or 0)
        if inf > 0:
            if t.shape[2] == inf:
                pass
            elif t.shape[1] == inf:
                t = t.transpose(1, 2)
        t = op(t)
        out = t.transpose(1, 2)
    else:
        if t.shape[1] == target_feats:
            out = t
        elif t.shape[2] == target_feats:
            out = t.transpose(1, 2)
        else:
            out = t.transpose(1, 2)
    if out.shape[1] != target_feats and out.shape[2] == target_feats:
        out = out.transpose(1, 2)
    return out


def _encoder_forward_with_layer_hooks(encoder, audio_signal, length, num_layers):
    layer_outs: List[torch.Tensor] = []
    hooks: List = []

    def _hook(_m, _inp, out):
        layer_outs.append(_layer_tensor_from_module_output(out))

    try:
        for lyr in encoder.layers:
            hooks.append(lyr.register_forward_hook(_hook))
        encoded, encoded_len = encoder(audio_signal=audio_signal, length=length)
    finally:
        for h in hooks:
            h.remove()

    if len(layer_outs) != num_layers:
        raise RuntimeError(
            f"Layer fusion hooks captured {len(layer_outs)} outputs, expected {num_layers}."
        )
    return encoded, encoded_len, layer_outs


def attach_layer_fusion_to_model(
    model: nn.Module,
    *,
    init_strategy: InitStrategy = "last_layer",
    module_name: str = "layer_fusion",
) -> LearnableLayerFusion:
    if not hasattr(model, "encoder") or not hasattr(model, "preprocessor"):
        raise TypeError("model must expose .encoder and .preprocessor (NeMo ASR).")

    enc = model.encoder
    if not hasattr(enc, "layers"):
        raise TypeError("model.encoder must have .layers (Conformer-style encoder).")

    num_layers = len(enc.layers)
    existing = getattr(model, module_name, None)
    if isinstance(existing, LearnableLayerFusion):
        if existing.num_layers != num_layers:
            raise ValueError(
                f"Checkpoint fusion layers {existing.num_layers} != encoder {num_layers}"
            )
        fusion = existing
    else:
        fusion = LearnableLayerFusion(num_layers, init_strategy=init_strategy)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        fusion = fusion.to(device=device, dtype=dtype)
        model.add_module(module_name, fusion)

    def _fused_forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = (
            processed_signal is not None and processed_signal_length is not None
        )
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments input_signal / input_signal_length are mutually "
                "exclusive with processed_signal / processed_signal_length."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(
                input_spec=processed_signal, length=processed_signal_length
            )

        fusion_mod: LearnableLayerFusion = getattr(self, module_name)

        try:
            encoded, encoded_len, raw_layer_tensors = _encoder_forward_with_layer_hooks(
                self.encoder,
                processed_signal,
                processed_signal_length,
                fusion_mod.num_layers,
            )
            layer_tensors = [
                _conformer_hook_to_fusion_bdt(self.encoder, t, encoded)
                for t in raw_layer_tensors
            ]
            fused = fusion_mod(layer_tensors)
        finally:
            rr = getattr(self.encoder, "reset_registry", None)
            if callable(rr):
                rr()

        del encoded
        return fused, encoded_len

    model.forward = types.MethodType(_fused_forward, model)
    return fusion


# ----- training configuration -----
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

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = 2
NUM_EPOCHS = 3
MAX_DURATION_SEC = 20.0
VAL_SPLIT = 0.2
LEARNING_RATE = 0.001
VAL_CHECK_INTERVAL = 0.5
FUSION_INIT = "last_layer"

os.makedirs(MANIFEST_DIR, exist_ok=True)

print(f"torch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"numpy: {np.__version__}")

USE_EXISTING_MANIFESTS = os.environ.get("USE_EXISTING_MANIFESTS", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
TRAIN_MANIFEST = os.environ.get(
    "TRAIN_MANIFEST",
    "/kaggle/input/datasets/akarshks/tr-manifest/train_manifest.jsonl",
)
VAL_MANIFEST = os.environ.get(
    "VAL_MANIFEST",
    "/kaggle/input/datasets/akarshks/val-manifest/val_manifest.jsonl",
)

if USE_EXISTING_MANIFESTS:
    if not os.path.isfile(TRAIN_MANIFEST):
        raise FileNotFoundError(
            f"Train manifest not found: {TRAIN_MANIFEST}. "
            "Attach the dataset or set TRAIN_MANIFEST / USE_EXISTING_MANIFESTS=0."
        )
    if not os.path.isfile(VAL_MANIFEST):
        raise FileNotFoundError(
            f"Val manifest not found: {VAL_MANIFEST}. "
            "Attach the dataset or set VAL_MANIFEST / USE_EXISTING_MANIFESTS=0."
        )
    train_path = TRAIN_MANIFEST
    val_path = VAL_MANIFEST
    print("Using existing manifests:")
    print("  Train:", train_path)
    print("  Val  :", val_path)
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
    if len(df) == 0:
        raise RuntimeError(
            "No training rows after JSONL + audio index. Check ASR_DATA_DIR / TALKBANK paths."
        )

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
            # Full checkpoints are .nemo via SaveFullNemoAtSteps + final save_to (saves disk vs .ckpt + .nemo)
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
    """Calls model.save_to() at specific trainer.global_step values (optimizer steps)."""

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
print(f"Checkpoints dir (early): {ckpt_dir}")

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

print("Attaching learnable layer fusion...")
attach_layer_fusion_to_model(
    model, init_strategy=FUSION_INIT, module_name="layer_fusion"
)
print(f"Fusion over {model.layer_fusion.num_layers} encoder layers")

if "spec_augment" in cfg.model:
    model.spec_augmentation = model.from_config_dict(cfg.model.spec_augment)
    print("SpecAugment enabled")

print("Setting up data...")
cfg.model.train_ds = update_model_cfg(model.cfg.train_ds, cfg.model.train_ds)
model.setup_training_data(cfg.model.train_ds)
cfg.model.validation_ds = update_model_cfg(
    model.cfg.validation_ds, cfg.model.validation_ds
)
model.setup_multiple_validation_data(cfg.model.validation_ds)

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

# Fused encoder features change the input distribution to decoder+joint; train them too.
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
    print("Decoder + joint: trainable (TRAIN_DECODER_JOINT=0 to keep frozen)")

# Optimizer must see final requires_grad mask (NeMo builds param groups here).
model.setup_optimization(cfg.model.optim)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,}, Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")
print("Initial fusion softmax:", model.layer_fusion.importance().cpu().numpy())

_save_steps = _parse_nemo_save_steps()
if _save_steps:
    trainer.callbacks.append(SaveFullNemoAtSteps(ckpt_dir, _save_steps))
    print(
        f"Full-model .nemo snapshots at global_steps {_save_steps} (set NEMO_SAVE_AT_STEPS to change)"
    )

print("=" * 60)
print(
    "STAGE 1: encoder frozen + adapters + layer fusion"
    + (" + decoder + joint" if TRAIN_DECODER_JOINT else "")
)
print("=" * 60)
trainer.fit(model)

print("Saving...")
model.save_adapters(os.path.join(ckpt_dir, "adapter_final.pt"))
torch.save(
    {"raw_weights": model.layer_fusion.raw_weights.detach().cpu()},
    os.path.join(ckpt_dir, "layer_fusion_weights.pt"),
)
model.save_to(os.path.join(ckpt_dir, "model_stage1_fusion.nemo"))
print(
    f"Saved adapters, fusion weights, and final model_stage1_fusion.nemo under {ckpt_dir}"
)
print("Final fusion softmax:", model.layer_fusion.importance().cpu().numpy())
print("STAGE 1 COMPLETE")
'''

os.makedirs("/kaggle/working", exist_ok=True)
with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
    f.write(train_code)

print(f"Training script written to {TRAIN_SCRIPT}\n")
print(
    "Optional env (set BEFORE this cell; child inherits):\n"
    "  USE_EXISTING_MANIFESTS, TRAIN_MANIFEST, VAL_MANIFEST\n"
    "  NEMO_SAVE_AT_STEPS — mid-run full .nemo steps (default 6000; empty disables)\n"
    "  TRAIN_DECODER_JOINT — default 1: train decoder+joint with fused encoder; set 0 to freeze\n"
)

# =============================================================================
# PART C: FRESH SUBPROCESS (avoids stale notebook imports / numpy)
# =============================================================================
print("=" * 60)
print("Launching training in fresh subprocess")
print("=" * 60)

_compile = subprocess.run(
    [sys.executable, "-m", "py_compile", TRAIN_SCRIPT],
    cwd="/kaggle/working",
    capture_output=True,
    text=True,
)
if _compile.returncode != 0:
    print("Syntax error in generated training script (py_compile failed):\n")
    print(_compile.stderr or _compile.stdout)
    raise RuntimeError(
        "train script failed py_compile — fix the generator or paste errors above."
    )

# Propagate manifest mode into child (inherits os.environ by default)
result = subprocess.run(
    [sys.executable, TRAIN_SCRIPT],
    cwd="/kaggle/working",
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
    stdout=None,
    stderr=subprocess.STDOUT,
)

if result.returncode != 0:
    print(
        f"\nTraining subprocess exited with code {result.returncode}.\n"
        "Scroll up for the Python traceback from the child process.\n"
        "Do not use a hand-edited copy of train_code: broken pastes cause SyntaxError "
        "or wrong __init__ / __future__ / torch.__version__."
    )
    raise RuntimeError(f"Training failed with exit code {result.returncode}")

print("\nTraining completed successfully.")
print("Check /kaggle/working/nemo_layer_fusion_stage1/ for outputs.")
