"""
V3: Three-stream chained memory adapter (safe phased rollout).

Paste this entire file into ONE Kaggle notebook cell (or run as script).
"""

import os
import subprocess
import sys


def _pip(*args):
    cmd = [sys.executable, "-m", "pip", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("\n[pip] command failed:")
        print(" ".join(cmd))
        print("\n[pip stdout]\n" + (proc.stdout or "")[-4000:])
        print("\n[pip stderr]\n" + (proc.stderr or "")[-4000:])
        raise RuntimeError(f"pip failed with exit code {proc.returncode}")


def _install_numpy_scipy():
    try:
        _pip(
            "install",
            "--no-cache-dir",
            "--prefer-binary",
            "numpy>=2.1,<2.3",
            "scipy>=1.14,<1.16",
        )
    except Exception as e:
        print(f"[warn] pinned numpy/scipy failed: {e}")
        _pip("install", "--no-cache-dir", "--prefer-binary", "numpy>=2.1", "scipy")


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
_install_numpy_scipy()

print("Step 3: PyTorch (CUDA 12.6 wheels)...")
_pip(
    "install",
    "--no-cache-dir",
    "--index-url",
    "https://download.pytorch.org/whl/cu126",
    "torch>=2.9.0",
    "torchaudio",
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
print("Installation complete.\n")

TRAIN_SCRIPT = "/kaggle/working/train_v3_chained_memory.py"

train_code = r'''from __future__ import annotations

import json
import math
import os
import random
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from lightning.pytorch.callbacks import Callback
from torch.optim import AdamW as TorchAdamW
from torch.optim.lr_scheduler import LambdaLR

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil, LinearAdapter
from nemo.core.classes import adapter_mixins
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf, open_dict


if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

EXPECTED_ENCODER_LAYERS = int(os.environ.get("EXPECTED_ENCODER_LAYERS", "42"))
TRAIN_DEBUG_STEPS = int(os.environ.get("TRAIN_DEBUG_STEPS", "3"))


class AdapterMemoryCell(nn.Module):
    # Memory reads full frozen 1024-d encoder output h_l, not bottleneck.
    def __init__(self, in_features: int = 1024, memory_dim: int = 64):
        super().__init__()
        self.memory_dim = memory_dim
        self.gates = nn.Linear(in_features + memory_dim, 4 * memory_dim)
        self.mem_out = nn.Linear(memory_dim, in_features, bias=False)
        with torch.no_grad():
            # forget gate bias = +2.0
            self.gates.bias[memory_dim : 2 * memory_dim].fill_(2.0)
        nn.init.normal_(self.mem_out.weight, std=0.01)

    def forward(self, h_l: torch.Tensor, prev_h: torch.Tensor, prev_c: torch.Tensor):
        z = torch.cat([h_l, prev_h], dim=-1)
        i, f, g, o = self.gates(z).chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * prev_c + i * g
        h = o * torch.tanh(c)
        return self.mem_out(h), h, c


class AdapterStreamState:
    def __init__(self, memory_dim: int = 64):
        self.memory_dim = memory_dim
        self.prev_bottleneck: Optional[torch.Tensor] = None
        self.prev_hidden: Optional[torch.Tensor] = None
        self.prev_cell: Optional[torch.Tensor] = None
        self.layer_counter = 0
        self.forward_counter = 0

    def reset(self):
        self.prev_bottleneck = None
        self.prev_hidden = None
        self.prev_cell = None
        self.layer_counter = 0
        self.forward_counter += 1

    def get_states(self, b: int, t: int, device: torch.device, dtype: torch.dtype):
        if self.prev_hidden is None or self.prev_cell is None:
            self.prev_hidden = torch.zeros(b, t, self.memory_dim, device=device, dtype=dtype)
            self.prev_cell = torch.zeros(b, t, self.memory_dim, device=device, dtype=dtype)
        return self.prev_hidden, self.prev_cell

    def update(self, bottleneck: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        self.prev_bottleneck = bottleneck.detach()
        self.prev_hidden = hidden
        self.prev_cell = cell
        self.layer_counter += 1


class ChainedLinearAdapterWithMemory(nn.Module, AdapterModuleUtil):
    def __init__(
        self,
        in_features: int = 1024,
        dim: int = 128,
        memory_dim: int = 64,
        dropout: float = 0.1,
        activation: str = "gelu",
        is_first: bool = False,
        state: Optional[AdapterStreamState] = None,
        return_full_residual: bool = True,
        adapter_strategy=None,
    ):
        super().__init__()
        self.down = nn.Linear(in_features, dim)
        self.up = nn.Linear(dim, in_features)
        self.norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout)
        self.memory = AdapterMemoryCell(in_features=in_features, memory_dim=memory_dim)
        self.is_first = is_first
        self.state = state
        self.return_full_residual = return_full_residual
        self.adapter_strategy = adapter_strategy
        self.hook_calls = 0
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "swish":
            self.act = nn.SiLU()
        else:
            self.act = nn.GELU()
        if not is_first:
            self.chain_proj = nn.Linear(dim, dim, bias=False)
            nn.init.orthogonal_(self.chain_proj.weight)
            with torch.no_grad():
                self.chain_proj.weight.mul_(0.9)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def _resize_time(self, x: torch.Tensor, t: int) -> torch.Tensor:
        if x.shape[1] == t:
            return x
        return F.interpolate(x.transpose(1, 2), size=t, mode="nearest").transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.hook_calls += 1
        b, t, _ = x.shape
        down = self.down(x)
        if (not self.is_first) and self.state is not None and self.state.prev_bottleneck is not None:
            prev_b = self._resize_time(self.state.prev_bottleneck, t)
            down = down + self.chain_proj(prev_b)
        bottleneck = self.act(down)
        adapter_out = self.up(bottleneck)
        if self.state is not None:
            prev_h, prev_c = self.state.get_states(b, t, x.device, x.dtype)
            prev_h = self._resize_time(prev_h, t)
            prev_c = self._resize_time(prev_c, t)
            memory_out, new_h, new_c = self.memory(x, prev_h, prev_c)
            self.state.update(bottleneck, new_h, new_c)
        else:
            memory_out = torch.zeros_like(adapter_out)
        combined = self.dropout(self.norm(adapter_out + memory_out))
        # IMPORTANT: we probe LinearAdapter behavior; return either full residual or delta to match NeMo.
        return x + combined if self.return_full_residual else combined


try:
    LinearAdapter.register(ChainedLinearAdapterWithMemory)
except Exception as e:
    print(f"[warn] LinearAdapter.register failed: {e}")


class WaveformAugmentor(nn.Module):
    def __init__(
        self,
        speed_min=0.90,
        speed_max=1.10,
        pitch_min=-1.0,
        pitch_max=3.0,
        sample_rate=16000,
        speed_prob=0.5,
        pitch_prob=0.5,
        gain_prob=0.3,
        gain_db_min=-6.0,
        gain_db_max=6.0,
    ):
        super().__init__()
        self.speed_min, self.speed_max = speed_min, speed_max
        self.pitch_min, self.pitch_max = pitch_min, pitch_max
        self.sample_rate = sample_rate
        self.speed_prob, self.pitch_prob = speed_prob, pitch_prob
        self.gain_prob, self.gain_db_min, self.gain_db_max = gain_prob, gain_db_min, gain_db_max

    @torch.no_grad()
    def forward(self, audio_signal, signal_length):
        if not self.training:
            return audio_signal, signal_length
        b, t = audio_signal.shape
        if torch.rand(1).item() < self.speed_prob:
            s = self.speed_min + torch.rand(1).item() * (self.speed_max - self.speed_min)
            nt = max(1, int(t / s))
            audio_signal = F.interpolate(audio_signal.unsqueeze(1), size=nt, mode="linear", align_corners=False).squeeze(1)
            signal_length = (signal_length.float() / s).long().clamp(min=1, max=nt)
        if torch.rand(1).item() < self.pitch_prob:
            n_steps = self.pitch_min + torch.rand(1).item() * (self.pitch_max - self.pitch_min)
            try:
                audio_signal = torchaudio.functional.pitch_shift(audio_signal, self.sample_rate, n_steps)
            except Exception as e:
                print(f"[warn] pitch_shift failed: {e}")
        if torch.rand(1).item() < self.gain_prob:
            gdb = self.gain_db_min + torch.rand(1).item() * (self.gain_db_max - self.gain_db_min)
            audio_signal = audio_signal * (10 ** (gdb / 20.0))
        return audio_signal, signal_length


class SaveSelectedEpochs(Callback):
    def __init__(self, directory: str, save_epochs: list[int]):
        super().__init__()
        self.directory = directory
        self.save_epochs = save_epochs

    def _save_adapter_fallback(self, pl_module, adapter_path: str):
        trainable = {n for n, p in pl_module.named_parameters() if p.requires_grad}
        sd = {k: v for k, v in pl_module.state_dict().items() if k in trainable}
        torch.save(sd, adapter_path)
        print(f"[save] fallback adapter subset -> {adapter_path} ({len(sd)} tensors)")

    def on_train_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        ep = trainer.current_epoch
        if ep not in self.save_epochs:
            return
        os.makedirs(self.directory, exist_ok=True)
        nemo_path = os.path.join(self.directory, f"model_epoch{ep+1}.nemo")
        adapter_path = os.path.join(self.directory, f"adapter_epoch{ep+1}.pt")
        pl_module.save_to(nemo_path)
        try:
            pl_module.save_adapters(adapter_path)
        except Exception as e:
            print(f"[warn] save_adapters failed: {e}")
            self._save_adapter_fallback(pl_module, adapter_path)
        print(f"[save] {nemo_path} + {adapter_path}")


def _encoder_target_key(model_cfg):
    enc = model_cfg.encoder
    if "_target_" in enc:
        return "_target_"
    if "target" in enc:
        return "target"
    raise AttributeError("model_cfg.encoder has neither _target_ nor target")


def _find_adapter_hosts(model):
    hosts = []
    for n, m in model.named_modules():
        al = getattr(m, "adapter_layer", None)
        if isinstance(al, nn.ModuleDict) and len(al) > 0:
            hosts.append((n, m))
    return hosts


def _print_runtime_inspection(model):
    print("\n=== Runtime inspection ===")
    print("model class:", model.__class__.__name__)
    print("encoder class:", getattr(model, "encoder", None).__class__.__name__)
    hosts = _find_adapter_hosts(model)
    print("adapter host count:", len(hosts))
    if hosts:
        print("encoder layer class:", hosts[0][1].__class__.__name__)
        print("adapter keys on first host:", list(hosts[0][1].adapter_layer.keys()))
        first = hosts[0][1]
        print("dispatch via adapter_layer:", hasattr(first, "adapter_layer"))
        print("dispatch via forward_enabled_adapters:", hasattr(first, "forward_enabled_adapters"))
    adapter_named = [n for n, _ in model.named_modules() if "adapter" in n.lower()]
    print("modules containing 'adapter':", len(adapter_named))
    for n in adapter_named[:30]:
        print("  ", n)
    if len(adapter_named) > 30:
        print("  ... truncated ...")

    joint_like = [n for n, _ in model.named_modules() if "joint" in n.lower()]
    dec_embed = [n for n, m in model.named_modules() if "decoder" in n.lower() and isinstance(m, nn.Embedding)]
    dec_rnn = [n for n, m in model.named_modules() if "decoder" in n.lower() and isinstance(m, nn.LSTM)]
    print("joint-like modules:", joint_like[:10])
    print("decoder embedding modules:", dec_embed[:10])
    print("decoder LSTM modules:", dec_rnn[:10])


def _probe_linear_adapter_semantics(base_adapter: nn.Module, device: torch.device):
    base_adapter = base_adapter.to(device).eval()
    x = torch.randn(2, 7, 1024, device=device)
    with torch.no_grad():
        y = base_adapter(x)
    d_x = float((y - x).abs().mean().item())
    d_0 = float(y.abs().mean().item())
    full_residual = d_x < d_0
    print(f"adapter semantics probe: mean|y-x|={d_x:.6f}, mean|y|={d_0:.6f} -> return_full_residual={full_residual}")
    return full_residual


def _phase1_unit_test(device: torch.device):
    print("\n[phase 1] unit test on dummy tensors")
    state = AdapterStreamState(memory_dim=64)
    mod = ChainedLinearAdapterWithMemory(
        in_features=1024,
        dim=128,
        memory_dim=64,
        is_first=False,
        state=state,
        return_full_residual=True,
    ).to(device)
    x = torch.randn(2, 11, 1024, device=device)
    state.prev_bottleneck = torch.randn(2, 9, 128, device=device)
    state.prev_hidden = torch.randn(2, 9, 64, device=device)
    state.prev_cell = torch.randn(2, 9, 64, device=device)
    y = mod(x)
    print("  x:", tuple(x.shape))
    print("  prev_b:", tuple(state.prev_bottleneck.shape))
    print("  prev_h/c:", tuple(state.prev_hidden.shape), tuple(state.prev_cell.shape))
    print("  y:", tuple(y.shape))
    assert y.shape == x.shape, "Phase 1 failed: output shape mismatch"
    print("[phase 1] PASS")


def _replace_host_adapter(host, key: str, module: nn.Module):
    host.adapter_layer[key] = module


def _collect_param_names(module: nn.Module):
    return {n for n, _ in module.named_parameters()}


def _print_trainable_summary(model):
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    print("\nTrainable parameters:")
    for n, c in trainable[:120]:
        print(f"  {n} -> {c:,}")
    if len(trainable) > 120:
        print("  ... truncated ...")
    tot = sum(c for _, c in trainable)
    print(f"Total trainable: {tot:,}")


MODEL_ID = "nvidia/parakeet-tdt-1.1b"
SAVE_DIR = "/kaggle/working/nemo_adapter_1.1b"
MANIFEST_DIR = os.path.join(SAVE_DIR, "manifests")
os.makedirs(MANIFEST_DIR, exist_ok=True)

TRAIN_MANIFEST = os.environ.get("TRAIN_MANIFEST", "/kaggle/input/datasets/akarshks/train-meta/train_manifest.jsonl")
USE_EXISTING_MANIFESTS = os.environ.get("USE_EXISTING_MANIFESTS", "1").strip().lower() in ("1", "true", "yes")
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "6"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "48"))

if USE_EXISTING_MANIFESTS:
    if not os.path.isfile(TRAIN_MANIFEST):
        raise FileNotFoundError(f"Train manifest not found: {TRAIN_MANIFEST}")
    train_path = TRAIN_MANIFEST
else:
    raise RuntimeError("This safe script expects pre-built TRAIN_MANIFEST for now.")

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
                "num_workers": 2,
                "pin_memory": False,
                "use_lhotse": False,
                "channel_selector": "average",
                "is_tarred": False,
                "shuffle": True,
            },
        },
        "exp_manager": {
            "exp_dir": SAVE_DIR,
            "name": "ParakeetAdapter1.1B_ChainedMemory",
            "create_tensorboard_logger": False,
            "create_checkpoint_callback": False,
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
    limit_val_batches=0,
    num_sanity_val_steps=0,
    logger=False,
    enable_checkpointing=False,
    log_every_n_steps=500,
)
exp_log_dir = exp_manager(trainer, cfg.exp_manager)
ckpt_dir = os.path.join(exp_log_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
trainer.callbacks.append(SaveSelectedEpochs(ckpt_dir, save_epochs=[2, 4]))

model_cfg = ASRModel.from_pretrained(MODEL_ID, return_config=True)
enc_key = _encoder_target_key(model_cfg)
with open_dict(model_cfg):
    md = adapter_mixins.get_registered_adapter(model_cfg.encoder[enc_key])
    if md is not None:
        model_cfg.encoder[enc_key] = md.adapter_class_path
model = ASRModel.from_pretrained(MODEL_ID, override_config_path=model_cfg, trainer=trainer)
if "spec_augment" in cfg.model:
    model.spec_augmentation = model.from_config_dict(cfg.model.spec_augment)

waveform_aug = WaveformAugmentor()
model.add_module("waveform_augmentor", waveform_aug)
state = AdapterStreamState(memory_dim=64)
model._adapter_stream_state = state

base_forward = model.forward.__func__
def _forward_wrap(self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None):
    self._adapter_stream_state.reset()
    if self.training and input_signal is not None and input_signal_length is not None:
        input_signal, input_signal_length = self.waveform_augmentor(input_signal, input_signal_length)
    out = base_forward(
        self,
        input_signal=input_signal,
        input_signal_length=input_signal_length,
        processed_signal=processed_signal,
        processed_signal_length=processed_signal_length,
    )
    if self._adapter_stream_state.forward_counter <= TRAIN_DEBUG_STEPS:
        print(
            f"[debug] forward #{self._adapter_stream_state.forward_counter} "
            f"layer_counter={self._adapter_stream_state.layer_counter}"
        )
    return out
model.forward = types.MethodType(_forward_wrap, model)

model.setup_training_data(cfg.model.train_ds)
adapter_full_name = "encoder:asr_children_adapter"
model.add_adapter(name=adapter_full_name, cfg=cfg.model.adapter.linear)
model.set_enabled_adapters(enabled=False)
model.set_enabled_adapters(name=adapter_full_name, enabled=True)

_print_runtime_inspection(model)
_phase1_unit_test(model.device)

# Phase 2: replace only layer 0 and verify hook firing.
hosts = _find_adapter_hosts(model)
if not hosts:
    raise RuntimeError("No adapter hosts found; cannot proceed.")
adapter_key = list(hosts[0][1].adapter_layer.keys())[0]
print(f"[phase 2] adapter key = {adapter_key}")
return_full_residual = _probe_linear_adapter_semantics(hosts[0][1].adapter_layer[adapter_key], model.device)

phase2_calls = {"n": 0, "in": None, "out": None}
first_custom = ChainedLinearAdapterWithMemory(
    is_first=True,
    state=state,
    return_full_residual=return_full_residual,
).to(model.device)
def _h(_m, inputs, output):
    phase2_calls["n"] += 1
    phase2_calls["in"] = tuple(inputs[0].shape)
    phase2_calls["out"] = tuple(output.shape)
first_custom.register_forward_hook(_h)
_replace_host_adapter(hosts[0][1], adapter_key, first_custom)

try:
    model.eval()
    with torch.no_grad():
        _ = model(
            input_signal=torch.randn(1, 16000, device=model.device),
            input_signal_length=torch.tensor([16000], dtype=torch.long, device=model.device),
        )
except Exception as e:
    raise RuntimeError(f"[phase 2] failed dummy forward: {e}")
if phase2_calls["n"] == 0:
    raise RuntimeError("[phase 2] hook did not fire; custom adapter not used.")
print(f"[phase 2] PASS hook_calls={phase2_calls['n']} in={phase2_calls['in']} out={phase2_calls['out']}")

# Phase 3: replace all adapters.
print("[phase 3] replacing all encoder adapters")
for i, (_hn, host) in enumerate(hosts):
    mod = ChainedLinearAdapterWithMemory(
        is_first=(i == 0),
        state=state,
        return_full_residual=return_full_residual,
    ).to(model.device)
    _replace_host_adapter(host, adapter_key, mod)
print(f"[phase 3] replaced hosts: {len(hosts)}")
if len(hosts) != EXPECTED_ENCODER_LAYERS:
    raise RuntimeError(f"Expected {EXPECTED_ENCODER_LAYERS} adapter hosts, found {len(hosts)}")

# Phase 4: freezing / unfreezing.
model.freeze()
for n, p in model.named_parameters():
    if "adapter_layer" in n and adapter_key in n:
        p.requires_grad = True

joint_prefixes = [n for n, m in model.named_modules() if "joint" in n.lower() and any(True for _ in m.parameters(recurse=False))]
for n, p in model.named_parameters():
    if any(n.startswith(pref + ".") or n == pref for pref in joint_prefixes):
        p.requires_grad = True

dec_emb = [n for n, m in model.named_modules() if "decoder" in n.lower() and isinstance(m, nn.Embedding)]
dec_lstm = [n for n, m in model.named_modules() if "decoder" in n.lower() and isinstance(m, nn.LSTM)]
if dec_emb:
    pref = dec_emb[-1]
    for n, p in model.named_parameters():
        if n.startswith(pref + "."):
            p.requires_grad = True
if dec_lstm:
    pref = dec_lstm[-1]
    for n, p in model.named_parameters():
        if n.startswith(pref + "."):
            p.requires_grad = True

if hasattr(model, "decoder"):
    model.decoder.train()
if hasattr(model, "spec_augmentation"):
    model.spec_augmentation.train()
model.waveform_augmentor.train()

counts = {"adapters": 0, "memory": 0, "joint": 0, "decoder": 0, "total": 0}
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    c = p.numel()
    counts["total"] += c
    if "adapter_layer" in n and adapter_key in n:
        counts["adapters"] += c
        if ".memory." in n:
            counts["memory"] += c
    elif any(n.startswith(pref + ".") for pref in joint_prefixes):
        counts["joint"] += c
    elif any(n.startswith(pref + ".") for pref in dec_emb + dec_lstm):
        counts["decoder"] += c
print("[phase 4] trainable counts:", counts)
_print_trainable_summary(model)

# Phase 5: optimizer groups.
adapter_params, joint_params, decoder_params = [], [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if "adapter_layer" in n and adapter_key in n:
        adapter_params.append(p)
    elif any(n.startswith(pref + ".") for pref in joint_prefixes):
        joint_params.append(p)
    elif any(n.startswith(pref + ".") for pref in dec_emb + dec_lstm):
        decoder_params.append(p)
print(
    "[phase 5] optimizer groups:",
    f"adapters+memory={sum(p.numel() for p in adapter_params):,}",
    f"joint={sum(p.numel() for p in joint_params):,}",
    f"decoder={sum(p.numel() for p in decoder_params):,}",
)
if not adapter_params:
    raise RuntimeError("No adapter params in optimizer group.")

groups = [
    {"params": adapter_params, "lr": 5e-4},
    {"params": joint_params, "lr": 1e-4},
    {"params": decoder_params, "lr": 5e-5},
]
optimizer = TorchAdamW(groups, betas=(0.9, 0.999), weight_decay=0.01)
steps_per_epoch = max(1, len(model._train_dl))
total_steps = max(1, steps_per_epoch * trainer.max_epochs)
warmup_steps = int(0.15 * total_steps)
def _lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))
scheduler = LambdaLR(optimizer, _lr_lambda)
def _custom_config_optimizers(self_model):
    return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
model.configure_optimizers = types.MethodType(_custom_config_optimizers, model)

print("\nStarting training with three-stream chained memory adapters...")
model.train()
trainer.fit(model)

print("Saving final model...")
model.save_to(os.path.join(ckpt_dir, "model_final.nemo"))
try:
    model.save_adapters(os.path.join(ckpt_dir, "adapter_final.pt"))
except Exception as e:
    print(f"[warn] save_adapters final failed: {e}")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    sd = {k: v for k, v in model.state_dict().items() if k in trainable}
    torch.save(sd, os.path.join(ckpt_dir, "adapter_final.pt"))
print("Done:", ckpt_dir)
'''

os.makedirs("/kaggle/working", exist_ok=True)
with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
    f.write(train_code)

print(f"Training script written to {TRAIN_SCRIPT}\n")
print("Launching in fresh subprocess...")
_compile = subprocess.run(
    [sys.executable, "-m", "py_compile", TRAIN_SCRIPT],
    cwd="/kaggle/working",
    capture_output=True,
    text=True,
)
if _compile.returncode != 0:
    print(_compile.stderr or _compile.stdout)
    raise RuntimeError("Generated train script failed py_compile")

result = subprocess.run(
    [sys.executable, TRAIN_SCRIPT],
    cwd="/kaggle/working",
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
    stdout=None,
    stderr=subprocess.STDOUT,
)
if result.returncode != 0:
    raise RuntimeError(f"Training failed with exit code {result.returncode}")
print("Training completed successfully.")
