# ==========================================
# CELL 1: INSTALL + TRAIN (SINGLE CELL)
# ==========================================
# This does EVERYTHING in one cell:
#   1. Installs packages
#   2. Writes training script (train_vnext) to disk
#   3. Runs training in a SUBPROCESS (fresh Python = no stale imports)

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

print("🧹 Step 1: Cleaning stale packages...")
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
     "webdataset>=0.2.80", "braceexpand>=0.1.7", "editdistance>=0.6.0",
     "whisper-normalizer")

print("🔥 Step 5: NeMo...")
_pip("install", "--no-cache-dir", "nemo_toolkit[asr]>=2.5.0")

print("🔒 Step 6: Re-pin numpy...")
_pip("install", "--no-cache-dir", "--force-reinstall",
     "numpy>=2.1,<2.3", "scipy>=1.14,<1.16")

print("✅ Installation complete!\n")

# ======================
# PART B: WRITE TRAINING SCRIPT TO DISK
# ======================
TRAIN_SCRIPT = "/kaggle/working/train_nemo_vnext.py"

train_code = r'''
# train_nemo_vnext.py
# Parakeet-TDT-1.1B child-speech adaptation — vNext
# LoRA + chained bottleneck adapters + LSTM memory + noise curriculum

from __future__ import annotations

import json
import logging
import math
import os
import re
import types
import warnings
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Type

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
from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil, LinearAdapter
from nemo.core.classes import adapter_mixins
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf, open_dict

# =============================================================================
# CONFIG
# =============================================================================
MODEL_ID = os.environ.get("MODEL_ID", "nvidia/parakeet-tdt-1.1b")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "7"))
MAX_DURATION_SEC = 20.0
EARLY_ADAPTER_LAYERS = 8

_lora_layers_env = os.environ.get("LORA_LAYERS", "").strip()
LORA_LAYERS = [int(x.strip()) for x in _lora_layers_env.split(",") if x.strip()] if _lora_layers_env else list(range(0, 21))
LORA_LAYER_SET = frozenset(LORA_LAYERS)
EARLY_BOTTLENECK_DIM = 256
LATE_BOTTLENECK_DIM = 256
MEMORY_DIM = int(os.environ.get("MEMORY_DIM", "128"))
LSTM_OUT_FUSE_WEIGHT = float(os.environ.get("LSTM_OUT_FUSE_WEIGHT", "0.7"))
ADAPTER_UP_FUSE_WEIGHT = float(os.environ.get("ADAPTER_UP_FUSE_WEIGHT", "0.3"))

LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = float(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))

NOISE_CURRICULUM = os.environ.get("NOISE_CURRICULUM", "1").strip().lower() in ("1", "true", "yes")
NOISE_AUG_PROB = float(os.environ.get("NOISE_AUG_PROB", "0.6"))
NOISE_SNR_MIN = float(os.environ.get("NOISE_SNR_MIN", "1.0"))
NOISE_SNR_MAX = float(os.environ.get("NOISE_SNR_MAX", "7.0"))
NOISE_CURR_E1_3 = (0.3, 5.0, 20.0)
NOISE_CURR_E4_5 = (0.4, 5.0, 15.0)
NOISE_CURR_E6_7 = (0.5, 3.0, 10.0)

LR_LORA = float(os.environ.get("LR_LORA", "3e-4"))
LR_ADAPTER = float(os.environ.get("LR_ADAPTER", "5e-4"))
LR_MEMORY = float(os.environ.get("LR_MEMORY", "3e-4"))
LR_JOINT = float(os.environ.get("LR_JOINT", "1e-4"))
LR_DECODER = float(os.environ.get("LR_DECODER", "5e-5"))

PHASE_LORA_LAST_EPOCH = int(os.environ.get("PHASE_LORA_LAST_EPOCH", "5"))
PHASE_JOINT_LAST_EPOCH = int(os.environ.get("PHASE_JOINT_LAST_EPOCH", "2"))
PHASE_DECODER_LAST_EPOCH = int(os.environ.get("PHASE_DECODER_LAST_EPOCH", "3"))
PHASE_ADAPTER_MEMORY_LAST_EPOCH = int(os.environ.get("PHASE_ADAPTER_MEMORY_LAST_EPOCH", "6"))

SAVE_DIR = os.environ.get("SAVE_DIR", "/kaggle/working/nemo_adapter_parakeet_vnext")
MANIFEST_DIR = os.path.join(SAVE_DIR, "manifests")
SAVE_EPOCHS = [4, 5, 6]
LEARNING_RATE_FALLBACK = 5e-4

try:
    from whisper_normalizer.english import EnglishTextNormalizer
    _whisper_normalizer = EnglishTextNormalizer()
    NORMALIZE_TEXT = True
    print("Whisper text normalizer loaded.")
except ImportError:
    NORMALIZE_TEXT = False
    _whisper_normalizer = None
    print("WARNING: whisper_normalizer not installed — using basic lowercase.")

# =============================================================================
# LoRA
# =============================================================================
LORA_LINEAR_TARGET_MODULES = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "self_attn.linear_q", "self_attn.linear_k", "self_attn.linear_v", "self_attn.linear_out",
    "ffn1.linear1", "ffn1.linear2", "ffn2.linear1", "ffn2.linear2",
    "feed_forward.linear1", "feed_forward.linear2",
]
LORA_CONV_TARGET_MODULES = ["conv_module.pointwise_conv1", "conv_module.pointwise_conv2"]


def _lora_path_matches_target(lfn, target):
    return all(part in lfn for part in target.lower().split("."))

def _linear_matches_lora_targets(lfn):
    return "adapter" not in lfn and any(_lora_path_matches_target(lfn, t) for t in LORA_LINEAR_TARGET_MODULES)

def _conv_matches_lora_targets(lfn):
    return "adapter" not in lfn and any(_lora_path_matches_target(lfn, t) for t in LORA_CONV_TARGET_MODULES)


class LinearWithLoRA(nn.Module):
    def __init__(self, linear, r, alpha, dropout=0.0):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False
        self.r = max(0, int(r))
        self.scale = (alpha / self.r) if self.r > 0 else 0.0
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.r, linear.in_features))
            self.lora_B = nn.Parameter(torch.zeros(linear.out_features, self.r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
        self.lora_dropout = nn.Dropout(dropout) if dropout and self.r > 0 else nn.Identity()

    def forward(self, x):
        out = self.linear(x)
        if self.r <= 0 or self.lora_A is None:
            return out
        return out + (self.scale * (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T)).to(out.dtype)


class Conv1dPointwiseLoRA(nn.Module):
    def __init__(self, conv, r, alpha, dropout=0.0):
        super().__init__()
        if conv.kernel_size[0] != 1 or conv.groups != 1:
            raise ValueError("Conv1dPointwiseLoRA: requires kernel_size=1, groups=1")
        self.conv = conv
        for p in self.conv.parameters():
            p.requires_grad = False
        self.r = max(0, int(r))
        self.scale = (alpha / self.r) if self.r > 0 else 0.0
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.r, conv.in_channels))
            self.lora_B = nn.Parameter(torch.zeros(conv.out_channels, self.r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
        self.lora_dropout = nn.Dropout(dropout) if dropout and self.r > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        if self.r <= 0 or self.lora_A is None:
            return out
        z = self.lora_dropout(x)
        b, c, t = z.shape
        delta = (z.transpose(1, 2).reshape(b * t, c) @ self.lora_A.T @ self.lora_B.T)
        return out + (self.scale * delta.reshape(b, t, out.shape[1]).transpose(1, 2).contiguous()).to(out.dtype)


def inject_lora_into_encoder_layers(encoder, layer_indices, r, alpha, dropout):
    paths = []
    if not hasattr(encoder, "layers"):
        print("WARNING: encoder has no .layers — skipping LoRA")
        return paths, 0
    seen = sorted({int(i) for i in layer_indices if 0 <= int(i) < len(encoder.layers)})
    for li in seen:
        layer = encoder.layers[li]
        for full_name, mod in list(layer.named_modules()):
            lfn = full_name.lower()
            parts = full_name.split(".")
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            child = parts[-1]
            if isinstance(getattr(parent, child), (LinearWithLoRA, Conv1dPointwiseLoRA)):
                continue
            if isinstance(mod, nn.Linear) and _linear_matches_lora_targets(lfn):
                setattr(parent, child, LinearWithLoRA(mod, r, alpha, dropout))
                paths.append(f"encoder.layers.{li}.{full_name}")
            elif isinstance(mod, nn.Conv1d) and mod.kernel_size[0] == 1 and _conv_matches_lora_targets(lfn):
                setattr(parent, child, Conv1dPointwiseLoRA(mod, r, alpha, dropout))
                paths.append(f"encoder.layers.{li}.{full_name}")
    return paths, len(paths)

# =============================================================================
# Stream state
# =============================================================================
class AdapterStreamState:
    def __init__(self, memory_dim):
        self.memory_dim = memory_dim
        self.prev_bottleneck = None
        self.prev_hidden = None
        self.prev_cell = None

    def reset(self):
        self.prev_bottleneck = None
        self.prev_hidden = None
        self.prev_cell = None

    def get_memory_states(self, B, T, device, dtype):
        if self.prev_hidden is None:
            self.prev_hidden = torch.zeros(B, T, self.memory_dim, device=device, dtype=dtype)
            self.prev_cell = torch.zeros(B, T, self.memory_dim, device=device, dtype=dtype)
        return self.prev_hidden, self.prev_cell

    def update_after_adapter(self, b_n, h, c):
        self.prev_bottleneck = b_n
        self.prev_hidden = h
        self.prev_cell = c


class AdapterMemoryCell(nn.Module):
    def __init__(self, bottleneck_dim, memory_dim, output_dim):
        super().__init__()
        self.memory_dim = memory_dim
        self.gates = nn.Linear(bottleneck_dim + memory_dim, 4 * memory_dim)
        self.mem_output_proj = nn.Linear(memory_dim, output_dim, bias=False)
        with torch.no_grad():
            self.gates.bias[memory_dim : 2 * memory_dim].fill_(2.0)
        nn.init.normal_(self.mem_output_proj.weight, std=0.002)

    def forward(self, bottleneck, prev_h, prev_c):
        gates = self.gates(torch.cat([bottleneck, prev_h], dim=-1))
        ig, fg, cc, og = gates.chunk(4, dim=-1)
        new_c = torch.sigmoid(fg) * prev_c + torch.sigmoid(ig) * torch.tanh(cc)
        new_h = torch.sigmoid(og) * torch.tanh(new_c)
        return self.mem_output_proj(new_h), new_h, new_c


def _match_seq_len(t, target_len):
    if t.shape[1] == target_len:
        return t
    return F.interpolate(t.transpose(1, 2), size=target_len, mode="nearest").transpose(1, 2)

# =============================================================================
# Adapters
# =============================================================================
class EarlyTwoBottleneckAdapter(nn.Module, AdapterModuleUtil):
    def __init__(self, in_features=1024, b1=256, b2=256, memory_dim=128,
                 dropout=0.1, is_first=False, stream_state_ref=None, adapter_strategy=None):
        nn.Module.__init__(self)
        self.stream_state_ref = stream_state_ref
        self.is_first = is_first
        self.down1 = nn.Linear(in_features, b1)
        self.down2 = nn.Linear(b1, b2)
        self.act = nn.GELU()
        self.norm_b = nn.LayerNorm(b2)
        self.up = nn.Linear(b2, in_features)
        self.norm_out = nn.LayerNorm(in_features)
        self.dropout_layer = nn.Dropout(dropout)
        if not is_first:
            self.chain_proj = nn.Linear(b2, b2, bias=False)
            nn.init.orthogonal_(self.chain_proj.weight)
            with torch.no_grad():
                self.chain_proj.weight.mul_(0.9)
        self.memory_cell = AdapterMemoryCell(b2, memory_dim, in_features)
        nn.init.normal_(self.up.weight, std=0.002)
        nn.init.zeros_(self.up.bias)
        self.setup_adapter_strategy(adapter_strategy)

    def forward(self, x):
        B, T, _ = x.shape
        h2_pre = self.down2(self.down1(x))
        if not self.is_first and self.stream_state_ref is not None:
            pb = self.stream_state_ref.prev_bottleneck
            if pb is not None:
                h2_pre = h2_pre + self.chain_proj(_match_seq_len(pb, T))
        b_n = self.norm_b(self.act(h2_pre))
        ph, pc = self.stream_state_ref.get_memory_states(B, T, x.device, x.dtype)
        mem_out, nh, nc = self.memory_cell(b_n, _match_seq_len(ph, T), _match_seq_len(pc, T))
        out = self.dropout_layer(self.norm_out(LSTM_OUT_FUSE_WEIGHT * mem_out + ADAPTER_UP_FUSE_WEIGHT * self.up(b_n)))
        if self.stream_state_ref is not None:
            self.stream_state_ref.update_after_adapter(b_n, nh, nc)
        return out


class LateChainedAdapter(nn.Module, AdapterModuleUtil):
    def __init__(self, in_features=1024, dim=256, memory_dim=128,
                 dropout=0.1, is_first=False, stream_state_ref=None, adapter_strategy=None):
        nn.Module.__init__(self)
        self.stream_state_ref = stream_state_ref
        self.is_first = is_first
        self.down = nn.Linear(in_features, dim)
        self.act = nn.GELU()
        self.norm_b = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, in_features)
        self.norm_out = nn.LayerNorm(in_features)
        self.dropout_layer = nn.Dropout(dropout)
        if not is_first:
            self.chain_proj = nn.Linear(dim, dim, bias=False)
            nn.init.orthogonal_(self.chain_proj.weight)
            with torch.no_grad():
                self.chain_proj.weight.mul_(0.9)
        self.memory_cell = AdapterMemoryCell(dim, memory_dim, in_features)
        nn.init.normal_(self.up.weight, std=0.002)
        nn.init.zeros_(self.up.bias)
        self.setup_adapter_strategy(adapter_strategy)

    def forward(self, x):
        B, T, _ = x.shape
        h_pre = self.down(x)
        if not self.is_first and self.stream_state_ref is not None:
            pb = self.stream_state_ref.prev_bottleneck
            if pb is not None:
                h_pre = h_pre + self.chain_proj(_match_seq_len(pb, T))
        b_n = self.norm_b(self.act(h_pre))
        ph, pc = self.stream_state_ref.get_memory_states(B, T, x.device, x.dtype)
        mem_out, nh, nc = self.memory_cell(b_n, _match_seq_len(ph, T), _match_seq_len(pc, T))
        out = self.dropout_layer(self.norm_out(LSTM_OUT_FUSE_WEIGHT * mem_out + ADAPTER_UP_FUSE_WEIGHT * self.up(b_n)))
        if self.stream_state_ref is not None:
            self.stream_state_ref.update_after_adapter(b_n, nh, nc)
        return out

# =============================================================================
# Waveform augmentation
# =============================================================================
class WaveformAugmentor(nn.Module):
    def __init__(self, speed_min=0.90, speed_max=1.10, pitch_min=-1.0, pitch_max=3.0,
                 sample_rate=16000, speed_prob=0.5, pitch_prob=0.5,
                 external_noise_prob=0.6, external_snr_min=1.0, external_snr_max=10.0,
                 noise_file_paths=None, gain_prob=0.3, gain_db_min=-6.0, gain_db_max=6.0):
        super().__init__()
        self.speed_min = speed_min; self.speed_max = speed_max
        self.pitch_min = pitch_min; self.pitch_max = pitch_max
        self.sample_rate = sample_rate
        self.speed_prob = speed_prob; self.pitch_prob = pitch_prob
        self.external_noise_prob = external_noise_prob
        self.external_snr_min = external_snr_min; self.external_snr_max = external_snr_max
        self.noise_file_paths = list(noise_file_paths or [])
        self.gain_prob = gain_prob; self.gain_db_min = gain_db_min; self.gain_db_max = gain_db_max

    @staticmethod
    def _mix_at_snr(speech, noise, snr_db):
        eps = 1e-8
        alpha = torch.sqrt(speech.pow(2).mean().clamp_min(eps) /
                           (10 ** (snr_db / 10.0) * noise.pow(2).mean().clamp_min(eps) + eps))
        return speech + alpha * noise

    def _load_noise_segment(self, num_samples, device, dtype):
        if not self.noise_file_paths:
            return None
        path = self.noise_file_paths[random.randint(0, len(self.noise_file_paths) - 1)]
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
            wav = wav.repeat((num_samples + n - 1) // n)[:num_samples]
        else:
            start = random.randint(0, n - num_samples)
            wav = wav[start : start + num_samples]
        return wav.to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def forward(self, audio_signal, signal_length):
        if not self.training:
            return audio_signal, signal_length
        B, T = audio_signal.shape
        if torch.rand(1).item() < self.speed_prob:
            sf = self.speed_min + torch.rand(1).item() * (self.speed_max - self.speed_min)
            new_T = max(1, int(T / sf))
            audio_signal = F.interpolate(audio_signal.unsqueeze(1), size=new_T, mode="linear", align_corners=False).squeeze(1)
            signal_length = (signal_length.float() / sf).long().clamp(min=1, max=new_T)
        if torch.rand(1).item() < self.pitch_prob:
            n_steps = self.pitch_min + torch.rand(1).item() * (self.pitch_max - self.pitch_min)
            try:
                audio_signal = torchaudio.functional.pitch_shift(audio_signal, self.sample_rate, n_steps)
            except Exception:
                pass
        if self.noise_file_paths and torch.rand(1).item() < self.external_noise_prob:
            for b in range(B):
                L = int(min(signal_length[b].item(), audio_signal.shape[1]))
                if L < 1:
                    continue
                seg = self._load_noise_segment(L, audio_signal.device, audio_signal.dtype)
                if seg is None:
                    continue
                snr = self.external_snr_min + torch.rand(1).item() * (self.external_snr_max - self.external_snr_min)
                audio_signal[b, :L] = self._mix_at_snr(audio_signal[b, :L].float(), seg, snr).to(audio_signal.dtype)
        if torch.rand(1).item() < self.gain_prob:
            gain_db = self.gain_db_min + torch.rand(1).item() * (self.gain_db_max - self.gain_db_min)
            audio_signal = audio_signal * (10 ** (gain_db / 20.0))
        return audio_signal, signal_length


def _collect_noise_files(dir_list):
    exts = (".wav", ".flac", ".mp3", ".ogg")
    out = []
    for d in dir_list:
        d = d.strip()
        if not d or not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(exts):
                    out.append(os.path.join(root, f))
    return out

# =============================================================================
# Callbacks
# =============================================================================
class SaveSelectedEpochs(Callback):
    def __init__(self, directory, save_epochs):
        super().__init__()
        self.directory = directory
        self.save_epochs = save_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        ep = trainer.current_epoch
        if ep not in self.save_epochs:
            return
        os.makedirs(self.directory, exist_ok=True)
        human = ep + 1
        nemo_path = os.path.join(self.directory, f"model_epoch{human}.nemo")
        try:
            pl_module.save_to(nemo_path)
            print(f"[SaveSelectedEpochs] saved {nemo_path}", flush=True)
        except Exception as e:
            print(f"[SaveSelectedEpochs] save_to warning: {e}", flush=True)
        try:
            pl_module.save_adapters(os.path.join(self.directory, f"adapter_epoch{human}.pt"))
        except Exception as e:
            print(f"[SaveSelectedEpochs] save_adapters warning: {e}", flush=True)


class ReinforceDecoderJointTrainMode(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        for attr in ("decoder", "joint", "waveform_augmentor", "spec_augmentation"):
            m = getattr(pl_module, attr, None)
            if m is not None:
                m.train()


class PhasedTrainabilityCallback(Callback):
    def __init__(self, adapter_short, last_lora, last_joint, last_decoder, last_adapter_mem):
        super().__init__()
        self.adapter_short = adapter_short.lower()
        self.last_lora = last_lora; self.last_joint = last_joint
        self.last_decoder = last_decoder; self.last_adapter_mem = last_adapter_mem

    def on_train_epoch_start(self, trainer, pl_module):
        ep = int(trainer.current_epoch)
        lora_on = ep <= self.last_lora
        joint_on = ep <= self.last_joint
        dec_on = ep <= self.last_decoder
        am_on = ep <= self.last_adapter_mem
        for n, p in pl_module.named_parameters():
            nl = n.lower()
            if "lora_a" in nl or "lora_b" in nl:
                p.requires_grad = lora_on
            elif "memory_cell" in nl:
                p.requires_grad = am_on
            elif "joint" in nl:
                p.requires_grad = joint_on
            elif "decoder" in nl or "prediction" in nl:
                p.requires_grad = dec_on
            elif self.adapter_short in nl or "asr_children" in nl:
                p.requires_grad = am_on
        if getattr(trainer, "global_rank", 0) == 0:
            print(f"[PhasedTrainability] human_epoch={ep+1}  LoRA={lora_on}  joint={joint_on}  decoder={dec_on}  adapter+mem={am_on}", flush=True)


class NoiseCurriculumCallback(Callback):
    def __init__(self, enabled=True):
        super().__init__()
        self.enabled = enabled

    def on_train_epoch_start(self, trainer, pl_module):
        if not self.enabled:
            return
        wa = getattr(pl_module, "waveform_augmentor", None)
        if wa is None:
            return
        human = int(trainer.current_epoch) + 1
        if human <= 3:
            p, lo, hi, label = *NOISE_CURR_E1_3, "1-3"
        elif human <= 5:
            p, lo, hi, label = *NOISE_CURR_E4_5, "4-5"
        else:
            p, lo, hi, label = *NOISE_CURR_E6_7, "6+"
        wa.external_noise_prob = p; wa.external_snr_min = lo; wa.external_snr_max = hi
        if getattr(trainer, "global_rank", 0) == 0:
            print(f"[NoiseCurriculum] human_epoch={human}  phase={label}  noise_prob={p}  SNR=[{lo},{hi}]", flush=True)

# =============================================================================
# Helpers
# =============================================================================
def _layer_idx_from_adapter_path(mod_name):
    m = re.search(r"layers\.(\d+)", mod_name)
    return int(m.group(1)) if m else -1

def _encoder_target_key(model_cfg):
    enc = model_cfg.encoder
    if "_target_" in enc:
        return "_target_"
    if "target" in enc:
        return "target"
    raise AttributeError("encoder target key missing")

def update_model_cfg(orig_cfg, new_cfg):
    with open_dict(orig_cfg):
        for k, v in new_cfg.items():
            orig_cfg[k] = v
    return orig_cfg

def _collect_trainable_groups(model, adapter_short):
    lora_p, adapter_p, mem_p, joint_p, dec_p = [], [], [], [], []
    seen = set()
    def add_once(lst, p):
        if id(p) not in seen:
            seen.add(id(p)); lst.append(p)
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nl = n.lower()
        if "lora_a" in nl or "lora_b" in nl:
            add_once(lora_p, p)
        elif "memory_cell" in nl:
            add_once(mem_p, p)
        elif "joint" in nl:
            add_once(joint_p, p)
        elif "decoder" in nl or "prediction" in nl:
            add_once(dec_p, p)
        elif adapter_short.lower() in nl or "asr_children" in nl:
            add_once(adapter_p, p)
    grouped = {id(x) for xs in (lora_p, adapter_p, mem_p, joint_p, dec_p) for x in xs}
    for n, p in model.named_parameters():
        if not p.requires_grad or id(p) in grouped:
            continue
        print(f"WARNING: unclassified trainable param -> adapter group: {n}")
        add_once(adapter_p, p)
    return lora_p, adapter_p, mem_p, joint_p, dec_p

def _build_configure_optimizers(model, lora_p, adapter_p, mem_p, joint_p, dec_p, warmup_ratio, num_epochs, steps_per_epoch):
    groups = []
    if lora_p:    groups.append({"params": lora_p,    "lr": LR_LORA,    "name": "lora"})
    if adapter_p: groups.append({"params": adapter_p, "lr": LR_ADAPTER, "name": "adapter"})
    if mem_p:     groups.append({"params": mem_p,     "lr": LR_MEMORY,  "name": "memory"})
    if joint_p:   groups.append({"params": joint_p,   "lr": LR_JOINT,   "name": "joint"})
    if dec_p:     groups.append({"params": dec_p,     "lr": LR_DECODER, "name": "decoder"})
    if not groups:
        raise RuntimeError("No trainable parameters for optimizer.")
    total_steps = max(1, num_epochs * max(1, steps_per_epoch))
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    def configure_optimizers(self):
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        self._optimizer = opt
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}
    return configure_optimizers, groups

# =============================================================================
# Paths & manifest
# =============================================================================
ASR_DATA_DIR = os.environ.get("ASR_DATA_DIR", "/kaggle/input/datasets/akarshkumarshukla/asr-data")
ASR_JSONL = os.environ.get("ASR_JSONL", os.path.join(ASR_DATA_DIR, "train_word_transcripts.jsonl"))
TALKBANK_DIR = os.environ.get("TALKBANK_DIR", "/kaggle/input/datasets/akarshkumarshukla/talk-bank-data")
TALKBANK_JSON = os.environ.get("TALKBANK_JSON", "/kaggle/input/datasets/akarshkumarshukla/talkbank-transcript-full/train_word_transcripts.jsonl")

NOISE_DIRS = []
_env = os.environ.get("CLASSROOM_NOISE_DIRS", "").strip()
if _env:
    NOISE_DIRS.extend(x.strip() for x in _env.split(",") if x.strip())
else:
    NOISE_DIRS.extend([
        "/kaggle/input/datasets/akarshkumarshukla/noise/noise_part_1",
        "/kaggle/input/datasets/akarshkumarshukla/asr-data/noise_part_0",
    ])

NOISE_FILES = _collect_noise_files(NOISE_DIRS)
print(f"Noise files: {len(NOISE_FILES)} from {len(NOISE_DIRS)} dir(s)")
if not NOISE_FILES:
    print("WARNING: No noise files — external noise augmentation disabled.")

os.makedirs(MANIFEST_DIR, exist_ok=True)

USE_EXISTING_MANIFESTS = os.environ.get("USE_EXISTING_MANIFESTS", "1").strip().lower() in ("1", "true", "yes")
TRAIN_MANIFEST = os.environ.get("TRAIN_MANIFEST", "/kaggle/input/datasets/akarshkumarshukla/noise/train_manifest.jsonl")

if USE_EXISTING_MANIFESTS:
    if not os.path.isfile(TRAIN_MANIFEST):
        raise FileNotFoundError(f"Train manifest not found: {TRAIN_MANIFEST}")
    df = pd.read_json(TRAIN_MANIFEST, lines=True)
    train_path = TRAIN_MANIFEST
    print(f"Using existing manifest: {train_path}  ({len(df):,} rows)")
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

    print("Indexing audio...")
    audio_index = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        for fut in (ex.submit(_index_directory, ASR_DATA_DIR), ex.submit(_index_directory, TALKBANK_DIR)):
            audio_index.update(fut.result())
    print(f"Indexed {len(audio_index):,} flac files")

    records = []
    for filepath in [ASR_JSONL] + ([TALKBANK_JSON] if os.path.exists(TALKBANK_JSON) else []):
        if not os.path.exists(filepath):
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("orthographic_text", "").strip()
                    text = _whisper_normalizer(text) if NORMALIZE_TEXT else text.lower()
                    dur = float(data.get("audio_duration_sec", 0.0))
                    path = audio_index.get(os.path.basename(data.get("audio_path", "")))
                    if path and text and (0.1 < dur <= MAX_DURATION_SEC):
                        records.append({"audio_filepath": path, "duration": dur, "text": text})
                except Exception:
                    continue

    df = pd.DataFrame(records)
    if len(df) == 0:
        raise RuntimeError("No training rows found.")
    train_path = os.path.join(MANIFEST_DIR, "train_manifest.jsonl")
    df.to_json(train_path, orient="records", lines=True)
    print(f"Wrote train manifest: {train_path}  ({len(df):,} rows)")

n_train = len(pd.read_json(train_path, lines=True))
steps_per_epoch = max(1, (n_train + BATCH_SIZE - 1) // BATCH_SIZE)
print(f"Train samples: {n_train:,}  |  Approx steps/epoch: {steps_per_epoch}  |  batch_size={BATCH_SIZE}")
print(f"torch {torch.__version__}  CUDA {torch.version.cuda}  numpy {np.__version__}")

cfg = OmegaConf.create({
    "model": {
        "pretrained_model": MODEL_ID, "log_prediction": False,
        "spec_augment": {
            "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
            "freq_masks": 2, "freq_width": 27, "time_masks": 10, "time_width": 0.05,
        },
        "adapter": {
            "adapter_name": "asr_children_adapter", "adapter_module_name": "encoder",
            "adapter_type": "linear",
            "linear": {
                "_target_": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
                "in_features": 1024, "dim": LATE_BOTTLENECK_DIM,
                "activation": "gelu", "norm_position": "post", "dropout": 0.1,
            },
        },
        "train_ds": {
            "manifest_filepath": train_path, "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS, "pin_memory": False, "use_lhotse": False,
            "channel_selector": "average", "is_tarred": False, "shuffle": True,
        },
        "optim": {
            "name": "adamw", "lr": LEARNING_RATE_FALLBACK, "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "sched": {"name": "CosineAnnealing", "warmup_ratio": 0.15, "min_lr": 1e-6},
        },
    },
    "exp_manager": {
        "exp_dir": SAVE_DIR, "name": "ParakeetVNext",
        "create_tensorboard_logger": True, "create_checkpoint_callback": False,
        "resume_if_exists": False, "resume_ignore_no_checkpoint": True,
    },
})

trainer = pl.Trainer(
    devices=1,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    precision="bf16-mixed" if torch.cuda.is_available() else 32,
    max_epochs=NUM_EPOCHS, limit_val_batches=0, num_sanity_val_steps=0,
    enable_progress_bar=True, log_every_n_steps=100, gradient_clip_val=1.0,
    logger=False, enable_checkpointing=False,
    callbacks=[ReinforceDecoderJointTrainMode()],
)

exp_log_dir = exp_manager(trainer, cfg.exp_manager)
ckpt_dir = os.path.join(exp_log_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
trainer.callbacks.append(SaveSelectedEpochs(ckpt_dir, save_epochs=SAVE_EPOCHS))
print(f"Experiment dir: {exp_log_dir}  |  Validation disabled (train only).")

print(f"Loading {MODEL_ID}...")
model_cfg = ASRModel.from_pretrained(MODEL_ID, return_config=True)
enc_key = _encoder_target_key(model_cfg)
with open_dict(model_cfg):
    meta = adapter_mixins.get_registered_adapter(model_cfg.encoder[enc_key])
    if meta is not None:
        model_cfg.encoder[enc_key] = meta.adapter_class_path
        print(f"Patched encoder: {meta.adapter_class_path}")

model = ASRModel.from_pretrained(MODEL_ID, override_config_path=model_cfg, trainer=trainer)
with open_dict(model.cfg):
    if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
        model.cfg.decoding.greedy.use_cuda_graph_decoder = False
if hasattr(model, "change_decoding_strategy"):
    model.change_decoding_strategy(model.cfg.decoding)

device = next(model.parameters()).device
_nprob, _nsnr0, _nsnr1 = NOISE_CURR_E1_3 if NOISE_CURRICULUM else (NOISE_AUG_PROB, NOISE_SNR_MIN, NOISE_SNR_MAX)
waveform_aug = WaveformAugmentor(
    speed_min=0.85, speed_max=1.15, pitch_min=-2.0, pitch_max=2.0, sample_rate=16000,
    speed_prob=0.5, pitch_prob=0.5,
    external_noise_prob=_nprob, external_snr_min=_nsnr0, external_snr_max=_nsnr1,
    noise_file_paths=NOISE_FILES if NOISE_FILES else None,
    gain_prob=0.3, gain_db_min=-8.0, gain_db_max=8.0,
).to(device)
model.waveform_augmentor = waveform_aug

stream_state = AdapterStreamState(memory_dim=MEMORY_DIM)
model.adapter_stream_state = stream_state
_orig_forward = model.forward.__func__

def _wrapped_forward(self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None):
    if hasattr(self, "adapter_stream_state"):
        self.adapter_stream_state.reset()
    if self.training and input_signal is not None and input_signal_length is not None:
        input_signal, input_signal_length = self.waveform_augmentor(input_signal, input_signal_length)
    return _orig_forward(self, input_signal=input_signal, input_signal_length=input_signal_length,
                         processed_signal=processed_signal, processed_signal_length=processed_signal_length)

model.forward = types.MethodType(_wrapped_forward, model)
model.spec_augmentation = model.from_config_dict(cfg.model.spec_augment)
cfg.model.train_ds = update_model_cfg(model.cfg.train_ds, cfg.model.train_ds)
model.setup_training_data(cfg.model.train_ds)

adapter_full_name = "encoder:asr_children_adapter"
adapter_short = "asr_children_adapter"
model.add_adapter(name=adapter_full_name, cfg=cfg.model.adapter.linear)
model.set_enabled_adapters(enabled=False)
model.set_enabled_adapters(name=adapter_full_name, enabled=True)
model.freeze()
model.unfreeze_enabled_adapters()

adapter_module_list = sorted(
    [(n, m) for n, m in model.named_modules() if isinstance(m, LinearAdapter) and adapter_short in n],
    key=lambda x: _layer_idx_from_adapter_path(x[0]),
)

print("Encoder layer map:")
for mod_name, _ in adapter_module_list:
    li = _layer_idx_from_adapter_path(mod_name)
    kind = ("early_two_bottleneck" if li < EARLY_ADAPTER_LAYERS else "late_single_bottleneck") + ("+chain" if li > 0 else "")
    print(f"  layer {li:2d}: LoRA={li in LORA_LAYER_SET}  adapter={kind}")

for mod_name, _old in adapter_module_list:
    li = _layer_idx_from_adapter_path(mod_name)
    parts = mod_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    child = parts[-1]
    kw = dict(in_features=1024, memory_dim=MEMORY_DIM, dropout=0.1, is_first=(li == 0), stream_state_ref=stream_state, adapter_strategy=None)
    new_mod = (EarlyTwoBottleneckAdapter(b1=EARLY_BOTTLENECK_DIM, b2=EARLY_BOTTLENECK_DIM, **kw)
               if li < EARLY_ADAPTER_LAYERS else LateChainedAdapter(dim=LATE_BOTTLENECK_DIM, **kw)).to(device)
    setattr(parent, child, new_mod)
    if isinstance(parent, nn.ModuleDict):
        parent[child] = new_mod
    if len(parts) >= 2 and parts[-2] == "adapter_layer":
        lm = model
        for p in parts[:-2]:
            lm = getattr(lm, p)
        if hasattr(lm, "adapter_layer") and adapter_short in lm.adapter_layer:
            lm.adapter_layer[adapter_short] = new_mod
    new_mod.setup_adapter_strategy(None)

LinearAdapter.register(EarlyTwoBottleneckAdapter)
LinearAdapter.register(LateChainedAdapter)

lora_paths, n_lora = inject_lora_into_encoder_layers(model.encoder, LORA_LAYERS, LORA_R, LORA_ALPHA, LORA_DROPOUT)
print(f"LoRA injected: {n_lora} modules  (r={LORA_R} alpha={LORA_ALPHA})")

for m in model.modules():
    la, lb = getattr(m, "lora_A", None), getattr(m, "lora_B", None)
    if la is not None: la.requires_grad = True
    if lb is not None: lb.requires_grad = True

for attr in ("joint", "decoder"):
    mod = getattr(model, attr, None)
    if mod is not None:
        for p in mod.parameters():
            p.requires_grad = True
        print(f"TRAINABLE {attr}: {sum(p.numel() for p in mod.parameters() if p.requires_grad):,} params")

for m in (model.decoder, model.joint, model.waveform_augmentor, model.spec_augmentation):
    if m is not None:
        m.train()

lora_p, adapter_p, mem_p, joint_p, dec_p = _collect_trainable_groups(model, adapter_short)
co_fn, opt_groups = _build_configure_optimizers(
    model, lora_p, adapter_p, mem_p, joint_p, dec_p,
    warmup_ratio=0.15, num_epochs=NUM_EPOCHS, steps_per_epoch=steps_per_epoch,
)
model.configure_optimizers = types.MethodType(co_fn, model)

print("Optimizer groups:")
for g in opt_groups:
    print(f"  {g['name']}: lr={g['lr']}  params={sum(p.numel() for p in g['params']):,}")

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,}  Trainable: {trainable:,} ({100 * trainable / max(1, total):.2f}%)")

trainer.callbacks.append(PhasedTrainabilityCallback(
    adapter_short=adapter_short,
    last_lora=PHASE_LORA_LAST_EPOCH,
    last_joint=PHASE_JOINT_LAST_EPOCH,
    last_decoder=PHASE_DECODER_LAST_EPOCH,
    last_adapter_mem=PHASE_ADAPTER_MEMORY_LAST_EPOCH,
))
trainer.callbacks.append(NoiseCurriculumCallback(enabled=NOISE_CURRICULUM))

print("=" * 60)
print("STARTING TRAINING (vNext adapters + continuous chain)")
print("=" * 60)
trainer.fit(model)
print("Done. Weights saved at human epochs 5-7 under checkpoints/")
'''

with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
    f.write(train_code)

print(f"📝 Training script written to {TRAIN_SCRIPT}\n")

# ======================
# PART C: RUN IN SUBPROCESS (FRESH PYTHON)
# ======================
print("=" * 60)
print("🚀 LAUNCHING TRAINING IN FRESH SUBPROCESS")
print("=" * 60)
print("(Fresh Python process — no stale imports from pip installs above)\n")

result = subprocess.run(
    [sys.executable, TRAIN_SCRIPT],
    cwd="/kaggle/working",
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
    stdout=None,          # Inherit stdout → prints live to notebook
    stderr=subprocess.STDOUT,  # Merge stderr into stdout
)

if result.returncode != 0:
    print(f"\n❌ Training failed with exit code {result.returncode}")
else:
    print(f"\n✅ Training completed successfully!")
    print("📁 Check /kaggle/working/nemo_adapter_parakeet_vnext/ for outputs")
