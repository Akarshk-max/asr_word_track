# =============================================================================
# Stage-2 Kaggle cell — same structure as your Stage-1 notebook:
#   Part A — pip
#   Part B — write /kaggle/working/train_nemo_adapter_stage2.py
#   Part C — subprocess (fresh interpreter)
#
# Continues from your Stage-1 .nemo: same waveform aug, SpecAugment, noise dirs,
# ChainedLinearAdapter registration + chain rewiring, ReinforceDecoderJointTrainMode,
# discriminative LR (adds encoder group). Unfreezes encoder backbone + adapters + joint +
# decoder embed + last LSTM. No validation. Default saves end of Lightning epochs 3,4,5
# (human epochs 4,5,6) + adapter_epoch*.pt — override with SAVE_EPOCHS (0-based indices).
# =============================================================================

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
# PART B: STAGE-2 TRAINING SCRIPT (self-contained; matches your Stage-1 aug + hooks)
# =============================================================================
TRAIN_SCRIPT = "/kaggle/working/train_nemo_adapter_stage2.py"

train_code = r'''from __future__ import annotations

import logging
import math
import os
import random
import types
import warnings

warnings.filterwarnings("ignore", category=Warning, module="numba")
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil, LinearAdapter
from nemo.utils import logging as nemo_logging

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

# ==========================================
# Whisper labels: Stage 1 normalizes when building manifests; use the same JSONL here.
# ==========================================
print(
    "Use the same train_manifest.jsonl as Stage 1 (Whisper-normalized labels if Stage 1 used it)."
)

# ==========================================
# Adapter chain + ChainedLinearAdapter (same as your Stage-1 script)
# ==========================================
class AdapterChainState:
    def __init__(self):
        self.prev_bottleneck = None
        self.layer_counter = 0

    def reset(self):
        self.prev_bottleneck = None
        self.layer_counter = 0

    def update(self, bottleneck):
        self.prev_bottleneck = bottleneck
        self.layer_counter += 1


class ChainedLinearAdapter(nn.Module, AdapterModuleUtil):
    def __init__(
        self,
        in_features,
        dim,
        activation="gelu",
        norm_position="post",
        dropout=0.1,
        is_first=False,
        chain_state_ref=None,
        adapter_strategy=None,
    ):
        nn.Module.__init__(self)
        assert norm_position == "post", "ChainedLinearAdapter only implements post-norm (NeMo parity)"
        self.down = nn.Linear(in_features, dim)
        self.up = nn.Linear(dim, in_features)
        self.norm = nn.LayerNorm(in_features)
        self.dropout_layer = nn.Dropout(dropout)
        self.is_first = is_first
        self.chain_state_ref = chain_state_ref
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "swish":
            self.act = nn.SiLU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.GELU()
        if not is_first:
            self.chain_proj = nn.Linear(dim, dim, bias=False)
            nn.init.orthogonal_(self.chain_proj.weight)
            with torch.no_grad():
                self.chain_proj.weight.mul_(0.9)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.setup_adapter_strategy(adapter_strategy)

    def forward(self, x):
        prev_bottleneck = None
        if self.chain_state_ref is not None and not self.is_first:
            prev_bottleneck = self.chain_state_ref.prev_bottleneck
        h = self.down(x)
        if prev_bottleneck is not None:
            if prev_bottleneck.shape[1] != h.shape[1]:
                prev_bottleneck = F.interpolate(
                    prev_bottleneck.transpose(1, 2),
                    size=h.shape[1],
                    mode="nearest",
                ).transpose(1, 2)
            h = h + self.chain_proj(prev_bottleneck)
        bottleneck = self.act(h)
        if self.chain_state_ref is not None:
            self.chain_state_ref.update(bottleneck)
        h_up = self.up(bottleneck)
        h_norm = self.norm(h_up)
        h_drop = self.dropout_layer(h_norm)
        return h_drop


# ==========================================
# Waveform augmentation (same hyperparameters as your Stage-1 attach block)
# ==========================================
class WaveformAugmentor(nn.Module):
    def __init__(
        self,
        speed_min: float = 0.90,
        speed_max: float = 1.10,
        pitch_min: float = -1.0,
        pitch_max: float = 3.0,
        sample_rate: int = 16000,
        speed_prob: float = 0.5,
        pitch_prob: float = 0.5,
        classroom_noise_prob: float = 0.5,
        classroom_snr_min: float = 5.0,
        classroom_snr_max: float = 10.0,
        noise_file_paths: list | None = None,
        gain_prob: float = 0.2,
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

    def _load_noise_segment(self, num_samples: int, device: torch.device, dtype: torch.dtype):
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
    def forward(self, audio_signal: torch.Tensor, signal_length: torch.Tensor):
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
            signal_length = (signal_length.float() / speed_factor).long().clamp(min=1, max=new_T)

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

        if self.noise_file_paths and torch.rand(1).item() < self.classroom_noise_prob:
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
            audio_signal = audio_signal * (10 ** (gain_db / 20.0))

        return audio_signal, signal_length


def _collect_classroom_noise_files(directories: list) -> list:
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    out = []
    for d in directories:
        if not d or not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    out.append(os.path.join(root, f))
    return out


def _parse_noise_dirs() -> list:
    raw = os.environ.get("CLASSROOM_NOISE_DIRS", "").strip()
    if raw:
        return [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
    d1 = os.environ.get("CLASSROOM_NOISE_DIR_1", "").strip()
    d2 = os.environ.get("CLASSROOM_NOISE_DIR_2", "").strip()
    if d1 or d2:
        return [d for d in (d1, d2) if d]
    return [
        "/kaggle/input/datasets/akarshkumarshukla/noise-1",
        "/kaggle/input/datasets/akarshkumarshukla/asr-data/noise_part_0",
    ]


# ==========================================
# Callbacks (same as Stage 1 + epoch indices 0-based like your SaveSelectedEpochs)
# ==========================================
class SaveSelectedEpochs(Callback):
    def __init__(self, directory: str, save_epochs: list):
        super().__init__()
        self.directory = directory
        self.save_epochs = list(save_epochs)

    def on_train_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        epoch = trainer.current_epoch
        if epoch not in self.save_epochs:
            print(f"[SaveSelectedEpochs] epoch {epoch} -> SKIPPED (not in save list)", flush=True)
            return
        human = epoch + 1
        nemo_path = os.path.join(self.directory, f"model_epoch{human}.nemo")
        adapter_path = os.path.join(self.directory, f"adapter_epoch{human}.pt")
        pl_module.save_to(nemo_path)
        try:
            pl_module.save_adapters(adapter_path)
        except Exception as e:
            print(f"[SaveSelectedEpochs] WARNING: save_adapters failed: {e}", flush=True)
        print(
            f"[SaveSelectedEpochs] lightning_epoch={epoch} (human {human}) -> {nemo_path}, {adapter_path}",
            flush=True,
        )


class ReinforceDecoderJointTrainMode(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        dec = getattr(pl_module, "decoder", None)
        if dec is not None:
            dec.train()
        j = getattr(pl_module, "joint", None)
        if j is not None:
            j.train()
        wa = getattr(pl_module, "waveform_augmentor", None)
        if wa is not None:
            wa.train()
        sa = getattr(pl_module, "spec_augmentation", None)
        if sa is not None:
            sa.train()


class ChainedAdapterTrainDebugCallback(Callback):
    def __init__(self, num_steps: int = 0):
        super().__init__()
        self.num_steps = int(num_steps)
        self._batch_ctr = 0
        self._handles = []
        self._adapters_ordered = []
        self._fwd_records = []
        self._grad_lines = []

    @staticmethod
    def _gmax(t):
        if t is None or t.grad is None:
            return float("nan")
        return t.grad.detach().float().abs().max().item()

    def on_train_start(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0 or self.num_steps <= 0:
            return
        self._adapters_ordered = [
            (n, m)
            for n, m in pl_module.named_modules()
            if isinstance(m, ChainedLinearAdapter)
        ]
        self._adapters_ordered.sort(key=lambda x: x[0])

        def _make_hook(idx, name):
            def _hook(mod, inp, out):
                x = inp[0]
                self._fwd_records.append(
                    {
                        "idx": idx,
                        "name": name,
                        "in": tuple(x.shape),
                        "out": tuple(out.shape) if torch.is_tensor(out) else None,
                        "is_first": mod.is_first,
                    }
                )

            return _hook

        for i, (name, mod) in enumerate(self._adapters_ordered):
            self._handles.append(mod.register_forward_hook(_make_hook(i, name)))
        print(
            f"[ChainedAdapterTrainDebug] hooks on {len(self._adapters_ordered)} adapters; "
            f"first {self.num_steps} batches",
            flush=True,
        )

    def on_train_end(self, trainer, pl_module):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if getattr(trainer, "global_rank", 0) != 0 or self._batch_ctr >= self.num_steps:
            return
        self._fwd_records.clear()

    def on_after_backward(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0 or self._batch_ctr >= self.num_steps:
            return
        lines = []
        for i, (name, mod) in enumerate(self._adapters_ordered):
            gd = self._gmax(mod.down.weight)
            gu = self._gmax(mod.up.weight)
            cp = getattr(mod, "chain_proj", None)
            gc = self._gmax(cp.weight) if cp is not None else float("nan")
            lines.append(f"    [{i:02d}] down={gd:.3e} up={gu:.3e} chain={gc:.3e}  {name}")
        self._grad_lines = lines

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if getattr(trainer, "global_rank", 0) != 0 or self._batch_ctr >= self.num_steps:
            return
        recs = sorted(self._fwd_records, key=lambda r: r["idx"])
        print(f"\n{'=' * 80}", flush=True)
        print(
            f"[ChainedAdapterTrainDebug] batch={self._batch_ctr} epoch={trainer.current_epoch}",
            flush=True,
        )
        if isinstance(outputs, dict):
            for k in ("loss", "train_loss"):
                if k in outputs and outputs[k] is not None and torch.is_tensor(outputs[k]):
                    print(f"  {k}={outputs[k].detach().float().item():.6f}", flush=True)
                    break
        for r in recs:
            print(
                f"    [{r['idx']:02d}] first={r['is_first']} in={r['in']} out={r['out']}  {r['name']}",
                flush=True,
            )
        for line in self._grad_lines:
            print(line, flush=True)
        self._grad_lines = []
        print(f"{'=' * 80}\n", flush=True)
        self._batch_ctr += 1


def rewire_adapter_chain_state(model: ASRModel) -> AdapterChainState:
    state = AdapterChainState()
    model.adapter_chain_state = state
    n = 0
    for mod in model.modules():
        if isinstance(mod, ChainedLinearAdapter):
            mod.chain_state_ref = state
            n += 1
    print(f"[stage2] Rewired chain_state_ref on {n} ChainedLinearAdapter module(s)", flush=True)
    return state


def spec_augment_cfg():
    return OmegaConf.create(
        {
            "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
            "freq_masks": 2,
            "freq_width": 27,
            "time_masks": 10,
            "time_width": 0.05,
        }
    )


def patch_augmented_forward(model: ASRModel, waveform_aug: nn.Module) -> None:
    _orig = model.forward.__func__

    def _augmented_forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        if hasattr(self, "adapter_chain_state"):
            self.adapter_chain_state.reset()
        if self.training and input_signal is not None and input_signal_length is not None:
            input_signal, input_signal_length = self.waveform_augmentor(
                input_signal, input_signal_length
            )
        return _orig(
            self,
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
        )

    model.add_module("waveform_augmentor", waveform_aug)
    model.forward = types.MethodType(_augmented_forward, model)


def _unfreeze_last_lstm_in_module(container):
    if container is None:
        return False
    unfroze = False
    for _, mod in container.named_modules():
        if isinstance(mod, nn.LSTM) and mod.num_layers >= 1:
            last_i = mod.num_layers - 1
            for pname, param in mod.named_parameters():
                if f"_l{last_i}" in pname:
                    param.requires_grad = True
                    unfroze = True
    return unfroze


def prepare_stage2_trainable(model: ASRModel, adapter_full_name: str) -> None:
    model.set_enabled_adapters(enabled=False)
    model.set_enabled_adapters(name=adapter_full_name, enabled=True)
    model.freeze()
    model.unfreeze_enabled_adapters()

    enc = getattr(model, "encoder", None)
    if enc is not None:
        for name, p in enc.named_parameters():
            nl = name.lower()
            if ".adapter" in nl or "adapter_layer" in nl:
                continue
            p.requires_grad = True

    try:
        if hasattr(model, "joint"):
            for p in model.joint.parameters():
                p.requires_grad = True
    except Exception as e:
        nemo_logging.warning(f"joint unfreeze: {e}")

    try:
        pred = model.decoder.prediction
        if hasattr(pred, "embed"):
            for p in pred.embed.parameters():
                p.requires_grad = True
    except Exception as e:
        nemo_logging.warning(f"decoder embed: {e}")

    try:
        pred = model.decoder.prediction
        if not _unfreeze_last_lstm_in_module(pred) and hasattr(pred, "dec_rnn"):
            _unfreeze_last_lstm_in_module(pred.dec_rnn)
    except Exception as e:
        nemo_logging.warning(f"decoder LSTM: {e}")

    if hasattr(model, "decoder"):
        model.decoder.train()
    if hasattr(model, "joint"):
        model.joint.train()
    if hasattr(model, "waveform_augmentor"):
        model.waveform_augmentor.train()
    if getattr(model, "spec_augmentation", None) is not None:
        model.spec_augmentation.train()

    if hasattr(model, "preprocessor"):
        for p in model.preprocessor.parameters():
            p.requires_grad = False

    n_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"Stage2 trainable: {n_t:,} / {n_all:,} ({100 * n_t / max(1, n_all):.2f}%)", flush=True)


def build_discriminative_configure_optimizers(model: ASRModel, num_epochs: int, warmup_ratio: float):
    lr_enc = float(os.environ.get("LR_ENCODER", "5e-6"))
    lr_ad = float(os.environ.get("LR_ADAPTER", "5e-4"))
    lr_j = float(os.environ.get("LR_JOINT", "1e-4"))
    lr_d = float(os.environ.get("LR_DECODER", "5e-5"))
    lr_o = float(os.environ.get("LR_OTHER", "1e-4"))
    wd = float(os.environ.get("WEIGHT_DECAY", "0.01"))

    adapter_p, enc_p, joint_p, dec_p, other_p = [], [], [], [], []

    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        nl = pname.lower()
        if ".adapter" in nl or "adapter_layer" in nl or "chain_proj" in nl:
            adapter_p.append(param)
        elif "joint" in nl:
            joint_p.append(param)
        elif "decoder" in nl or "prediction" in nl or "embed" in nl:
            dec_p.append(param)
        elif "encoder" in nl:
            enc_p.append(param)
        else:
            other_p.append(param)

    groups = []
    if enc_p:
        groups.append({"params": enc_p, "lr": lr_enc})
    if adapter_p:
        groups.append({"params": adapter_p, "lr": lr_ad})
    if joint_p:
        groups.append({"params": joint_p, "lr": lr_j})
    if dec_p:
        groups.append({"params": dec_p, "lr": lr_d})
    if other_p:
        groups.append({"params": other_p, "lr": lr_o})
    if not groups:
        raise RuntimeError("No trainable parameters for optimizer.")

    for i, g in enumerate(groups):
        n = sum(p.numel() for p in g["params"])
        print(f"Param group {i}: {n:,} params, lr={g['lr']}", flush=True)

    try:
        num_batches = len(model._train_dl)
    except Exception:
        num_batches = 1000
    total_steps = max(1, num_epochs * num_batches)
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=wd)

    def _lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    def _configure_optimizers(_self):
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }

    return types.MethodType(_configure_optimizers, model)


# ==========================================
# Main
# ==========================================
INIT_NEMO = os.environ.get(
    "INIT_NEMO",
    "/kaggle/input/models/akarshkumarshukla/chained-adapters-model/pytorch/default/1/model_epoch4.nemo",
)
TRAIN_MANIFEST = os.environ.get(
    "TRAIN_MANIFEST",
    "/kaggle/input/datasets/akarshkumarshukla/earlier-manifest/train_manifest.jsonl",
).strip()
SAVE_DIR = os.environ.get("SAVE_DIR", "/kaggle/working/nemo_adapter_stage2")
ADAPTER_FULL_NAME = os.environ.get("ADAPTER_FULL_NAME", "encoder:asr_children_adapter").strip()

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "6"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.15"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "1.0"))
PRECISION = os.environ.get("PRECISION", "bf16-mixed")

# SAVE_EPOCHS: Lightning 0-based indices to save at end of epoch (default 3,4,5 -> human 4,5,6)
_se = os.environ.get("SAVE_EPOCHS", "").strip()
if _se:
    SAVE_EPOCHS = [int(x.strip()) for x in _se.split(",") if x.strip().isdigit()]
else:
    SAVE_EPOCHS = [3, 4, 5]

if not os.path.isfile(INIT_NEMO):
    raise FileNotFoundError(f"INIT_NEMO not found: {INIT_NEMO}")
if not os.path.isfile(TRAIN_MANIFEST):
    raise FileNotFoundError(f"TRAIN_MANIFEST not found: {TRAIN_MANIFEST}")

os.makedirs(SAVE_DIR, exist_ok=True)
ckpt_dir = os.path.join(SAVE_DIR, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

CLASSROOM_NOISE_DIRS = _parse_noise_dirs()
CLASSROOM_NOISE_FILES = _collect_classroom_noise_files(CLASSROOM_NOISE_DIRS)
print(
    f"Classroom noise: {len(CLASSROOM_NOISE_FILES)} clips from {len(CLASSROOM_NOISE_DIRS)} folder(s)"
)
for _dir in CLASSROOM_NOISE_DIRS:
    print(f"  - {_dir}  (exists={os.path.isdir(_dir)})")

LinearAdapter.register(ChainedLinearAdapter)

trainer = pl.Trainer(
    devices=1,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    precision=PRECISION if torch.cuda.is_available() else 32,
    max_epochs=NUM_EPOCHS,
    limit_val_batches=0,
    num_sanity_val_steps=0,
    enable_progress_bar=True,
    log_every_n_steps=2000,
    gradient_clip_val=GRAD_CLIP,
    logger=False,
    enable_checkpointing=False,
)

trainer.callbacks.append(SaveSelectedEpochs(ckpt_dir, save_epochs=SAVE_EPOCHS))
trainer.callbacks.append(ReinforceDecoderJointTrainMode())
_dbg = int(os.environ.get("TRAIN_DEBUG_STEPS", "0"))
if _dbg > 0:
    trainer.callbacks.append(ChainedAdapterTrainDebugCallback(num_steps=_dbg))

print(
    f"Saving .nemo + adapter at end of Lightning epochs {SAVE_EPOCHS} "
    f"(human epochs {[e + 1 for e in SAVE_EPOCHS]}) -> {ckpt_dir}"
)
print("Validation: DISABLED")

map_loc = "cuda" if torch.cuda.is_available() else "cpu"
model = ASRModel.restore_from(INIT_NEMO, map_location=map_loc, trainer=trainer)

with open_dict(model.cfg):
    if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
        model.cfg.decoding.greedy.use_cuda_graph_decoder = False
if hasattr(model, "change_decoding_strategy"):
    model.change_decoding_strategy(model.cfg.decoding)

rewire_adapter_chain_state(model)

sr = int(getattr(model.preprocessor, "_sample_rate", 16000))
if not CLASSROOM_NOISE_FILES:
    print("WARNING: No classroom noise files; noise mixing disabled.", flush=True)

waveform_aug = WaveformAugmentor(
    speed_min=0.90,
    speed_max=1.10,
    pitch_min=-1.0,
    pitch_max=3.0,
    sample_rate=sr,
    speed_prob=0.5,
    pitch_prob=0.5,
    classroom_noise_prob=0.5,
    classroom_snr_min=5.0,
    classroom_snr_max=10.0,
    noise_file_paths=CLASSROOM_NOISE_FILES,
    gain_prob=0.2,
    gain_db_min=-6.0,
    gain_db_max=6.0,
).to(next(model.parameters()).device)

patch_augmented_forward(model, waveform_aug)
model.spec_augmentation = model.from_config_dict(spec_augment_cfg())
print("SpecAugment: freq_masks=2 w=27, time_masks=10 w=0.05", flush=True)

prepare_stage2_trainable(model, ADAPTER_FULL_NAME)

with open_dict(model.cfg.train_ds):
    model.cfg.train_ds.manifest_filepath = TRAIN_MANIFEST
    model.cfg.train_ds.batch_size = BATCH_SIZE
    model.cfg.train_ds.num_workers = NUM_WORKERS
    model.cfg.train_ds.shuffle = True

model.setup_training_data(model.cfg.train_ds)
model.configure_optimizers = build_discriminative_configure_optimizers(
    model, NUM_EPOCHS, WARMUP_RATIO
)

print("=" * 60)
print("STAGE-2 TRAINING")
print(f"  INIT_NEMO:     {INIT_NEMO}")
print(f"  Epochs:        {NUM_EPOCHS}")
print(f"  Batch size:    {BATCH_SIZE}")
print(f"  LR groups:     encoder={os.environ.get('LR_ENCODER', '5e-6')} adapter={os.environ.get('LR_ADAPTER', '5e-4')} "
      f"joint={os.environ.get('LR_JOINT', '1e-4')} decoder={os.environ.get('LR_DECODER', '5e-5')}")
print(f"  Aug: speed 0.90-1.10, pitch -1..+3 sem, noise SNR 5-10 dB (p=0.5), gain +/-6 dB (p=0.2)")
print("=" * 60)

trainer.fit(model)

model.save_adapters(os.path.join(ckpt_dir, "adapter_final.pt"))
model.save_to(os.path.join(ckpt_dir, "model_final.nemo"))
print(f"Saved adapter_final.pt and model_final.nemo under {ckpt_dir}")
print("STAGE-2 COMPLETE.")
'''

os.makedirs("/kaggle/working", exist_ok=True)
with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
    f.write(train_code)

print(f"Training script written to {TRAIN_SCRIPT}\n")
print(
    "Env: INIT_NEMO, TRAIN_MANIFEST, SAVE_DIR, BATCH_SIZE, SAVE_EPOCHS (0-based, default 3,4,5), "
    "LR_ENCODER, LR_ADAPTER, LR_JOINT, LR_DECODER, CLASSROOM_NOISE_DIRS, TRAIN_DEBUG_STEPS\n"
)

# =============================================================================
# PART C: fresh subprocess
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
    print(_compile.stderr or _compile.stdout)
    raise RuntimeError("py_compile failed for stage-2 script")

result = subprocess.run(
    [sys.executable, TRAIN_SCRIPT],
    cwd="/kaggle/working",
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
    stdout=None,
    stderr=subprocess.STDOUT,
)
if result.returncode != 0:
    raise RuntimeError(f"Training failed with exit code {result.returncode}")

print("\nTraining completed successfully.")
