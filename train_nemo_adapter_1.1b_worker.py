from __future__ import annotations

import glob
import json
import logging
import math
import os
import types
import warnings
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List

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

# -----------------------------------------------------------------------------
# Whisper text normalizer
# -----------------------------------------------------------------------------
try:
    from whisper_normalizer.english import EnglishTextNormalizer

    _whisper_normalizer = EnglishTextNormalizer()
    NORMALIZE_TEXT = True
    print("Whisper text normalizer loaded — will normalize training labels")
except ImportError:
    NORMALIZE_TEXT = False
    _whisper_normalizer = None
    print("WARNING: whisper_normalizer not installed. Training labels will use basic lowercase only.")


# -----------------------------------------------------------------------------
# Adapter chain state
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Chained linear adapter (replaces NeMo LinearAdapter instances)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Waveform augmentation (same strategy as train_nemo_vnext / code A)
# -----------------------------------------------------------------------------
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
        external_noise_prob=0.6,
        external_snr_min=1.0,
        external_snr_max=7.0,
        noise_file_paths=None,
        gain_prob=0.3,
        gain_db_min=-6.0,
        gain_db_max=6.0,
    ):
        super().__init__()
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.sample_rate = sample_rate
        self.speed_prob = speed_prob
        self.pitch_prob = pitch_prob
        self.external_noise_prob = external_noise_prob
        self.external_snr_min = external_snr_min
        self.external_snr_max = external_snr_max
        self.noise_file_paths = list(noise_file_paths or [])
        self.gain_prob = gain_prob
        self.gain_db_min = gain_db_min
        self.gain_db_max = gain_db_max

    @staticmethod
    def _mix_at_snr(speech, noise, snr_db):
        eps = 1e-8
        alpha = torch.sqrt(
            speech.pow(2).mean().clamp_min(eps)
            / (10 ** (snr_db / 10.0) * noise.pow(2).mean().clamp_min(eps) + eps)
        )
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
            audio_signal = F.interpolate(
                audio_signal.unsqueeze(1), size=new_T, mode="linear", align_corners=False
            ).squeeze(1)
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
                snr = self.external_snr_min + torch.rand(1).item() * (
                    self.external_snr_max - self.external_snr_min
                )
                audio_signal[b, :L] = self._mix_at_snr(
                    audio_signal[b, :L].float(), seg, snr
                ).to(audio_signal.dtype)
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


# -----------------------------------------------------------------------------
# Noise curriculum (code A)
# -----------------------------------------------------------------------------
NOISE_CURRICULUM = os.environ.get("NOISE_CURRICULUM", "1").strip().lower() in ("1", "true", "yes")
NOISE_AUG_PROB = float(os.environ.get("NOISE_AUG_PROB", "0.6"))
NOISE_SNR_MIN = float(os.environ.get("NOISE_SNR_MIN", "1.0"))
NOISE_SNR_MAX = float(os.environ.get("NOISE_SNR_MAX", "7.0"))
NOISE_CURR_E1_2 = (0.3, 5.0, 20.0)
NOISE_CURR_E3_4 = (0.4, 5.0, 15.0)
NOISE_CURR_E5 = (0.45, 3.0, 12.0)


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
        if human <= 2:
            p, lo, hi, label = *NOISE_CURR_E1_2, "1-2"
        elif human <= 4:
            p, lo, hi, label = *NOISE_CURR_E3_4, "3-4"
        else:
            p, lo, hi, label = *NOISE_CURR_E5, "5+"
        wa.external_noise_prob = p
        wa.external_snr_min = lo
        wa.external_snr_max = hi
        if getattr(trainer, "global_rank", 0) == 0:
            print(
                f"[NoiseCurriculum] human_epoch={human}  phase={label}  noise_prob={p}  SNR=[{lo},{hi}]",
                flush=True,
            )


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
class SaveSelectedEpochs(Callback):
    def __init__(self, directory: str, save_epochs: list):
        super().__init__()
        self.directory = directory
        self.save_epochs = save_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        epoch = trainer.current_epoch
        if epoch not in self.save_epochs:
            return
        os.makedirs(self.directory, exist_ok=True)
        human = epoch + 1
        nemo_path = os.path.join(self.directory, f"model_epoch{human}.nemo")
        adapter_path = os.path.join(self.directory, f"adapter_epoch{human}.pt")
        try:
            pl_module.save_to(nemo_path)
        except Exception as e:
            print(f"[SaveSelectedEpochs] save_to warning: {e}", flush=True)
        try:
            pl_module.save_adapters(adapter_path)
        except Exception as e:
            print(f"[SaveSelectedEpochs] save_adapters warning: {e}", flush=True)
        print(f"[SaveSelectedEpochs] human_epoch={human} -> {nemo_path}, {adapter_path}", flush=True)


class ReinforceDecoderJointTrainMode(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        for attr in ("decoder", "joint", "waveform_augmentor", "spec_augmentation"):
            m = getattr(pl_module, attr, None)
            if m is not None:
                m.train()


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
        if getattr(trainer, "global_rank", 0) != 0:
            return
        if self.num_steps <= 0:
            return
        self._adapters_ordered = [
            (n, m) for n, m in pl_module.named_modules() if isinstance(m, ChainedLinearAdapter)
        ]
        self._adapters_ordered.sort(key=lambda x: x[0])
        for h in self._handles:
            h.remove()
        self._handles.clear()

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
        if getattr(trainer, "global_rank", 0) != 0:
            return
        if self._batch_ctr >= self.num_steps:
            return
        self._fwd_records.clear()

    def on_after_backward(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        if self._batch_ctr >= self.num_steps:
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
        if getattr(trainer, "global_rank", 0) != 0:
            return
        if self._batch_ctr >= self.num_steps:
            return
        recs = sorted(self._fwd_records, key=lambda r: r["idx"])
        print(f"\n{'=' * 80}", flush=True)
        print(
            f"[ChainedAdapterTrainDebug] batch={self._batch_ctr} "
            f"(epoch={trainer.current_epoch}, batch_idx={batch_idx})",
            flush=True,
        )
        loss_line = ""
        if isinstance(outputs, dict):
            for k in ("loss", "train_loss"):
                if k in outputs and outputs[k] is not None:
                    v = outputs[k]
                    loss_line = f"{k}={v.detach().float().item():.6f}" if torch.is_tensor(v) else f"{k}={v}"
                    break
        print(f"  {loss_line or type(outputs).__name__}", flush=True)
        for r in recs:
            print(
                f"    [{r['idx']:02d}] first={r['is_first']} in={r['in']} out={r['out']}  {r['name']}",
                flush=True,
            )
        if self._grad_lines:
            for line in self._grad_lines:
                print(line, flush=True)
        self._grad_lines = []
        print(f"{'=' * 80}\n", flush=True)
        self._batch_ctr += 1


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MODEL_ID = "nvidia/parakeet-tdt-1.1b"
BOTTLENECK_DIM = int(os.environ.get("BOTTLENECK_DIM", "256"))

ASR_DATA_DIR = os.environ.get(
    "ASR_DATA_DIR", "/kaggle/input/datasets/akarshkumarshukla/asr-data"
)
ASR_JSONL = os.environ.get(
    "ASR_JSONL", os.path.join(ASR_DATA_DIR, "train_word_transcripts.jsonl")
)
TALKBANK_DIR = os.environ.get(
    "TALKBANK_DIR", "/kaggle/input/datasets/akarshkumarshukla/talk-bank-data/audio"
)
TALKBANK_JSON = os.environ.get(
    "TALKBANK_JSON",
    "/kaggle/input/datasets/akarshkumarshukla/talkbank-transcript-full/train_word_transcripts.jsonl",
)

NOISE_DIRS: List[str] = []
_env_nd = os.environ.get("CLASSROOM_NOISE_DIRS", "").strip()
if _env_nd:
    for x in _env_nd.replace(";", ",").split(","):
        x = x.strip()
        if x:
            NOISE_DIRS.append(x)
else:
    NOISE_DIRS.extend(
        [
            "/kaggle/input/datasets/akarshks/noise/noise_part_1",
            "/kaggle/input/datasets/akarshkumarshukla/asr-data/noise_part_0",
        ]
    )

NOISE_FILES = _collect_noise_files(NOISE_DIRS)
print(f"Noise files: {len(NOISE_FILES)} from {len(NOISE_DIRS)} dir(s)")
if not NOISE_FILES:
    print("WARNING: No noise files — external noise augmentation disabled.")

SAVE_DIR = os.environ.get("SAVE_DIR", "/kaggle/working/nemo_adapter_1.1b")
MANIFEST_DIR = os.path.join(SAVE_DIR, "manifests")

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "7"))
MAX_DURATION_SEC = 20.0
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-4"))

SAVE_EPOCHS = [1, 2, 4]

os.makedirs(MANIFEST_DIR, exist_ok=True)

print(f"torch:  {torch.__version__}")
print(f"CUDA:   {torch.version.cuda}")
print(f"numpy:  {np.__version__}")
print(
    "[worker] manifest resolver v3: env + fixed paths + glob under /kaggle/input",
    flush=True,
)


def _discovered_train_manifest_jsonl(max_n: int = 40) -> list[str]:
    base = "/kaggle/input"
    if not os.path.isdir(base):
        return []
    pattern = os.path.join(base, "**", "train_manifest.jsonl")
    paths = sorted(
        (os.path.abspath(p) for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)),
        key=lambda x: (len(x), x),
    )
    return paths[:max_n]


USE_EXISTING_MANIFESTS = os.environ.get("USE_EXISTING_MANIFESTS", "1").strip().lower() in ("1", "true", "yes")
_env_train_manifest = os.environ.get("TRAIN_MANIFEST", "").strip()
_MANIFEST_CANDIDATES = [
    "/kaggle/input/datasets/akarshks/noise/train_manifest.jsonl",
    "/kaggle/input/datasets/akarshkumarshukla/earlier-manifest/train_manifest.jsonl",
    "/kaggle/input/datasets/akarshks/train-meta/train_manifest.jsonl",
]

if USE_EXISTING_MANIFESTS:
    _manifest_try_order: list[str] = []
    if _env_train_manifest:
        _manifest_try_order.append(os.path.abspath(os.path.expanduser(_env_train_manifest)))
    for c in _MANIFEST_CANDIDATES:
        ac = os.path.abspath(c)
        if ac not in _manifest_try_order:
            _manifest_try_order.append(ac)
    for p in _discovered_train_manifest_jsonl():
        if p not in _manifest_try_order:
            _manifest_try_order.append(p)
    train_path = None
    for i, p in enumerate(_manifest_try_order):
        if os.path.isfile(p):
            train_path = p
            if i > 0 and _env_train_manifest:
                print(
                    f"[manifest] TRAIN_MANIFEST {_env_train_manifest!r} not on disk — using {p}",
                    flush=True,
                )
            break
    if train_path is None:
        raise FileNotFoundError(
            "Train manifest not found. Tried (in order):\n  "
            + "\n  ".join(_manifest_try_order)
            + "\nAttach a dataset, fix TRAIN_MANIFEST, or USE_EXISTING_MANIFESTS=0."
        )
    print("Using existing manifest:", train_path)
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
        for fut in (ex.submit(_index_directory, ASR_DATA_DIR), ex.submit(_index_directory, TALKBANK_DIR)):
            audio_index.update(fut.result())
    print(f"Indexed {len(audio_index):,} files")

    def _find_audio(path):
        return audio_index.get(os.path.basename(path), None)

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
                    text = data.get("orthographic_text", "").strip()
                    if NORMALIZE_TEXT:
                        text = _whisper_normalizer(text)
                    else:
                        text = text.lower()
                    dur = float(data.get("audio_duration_sec", 0.0))
                    path = _find_audio(data.get("audio_path", ""))
                    if path and text and (0.1 < dur <= MAX_DURATION_SEC):
                        records.append({"audio_filepath": path, "duration": dur, "text": text})
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue

    df = pd.DataFrame(records)
    if len(df) == 0:
        raise RuntimeError("No training rows after JSONL + audio index.")
    train_path = os.path.join(MANIFEST_DIR, "train_manifest.jsonl")
    df.to_json(train_path, orient="records", lines=True)
    print(f"Train manifest written: {train_path} ({len(df):,} rows)")


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
                    "dim": BOTTLENECK_DIM,
                    "activation": "gelu",
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
                    "warmup_ratio": 0.15,
                    "min_lr": 1e-6,
                },
            },
        },
        "exp_manager": {
            "exp_dir": SAVE_DIR,
            "name": "ParakeetAdapter1.1B",
            "create_tensorboard_logger": True,
            "create_checkpoint_callback": False,
            "resume_if_exists": False,
            "resume_ignore_no_checkpoint": True,
        },
    }
)

_use_gpu = torch.cuda.is_available()
_accelerator = "gpu" if _use_gpu else "cpu"
_precision = "bf16-mixed" if _use_gpu else 32

print("Initializing Trainer (training only — no validation)...")
trainer = pl.Trainer(
    devices=1,
    accelerator=_accelerator,
    precision=_precision,
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

trainer.callbacks.append(SaveSelectedEpochs(ckpt_dir, save_epochs=SAVE_EPOCHS))
trainer.callbacks.append(ReinforceDecoderJointTrainMode())
trainer.callbacks.append(NoiseCurriculumCallback(enabled=NOISE_CURRICULUM))
_train_dbg = int(os.environ.get("TRAIN_DEBUG_STEPS", "0"))
if _train_dbg > 0:
    trainer.callbacks.append(ChainedAdapterTrainDebugCallback(num_steps=_train_dbg))
print(
    f"Saving checkpoints at human epochs {[e + 1 for e in SAVE_EPOCHS]} under {ckpt_dir}",
    flush=True,
)

print(f"Loading {MODEL_ID}...")
model_cfg = ASRModel.from_pretrained(MODEL_ID, return_config=True)
enc_key = _encoder_target_key(model_cfg)

with open_dict(model_cfg):
    adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder[enc_key])
    if adapter_metadata is not None:
        model_cfg.encoder[enc_key] = adapter_metadata.adapter_class_path
        print(f"Patched encoder to: {adapter_metadata.adapter_class_path}")

model = ASRModel.from_pretrained(MODEL_ID, override_config_path=model_cfg, trainer=trainer)

with open_dict(model.cfg):
    if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
        model.cfg.decoding.greedy.use_cuda_graph_decoder = False
if hasattr(model, "change_decoding_strategy"):
    model.change_decoding_strategy(model.cfg.decoding)

device = next(model.parameters()).device
_nprob, _nsnr0, _nsnr1 = (
    NOISE_CURR_E1_2
    if NOISE_CURRICULUM
    else (NOISE_AUG_PROB, NOISE_SNR_MIN, NOISE_SNR_MAX)
)
waveform_aug = WaveformAugmentor(
    speed_min=0.85,
    speed_max=1.15,
    pitch_min=-2.0,
    pitch_max=2.0,
    sample_rate=16000,
    speed_prob=0.5,
    pitch_prob=0.5,
    external_noise_prob=_nprob,
    external_snr_min=_nsnr0,
    external_snr_max=_nsnr1,
    noise_file_paths=NOISE_FILES if NOISE_FILES else None,
    gain_prob=0.3,
    gain_db_min=-8.0,
    gain_db_max=8.0,
).to(device)
model.add_module("waveform_augmentor", waveform_aug)

_original_forward = model.forward.__func__


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
    return _original_forward(
        self,
        input_signal=input_signal,
        input_signal_length=input_signal_length,
        processed_signal=processed_signal,
        processed_signal_length=processed_signal_length,
    )


model.forward = types.MethodType(_augmented_forward, model)

if "spec_augment" in cfg.model:
    model.spec_augmentation = model.from_config_dict(cfg.model.spec_augment)

cfg.model.train_ds = update_model_cfg(model.cfg.train_ds, cfg.model.train_ds)
model.setup_training_data(cfg.model.train_ds)

adapter_full_name = "encoder:asr_children_adapter"
adapter_short_name = "asr_children_adapter"
adapter_type_cfg = cfg.model.adapter.linear
model.add_adapter(name=adapter_full_name, cfg=adapter_type_cfg)
model.set_enabled_adapters(enabled=False)
model.set_enabled_adapters(name=adapter_full_name, enabled=True)

model.freeze()
model.unfreeze_enabled_adapters()

chain_state = AdapterChainState()
model.adapter_chain_state = chain_state

adapter_module_list = []
for mod_name, module in list(model.named_modules()):
    if isinstance(module, LinearAdapter) and adapter_short_name in mod_name:
        adapter_module_list.append((mod_name, module))
adapter_module_list.sort(key=lambda x: x[0])

for i, (mod_name, _orig) in enumerate(adapter_module_list):
    is_first_layer = i == 0
    new_adapter = ChainedLinearAdapter(
        in_features=1024,
        dim=BOTTLENECK_DIM,
        activation="gelu",
        norm_position="post",
        dropout=0.1,
        is_first=is_first_layer,
        chain_state_ref=chain_state,
        adapter_strategy=None,
    ).to(device)

    parts = mod_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    child = parts[-1]
    setattr(parent, child, new_adapter)
    if isinstance(parent, nn.ModuleDict):
        parent[child] = new_adapter
    if len(parts) >= 2 and parts[-2] == "adapter_layer":
        layer_mod = model
        for p in parts[:-2]:
            layer_mod = getattr(layer_mod, p)
        if hasattr(layer_mod, "adapter_layer") and adapter_short_name in layer_mod.adapter_layer:
            layer_mod.adapter_layer[adapter_short_name] = new_adapter
    new_adapter.setup_adapter_strategy(None)

LinearAdapter.register(ChainedLinearAdapter)
print(f"Replaced {len(adapter_module_list)} adapters with ChainedLinearAdapter (dim={BOTTLENECK_DIM})")

try:
    if hasattr(model, "joint"):
        for param in model.joint.parameters():
            param.requires_grad = True
        joint_count = sum(p.numel() for p in model.joint.parameters() if p.requires_grad)
        print(f"Unfroze joint network: {joint_count:,} params")
except Exception as e:
    print(f"WARNING: joint unfreeze: {e}")

try:
    if hasattr(model, "decoder") and hasattr(model.decoder, "prediction"):
        if hasattr(model.decoder.prediction, "embed"):
            for param in model.decoder.prediction.embed.parameters():
                param.requires_grad = True
            embed_count = sum(
                p.numel() for p in model.decoder.prediction.embed.parameters() if p.requires_grad
            )
            print(f"Unfroze decoder embedding: {embed_count:,} params")
except Exception as e:
    print(f"WARNING: decoder embed: {e}")


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


try:
    unfroze_lstm = False
    if hasattr(model, "decoder") and hasattr(model.decoder, "prediction"):
        pred = model.decoder.prediction
        unfroze_lstm = _unfreeze_last_lstm_in_module(pred)
        if not unfroze_lstm and hasattr(pred, "dec_rnn"):
            unfroze_lstm = _unfreeze_last_lstm_in_module(pred.dec_rnn)
    if unfroze_lstm:
        print("Unfroze last decoder LSTM layer")
    else:
        print("WARNING: No nn.LSTM found under decoder.prediction")
except Exception as e:
    print(f"WARNING: decoder LSTM: {e}")

for m in (getattr(model, "decoder", None), getattr(model, "joint", None), model.waveform_augmentor, getattr(model, "spec_augmentation", None)):
    if m is not None:
        m.train()

model.setup_optimization(cfg.model.optim)

try:
    from torch.optim import AdamW as TorchAdamW
    from torch.optim.lr_scheduler import LambdaLR

    adapter_params = []
    joint_params = []
    decoder_params = []

    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = pname.lower()
        if "adapter" in name_lower or "chain" in name_lower:
            adapter_params.append(param)
        elif "joint" in name_lower:
            joint_params.append(param)
        elif "decoder" in name_lower or "prediction" in name_lower or "embed" in name_lower:
            decoder_params.append(param)
        else:
            adapter_params.append(param)

    param_groups = []
    if adapter_params:
        param_groups.append({"params": adapter_params, "lr": 5e-4})
    if joint_params:
        param_groups.append({"params": joint_params, "lr": 1e-4})
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": 5e-5})

    if not param_groups:
        raise RuntimeError("No trainable parameters found for param groups")

    for i, pg in enumerate(param_groups):
        n = sum(p.numel() for p in pg["params"])
        print(f"Param group {i}: {n:,} params, lr={pg['lr']}")

    try:
        num_batches = len(model._train_dl)
    except (TypeError, AttributeError):
        num_batches = 1000
    total_steps = NUM_EPOCHS * num_batches
    warmup_steps = int(0.15 * total_steps)

    _disc_optimizer = TorchAdamW(param_groups, betas=(0.9, 0.999), weight_decay=0.01)

    def _lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    _disc_scheduler = LambdaLR(_disc_optimizer, _lr_lambda)

    def _custom_configure_optimizers(self_model):
        return {
            "optimizer": _disc_optimizer,
            "lr_scheduler": {
                "scheduler": _disc_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    model.configure_optimizers = types.MethodType(_custom_configure_optimizers, model)
    print(f"Discriminative LR (warmup={warmup_steps}, total={total_steps} steps)")

except Exception as e:
    print(f"WARNING: discriminative LR failed: {e} — using NeMo default optim")

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params:     {total:,}")
print(f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)")

print("=" * 60)
print("STARTING TRAINING (train only; limit_val_batches=0)")
print(f"  Adapter: 1024 -> {BOTTLENECK_DIM} -> 1024, GELU + LayerNorm + dropout=0.1")
print(f"  Epochs: {NUM_EPOCHS}  |  Aug: speed 0.85–1.15×, pitch −2..+2 st, gain ±8 dB @0.3")
print(f"  Noise curriculum: {NOISE_CURRICULUM}  (env NOISE_CURRICULUM=0 for fixed SNR)")
print(f"  SpecAugment: freq_masks=2, time_masks=10")
print("=" * 60)

trainer.fit(model)

print("Saving final model...")
model.save_adapters(os.path.join(ckpt_dir, "adapter_final.pt"))
model.save_to(os.path.join(ckpt_dir, "model_final.nemo"))
print(f"Saved under {ckpt_dir}")
print("TRAINING COMPLETE.")

