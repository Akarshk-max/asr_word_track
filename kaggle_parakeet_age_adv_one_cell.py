#!/usr/bin/env python3
# ruff: noqa
"""
Single Kaggle cell: pip installs, writes full training stack to working, runs train in subprocess.

Typical env:
  USE_CHAINED_ADAPTER=1
  TRAIN_MANIFEST=/path/to/train.jsonl
  SAVE_DIR=/kaggle/working/nemo_parakeet_age_adv

Mixed child + Libri (resolved inside this cell before training):
  TRAIN_MANIFEST=/path/child.jsonl,/path/libri_clean_age.jsonl
  TRAIN_MANIFEST_COHORTS=child,adult
  MERGED_TRAIN_MANIFEST_PATH=/kaggle/working/merged_train_manifest.jsonl  (optional)
  Child JSONL may use age_bucket (3-4, 5-7, 8-11, 12+, unknown) with or without age_years.

Optional:
  SKIP_KAGGLE_PIP=1
  KAGGLE_WORKING_DIR=/kaggle/working
  RUN_AGE_ADV_VERIFY=1
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

os.environ.setdefault("USE_CHAINED_ADAPTER", "1")

WORK = os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working").strip() or "/kaggle/working"
PATH_EXTRAS = os.path.join(WORK, "parakeet_age_adv_nemo_extras.py")
PATH_TRAIN = os.path.join(WORK, "train_parakeet_age_adversarial.py")
PATH_VERIFY = os.path.join(WORK, "verify_age_adv_training_ready.py")

# region agent log
def _agent_log(message: str, data: dict | None = None, hypothesis_id: str = "?") -> None:
    try:
        _p = os.path.join(WORK, "debug-f2babd.log")
        payload = {
            "sessionId": "f2babd",
            "timestamp": int(time.time() * 1000),
            "location": "kaggle_parakeet_age_adv_one_cell:driver",
            "message": message,
            "data": data or {},
            "hypothesisId": hypothesis_id,
        }
        with open(_p, "a", encoding="utf-8") as _lf:
            _lf.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# endregion agent log


def _pip(*args: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


_skip = os.environ.get("SKIP_KAGGLE_PIP", "0").strip().lower() in ("1", "true", "yes")
if not _skip:
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
else:
    print("SKIP_KAGGLE_PIP=1 — skipping pip installs.\n")

os.makedirs(WORK, exist_ok=True)


def _age_bucket_to_years(bucket) -> float:
    """Map Talkbank-style age_bucket to a numeric age for training (midpoint of band)."""
    b = str(bucket).strip().lower()
    if b == "3-4":
        return 3.5
    if b == "5-7":
        return 6.0
    if b == "8-11":
        return 9.5
    if b == "12+":
        return 13.0
    if b == "unknown":
        return 8.0
    return 8.0


def _ensure_trainable_age_fields(row: dict) -> bool:
    """
    Training expects age_years, age_target, or age. Child manifests may only have age_bucket;
    Libri rows typically have age_years (+ age_cohort).
    """
    if "age_target" in row:
        return True
    if "age" in row:
        return True
    if "age_years" in row:
        try:
            float(row["age_years"])
            return True
        except (TypeError, ValueError):
            pass
    if "age_bucket" in row:
        row["age_years"] = _age_bucket_to_years(row.get("age_bucket"))
        return True
    return False


def _merge_train_manifests_if_needed() -> None:
    """If TRAIN_MANIFEST lists multiple comma-separated JSONLs, merge and set TRAIN_MANIFEST to output."""
    raw = os.environ.get("TRAIN_MANIFEST", "").strip()
    if not raw:
        _agent_log("merge_skip_empty_TRAIN_MANIFEST", {}, "H5")
        return
    paths = [p.strip() for p in raw.split(",") if p.strip()]
    # region agent log
    _agent_log(
        "merge_inspect_TRAIN_MANIFEST",
        {"n_paths": len(paths), "raw_has_comma": "," in raw},
        "H1",
    )
    # endregion agent log
    if len(paths) <= 1:
        _agent_log("merge_skipped_single_path", {"n_paths": len(paths)}, "H5")
        return
    cohorts_raw = os.environ.get("TRAIN_MANIFEST_COHORTS", "").strip()
    if not cohorts_raw:
        raise ValueError(
            "TRAIN_MANIFEST has multiple files (comma-separated). Set TRAIN_MANIFEST_COHORTS with the "
            "same number of labels, e.g. TRAIN_MANIFEST_COHORTS=child,adult in matching order."
        )
    cohorts = [c.strip().lower() for c in cohorts_raw.split(",") if c.strip()]
    if len(cohorts) != len(paths):
        raise ValueError(
            f"TRAIN_MANIFEST has {len(paths)} paths but TRAIN_MANIFEST_COHORTS has {len(cohorts)} labels."
        )
    _allowed = {"child", "adult"}
    for c in cohorts:
        if c not in _allowed:
            raise ValueError(f"Invalid cohort {c!r}; use child or adult.")
    merged_path = os.environ.get("MERGED_TRAIN_MANIFEST_PATH", "").strip()
    if not merged_path:
        merged_path = os.path.join(WORK, "merged_train_manifest.jsonl")
    written = 0
    with open(merged_path, "w", encoding="utf-8") as out:
        for mf_path, cohort in zip(paths, cohorts):
            if not os.path.isfile(mf_path):
                raise FileNotFoundError(f"Manifest not found: {mf_path}")
            is_child = cohort == "child"
            with open(mf_path, "r", encoding="utf-8") as inf:
                for line in inf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    ap = row.get("audio_filepath") or row.get("audio_file") or row.get("audio_filename")
                    if not ap:
                        continue
                    text = (row.get("text") or "").strip()
                    if not text:
                        continue
                    if "duration" not in row:
                        continue
                    try:
                        row["duration"] = float(row["duration"])
                    except (TypeError, ValueError):
                        continue
                    if not _ensure_trainable_age_fields(row):
                        continue
                    row["audio_filepath"] = ap
                    row["text"] = text
                    row["age_cohort"] = cohort
                    row["is_child"] = is_child
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
    if written == 0:
        raise RuntimeError(
            "Merged manifest is empty. Check file paths, JSONL rows (audio_filepath, text, duration, age_*)."
        )
    os.environ["TRAIN_MANIFEST"] = merged_path
    # region agent log
    _agent_log(
        "merge_done",
        {"merged_path": merged_path, "written": written, "n_sources": len(paths)},
        "H5",
    )
    # endregion agent log
    print(f"Merged {len(paths)} manifests -> {merged_path} ({written} rows)\n", flush=True)


_CODE_EXTRAS = r'''"""
NeMo Parakeet age-adversarial Kaggle extras: chained bottleneck adapters, waveform aug, noise helpers.

Used by train_parakeet_age_adversarial.py when USE_CHAINED_ADAPTER=1 (notebook-style PEFT + aug).
"""
from __future__ import annotations

import math
import os
import random
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil, LinearAdapter


# -----------------------------------------------------------------------------
# Adapter chain
# -----------------------------------------------------------------------------


class AdapterChainState:
    def __init__(self) -> None:
        self.prev_bottleneck: Optional[torch.Tensor] = None
        self.layer_counter = 0

    def reset(self) -> None:
        self.prev_bottleneck = None
        self.layer_counter = 0

    def update(self, bottleneck: torch.Tensor) -> None:
        self.prev_bottleneck = bottleneck
        self.layer_counter += 1


class ChainedLinearAdapter(nn.Module, AdapterModuleUtil):
    """
    Bottleneck adapter for NeMo encoder layers. Initialized so the branch output is zero before
    training: with the encoder's residual ``x + adapter(x)`` convention, the net acts as identity
    at step zero (up/down/chain_proj zeros; LayerNorm of zeros stays zero).
    """

    def __init__(
        self,
        in_features: int,
        dim: int,
        activation: str = "gelu",
        norm_position: str = "post",
        dropout: float = 0.1,
        is_first: bool = False,
        chain_state_ref: Optional[AdapterChainState] = None,
        adapter_strategy: Any = None,
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
            nn.init.zeros_(self.chain_proj.weight)
        # Identity at init: zero bottleneck contribution so residual path is unchanged.
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.setup_adapter_strategy(adapter_strategy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.dropout_layer(h_norm)


# -----------------------------------------------------------------------------
# Waveform augmentation
# -----------------------------------------------------------------------------


class WaveformAugmentor(nn.Module):
    def __init__(
        self,
        speed_min: float = 0.85,
        speed_max: float = 1.15,
        pitch_min: float = -2.0,
        pitch_max: float = 2.0,
        sample_rate: int = 16000,
        speed_prob: float = 0.5,
        pitch_prob: float = 0.5,
        external_noise_prob: float = 0.6,
        external_snr_min: float = 1.0,
        external_snr_max: float = 7.0,
        noise_file_paths: Optional[Sequence[str]] = None,
        gain_prob: float = 0.3,
        gain_db_min: float = -8.0,
        gain_db_max: float = 8.0,
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
    def _mix_at_snr(speech: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
        eps = 1e-8
        alpha = torch.sqrt(
            speech.pow(2).mean().clamp_min(eps)
            / (10 ** (snr_db / 10.0) * noise.pow(2).mean().clamp_min(eps) + eps)
        )
        return speech + alpha * noise

    def _load_noise_segment(self, num_samples: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
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
            wav = torchaudio.functional.resample(wav, int(sr), self.sample_rate)
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
    def forward(
        self, audio_signal: torch.Tensor, signal_length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return audio_signal, signal_length
        b, t = audio_signal.shape
        if torch.rand(1).item() < self.speed_prob:
            sf = self.speed_min + torch.rand(1).item() * (self.speed_max - self.speed_min)
            new_t = max(1, int(t / sf))
            audio_signal = F.interpolate(
                audio_signal.unsqueeze(1), size=new_t, mode="linear", align_corners=False
            ).squeeze(1)
            signal_length = (signal_length.float() / sf).long().clamp(min=1, max=new_t)
        if torch.rand(1).item() < self.pitch_prob:
            n_steps = self.pitch_min + torch.rand(1).item() * (self.pitch_max - self.pitch_min)
            try:
                audio_signal = torchaudio.functional.pitch_shift(
                    audio_signal, self.sample_rate, float(n_steps)
                )
            except Exception:
                pass
        if self.noise_file_paths and torch.rand(1).item() < self.external_noise_prob:
            for i in range(b):
                L = int(min(signal_length[i].item(), audio_signal.shape[1]))
                if L < 1:
                    continue
                seg = self._load_noise_segment(L, audio_signal.device, audio_signal.dtype)
                if seg is None:
                    continue
                snr = self.external_snr_min + torch.rand(1).item() * (
                    self.external_snr_max - self.external_snr_min
                )
                audio_signal[i, :L] = self._mix_at_snr(
                    audio_signal[i, :L].float(), seg, float(snr)
                ).to(dtype=audio_signal.dtype)
        if torch.rand(1).item() < self.gain_prob:
            gain_db = self.gain_db_min + torch.rand(1).item() * (self.gain_db_max - self.gain_db_min)
            audio_signal = audio_signal * (10 ** (gain_db / 20.0))
        return audio_signal, signal_length


def collect_noise_audio_files(dir_list: Sequence[str]) -> List[str]:
    exts = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
    out: List[str] = []
    for d in dir_list:
        d = (d or "").strip()
        if not d or not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(exts):
                    out.append(os.path.join(root, f))
    return out


def parse_noise_dirs_from_env() -> List[str]:
    raw = os.environ.get("CLASSROOM_NOISE_DIRS", "").strip()
    if raw:
        return [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
    d1 = os.environ.get("CLASSROOM_NOISE_DIR_1", "").strip()
    d2 = os.environ.get("CLASSROOM_NOISE_DIR_2", "").strip()
    return [d for d in (d1, d2) if d]


NOISE_CURRICULUM = os.environ.get("NOISE_CURRICULUM", "1").strip().lower() in ("1", "true", "yes")
NOISE_CURR_E1_2 = (0.3, 5.0, 20.0)
NOISE_CURR_E3_4 = (0.4, 5.0, 15.0)
NOISE_CURR_E5 = (0.45, 3.0, 12.0)


class NoiseCurriculumCallback(Callback):
    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
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
                f"[NoiseCurriculum] human_epoch={human} phase={label} noise_prob={p} SNR=[{lo},{hi}]",
                flush=True,
            )


# -----------------------------------------------------------------------------
# Install chained adapters + aug on model
# -----------------------------------------------------------------------------


def spec_augment_cfg_dict() -> Dict[str, Any]:
    return {
        "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
        "freq_masks": 2,
        "freq_width": 27,
        "time_masks": 10,
        "time_width": 0.05,
    }


def replace_linear_adapters_with_chained(
    model: ASRModel,
    chain_state: AdapterChainState,
    in_features: int,
    bottleneck_dim: int,
    adapter_short_name: str = "asr_children_adapter",
) -> int:
    LinearAdapter.register(ChainedLinearAdapter)
    device = next(model.parameters()).device
    adapter_module_list: List[Tuple[str, LinearAdapter]] = []
    for mod_name, module in list(model.named_modules()):
        if isinstance(module, LinearAdapter) and adapter_short_name in mod_name:
            adapter_module_list.append((mod_name, module))
    adapter_module_list.sort(key=lambda x: x[0])
    for i, (mod_name, _orig) in enumerate(adapter_module_list):
        new_adapter = ChainedLinearAdapter(
            in_features=in_features,
            dim=bottleneck_dim,
            activation="gelu",
            norm_position="post",
            dropout=0.1,
            is_first=(i == 0),
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
    return len(adapter_module_list)


def attach_chained_adapter_stack(
    model: ASRModel,
    bottleneck_dim: int,
    in_features: int = 1024,
    adapter_full_name: str = "encoder:asr_children_adapter",
    adapter_short_name: str = "asr_children_adapter",
) -> AdapterChainState:
    linear_cfg = OmegaConf.create(
        {
            "_target_": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
            "in_features": in_features,
            "dim": bottleneck_dim,
            "activation": "gelu",
            "norm_position": "post",
            "dropout": 0.1,
        }
    )
    model.add_adapter(name=adapter_full_name, cfg=linear_cfg)
    model.set_enabled_adapters(enabled=False)
    model.set_enabled_adapters(name=adapter_full_name, enabled=True)
    model.freeze()
    model.unfreeze_enabled_adapters()
    chain_state = AdapterChainState()
    model.adapter_chain_state = chain_state
    n = replace_linear_adapters_with_chained(
        model, chain_state, in_features, bottleneck_dim, adapter_short_name
    )
    print(f"[age_adv] Replaced {n} LinearAdapter modules with ChainedLinearAdapter (dim={bottleneck_dim})")
    return chain_state


def patch_forward_waveform_and_chain_reset(model: ASRModel, waveform_aug: nn.Module) -> None:
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


def apply_spec_augmentation(model: ASRModel) -> None:
    model.spec_augmentation = model.from_config_dict(OmegaConf.create(spec_augment_cfg_dict()))


def partial_unfreeze_joint_decoder_lstm(model: ASRModel) -> None:
    try:
        if hasattr(model, "joint"):
            for p in model.joint.parameters():
                p.requires_grad = True
    except Exception as e:
        print(f"[age_adv] WARN joint unfreeze: {e}")
    try:
        if hasattr(model, "decoder") and hasattr(model.decoder, "prediction"):
            pred = model.decoder.prediction
            if hasattr(pred, "embed"):
                for p in pred.embed.parameters():
                    p.requires_grad = True
    except Exception as e:
        print(f"[age_adv] WARN decoder embed: {e}")

    def _unfreeze_last_lstm(container: Optional[nn.Module]) -> bool:
        if container is None:
            return False
        ok = False
        for _, mod in container.named_modules():
            if isinstance(mod, nn.LSTM) and mod.num_layers >= 1:
                last_i = mod.num_layers - 1
                for pname, param in mod.named_parameters():
                    if f"_l{last_i}" in pname:
                        param.requires_grad = True
                        ok = True
        return ok

    try:
        if hasattr(model, "decoder") and hasattr(model.decoder, "prediction"):
            pred = model.decoder.prediction
            if not _unfreeze_last_lstm(pred) and hasattr(pred, "dec_rnn"):
                _unfreeze_last_lstm(pred.dec_rnn)
    except Exception as e:
        print(f"[age_adv] WARN decoder LSTM: {e}")

    for m in (
        getattr(model, "decoder", None),
        getattr(model, "joint", None),
        getattr(model, "waveform_augmentor", None),
        getattr(model, "spec_augmentation", None),
    ):
        if m is not None:
            m.train()


def build_discriminative_configure_optimizers(
    model: ASRModel,
    age_disc: nn.Module,
    num_epochs: int,
    lr_adapter: float = 5e-4,
    lr_joint: float = 1e-4,
    lr_decoder: float = 5e-5,
    lr_age_disc: float = 1e-3,
    warmup_ratio: float = 0.15,
) -> Any:
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    disc_ids = {id(p) for p in age_disc.parameters()}
    adapter_params: List[torch.nn.Parameter] = []
    joint_params: List[torch.nn.Parameter] = []
    decoder_params: List[torch.nn.Parameter] = []
    age_disc_params: List[torch.nn.Parameter] = []

    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in disc_ids:
            age_disc_params.append(param)
            continue
        nl = pname.lower()
        if "adapter" in nl or "chain" in nl:
            adapter_params.append(param)
        elif "joint" in nl:
            joint_params.append(param)
        elif "decoder" in nl or "prediction" in nl or "embed" in nl:
            decoder_params.append(param)
        else:
            adapter_params.append(param)

    groups = []
    if age_disc_params:
        groups.append({"params": age_disc_params, "lr": lr_age_disc})
    if adapter_params:
        groups.append({"params": adapter_params, "lr": lr_adapter})
    if joint_params:
        groups.append({"params": joint_params, "lr": lr_joint})
    if decoder_params:
        groups.append({"params": decoder_params, "lr": lr_decoder})

    if not groups:
        raise RuntimeError("No trainable parameters for discriminative optimizer")

    try:
        num_batches = len(model._train_dl)
    except (TypeError, AttributeError):
        num_batches = 1000
    total_steps = max(1, num_epochs * num_batches)
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    optimizer = AdamW(groups, betas=(0.9, 0.999), weight_decay=0.01)

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, _lr_lambda)

    def _custom_configure_optimizers(_self) -> Dict[str, Any]:
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    return types.MethodType(_custom_configure_optimizers, model)


def rewire_adapter_chain_state(model: ASRModel) -> AdapterChainState:
    """After ``restore_from``, shared ``AdapterChainState`` may be missing; reattach for ChainedLinearAdapter."""
    state = AdapterChainState()
    model.adapter_chain_state = state
    n = 0
    for mod in model.modules():
        if isinstance(mod, ChainedLinearAdapter):
            mod.chain_state_ref = state
            n += 1
    print(f"[age_adv] Rewired chain_state_ref on {n} ChainedLinearAdapter module(s)", flush=True)
    return state


def stage2_prepare_trainable(model: ASRModel, adapter_full_name: str = "encoder:asr_children_adapter") -> None:
    """
    Full fine-tune mask after Stage 1 (frozen encoder + adapters): encoder backbone, adapters, full joint,
    partial decoder (embed + last LSTM per ``partial_unfreeze_joint_decoder_lstm``), and age head.
    Preprocessor weights stay frozen.
    """
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

    partial_unfreeze_joint_decoder_lstm(model)

    head = getattr(model, "parakeet_age_adv", None)
    if head is not None:
        for p in head.parameters():
            p.requires_grad = True

    pre = getattr(model, "preprocessor", None)
    if pre is not None:
        for p in pre.parameters():
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[age_adv] Stage2 trainable: {trainable:,} / {total:,} ({100.0 * trainable / max(1, total):.2f}%)",
        flush=True,
    )


def build_stage2_discriminative_configure_optimizers(
    model: ASRModel,
    age_disc: nn.Module,
    num_epochs: int,
    lr_encoder: float = 5e-6,
    lr_adapter: float = 3e-4,
    lr_joint: float = 1e-4,
    lr_decoder: float = 5e-5,
    lr_age_disc: float = 5e-4,
    warmup_ratio: float = 0.15,
) -> Any:
    """
    Five LR groups: encoder backbone (lowest), bottleneck adapters, joint, partial decoder, age discriminator.
    """
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    disc_ids = {id(p) for p in age_disc.parameters()}
    encoder_p: List[nn.Parameter] = []
    adapter_p: List[nn.Parameter] = []
    joint_p: List[nn.Parameter] = []
    decoder_p: List[nn.Parameter] = []
    age_disc_p: List[nn.Parameter] = []
    other_p: List[nn.Parameter] = []

    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in disc_ids:
            age_disc_p.append(param)
            continue
        nl = pname.lower()
        if ".adapter" in nl or "adapter_layer" in nl:
            adapter_p.append(param)
            continue
        if "joint" in nl:
            joint_p.append(param)
            continue
        if "decoder" in nl or "prediction" in nl:
            decoder_p.append(param)
            continue
        if "encoder" in nl or nl.startswith("encoder."):
            encoder_p.append(param)
            continue
        other_p.append(param)

    if other_p:
        op_ids = {id(p) for p in other_p}
        extra_names = [n for n, p in model.named_parameters() if id(p) in op_ids][:12]
        print(
            f"[age_adv] Stage2 optimizer: {len(other_p)} tensor(s) in 'other' group (lr=lr_joint). "
            f"Examples: {extra_names}",
            flush=True,
        )

    groups: List[Dict[str, Any]] = []
    if age_disc_p:
        groups.append({"params": age_disc_p, "lr": lr_age_disc})
    if encoder_p:
        groups.append({"params": encoder_p, "lr": lr_encoder})
    if adapter_p:
        groups.append({"params": adapter_p, "lr": lr_adapter})
    if joint_p:
        groups.append({"params": joint_p, "lr": lr_joint})
    if decoder_p:
        groups.append({"params": decoder_p, "lr": lr_decoder})
    if other_p:
        groups.append({"params": other_p, "lr": lr_joint})

    if not groups:
        raise RuntimeError("No trainable parameters for stage-2 optimizer")

    try:
        num_batches = len(model._train_dl)
    except (TypeError, AttributeError):
        num_batches = 1000
    total_steps = max(1, num_epochs * num_batches)
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    optimizer = AdamW(groups, betas=(0.9, 0.999), weight_decay=0.01)

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, _lr_lambda)

    def _custom_configure_optimizers(_self) -> Dict[str, Any]:
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    return types.MethodType(_custom_configure_optimizers, model)


class ChainedAdapterTrainDebugCallback(Callback):
    """Log chained adapter forwards + grads for first N batches (rank 0)."""

    def __init__(self, num_steps: int = 0):
        super().__init__()
        self.num_steps = int(num_steps)
        self._batch_ctr = 0
        self._handles: List[Any] = []
        self._adapters_ordered: List[Tuple[str, ChainedLinearAdapter]] = []
        self._fwd_records: List[dict] = []
        self._grad_lines: List[str] = []

    @staticmethod
    def _gmax(t: Optional[torch.Tensor]) -> float:
        if t is None or t.grad is None:
            return float("nan")
        return float(t.grad.detach().float().abs().max().item())

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        if getattr(trainer, "global_rank", 0) != 0 or self.num_steps <= 0:
            return
        self._adapters_ordered = [
            (n, m)
            for n, m in pl_module.named_modules()
            if isinstance(m, ChainedLinearAdapter)
        ]
        self._adapters_ordered.sort(key=lambda x: x[0])

        def _make_hook(idx: int, name: str):
            def _hook(mod: Any, inp: Any, out: Any) -> None:
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

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def on_train_batch_start(self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int) -> None:
        if getattr(trainer, "global_rank", 0) != 0 or self._batch_ctr >= self.num_steps:
            return
        self._fwd_records.clear()

    def on_after_backward(self, trainer: Any, pl_module: Any) -> None:
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

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
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
'''

_CODE_TRAIN = r'''#!/usr/bin/env python3
"""
Adversarial age-invariant ASR training for NeMo Parakeet TDT (paper-style).

Scalar loss (implementation convenience; one Lightning backward):

    L_impl = L_asr + λ_adv * L_adv + s_disc * L_disc

``s_disc`` (env ``DISC_LOSS_SCALE``, alias ``LAMBDA_AGE``) scales the **discriminator-only** supervised
BCE term; it is not an ASR-vs-side-task tradeoff like ``λ_adv``.

Parameter-isolated routing (approximates split optimization: discriminator on L_disc;
encoder + ASR on L_asr + λ_adv * L_adv). Intended zero partials:

    ∂L_disc / ∂θ_enc = 0 ,  ∂L_disc / ∂θ_asr = 0   — z_sup = disc(f.detach()), logits + BCEWithLogits
    ∂L_adv / ∂θ_disc = 0                         — adversarial branch: disc_forward_detached_weights(f)
    ∂L_asr / ∂θ_disc = 0                         — RNNT path has no discriminator

Gradients flow where intended: L_disc → θ_disc only; L_adv → θ_enc (via f); L_asr → θ_enc and θ_asr.

Losses: ``L_disc`` / ``L_adv`` use ``binary_cross_entropy_with_logits`` on discriminator logits vs soft
targets (age target a_i and 0.5); ``L_asr`` = NeMo RNNT/TDT (not CTC).

Age targets (paper-style): youngest child in span → 0; oldest child in corpus span → 0.8 (linear in
between); any age above that span → 1.0 (no adolescent ramp). Knots: ``AGE_CHILD_YOUNGEST``,
``AGE_CHILD_OLDEST_CORPUS`` (defaults 2–12 for a 2–12y child set mixed with Libri adults).

Manifest JSONL: audio_filepath, text, duration; age_years (recommended) or precomputed age_target.
Optional ``age_cohort`` (``child`` / ``adult``) or ``is_child`` (bool) enables **balanced batches** when
``BALANCED_CHILD_ADULT_BATCHES=1``: each batch is half child-cohort and half adult-cohort (oversamples
the smaller group within an epoch). Build LibriSpeech manifests with ``build_librispeech_age_manifest.py``.

**Mixed child + Libri in one run:** comma-separate paths in ``TRAIN_MANIFEST`` (or ``--manifest``) and set
``TRAIN_MANIFEST_COHORTS`` to matching labels, e.g.
``TRAIN_MANIFEST=/path/child.jsonl,/path/libri_clean_age.jsonl`` and ``TRAIN_MANIFEST_COHORTS=child,adult``.
The trainer writes ``SAVE_DIR/merged_train_manifest.jsonl`` (override with ``MERGED_TRAIN_MANIFEST_PATH``).
Alternatively pre-merge offline: ``merge_child_adult_training_manifests.py``.

Run: ``python train_parakeet_age_adversarial.py --manifest /path/to/train.jsonl`` (or set TRAIN_MANIFEST).

**Kaggle / notebook-style PEFT:** set ``USE_CHAINED_ADAPTER=1`` for chained bottleneck adapters (env
``BOTTLENECK_DIM``), waveform + SpecAugment, optional noise dirs (``CLASSROOM_NOISE_DIRS``), noise
curriculum, partial unfreeze (joint + decoder embed + last LSTM), and discriminative AdamW including
a dedicated ``age_disc`` LR (``LR_AGE_DISC``, etc.). Use ``kaggle_run_parakeet_age_adv.py`` for pip
clean + subprocess launch.

For **encoder + full joint + partial decoder** fine-tuning after a frozen-encoder Stage-1 ``.nemo``, use
``train_parakeet_age_adv_stage2.py`` (``INIT_NEMO``, discriminative ``LR_ENCODER``, same aug/noise hooks).

Env: SAVE_DIR, BATCH_SIZE, NUM_EPOCHS, USE_CHAINED_ADAPTER, BOTTLENECK_DIM, CLASSROOM_NOISE_DIRS,
NOISE_CURRICULUM, TRAIN_DEBUG_STEPS, LAMBDA_ADV (λ_max), LAMBDA_ADV_SCHEDULE, LAMBDA_ADV_RAMP_*,
DISC_LOSS_SCALE, LR_AGE_DISC / LR_ADAPTER / LR_JOINT / LR_DECODER, AGE_CHILD_* , WER_PROBE_N,
BALANCED_CHILD_ADULT_BATCHES, TRAIN_MANIFEST (comma-separated + TRAIN_MANIFEST_COHORTS for merge),
MERGED_TRAIN_MANIFEST_PATH, ...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import types
import warnings
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=Warning, module="numba")
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

import jiwer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.data.audio_to_text import AudioToBPEDataset, _speech_collate_fn
from nemo.collections.asr.models import ASRModel
from nemo.core.classes import adapter_mixins
from nemo.core.classes.mixins import AccessMixin
from nemo.utils import logging as nemo_logging

# -----------------------------------------------------------------------------
# Paper-style age target: youngest child → 0, oldest child in span → 0.8, else → 1.0 (no 0.8→1 ramp)
# -----------------------------------------------------------------------------


def paper_age_target(age_years: float, child_youngest: float, child_oldest: float) -> float:
    """
    Soft label in [0, 1] for BCEWithLogits (targets need not be binary).

    - At or below ``child_youngest`` -> 0; at ``child_oldest`` -> 0.8; linear between.
    - Strictly above ``child_oldest`` -> 1.0 (adults and any non-child ages).
    """
    a = float(age_years)
    if a <= child_youngest:
        return 0.0
    if a <= child_oldest:
        if child_oldest <= child_youngest:
            return 0.8
        t = (a - child_youngest) / (child_oldest - child_youngest)
        return float(0.8 * t)
    return 1.0


def scheduled_lambda_adv(
    current_epoch_zero_based: int,
    lambda_max: float,
    ramp_first_epoch_1based: int = 10,
    ramp_last_epoch_1based: int = 40,
) -> float:
    """
    Piecewise λ_adv(e) with e = trainer epoch **1-based** (first epoch has e=1).

    λ_adv = 0 for e < ramp_first; linear from 0 to λ_max for e in [ramp_first, ramp_last];
    λ_max for e > ramp_last. Matches: 0 for e<10, ramp 10–40, full after 40 when defaults are used.
    """
    e = int(current_epoch_zero_based) + 1
    rf = int(ramp_first_epoch_1based)
    rl = int(ramp_last_epoch_1based)
    if e < rf:
        return 0.0
    if e > rl:
        return float(lambda_max)
    denom = float(rl - rf)
    if denom <= 0.0:
        return float(lambda_max)
    return float(lambda_max) * float(e - rf) / denom


def build_age_lookup_from_manifests(
    manifest_filepaths: List[str],
    child_youngest: float,
    child_oldest: float,
    use_precomputed_target: bool = False,
) -> Dict[str, float]:
    """Map normalized absolute audio path -> soft target in [0,1]."""
    out: Dict[str, float] = {}
    for mf in manifest_filepaths:
        if not mf or not os.path.isfile(mf):
            continue
        with open(mf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ap = row.get("audio_filepath") or row.get("audio_file") or row.get("audio_filename")
                if not ap:
                    continue
                abspath = os.path.normpath(os.path.abspath(os.path.expanduser(ap)))
                if use_precomputed_target and "age_target" in row:
                    t = float(row["age_target"])
                elif "age_years" in row:
                    t = paper_age_target(float(row["age_years"]), child_youngest, child_oldest)
                elif "age" in row:
                    # If manifest stores already-normalized soft label in [0,1]
                    t = float(row["age"])
                else:
                    continue
                t = float(min(1.0, max(0.0, t)))
                out[abspath] = t
                out[os.path.basename(abspath)] = t
    return out


def build_cohort_lookup_from_manifests(manifest_filepaths: List[str]) -> Optional[Dict[str, bool]]:
    """
    Map normalized absolute path -> True if child cohort, False if adult, for balanced batching.
    Returns None if no row in any manifest defines a cohort field.
    """
    out: Dict[str, bool] = {}
    any_cohort = False
    for mf in manifest_filepaths:
        if not mf or not os.path.isfile(mf):
            continue
        with open(mf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ap = row.get("audio_filepath") or row.get("audio_file") or row.get("audio_filename")
                if not ap:
                    continue
                is_child: Optional[bool] = None
                if "age_cohort" in row:
                    any_cohort = True
                    v = str(row["age_cohort"]).strip().lower()
                    is_child = v == "child"
                elif "balance_cohort" in row:
                    any_cohort = True
                    v = str(row["balance_cohort"]).strip().lower()
                    is_child = v == "child"
                elif "is_child" in row:
                    any_cohort = True
                    is_child = bool(row["is_child"])
                if is_child is None:
                    continue
                abspath = os.path.normpath(os.path.abspath(os.path.expanduser(ap)))
                out[abspath] = is_child
                out[os.path.basename(abspath)] = is_child
    return out if any_cohort else None


# -----------------------------------------------------------------------------
# Balanced child / adult batches
# -----------------------------------------------------------------------------


class BalancedTwoGroupBatchSampler(torch.utils.data.Sampler[List[int]]):
    """
    Yields batches of indices with n_child = floor(batch_size/2) from the child pool and the rest
    from the adult pool. Pads the shorter pool by cycling so every epoch has the same number of
    batches: max(ceil(|C|/n_c), ceil(|A|/n_a)).
    """

    def __init__(
        self,
        child_indices: List[int],
        adult_indices: List[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if batch_size < 2:
            raise ValueError("BalancedTwoGroupBatchSampler needs batch_size >= 2")
        if not child_indices or not adult_indices:
            raise ValueError("BalancedTwoGroupBatchSampler needs non-empty child and adult index lists")
        self.child_indices = list(child_indices)
        self.adult_indices = list(adult_indices)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _num_batches(self) -> int:
        n_c = self.batch_size // 2
        n_a = self.batch_size - n_c
        n_batches = max(
            (len(self.child_indices) + n_c - 1) // n_c,
            (len(self.adult_indices) + n_a - 1) // n_a,
        )
        if self.drop_last:
            cap = (len(self.child_indices) + len(self.adult_indices)) // self.batch_size
            return min(n_batches, cap)
        return n_batches

    def __len__(self) -> int:
        return self._num_batches()

    def __iter__(self):
        n_c = self.batch_size // 2
        n_a = self.batch_size - n_c
        rng = random.Random(self.seed + self.epoch)
        child_order = list(self.child_indices)
        adult_order = list(self.adult_indices)
        if self.shuffle:
            rng.shuffle(child_order)
            rng.shuffle(adult_order)
        n_batches = self._num_batches()
        ic = ia = 0
        for _ in range(n_batches):
            batch: List[int] = []
            for _ in range(n_c):
                batch.append(child_order[ic % len(child_order)])
                ic += 1
            for _ in range(n_a):
                batch.append(adult_order[ia % len(adult_order)])
                ia += 1
            if self.shuffle:
                rng.shuffle(batch)
            yield batch


class BalancedBatchEpochCallback(Callback):
    """Advance BalancedTwoGroupBatchSampler epoch for reproducible shuffles each training epoch."""

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: Any) -> None:
        dl = getattr(pl_module, "_train_dl", None)
        if dl is None:
            return
        bs = getattr(dl, "batch_sampler", None)
        if bs is not None and hasattr(bs, "set_epoch"):
            bs.set_epoch(int(trainer.current_epoch))


# -----------------------------------------------------------------------------
# f0 normalization hook (integrate paper-specific DSP here)
# -----------------------------------------------------------------------------

_F0_LOGGED = False


def apply_f0_normalization(audio: torch.Tensor, sr: int, enabled: bool) -> torch.Tensor:
    """
    Time-domain waveform [B,T] or [T]. Placeholder: returns input.
    Replace with paper f0 estimation + warping (e.g. librosa/pyworld) when enabled.
    """
    global _F0_LOGGED
    if not enabled:
        return audio
    if not _F0_LOGGED:
        nemo_logging.warning(
            "apply_f0_normalization: enabled but using identity — implement paper f0 warping here."
        )
        _F0_LOGGED = True
    return audio


# -----------------------------------------------------------------------------
# Collate with age
# -----------------------------------------------------------------------------


def collate_age_bpe_batch(batch, pad_id: int):
    """batch items: (sig, sig_len, tok, tok_len, age_scalar_tensor)."""
    ages = torch.stack([b[4] for b in batch]).view(-1, 1)
    core = [b[:4] for b in batch]
    sig, sl, tok, tl = _speech_collate_fn(core, pad_id)
    return sig, sl, tok, tl, ages.to(dtype=torch.float32)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class AudioToBPEDatasetWithAge(AudioToBPEDataset):
    def __init__(
        self,
        *args,
        age_by_path: Optional[Dict[str, float]] = None,
        cohort_by_path: Optional[Dict[str, bool]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.age_by_path = age_by_path or {}
        self.cohort_by_path = cohort_by_path or {}
        self._missing_age = 0
        self._missing_cohort = 0

    def _lookup_age(self, audio_file: str) -> float:
        n = os.path.normpath(audio_file)
        if n in self.age_by_path:
            return float(self.age_by_path[n])
        bn = os.path.basename(n)
        if bn in self.age_by_path:
            return float(self.age_by_path[bn])
        if self._missing_age < 5:
            nemo_logging.warning(f"No age entry for {audio_file}; using target 0.5")
            self._missing_age += 1
        return 0.5

    def _lookup_cohort_is_child(self, audio_file: str) -> Optional[bool]:
        """If cohort dict is empty, balanced batching is disabled. Unknown path -> None (omit from both lists)."""
        if not self.cohort_by_path:
            return None
        n = os.path.normpath(audio_file)
        if n in self.cohort_by_path:
            return bool(self.cohort_by_path[n])
        bn = os.path.basename(n)
        if bn in self.cohort_by_path:
            return bool(self.cohort_by_path[bn])
        return None

    def balance_index_lists(self) -> Tuple[List[int], List[int]]:
        child: List[int] = []
        adult: List[int] = []
        for i in range(len(self.manifest_processor.collection)):
            sample = self.manifest_processor.collection[i]
            v = self._lookup_cohort_is_child(sample.audio_file)
            if v is True:
                child.append(i)
            elif v is False:
                adult.append(i)
            elif self.cohort_by_path:
                if self._missing_cohort < 5:
                    nemo_logging.warning(
                        f"No cohort entry for {sample.audio_file}; treating as adult for batch balance."
                    )
                    self._missing_cohort += 1
                adult.append(i)
        return child, adult

    def _process_sample(self, index):
        sample = self.manifest_processor.collection[index]
        offset = sample.offset
        if offset is None:
            offset = 0
        features = self.featurizer.process(
            sample.audio_file,
            offset=offset,
            duration=sample.duration,
            trim=self.trim,
            orig_sr=sample.orig_sr,
            channel_selector=self.channel_selector,
        )
        f, fl = features, torch.tensor(features.shape[0]).long()
        t, tl = self.manifest_processor.process_text_by_sample(sample=sample)
        age = torch.tensor(self._lookup_age(sample.audio_file), dtype=torch.float32)
        if self.return_sample_id:
            return f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), age, index
        return f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), age

    def _collate_fn(self, batch):
        if self.return_sample_id:
            raise NotImplementedError("return_sample_id + age collate not supported in this script")
        return collate_age_bpe_batch(batch, self.manifest_processor.pad_id)


# -----------------------------------------------------------------------------
# Pooling (encoder layout B, D, T)
# -----------------------------------------------------------------------------


def masked_mean_std_pool_bdt(encoded: torch.Tensor, lengths: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """encoded: [B, D, T], lengths: [B] — returns [B, 2D] concat(mean, std) over valid frames."""
    _, d, t_max = encoded.shape
    idx = torch.arange(t_max, device=encoded.device, dtype=torch.long).unsqueeze(0)
    mask = (idx < lengths.unsqueeze(1)).float().unsqueeze(1)  # [B, 1, T]

    denom = mask.sum(dim=2).clamp(min=1.0)  # [B, 1]
    mean = (encoded * mask).sum(dim=2) / denom  # [B, D]

    var = ((encoded - mean.unsqueeze(2)) ** 2 * mask).sum(dim=2) / denom
    std = torch.sqrt(var + eps)

    return torch.cat([mean, std], dim=1)


# -----------------------------------------------------------------------------
# Discriminator
# -----------------------------------------------------------------------------


class AgeDiscriminator(nn.Module):
    def __init__(self, d_in: int, hidden1: int = 512, hidden2: int = 256, dropout: float = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(d_in, hidden1)
        self.lin2 = nn.Linear(hidden1, hidden2)
        self.lin3 = nn.Linear(hidden2, 1)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.lin1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.lin2(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin3(h)

    def forward_detached_weights(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(F.linear(x, self.lin1.weight.detach(), self.lin1.bias.detach()))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(F.linear(h, self.lin2.weight.detach(), self.lin2.bias.detach()))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.linear(h, self.lin3.weight.detach(), self.lin3.bias.detach())


class ParakeetWithAgeAdversarial(nn.Module):
    """
    Logical head for encoder features + age discriminator. The pretrained ASR body stays on the
    NeMo ``ASRModel`` instance; this module only owns ``age_disc`` so optimizers can include it
    (training also attaches a copy as ``model.age_disc`` for ``training_step``).

    Pooling is mean+std over time (``2 * encoder_dim``) so the discriminator sees temporal spread.
    """

    def __init__(
        self,
        encoder_dim: int,
        hidden1: int = 512,
        hidden2: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.age_disc = AgeDiscriminator(2 * encoder_dim, hidden1, hidden2, dropout)

    def forward(
        self,
        asr: ASRModel,
        input_signal: torch.Tensor,
        input_signal_length: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        encoded, encoded_len = asr.forward(
            input_signal=input_signal, input_signal_length=input_signal_length
        )
        f = masked_mean_std_pool_bdt(encoded, encoded_len)
        age_logits = self.age_disc(f)
        return {
            "encoded": encoded,
            "encoded_len": encoded_len,
            "features": f,
            "age_logits": age_logits,
        }


# -----------------------------------------------------------------------------
# RNNT loss (non-fused + fused), mirrors EncDecRNNTModel.training_step
# -----------------------------------------------------------------------------


def transducer_loss_from_encoded(
    model: ASRModel,
    encoded: torch.Tensor,
    encoded_len: torch.Tensor,
    transcript: torch.Tensor,
    transcript_len: torch.Tensor,
) -> torch.Tensor:
    decoder, target_length, _states = model.decoder(
        targets=transcript, target_length=transcript_len
    )
    if not model.joint.fuse_loss_wer:
        joint = model.joint(encoder_outputs=encoded, decoder_outputs=decoder)
        loss_value = model.loss(
            log_probs=joint,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=target_length,
        )
        return model.add_auxiliary_losses(loss_value)
    loss_value, _wer, _, _ = model.joint(
        encoder_outputs=encoded,
        decoder_outputs=decoder,
        encoder_lengths=encoded_len,
        transcripts=transcript,
        transcript_lengths=transcript_len,
        compute_wer=False,
    )
    return model.add_auxiliary_losses(loss_value)


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------


class ReinforceDecoderJointTrainMode(Callback):
    """Keep decoder/joint (and aug) in train mode for RNN backward + augmentation gates."""

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


class SaveEpochsAndWERProbe(Callback):
    """Save .nemo at human epochs 5,6,7 and run greedy WER on a fixed train subset."""

    def __init__(
        self,
        ckpt_dir: str,
        save_human_epochs: List[int],
        manifest_path: str,
        probe_n: int,
        seed: int = 42,
        save_adapter_weights: bool = False,
    ):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.save_human_epochs = set(save_human_epochs)
        self.manifest_path = manifest_path
        self.probe_n = probe_n
        self.seed = seed
        self.save_adapter_weights = save_adapter_weights
        self._probe_paths: Optional[List[str]] = None
        self._probe_refs: Optional[List[str]] = None
        self.best_wer = float("inf")

    def _ensure_probe_lists(self, model: ASRModel):
        if self._probe_paths is not None:
            return
        if self.probe_n <= 0:
            self._probe_paths = []
            self._probe_refs = []
            nemo_logging.info("WER probe disabled (WER_PROBE_N<=0); only epoch .nemo saves run.")
            return
        rng = random.Random(self.seed)
        rows = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        rng.shuffle(rows)
        paths, refs = [], []
        for r in rows:
            if len(paths) >= self.probe_n:
                break
            ap = r.get("audio_filepath") or r.get("audio_file")
            tx = (r.get("text") or "").strip()
            if not ap or not tx:
                continue
            ap = os.path.normpath(os.path.abspath(os.path.expanduser(ap)))
            if os.path.isfile(ap):
                paths.append(ap)
                refs.append(tx.lower())
        self._probe_paths = paths
        self._probe_refs = refs
        nemo_logging.info(f"WER probe: {len(paths)} utterances from manifest.")

    def on_train_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        he = trainer.current_epoch + 1
        model = pl_module

        self._ensure_probe_lists(model)
        if self._probe_paths and self._probe_refs:
            model.eval()
            try:
                preds_raw = model.transcribe(audio=self._probe_paths, batch_size=min(8, len(self._probe_paths)))
            except TypeError:
                preds_raw = model.transcribe(
                    self._probe_paths, batch_size=min(8, len(self._probe_paths))
                )
            if isinstance(preds_raw, tuple):
                preds_raw = preds_raw[0]
            hyps = []
            for h in preds_raw:
                t = getattr(h, "text", None)
                hyps.append((t or str(h)).strip().lower() if t is not None else str(h).strip().lower())
            wer = float(jiwer.wer(self._probe_refs, hyps)) if self._probe_refs else 1.0
            nemo_logging.info(f"Epoch {he} train-subset WER (n={len(self._probe_refs)}): {wer:.4f}")
            log = getattr(trainer, "logger", None)
            if log is not None and callable(getattr(log, "log_metrics", None)):
                log.log_metrics({"train_probe_wer": wer}, step=trainer.global_step)
            if wer < self.best_wer:
                self.best_wer = wer
                best_path = os.path.join(self.ckpt_dir, "best.nemo")
                model.save_to(best_path)
                nemo_logging.info(f"New best WER {wer:.4f} -> {best_path}")
            model.train()

        if he in self.save_human_epochs:
            path = os.path.join(self.ckpt_dir, f"model_epoch{he}.nemo")
            model.save_to(path)
            nemo_logging.info(f"Saved checkpoint epoch {he} -> {path}")
            if self.save_adapter_weights and hasattr(model, "save_adapters"):
                ap = os.path.join(self.ckpt_dir, f"adapter_epoch{he}.pt")
                try:
                    model.save_adapters(ap)
                    nemo_logging.info(f"Saved adapter weights -> {ap}")
                except Exception as e:
                    nemo_logging.warning(f"save_adapters epoch {he}: {e}")


# -----------------------------------------------------------------------------
# Training step factory
# -----------------------------------------------------------------------------


def make_training_step_with_age(
    lambda_adv_max: float,
    disc_loss_scale: float,
    use_f0: bool,
    use_lambda_adv_schedule: bool,
    lambda_adv_ramp_start_1based: int,
    lambda_adv_ramp_end_1based: int,
):
    def training_step(self, batch, batch_nb):
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if len(batch) != 5:
            raise ValueError(f"Expected batch of 5 tensors (with age), got {len(batch)}")
        signal, signal_len, transcript, transcript_len, age_target = batch

        sr = int(getattr(self.preprocessor, "_sample_rate", 16000))
        signal = apply_f0_normalization(signal, sr, use_f0)

        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        f = masked_mean_std_pool_bdt(encoded, encoded_len)
        z_sup = self.age_disc(f.detach())
        z_adv = self.age_disc.forward_detached_weights(f)
        tgt = age_target.to(dtype=z_sup.dtype)
        half = torch.full_like(z_adv, 0.5)
        loss_disc = F.binary_cross_entropy_with_logits(z_sup, tgt, reduction="mean")
        loss_adv = F.binary_cross_entropy_with_logits(z_adv, half, reduction="mean")
        loss_asr = transducer_loss_from_encoded(
            self, encoded, encoded_len, transcript, transcript_len
        )
        if use_lambda_adv_schedule and self.trainer is not None:
            lam_adv = scheduled_lambda_adv(
                int(self.trainer.current_epoch),
                lambda_adv_max,
                ramp_first_epoch_1based=lambda_adv_ramp_start_1based,
                ramp_last_epoch_1based=lambda_adv_ramp_end_1based,
            )
        else:
            lam_adv = float(lambda_adv_max)
        loss = loss_asr + lam_adv * loss_adv + disc_loss_scale * loss_disc

        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        try:
            opt = self.trainer.optimizers[0]
            lr = opt.param_groups[0]["lr"]
        except Exception:
            lr = 0.0

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("loss_asr", loss_asr.detach(), prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss_adv", loss_adv.detach(), prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss_disc", loss_disc.detach(), prog_bar=False, on_step=True, on_epoch=True)
        self.log("lambda_adv", float(lam_adv), prog_bar=False, on_step=True, on_epoch=True)
        self.log("lr", lr, prog_bar=False, on_step=True, on_epoch=True)

        if getattr(self, "_optim_normalize_joint_txu", False):
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {"loss": loss}

    return training_step


def patch_configure_optimizers(model: ASRModel, age_disc: nn.Module):
    """Put age_disc params first in group 0 without duplicating (they are already in model.parameters())."""
    user_conf = model.configure_optimizers

    def _wrapped():
        out = user_conf()
        if isinstance(out, dict):
            opt = out["optimizer"]
        else:
            opt = out[0] if isinstance(out, (list, tuple)) else out
        disc_params = list(age_disc.parameters())
        if not disc_params:
            return out
        disc_ids = {id(p) for p in disc_params}
        g0 = opt.param_groups[0]
        rest = [p for p in g0["params"] if id(p) not in disc_ids]
        g0["params"] = disc_params + rest
        return out

    model.configure_optimizers = types.MethodType(_wrapped, model)


def infer_encoder_dim(model: ASRModel, device: torch.device) -> int:
    sr = int(getattr(model.preprocessor, "_sample_rate", 16000))
    x = torch.zeros(1, int(sr * 0.3), device=device)
    xl = torch.tensor([x.shape[1]], device=device, dtype=torch.long)
    with torch.no_grad():
        enc, el = model.forward(input_signal=x, input_signal_length=xl)
    return int(enc.shape[1])


def build_dataloader_with_age(
    model: ASRModel,
    train_cfg: Any,
    age_map: Dict[str, float],
    cohort_map: Optional[Dict[str, bool]] = None,
    use_balanced_batches: bool = False,
    balanced_sampler_seed: int = 0,
):
    use_start = train_cfg.get("use_start_end_token", True)
    ds = AudioToBPEDatasetWithAge(
        manifest_filepath=train_cfg.manifest_filepath,
        tokenizer=model.tokenizer,
        sample_rate=train_cfg.sample_rate,
        int_values=train_cfg.get("int_values", False),
        augmentor=None,
        max_duration=train_cfg.get("max_duration", None),
        min_duration=train_cfg.get("min_duration", None),
        max_utts=train_cfg.get("max_utts", 0),
        trim=train_cfg.get("trim_silence", False),
        use_start_end_token=use_start,
        return_sample_id=False,
        channel_selector=train_cfg.get("channel_selector", None),
        age_by_path=age_map,
        cohort_by_path=cohort_map if cohort_map is not None else {},
    )

    drop_last = bool(train_cfg.get("drop_last", False))
    nw = int(train_cfg.get("num_workers", 0))
    pin = bool(train_cfg.get("pin_memory", False))
    bs = int(train_cfg.batch_size)
    shuffle = bool(train_cfg.shuffle)

    if use_balanced_batches:
        if not cohort_map:
            nemo_logging.warning(
                "BALANCED_CHILD_ADULT_BATCHES=1 but manifest has no age_cohort / is_child / balance_cohort; "
                "using standard shuffle. Use build_librispeech_age_manifest.py or add cohort fields."
            )
        else:
            ch, ad = ds.balance_index_lists()
            if len(ch) == 0 or len(ad) == 0 or bs < 2:
                nemo_logging.warning(
                    f"Balanced batches disabled (child={len(ch)} adult={len(ad)} batch_size={bs}); "
                    "using standard shuffle."
                )
            else:
                try:
                    batch_sampler = BalancedTwoGroupBatchSampler(
                        ch,
                        ad,
                        batch_size=bs,
                        shuffle=shuffle,
                        seed=balanced_sampler_seed,
                        drop_last=drop_last,
                    )
                except ValueError as e:
                    nemo_logging.warning(f"Balanced batching disabled ({e}); using standard shuffle.")
                else:
                    nemo_logging.info(
                        f"Balanced batches: child_idx={len(ch)} adult_idx={len(ad)} batch_size={bs}"
                    )
                    return torch.utils.data.DataLoader(
                        dataset=ds,
                        batch_sampler=batch_sampler,
                        num_workers=nw,
                        pin_memory=pin,
                        collate_fn=ds.collate_fn,
                    )

    return torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=pin,
        drop_last=drop_last,
        collate_fn=ds.collate_fn,
    )


def main():
    parser = argparse.ArgumentParser(
        description="NeMo Parakeet TDT + age-adversarial BCE head (see module docstring)."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Train JSONL path (overrides TRAIN_MANIFEST env if set)",
    )
    args, _unknown = parser.parse_known_args()

    MODEL_ID = os.environ.get("MODEL_ID", "nvidia/parakeet-tdt-1.1b")
    raw_manifest = (args.manifest or os.environ.get("TRAIN_MANIFEST", "")).strip()
    if not raw_manifest:
        raise FileNotFoundError(
            "Set TRAIN_MANIFEST or pass --manifest to a JSONL with audio_filepath, text, duration, "
            "age_years (or age_target / age). Use comma-separated paths + TRAIN_MANIFEST_COHORTS to merge "
            "child and adult manifests (see module docstring)."
        )

    SAVE_DIR = os.environ.get("SAVE_DIR", "./nemo_parakeet_age_adv")
    os.makedirs(SAVE_DIR, exist_ok=True)

    manifest_paths = [p.strip() for p in raw_manifest.split(",") if p.strip()]
    if not manifest_paths:
        raise FileNotFoundError("TRAIN_MANIFEST is empty after splitting on commas.")

    for _mp in manifest_paths:
        if not os.path.isfile(_mp):
            raise FileNotFoundError(f"Manifest file not found: {_mp}")

    if len(manifest_paths) == 1:
        TRAIN_MANIFEST = manifest_paths[0]
    else:
        cohorts_raw = os.environ.get("TRAIN_MANIFEST_COHORTS", "").strip()
        if not cohorts_raw:
            raise ValueError(
                "Multiple comma-separated manifests require TRAIN_MANIFEST_COHORTS with the same number "
                "of labels (child or adult), e.g. TRAIN_MANIFEST_COHORTS=child,adult matching file order."
            )
        cohort_labels = [c.strip().lower() for c in cohorts_raw.split(",") if c.strip()]
        if len(cohort_labels) != len(manifest_paths):
            raise ValueError(
                f"TRAIN_MANIFEST has {len(manifest_paths)} files but TRAIN_MANIFEST_COHORTS has "
                f"{len(cohort_labels)} labels; counts must match."
            )
        _allowed_c = {"child", "adult"}
        for _c in cohort_labels:
            if _c not in _allowed_c:
                raise ValueError(f"Invalid TRAIN_MANIFEST_COHORTS entry {_c!r}; use child or adult.")

        from merge_child_adult_training_manifests import merge_labeled_manifests

        merged_path = os.environ.get("MERGED_TRAIN_MANIFEST_PATH", "").strip()
        if not merged_path:
            merged_path = os.path.join(SAVE_DIR, "merged_train_manifest.jsonl")
        _pairs = list(zip(manifest_paths, cohort_labels))
        _stats = merge_labeled_manifests(_pairs, merged_path)
        nemo_logging.info(
            f"Merged {len(manifest_paths)} manifests -> {merged_path} ({_stats.written} rows; "
            f"skipped no_age={_stats.skipped_no_age} no_duration={_stats.skipped_no_duration})"
        )
        if _stats.written == 0:
            raise RuntimeError(
                "Merged manifest is empty. Check inputs, age_years/age_target/age, and duration on each row."
            )
        TRAIN_MANIFEST = merged_path
    ckpt_dir = os.path.join(SAVE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
    NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "7"))
    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
    LAMBDA_ADV = float(os.environ.get("LAMBDA_ADV", "0.1"))
    LAMBDA_ADV_SCHEDULE = os.environ.get("LAMBDA_ADV_SCHEDULE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    LAMBDA_ADV_RAMP_START = int(os.environ.get("LAMBDA_ADV_RAMP_START", "10"))
    LAMBDA_ADV_RAMP_END = int(os.environ.get("LAMBDA_ADV_RAMP_END", "40"))
    DISC_LOSS_SCALE = float(
        os.environ.get("DISC_LOSS_SCALE", os.environ.get("LAMBDA_AGE", "1.0"))
    )
    USE_F0 = os.environ.get("USE_F0_NORM", "0").strip().lower() in ("1", "true", "yes")
    WER_PROBE_N = int(os.environ.get("WER_PROBE_N", "500"))
    PRECISION = os.environ.get("PRECISION", "bf16-mixed")

    CHILD_YOUNGEST = float(os.environ.get("AGE_CHILD_YOUNGEST", "2.0"))
    CHILD_OLDEST = float(os.environ.get("AGE_CHILD_OLDEST_CORPUS", "12.0"))
    USE_PRECOMPUTED_AGE_TARGET = os.environ.get("USE_PRECOMPUTED_AGE_TARGET", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    USE_BALANCED_BATCHES = os.environ.get("BALANCED_CHILD_ADULT_BATCHES", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    BALANCED_SAMPLER_SEED = int(os.environ.get("BALANCED_SAMPLER_SEED", "0"))

    USE_CHAINED_ADAPTER = os.environ.get("USE_CHAINED_ADAPTER", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    BOTTLENECK_DIM = int(os.environ.get("BOTTLENECK_DIM", "256"))
    LR_AGE_DISC = float(os.environ.get("LR_AGE_DISC", "1e-3"))
    LR_ADAPTER = float(os.environ.get("LR_ADAPTER", "5e-4"))
    LR_JOINT = float(os.environ.get("LR_JOINT", "1e-4"))
    LR_DECODER = float(os.environ.get("LR_DECODER", "5e-5"))
    _save_ep_raw = os.environ.get("SAVE_HUMAN_EPOCHS", "5,6,7").strip()
    SAVE_HUMAN_EPOCHS: List[int] = []
    for x in _save_ep_raw.split(","):
        x = x.strip()
        if x.isdigit():
            SAVE_HUMAN_EPOCHS.append(int(x))
    if not SAVE_HUMAN_EPOCHS:
        SAVE_HUMAN_EPOCHS = [5, 6, 7]

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    manifest_list = [TRAIN_MANIFEST]
    age_map = build_age_lookup_from_manifests(
        manifest_list,
        CHILD_YOUNGEST,
        CHILD_OLDEST,
        use_precomputed_target=USE_PRECOMPUTED_AGE_TARGET,
    )
    cohort_map = build_cohort_lookup_from_manifests(manifest_list)
    if not age_map:
        raise RuntimeError(
            "No age entries loaded from manifest. Add age_years, age_target, or age fields per row."
        )

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=PRECISION if torch.cuda.is_available() else 32,
        max_epochs=NUM_EPOCHS,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        logger=pl.loggers.TensorBoardLogger(save_dir=SAVE_DIR, name="age_adv"),
        gradient_clip_val=float(os.environ.get("GRAD_CLIP", "1.0")),
    )

    nemo_logging.info(f"Loading {MODEL_ID}...")
    model_cfg = ASRModel.from_pretrained(MODEL_ID, return_config=True)
    enc_key = "_target_" if "_target_" in model_cfg.encoder else "target"
    with open_dict(model_cfg):
        adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder[enc_key])
        if adapter_metadata is not None:
            model_cfg.encoder[enc_key] = adapter_metadata.adapter_class_path

    model = ASRModel.from_pretrained(MODEL_ID, override_config_path=model_cfg, trainer=trainer)

    with open_dict(model.cfg):
        if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
            model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    if hasattr(model, "change_decoding_strategy"):
        model.change_decoding_strategy(model.cfg.decoding)

    device = next(model.parameters()).device

    if USE_CHAINED_ADAPTER:
        from parakeet_age_adv_nemo_extras import (
            NOISE_CURRICULUM,
            ChainedAdapterTrainDebugCallback,
            NoiseCurriculumCallback,
            WaveformAugmentor,
            apply_spec_augmentation,
            attach_chained_adapter_stack,
            build_discriminative_configure_optimizers,
            collect_noise_audio_files,
            parse_noise_dirs_from_env,
            partial_unfreeze_joint_decoder_lstm,
            patch_forward_waveform_and_chain_reset,
        )

        noise_dirs = parse_noise_dirs_from_env()
        if not noise_dirs:
            noise_dirs = [
                "/kaggle/input/datasets/akarshks/noise/noise_part_1",
                "/kaggle/input/datasets/akarshkumarshukla/asr-data/noise_part_0",
            ]
        noise_files = collect_noise_audio_files(noise_dirs)
        nemo_logging.info(
            f"USE_CHAINED_ADAPTER=1: {len(noise_files)} noise clips from {len(noise_dirs)} dir(s)"
        )
        if NOISE_CURRICULUM:
            _nprob, _nsnr0, _nsnr1 = (0.3, 5.0, 20.0)
        else:
            _nprob = float(os.environ.get("NOISE_AUG_PROB", "0.6"))
            _nsnr0 = float(os.environ.get("NOISE_SNR_MIN", "1.0"))
            _nsnr1 = float(os.environ.get("NOISE_SNR_MAX", "7.0"))

        attach_chained_adapter_stack(model, BOTTLENECK_DIM, in_features=1024)
        waug = WaveformAugmentor(
            speed_min=0.85,
            speed_max=1.15,
            pitch_min=-2.0,
            pitch_max=2.0,
            sample_rate=int(getattr(model.preprocessor, "_sample_rate", 16000)),
            speed_prob=0.5,
            pitch_prob=0.5,
            external_noise_prob=_nprob,
            external_snr_min=_nsnr0,
            external_snr_max=_nsnr1,
            noise_file_paths=noise_files if noise_files else None,
            gain_prob=0.3,
            gain_db_min=-8.0,
            gain_db_max=8.0,
        ).to(device)
        patch_forward_waveform_and_chain_reset(model, waug)
        apply_spec_augmentation(model)
        partial_unfreeze_joint_decoder_lstm(model)

    d_enc = infer_encoder_dim(model, device)
    _age_head = ParakeetWithAgeAdversarial(d_enc).to(device)
    model.add_module("parakeet_age_adv", _age_head)
    model.age_disc = model.parakeet_age_adv.age_disc

    model.training_step = types.MethodType(
        make_training_step_with_age(
            LAMBDA_ADV,
            DISC_LOSS_SCALE,
            USE_F0,
            use_lambda_adv_schedule=LAMBDA_ADV_SCHEDULE,
            lambda_adv_ramp_start_1based=LAMBDA_ADV_RAMP_START,
            lambda_adv_ramp_end_1based=LAMBDA_ADV_RAMP_END,
        ),
        model,
    )

    with open_dict(model.cfg.train_ds):
        model.cfg.train_ds.manifest_filepath = TRAIN_MANIFEST
        model.cfg.train_ds.batch_size = BATCH_SIZE
        model.cfg.train_ds.num_workers = NUM_WORKERS
        model.cfg.train_ds.shuffle = True

    model.setup_training_data(model.cfg.train_ds)
    model._train_dl = build_dataloader_with_age(
        model,
        model.cfg.train_ds,
        age_map,
        cohort_map=cohort_map,
        use_balanced_batches=USE_BALANCED_BATCHES,
        balanced_sampler_seed=BALANCED_SAMPLER_SEED,
    )

    if USE_CHAINED_ADAPTER:
        from parakeet_age_adv_nemo_extras import build_discriminative_configure_optimizers

        model.configure_optimizers = build_discriminative_configure_optimizers(
            model,
            model.age_disc,
            NUM_EPOCHS,
            lr_adapter=LR_ADAPTER,
            lr_joint=LR_JOINT,
            lr_decoder=LR_DECODER,
            lr_age_disc=LR_AGE_DISC,
        )
        nemo_logging.info(
            f"Discriminative optimizer: LR_AGE_DISC={LR_AGE_DISC} LR_ADAPTER={LR_ADAPTER} "
            f"LR_JOINT={LR_JOINT} LR_DECODER={LR_DECODER}"
        )
    else:
        patch_configure_optimizers(model, model.age_disc)

    trainer.callbacks.append(BalancedBatchEpochCallback())
    trainer.callbacks.append(ReinforceDecoderJointTrainMode())
    if USE_CHAINED_ADAPTER and NOISE_CURRICULUM:
        trainer.callbacks.append(NoiseCurriculumCallback(enabled=True))
    _dbg = int(os.environ.get("TRAIN_DEBUG_STEPS", "0"))
    if USE_CHAINED_ADAPTER and _dbg > 0:
        trainer.callbacks.append(ChainedAdapterTrainDebugCallback(num_steps=_dbg))
    trainer.callbacks.append(
        SaveEpochsAndWERProbe(
            ckpt_dir,
            save_human_epochs=SAVE_HUMAN_EPOCHS,
            manifest_path=TRAIN_MANIFEST,
            probe_n=WER_PROBE_N,
            save_adapter_weights=USE_CHAINED_ADAPTER,
        )
    )

    if LAMBDA_ADV_SCHEDULE:
        nemo_logging.info(
            f"λ_adv schedule: 0 until epoch<{LAMBDA_ADV_RAMP_START} (1-based), "
            f"linear ramp to λ_max={LAMBDA_ADV} through epoch {LAMBDA_ADV_RAMP_END}, then constant."
        )
    nemo_logging.info(
        f"Training: chained_adapter={USE_CHAINED_ADAPTER} lambda_adv_max={LAMBDA_ADV} "
        f"schedule={LAMBDA_ADV_SCHEDULE} disc_loss_scale={DISC_LOSS_SCALE} epochs={NUM_EPOCHS} "
        f"encoder_dim={d_enc} age_disc_in={2 * d_enc} age_keys={len(age_map)}"
    )
    trainer.fit(model)

    final_path = os.path.join(ckpt_dir, "model_final.nemo")
    model.save_to(final_path)
    nemo_logging.info(f"Saved final model -> {final_path}")
    if USE_CHAINED_ADAPTER and hasattr(model, "save_adapters"):
        try:
            model.save_adapters(os.path.join(ckpt_dir, "adapter_final.pt"))
            nemo_logging.info("Saved adapter_final.pt")
        except Exception as e:
            nemo_logging.warning(f"save_adapters final: {e}")


if __name__ == "__main__":
    main()
'''

_CODE_VERIFY = r'''#!/usr/bin/env python3
"""Lightweight checks before / after age-adversarial training (Kaggle optional subprocess)."""
from __future__ import annotations

import json
import os
import sys


def main() -> int:
    manifest = (os.environ.get("TRAIN_MANIFEST") or os.environ.get("VERIFY_MANIFEST", "")).strip()
    if manifest:
        if not os.path.isfile(manifest):
            print(f"FAIL: manifest not found: {manifest}", file=sys.stderr)
            return 1
        n = 0
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    n += 1
                except json.JSONDecodeError:
                    print(f"FAIL: invalid JSONL line in {manifest}", file=sys.stderr)
                    return 1
        print(f"OK: manifest {manifest} has {n} JSON lines")
    else:
        print("SKIP: no TRAIN_MANIFEST / VERIFY_MANIFEST set")

    try:
        import torch  # noqa: F401
    except ImportError:
        print("FAIL: torch not importable", file=sys.stderr)
        return 1
    try:
        from nemo.collections.asr.models import ASRModel  # noqa: F401
    except ImportError as e:
        print(f"FAIL: NeMo ASR import: {e}", file=sys.stderr)
        return 1
    print("OK: torch + nemo ASR import")

    try:
        import parakeet_age_adv_nemo_extras  # noqa: F401
    except ImportError as e:
        print(f"WARN: parakeet_age_adv_nemo_extras: {e}", file=sys.stderr)
    else:
        print("OK: parakeet_age_adv_nemo_extras import")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''



with open(PATH_EXTRAS, "w", encoding="utf-8") as _f:
    _f.write(_CODE_EXTRAS)
with open(PATH_TRAIN, "w", encoding="utf-8") as _f:
    _f.write(_CODE_TRAIN)
with open(PATH_VERIFY, "w", encoding="utf-8") as _f:
    _f.write(_CODE_VERIFY)

print(f"Wrote {PATH_EXTRAS}")
print(f"Wrote {PATH_TRAIN}")
print(f"Wrote {PATH_VERIFY}\n")

_merge_train_manifests_if_needed()

# region agent log
_tm = os.environ.get("TRAIN_MANIFEST", "").strip()
_agent_log(
    "pre_train_env",
    {
        "TRAIN_MANIFEST": _tm[:500] if _tm else "",
        "manifest_has_comma": "," in _tm,
        "SAVE_DIR": os.environ.get("SAVE_DIR", ""),
        "BALANCED_CHILD_ADULT_BATCHES": os.environ.get("BALANCED_CHILD_ADULT_BATCHES", ""),
    },
    "H1",
)
# endregion agent log

print("=" * 60)
print("Launching train_parakeet_age_adversarial.py (subprocess)")
print("=" * 60)

_env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": WORK}
_comp = subprocess.run(
    [sys.executable, "-m", "py_compile", PATH_EXTRAS, PATH_TRAIN, PATH_VERIFY],
    cwd=WORK,
    capture_output=True,
    text=True,
)
if _comp.returncode != 0:
    print(_comp.stderr or _comp.stdout)
    raise RuntimeError("py_compile failed")

_result = subprocess.run(
    [sys.executable, PATH_TRAIN],
    cwd=WORK,
    env=_env,
    capture_output=True,
    text=True,
)
# region agent log
_out = (_result.stdout or "") + ("\n" + _result.stderr if _result.stderr else "")
_agent_log(
    "train_subprocess_finished",
    {
        "returncode": _result.returncode,
        "stdout_tail": _out[-8000:] if _out else "",
    },
    "H3",
)
# endregion agent log
if _result.returncode != 0:
    print("--- train subprocess stdout/stderr (last 12k chars) ---")
    print(_out[-12000:] if _out else "(empty)")
    raise RuntimeError(f"Training exited with code {_result.returncode}")
if _out.strip():
    print(_out[-2000:] if len(_out) > 2000 else _out, flush=True)

if os.environ.get("RUN_AGE_ADV_VERIFY", "0").strip().lower() in ("1", "true", "yes"):
    print("\n" + "=" * 60)
    print("RUN_AGE_ADV_VERIFY=1 — verify_age_adv_training_ready.py")
    print("=" * 60)
    subprocess.check_call(
        [sys.executable, PATH_VERIFY],
        cwd=WORK,
        env=_env,
    )

print("\nDone. Check SAVE_DIR for checkpoints.")
