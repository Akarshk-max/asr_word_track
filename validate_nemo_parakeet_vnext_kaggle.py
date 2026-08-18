"""
Kaggle one-cell (or local): Part A pip, Part B write eval worker, Part C subprocess.

Validates WER/CER on a manifest for Parakeet-TDT-1.1B vNext checkpoints produced by
train_nemo_adapter_parakeet_vnext.py (EarlyTwoBottleneckAdapter + LateChainedAdapter + LoRA + stream).

Use full model_epoch*.nemo from SaveSelectedEpochs (not adapter-only .pt for this script).

Env (optional, inherited by child):
  MODEL_NEMO or MODEL_PATH, VAL_MANIFEST, BATCH_SIZE, NUM_WORKERS,
  RESULTS_JSON, PREDICTIONS_JSONL, CHUNK_MULTIPLIER

Class definitions in the worker must stay in sync with train_nemo_vnext.py (train_code) for unpickling.
"""

from __future__ import annotations

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
# PART A: INSTALLATION (aligned with train / validate 1.1b)
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
# PART B: WRITE EVAL SCRIPT
# =============================================================================
EVAL_SCRIPT = "/kaggle/working/compute_val_wer_parakeet_vnext.py"

eval_code = r'''
from __future__ import annotations

# keep in sync with train_nemo_adapter_parakeet_vnext.py train_code (unpickling + NeMo adapter registry)

import json
import logging
import math
import os
import random
import time
import types
import warnings

warnings.filterwarnings("ignore", category=Warning, module="numba")
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

import jiwer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil, LinearAdapter


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


class AdapterStreamState:
    def __init__(self):
        self.prev_bottleneck = None

    def reset(self):
        self.prev_bottleneck = None

    def update_after_adapter(self, b_n):
        self.prev_bottleneck = b_n


def _match_seq_len(t, target_len):
    if t.shape[1] == target_len:
        return t
    return F.interpolate(t.transpose(1, 2), size=target_len, mode="nearest").transpose(1, 2)


class EarlyTwoBottleneckAdapter(nn.Module, AdapterModuleUtil):
    def __init__(self, in_features=1024, b1=256, b2=256,
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
        nn.init.normal_(self.up.weight, std=0.002)
        nn.init.zeros_(self.up.bias)
        self.setup_adapter_strategy(adapter_strategy)

    def forward(self, x):
        _, T, _ = x.shape
        h2_pre = self.down2(self.down1(x))
        if not self.is_first and self.stream_state_ref is not None:
            pb = self.stream_state_ref.prev_bottleneck
            if pb is not None:
                h2_pre = h2_pre + self.chain_proj(_match_seq_len(pb, T))
        b_n = self.norm_b(self.act(h2_pre))
        out = self.dropout_layer(self.norm_out(self.up(b_n)))
        if self.stream_state_ref is not None:
            self.stream_state_ref.update_after_adapter(b_n)
        return out


class LateChainedAdapter(nn.Module, AdapterModuleUtil):
    def __init__(self, in_features=1024, dim=256,
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
        nn.init.normal_(self.up.weight, std=0.002)
        nn.init.zeros_(self.up.bias)
        self.setup_adapter_strategy(adapter_strategy)

    def forward(self, x):
        _, T, _ = x.shape
        h_pre = self.down(x)
        if not self.is_first and self.stream_state_ref is not None:
            pb = self.stream_state_ref.prev_bottleneck
            if pb is not None:
                h_pre = h_pre + self.chain_proj(_match_seq_len(pb, T))
        b_n = self.norm_b(self.act(h_pre))
        out = self.dropout_layer(self.norm_out(self.up(b_n)))
        if self.stream_state_ref is not None:
            self.stream_state_ref.update_after_adapter(b_n)
        return out


class WaveformAugmentor(nn.Module):
    def __init__(self, speed_min=0.90, speed_max=1.10, pitch_min=-1.0, pitch_max=3.0,
                 sample_rate=16000, speed_prob=0.5, pitch_prob=0.5,
                 external_noise_prob=0.6, external_snr_min=1.0, external_snr_max=10.0,
                 noise_file_paths=None, gain_prob=0.3, gain_db_min=-6.0, gain_db_max=6.0):
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
                audio_signal = torchaudio.functional.pitch_shift(
                    audio_signal, self.sample_rate, n_steps
                )
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


LinearAdapter.register(EarlyTwoBottleneckAdapter)
LinearAdapter.register(LateChainedAdapter)

try:
    from whisper_normalizer.english import EnglishTextNormalizer

    _whisper_normalizer = EnglishTextNormalizer()
    HAS_WHISPER_NORM = True
except ImportError:
    _whisper_normalizer = None
    HAS_WHISPER_NORM = False


def normalize_ref(text: str) -> str:
    text = (text or "").strip()
    if HAS_WHISPER_NORM and _whisper_normalizer:
        return _whisper_normalizer(text)
    return text.lower()


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
MODEL_PATH = os.environ.get(
    "MODEL_NEMO",
    os.environ.get(
        "MODEL_PATH",
        "/kaggle/working/nemo_adapter_parakeet_vnext/ParakeetVNext/2026-04-03_23-41-51/checkpoints/model_epoch5.nemo",
    ),
)
VAL_MANIFEST = os.environ.get(
    "VAL_MANIFEST",
    "/kaggle/input/datasets/akarshkumarshukla/vel-mani/val_manifest.jsonl",
)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
RESULTS_JSON = os.environ.get(
    "RESULTS_JSON", "/kaggle/working/val_wer_results_parakeet_vnext.json"
)
PREDICTIONS_JSONL = os.environ.get(
    "PREDICTIONS_JSONL", "/kaggle/working/val_predictions_parakeet_vnext.jsonl"
)
CHUNK_MULTIPLIER = int(os.environ.get("CHUNK_MULTIPLIER", "10"))


def _patched_transcribe_dataloader(model, config, num_workers: int):
    if "manifest_filepath" in config:
        manifest_filepath = config["manifest_filepath"]
        batch_size = config["batch_size"]
    else:
        manifest_filepath = os.path.join(config["temp_dir"], "manifest.json")
        batch_size = min(config["batch_size"], len(config["paths2audio_files"]))

    use_ste = False
    if hasattr(model.cfg, "validation_ds") and model.cfg.validation_ds is not None:
        use_ste = model.cfg.validation_ds.get("use_start_end_token", False)

    dl_config = {
        "use_lhotse": False,
        "manifest_filepath": manifest_filepath,
        "sample_rate": model.preprocessor._sample_rate,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": min(batch_size, num_workers),
        "pin_memory": False,
        "channel_selector": config.get("channel_selector", "average"),
        "use_start_end_token": use_ste,
    }
    if config.get("augmentor"):
        dl_config["augmentor"] = config["augmentor"]
    return model._setup_dataloader_from_config(config=DictConfig(dl_config))


def _reattach_adapter_stream_state(model: ASRModel) -> None:
    st = AdapterStreamState()
    model.adapter_stream_state = st
    n = 0
    for m in model.modules():
        if isinstance(m, (EarlyTwoBottleneckAdapter, LateChainedAdapter)):
            m.stream_state_ref = st
            n += 1
    if n:
        print(
            f"Re-attached AdapterStreamState to {n} vNext adapter module(s)",
            flush=True,
        )


def _ensure_vnext_inference_forward(model: ASRModel) -> None:
    _original_forward = model.forward.__func__

    def _fw(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        if hasattr(self, "adapter_stream_state"):
            self.adapter_stream_state.reset()
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

    model.forward = types.MethodType(_fw, model)


def main() -> None:
    print("=" * 70)
    print("VALIDATION WER (Parakeet-TDT-1.1B vNext .nemo)")
    print("=" * 70)

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.isfile(VAL_MANIFEST):
        raise FileNotFoundError(f"Val manifest not found: {VAL_MANIFEST}")

    print(f"Model: {MODEL_PATH} ({os.path.getsize(MODEL_PATH) / 1024**3:.2f} GB)")
    print(f"Manifest: {VAL_MANIFEST}")
    print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    t0 = time.time()
    try:
        model = ASRModel.restore_from(MODEL_PATH, map_location=device, strict=False)
    except TypeError:
        model = ASRModel.restore_from(MODEL_PATH, map_location=device)

    with open_dict(model.cfg):
        if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
            model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    if hasattr(model, "change_decoding_strategy"):
        model.change_decoding_strategy(model.cfg.decoding)

    _reattach_adapter_stream_state(model)
    _ensure_vnext_inference_forward(model)

    model = model.to(device)
    model.eval()
    model.freeze()
    if hasattr(model, "waveform_augmentor") and model.waveform_augmentor is not None:
        model.waveform_augmentor.eval()

    def _patched_setup_transcribe(self, config):
        return _patched_transcribe_dataloader(self, config, NUM_WORKERS)

    model._setup_transcribe_dataloader = types.MethodType(_patched_setup_transcribe, model)

    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    samples = []
    skipped_empty_norm = 0
    with open(VAL_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                audio_path = (
                    data.get("audio_filepath")
                    or data.get("audio_path")
                    or ""
                ).strip()
                text = (data.get("text") or data.get("orthographic_text") or "").strip()
                duration = float(data.get("duration") or data.get("audio_duration_sec") or 0.0)
                if audio_path and text and os.path.isfile(audio_path):
                    ref_norm = normalize_ref(text)
                    if not (ref_norm or "").strip():
                        skipped_empty_norm += 1
                        continue
                    samples.append(
                        {
                            "path": audio_path,
                            "ref_raw": text,
                            "ref_norm": ref_norm,
                            "dur": duration,
                        }
                    )
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

    if not samples:
        raise RuntimeError(
            "No valid rows (need audio path + non-empty text after normalization + existing file)."
        )

    references = [s["ref_norm"] for s in samples]
    audio_files = [s["path"] for s in samples]
    total_duration = sum(s["dur"] for s in samples)

    print(
        f"Samples: {len(samples):,}  |  duration: {total_duration / 3600:.2f} h"
        + (
            f"  |  skipped {skipped_empty_norm:,} rows (empty ref after norm)"
            if skipped_empty_norm
            else ""
        ),
        flush=True,
    )

    chunk_size = max(BATCH_SIZE, BATCH_SIZE * CHUNK_MULTIPLIER)
    all_predictions = []
    t1 = time.time()
    for i in tqdm(range(0, len(audio_files), chunk_size), desc="Transcribing"):
        chunk = audio_files[i : i + chunk_size]
        raw = model.transcribe(
            chunk,
            batch_size=BATCH_SIZE,
            channel_selector="average",
            verbose=False,
        )
        if isinstance(raw, tuple):
            raw = raw[0]
        for h in raw:
            txt = getattr(h, "text", None)
            all_predictions.append(txt if isinstance(txt, str) else str(h))

    elapsed = time.time() - t1
    predictions_normalized = [normalize_ref(p) for p in all_predictions]

    if len(predictions_normalized) != len(references):
        raise RuntimeError(
            f"Prediction count {len(predictions_normalized)} != references {len(references)}"
        )

    wer = jiwer.wer(references, predictions_normalized)
    cer = jiwer.cer(references, predictions_normalized)
    measures = jiwer.compute_measures(references, predictions_normalized)
    exact_matches = sum(1 for r, p in zip(references, predictions_normalized) if r == p)

    per_wer = []
    for i, (ref, pred) in enumerate(zip(references, predictions_normalized)):
        try:
            per_wer.append((i, jiwer.wer([ref], [pred]), ref, pred))
        except Exception:
            per_wer.append((i, 1.0, ref, pred))

    worst = sorted(per_wer, key=lambda x: x[1], reverse=True)
    best = sorted(per_wer, key=lambda x: x[1])

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"WER:  {wer:.4f} ({wer * 100:.2f}%)")
    print(f"CER:  {cer:.4f} ({cer * 100:.2f}%)")
    print(
        f"Exact match: {exact_matches:,}/{len(samples):,} ({100 * exact_matches / len(samples):.1f}%)"
    )
    print(
        f"Time: {elapsed:.1f}s  |  {len(audio_files) / max(elapsed, 1e-6):.2f} files/s  |  RTF {elapsed / max(total_duration, 1e-6):.3f}x"
    )
    print(
        f"Substitutions: {measures['substitutions']:,}  Deletions: {measures['deletions']:,}  Insertions: {measures['insertions']:,}"
    )

    print("\n10 worst (by utterance WER):")
    for idx, (i, w, ref, pred) in enumerate(worst[:10]):
        print(f"\n[{idx + 1}] utt={i} WER={w:.2f}")
        print(f"  REF:  {ref[:120]}{'...' if len(ref) > 120 else ''}")
        print(f"  HYP:  {pred[:120]}{'...' if len(pred) > 120 else ''}")

    print("\n10 best (by utterance WER):")
    for idx, (i, w, ref, pred) in enumerate(best[:10]):
        print(f"\n[{idx + 1}] utt={i} WER={w:.2f}")
        print(f"  REF:  {ref[:120]}{'...' if len(ref) > 120 else ''}")
        print(f"  HYP:  {pred[:120]}{'...' if len(pred) > 120 else ''}")

    results = {
        "model_path": MODEL_PATH,
        "val_manifest": VAL_MANIFEST,
        "num_samples": len(samples),
        "skipped_empty_normalized_reference": skipped_empty_norm,
        "total_duration_hours": total_duration / 3600,
        "wer": wer,
        "cer": cer,
        "exact_match_rate": exact_matches / len(samples),
        "substitutions": measures["substitutions"],
        "deletions": measures["deletions"],
        "insertions": measures["insertions"],
        "hits": measures["hits"],
        "inference_time_sec": elapsed,
        "files_per_sec": len(audio_files) / max(elapsed, 1e-6),
    }
    os.makedirs(os.path.dirname(RESULTS_JSON) or ".", exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics: {RESULTS_JSON}")

    with open(PREDICTIONS_JSONL, "w", encoding="utf-8") as f:
        wer_by_idx = {i: w for i, w, _, _ in per_wer}
        for i, s in enumerate(samples):
            f.write(
                json.dumps(
                    {
                        "idx": i,
                        "audio_path": s["path"],
                        "reference_raw": s["ref_raw"],
                        "reference_norm": references[i],
                        "prediction_raw": all_predictions[i],
                        "prediction_norm": predictions_normalized[i],
                        "utterance_wer": wer_by_idx.get(i),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Saved predictions: {PREDICTIONS_JSONL}")
    print("=" * 70)


if __name__ == "__main__":
    main()
'''

os.makedirs("/kaggle/working", exist_ok=True)
with open(EVAL_SCRIPT, "w", encoding="utf-8") as f:
    f.write(eval_code)

print(f"Eval script written to {EVAL_SCRIPT}\n")
print(
    "Env: MODEL_NEMO or MODEL_PATH, VAL_MANIFEST, BATCH_SIZE, NUM_WORKERS, "
    "RESULTS_JSON, PREDICTIONS_JSONL, CHUNK_MULTIPLIER\n"
)

# =============================================================================
# PART C: SUBPROCESS
# =============================================================================
print("=" * 60)
print("Running validation (fresh interpreter)")
print("=" * 60)

_compile = subprocess.run(
    [sys.executable, "-m", "py_compile", EVAL_SCRIPT],
    cwd="/kaggle/working",
    capture_output=True,
    text=True,
)
if _compile.returncode != 0:
    print(_compile.stderr or _compile.stdout)
    raise RuntimeError("eval script py_compile failed")

result = subprocess.run(
    [sys.executable, EVAL_SCRIPT],
    cwd="/kaggle/working",
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
    stdout=None,
    stderr=subprocess.STDOUT,
)

if result.returncode != 0:
    raise RuntimeError(f"Validation failed with exit code {result.returncode}")

print("\nValidation finished successfully.")
