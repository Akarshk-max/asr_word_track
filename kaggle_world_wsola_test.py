# -*- coding: utf-8 -*-
"""
======================================================================
Kaggle: Child → Adult Speech Transform — ASR WER Test
======================================================================
Paste this entire file into ONE Kaggle notebook cell.

Part A — pip install (NeMo + audio deps)
Part B — write self-contained transform + eval script (torchaudio-based)
Part C — subprocess.run (fresh Python interpreter)

What it does:
  1. Picks a subset of child speech audio files
  2. Applies: pitch ↓ (torchaudio phase vocoder) + speed ↑ (resample trick)
  3. Transcribes BOTH original and transformed audio with Parakeet-TDT
  4. Compares WER to measure whether the transform helps ASR

Transform uses torchaudio (GPU-accelerated) — matching the playground.

Env vars (optional — set BEFORE running this cell):
  SKIP_PIP=1                     Skip installation if already done
  MODEL_SOURCE=pretrained        Use base pretrained model (default)
  MODEL_SOURCE=/path/to.nemo     Use your fine-tuned .nemo checkpoint
  VAL_MANIFEST=/path/to.jsonl    NeMo manifest (audio_filepath, text, duration)
  MAX_SAMPLES=200                Limit samples for quick test
  BATCH_SIZE=16                  ASR transcription batch size
  PITCH_SEMITONES=-6             Pitch shift in semitones (negative = lower)
  SPEED_FACTOR=1.08              Speed-up factor (> 1 = faster)
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
SKIP_PIP = os.environ.get("SKIP_PIP", "0").strip().lower() in ("1", "true", "yes")

if not SKIP_PIP:
    print("Step 1: Cleaning conflicting packages...")
    for _ in range(2):
        subprocess.run(
            [
                sys.executable, "-m", "pip", "uninstall", "-y",
                "numpy", "scipy", "nemo_toolkit", "lightning", "pytorch-lightning",
                "datasets", "diffusers", "gradio", "peft", "sentence-transformers",
                "transformers", "huggingface_hub", "torch", "torchaudio",
                "torchvision", "numba",
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    print("Step 2: numpy + scipy...")
    _pip("install", "--no-cache-dir", "numpy>=2.1,<2.3", "scipy>=1.14,<1.16")

    print("Step 3: PyTorch (CUDA 12.6)...")
    subprocess.check_call(
        [
            sys.executable, "-m", "pip", "install", "--no-cache-dir",
            "--index-url", "https://download.pytorch.org/whl/cu126",
            "torch>=2.9.0", "torchaudio",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    print("Step 4: ASR + audio dependencies...")
    _pip(
        "install", "--no-cache-dir",
        "transformers>=4.57.6,<4.58",
        "huggingface_hub>=0.30.0",
        "lightning>=2.2.0",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.2",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        "sentencepiece>=0.2.0",
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
        "install", "--no-cache-dir", "--force-reinstall",
        "numpy>=2.1,<2.3", "scipy>=1.14,<1.16",
    )

    print("Installation complete.\n")
else:
    print("SKIP_PIP=1 — using current kernel packages.\n")


# =============================================================================
# PART B: WRITE SELF-CONTAINED SCRIPT
# =============================================================================
SCRIPT_PATH = "/kaggle/working/world_wsola_asr_test.py"

script_code = r'''#!/usr/bin/env python3
"""
Child → Adult speech transformation + ASR WER comparison.

Transforms use torchaudio (GPU-accelerated, matching the playground):
  1. Pitch shift down  →  torchaudio.functional.pitch_shift (phase vocoder)
  2. Speed up          →  torchaudio.transforms.Resample trick
"""
from __future__ import annotations

import json
import logging
import os
import time
import warnings
from typing import List

warnings.filterwarnings("ignore", category=Warning, module="numba")
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

import jiwer
import numpy as np
import soundfile as sf_lib
import torch
import torchaudio
from omegaconf import open_dict
from tqdm import tqdm

from nemo.collections.asr.models import ASRModel


# ─────────────────────────────────────────────────────────────────────────────
# Configuration (override with env vars)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_SOURCE     = os.environ.get("MODEL_SOURCE", "pretrained").strip()
MODEL_ID         = "nvidia/parakeet-tdt-1.1b"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE       = int(os.environ.get("BATCH_SIZE", "16"))
MAX_SAMPLES      = os.environ.get("MAX_SAMPLES", "200").strip()
MAX_SAMPLES_N    = int(MAX_SAMPLES) if MAX_SAMPLES.isdigit() else 200

# Transform knobs — matching the playground
PITCH_SEMITONES  = int(os.environ.get("PITCH_SEMITONES", "-6"))     # negative = lower
SPEED_FACTOR     = float(os.environ.get("SPEED_FACTOR", "1.08"))    # > 1 = faster

# Manifest
_val_env = os.environ.get("VAL_MANIFEST", "").strip()
if _val_env:
    VAL_MANIFEST = _val_env
else:
    _candidates = [
        "/kaggle/input/datasets/akarshks/val-meta/val_manifest.jsonl",
        "/kaggle/working/nemo_layer_fusion_stage1/manifests/val_manifest.jsonl",
        "/kaggle/input/datasets/akarshks/val-manifest/val_manifest.jsonl",
        "/kaggle/input/datasets/akarshkumarshukla/val-train/val_manifest.jsonl",
        "/kaggle/working/nemo_adapter_run/manifests/val_manifest.jsonl",
    ]
    VAL_MANIFEST = next((p for p in _candidates if os.path.isfile(p)), _candidates[0])

TRANSFORMED_DIR = "/kaggle/working/transformed_audio"
os.makedirs(TRANSFORMED_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Transform Functions (torchaudio — matching playground exactly)
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str):
    """Load audio with soundfile → torch tensor (bypasses torchcodec/FFmpeg)."""
    data, sr = sf_lib.read(path, dtype="float32")
    t = torch.from_numpy(data)
    if t.ndim == 1:
        t = t.unsqueeze(0)                      # [samples] → [1, samples]
    else:
        t = t.T                                  # [samples, ch] → [ch, samples]
        t = t.mean(dim=0, keepdim=True)           # stereo → mono
    return t, sr


def pitch_shift(waveform: torch.Tensor, sr: int, n_steps: int) -> torch.Tensor:
    """
    Shift pitch by n_steps semitones using torchaudio's phase vocoder.
    GPU-accelerated when available.

    n_steps < 0 → lower pitch  (child → adult)
    """
    if n_steps == 0:
        return waveform
    return torchaudio.functional.pitch_shift(
        waveform.to(DEVICE), sr, n_steps=n_steps
    ).cpu()


def speed_change(waveform: torch.Tensor, sr: int, factor: float) -> torch.Tensor:
    """
    Change speed WITHOUT changing pitch using resample trick.

    factor > 1.0 → faster speech (child speaks slower → make adult-like).

    How: treat audio as if recorded at (sr * factor), resample back to sr.
    This compresses time without affecting pitch.
    """
    if abs(factor - 1.0) < 0.005:
        return waveform
    orig_freq = int(sr * factor)
    resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=sr)
    return resampler(waveform)


def transform_child_to_adult(
    waveform: torch.Tensor,
    sr: int,
    pitch_semitones: int = -6,
    speed_factor: float = 1.08,
) -> torch.Tensor:
    """
    Full child → adult-like pipeline (torchaudio):

      child audio
         ↓  pitch_shift (phase vocoder, GPU)  →  lower F0
         ↓  resample speed trick              →  faster speech
         ↓  normalize
      adult-like audio
    """
    w = pitch_shift(waveform, sr, pitch_semitones)
    w = speed_change(w, sr, speed_factor)

    # Normalize to prevent clipping
    peak = w.abs().max()
    if peak > 0:
        w = w / peak * 0.95
    return w


# ─────────────────────────────────────────────────────────────────────────────
# ASR Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    """Load Parakeet-TDT model (pretrained or from .nemo checkpoint)."""
    if MODEL_SOURCE.lower() == "pretrained":
        print(f"Loading pretrained {MODEL_ID}...")
        model = ASRModel.from_pretrained(MODEL_ID, map_location=torch.device(DEVICE))
    else:
        print(f"Restoring from checkpoint: {MODEL_SOURCE}...")
        try:
            model = ASRModel.restore_from(MODEL_SOURCE, map_location=torch.device(DEVICE), strict=False)
        except TypeError:
            model = ASRModel.restore_from(MODEL_SOURCE, map_location=torch.device(DEVICE))

    model = model.to(DEVICE)
    model.eval()

    with open_dict(model.cfg):
        if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
            model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    if hasattr(model, "change_decoding_strategy"):
        model.change_decoding_strategy(model.cfg.decoding)

    # Patch dataloader to disable lhotse (crashes on mono 1D audio tensors)
    from omegaconf import DictConfig

    def _patched_transcribe_dataloader(config):
        if "manifest_filepath" in config:
            manifest_filepath = config["manifest_filepath"]
            batch_size = config["batch_size"]
        else:
            manifest_filepath = os.path.join(config["temp_dir"], "manifest.json")
            batch_size = min(config["batch_size"], len(config["paths2audio_files"]))
        dl_config = {
            "use_lhotse": False,
            "manifest_filepath": manifest_filepath,
            "sample_rate": model.preprocessor._sample_rate,
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": min(batch_size, 4),
            "pin_memory": False,
            "channel_selector": config.get("channel_selector", None),
            "use_start_end_token": model.cfg.validation_ds.get(
                "use_start_end_token", False
            ),
        }
        return model._setup_dataloader_from_config(config=DictConfig(dl_config))

    model._setup_transcribe_dataloader = _patched_transcribe_dataloader
    print("  (Lhotse patched for mono audio)")

    return model


def transcribe_files(model, audio_paths: List[str]) -> List[str]:
    """Transcribe a list of audio file paths → list of text strings."""
    all_preds = []
    for start in range(0, len(audio_paths), BATCH_SIZE):
        chunk = audio_paths[start : start + BATCH_SIZE]
        try:
            raw = model.transcribe(audio=chunk, batch_size=len(chunk), channel_selector="average")
        except TypeError:
            raw = model.transcribe(paths2audio_files=chunk, batch_size=len(chunk), channel_selector="average")
        if isinstance(raw, tuple):
            raw = raw[0]
        for h in raw:
            if h is None:
                all_preds.append("")
            elif isinstance(h, str):
                all_preds.append(h.strip().lower())
            else:
                t = getattr(h, "text", str(h))
                all_preds.append(str(t).strip().lower())
    return all_preds


def load_manifest(path: str):
    """Load NeMo manifest → list of (audio_path, reference_text, duration)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ap = d.get("audio_filepath") or d.get("audio_path", "")
            text = (d.get("text") or "").strip().lower()
            dur = float(d.get("duration", 0.0))
            if ap and text and os.path.isfile(ap) and dur > 0.1:
                rows.append((ap, text, dur))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Child → Adult Speech Transform — ASR WER Test (torchaudio)")
    print("=" * 70)
    print(f"  torch:             {torch.__version__}")
    print(f"  CUDA:              {torch.version.cuda}")
    print(f"  device:            {DEVICE}")
    print(f"  Pitch:             {PITCH_SEMITONES:+d} semitones")
    print(f"  Speed:             {SPEED_FACTOR}x")
    print(f"  Model:             {MODEL_SOURCE}")
    print(f"  Manifest:          {VAL_MANIFEST}")
    print(f"  Max samples:       {MAX_SAMPLES_N}")
    print()

    # ---- Load manifest ----
    if not os.path.isfile(VAL_MANIFEST):
        raise FileNotFoundError(
            f"Manifest not found: {VAL_MANIFEST}\n"
            "Set VAL_MANIFEST env var to your NeMo JSONL manifest."
        )

    rows = load_manifest(VAL_MANIFEST)
    if not rows:
        raise RuntimeError("No usable rows in manifest (need audio_filepath + text).")

    if MAX_SAMPLES_N and len(rows) > MAX_SAMPLES_N:
        rows = rows[:MAX_SAMPLES_N]
    print(f"Using {len(rows)} samples for evaluation\n")

    # ---- Transform audio ----
    print("─" * 70)
    print("Step 1: Transforming child speech → adult-like speech")
    print(f"        (pitch {PITCH_SEMITONES:+d}st, speed {SPEED_FACTOR}x)")
    print("─" * 70)

    original_paths: List[str] = []
    transformed_paths: List[str] = []
    references: List[str] = []
    skipped = 0

    for i, (audio_path, ref_text, dur) in enumerate(tqdm(rows, desc="Transforming")):
        try:
            # Load audio with soundfile (bypasses torchcodec)
            waveform, sr = load_audio(audio_path)

            # Apply transform (torchaudio, GPU-accelerated)
            waveform_t = transform_child_to_adult(
                waveform, sr,
                pitch_semitones=PITCH_SEMITONES,
                speed_factor=SPEED_FACTOR,
            )

            # Save transformed audio
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            out_path = os.path.join(
                TRANSFORMED_DIR,
                f"{basename}_p{PITCH_SEMITONES}_sp{SPEED_FACTOR}.wav",
            )
            sf_lib.write(out_path, waveform_t.squeeze().numpy(), sr)

            original_paths.append(audio_path)
            transformed_paths.append(out_path)
            references.append(ref_text)

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  [skip] {os.path.basename(audio_path)}: {e}")
            continue

    print(f"\nTransformed: {len(original_paths)}, Skipped: {skipped}\n")

    if not original_paths:
        raise RuntimeError("No audio files were successfully transformed.")

    # ---- Load ASR model ----
    print("─" * 70)
    print("Step 2: Loading ASR model")
    print("─" * 70)
    model = load_model()
    print("Model ready.\n")

    # ---- Transcribe original audio ----
    print("─" * 70)
    print("Step 3: Transcribing ORIGINAL child speech")
    print("─" * 70)
    t0 = time.time()
    preds_original = transcribe_files(model, original_paths)
    t_orig = time.time() - t0
    print(f"  Done in {t_orig:.1f}s\n")

    # ---- Transcribe transformed audio ----
    print("─" * 70)
    print("Step 4: Transcribing TRANSFORMED (adult-like) speech")
    print("─" * 70)
    t0 = time.time()
    preds_transformed = transcribe_files(model, transformed_paths)
    t_trans = time.time() - t0
    print(f"  Done in {t_trans:.1f}s\n")

    # ---- Compute WER ----
    print("─" * 70)
    print("Step 5: RESULTS")
    print("─" * 70)

    valid_refs_o, valid_preds_o = [], []
    valid_refs_t, valid_preds_t = [], []

    for ref, pred_o, pred_t in zip(references, preds_original, preds_transformed):
        if not ref.strip():
            continue
        valid_refs_o.append(ref)
        valid_preds_o.append(pred_o if pred_o.strip() else "<empty>")
        valid_refs_t.append(ref)
        valid_preds_t.append(pred_t if pred_t.strip() else "<empty>")

    wer_original    = jiwer.wer(valid_refs_o, valid_preds_o)
    wer_transformed = jiwer.wer(valid_refs_t, valid_preds_t)
    delta           = wer_original - wer_transformed

    print(f"\n  Samples evaluated:  {len(valid_refs_o)}")
    print(f"  Settings:           pitch={PITCH_SEMITONES:+d}st  speed={SPEED_FACTOR}x")
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │  WER (Original child speech):   {wer_original:.4f}    │")
    print(f"  │  WER (Transformed adult-like):  {wer_transformed:.4f}    │")
    print(f"  │  Delta (positive = improvement): {delta:+.4f}   │")
    print(f"  └─────────────────────────────────────────────┘")

    if delta > 0.01:
        print(f"\n  ✅ TRANSFORM HELPS! WER improved by {delta:.4f} ({delta/wer_original*100:.1f}% relative)")
    elif delta < -0.01:
        print(f"\n  ❌ TRANSFORM HURTS. WER worsened by {abs(delta):.4f} ({abs(delta)/wer_original*100:.1f}% relative)")
        print(f"     Try different settings: PITCH_SEMITONES, SPEED_FACTOR")
    else:
        print(f"\n  ➖ Minimal difference. Try more aggressive settings.")

    # ---- Show sample predictions ----
    print(f"\n{'─' * 70}")
    print("Sample predictions (first 10):")
    print(f"{'─' * 70}")
    for i in range(min(10, len(valid_refs_o))):
        print(f"\n  [{i+1}] Reference:   {valid_refs_o[i]}")
        print(f"      Original:    {valid_preds_o[i]}")
        print(f"      Transformed: {valid_preds_t[i]}")
        wer_o = jiwer.wer([valid_refs_o[i]], [valid_preds_o[i]])
        wer_t = jiwer.wer([valid_refs_t[i]], [valid_preds_t[i]])
        if wer_t < wer_o:
            print(f"      → Transform better (WER {wer_o:.2f} → {wer_t:.2f})")
        elif wer_t > wer_o:
            print(f"      → Original better  (WER {wer_o:.2f} → {wer_t:.2f})")
        else:
            print(f"      → Same (WER {wer_o:.2f})")

    # ---- Next steps ----
    print(f"\n{'═' * 70}")
    print("NEXT STEPS:")
    print(f"{'═' * 70}")
    print("""
  If WER improved:
    → Use these settings in your training pipeline
    → Transform ALL training audio before fine-tuning

  If WER didn't improve:
    → Try a grid search over transform parameters:
      PITCH_SEMITONES: -2, -4, -6, -8
      SPEED_FACTOR:    1.00, 1.05, 1.08, 1.10, 1.15

    → Set env vars and re-run, e.g.:
      PITCH_SEMITONES=-4 SPEED_FACTOR=1.10
    """)

    print("DONE.")


if __name__ == "__main__":
    main()
'''

os.makedirs("/kaggle/working", exist_ok=True)
with open(SCRIPT_PATH, "w", encoding="utf-8") as f:
    f.write(script_code)
print(f"Script written to {SCRIPT_PATH}")


# =============================================================================
# PART C: FRESH SUBPROCESS
# =============================================================================
print("=" * 60)
print("Syntax check...")
_compile = subprocess.run(
    [sys.executable, "-m", "py_compile", SCRIPT_PATH],
    cwd="/kaggle/working",
    capture_output=True,
    text=True,
)
if _compile.returncode != 0:
    print(_compile.stderr or _compile.stdout)
    raise RuntimeError("Script failed py_compile")

print("Launching transform + evaluation in fresh subprocess...")
print("=" * 60)

result = subprocess.run(
    [sys.executable, SCRIPT_PATH],
    cwd="/kaggle/working",
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
    stdout=None,
    stderr=subprocess.STDOUT,
)

if result.returncode != 0:
    raise RuntimeError(f"Script exited with code {result.returncode}")

print("\nAll done. Check output above for WER comparison.")
