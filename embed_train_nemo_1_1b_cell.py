"""Rebuild train_nemo_adapter_1.1b.py (Kaggle one-cell) from train_nemo_adapter_1.1b_worker.py."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKER = ROOT / "train_nemo_adapter_1.1b_worker.py"
OUT = ROOT / "train_nemo_adapter_1.1b.py"

HEADER = '''import os
import subprocess
import sys

os.environ.setdefault("USE_EXISTING_MANIFESTS", "1")
os.environ.setdefault("TRAIN_DEBUG_STEPS", "0")
"""
Paste this entire file into ONE Kaggle notebook cell (or run locally).

  Part A — pip install (clean env + NeMo)
  Part B — write training script to /kaggle/working/train_nemo_adapter_1.1b_train.py
  Part C — subprocess.run (fresh Python interpreter)

Pipeline (methodology):
  - nvidia/parakeet-tdt-1.1b; ChainedLinearAdapter bottleneck dim=256 (BOTTLENECK_DIM); 7 epochs (NUM_EPOCHS)
  - Waveform aug + noise curriculum (vNext-style); SpecAugment; Whisper label normalizer
  - Manifest: env TRAIN_MANIFEST, fixed Kaggle paths, then glob **/train_manifest.jsonl under /kaggle/input
  - Joint + decoder embed + last LSTM; discriminative AdamW; train only (no validation)

Outputs: /kaggle/working/nemo_adapter_1.1b/
"""

# Part B mirrors train_nemo_adapter_1.1b_worker.py — edit that file, then run:
#   python embed_train_nemo_1_1b_cell.py


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

print("Installation complete.\\n")

# =============================================================================
# PART B: WRITE TRAINING SCRIPT
# =============================================================================
TRAIN_SCRIPT = "/kaggle/working/train_nemo_adapter_1.1b_train.py"

train_code = r\'\'\'
'''

FOOTER = '''\'\'\'

os.makedirs("/kaggle/working", exist_ok=True)
with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
    f.write(train_code)

print(f"Training script written to {TRAIN_SCRIPT}\\n")
print(
    "Optional env (child inherits): USE_EXISTING_MANIFESTS, TRAIN_MANIFEST, BATCH_SIZE, "
    "BOTTLENECK_DIM, NUM_EPOCHS, CLASSROOM_NOISE_DIRS, NOISE_CURRICULUM, TRAIN_DEBUG_STEPS, SAVE_DIR, …\\n"
)

# =============================================================================
# PART C: FRESH SUBPROCESS
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
    print("Syntax error in training script (py_compile failed):\\n")
    print(_compile.stderr or _compile.stdout)
    raise RuntimeError("train script failed py_compile")

result = subprocess.run(
    [sys.executable, TRAIN_SCRIPT],
    cwd="/kaggle/working",
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
    stdout=None,
    stderr=subprocess.STDOUT,
)

if result.returncode != 0:
    raise RuntimeError(f"Training failed with exit code {result.returncode}")

print("\\nTraining completed successfully.")
print("Check /kaggle/working/nemo_adapter_1.1b/ for outputs.")
'''


def main() -> None:
    worker_src = WORKER.read_text(encoding="utf-8")
    if "'''" in worker_src:
        raise SystemExit("worker must not contain triple-single-quotes (breaks train_code r'''…''')")
    out = HEADER + worker_src + FOOTER
    OUT.write_text(out, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes) from {WORKER}")


if __name__ == "__main__":
    main()
