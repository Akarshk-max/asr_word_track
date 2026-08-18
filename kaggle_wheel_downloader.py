# ==========================================
# ONE-TIME WHEEL DOWNLOADER FOR KAGGLE
# ==========================================
# Run this ONCE in a Kaggle notebook with INTERNET ENABLED.
# It downloads all required packages (exact versions) as .whl files.
# After this cell completes:
#   1. Go to kaggle.com → Datasets → New Dataset
#   2. Upload the contents of /kaggle/working/nemo_wheels/
#   3. Name the dataset something like: nemo-asr-wheels
#   4. Make it PRIVATE, then add it to your training notebook
# From then on your training notebook needs NO internet.

import subprocess
import sys
import os
import shutil

WHEEL_DIR = "/kaggle/working/nemo_wheels"
os.makedirs(WHEEL_DIR, exist_ok=True)

# ─── EXACT VERSION CONSTRAINTS ───────────────────────────────────────────────
# These match the competition runtime spec:
#   Python 3.11 / CUDA 12.6 / manylinux2014_x86_64
# torch / torchaudio are downloaded separately (cu126 index).
# Everything else is downloaded from PyPI.
TORCH_PACKAGES = [
    "torch>=2.9.0",
    "torchaudio",
]

OTHER_PACKAGES = [
    "numpy>=2.1,<2.3",
    "scipy>=1.14,<1.16",
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
    "nemo_toolkit[asr]>=2.5.0",
]

# ─── Step 1: Download torch from cu126 index ─────────────────────────────────
print("⚡ Downloading torch + torchaudio (cu126)...")
subprocess.check_call([
    sys.executable, "-m", "pip", "download",
    "--no-cache-dir",
    "--dest", WHEEL_DIR,
    "--index-url", "https://download.pytorch.org/whl/cu126",
    *TORCH_PACKAGES,
])
print("  torch wheels done.\n")

# ─── Step 2: Download all other packages from PyPI ───────────────────────────
print("📦 Downloading all other packages (PyPI)...")
subprocess.check_call([
    sys.executable, "-m", "pip", "download",
    "--no-cache-dir",
    "--dest", WHEEL_DIR,
    *OTHER_PACKAGES,
])
print("  All dependency wheels done.\n")

# ─── Step 3: Report ──────────────────────────────────────────────────────────
wheel_files = sorted(os.listdir(WHEEL_DIR))
total_bytes = sum(
    os.path.getsize(os.path.join(WHEEL_DIR, f)) for f in wheel_files
)
print(f"✅ Downloaded {len(wheel_files)} files  ({total_bytes / 1e9:.2f} GB total)")
print(f"📁 Saved to: {WHEEL_DIR}")
print()
print("─── File list ───────────────────────────────────────────────────────────")
for f in wheel_files:
    sz = os.path.getsize(os.path.join(WHEEL_DIR, f)) / 1e6
    print(f"  {sz:7.1f} MB  {f}")

print()
print("─── NEXT STEPS ─────────────────────────────────────────────────────────")
print("1. Click the [Data] tab in this Kaggle notebook")
print("2. Click 'Upload' → 'New Dataset'")
print("3. Upload ALL files from /kaggle/working/nemo_wheels/")
print("4. Name it: nemo-asr-wheels  (private)")
print("5. Add this dataset to your training notebook")
print("6. In the training notebook PART A, use:")
print("   WHEEL_DIR = '/kaggle/input/nemo-asr-wheels/nemo_wheels'")
print("   (or adjust path to match the dataset slug Kaggle assigns)")
