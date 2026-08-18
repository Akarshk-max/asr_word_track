# -*- coding: utf-8 -*-
import os; os.environ["USE_EXISTING_MANIFESTS"] = "1"
"""
Paste this entire file into ONE Kaggle notebook cell (or run locally).

  Part A — pip install (clean env + NeMo)
  Part B — write training script to /kaggle/working
  Part C — subprocess.run (fresh Python interpreter)
  Part D (optional) — RUN_ADAPTER_VERIFY=1: verify_nemo_adapter_pipeline.py via subprocess

Pipeline (vNext — see train_nemo_adapter_parakeet_vnext.py):
  - Model: nvidia/parakeet-tdt-1.1b
  - LoRA on encoder layers 0–4 (attention + FFN linears); early adapters 0–7 two-bottleneck;
    layers 8+ chained single-bottleneck; LSTM-style memory; chain without .detach
  - Joint + full decoder trainable; five LR groups; DEBUG_SMALL_RUN caps train rows
  - External noise SNR 1–10 dB @ 0.6 prob; speed/pitch/gain; SpecAugment
  - Training only (no validation). Script loaded from repo file next to this cell or /kaggle/working.

Outputs: /kaggle/working/nemo_adapter_parakeet_vnext/ (copy train_nemo_adapter_parakeet_vnext.py to Kaggle)
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
        if proc.stdout:
            print("\n[pip stdout]\n" + proc.stdout[-4000:])
        if proc.stderr:
            print("\n[pip stderr]\n" + proc.stderr[-4000:])
        raise RuntimeError(f"pip failed with exit code {proc.returncode}")


def _install_numpy_scipy_with_fallback():
    try:
        _pip(
            "install",
            "--no-cache-dir",
            "--prefer-binary",
            "numpy>=2.1,<2.3",
            "scipy>=1.14,<1.16",
        )
        return
    except Exception as e:
        print(f"[warn] pinned numpy/scipy install failed: {e}")

    _pip(
        "install",
        "--no-cache-dir",
        "--prefer-binary",
        "numpy>=2.1",
        "scipy",
    )


# =============================================================================
# PART A: INSTALLATION (Kaggle-compatible pins; verbose pip errors on failure)
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

print("Step 2: pip toolchain + numpy + scipy...")
_pip("install", "--no-cache-dir", "--upgrade", "pip", "setuptools", "wheel")
_install_numpy_scipy_with_fallback()

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
# PART B: WRITE SELF-CONTAINED TRAINING SCRIPT
# =============================================================================
TRAIN_SCRIPT = "/kaggle/working/train_nemo_adapter_parakeet_vnext.py"


def _load_train_script_source():
    candidates = []
    if "__file__" in globals():
        candidates.append(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "train_nemo_adapter_parakeet_vnext.py",
            )
        )
    candidates.extend(
        [
            "/kaggle/working/train_nemo_adapter_parakeet_vnext.py",
            os.path.join(os.getcwd(), "train_nemo_adapter_parakeet_vnext.py"),
        ]
    )
    for path in candidates:
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return path, f.read()
    return None, None


_src_path, train_code = _load_train_script_source()
if train_code is None:
    raise FileNotFoundError(
        "train_nemo_adapter_parakeet_vnext.py not found. Copy ASR/train_nemo_adapter_parakeet_vnext.py "
        "to /kaggle/working or place next to this .py file in the repo."
    )
print(f"Loaded training script ({len(train_code)} chars) from: {_src_path}")


os.makedirs("/kaggle/working", exist_ok=True)
# Materialize on /kaggle/working so the subprocess always finds the same path.
with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
    f.write(train_code)

print(f"Training script written to {TRAIN_SCRIPT}\n")
print(
    "Optional env (set BEFORE this cell; child inherits):\n"
    "  USE_EXISTING_MANIFESTS — 1 to use pre-built manifests\n"
    "  TRAIN_MANIFEST — override train manifest path\n"
    "  BATCH_SIZE — default 64 (vNext)\n"
    "  DEBUG_SMALL_RUN — 1 (default) caps train manifest; DEBUG_TRAIN_SAMPLES=640\n"
    "  MODEL_ID, LORA_R, LORA_ALPHA, MEMORY_DIM, LR_LORA, LR_ADAPTER, LR_MEMORY, LR_JOINT, LR_DECODER\n"
    "  CLASSROOM_NOISE_DIRS — comma-separated paths to noise clip folders\n"
    "  CLASSROOM_NOISE_DIR_1 / CLASSROOM_NOISE_DIR_2 — two folders (defaults under /kaggle/input/)\n"
    "  RUN_ADAPTER_VERIFY — 1 to run verify_nemo_adapter_pipeline.py after training (subprocess)\n"
    "  VERIFY_SCRIPT — path to verify script (default /kaggle/working/verify_nemo_adapter_pipeline.py)\n"
    "  VERIFY_OUT — report directory (default /kaggle/working/adapter_verify_report)\n"
    "  VERIFY_QUICK — 1 to skip heavy dataloader checks\n"
    "  TRAIN_DEBUG_STEPS — N>0: log ALL chained-adapter forward shapes + grads for first N batches\n"
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
print("Check /kaggle/working/nemo_adapter_parakeet_vnext/ for outputs.")

# =============================================================================
# OPTIONAL Part D — adapter pipeline verify (FRESH SUBPROCESS, same safety as Part C)
# =============================================================================
# Do NOT %run verify_nemo_adapter_pipeline.py in the notebook kernel — that imports NeMo in
# IPython and can hit the same numpy/scipy breakage as pre-Part-C training.
# Set RUN_ADAPTER_VERIFY=1 (and copy verify_nemo_adapter_pipeline.py to /kaggle/working).
VERIFY_SCRIPT = os.environ.get(
    "VERIFY_SCRIPT",
    "/kaggle/working/verify_nemo_adapter_pipeline.py",
)
if os.environ.get("RUN_ADAPTER_VERIFY", "").strip().lower() in ("1", "true", "yes"):
    print("\n" + "=" * 60)
    print("Part D: Launching verify_nemo_adapter_pipeline.py in fresh subprocess")
    print("=" * 60)
    if not os.path.isfile(VERIFY_SCRIPT):
        print(
            f"SKIP: VERIFY_SCRIPT not found: {VERIFY_SCRIPT}\n"
            "Upload verify_nemo_adapter_pipeline.py to /kaggle/working/ or set VERIFY_SCRIPT."
        )
    else:
        _vout = os.environ.get("VERIFY_OUT", "/kaggle/working/adapter_verify_report")
        _vmanifest = os.environ.get("TRAIN_MANIFEST", "").strip()
        _vquick = os.environ.get("VERIFY_QUICK", "").strip().lower() in ("1", "true", "yes")
        _vcmd = [
            sys.executable,
            VERIFY_SCRIPT,
            "--out",
            _vout,
        ]
        if _vmanifest:
            _vcmd.extend(["--manifest", _vmanifest])
        if _vquick:
            _vcmd.append("--quick")
        _vc = subprocess.run(
            _vcmd,
            cwd="/kaggle/working",
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=None,
            stderr=subprocess.STDOUT,
        )
        if _vc.returncode != 0:
            raise RuntimeError(
                f"Verification subprocess exited with code {_vc.returncode}. "
                f"See log above; report under {_vout}/VERIFY_REPORT.md if partial."
            )
        print(f"Verification OK. Report: {_vout}/VERIFY_REPORT.md")
