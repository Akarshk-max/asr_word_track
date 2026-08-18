#!/usr/bin/env python3
"""
Kaggle notebook driver: pip clean + copy training files + subprocess (fresh Python).

Mirrors the train_nemo_adapter_1.1b cell pattern for ``train_parakeet_age_adversarial.py``.

Set before running (typical):
  USE_CHAINED_ADAPTER=1
  TRAIN_MANIFEST=/path/to/train.jsonl
  SAVE_DIR=/kaggle/working/nemo_parakeet_age_adv

Optional:
  RUN_AGE_ADV_VERIFY=1  — run verify_age_adv_training_ready.py after training
  VERIFY_SCRIPT — override path to verify script

One notebook cell (pip + embed sources + subprocess): run ``python embed_parakeet_age_adv_one_cell.py``
locally to regenerate ``kaggle_parakeet_age_adv_one_cell.py``, then paste that file into a single Kaggle cell.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

os.environ.setdefault("USE_CHAINED_ADAPTER", "1")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
WORKING_TRAIN = os.environ.get("KAGGLE_TRAIN_SCRIPT", "/kaggle/working/train_parakeet_age_adversarial.py")
WORKING_EXTRAS = os.environ.get("KAGGLE_EXTRAS_SCRIPT", "/kaggle/working/parakeet_age_adv_nemo_extras.py")
WORKING_MERGE = os.environ.get(
    "KAGGLE_MERGE_SCRIPT", "/kaggle/working/merge_child_adult_training_manifests.py"
)


def _pip(*args: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def part_a_install() -> None:
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


def part_b_copy_scripts() -> None:
    dst_root = os.path.dirname(WORKING_TRAIN) or "/kaggle/working"
    os.makedirs(dst_root, exist_ok=True)
    src_train = os.path.join(REPO_ROOT, "train_parakeet_age_adversarial.py")
    src_extras = os.path.join(REPO_ROOT, "parakeet_age_adv_nemo_extras.py")
    src_merge = os.path.join(REPO_ROOT, "merge_child_adult_training_manifests.py")
    if not os.path.isfile(src_train):
        raise FileNotFoundError(f"Missing {src_train}")
    if not os.path.isfile(src_extras):
        raise FileNotFoundError(f"Missing {src_extras}")
    if not os.path.isfile(src_merge):
        raise FileNotFoundError(f"Missing {src_merge}")
    shutil.copy2(src_train, WORKING_TRAIN)
    shutil.copy2(src_extras, WORKING_EXTRAS)
    shutil.copy2(src_merge, WORKING_MERGE)
    print(f"Copied -> {WORKING_TRAIN}")
    print(f"Copied -> {WORKING_EXTRAS}")
    print(f"Copied -> {WORKING_MERGE}\n")


def part_c_subprocess_train() -> None:
    print("=" * 60)
    print("Launching train_parakeet_age_adversarial.py (fresh subprocess)")
    print("=" * 60)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": os.path.dirname(WORKING_TRAIN)}
    comp = subprocess.run(
        [sys.executable, "-m", "py_compile", WORKING_TRAIN, WORKING_EXTRAS, WORKING_MERGE],
        cwd=os.path.dirname(WORKING_TRAIN) or ".",
        capture_output=True,
        text=True,
    )
    if comp.returncode != 0:
        print(comp.stderr or comp.stdout)
        raise RuntimeError("py_compile failed for training scripts")
    result = subprocess.run(
        [sys.executable, WORKING_TRAIN],
        cwd=os.path.dirname(WORKING_TRAIN) or ".",
        env=env,
        stdout=None,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Training exited with code {result.returncode}")


def part_d_verify() -> None:
    if os.environ.get("RUN_AGE_ADV_VERIFY", "0").strip().lower() not in ("1", "true", "yes"):
        return
    vpath = os.environ.get("VERIFY_SCRIPT", os.path.join(REPO_ROOT, "verify_age_adv_training_ready.py"))
    if not os.path.isfile(vpath):
        print(f"WARN: VERIFY_SCRIPT not found: {vpath}")
        return
    shutil.copy2(vpath, "/kaggle/working/verify_age_adv_training_ready.py")
    subprocess.check_call(
        [sys.executable, "/kaggle/working/verify_age_adv_training_ready.py"],
        env={**os.environ, "PYTHONPATH": "/kaggle/working"},
    )


def main() -> None:
    skip_pip = os.environ.get("SKIP_KAGGLE_PIP", "0").strip().lower() in ("1", "true", "yes")
    if not skip_pip:
        part_a_install()
    part_b_copy_scripts()
    part_c_subprocess_train()
    part_d_verify()
    print("\nDone. Check SAVE_DIR (default ./nemo_parakeet_age_adv or /kaggle/working/...).")


if __name__ == "__main__":
    main()
