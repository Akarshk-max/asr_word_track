"""
Rebuild ``kaggle_parakeet_age_adv_one_cell.py`` — one Kaggle notebook cell:

  Part A — pip install
  Part B — write extras + train + verify to working; inline merge when TRAIN_MANIFEST is comma-separated
  Part C — py_compile + subprocess train
  Part D — optional RUN_AGE_ADV_VERIFY

Run::

  python embed_parakeet_age_adv_one_cell.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_TRAIN = ROOT / "train_parakeet_age_adversarial.py"
SRC_EXTRAS = ROOT / "parakeet_age_adv_nemo_extras.py"
SRC_VERIFY = ROOT / "verify_age_adv_training_ready.py"
OUT = ROOT / "kaggle_parakeet_age_adv_one_cell.py"

# Ends before ``_CODE_EXTRAS = r'''`` (that fragment is concatenated in main — cannot live inside '''...''').
PREFIX = '''#!/usr/bin/env python3
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

os.environ.setdefault("USE_CHAINED_ADAPTER", "1")

WORK = os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working").strip() or "/kaggle/working"
PATH_EXTRAS = os.path.join(WORK, "parakeet_age_adv_nemo_extras.py")
PATH_TRAIN = os.path.join(WORK, "train_parakeet_age_adversarial.py")
PATH_VERIFY = os.path.join(WORK, "verify_age_adv_training_ready.py")


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
    print("Installation complete.\\n")
else:
    print("SKIP_KAGGLE_PIP=1 — skipping pip installs.\\n")

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
        return
    paths = [p.strip() for p in raw.split(",") if p.strip()]
    if len(paths) <= 1:
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
                    out.write(json.dumps(row, ensure_ascii=False) + "\\n")
                    written += 1
    if written == 0:
        raise RuntimeError(
            "Merged manifest is empty. Check file paths, JSONL rows (audio_filepath, text, duration, age_*)."
        )
    os.environ["TRAIN_MANIFEST"] = merged_path
    print(f"Merged {len(paths)} manifests -> {merged_path} ({written} rows)\\n", flush=True)


'''

SUFFIX = '''

with open(PATH_EXTRAS, "w", encoding="utf-8") as _f:
    _f.write(_CODE_EXTRAS)
with open(PATH_TRAIN, "w", encoding="utf-8") as _f:
    _f.write(_CODE_TRAIN)
with open(PATH_VERIFY, "w", encoding="utf-8") as _f:
    _f.write(_CODE_VERIFY)

print(f"Wrote {PATH_EXTRAS}")
print(f"Wrote {PATH_TRAIN}")
print(f"Wrote {PATH_VERIFY}\\n")

_merge_train_manifests_if_needed()

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
    stdout=None,
    stderr=subprocess.STDOUT,
)
if _result.returncode != 0:
    raise RuntimeError(f"Training exited with code {_result.returncode}")

if os.environ.get("RUN_AGE_ADV_VERIFY", "0").strip().lower() in ("1", "true", "yes"):
    print("\\n" + "=" * 60)
    print("RUN_AGE_ADV_VERIFY=1 — verify_age_adv_training_ready.py")
    print("=" * 60)
    subprocess.check_call(
        [sys.executable, PATH_VERIFY],
        cwd=WORK,
        env=_env,
    )

print("\\nDone. Check SAVE_DIR for checkpoints.")
'''


_VERIFY_STUB = '''#!/usr/bin/env python3
def main() -> int:
    print("SKIP: verify_age_adv_training_ready.py was not bundled; add file to repo and re-run embed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def main() -> None:
    if not SRC_TRAIN.is_file() or not SRC_EXTRAS.is_file():
        raise SystemExit(f"Missing {SRC_TRAIN} or {SRC_EXTRAS}")
    extras = SRC_EXTRAS.read_text(encoding="utf-8")
    train = SRC_TRAIN.read_text(encoding="utf-8")
    verify = SRC_VERIFY.read_text(encoding="utf-8") if SRC_VERIFY.is_file() else _VERIFY_STUB
    for label, text in (
        ("parakeet_age_adv_nemo_extras.py", extras),
        ("train_parakeet_age_adversarial.py", train),
        ("verify_age_adv_training_ready.py", verify),
    ):
        if text and "'''" in text:
            raise SystemExit(
                f"{label} contains ''' — remove/replace triple-single-quoted strings in source, "
                "or change this embedder to use base64."
            )

    middle = (
        "_CODE_EXTRAS = r'''"
        + extras
        + "'''\n\n_CODE_TRAIN = r'''"
        + train
        + "'''\n\n_CODE_VERIFY = r'''"
        + verify
        + "'''\n\n"
    )
    out = PREFIX + middle + SUFFIX
    OUT.write_text(out, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
