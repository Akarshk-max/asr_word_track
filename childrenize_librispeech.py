#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
childrenize_librispeech.py
==========================
Convert LibriSpeech adult speech into child-like synthetic speech using the
WORLD-vocoder "childrenization" method.

Pipeline:
    1. Load .flac audio → resample to 16 kHz mono
    2. (Optional) speech enhancement / denoising placeholder
    3. WORLD vocoder decomposition (F0, spectral envelope, aperiodicity)
    4. Spectral warping (linear for male, piecewise for female)
    5. F0 shifting toward child range [240–300 Hz]
    6. Vowel-length stretching on voiced segments
    7. WORLD resynthesis → save .wav

Reference: https://github.com/zhao-shuyang/childrenize
Reimplemented cleanly with multiprocessing, resume, and error handling.

Usage:
    python childrenize_librispeech.py \
        --input_dir  LibriSpeech/train-other-500 \
        --output_dir child_synthetic_librispeech \
        --num_workers 4

Dependencies:
    pip install pyworld soundfile librosa numpy scipy tqdm
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import traceback
from multiprocessing import Pool, cpu_count, get_context
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:
    librosa = None  # fallback: scipy.signal.resample

try:
    import pyworld as pw
except ImportError:
    raise ImportError(
        "pyworld is required.  Install with: pip install pyworld"
    )

from scipy.interpolate import InterpolatedUnivariateSpline
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

TARGET_SR: int = 16_000          # output sample-rate
EPSILON: float = 1e-8
F0_MIN: float = 50.0            # ignore unvoiced / low-F0 frames
F0_MAX: float = 600.0           # hard cap on synthesized F0
GENDER_F0_THRESHOLD: float = 160.0  # mean F0 > 160 → female

# Randomization ranges
TARGET_F0_RANGE: Tuple[float, float] = (240.0, 300.0)
MALE_ALPHA_RANGE: Tuple[float, float] = (1.2, 1.4)
FEMALE_ALPHA_RANGE: Tuple[float, float] = (1.1, 1.25)
VOWEL_STRETCH_RANGE: Tuple[float, float] = (1.1, 1.4)

# Piecewise warping breakpoints (Hz)
PIECEWISE_LOW_CUTOFF: float = 300.0
PIECEWISE_HIGH_CUTOFF: float = 5500.0

# WORLD frame period (ms) — default used in reference
FRAME_PERIOD_MS: float = 5.0

logger = logging.getLogger("childrenize")

# ═══════════════════════════════════════════════════════════════════════════
# §1  Audio I/O Utilities
# ═══════════════════════════════════════════════════════════════════════════

def load_audio(
    path: str | Path,
    target_sr: int = TARGET_SR,
) -> Tuple[np.ndarray, int]:
    """Load audio file, convert to mono float64, resample to *target_sr*.

    Returns
    -------
    audio : np.ndarray   — shape (n_samples,), dtype float64
    sr    : int           — always *target_sr*
    """
    audio, sr = sf.read(str(path), dtype="float64")

    # stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # resample if necessary
    if sr != target_sr:
        if librosa is not None:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=target_sr
            )
        else:
            # fallback: scipy
            from scipy.signal import resample as scipy_resample
            n_target = int(len(audio) * target_sr / sr)
            audio = scipy_resample(audio, n_target)
        sr = target_sr

    return audio.astype(np.float64, copy=False), sr

def save_audio(
    path: str | Path,
    audio: np.ndarray,
    sr: int = TARGET_SR,
) -> None:
    """Write waveform to 16-bit PCM .wav."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")

def discover_flac_files(root: Path) -> List[Path]:
    """Recursively find all .flac files, sorted for determinism."""
    return sorted(root.rglob("*.flac"))

def output_wav_path(
    input_root: Path,
    output_root: Path,
    flac_path: Path,
) -> Path:
    """Mirror the directory structure, replacing .flac → .wav."""
    rel = flac_path.resolve().relative_to(input_root.resolve())
    return (output_root / rel).with_suffix(".wav")

# ═══════════════════════════════════════════════════════════════════════════
# §1b  LibriSpeech Transcript Loader
# ═══════════════════════════════════════════════════════════════════════════

def load_transcripts(root: Path) -> Dict[str, str]:
    """Parse all LibriSpeech .trans.txt files into {utterance_id: text}.

    LibriSpeech layout:  speaker/chapter/speaker-chapter.trans.txt
    Each line:           <utterance_id> <transcript text>
    """
    transcripts: Dict[str, str] = {}
    for trans_file in root.rglob("*.trans.txt"):
        try:
            with trans_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        utt_id, text = parts
                        transcripts[utt_id] = text
        except Exception as e:
            logger.warning("Failed to parse %s: %s", trans_file, e)
    return transcripts

def get_utterance_id(flac_path: Path) -> str:
    """Extract utterance ID from a LibriSpeech .flac filename.

    e.g.  /data/1234/5678/1234-5678-0001.flac  →  '1234-5678-0001'
    """
    return flac_path.stem

# ═══════════════════════════════════════════════════════════════════════════
# §2  (Optional) Speech Enhancement / Denoising
# ═══════════════════════════════════════════════════════════════════════════

def denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Optional denoising step (placeholder).

    Currently a no-op.  Replace with noisereduce, DeepFilterNet, etc.
    when --enable_denoise is passed.

    To integrate e.g. noisereduce:
        import noisereduce as nr
        return nr.reduce_noise(y=audio, sr=sr)
    """
    return audio

# ═══════════════════════════════════════════════════════════════════════════
# §3  WORLD Vocoder Decomposition & Resynthesis
# ═══════════════════════════════════════════════════════════════════════════

def world_decompose(
    audio: np.ndarray,
    sr: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract F0, spectral envelope, and aperiodicity via WORLD.

    Returns
    -------
    f0        : (T,)    fundamental frequency per frame
    sp        : (T, D)  spectral envelope
    ap        : (T, D)  aperiodicity
    timeaxis  : (T,)    frame centre times
    """
    # Harvest gives raw F0 + time axis
    _f0, timeaxis = pw.harvest(
        audio, sr,
        f0_floor=F0_MIN,
        f0_ceil=F0_MAX,
        frame_period=FRAME_PERIOD_MS,
    )
    # Stonemask refines the F0 estimate
    f0 = pw.stonemask(audio, _f0, timeaxis, sr)
    # Spectral envelope
    sp = pw.cheaptrick(audio, f0, timeaxis, sr)
    # Aperiodicity
    ap = pw.d4c(audio, f0, timeaxis, sr)

    return f0, sp, ap, timeaxis

def synthesize_segments(
    f0: np.ndarray,
    sp: np.ndarray,
    ap: np.ndarray,
    sr: int,
    vowel_stretch_factor: float = 1.0,
) -> np.ndarray:
    """Segment-wise resynthesis with vowel stretching.

    Voiced (harmonic) segments are synthesized with a stretched frame period
    to lengthen vowels, while unvoiced segments keep the original timing.
    """
    voiced_mask = f0 != 0

    # Find boundaries where voicing changes
    change_points = np.nonzero(voiced_mask != np.roll(voiced_mask, 1))[0] + 1
    if len(change_points) > 1 and change_points[-1] >= len(voiced_mask):
        change_points = change_points[:-1]

    # Determine voicing label of each segment
    segment_starts = np.insert(change_points, 0, 0)
    is_voiced = voiced_mask[segment_starts]
    n_segs = len(is_voiced)

    # Split arrays at change-points
    f0_segs = np.split(f0, change_points)
    sp_segs = np.split(sp, change_points)
    ap_segs = np.split(ap, change_points)

    wav_segs: List[np.ndarray] = []
    for i in range(n_segs):
        if is_voiced[i]:
            # Slower frame rate → stretched vowels
            frame_period = FRAME_PERIOD_MS * vowel_stretch_factor
        else:
            frame_period = FRAME_PERIOD_MS

        seg = pw.synthesize(
            f0_segs[i].astype(np.float64),
            sp_segs[i].astype(np.float64),
            ap_segs[i].astype(np.float64),
            sr,
            frame_period,
        )
        wav_segs.append(seg)

    return np.concatenate(wav_segs) if wav_segs else np.array([], dtype=np.float64)

# ═══════════════════════════════════════════════════════════════════════════
# §4  Spectral Warping  (formant shift)
# ═══════════════════════════════════════════════════════════════════════════

def _linear_warping_fn(alpha: float):
    """Linear frequency scaling:  f' = α·f"""
    def fn(f: float) -> float:
        return alpha * f
    return fn

def _piecewise_warping_fn(beta: float, fs: int):
    """Three-band piecewise frequency warping.

    Band       | Input range         | Scale
    -----------|---------------------|------
    Low        | [0, 300]            | β²
    Mid        | [300, 5500]         | β
    High       | [5500, fs/2]        | adaptive (map to [F_high, fs/2])

    This shifts low/mid formants upward (child-like) while keeping high
    frequencies within the Nyquist limit.
    """
    low_cut = PIECEWISE_LOW_CUTOFF
    high_cut = PIECEWISE_HIGH_CUTOFF
    beta_low = beta ** 2
    F_low = low_cut * beta_low
    F_high = F_low + beta * (high_cut - low_cut)
    nyquist = fs / 2.0
    beta_high = (nyquist - F_high) / (nyquist - high_cut)

    def fn(f: float) -> float:
        if f <= low_cut:
            return beta_low * f
        elif f <= high_cut:
            return F_low + beta * (f - low_cut)
        else:
            return F_high + beta_high * (f - high_cut)

    return fn

def warp_spectral_envelope(
    sp: np.ndarray,
    alpha: float,
    fs: int,
    mode: str = "linear",
) -> np.ndarray:
    """Warp the spectral envelope along the frequency axis.

    Parameters
    ----------
    sp    : (T, D) spectral envelope from WORLD
    alpha : warping factor (e.g. 1.3 for male linear)
    fs    : sample rate
    mode  : 'linear' or 'piecewise'

    Returns
    -------
    sp_warped : (T, D) warped spectral envelope
    """
    N, D = sp.shape
    sp_out = sp.copy()

    half_fft = D - 1
    # Frequency bins (skip DC bin at index 0)
    f = np.arange(1, half_fft + 1) / half_fft * (fs // 2)

    # Build warping function
    if mode == "linear":
        warp_fn = _linear_warping_fn(alpha)
    elif mode == "piecewise":
        warp_fn = _piecewise_warping_fn(alpha, fs)
    else:
        raise ValueError(f"Unknown warping mode: {mode!r}")

    # Compute warped frequency axis
    f_warped = np.array([warp_fn(fi) for fi in f])

    # Interpolate each frame's spectral envelope onto the original freq grid
    for frame_i in range(N):
        sp_slice = sp[frame_i, 1:]  # exclude DC
        interp_fn = InterpolatedUnivariateSpline(f_warped, sp_slice, k=1)
        sp_out[frame_i, 1:] = interp_fn(f)

    return sp_out

# ═══════════════════════════════════════════════════════════════════════════
# §5  F0 (Pitch) Shifting
# ═══════════════════════════════════════════════════════════════════════════

def shift_f0(
    f0: np.ndarray,
    target_mean: float,
    f0_min: float = F0_MIN,
    f0_max: float = F0_MAX,
) -> np.ndarray:
    """Shift F0 so voiced-frame mean equals *target_mean*.

    Unvoiced frames (f0 == 0 or f0 < f0_min) stay at 0.
    Voiced frames are shifted by (target_mean - original_mean),
    then clamped to [f0_min, f0_max].
    """
    f0_new = f0.copy()
    voiced = f0 > f0_min

    if not np.any(voiced):
        return f0_new  # entirely unvoiced — nothing to shift

    original_mean = np.mean(f0[voiced])
    delta = target_mean - original_mean

    f0_new[voiced] = f0[voiced] + delta
    # Clamp: no negatives, cap at max
    f0_new[f0_new < 0] = 0.0
    f0_new[f0_new > f0_max] = f0_max
    # Preserve unvoiced
    f0_new[~voiced] = 0.0

    return f0_new

# ═══════════════════════════════════════════════════════════════════════════
# §6  Parameter Randomization
# ═══════════════════════════════════════════════════════════════════════════

def randomize_params(
    f0: np.ndarray,
    rng: np.random.Generator | None = None,
) -> Dict[str, object]:
    """Choose random childrenization parameters based on estimated gender.

    Returns dict with keys:
        target_f0, warping_factor, warping_mode, vowel_stretch_factor
    """
    if rng is None:
        rng = np.random.default_rng()

    voiced = f0[f0 > F0_MIN]
    if len(voiced) == 0:
        f0_mean = 0.0
    else:
        f0_mean = float(np.mean(voiced))

    params: Dict[str, object] = {}

    # Target child F0
    params["target_f0"] = float(rng.uniform(*TARGET_F0_RANGE))

    # Vowel stretch factor
    params["vowel_stretch_factor"] = float(rng.uniform(*VOWEL_STRETCH_RANGE))

    # Gender-dependent spectral warping
    if f0_mean < GENDER_F0_THRESHOLD:
        # Male → linear warping with larger factor
        params["warping_mode"] = "linear"
        params["warping_factor"] = float(rng.uniform(*MALE_ALPHA_RANGE))
    else:
        # Female → piecewise warping with smaller factor
        params["warping_mode"] = "piecewise"
        params["warping_factor"] = float(rng.uniform(*FEMALE_ALPHA_RANGE))

    return params

# ═══════════════════════════════════════════════════════════════════════════
# §7  Full Childrenization Transform
# ═══════════════════════════════════════════════════════════════════════════

def childrenize(
    audio: np.ndarray,
    sr: int,
    params: Dict[str, object] | None = None,
    rng: np.random.Generator | None = None,
    enable_denoise: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """End-to-end childrenization of a single utterance.

    Parameters
    ----------
    audio          : mono float64 waveform
    sr             : sample rate (should be 16 kHz)
    params         : dict of transform params; randomized if None
    rng            : numpy RNG for reproducibility
    enable_denoise : whether to apply the denoising step

    Returns
    -------
    audio_out : transformed waveform (float64)
    params    : dict of transform params actually used
    """
    # ── 0. Optional denoising ──────────────────────────────────────────
    if enable_denoise:
        audio = denoise(audio, sr)

    # ── 1. WORLD decomposition ─────────────────────────────────────────
    f0, sp, ap, _ = world_decompose(audio, sr)

    # ── 2. Randomize parameters if not supplied ────────────────────────
    if params is None:
        params = randomize_params(f0, rng=rng)

    # ── 3. Spectral warping (formant shift) ────────────────────────────
    sp_warped = warp_spectral_envelope(
        sp,
        alpha=params["warping_factor"],
        fs=sr,
        mode=params["warping_mode"],
    )

    # ── 4. F0 shifting ────────────────────────────────────────────────
    f0_shifted = shift_f0(f0, target_mean=params["target_f0"])

    # ── 5. Resynthesis with vowel stretching ──────────────────────────
    audio_out = synthesize_segments(
        f0_shifted, sp_warped, ap, sr,
        vowel_stretch_factor=params["vowel_stretch_factor"],
    )

    return audio_out, params

# ═══════════════════════════════════════════════════════════════════════════
# §8  Multiprocessing Worker
# ═══════════════════════════════════════════════════════════════════════════

# Per-worker state (initialised once per child process)
_WORKER_RNG: Optional[np.random.Generator] = None
_WORKER_DENOISE: bool = False
_WORKER_TRANSCRIPTS: Dict[str, str] = {}
_WORKER_INPUT_ROOT: str = ""
_WORKER_OUTPUT_ROOT: str = ""

def _worker_init(
    seed: int,
    enable_denoise: bool,
    transcripts: Dict[str, str],
    input_root: str,
    output_root: str,
) -> None:
    """Initialise per-worker RNG with a unique seed."""
    global _WORKER_RNG, _WORKER_DENOISE, _WORKER_TRANSCRIPTS
    global _WORKER_INPUT_ROOT, _WORKER_OUTPUT_ROOT
    _WORKER_RNG = np.random.default_rng(seed + os.getpid())
    _WORKER_DENOISE = enable_denoise
    _WORKER_TRANSCRIPTS = transcripts
    _WORKER_INPUT_ROOT = input_root
    _WORKER_OUTPUT_ROOT = output_root

def _process_one(
    args: Tuple[str, str],
) -> Tuple[str, Optional[str], Optional[dict]]:
    """Process a single file.

    Returns (rel_path, error_or_None, metadata_dict_or_None).
    """
    src_path, dst_path = args
    rel = src_path  # used for logging

    try:
        # Skip if output already exists (resume-safe)
        dst = Path(dst_path)
        if dst.is_file() and dst.stat().st_size > 0:
            # Still produce metadata for already-done files
            info = sf.info(str(dst))
            utt_id = get_utterance_id(Path(src_path))
            out_rel = str(Path(dst_path).resolve().relative_to(
                Path(_WORKER_OUTPUT_ROOT).resolve()
            ))
            meta = {
                "audio_filepath": out_rel.replace("\\", "/"),
                "duration": info.duration,
                "text": _WORKER_TRANSCRIPTS.get(utt_id, ""),
                "speaker_id": utt_id.split("-")[0] if "-" in utt_id else "",
                "original_filepath": str(
                    Path(src_path).resolve().relative_to(
                        Path(_WORKER_INPUT_ROOT).resolve()
                    )
                ).replace("\\", "/"),
            }
            return (rel, None, meta)

        # Load
        audio, sr = load_audio(src_path)
        if len(audio) == 0:
            return (rel, "zero-length audio -- skipped", None)

        # Transform
        audio_out, params_used = childrenize(
            audio, sr,
            rng=_WORKER_RNG,
            enable_denoise=_WORKER_DENOISE,
        )

        # Save
        save_audio(dst_path, audio_out, sr)

        # Build metadata record
        utt_id = get_utterance_id(Path(src_path))
        out_rel = str(Path(dst_path).resolve().relative_to(
            Path(_WORKER_OUTPUT_ROOT).resolve()
        ))
        meta = {
            "audio_filepath": out_rel.replace("\\", "/"),
            "duration": round(len(audio_out) / sr, 4),
            "text": _WORKER_TRANSCRIPTS.get(utt_id, ""),
            "speaker_id": utt_id.split("-")[0] if "-" in utt_id else "",
            "original_filepath": str(
                Path(src_path).resolve().relative_to(
                    Path(_WORKER_INPUT_ROOT).resolve()
                )
            ).replace("\\", "/"),
            "target_f0": round(float(params_used["target_f0"]), 2),
            "warping_mode": params_used["warping_mode"],
            "warping_factor": round(float(params_used["warping_factor"]), 4),
            "vowel_stretch_factor": round(float(params_used["vowel_stretch_factor"]), 4),
        }
        return (rel, None, meta)

    except Exception:
        return (rel, traceback.format_exc(), None)

# ═══════════════════════════════════════════════════════════════════════════
# §9  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Childrenize LibriSpeech: convert adult speech to child-like "
            "speech via WORLD-vocoder spectral warping, F0 shifting, "
            "and vowel stretching."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Batch splitting examples (for multiple Kaggle sessions):\n"
            "  Session 1:  --batch_index 0 --batch_total 3\n"
            "  Session 2:  --batch_index 1 --batch_total 3\n"
            "  Session 3:  --batch_index 2 --batch_total 3\n"
            "\n"
            "If the corpus has exactly 10k sorted .flac files, second half:\n"
            "  --batch_index 1 --batch_total 2  (omit --max_files)\n"
            "\n"
            "Subprocess spawning (auto-splits into N independent processes):\n"
            "  --num_subprocesses 4  (spawns 4 python subprocesses, each\n"
            "   processing 1/4 of the data with --num_workers cores each)\n"
            "\n"
            "Fixed sorted window (e.g. files 5000–9999 on full train-clean-100):\n"
            "  --skip_files 5000 --max_files 5000"
        ),
    )
    p.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Root of LibriSpeech (e.g. LibriSpeech/train-other-500).",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("child_synthetic_librispeech"),
        help="Output directory (default: child_synthetic_librispeech).",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (cpu_count() or 4)),
        help="Parallel workers per process (default: all CPUs).",
    )
    p.add_argument(
        "--enable_denoise",
        action="store_true",
        help="Enable speech enhancement before processing.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed for reproducibility (default: 42).",
    )
    # ── Batch splitting (for multiple Kaggle sessions) ─────────────────
    p.add_argument(
        "--batch_index",
        type=int,
        default=None,
        metavar="I",
        help="Process only batch I of batch_total (0-indexed). "
             "Use with --batch_total for multi-session parallelism.",
    )
    p.add_argument(
        "--batch_total",
        type=int,
        default=None,
        metavar="N",
        help="Total number of batches to split the dataset into.",
    )
    # ── Subprocess spawning ────────────────────────────────────────────
    p.add_argument(
        "--num_subprocesses",
        type=int,
        default=0,
        metavar="N",
        help="Spawn N independent Python subprocesses, each handling 1/N "
             "of the data. 0 = disabled (default). Great for maximizing "
             "throughput on multi-core machines.",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=0,
        help="Pool imap chunksize (0 = auto-tune based on dataset size).",
    )
    p.add_argument(
        "--skip_files",
        type=int,
        default=0,
        metavar="K",
        help=(
            "Skip the first K sorted .flac files after discovery (default: 0). "
            "Use with --max_files for a fixed window, e.g. --skip_files 5000 "
            "--max_files 5000 for sorted indices 5000–9999."
        ),
    )
    p.add_argument(
        "--max_files",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N files (sorted). Example: --max_files 5000",
    )
    return p.parse_args(argv)

# ═══════════════════════════════════════════════════════════════════════════
# §10b  Subprocess Spawner
# ═══════════════════════════════════════════════════════════════════════════

def run_subprocesses(args: argparse.Namespace) -> int:
    """Spawn N independent Python subprocesses, each handling 1/N of data.

    This is the fastest mode: each subprocess gets its own pool of workers,
    completely avoiding GIL contention and maximizing I/O + CPU overlap.
    """
    n_sub = args.num_subprocesses
    logger.info("Spawning %d subprocesses...", n_sub)

    # Build base command
    script = os.path.abspath(__file__)
    base_cmd = [
        sys.executable, script,
        "--input_dir", str(args.input_dir),
        "--output_dir", str(args.output_dir),
        "--num_workers", str(max(1, args.num_workers // n_sub)),
        "--seed", str(args.seed),
        "--num_subprocesses", "0",  # don't recurse
    ]
    if args.enable_denoise:
        base_cmd.append("--enable_denoise")
    if args.max_files is not None:
        base_cmd.extend(["--max_files", str(args.max_files)])
    if args.skip_files:
        base_cmd.extend(["--skip_files", str(args.skip_files)])

    # Launch each subprocess with its batch slice
    procs = []
    for i in range(n_sub):
        cmd = base_cmd + [
            "--batch_index", str(i),
            "--batch_total", str(n_sub),
        ]
        logger.info("  Subprocess %d/%d: batch %d of %d", i + 1, n_sub, i, n_sub)
        p = subprocess.Popen(
            cmd,
            # Let BOTH stdout and stderr flow to the terminal.
            # If stdout=subprocess.PIPE is used and not read continuously,
            # the OS pipe buffer fills up and blocks the subprocess,
            # making them run sequentially instead of in parallel.
            stdout=None,
            stderr=None,
        )
        procs.append((i, p))

    # Wait for all subprocesses
    any_failed = False
    for i, p in procs:
        p.wait()
        rc = p.returncode
        if rc != 0:
            any_failed = True
            logger.error("Subprocess %d failed (rc=%d)", i, rc)
        else:
            logger.info("Subprocess %d finished successfully.", i)

    # Merge partial manifests into one
    _merge_partial_manifests(args.output_dir.resolve())

    return 2 if any_failed else 0

def _merge_partial_manifests(output_root: Path) -> None:
    """Merge all manifest_batch_*.jsonl into a single manifest.jsonl."""
    partials = sorted(output_root.glob("manifest_batch_*.jsonl"))
    if not partials:
        logger.warning("No partial manifests found to merge.")
        return

    all_records: List[dict] = []
    for pf in partials:
        with pf.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))

    # Deduplicate by audio_filepath (in case of overlap)
    seen = set()
    unique = []
    for r in all_records:
        key = r.get("audio_filepath", "")
        if key not in seen:
            seen.add(key)
            unique.append(r)

    unique.sort(key=lambda m: m["audio_filepath"])

    merged_path = output_root / "manifest.jsonl"
    with merged_path.open("w", encoding="utf-8") as mf:
        for record in unique:
            mf.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Merged %d partial manifests -> %s (%d entries)",
        len(partials), merged_path, len(unique),
    )

    # Clean up partials
    for pf in partials:
        pf.unlink()
    logger.info("Removed %d partial manifest files.", len(partials))

# ═══════════════════════════════════════════════════════════════════════════
# §10  Main
# ═══════════════════════════════════════════════════════════════════════════

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    # ── Subprocess spawning mode ───────────────────────────────────────
    if args.num_subprocesses > 1:
        return run_subprocesses(args)

    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()

    if not input_root.is_dir():
        logger.error("input_dir is not a directory: %s", input_root)
        return 1

    # ── Discover files ─────────────────────────────────────────────────
    logger.info("Scanning for .flac files under %s ...", input_root)
    all_flac = discover_flac_files(input_root)
    if not all_flac:
        logger.error("No .flac files found under %s", input_root)
        return 1
    logger.info("Found %d .flac files", len(all_flac))

    # ── Skip leading files (offset in sorted list) ────────────────────
    if args.skip_files:
        if args.skip_files < 0:
            logger.error("skip_files must be >= 0, got %d", args.skip_files)
            return 1
        if args.skip_files >= len(all_flac):
            logger.error(
                "skip_files (%d) >= total files (%d)",
                args.skip_files,
                len(all_flac),
            )
            return 1
        all_flac = all_flac[args.skip_files:]
        logger.info(
            "Skipped first %d files (--skip_files); %d remaining",
            args.skip_files,
            len(all_flac),
        )

    # ── Limit file count ───────────────────────────────────────────────
    if args.max_files is not None and args.max_files < len(all_flac):
        all_flac = all_flac[:args.max_files]
        logger.info("Limited to first %d files (--max_files)", args.max_files)

    # ── Batch slicing (for multi-session / subprocess splits) ──────────
    if args.batch_index is not None and args.batch_total is not None:
        total = args.batch_total
        idx = args.batch_index
        if idx < 0 or idx >= total:
            logger.error("batch_index %d out of range [0, %d)", idx, total)
            return 1
        chunk_size = math.ceil(len(all_flac) / total)
        start = idx * chunk_size
        end = min(start + chunk_size, len(all_flac))
        all_flac = all_flac[start:end]
        logger.info(
            "Batch %d/%d: processing files %d-%d (%d files)",
            idx, total, start, end - 1, len(all_flac),
        )

    # ── Load transcripts ───────────────────────────────────────────────
    logger.info("Loading transcripts from .trans.txt files...")
    transcripts = load_transcripts(input_root)
    logger.info("Loaded %d transcripts", len(transcripts))

    # ── Build task list ────────────────────────────────────────────────
    tasks: List[Tuple[str, str]] = []
    for fp in all_flac:
        dst = output_wav_path(input_root, output_root, fp)
        tasks.append((str(fp), str(dst)))

    logger.info("Total files to handle: %d", len(tasks))

    if not tasks:
        logger.info("Nothing to do.")
        return 0

    # ── Error log & output dir ─────────────────────────────────────────
    error_log = output_root / "childrenize_errors.log"
    output_root.mkdir(parents=True, exist_ok=True)

    # Use batch-specific manifest name if in batch mode
    if args.batch_index is not None:
        manifest_path = output_root / f"manifest_batch_{args.batch_index}.jsonl"
    else:
        manifest_path = output_root / "manifest.jsonl"

    ok_count = 0
    err_count = 0
    all_meta: List[dict] = []

    # ── Dynamic chunksize ──────────────────────────────────────────────
    num_workers = max(1, args.num_workers)
    if args.chunksize > 0:
        chunksize = args.chunksize
    else:
        # Auto-tune: larger chunks reduce IPC overhead
        chunksize = max(1, min(32, len(tasks) // (num_workers * 4)))
    logger.info(
        "Workers: %d, chunksize: %d, spawn context: fork-safe",
        num_workers, chunksize,
    )

    # ── Process ────────────────────────────────────────────────────────
    if num_workers <= 1:
        # Single-process mode
        global _WORKER_RNG, _WORKER_DENOISE, _WORKER_TRANSCRIPTS
        global _WORKER_INPUT_ROOT, _WORKER_OUTPUT_ROOT
        _WORKER_RNG = np.random.default_rng(args.seed)
        _WORKER_DENOISE = args.enable_denoise
        _WORKER_TRANSCRIPTS = transcripts
        _WORKER_INPUT_ROOT = str(input_root)
        _WORKER_OUTPUT_ROOT = str(output_root)

        for task in tqdm(tasks, desc="Childrenizing", unit="file"):
            rel, err, meta = _process_one(task)
            if err is not None:
                err_count += 1
                msg = f"[ERROR] {rel}\n{err}\n"
                logger.error(msg.strip())
                with error_log.open("a", encoding="utf-8") as f:
                    f.write(msg)
            else:
                ok_count += 1
                if meta is not None:
                    all_meta.append(meta)
    else:
        # Multiprocessing with spawn-safe context
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(
                args.seed, args.enable_denoise, transcripts,
                str(input_root), str(output_root),
            ),
        ) as pool:
            results = pool.imap_unordered(
                _process_one, tasks, chunksize=chunksize,
            )
            for rel, err, meta in tqdm(
                results, total=len(tasks), desc="Childrenizing", unit="file"
            ):
                if err is not None:
                    err_count += 1
                    msg = f"[ERROR] {rel}\n{err}\n"
                    logger.error(msg.strip())
                    with error_log.open("a", encoding="utf-8") as f:
                        f.write(msg)
                else:
                    ok_count += 1
                    if meta is not None:
                        all_meta.append(meta)

    # ── Write JSONL manifest ───────────────────────────────────────────
    all_meta.sort(key=lambda m: m["audio_filepath"])
    with manifest_path.open("w", encoding="utf-8") as mf:
        for record in all_meta:
            mf.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(
        "Wrote manifest with %d entries -> %s", len(all_meta), manifest_path
    )
    # ── Summary ────────────────────────────────────────────────────────
    logger.info(
        "Done.  ok=%d  errors=%d  manifest_entries=%d  total_flac=%d",
        ok_count, err_count, len(all_meta), len(all_flac),
    )
    if err_count > 0:
        logger.warning("Errors logged to %s", error_log)

    return 0 if err_count == 0 else 2

if __name__ == "__main__":
    raise SystemExit(main())