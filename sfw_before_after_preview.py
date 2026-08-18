#!/usr/bin/env python3
"""
Pair LibriSpeech originals (.flac) with SFW outputs (.wav) by utterance stem and
preview them (notebook Audio widgets) or write side-by-side WAVs for download.

Kaggle notebook (same kernel as training):
  from sfw_before_after_preview import preview_pairs
  preview_pairs(
      librispeech_root="/kaggle/input/.../train-clean-100",
      sfw_wav_dir="/kaggle/working/sfw_audio",
      n=5,
  )

CLI:
  python sfw_before_after_preview.py \\
    --librispeech-root /path/to/train-clean-100 \\
    --sfw-wav-dir /path/to/sfw_audio \\
    --save-dir /path/to/preview_out \\
    --n 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import librosa
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install librosa") from e

try:
    import soundfile as sf
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install soundfile") from e

from sfw_augment import SFW_SAMPLE_RATE


def _in_ipython() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except Exception:
        return False


def find_flac(librispeech_root: Path, stem: str) -> Path | None:
    """Utterance id is unique in LibriSpeech; first match wins."""
    for p in librispeech_root.rglob(f"{stem}.flac"):
        return p
    return None


def _peak_normalize(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    m = float(np.max(np.abs(y))) if y.size else 0.0
    if m < 1e-8:
        return y
    return (y / m * peak).astype(np.float32)


def load_before_after(
    flac_path: Path,
    wav_path: Path,
    *,
    target_sr: int = SFW_SAMPLE_RATE,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    y_before, _ = librosa.load(str(flac_path), sr=target_sr, mono=True)
    y_after, sr_a = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if y_after.ndim > 1:
        y_after = np.mean(y_after, axis=-1)
    if sr_a != target_sr:
        y_after = librosa.resample(
            np.asarray(y_after, dtype=np.float32),
            orig_sr=int(sr_a),
            target_sr=target_sr,
        )
    else:
        y_after = np.asarray(y_after, dtype=np.float32)
    if normalize:
        y_before = _peak_normalize(y_before)
        y_after = _peak_normalize(y_after)
    return y_before, y_after, target_sr


def preview_pairs(
    librispeech_root: str | Path,
    sfw_wav_dir: str | Path,
    *,
    n: int = 5,
    utterance: str | None = None,
    normalize: bool = False,
    save_dir: str | Path | None = None,
    verbose: bool = True,
) -> list[tuple[str, Path, Path]]:
    """
    Load and show (or save) before/after pairs.

    Returns list of (stem, flac_path, wav_path) for pairs that were found.
    """
    root = Path(librispeech_root).resolve()
    wav_dir = Path(sfw_wav_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"librispeech root not found: {root}")
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"sfw wav dir not found: {wav_dir}")

    wavs = sorted(wav_dir.glob("*.wav"))
    if utterance:
        stem = utterance.replace(".wav", "")
        wavs = [wav_dir / f"{stem}.wav"] if (wav_dir / f"{stem}.wav").is_file() else []

    pairs: list[tuple[str, Path, Path]] = []
    for wav in wavs:
        stem = wav.stem
        flac = find_flac(root, stem)
        if flac is None:
            if verbose:
                print(f"[skip] no .flac for stem {stem}", file=sys.stderr)
            continue
        pairs.append((stem, flac, wav))
        if len(pairs) >= n and not utterance:
            break

    if not pairs:
        print("No matching flac+wav pairs found.", file=sys.stderr)
        return []

    save_path = Path(save_dir).resolve() if save_dir else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    show_ipy = _in_ipython() and save_path is None

    if show_ipy:
        from IPython.display import Audio, Markdown, display

    for stem, flac, wav in pairs:
        y_b, y_a, sr = load_before_after(flac, wav, normalize=normalize)
        if verbose:
            print(f"--- {stem} ---\n  before: {flac}\n  after:  {wav}\n  samples @ {sr} Hz: {len(y_b)} / {len(y_a)}")

        if save_path is not None:
            sf.write(str(save_path / f"{stem}_before.wav"), y_b, sr, subtype="PCM_16")
            sf.write(str(save_path / f"{stem}_after.wav"), y_a, sr, subtype="PCM_16")
            if verbose:
                print(f"  wrote {save_path / (stem + '_before.wav')}")
                print(f"  wrote {save_path / (stem + '_after.wav')}")

        if show_ipy:
            display(Markdown(f"### `{stem}` — **before** (LibriSpeech)"))
            display(Audio(y_b, rate=sr))
            display(Markdown(f"### `{stem}` — **after** (SFW)"))
            display(Audio(y_a, rate=sr))

    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description="Preview LibriSpeech vs SFW wav pairs.")
    ap.add_argument("--librispeech-root", type=str, required=True)
    ap.add_argument("--sfw-wav-dir", type=str, required=True)
    ap.add_argument("--n", type=int, default=5, help="Max pairs (sorted by wav name)")
    ap.add_argument("--utterance", type=str, default="", help="Single stem, e.g. 103-1240-0000")
    ap.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="Write *_before.wav and *_after.wav here (recommended for !python in notebook)",
    )
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="Peak-normalize each clip for comparable loudness",
    )
    args = ap.parse_args()

    save_dir = args.save_dir.strip() or None
    preview_pairs(
        args.librispeech_root,
        args.sfw_wav_dir,
        n=args.n,
        utterance=args.utterance.strip() or None,
        normalize=args.normalize,
        save_dir=save_dir,
        verbose=True,
    )

    if not _in_ipython() and save_dir is None:
        print(
            "\nTip: use --save-dir ./previews to export WAVs, or in a notebook run:\n"
            "  from sfw_before_after_preview import preview_pairs\n"
            "  preview_pairs('.../train-clean-100', '.../sfw_audio', n=5)\n",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
