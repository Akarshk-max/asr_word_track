#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_librispeech_manifest.py
================================
Build an incremental NeMo-compatible manifest.jsonl for LibriSpeech /
childrenized-LibriSpeech data.

LibriSpeech transcripts are UPPERCASE; this script normalises them to
lowercase with ``whisper-normalizer`` so they match child-speech manifests.

Modes
-----
1. From an existing manifest  (e.g. childrenize_librispeech.py output):

    python prepare_librispeech_manifest.py \\
        --input_manifest  /data/child_synthetic_librispeech/manifest.jsonl \\
        --audio_root      /data/child_synthetic_librispeech \\
        --manifest_path   /kaggle/working/librispeech_normalised.jsonl

2. From a raw LibriSpeech tree  (audio + .trans.txt co-located):

    python prepare_librispeech_manifest.py \\
        --librispeech_dir /data/LibriSpeech/train-clean-100 \\
        --manifest_path   librispeech_manifest.jsonl

3. Flat SFW audio + separate transcript tree:

    python prepare_librispeech_manifest.py \\
        --audio_dir       /data/sfw_audio \\
        --transcript_dir  /data/LibriSpeech \\
        --manifest_path   librispeech_manifest.jsonl

4. Rebuild from scratch  (rescan, ignore existing output manifest):

    python prepare_librispeech_manifest.py \\
        --input_manifest  /data/manifest.jsonl \\
        --audio_root      /data/child_synthetic_librispeech \\
        --manifest_path   out.jsonl --rebuild_manifest

Kaggle example (childrenized data uploaded as a Kaggle dataset):

    !python prepare_librispeech_manifest.py \\
        --input_manifest  /kaggle/input/librispeech-childrenized-5000/child_synthetic_librispeech/manifest.jsonl \\
        --audio_root      /kaggle/input/librispeech-childrenized-5000/child_synthetic_librispeech \\
        --manifest_path   /kaggle/working/librispeech_normalised.jsonl

Then in the training notebook:

    os.environ["LIBRISPEECH_MANIFEST"] = "/kaggle/working/librispeech_normalised.jsonl"

Features
--------
* Incremental — loads existing output manifest, skips already-processed
  entries, appends only new ones.  No duplicate ``audio_filepath`` values.
* Whisper text normalisation (UPPER → lower, punctuation cleanup).
* Multiprocessing for audio-scan mode (parallel duration extraction).
* ``--rebuild_manifest`` rewrites the output from scratch.
* Graceful handling of malformed manifest lines.
* Summary statistics printed at the end.

Dependencies
------------
    pip install whisper-normalizer tqdm
    pip install torchaudio          # or soundfile (fallback for duration)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from multiprocessing import cpu_count, get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except ImportError:
    _HAS_TORCHAUDIO = False

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False

try:
    from whisper_normalizer import EnglishTextNormalizer
    _normalizer = EnglishTextNormalizer()
except ImportError:
    _normalizer = None

AUDIO_EXTENSIONS = {".flac", ".wav"}


# ═══════════════════════════════════════════════════════════════════════════
# Manifest I/O
# ═══════════════════════════════════════════════════════════════════════════

def load_existing_manifest(manifest_path: Path) -> Dict[str, dict]:
    """Load a JSONL manifest keyed by ``audio_filepath``.

    Malformed / empty lines are warned and skipped.
    """
    entries: Dict[str, dict] = {}
    if not manifest_path.is_file():
        return entries
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line_num, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                key = rec.get("audio_filepath", "")
                if key:
                    entries[key] = rec
                else:
                    print(
                        f"  [warn] line {line_num}: "
                        "missing audio_filepath — skipped"
                    )
            except (json.JSONDecodeError, KeyError) as exc:
                print(f"  [warn] malformed line {line_num}: {exc}")
    return entries


def append_manifest_entries(
    manifest_path: Path,
    records: List[dict],
) -> int:
    """Append *records* to *manifest_path*.  Returns count written."""
    if not records:
        return 0
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


# ═══════════════════════════════════════════════════════════════════════════
# Text normalisation
# ═══════════════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """Whisper-normalise (UPPERCASE → lowercase + cleanup).

    Falls back to ``str.lower()`` when whisper-normalizer is not installed.
    """
    if _normalizer is not None:
        return _normalizer(text)
    return text.lower().strip()


# ═══════════════════════════════════════════════════════════════════════════
# Audio helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_audio_duration(path: str) -> float:
    """Return duration in seconds (prefers torchaudio, falls back to sf)."""
    if _HAS_TORCHAUDIO:
        info = torchaudio.info(path)
        return info.num_frames / float(info.sample_rate)
    if _HAS_SOUNDFILE:
        info = sf.info(path)
        return float(info.duration)
    raise ImportError(
        "Install torchaudio or soundfile for duration extraction."
    )


def ensure_mono(audio_path: str, cache_dir: str) -> str:
    """Return *audio_path* if mono, else write a mono copy to *cache_dir*."""
    if not _HAS_TORCHAUDIO:
        return audio_path
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] <= 1:
        return audio_path
    waveform = waveform.mean(dim=0, keepdim=True)
    os.makedirs(cache_dir, exist_ok=True)
    ext = os.path.splitext(audio_path)[1] or ".wav"
    h = hashlib.sha256(
        os.path.abspath(audio_path).encode()
    ).hexdigest()[:32]
    out = os.path.join(cache_dir, h + ext)
    if not os.path.isfile(out):
        torchaudio.save(out, waveform, sr)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# LibriSpeech transcript loading  (for audio-scan mode only)
# ═══════════════════════════════════════════════════════════════════════════

def load_transcripts(roots: List[str]) -> Dict[str, str]:
    """Parse ``*.trans.txt`` under each root → ``{utt_id: raw_text}``."""
    index: Dict[str, str] = {}
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if not fname.endswith(".trans.txt"):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split(" ", 1)
                            if len(parts) == 2:
                                index[parts[0]] = parts[1]
                except OSError:
                    continue
    return index


# ═══════════════════════════════════════════════════════════════════════════
# Audio file discovery  (for audio-scan mode only)
# ═══════════════════════════════════════════════════════════════════════════

def discover_audio_files(roots: List[str]) -> List[str]:
    """Recursively find all .flac / .wav under *roots*, sorted."""
    _skip = {"__pycache__", ".git", ".hg", ".ipynb_checkpoints"}
    found: List[str] = []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _skip]
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in AUDIO_EXTENSIONS:
                    found.append(os.path.join(dirpath, fn))
    return sorted(found)


# ═══════════════════════════════════════════════════════════════════════════
# Multiprocessing worker  (for audio-scan mode)
# ═══════════════════════════════════════════════════════════════════════════

_W_TRANSCRIPTS: Dict[str, str] = {}
_W_MONO_CACHE: str = ""
_W_ENSURE_MONO: bool = False


def _worker_init(
    transcripts: Dict[str, str],
    mono_cache: str,
    do_mono: bool,
) -> None:
    global _W_TRANSCRIPTS, _W_MONO_CACHE, _W_ENSURE_MONO
    _W_TRANSCRIPTS = transcripts
    _W_MONO_CACHE = mono_cache
    _W_ENSURE_MONO = do_mono


def _process_one_audio(
    audio_path: str,
) -> Tuple[str, Optional[str], Optional[dict]]:
    """Process one audio file → (path, error_or_None, record_or_None)."""
    try:
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        raw_text = _W_TRANSCRIPTS.get(stem)
        if raw_text is None:
            return (audio_path, "no transcript", None)

        final_path = audio_path
        if _W_ENSURE_MONO and _W_MONO_CACHE:
            final_path = ensure_mono(audio_path, _W_MONO_CACHE)

        duration = get_audio_duration(final_path)
        text = normalize_text(raw_text)
        speaker_id = stem.split("-")[0] if "-" in stem else ""

        record = {
            "audio_filepath": final_path,
            "duration": round(duration, 4),
            "text": text,
            "speaker_id": speaker_id,
        }
        return (audio_path, None, record)
    except Exception:
        return (audio_path, traceback.format_exc(), None)


# ═══════════════════════════════════════════════════════════════════════════
# MODE A — Transform an existing manifest
# ═══════════════════════════════════════════════════════════════════════════

def process_from_manifest(
    input_manifest_path: str,
    audio_root: str,
    output_manifest_path: Path,
    rebuild: bool,
    ensure_mono_flag: bool,
    mono_cache_dir: str,
    verify_audio: bool,
) -> int:
    """Read *input_manifest_path*, normalise text, resolve paths, write output.

    This is fast because duration is taken from the input manifest —
    no audio I/O unless ``--verify_audio`` or ``--ensure_mono`` is set.
    """
    from tqdm import tqdm

    # Load input manifest (source of truth for transcripts + durations)
    print(f"Reading input manifest: {input_manifest_path}")
    input_records: List[dict] = []
    bad_lines = 0
    with open(input_manifest_path, "r", encoding="utf-8") as fh:
        for line_num, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                input_records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                bad_lines += 1
                print(f"  [warn] input line {line_num}: {exc}")
    print(f"  {len(input_records)} records read  ({bad_lines} bad lines)")

    # Load existing output manifest (for incremental skip)
    if rebuild:
        existing: Dict[str, dict] = {}
    else:
        print(f"Loading existing output manifest: {output_manifest_path}")
        existing = load_existing_manifest(output_manifest_path)
    loaded_count = len(existing)
    existing_keys = set(existing.keys())
    print(f"  {loaded_count} existing entries")

    # Process each input record
    new_records: List[dict] = []
    skipped = 0
    missing_audio = 0
    no_text = 0

    for rec in tqdm(input_records, desc="Normalising", unit="rec"):
        raw_audio = rec.get("audio_filepath", "")
        raw_text = rec.get("text", "")
        duration = rec.get("duration")

        if not raw_audio:
            continue

        # Resolve relative → absolute using audio_root
        if audio_root and not os.path.isabs(raw_audio):
            abs_audio = os.path.join(audio_root, raw_audio)
        else:
            abs_audio = raw_audio

        # Mono conversion (changes the final path)
        if ensure_mono_flag and mono_cache_dir:
            if os.path.isfile(abs_audio):
                abs_audio = ensure_mono(abs_audio, mono_cache_dir)

        # Skip if already in output manifest
        if abs_audio in existing_keys:
            skipped += 1
            continue

        # Verify audio exists on disk
        if verify_audio and not os.path.isfile(abs_audio):
            missing_audio += 1
            continue

        if not raw_text.strip():
            no_text += 1
            continue

        # Get duration from input record, or read from audio file
        if duration is None or duration <= 0:
            if os.path.isfile(abs_audio):
                try:
                    duration = get_audio_duration(abs_audio)
                except Exception:
                    missing_audio += 1
                    continue
            else:
                missing_audio += 1
                continue

        text = normalize_text(raw_text)
        speaker_id = rec.get("speaker_id", "")
        if not speaker_id:
            stem = os.path.splitext(os.path.basename(abs_audio))[0]
            speaker_id = stem.split("-")[0] if "-" in stem else ""

        new_records.append({
            "audio_filepath": abs_audio,
            "duration": round(float(duration), 4),
            "text": text,
            "speaker_id": speaker_id,
        })

    # Deduplicate new records among themselves
    seen: set = set()
    deduped: List[dict] = []
    for r in new_records:
        k = r["audio_filepath"]
        if k not in seen and k not in existing_keys:
            seen.add(k)
            deduped.append(r)
    deduped.sort(key=lambda r: r["audio_filepath"])

    # Write
    if rebuild:
        output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with output_manifest_path.open("w", encoding="utf-8") as fh:
            for r in deduped:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        appended = len(deduped)
    else:
        appended = append_manifest_entries(output_manifest_path, deduped)

    # Summary
    print(
        f"\nSummary:  loaded={loaded_count}  skipped={skipped}  "
        f"newly_processed={len(new_records)}  appended={appended}"
    )
    if missing_audio:
        print(f"  Missing audio files: {missing_audio}")
    if no_text:
        print(f"  Entries with no text: {no_text}")
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# MODE B — Build from audio files + .trans.txt
# ═══════════════════════════════════════════════════════════════════════════

def process_from_audio_scan(
    audio_roots: List[str],
    transcript_roots: List[str],
    output_manifest_path: Path,
    rebuild: bool,
    ensure_mono_flag: bool,
    mono_cache_dir: str,
    num_workers: int,
    chunksize: int,
    max_files: Optional[int],
) -> int:
    """Scan audio files, look up transcripts from .trans.txt, write manifest."""
    from tqdm import tqdm

    print("Loading transcripts from .trans.txt files ...")
    transcripts = load_transcripts(transcript_roots)
    print(f"  {len(transcripts)} transcript entries")

    if rebuild:
        existing: Dict[str, dict] = {}
    else:
        print(f"Loading existing output manifest: {output_manifest_path}")
        existing = load_existing_manifest(output_manifest_path)
    loaded_count = len(existing)
    existing_keys = set(existing.keys())
    print(f"  {loaded_count} existing entries")

    print("Discovering audio files ...")
    all_audio = discover_audio_files(audio_roots)
    print(f"  {len(all_audio)} audio files found")

    if max_files is not None and max_files < len(all_audio):
        all_audio = all_audio[:max_files]
        print(f"  Limited to {max_files} files (--max_files)")

    # Filter out files already in output manifest
    to_process: List[str] = []
    skipped = 0
    for path in all_audio:
        if path in existing_keys:
            skipped += 1
            continue
        if ensure_mono_flag and mono_cache_dir:
            ext = os.path.splitext(path)[1] or ".wav"
            h = hashlib.sha256(
                os.path.abspath(path).encode()
            ).hexdigest()[:32]
            mono_path = os.path.join(mono_cache_dir, h + ext)
            if mono_path in existing_keys:
                skipped += 1
                continue
        to_process.append(path)

    print(
        f"To process: {len(to_process)}  "
        f"(skipped {skipped} already in manifest)"
    )

    if not to_process:
        print(
            f"\nSummary:  loaded={loaded_count}  skipped={skipped}  "
            "newly_processed=0  appended=0"
        )
        return 0

    # Process
    nw = max(1, num_workers)
    cs = (
        chunksize
        if chunksize > 0
        else max(1, min(32, len(to_process) // (nw * 4)))
    )
    print(f"Workers: {nw}   chunksize: {cs}")

    ok_count = 0
    err_count = 0
    no_transcript = 0
    new_records: List[dict] = []

    if nw <= 1:
        global _W_TRANSCRIPTS, _W_MONO_CACHE, _W_ENSURE_MONO
        _W_TRANSCRIPTS = transcripts
        _W_MONO_CACHE = mono_cache_dir
        _W_ENSURE_MONO = ensure_mono_flag
        for path in tqdm(to_process, desc="Processing", unit="file"):
            _, err, rec = _process_one_audio(path)
            if err:
                err_count += 1
                if "no transcript" in str(err):
                    no_transcript += 1
            else:
                ok_count += 1
                if rec:
                    new_records.append(rec)
    else:
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=nw,
            initializer=_worker_init,
            initargs=(transcripts, mono_cache_dir, ensure_mono_flag),
        ) as pool:
            results = pool.imap_unordered(
                _process_one_audio, to_process, chunksize=cs,
            )
            for _, err, rec in tqdm(
                results, total=len(to_process),
                desc="Processing", unit="file",
            ):
                if err:
                    err_count += 1
                    if "no transcript" in str(err):
                        no_transcript += 1
                else:
                    ok_count += 1
                    if rec:
                        new_records.append(rec)

    # Deduplicate & write
    deduped = [
        r for r in new_records if r["audio_filepath"] not in existing_keys
    ]
    deduped.sort(key=lambda r: r["audio_filepath"])

    if rebuild:
        output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with output_manifest_path.open("w", encoding="utf-8") as fh:
            for r in deduped:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        appended = len(deduped)
    else:
        appended = append_manifest_entries(output_manifest_path, deduped)

    print(
        f"\nSummary:  loaded={loaded_count}  skipped={skipped}  "
        f"newly_processed={ok_count}  appended={appended}"
    )
    if no_transcript:
        print(f"  Files with no matching transcript: {no_transcript}")
    if err_count - no_transcript > 0:
        print(
            f"  Other errors (audio read failures): "
            f"{err_count - no_transcript}"
        )
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build an incremental NeMo-compatible manifest for LibriSpeech "
            "/ childrenized-LibriSpeech data.  Whisper-normalises UPPER → "
            "lower text."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "\n"
            "  From existing childrenized manifest:\n"
            "    python prepare_librispeech_manifest.py \\\n"
            "        --input_manifest  child_synthetic_librispeech/manifest.jsonl \\\n"
            "        --audio_root      child_synthetic_librispeech \\\n"
            "        --manifest_path   librispeech_normalised.jsonl\n"
            "\n"
            "  From raw LibriSpeech tree:\n"
            "    python prepare_librispeech_manifest.py \\\n"
            "        --librispeech_dir /data/LibriSpeech/train-clean-100 \\\n"
            "        --manifest_path   librispeech_manifest.jsonl\n"
            "\n"
            "  Flat SFW audio + transcript tree:\n"
            "    python prepare_librispeech_manifest.py \\\n"
            "        --audio_dir      /data/sfw_audio \\\n"
            "        --transcript_dir /data/LibriSpeech \\\n"
            "        --manifest_path  librispeech_manifest.jsonl\n"
        ),
    )

    # --- Mode A: existing manifest ---
    g1 = p.add_argument_group("Mode A — from existing manifest")
    g1.add_argument(
        "--input_manifest",
        type=str,
        default="",
        help="Path to an existing manifest.jsonl whose text will be "
             "Whisper-normalised.  Durations are reused (no audio I/O).",
    )
    g1.add_argument(
        "--audio_root",
        type=str,
        default="",
        help="Root directory prepended to relative audio_filepath values "
             "in --input_manifest.",
    )
    g1.add_argument(
        "--verify_audio",
        action="store_true",
        help="When using --input_manifest, skip entries whose audio file "
             "does not exist on disk.",
    )

    # --- Mode B: audio-scan ---
    g2 = p.add_argument_group("Mode B — from audio files + .trans.txt")
    g2.add_argument(
        "--librispeech_dir",
        type=str,
        default="",
        help="Root of a LibriSpeech tree (audio + .trans.txt co-located). "
             "Comma-separated for multiple roots.",
    )
    g2.add_argument(
        "--audio_dir",
        type=str,
        default="",
        help="Flat audio directory (SFW mode).  Pair with --transcript_dir.",
    )
    g2.add_argument(
        "--transcript_dir",
        type=str,
        default="",
        help="Separate .trans.txt tree (SFW mode).  Searched recursively.",
    )

    # --- Shared ---
    g3 = p.add_argument_group("Shared options")
    g3.add_argument(
        "--manifest_path",
        type=str,
        required=True,
        help="Output manifest JSONL path.",
    )
    g3.add_argument(
        "--rebuild_manifest",
        action="store_true",
        help="Rebuild output manifest from scratch (ignore existing output).",
    )
    g3.add_argument(
        "--ensure_mono",
        action="store_true",
        help="Convert stereo files to mono (cached in --mono_cache_dir).",
    )
    g3.add_argument(
        "--mono_cache_dir",
        type=str,
        default="",
        help="Cache directory for mono-converted files.",
    )
    g3.add_argument(
        "--num_workers",
        type=int,
        default=max(1, cpu_count() or 4),
        help="Parallel workers for audio-scan mode (default: all CPUs).",
    )
    g3.add_argument(
        "--max_files",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N files (for testing).",
    )
    g3.add_argument(
        "--chunksize",
        type=int,
        default=0,
        help="Pool imap chunksize (0 = auto).",
    )
    return p.parse_args(argv)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    output_path = Path(args.manifest_path)

    # ------------------------------------------------------------------
    # Mode A: from an existing manifest
    # ------------------------------------------------------------------
    if args.input_manifest:
        if not os.path.isfile(args.input_manifest):
            print(f"Error: --input_manifest not found: {args.input_manifest}")
            return 1
        return process_from_manifest(
            input_manifest_path=args.input_manifest,
            audio_root=args.audio_root,
            output_manifest_path=output_path,
            rebuild=args.rebuild_manifest,
            ensure_mono_flag=args.ensure_mono,
            mono_cache_dir=args.mono_cache_dir,
            verify_audio=args.verify_audio,
        )

    # ------------------------------------------------------------------
    # Mode B: audio-scan + .trans.txt
    # ------------------------------------------------------------------
    audio_roots: List[str] = []
    transcript_roots: List[str] = []

    for d in args.librispeech_dir.split(","):
        d = d.strip()
        if d:
            audio_roots.append(d)
            transcript_roots.append(d)
    for d in args.audio_dir.split(","):
        d = d.strip()
        if d:
            audio_roots.append(d)
    for d in args.transcript_dir.split(","):
        d = d.strip()
        if d:
            transcript_roots.append(d)

    if not audio_roots:
        print(
            "Error: supply --input_manifest, --librispeech_dir, "
            "or --audio_dir"
        )
        return 1
    if not transcript_roots:
        print("Error: supply --librispeech_dir or --transcript_dir")
        return 1

    return process_from_audio_scan(
        audio_roots=audio_roots,
        transcript_roots=transcript_roots,
        output_manifest_path=output_path,
        rebuild=args.rebuild_manifest,
        ensure_mono_flag=args.ensure_mono,
        mono_cache_dir=args.mono_cache_dir,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    sys.exit(main())
