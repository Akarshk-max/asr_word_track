#!/usr/bin/env python3
"""
Build SFW-augmented wav corpus + NeMo JSONL manifest from LibriSpeech tree or existing manifest.
Transcripts use Whisper EnglishTextNormalizer when whisper-normalizer is installed.

Full clean-100 + clean-360 (Kaggle):
  python build_sfw_manifest.py \\
    --librispeech-data-root /kaggle/input/.../LibriSpeech \\
    --librispeech-splits train-clean-100 train-clean-360 \\
    --output-wav-dir /kaggle/working/sfw_audio \\
    --output-manifest /kaggle/working/sfw_train_clean_manifest.jsonl \\
    --num-workers 4 \\
    --skip-existing

Single split:
  python build_sfw_manifest.py \\
    --librispeech-root /kaggle/input/.../LibriSpeech/train-clean-100 \\
    --output-wav-dir /kaggle/working/sfw_audio \\
    --output-manifest /kaggle/working/sfw_train_manifest.jsonl

NeMo JSONL source:
  python build_sfw_manifest.py --manifest /path/to/train.jsonl --output-wav-dir ... --output-manifest ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import soundfile as sf
except ImportError as e:  # pragma: no cover
    raise SystemExit("install soundfile: pip install soundfile") from e

try:
    import librosa
except ImportError as e:  # pragma: no cover
    raise SystemExit("install librosa") from e

try:
    from whisper_normalizer.english import EnglishTextNormalizer
except ImportError:
    EnglishTextNormalizer = None  # type: ignore[misc, assignment]

from sfw_augment import SFW_SAMPLE_RATE, apply_sfw_to_waveform


def _whisper_normalize(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if EnglishTextNormalizer is not None:
        return EnglishTextNormalizer()(text).strip()
    return text.lower().strip()


def _load_chapter_transcripts(trans_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not trans_path.is_file():
        return out
    with open(trans_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            uid, transcript = parts[0], parts[1]
            out[uid] = transcript
    return out


_trans_cache: dict[str, dict[str, str]] = {}


def iter_librispeech_flacs(root: Path) -> Iterator[tuple[Path, str]]:
    """Yield (flac_path, raw_transcript)."""
    root = root.resolve()
    for flac in sorted(root.rglob("*.flac")):
        chapter_dir = flac.parent
        try:
            speaker = chapter_dir.parent.name
            chapter = chapter_dir.name
        except Exception:
            continue
        trans_name = f"{speaker}-{chapter}.trans.txt"
        trans_path = chapter_dir / trans_name
        key = str(trans_path)
        if key not in _trans_cache:
            _trans_cache[key] = _load_chapter_transcripts(trans_path)
        uid = flac.stem
        raw = _trans_cache[key].get(uid)
        if raw is None:
            continue
        yield flac, raw


def iter_librispeech_flacs_multi(
    data_root: Path,
    splits: list[str],
) -> Iterator[tuple[Path, str]]:
    """Walk train-clean-* subtrees under a common LibriSpeech folder."""
    data_root = data_root.resolve()
    for split in splits:
        sub = data_root / split
        if not sub.is_dir():
            print(f"[warn] missing split dir, skip: {sub}", file=sys.stderr)
            continue
        yield from iter_librispeech_flacs(sub)


def iter_manifest_jsonl(path: Path) -> Iterator[tuple[Path, str, float | None]]:
    """Yield (audio_path, text, duration or None)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ap = row.get("audio_filepath") or row.get("audio_path")
            if not ap:
                continue
            text = row.get("text", "")
            dur = row.get("duration")
            try:
                dur_f = float(dur) if dur is not None else None
            except (TypeError, ValueError):
                dur_f = None
            yield Path(ap), str(text), dur_f


def process_one(
    in_path: Path,
    out_wav: Path,
    raw_text: str,
    rng: np.random.Generator,
    griffin_iter: int,
) -> tuple[bool, str, float]:
    """
    Returns (ok, whisper_text, duration_sec).
    """
    norm = _whisper_normalize(raw_text)
    if not norm:
        return False, "", 0.0

    try:
        y, sr = librosa.load(str(in_path), sr=None, mono=True)
    except Exception:
        return False, "", 0.0

    if y.size == 0:
        return False, "", 0.0

    y_out, _, _ = apply_sfw_to_waveform(
        y.astype(np.float32),
        int(sr),
        rng=rng,
        griffin_iter=griffin_iter,
    )
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), y_out, SFW_SAMPLE_RATE, subtype="PCM_16")
    dur = float(len(y_out) / SFW_SAMPLE_RATE)
    return True, norm, dur


def _mp_process_one(job: tuple[str, str, str, int, int]) -> dict | None:
    """Picklable worker: (flac_path, raw_transcript, out_wav_path, seed, griffin_iter)."""
    flac_s, raw, out_wav_s, seed, griffin_iter = job
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    ok, norm, dur = process_one(Path(flac_s), Path(out_wav_s), raw, rng, int(griffin_iter))
    if not ok:
        return None
    return {"audio_filepath": out_wav_s, "duration": dur, "text": norm}


def main() -> int:
    p = argparse.ArgumentParser(description="Build SFW-augmented LibriSpeech / manifest corpus.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--librispeech-root",
        type=str,
        help="One split root: speaker/chapter/*.flac + *.trans.txt",
    )
    src.add_argument(
        "--librispeech-data-root",
        type=str,
        help="Parent folder containing split subdirs (e.g. .../LibriSpeech with train-clean-100 inside)",
    )
    src.add_argument(
        "--manifest",
        type=str,
        help="NeMo JSONL with audio_filepath and text",
    )
    p.add_argument(
        "--librispeech-splits",
        nargs="+",
        default=["train-clean-100", "train-clean-360"],
        help="Used with --librispeech-data-root only",
    )
    p.add_argument("--output-wav-dir", type=str, required=True, help="Directory for augmented .wav files")
    p.add_argument("--output-manifest", type=str, required=True, help="Output NeMo JSONL path")
    p.add_argument("--max-utterances", type=int, default=0, help="Cap utterances (0 = no cap)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--griffin-iter",
        type=int,
        default=8,
        help="Griffin–Lim iterations (try 16 if quality is poor)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="If output wav exists, reuse it and only append manifest row",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel processes for SFW (1=sequential). Try 4–8 on Kaggle CPU",
    )
    args = p.parse_args()

    if EnglishTextNormalizer is None:
        print(
            "WARNING: whisper-normalizer not installed; using lowercase only. "
            "pip install whisper-normalizer",
            file=sys.stderr,
        )

    out_wav_dir = Path(args.output_wav_dir).resolve()
    out_manifest = Path(args.output_manifest).resolve()
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    rows_written = 0
    skipped_empty = 0
    skipped_audio = 0
    n_max = args.max_utterances if args.max_utterances and args.max_utterances > 0 else None

    def _librispeech_iter() -> Iterator[tuple[Path, str]]:
        if args.librispeech_data_root:
            dr = Path(args.librispeech_data_root)
            if not dr.is_dir():
                print(f"Not a directory: {dr}", file=sys.stderr)
                return iter(())
            return iter_librispeech_flacs_multi(dr, list(args.librispeech_splits))
        if args.librispeech_root:
            root = Path(args.librispeech_root)
            if not root.is_dir():
                print(f"Not a directory: {root}", file=sys.stderr)
                return iter(())
            return iter_librispeech_flacs(root)
        return iter(())

    if args.librispeech_data_root:
        dr = Path(args.librispeech_data_root)
        if not dr.is_dir():
            return 1
    if args.librispeech_root:
        root = Path(args.librispeech_root)
        if not root.is_dir():
            return 1

    with open(out_manifest, "w", encoding="utf-8") as mf:
        if args.librispeech_root or args.librispeech_data_root:
            jobs: list[tuple[str, str, str, int, int]] = []
            for flac, raw in _librispeech_iter():
                if n_max is not None and rows_written >= n_max:
                    break
                norm = _whisper_normalize(raw)
                if not norm:
                    skipped_empty += 1
                    continue
                out_wav = out_wav_dir / (flac.stem + ".wav")
                if args.skip_existing and out_wav.is_file():
                    try:
                        info = sf.info(str(out_wav))
                        dur = float(info.duration)
                    except Exception:
                        dur = float(len(sf.read(str(out_wav))[0]) / SFW_SAMPLE_RATE)
                    line = (
                        json.dumps(
                            {
                                "audio_filepath": str(out_wav),
                                "duration": dur,
                                "text": norm,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    mf.write(line)
                    rows_written += 1
                    mf.flush()
                    continue
                if n_max is not None and rows_written + len(jobs) >= n_max:
                    break
                seed_i = (args.seed + zlib.adler32(flac.stem.encode("utf-8"))) & 0xFFFFFFFF
                jobs.append(
                    (str(flac.resolve()), raw, str(out_wav.resolve()), seed_i, int(args.griffin_iter))
                )

            nw = max(1, int(args.num_workers))
            if nw == 1:
                for job in jobs:
                    row = _mp_process_one(job)
                    if row is None:
                        skipped_audio += 1
                        continue
                    mf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    mf.flush()
                    rows_written += 1
            else:
                done: list[dict] = []
                with ProcessPoolExecutor(max_workers=nw) as ex:
                    futs = {ex.submit(_mp_process_one, job): job for job in jobs}
                    for fut in as_completed(futs):
                        try:
                            row = fut.result()
                        except Exception as e:  # pragma: no cover
                            print(f"[worker error] {e}", file=sys.stderr)
                            skipped_audio += 1
                            continue
                        if row is None:
                            skipped_audio += 1
                            continue
                        done.append(row)
                done.sort(key=lambda r: Path(r["audio_filepath"]).stem)
                for row in done:
                    mf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows_written += 1
                mf.flush()
        else:
            man = Path(args.manifest)
            if not man.is_file():
                print(f"Manifest not found: {man}", file=sys.stderr)
                return 1
            for ap, text, _dur_hint in iter_manifest_jsonl(man):
                if n_max is not None and rows_written >= n_max:
                    break
                if not ap.is_file():
                    skipped_audio += 1
                    continue
                raw = text
                norm = _whisper_normalize(raw)
                if not norm:
                    skipped_empty += 1
                    continue
                ap_key = str(ap.resolve())
                h = hashlib.md5(ap_key.encode("utf-8")).hexdigest()[:10]
                safe = f"{h}_{ap.stem.replace(' ', '_')}.wav"
                out_wav = out_wav_dir / safe
                if args.skip_existing and out_wav.is_file():
                    try:
                        info = sf.info(str(out_wav))
                        dur = float(info.duration)
                    except Exception:
                        dur = float(len(sf.read(str(out_wav))[0]) / SFW_SAMPLE_RATE)
                    line = (
                        json.dumps(
                            {
                                "audio_filepath": str(out_wav),
                                "duration": dur,
                                "text": norm,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    mf.write(line)
                    rows_written += 1
                    continue
                ok, norm2, dur = process_one(ap, out_wav, raw, rng, args.griffin_iter)
                if not ok:
                    skipped_audio += 1
                    continue
                mf.write(
                    json.dumps(
                        {"audio_filepath": str(out_wav), "duration": dur, "text": norm2},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                mf.flush()
                rows_written += 1

    print(
        f"Done. Wrote {rows_written} rows to {out_manifest}. "
        f"skipped_empty_norm={skipped_empty} skipped_audio={skipped_audio}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
