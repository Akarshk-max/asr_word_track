#!/usr/bin/env python3
"""
NeMo JSONL manifest from LibriSpeech train-clean-100 / train-clean-360 for age-adversarial training.

Each line: audio_filepath, text, duration, age_years, age_cohort ("child" | "adult").

**Text normalization:** Both ``child`` and ``adult`` cohort rows use the **same** pipeline: Whisper
``EnglishTextNormalizer`` when ``whisper-normalizer`` is installed, otherwise lowercase-only fallback.
Use ``--require-whisper-normalizer`` when merging with child manifests built with Whisper so labels match.

LibriSpeech is **adult read speech** — there are no child speakers. By default every row is labeled
``age_cohort=adult`` with ``--adult-age-years`` (see ``--pseudo-cohort all_adult``). Merge a separate
child-corpus JSONL for real child data. For **synthetic** half/half batching experiments only, you can
pass ``--pseudo-cohort speaker_parity`` or ``speaker_hash`` (misleading vs. true acoustics).

Kaggle example::

    python build_librispeech_age_manifest.py \\
      --librispeech-data-root /kaggle/input/librispeech/LibriSpeech \\
      --splits train-clean-100 train-clean-360 \\
      --output /kaggle/working/libri_clean_train_age.jsonl \\
      --max-duration 20.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

try:
    import soundfile as sf
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install soundfile") from e

try:
    from whisper_normalizer.english import EnglishTextNormalizer
except ImportError:
    EnglishTextNormalizer = None  # type: ignore[misc, assignment]

# Singleton (same pattern as adapter / Kaggle training notebooks)
_WHISPER_INSTANCE = EnglishTextNormalizer() if EnglishTextNormalizer is not None else None

_trans_cache: dict[str, dict[str, str]] = {}
_WHISPER_STATUS_PRINTED = False


def _normalize_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if _WHISPER_INSTANCE is not None:
        return _WHISPER_INSTANCE(text).strip()
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


def iter_librispeech_flacs(root: Path) -> Iterator[tuple[Path, str]]:
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


def iter_librispeech_flacs_multi(data_root: Path, splits: list[str]) -> Iterator[tuple[Path, str]]:
    data_root = data_root.resolve()
    for split in splits:
        sub = data_root / split
        if not sub.is_dir():
            print(f"[warn] missing split dir, skip: {sub}", file=sys.stderr)
            continue
        yield from iter_librispeech_flacs(sub)


def _flac_duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.duration)


def _speaker_id_from_uid(uid: str) -> str:
    parts = uid.split("-")
    return parts[0] if parts else uid


def _pseudo_child_cohort(speaker_id: str, strategy: str) -> bool:
    if strategy == "all_adult":
        return False
    if strategy == "speaker_parity":
        try:
            return int(speaker_id) % 2 == 0
        except ValueError:
            return hash(speaker_id) % 2 == 0
    if strategy == "speaker_hash":
        return hash(speaker_id) % 2 == 0
    raise ValueError(f"unknown pseudo-cohort strategy: {strategy}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LibriSpeech → NeMo JSONL. Default: all rows adult; optional pseudo child/adult for tests."
    )
    ap.add_argument(
        "--librispeech-root",
        type=str,
        default="",
        help="Single split root (e.g. .../train-clean-100)",
    )
    ap.add_argument(
        "--librispeech-data-root",
        type=str,
        default="",
        help="Parent folder containing split subdirs (e.g. .../LibriSpeech)",
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["train-clean-100", "train-clean-360"],
        help="Split directory names under --librispeech-data-root",
    )
    ap.add_argument("--output", "-o", type=str, required=True, help="Output JSONL path")
    ap.add_argument("--min-duration", type=float, default=0.1)
    ap.add_argument("--max-duration", type=float, default=200.0)
    ap.add_argument(
        "--pseudo-cohort",
        choices=("all_adult", "speaker_parity", "speaker_hash"),
        default="all_adult",
        help="Libri is all adult: default all_adult. speaker_* = fake child/adult split for debugging batch samplers only.",
    )
    ap.add_argument(
        "--child-age-years",
        type=float,
        default=8.0,
        help="age_years for pseudo-child rows (should fall in child band vs AGE_CHILD_* env in training).",
    )
    ap.add_argument(
        "--adult-age-years",
        type=float,
        default=35.0,
        help="age_years for adult cohort rows.",
    )
    ap.add_argument(
        "--require-whisper-normalizer",
        action="store_true",
        help="Exit if whisper-normalizer is not installed (use when merging with Whisper-normalized child JSONL).",
    )
    args = ap.parse_args()

    global _WHISPER_STATUS_PRINTED
    if not _WHISPER_STATUS_PRINTED:
        _WHISPER_STATUS_PRINTED = True
        if _WHISPER_INSTANCE is not None:
            print(
                "build_librispeech_age_manifest: Whisper EnglishTextNormalizer active "
                "(child and adult rows use the same normalizer).",
                file=sys.stderr,
            )
        else:
            print(
                "build_librispeech_age_manifest: WARNING: whisper-normalizer not installed — "
                "using lowercase-only fallback for ALL rows. Install whisper-normalizer for parity "
                "with child manifests.",
                file=sys.stderr,
            )
    if args.require_whisper_normalizer and _WHISPER_INSTANCE is None:
        raise SystemExit(
            "--require-whisper-normalizer set but whisper-normalizer is not installed "
            "(pip install whisper-normalizer)."
        )

    dr = (args.librispeech_data_root or "").strip()
    one_root = (args.librispeech_root or "").strip()
    if not dr and not one_root:
        ap.error("set --librispeech-data-root or --librispeech-root")
    if dr and one_root:
        ap.error("set only one of --librispeech-data-root or --librispeech-root")

    if dr:
        flac_iter = iter_librispeech_flacs_multi(Path(dr), list(args.splits))
    else:
        flac_iter = iter_librispeech_flacs(Path(one_root))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.pseudo_cohort == "all_adult":
        print(
            "pseudo-cohort=all_adult: every Libri row → adult (LibriSpeech is adult read speech).",
            file=sys.stderr,
        )
    else:
        print(
            f"WARNING: pseudo-cohort={args.pseudo_cohort} invents child labels on Libri — "
            "not acoustically valid; for batching experiments only.",
            file=sys.stderr,
        )
    n_out = 0
    n_skip_dur = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for flac, raw in flac_iter:
            try:
                dur = _flac_duration(flac)
            except Exception:
                continue
            if dur < args.min_duration or dur > args.max_duration:
                n_skip_dur += 1
                continue
            uid = flac.stem
            spk = _speaker_id_from_uid(uid)
            is_child = _pseudo_child_cohort(spk, args.pseudo_cohort)
            age_years = float(args.child_age_years if is_child else args.adult_age_years)
            cohort = "child" if is_child else "adult"
            text = _normalize_text(raw)
            if not text:
                continue
            row = {
                "audio_filepath": str(flac.resolve()),
                "text": text,
                "duration": round(dur, 3),
                "age_years": age_years,
                "age_cohort": cohort,
            }
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    print(
        f"wrote {n_out} rows -> {out_path} (skipped {n_skip_dur} by duration bounds)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
