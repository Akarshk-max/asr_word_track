#!/usr/bin/env python3
"""
Join ``age_bucket`` (child age group) from transcript JSONLs onto ``train_manifest.jsonl``.

Reads, in order (later files only fill keys still missing):

1. ``train_word_transcripts_t.jsonl`` — primary word-level metadata
2. ``train_word_transcripts.jsonl`` — gap-fill for utterances missing from (1), unless
   ``--no-word-full``
3. Optionally ``--phon-transcripts`` — phonetic JSONL (subset of utterances), if you have it

Match key: basename of ``audio_filepath`` without extension, equal to ``utterance_id``
(e.g. ``U_890c3ddd00b291ce``).

Writes ``age_bucket`` plus numeric ``age_years`` (bucket midpoint) for downstream
``train_parakeet_age_adversarial.py``. Rows with no metadata get ``age_bucket`` ``unknown``
and ``age_years`` 8.0.

Notebook / Kaggle: ``__file__`` is undefined when pasted into a cell; defaults use
``os.getcwd()`` instead. Run as a ``.py`` file or set working directory to the folder
that contains the JSONLs.

**Kaggle (run as script, write to working — input is read-only):**

.. code-block:: bash

   cd /kaggle/input/<your-dataset>
   python /kaggle/working/enrich_submission_train_manifest_age.py \\
     --manifest ./submission_1_meta_data/train_manifest.jsonl \\
     --word-transcripts-t ./train_word_transcripts_t.jsonl \\
     --word-transcripts-full ./train_word_transcripts.jsonl \\
     --output /kaggle/working/train_manifest_with_age.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence


def _script_dir() -> Path:
    """Directory containing this file, or cwd when ``__file__`` is missing (e.g. notebook)."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def _iter_jsonl_rows(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_age_bucket_map(paths: Sequence[Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        if not p.is_file():
            continue
        for row in _iter_jsonl_rows(p):
            uid = row.get("utterance_id")
            ab = row.get("age_bucket")
            if not uid or not ab:
                continue
            if uid not in out:
                out[str(uid)] = str(ab)
    return out


def age_bucket_to_years(bucket: str) -> float:
    b = str(bucket).strip()
    if b == "3-4":
        return 3.5
    if b == "5-7":
        return 6.0
    if b == "8-11":
        return 9.5
    if b == "12+":
        return 13.0
    if b.lower() == "unknown":
        return 8.0
    return 8.0


def utterance_id_from_manifest_row(row: dict) -> Optional[str]:
    ap = row.get("audio_filepath") or row.get("audio_file") or row.get("audio_filename")
    if not ap:
        return None
    stem = Path(str(ap).replace("\\", "/")).stem
    return stem or None


def main() -> None:
    here = _script_dir()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=here / "submission_1_meta_data" / "train_manifest.jsonl",
        help="Manifest JSONL to enrich.",
    )
    ap.add_argument(
        "--word-transcripts-t",
        type=Path,
        default=here / "train_word_transcripts_t.jsonl",
    )
    ap.add_argument(
        "--phon-transcripts",
        type=Path,
        default=None,
        help="Optional phonetic JSONL; only used if this path exists.",
    )
    ap.add_argument(
        "--word-transcripts-full",
        type=Path,
        default=here / "train_word_transcripts.jsonl",
        help="Full word manifest (fills utterances missing from _t).",
    )
    ap.add_argument(
        "--no-word-full",
        action="store_true",
        help="Do not read train_word_transcripts.jsonl; unmatched rows get age_bucket unknown.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: overwrite --manifest in place via temp file).",
    )
    args = ap.parse_args()

    sources: List[Path] = [args.word_transcripts_t]
    if args.phon_transcripts is not None:
        sources.append(args.phon_transcripts)
    if not args.no_word_full:
        sources.append(args.word_transcripts_full)

    age_map = load_age_bucket_map(sources)
    out_path = args.output or args.manifest
    tmp_path = Path(str(out_path) + ".tmp")

    n = n_miss = 0
    with open(args.manifest, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            uid = utterance_id_from_manifest_row(row)
            if uid and uid in age_map:
                ab = age_map[uid]
            else:
                ab = "unknown"
                n_miss += 1
            row["age_bucket"] = ab
            row["age_years"] = age_bucket_to_years(ab)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    os.replace(tmp_path, out_path)
    print(f"Wrote {n} rows to {out_path} ({n_miss} without transcript metadata -> unknown).")


if __name__ == "__main__":
    main()
