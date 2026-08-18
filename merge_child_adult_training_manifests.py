#!/usr/bin/env python3
"""
Concatenate NeMo JSONL manifests (e.g. child ASR + Libri adult) into one training file.

For each row, enforces ``age_cohort`` (``child`` | ``adult``) and ``is_child`` (bool) from the
source group so ``BALANCED_CHILD_ADULT_BATCHES=1`` can split batches. Rows must already carry
``age_years``, ``age_target``, or ``age`` for age-adversarial training.

CLI (two-file shortcut)::

    python merge_child_adult_training_manifests.py \\
      --child-manifest /path/child.jsonl \\
      --adult-manifest /path/libri_clean_age.jsonl \\
      --output /kaggle/working/merged_train.jsonl

General (any order, more than two files)::

    python merge_child_adult_training_manifests.py \\
      --output merged.jsonl \\
      --add /path/a.jsonl --cohort child \\
      --add /path/b.jsonl --cohort adult

File-level cohort **overwrites** existing ``age_cohort`` / ``is_child`` on each row (so Libri is
always adult and child corpus always child when you label files correctly).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

Cohort = Literal["child", "adult"]


@dataclass
class MergeStats:
    written: int = 0
    skipped_no_audio: int = 0
    skipped_no_text: int = 0
    skipped_no_age: int = 0
    skipped_no_duration: int = 0
    skipped_bad_json: int = 0


def _row_has_age(row: Dict[str, Any]) -> bool:
    if "age_target" in row:
        return True
    if "age_years" in row:
        return True
    if "age" in row:
        return True
    return False


def merge_labeled_manifests(
    sources: List[Tuple[str, Cohort]],
    output_path: str,
) -> MergeStats:
    """
    Concatenate manifests; each row from ``sources[i]`` gets ``sources[i][1]`` cohort tags.

    Returns aggregate stats. Skips rows missing ``audio_filepath`` (or aliases), ``text``, age field,
    or ``duration`` (required for NeMo ASR manifests).
    """
    stats = MergeStats()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with open(out_p, "w", encoding="utf-8") as out:
        for manifest_path, cohort in sources:
            mp = Path(manifest_path)
            if not mp.is_file():
                raise FileNotFoundError(f"Manifest not found: {mp}")
            is_child = cohort == "child"
            with open(mp, "r", encoding="utf-8") as inf:
                for line in inf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        stats.skipped_bad_json += 1
                        continue
                    if not isinstance(row, dict):
                        stats.skipped_bad_json += 1
                        continue
                    ap = row.get("audio_filepath") or row.get("audio_file") or row.get("audio_filename")
                    if not ap:
                        stats.skipped_no_audio += 1
                        continue
                    text = (row.get("text") or "").strip()
                    if not text:
                        stats.skipped_no_text += 1
                        continue
                    if not _row_has_age(row):
                        stats.skipped_no_age += 1
                        continue
                    if "duration" not in row:
                        stats.skipped_no_duration += 1
                        continue
                    row["audio_filepath"] = ap
                    row["text"] = text
                    row["age_cohort"] = cohort
                    row["is_child"] = is_child
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats.written += 1
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", "-o", type=str, required=True, help="Merged JSONL path")
    ap.add_argument("--child-manifest", type=str, default="", help="Child-cohort JSONL (shortcut)")
    ap.add_argument("--adult-manifest", type=str, default="", help="Adult-cohort JSONL (shortcut)")
    ap.add_argument("--add", action="append", default=[], metavar="PATH", help="Manifest path (repeatable)")
    ap.add_argument(
        "--cohort",
        action="append",
        default=[],
        choices=("child", "adult"),
        help="Cohort for the preceding --add order (use once per --add)",
    )
    args = ap.parse_args()

    sources: List[Tuple[str, Cohort]] = []

    if args.child_manifest.strip():
        sources.append((args.child_manifest.strip(), "child"))
    if args.adult_manifest.strip():
        sources.append((args.adult_manifest.strip(), "adult"))

    if args.add:
        if len(args.add) != len(args.cohort):
            print(
                "ERROR: repeat --add PATH and --cohort {child,adult} the same number of times.",
                file=sys.stderr,
            )
            return 1
        for p, c in zip(args.add, args.cohort):
            p = (p or "").strip()
            if not p:
                continue
            sources.append((p, c))

    if not sources:
        print(
            "ERROR: provide --child-manifest/--adult-manifest and/or paired --add/--cohort.",
            file=sys.stderr,
        )
        return 1

    stats = merge_labeled_manifests(sources, args.output)
    print(
        f"Wrote {stats.written} lines -> {args.output} "
        f"(skipped: no_audio={stats.skipped_no_audio} no_text={stats.skipped_no_text} "
        f"no_age={stats.skipped_no_age} no_duration={stats.skipped_no_duration} "
        f"bad_json={stats.skipped_bad_json})",
        flush=True,
    )
    if stats.written == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
