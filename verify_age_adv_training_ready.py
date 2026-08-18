#!/usr/bin/env python3
"""Lightweight checks before / after age-adversarial training (Kaggle optional subprocess)."""
from __future__ import annotations

import json
import os
import sys


def main() -> int:
    manifest = (os.environ.get("TRAIN_MANIFEST") or os.environ.get("VERIFY_MANIFEST", "")).strip()
    if manifest:
        if not os.path.isfile(manifest):
            print(f"FAIL: manifest not found: {manifest}", file=sys.stderr)
            return 1
        n = 0
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    n += 1
                except json.JSONDecodeError:
                    print(f"FAIL: invalid JSONL line in {manifest}", file=sys.stderr)
                    return 1
        print(f"OK: manifest {manifest} has {n} JSON lines")
    else:
        print("SKIP: no TRAIN_MANIFEST / VERIFY_MANIFEST set")

    try:
        import torch  # noqa: F401
    except ImportError:
        print("FAIL: torch not importable", file=sys.stderr)
        return 1
    try:
        from nemo.collections.asr.models import ASRModel  # noqa: F401
    except ImportError as e:
        print(f"FAIL: NeMo ASR import: {e}", file=sys.stderr)
        return 1
    print("OK: torch + nemo ASR import")

    try:
        import parakeet_age_adv_nemo_extras  # noqa: F401
    except ImportError as e:
        print(f"WARN: parakeet_age_adv_nemo_extras: {e}", file=sys.stderr)
    else:
        print("OK: parakeet_age_adv_nemo_extras import")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
