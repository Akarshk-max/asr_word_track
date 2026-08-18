# -*- coding: utf-8 -*-
"""
Optional thin entrypoint: runs the full one-cell script from disk.

The canonical Kaggle cell is **train_nemo_adapter_1.1b.py** (Part A pip + Part B train_code
string + Part C subprocess). Paste that file into a notebook cell, or run:

  python train_nemo_adapter_1.1b.py

This wrapper only locates train_nemo_adapter_1.1b.py next to this file, under
/kaggle/working, or cwd, then execs it in a subprocess (fresh interpreter after your
own optional pip cell).
"""

import os
import subprocess
import sys


def _find_main():
    cands = []
    if "__file__" in globals():
        cands.append(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_nemo_adapter_1.1b.py")
        )
    cands.extend(
        [
            "/kaggle/working/train_nemo_adapter_1.1b.py",
            os.path.join(os.getcwd(), "train_nemo_adapter_1.1b.py"),
        ]
    )
    for p in cands:
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        "train_nemo_adapter_1.1b.py not found. Copy it from the ASR repo next to this script "
        "or to /kaggle/working/."
    )


if __name__ == "__main__":
    main_py = _find_main()
    r = subprocess.run(
        [sys.executable, main_py],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        cwd=os.path.dirname(main_py) or ".",
    )
    raise SystemExit(r.returncode)
