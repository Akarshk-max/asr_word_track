"""
Load Stage 1 ``model_stage1_fusion.nemo``, re-bind layer fusion forward, print fusion weights,
and compute WER on a NeMo JSON manifest (``audio_filepath``, ``text`` per line).

Run locally next to ``nemo_layer_fusion.py`` or set ``PYTHONPATH``. On Kaggle, set env:

  STAGE1_NEMO   — path to ``model_stage1_fusion.nemo`` (optional; else glob under stage1 dir)
  VAL_MANIFEST  — path to ``val_manifest.jsonl``
  BATCH_SIZE    — transcribe batch size (default 16)
  MAX_SAMPLES   — if set, only first N utterances (smoke test)
"""

from __future__ import annotations

import glob
import io
import json
import os
import sys
import tarfile
import warnings

warnings.filterwarnings("ignore", category=Warning, module="numba")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import logging

logging.getLogger("nemo_logger").setLevel(logging.ERROR)

import jiwer
import torch
from omegaconf import open_dict

from nemo.collections.asr.models import ASRModel

from nemo_layer_fusion import attach_layer_fusion_to_model

STAGE1_NEMO_EXPLICIT = os.environ.get(
    "STAGE1_NEMO",
    "/kaggle/working/nemo_layer_fusion_stage1/model_stage1_fusion.nemo",
)
VAL_MANIFEST = os.environ.get(
    "VAL_MANIFEST",
    "/kaggle/working/nemo_layer_fusion_stage1/manifests/val_manifest.jsonl",
)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
MAX_SAMPLES = os.environ.get("MAX_SAMPLES", "").strip()
MAX_SAMPLES_N = int(MAX_SAMPLES) if MAX_SAMPLES.isdigit() else None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_stage1_nemo() -> str:
    if os.path.isfile(STAGE1_NEMO_EXPLICIT):
        return os.path.abspath(STAGE1_NEMO_EXPLICIT)
    search_roots = [
        "/kaggle/working/nemo_layer_fusion_stage1",
        os.path.join(SCRIPT_DIR, "nemo_layer_fusion_stage1"),
        os.path.join(os.getcwd(), "nemo_layer_fusion_stage1"),
    ]
    matches: list[str] = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        matches.extend(
            glob.glob(
                os.path.join(root, "**", "model_stage1_fusion.nemo"),
                recursive=True,
            )
        )
        matches.extend(
            glob.glob(
                os.path.join(root, "**", "model_stage1_step*.nemo"),
                recursive=True,
            )
        )
    matches = sorted(set(matches), key=os.path.getmtime, reverse=True)
    final = [p for p in matches if os.path.basename(p) == "model_stage1_fusion.nemo"]
    if final:
        return max(final, key=os.path.getmtime)
    step = [p for p in matches if os.path.basename(p).startswith("model_stage1_step")]
    if step:
        best = max(step, key=os.path.getmtime)
        print(f"[resolve] Using newest step checkpoint: {best}", flush=True)
        return best
    raise FileNotFoundError(
        "Could not find model_stage1_fusion.nemo or model_stage1_step*.nemo. Set STAGE1_NEMO."
    )


def _load_manifest_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _hyp_to_text(h) -> str:
    if h is None:
        return ""
    if isinstance(h, str):
        return h.strip().lower()
    t = getattr(h, "text", None)
    if t is not None:
        return str(t).strip().lower()
    return str(h).strip().lower()


def _transcribe_paths(model: ASRModel, paths: list[str]) -> list[str]:
    try:
        raw = model.transcribe(audio=paths, batch_size=BATCH_SIZE)
    except TypeError:
        raw = model.transcribe(paths2audio_files=paths, batch_size=BATCH_SIZE)
    if isinstance(raw, tuple):
        raw = raw[0]
    return [_hyp_to_text(h) for h in raw]


def _state_dict_from_nemo(nemo_path: str) -> dict:
    with tarfile.open(nemo_path, "r:*") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            if not (
                m.name.endswith(".ckpt")
                or m.name.endswith(".pth")
                or "model_weights" in m.name.replace("\\", "/")
            ):
                continue
            try:
                with tar.extractfile(m) as f:
                    blob = f.read()
                obj = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                return obj["state_dict"]
            if any(
                isinstance(k, str)
                and (
                    k.startswith("encoder.")
                    or k.startswith("decoder.")
                    or k.startswith("layer_fusion.")
                )
                for k in obj
            ):
                return obj
    raise RuntimeError(
        f"Could not read state_dict from {nemo_path} (no suitable .ckpt in .nemo tar)."
    )


def _load_layer_fusion_weights_from_nemo(model: torch.nn.Module, nemo_path: str) -> None:
    fusion = getattr(model, "layer_fusion", None)
    if fusion is None:
        print("[warn] No layer_fusion — skip fusion weight load.", flush=True)
        return
    sd = _state_dict_from_nemo(nemo_path)
    prefix = "layer_fusion."
    lf = {
        k[len(prefix) :]: v
        for k, v in sd.items()
        if isinstance(k, str) and k.startswith(prefix)
    }
    if not lf:
        print(
            "[warn] No layer_fusion.* in checkpoint — fusion at default init.",
            flush=True,
        )
        return
    dev = next(fusion.parameters()).device
    lf = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in lf.items()}
    fusion.load_state_dict(lf, strict=True)
    print(f"Loaded fusion submodule ({len(lf)} tensor(s)).", flush=True)


def _print_fusion_inspection(model: ASRModel) -> None:
    fusion = getattr(model, "layer_fusion", None)
    if fusion is None:
        print("No ``layer_fusion`` submodule on model — fusion weights N/A.")
        return
    raw = fusion.raw_weights.detach().float().cpu()
    w = torch.softmax(raw, dim=0).numpy()
    n = len(w)
    print(f"\nLayer fusion ({n} encoder layers)")
    print(f"  raw_weights (pre-softmax): {raw.numpy().tolist()}")
    print("  per-layer softmax weights (importance):")
    order = sorted(range(n), key=lambda i: w[i], reverse=True)
    for rank, i in enumerate(order):
        bar = "#" * max(1, int(w[i] * 50))
        print(f"    layer {i:3d}  p={w[i]:.6f}  raw={raw[i].item():.4f}  {bar}")
    top5 = order[:5]
    print(f"  top-5 layers by weight: {top5}")


def main() -> None:
    nemo_path = _resolve_stage1_nemo()
    if not os.path.isfile(VAL_MANIFEST):
        raise FileNotFoundError(
            f"VAL_MANIFEST not found: {VAL_MANIFEST}. Set env VAL_MANIFEST to your val_manifest.jsonl."
        )

    print(f"Device: {DEVICE}")
    print(f"Restore: {nemo_path}")
    print(f"Manifest: {VAL_MANIFEST}")

    try:
        model = ASRModel.restore_from(
            nemo_path, map_location=torch.device(DEVICE), strict=False
        )
    except TypeError:
        model = ASRModel.restore_from(nemo_path, map_location=torch.device(DEVICE))
    model = model.to(DEVICE)
    model.eval()

    with open_dict(model.cfg):
        if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
            model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    if hasattr(model, "change_decoding_strategy"):
        model.change_decoding_strategy(model.cfg.decoding)

    print(
        "Re-binding fusion forward; loading layer_fusion.* from archive (strict=False restore).",
        flush=True,
    )
    attach_layer_fusion_to_model(model, init_strategy="last_layer", module_name="layer_fusion")
    _load_layer_fusion_weights_from_nemo(model, nemo_path)
    _print_fusion_inspection(model)

    rows = _load_manifest_rows(VAL_MANIFEST)
    paths: list[str] = []
    refs: list[str] = []
    for r in rows:
        ap = r.get("audio_filepath") or r.get("audio_path")
        text = (r.get("text") or "").strip().lower()
        if not ap or not text:
            continue
        if not os.path.isfile(ap):
            continue
        paths.append(ap)
        refs.append(text)

    if MAX_SAMPLES_N is not None:
        paths = paths[:MAX_SAMPLES_N]
        refs = refs[:MAX_SAMPLES_N]

    if not paths:
        raise RuntimeError(
            "No usable rows (need audio_filepath + text, file must exist). Check paths in manifest."
        )

    print(f"\nTranscribing {len(paths):,} utterances (batch_size={BATCH_SIZE})...")
    preds: list[str] = []
    for start in range(0, len(paths), BATCH_SIZE):
        chunk = paths[start : start + BATCH_SIZE]
        preds.extend(_transcribe_paths(model, chunk))

    valid_refs: list[str] = []
    valid_preds: list[str] = []
    for ref, pred in zip(refs, preds):
        if not ref.strip():
            continue
        valid_refs.append(ref)
        valid_preds.append(pred if pred.strip() else "<empty>")

    wer = jiwer.wer(valid_refs, valid_preds)
    print(f"\nWER on manifest: {wer:.4f}  (n={len(valid_refs):,})")

    n_show = min(5, len(valid_preds))
    if n_show:
        print("\nSample predictions:")
        for i in range(n_show):
            print(f"  P: {valid_preds[i]}")
            print(f"  T: {valid_refs[i]}\n")


if __name__ == "__main__":
    main()
