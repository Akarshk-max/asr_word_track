# -*- coding: utf-8 -*-
"""
Kaggle NOTEBOOK cell 2 — run AFTER your helper cell that defines
``attach_layer_fusion_to_model`` (and related classes).

Requires the same NeMo stack as training (reuse kernel + ``SKIP_PIP=1`` on the
full infer one-cell driver, or run training install first).

Set env in the notebook before this cell if needed:
  os.environ["STAGE1_NEMO"] = "/kaggle/working/.../model_stage1_fusion.nemo"
  os.environ["VAL_MANIFEST"] = "/path/to/val_manifest.jsonl"
  os.environ["BATCH_SIZE"] = "16"
  os.environ["MAX_SAMPLES"] = "200"   # optional
"""

from __future__ import annotations

import glob
import io
import json
import logging
import os
import tarfile
import warnings

warnings.filterwarnings("ignore", category=Warning, module="numba")
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

import jiwer
import torch
from omegaconf import open_dict

from nemo.collections.asr.models import ASRModel

if "attach_layer_fusion_to_model" not in globals():
    raise RuntimeError(
        "Run your fusion helper cell first so attach_layer_fusion_to_model is defined."
    )

STAGE1_NEMO_EXPLICIT = os.environ.get(
    "STAGE1_NEMO",
    "/kaggle/working/nemo_layer_fusion_stage1/model_stage1_fusion.nemo",
)
_val_env = os.environ.get("VAL_MANIFEST", "").strip()
if _val_env:
    VAL_MANIFEST = _val_env
else:
    _candidates = [
        "/kaggle/working/nemo_layer_fusion_stage1/manifests/val_manifest.jsonl",
        "/kaggle/input/datasets/akarshks/val-manifest/val_manifest.jsonl",
    ]
    VAL_MANIFEST = next((p for p in _candidates if os.path.isfile(p)), _candidates[0])

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
MAX_SAMPLES = os.environ.get("MAX_SAMPLES", "").strip()
MAX_SAMPLES_N = int(MAX_SAMPLES) if MAX_SAMPLES.isdigit() else None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_stage1_nemo() -> str:
    if os.path.isfile(STAGE1_NEMO_EXPLICIT):
        return os.path.abspath(STAGE1_NEMO_EXPLICIT)
    roots = ["/kaggle/working/nemo_layer_fusion_stage1", "/kaggle/working"]
    matches: list[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        matches.extend(
            glob.glob(os.path.join(root, "**", "model_stage1_fusion.nemo"), recursive=True)
        )
        matches.extend(
            glob.glob(os.path.join(root, "**", "model_stage1_step*.nemo"), recursive=True)
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
        "Set STAGE1_NEMO to your .nemo path (check .../checkpoints/ under exp_manager output)."
    )


def _hyp_to_text(h) -> str:
    if h is None:
        return ""
    if isinstance(h, str):
        return h.strip().lower()
    t = getattr(h, "text", None)
    if t is not None:
        return str(t).strip().lower()
    return str(h).strip().lower()


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
    raise RuntimeError(f"No state_dict in {nemo_path}")


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
        print("[warn] No layer_fusion.* in checkpoint.", flush=True)
        return
    dev = next(fusion.parameters()).device
    lf = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in lf.items()}
    fusion.load_state_dict(lf, strict=True)
    print(f"Loaded fusion submodule ({len(lf)} tensor(s)).", flush=True)


nemo_path = _resolve_stage1_nemo()
if not os.path.isfile(VAL_MANIFEST):
    raise FileNotFoundError(f"VAL_MANIFEST: {VAL_MANIFEST}")

print(f"torch {torch.__version__}  CUDA {torch.version.cuda}  {DEVICE}")
print("Restore:", nemo_path)
print("Manifest:", VAL_MANIFEST)

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

attach_layer_fusion_to_model(model, init_strategy="last_layer", module_name="layer_fusion")
_load_layer_fusion_weights_from_nemo(model, nemo_path)

fusion = getattr(model, "layer_fusion", None)
if fusion is not None:
    raw = fusion.raw_weights.detach().float().cpu()
    w = torch.softmax(raw, dim=0).numpy()
    print(f"\n=== Layer fusion ({len(w)} layers) ===")
    print("raw_weights:", raw.numpy().tolist())
    order = sorted(range(len(w)), key=lambda i: w[i], reverse=True)
    for i in order:
        bar = "#" * max(1, int(w[i] * 50))
        print(f"  layer {i:3d}  p={w[i]:.6f}  {bar}")
    print("top-5:", order[:5])

paths, refs = [], []
with open(VAL_MANIFEST, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ap = r.get("audio_filepath") or r.get("audio_path")
        text = (r.get("text") or "").strip().lower()
        if ap and text and os.path.isfile(ap):
            paths.append(ap)
            refs.append(text)

if MAX_SAMPLES_N is not None:
    paths = paths[:MAX_SAMPLES_N]
    refs = refs[:MAX_SAMPLES_N]

print(f"\nTranscribing {len(paths):,} files...")
preds: list[str] = []
for start in range(0, len(paths), BATCH_SIZE):
    chunk = paths[start : start + BATCH_SIZE]
    try:
        raw = model.transcribe(audio=chunk, batch_size=BATCH_SIZE)
    except TypeError:
        raw = model.transcribe(paths2audio_files=chunk, batch_size=BATCH_SIZE)
    if isinstance(raw, tuple):
        raw = raw[0]
    preds.extend(_hyp_to_text(h) for h in raw)

vr, vp = [], []
for ref, pred in zip(refs, preds):
    if ref.strip():
        vr.append(ref)
        vp.append(pred if pred.strip() else "<empty>")

wer = jiwer.wer(vr, vp)
print(f"\n=== WER: {wer:.4f} (n={len(vr):,}) ===")
for i in range(min(5, len(vp))):
    print(f"  P: {vp[i]}\n  T: {vr[i]}\n")
