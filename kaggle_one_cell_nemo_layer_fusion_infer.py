# -*- coding: utf-8 -*-
"""
Kaggle: Stage 1 layer-fusion inference — layer-weight report + val WER.

Layer outputs use forward hooks on ``encoder.layers`` (not InterCTC), so ``model.transcribe()``
works. Older ``infer-r5`` cells that read the InterCTC registry fail under transcribe with an
empty registry. Revision ``infer-r7`` applies ``encoder.out_proj`` to hook outputs so fused
features match the joint (fixes 1024 vs projected-dim matmul errors).

Paste this entire file into ONE notebook cell, OR use ``SKIP_PIP=1`` in the same kernel
after the training install cell.

  Part A — pip (same pins as training; set SKIP_PIP=1 to skip)
  Part B — write ``/kaggle/working/infer_layer_fusion_stage1_standalone.py`` (fusion inlined)
  Part C — ``subprocess.run`` fresh Python

Env (optional, inherited by child):
  SKIP_PIP=1              Skip Part A (NeMo already installed)
  STAGE1_NEMO=...         Full path to .nemo (else auto-find under /kaggle/working)
  VAL_MANIFEST=...        NeMo JSONL with audio_filepath + text
  BATCH_SIZE=16
  MAX_SAMPLES=500         Limit utterances for a quick run (digits only)

If you already run a **helper cell** with ``attach_layer_fusion_to_model`` in the notebook,
see ``kaggle_notebook_infer_cell_using_helpers.py`` instead (no subprocess).

**Do not** hand-edit the embedded ``infer_code`` string in a copy-paste (broken
``from __future__`` / ``__init__`` breaks the child script). Use this file from the repo
or upload it whole to Kaggle.
"""

import os
import subprocess
import sys


def _pip(*args):
    subprocess.check_call(
        [sys.executable, "-m", "pip", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


SKIP_PIP = os.environ.get("SKIP_PIP", "0").strip().lower() in ("1", "true", "yes")

if not SKIP_PIP:
    # =============================================================================
    # PART A: INSTALLATION (match kaggle_one_cell_nemo_layer_fusion.py)
    # =============================================================================
    print("Step 1: Cleaning...")
    for _ in range(2):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "numpy",
                "scipy",
                "nemo_toolkit",
                "lightning",
                "pytorch-lightning",
                "datasets",
                "diffusers",
                "gradio",
                "peft",
                "sentence-transformers",
                "transformers",
                "huggingface_hub",
                "torch",
                "torchaudio",
                "torchvision",
                "numba",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    print("Step 2: numpy + scipy...")
    _pip("install", "--no-cache-dir", "numpy>=2.1,<2.3", "scipy>=1.14,<1.16")

    print("Step 3: PyTorch (CUDA 12.6 wheels)...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
            "torch>=2.9.0",
            "torchaudio",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    print("Step 4: Dependencies...")
    _pip(
        "install",
        "--no-cache-dir",
        "transformers>=4.57.6,<4.58",
        "huggingface_hub>=0.30.0",
        "lightning>=2.2.0",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.2",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        "sentencepiece>=0.2.0",
        "datasets>=2.18.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.4.0",
        "loguru>=0.7.0",
        "jiwer>=3.0.0",
        "tqdm>=4.60.0",
        "webdataset>=0.2.80",
        "braceexpand>=0.1.7",
        "editdistance>=0.6.0",
    )

    print("Step 5: NeMo...")
    _pip("install", "--no-cache-dir", "nemo_toolkit[asr]>=2.5.0")

    print("Step 6: Re-pin numpy...")
    _pip(
        "install",
        "--no-cache-dir",
        "--force-reinstall",
        "numpy>=2.1,<2.3",
        "scipy>=1.14,<1.16",
    )

    print("Installation complete.\n")
else:
    print("SKIP_PIP=1 — skipping pip (using current kernel packages).\n")

# =============================================================================
# PART B: STANDALONE INFERENCE SCRIPT (fusion inlined — no import from notebook)
# =============================================================================
INFER_SCRIPT = "/kaggle/working/infer_layer_fusion_stage1_standalone.py"

infer_code = r'''from __future__ import annotations

import glob
import io
import json
import logging
import os
import tarfile
import time
import types
import warnings
from typing import List, Literal

warnings.filterwarnings("ignore", category=Warning, module="numba")
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

import jiwer
import librosa
import torch
import torch.nn as nn
from omegaconf import open_dict

from nemo.collections.asr.models import ASRModel

SCRIPT_REV = "infer-r7-out-proj-bdt"
InitStrategy = Literal["last_layer", "uniform", "edges"]


class LearnableLayerFusion(nn.Module):
    def __init__(self, num_layers: int, init_strategy: InitStrategy = "last_layer"):
        super().__init__()
        self.num_layers = num_layers
        self.raw_weights = nn.Parameter(torch.zeros(num_layers))
        if init_strategy == "last_layer":
            self.raw_weights.data[-1] = 5.0
        elif init_strategy == "uniform":
            self.raw_weights.data.zero_()
        elif init_strategy == "edges":
            for i in range(num_layers):
                if i < 4 or i >= num_layers - 4:
                    self.raw_weights.data[i] = 2.0
        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")

    def forward(self, layer_tensors: List[torch.Tensor]) -> torch.Tensor:
        if len(layer_tensors) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layer tensors, got {len(layer_tensors)}"
            )
        w = torch.softmax(self.raw_weights, dim=0)
        stacked = torch.stack(layer_tensors, dim=0)
        return (stacked * w.view(-1, 1, 1, 1)).sum(dim=0)

    @torch.no_grad()
    def importance(self) -> torch.Tensor:
        return torch.softmax(self.raw_weights, dim=0)


def _layer_tensor_from_module_output(out):
    return out[0] if isinstance(out, tuple) else out


def _encoder_out_proj_module(encoder: nn.Module):
    for mod in (encoder, getattr(encoder, "encoder", None)):
        if mod is None:
            continue
        op = getattr(mod, "out_proj", None)
        if isinstance(op, nn.Module):
            return op
    return None


def _conformer_hook_to_fusion_bdt(encoder: nn.Module, hook_out: torch.Tensor, encoded_ref: torch.Tensor):
    """Match NeMo interctc: layer (B,T,d_in) -> optional out_proj -> (B,d_out,T)."""
    t = hook_out
    if t.dim() != 3:
        return t
    op = _encoder_out_proj_module(encoder)
    target_feats = encoded_ref.shape[1]
    if op is not None:
        inf = int(getattr(op, "in_features", 0) or 0)
        if inf > 0:
            if t.shape[2] == inf:
                pass
            elif t.shape[1] == inf:
                t = t.transpose(1, 2)
        t = op(t)
        out = t.transpose(1, 2)
    else:
        if t.shape[1] == target_feats:
            out = t
        elif t.shape[2] == target_feats:
            out = t.transpose(1, 2)
        else:
            out = t.transpose(1, 2)
    if out.shape[1] != target_feats and out.shape[2] == target_feats:
        out = out.transpose(1, 2)
    return out


def _encoder_forward_with_layer_hooks(encoder, audio_signal, length, num_layers):
    layer_outs: List[torch.Tensor] = []
    hooks: List = []

    def _hook(_m, _inp, out):
        layer_outs.append(_layer_tensor_from_module_output(out))

    try:
        for lyr in encoder.layers:
            hooks.append(lyr.register_forward_hook(_hook))
        encoded, encoded_len = encoder(audio_signal=audio_signal, length=length)
    finally:
        for h in hooks:
            h.remove()

    if len(layer_outs) != num_layers:
        raise RuntimeError(
            f"Layer fusion hooks captured {len(layer_outs)} outputs, expected {num_layers}."
        )
    return encoded, encoded_len, layer_outs


def attach_layer_fusion_to_model(
    model: nn.Module,
    *,
    init_strategy: InitStrategy = "last_layer",
    module_name: str = "layer_fusion",
) -> LearnableLayerFusion:
    if not hasattr(model, "encoder") or not hasattr(model, "preprocessor"):
        raise TypeError("model must expose .encoder and .preprocessor (NeMo ASR).")

    enc = model.encoder
    if not hasattr(enc, "layers"):
        raise TypeError("model.encoder must have .layers (Conformer-style encoder).")

    num_layers = len(enc.layers)
    existing = getattr(model, module_name, None)
    if isinstance(existing, LearnableLayerFusion):
        if existing.num_layers != num_layers:
            raise ValueError(
                f"Checkpoint fusion layers {existing.num_layers} != encoder {num_layers}"
            )
        fusion = existing
    else:
        fusion = LearnableLayerFusion(num_layers, init_strategy=init_strategy)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        fusion = fusion.to(device=device, dtype=dtype)
        model.add_module(module_name, fusion)

    _fusion_dbg_flags = {"entry": False, "reg": False}

    def _fused_forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = (
            processed_signal is not None and processed_signal_length is not None
        )
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments input_signal / input_signal_length are mutually "
                "exclusive with processed_signal / processed_signal_length."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(
                input_spec=processed_signal, length=processed_signal_length
            )

        guid = getattr(self, "model_guid", None)
        enc_guid = getattr(self.encoder, "model_guid", None)
        fusion_mod: LearnableLayerFusion = getattr(self, module_name)

        # #region agent log
        def _agent_dbg(hypothesis_id, location, message, data):
            if os.environ.get("AGENT_DEBUG_LOG", "1").strip().lower() in (
                "0",
                "false",
                "no",
            ):
                return
            payload = {
                "sessionId": "41b2c4",
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000),
            }
            line = json.dumps(payload, default=str) + "\n"
            for _p in (
                os.path.join(os.getcwd(), "debug-41b2c4.log"),
                "/kaggle/working/debug-41b2c4.log",
            ):
                try:
                    with open(_p, "a", encoding="utf-8") as _df:
                        _df.write(line)
                    break
                except OSError:
                    continue

        if not _fusion_dbg_flags["entry"]:
            _agent_dbg(
                "H1",
                "infer_standalone:_fused_forward:entry",
                "before encoder (hook capture)",
                {
                    "torch_is_inference_mode": torch.is_inference_mode_enabled(),
                    "grad_enabled": torch.is_grad_enabled(),
                    "self_training": bool(self.training),
                    "model_guid_is_none": guid is None,
                    "encoder_guid_is_none": enc_guid is None,
                    "guid_equal": guid == enc_guid,
                    "encoder_cls": type(self.encoder).__name__,
                    "script_rev": SCRIPT_REV,
                },
            )
            _fusion_dbg_flags["entry"] = True

        # #endregion

        try:
            encoded, encoded_len, raw_layer_tensors = _encoder_forward_with_layer_hooks(
                self.encoder,
                processed_signal,
                processed_signal_length,
                fusion_mod.num_layers,
            )
            layer_tensors = [
                _conformer_hook_to_fusion_bdt(self.encoder, t, encoded)
                for t in raw_layer_tensors
            ]
            # #region agent log
            if not _fusion_dbg_flags["reg"]:
                _reg = getattr(self.encoder, "_registry", None)
                _ik = (
                    [k for k in _reg if str(k).startswith("interctc/")]
                    if isinstance(_reg, dict)
                    else []
                )
                _agent_dbg(
                    "H7",
                    "infer_standalone:after_encoder",
                    "hook layer capture (+ registry snapshot)",
                    {
                        "n_hook_layers": len(layer_tensors),
                        "expected_layers": fusion_mod.num_layers,
                        "n_interctc_keys": len(_ik),
                        "model_guid": str(guid),
                        "encoder_guid": str(enc_guid),
                        "script_rev": SCRIPT_REV,
                    },
                )
                _fusion_dbg_flags["reg"] = True
            # #endregion
            fused = fusion_mod(layer_tensors)
        finally:
            rr = getattr(self.encoder, "reset_registry", None)
            if callable(rr):
                rr()

        del encoded
        return fused, encoded_len

    model.forward = types.MethodType(_fused_forward, model)
    return fusion


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


def _discover_stage1_nemo_paths() -> list[str]:
    """NeMo exp_manager nests checkpoints under .../<name>/<date>/checkpoints/."""
    roots = [
        "/kaggle/working/nemo_layer_fusion_stage1",
        "/kaggle/working",
    ]
    found: list[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        found.extend(
            glob.glob(os.path.join(root, "**", "model_stage1_fusion.nemo"), recursive=True)
        )
        found.extend(
            glob.glob(os.path.join(root, "**", "model_stage1_step*.nemo"), recursive=True)
        )
    return sorted(set(found), key=os.path.getmtime, reverse=True)


def _resolve_stage1_nemo() -> str:
    if os.path.isfile(STAGE1_NEMO_EXPLICIT):
        return os.path.abspath(STAGE1_NEMO_EXPLICIT)
    candidates = _discover_stage1_nemo_paths()
    final_name = [p for p in candidates if os.path.basename(p) == "model_stage1_fusion.nemo"]
    if final_name:
        return max(final_name, key=os.path.getmtime)
    step_ckpts = [
        p for p in candidates if os.path.basename(p).startswith("model_stage1_step")
    ]
    if step_ckpts:
        best = max(step_ckpts, key=os.path.getmtime)
        print(
            f"[resolve] Using mid-train checkpoint (newest step file): {best}",
            flush=True,
        )
        return best
    hints: list[str] = []
    hr = "/kaggle/working/nemo_layer_fusion_stage1"
    if os.path.isdir(hr):
        for dp, _, fns in os.walk(hr):
            for fn in fns:
                if fn.endswith(".nemo"):
                    hints.append(os.path.join(dp, fn))
            if len(hints) >= 35:
                break
    msg = (
        "Could not find model_stage1_fusion.nemo or model_stage1_step*.nemo under "
        "/kaggle/working. Training writes under exp subfolders, e.g. "
        ".../ParakeetAdapterLayerFusion/<timestamp>/checkpoints/. "
        "Set STAGE1_NEMO to the full .nemo path."
    )
    if hints:
        msg += "\n.nemo files under nemo_layer_fusion_stage1:\n  " + "\n  ".join(
            hints[:30]
        )
    else:
        msg += (
            "\n(No .nemo under /kaggle/working/nemo_layer_fusion_stage1 — run Stage 1 in "
            "this workspace or add your checkpoint as a dataset and set STAGE1_NEMO.)"
        )
    raise FileNotFoundError(msg)


def _load_manifest_rows(path: str):
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


def _agent_dbg_global(hypothesis_id, location, message, data):
    # #region agent log
    if os.environ.get("AGENT_DEBUG_LOG", "1").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return
    payload = {
        "sessionId": "41b2c4",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    line = json.dumps(payload, default=str) + "\n"
    for _p in (
        os.path.join(os.getcwd(), "debug-41b2c4.log"),
        "/kaggle/working/debug-41b2c4.log",
    ):
        try:
            with open(_p, "a", encoding="utf-8") as _df:
                _df.write(line)
            break
        except OSError:
            continue
    # #endregion


def _hyp_to_text(h) -> str:
    if h is None:
        return ""
    if isinstance(h, str):
        return h.strip().lower()
    t = getattr(h, "text", None)
    if t is not None:
        return str(t).strip().lower()
    return str(h).strip().lower()


def _transcribe_call(model, paths: List[str], bs: int):
    try:
        return model.transcribe(audio=paths, batch_size=bs)
    except TypeError as e:
        # Only fallback when API truly does not accept ``audio=``.
        if "unexpected keyword argument 'audio'" not in str(e):
            raise
        return model.transcribe(paths2audio_files=paths, batch_size=bs)


def _is_channel_shape_error(exc: Exception) -> bool:
    s = str(exc)
    return (
        "Input shape mismatch occured for input_signal" in s
        or "Dimension out of range" in s
        or "collate_audio" in s
        or "Channel selector average not found in cut.custom" in s
    )


def _transcribe_paths(model, paths: List[str]) -> List[str]:
    # #region agent log
    _agent_dbg_global(
        "H4",
        "infer_standalone:_transcribe_paths:start",
        "starting batch transcribe",
        {
            "n_paths": len(paths),
            "sample_paths": paths[:3],
            "batch_size": BATCH_SIZE,
        },
    )
    # #endregion
    try:
        raw = _transcribe_call(model, paths, BATCH_SIZE)
    except Exception as e:
        # Fallback for mixed/stereo clips when NeMo/Lhotse collation path fails.
        if _is_channel_shape_error(e):
            # #region agent log
            _agent_dbg_global(
                "H6",
                "infer_standalone:_transcribe_paths:mono_fallback",
                "channel-shape error; retrying with librosa mono arrays",
                {"error": repr(e), "n_paths": len(paths)},
            )
            # #endregion
            mono_audio = []
            for p in paths:
                y, _sr = librosa.load(p, sr=16000, mono=True)
                mono_audio.append(y)
            raw = model.transcribe(audio=mono_audio, batch_size=BATCH_SIZE)
            if isinstance(raw, tuple):
                raw = raw[0]
            # #region agent log
            _agent_dbg_global(
                "H6",
                "infer_standalone:_transcribe_paths:mono_fallback_done",
                "mono fallback succeeded",
                {"n_preds": len(raw), "n_paths": len(paths)},
            )
            # #endregion
            return [_hyp_to_text(h) for h in raw]
        # #region agent log
        _agent_dbg_global(
            "H5",
            "infer_standalone:_transcribe_paths:batch_exception",
            "batch transcribe exception; probing per-file",
            {"error": repr(e), "n_paths": len(paths), "sample_paths": paths[:5]},
        )
        # #endregion
        bad = []
        for p in paths:
            try:
                _ = _transcribe_call(model, [p], 1)
            except Exception as e1:
                bad.append({"path": p, "error": repr(e1)})
                if len(bad) >= 5:
                    break
        # #region agent log
        _agent_dbg_global(
            "H4",
            "infer_standalone:_transcribe_paths:per_file_probe",
            "per-file probe results",
            {"n_bad": len(bad), "bad_examples": bad},
        )
        # #endregion
        raise
    if isinstance(raw, tuple):
        raw = raw[0]
    return [_hyp_to_text(h) for h in raw]


def _print_fusion_report(model) -> None:
    fusion = getattr(model, "layer_fusion", None)
    if fusion is None:
        print("No layer_fusion submodule — weights N/A.")
        return
    raw = fusion.raw_weights.detach().float().cpu()
    w = torch.softmax(raw, dim=0).numpy()
    n = len(w)
    print(f"\n=== Layer fusion ({n} encoder layers) ===")
    print(f"raw_weights (pre-softmax): {raw.numpy().tolist()}")
    print("softmax weights (sorted by importance):")
    order = sorted(range(n), key=lambda i: w[i], reverse=True)
    for i in order:
        bar = "#" * max(1, int(w[i] * 50))
        print(f"  layer {i:3d}  p={w[i]:.6f}  raw={raw[i].item():.4f}  {bar}")
    print(f"top-5 layers: {order[:5]}")
    print(f"importance() tensor: {fusion.importance().cpu().numpy().tolist()}")


def _state_dict_from_nemo(nemo_path: str) -> dict:
    """Read PyTorch state_dict from inside a NeMo .nemo (tar archive)."""
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
        f"Could not read state_dict from {nemo_path} (no suitable .ckpt inside .nemo tar)."
    )


def _load_layer_fusion_weights_from_nemo(model: torch.nn.Module, nemo_path: str) -> None:
    fusion = getattr(model, "layer_fusion", None)
    if fusion is None:
        print("[warn] No layer_fusion submodule — skip fusion weight load.", flush=True)
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
            "[warn] Checkpoint has no layer_fusion.* keys — fusion left at init_strategy.",
            flush=True,
        )
        return
    dev = next(fusion.parameters()).device
    lf = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in lf.items()}
    fusion.load_state_dict(lf, strict=True)
    print(f"Loaded fusion submodule weights ({len(lf)} tensor(s)).", flush=True)


def main() -> None:
    nemo_path = _resolve_stage1_nemo()
    if not os.path.isfile(VAL_MANIFEST):
        raise FileNotFoundError(
            f"VAL_MANIFEST not found: {VAL_MANIFEST}. "
            "Set env VAL_MANIFEST to a NeMo JSONL (audio_filepath, text)."
        )

    print(f"[SCRIPT_REV] {SCRIPT_REV}")
    print(f"torch {torch.__version__}  CUDA {torch.version.cuda}  device {DEVICE}")
    print(f"Restore: {nemo_path}")
    print(f"Manifest: {VAL_MANIFEST}")

    # EncDecRNNT config has no layer_fusion; save_to still stores layer_fusion.raw_weights.
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
        "Re-binding layer fusion forward, then loading layer_fusion.* from .nemo "
        "(strict restore skips unknown keys).",
        flush=True,
    )
    attach_layer_fusion_to_model(model, init_strategy="last_layer", module_name="layer_fusion")
    _load_layer_fusion_weights_from_nemo(model, nemo_path)
    _print_fusion_report(model)

    rows = _load_manifest_rows(VAL_MANIFEST)
    paths: List[str] = []
    refs: List[str] = []
    for r in rows:
        ap = r.get("audio_filepath") or r.get("audio_path")
        text = (r.get("text") or "").strip().lower()
        if not ap or not text:
            continue
        if not os.path.isfile(ap):
            continue
        paths.append(ap)
        refs.append(text)
    # #region agent log
    _agent_dbg_global(
        "H4",
        "infer_standalone:manifest_filtered",
        "manifest rows filtered for usable audio/text",
        {"n_total_rows": len(rows), "n_usable": len(paths), "sample_paths": paths[:3]},
    )
    # #endregion

    if MAX_SAMPLES_N is not None:
        paths = paths[:MAX_SAMPLES_N]
        refs = refs[:MAX_SAMPLES_N]

    if not paths:
        raise RuntimeError(
            "No usable rows (audio_filepath + text, file must exist on disk)."
        )

    print(f"\nTranscribing {len(paths):,} utterances (batch_size={BATCH_SIZE})...")
    preds: List[str] = []
    for start in range(0, len(paths), BATCH_SIZE):
        chunk = paths[start : start + BATCH_SIZE]
        preds.extend(_transcribe_paths(model, chunk))

    valid_refs: List[str] = []
    valid_preds: List[str] = []
    for ref, pred in zip(refs, preds):
        if not ref.strip():
            continue
        valid_refs.append(ref)
        valid_preds.append(pred if pred.strip() else "<empty>")

    wer = jiwer.wer(valid_refs, valid_preds)
    print(f"\n=== WER on manifest: {wer:.4f}  (n={len(valid_refs):,}) ===")

    for i in range(min(5, len(valid_preds))):
        print(f"  P: {valid_preds[i]}")
        print(f"  T: {valid_refs[i]}\n")

    print("INFERENCE COMPLETE")


if __name__ == "__main__":
    main()
'''

os.makedirs("/kaggle/working", exist_ok=True)
with open(INFER_SCRIPT, "w", encoding="utf-8") as f:
    f.write(infer_code)

print(f"Inference script written to {INFER_SCRIPT}")
print(
    "Env: SKIP_PIP, STAGE1_NEMO, VAL_MANIFEST (or defaults: working manifests / akarshks), "
    "BATCH_SIZE, MAX_SAMPLES\n"
)

# =============================================================================
# PART C: FRESH SUBPROCESS
# =============================================================================
_compile = subprocess.run(
    [sys.executable, "-m", "py_compile", INFER_SCRIPT],
    cwd="/kaggle/working",
    capture_output=True,
    text=True,
)
if _compile.returncode != 0:
    print(_compile.stderr or _compile.stdout)
    raise RuntimeError("infer script failed py_compile")

result = subprocess.run(
    [sys.executable, INFER_SCRIPT],
    cwd="/kaggle/working",
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
    stdout=None,
    stderr=subprocess.STDOUT,
)
if result.returncode != 0:
    raise RuntimeError(f"Inference subprocess exit {result.returncode}")

print("\nDone.")
