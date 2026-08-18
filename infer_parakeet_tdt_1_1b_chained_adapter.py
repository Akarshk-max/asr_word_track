"""
Inference for Parakeet-TDT-1.1B fine-tuned with the ChainedLinearAdapter pipeline
(same as train_nemo_adapter_1.1b / the Kaggle single-cell trainer).

Preferred artifact: full checkpoint from training ``save_to`` (e.g. model_final.nemo or
model_epoch6.nemo). That file already includes adapter + joint + partial decoder weights.

Usage (local):
  python infer_parakeet_tdt_1_1b_chained_adapter.py \\
    --model /path/to/model_final.nemo \\
    audio1.wav audio2.flac

Optional:
  --manifest path/to.jsonl   (uses audio_filepath or audio_path per line; ignores text)
  --adapter-pt path.pt       (only if you need to override adapter weights on top of .nemo)
  --batch-size 8
  --out predictions.jsonl

Kaggle: set MODEL_NEMO to your checkpoint under /kaggle/working/nemo_adapter_1.1b/.../checkpoints/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import types
import warnings
from typing import Iterable

warnings.filterwarnings("ignore", category=Warning, module="numba")
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil, LinearAdapter

# -----------------------------------------------------------------------------
# Must match training (ChainedLinearAdapter + forward patch semantics)
# -----------------------------------------------------------------------------
class AdapterChainState:
    def __init__(self):
        self.prev_bottleneck = None
        self.layer_counter = 0

    def reset(self):
        self.prev_bottleneck = None
        self.layer_counter = 0

    def update(self, bottleneck):
        self.prev_bottleneck = bottleneck
        self.layer_counter += 1


class ChainedLinearAdapter(nn.Module, AdapterModuleUtil):
    def __init__(
        self,
        in_features,
        dim,
        activation="gelu",
        norm_position="post",
        dropout=0.1,
        is_first=False,
        chain_state_ref=None,
        adapter_strategy=None,
    ):
        nn.Module.__init__(self)
        assert norm_position == "post", "ChainedLinearAdapter only implements post-norm (NeMo parity)"
        self.down = nn.Linear(in_features, dim)
        self.up = nn.Linear(dim, in_features)
        self.norm = nn.LayerNorm(in_features)
        self.dropout_layer = nn.Dropout(dropout)
        self.is_first = is_first
        self.chain_state_ref = chain_state_ref
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "swish":
            self.act = nn.SiLU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.GELU()
        if not is_first:
            self.chain_proj = nn.Linear(dim, dim, bias=False)
            nn.init.orthogonal_(self.chain_proj.weight)
            with torch.no_grad():
                self.chain_proj.weight.mul_(0.9)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.setup_adapter_strategy(adapter_strategy)

    def forward(self, x):
        prev_bottleneck = None
        if self.chain_state_ref is not None and not self.is_first:
            prev_bottleneck = self.chain_state_ref.prev_bottleneck
        h = self.down(x)
        if prev_bottleneck is not None:
            if prev_bottleneck.shape[1] != h.shape[1]:
                prev_bottleneck = F.interpolate(
                    prev_bottleneck.transpose(1, 2),
                    size=h.shape[1],
                    mode="nearest",
                ).transpose(1, 2)
            h = h + self.chain_proj(prev_bottleneck)
        bottleneck = self.act(h)
        if self.chain_state_ref is not None:
            self.chain_state_ref.update(bottleneck)
        h_up = self.up(bottleneck)
        h_norm = self.norm(h_up)
        h_drop = self.dropout_layer(h_norm)
        return h_drop


LinearAdapter.register(ChainedLinearAdapter)


def _reattach_chain_state(model: ASRModel) -> None:
    cs = AdapterChainState()
    model.adapter_chain_state = cs
    n = 0
    for m in model.modules():
        if isinstance(m, ChainedLinearAdapter):
            m.chain_state_ref = cs
            n += 1
    if n:
        print(f"Re-attached AdapterChainState to {n} ChainedLinearAdapter module(s)")


def _ensure_inference_forward(model: ASRModel) -> None:
    if getattr(model, "_chain_forward_patched", False):
        return
    _original_forward = model.forward.__func__

    def _fw(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        if hasattr(self, "adapter_chain_state"):
            self.adapter_chain_state.reset()
        if self.training and input_signal is not None and input_signal_length is not None:
            wa = getattr(self, "waveform_augmentor", None)
            if wa is not None:
                input_signal, input_signal_length = wa(input_signal, input_signal_length)
        return _original_forward(
            self,
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
        )

    model.forward = types.MethodType(_fw, model)
    model._chain_forward_patched = True


def _patched_transcribe_dataloader(model, config, num_workers: int):
    if "manifest_filepath" in config:
        manifest_filepath = config["manifest_filepath"]
        batch_size = config["batch_size"]
    else:
        manifest_filepath = os.path.join(config["temp_dir"], "manifest.json")
        batch_size = min(config["batch_size"], len(config["paths2audio_files"]))

    use_ste = False
    if hasattr(model.cfg, "validation_ds") and model.cfg.validation_ds is not None:
        use_ste = model.cfg.validation_ds.get("use_start_end_token", False)

    dl_config = {
        "use_lhotse": False,
        "manifest_filepath": manifest_filepath,
        "sample_rate": model.preprocessor._sample_rate,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": min(batch_size, num_workers),
        "pin_memory": False,
        "channel_selector": config.get("channel_selector", "average"),
        "use_start_end_token": use_ste,
    }
    if config.get("augmentor"):
        dl_config["augmentor"] = config["augmentor"]
    return model._setup_dataloader_from_config(config=DictConfig(dl_config))


def load_model_for_inference(
    nemo_path: str,
    *,
    adapter_pt: str | None = None,
    num_workers: int = 4,
) -> ASRModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = ASRModel.restore_from(nemo_path, map_location=device, strict=False)
    except TypeError:
        model = ASRModel.restore_from(nemo_path, map_location=device)

    with open_dict(model.cfg):
        if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
            model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    if hasattr(model, "change_decoding_strategy"):
        model.change_decoding_strategy(model.cfg.decoding)

    _reattach_chain_state(model)
    _ensure_inference_forward(model)

    model = model.to(device)
    model.eval()
    model.freeze()
    if hasattr(model, "waveform_augmentor") and model.waveform_augmentor is not None:
        model.waveform_augmentor.eval()

    def _patched_setup_transcribe(self, config):
        return _patched_transcribe_dataloader(self, config, num_workers)

    model._setup_transcribe_dataloader = types.MethodType(_patched_setup_transcribe, model)

    if adapter_pt and os.path.isfile(adapter_pt):
        try:
            model.load_adapters(adapter_pt)
            print(f"Loaded adapter weights from {adapter_pt}")
        except Exception as e:
            print(f"[warn] load_adapters failed ({e}); using .nemo weights only.")

    return model


def transcribe_paths(
    model: ASRModel,
    paths: list[str],
    *,
    batch_size: int = 8,
    channel_selector: str = "average",
) -> list[str]:
    raw = model.transcribe(
        paths,
        batch_size=batch_size,
        channel_selector=channel_selector,
        verbose=False,
    )
    if isinstance(raw, tuple):
        raw = raw[0]
    out: list[str] = []
    for h in raw:
        txt = getattr(h, "text", None)
        out.append(txt if isinstance(txt, str) else str(h))
    return out


def _paths_from_manifest(manifest_path: str) -> list[str]:
    paths: list[str] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            p = (data.get("audio_filepath") or data.get("audio_path") or "").strip()
            if p and os.path.isfile(p):
                paths.append(p)
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transcribe with Parakeet 1.1B + chained adapter .nemo")
    p.add_argument(
        "--model",
        default=os.environ.get("MODEL_NEMO", ""),
        help="Path to .nemo from training save_to (e.g. model_final.nemo)",
    )
    p.add_argument(
        "--adapter-pt",
        default="",
        help="Optional adapter .pt to load after restore (usually unnecessary if .nemo is full)",
    )
    p.add_argument("--manifest", default="", help="JSONL manifest with audio_filepath or audio_path")
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")))
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "4")))
    p.add_argument("--out", default="", help="Write JSONL with audio_path and prediction")
    p.add_argument("audio", nargs="*", help="Audio files (wav/flac/...)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    nemo = args.model.strip()
    if not nemo:
        raise SystemExit("Pass --model / set MODEL_NEMO to your checkpoint .nemo")

    paths: list[str] = []
    if args.manifest:
        paths.extend(_paths_from_manifest(args.manifest))
    paths.extend(args.audio)
    paths = [os.path.abspath(p) for p in paths if os.path.isfile(p)]
    if not paths:
        raise SystemExit("No audio files: pass file paths and/or --manifest with valid paths")

    adapter_pt = args.adapter_pt.strip() or None
    print(f"Loading {nemo} …")
    model = load_model_for_inference(nemo, adapter_pt=adapter_pt, num_workers=args.num_workers)
    print(f"Transcribing {len(paths)} file(s), batch_size={args.batch_size} …")
    hyps = transcribe_paths(model, paths, batch_size=args.batch_size)

    for p, t in zip(paths, hyps):
        print(f"{p}\n  {t}\n")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for p, t in zip(paths, hyps):
                f.write(json.dumps({"audio_filepath": p, "text": t}, ensure_ascii=False) + "\n")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
