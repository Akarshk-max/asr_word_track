"""
Kaggle / DrivenData submission: Parakeet-TDT-1.1B + ChainedLinearAdapter.

Faster path (no model.transcribe / no Lhotse dataloader): soundfile + torchaudio resample,
manual forward + RNN-T tensor decode, duration-based adaptive batching, OOM single-file fallback.

Bundle ``model.nemo`` next to this script (full ``save_to`` checkpoint — no separate adapter files).
Override path with argv[1] or MODEL_NEMO.

Layout:
  data/utterance_metadata.jsonl
  data/submission_format.jsonl
  data/audio/...

Writes:
  submission/submission.jsonl
"""

from __future__ import annotations

import json
import os
import sys
import types
import warnings
from pathlib import Path
from typing import List

warnings.filterwarnings("ignore", category=Warning, module="numba")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from loguru import logger
from omegaconf import open_dict

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil, LinearAdapter

torch.set_float32_matmul_precision("medium")

TARGET_SR = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        logger.info("Re-attached AdapterChainState to {} ChainedLinearAdapter module(s)", n)


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


def load_model_for_inference(nemo_path: str | Path) -> ASRModel:
    nemo_str = str(nemo_path)
    try:
        model = ASRModel.restore_from(nemo_str, map_location=str(DEVICE), strict=False)
    except TypeError:
        model = ASRModel.restore_from(nemo_str, map_location=str(DEVICE))

    with open_dict(model.cfg):
        if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
            model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    if hasattr(model, "change_decoding_strategy"):
        model.change_decoding_strategy(model.cfg.decoding)

    _reattach_chain_state(model)
    _ensure_inference_forward(model)

    model = model.to(DEVICE)
    model.eval()
    model.freeze()
    if hasattr(model, "waveform_augmentor") and model.waveform_augmentor is not None:
        model.waveform_augmentor.eval()

    return model


# -----------------------------------------------------------------------------
# Audio: mono @ 16 kHz (soundfile + torchaudio resample; no librosa)
# -----------------------------------------------------------------------------
def load_audio_mono_16k(path: str) -> torch.Tensor:
    wav, sr = sf.read(path, always_2d=True)
    wav = np.asarray(wav, dtype=np.float32)
    wav = wav.mean(axis=1)
    wav_t = torch.from_numpy(wav).float()
    if int(sr) != TARGET_SR:
        wav_t = torchaudio.functional.resample(wav_t, int(sr), TARGET_SR)
    return wav_t.contiguous()


def collate_audio(batch_paths: List[str], device: torch.device):
    waves = []
    for p in batch_paths:
        try:
            waves.append(load_audio_mono_16k(p))
        except Exception as e:
            logger.warning("Failed to load {}: {}; using silence", p, e)
            waves.append(torch.zeros(TARGET_SR, dtype=torch.float32))

    lengths = torch.tensor([w.shape[0] for w in waves], dtype=torch.long, device=device)
    max_len = int(lengths.max().item())
    padded = torch.zeros(len(waves), max_len, dtype=torch.float32, device=device)
    for i, w in enumerate(waves):
        padded[i, : w.shape[0]] = w.to(device=device, dtype=torch.float32)
    return padded, lengths


def _hypothesis_to_text(h) -> str:
    if isinstance(h, tuple):
        h = h[0]
    txt = getattr(h, "text", None)
    if isinstance(txt, str):
        return txt
    return str(h) if h is not None else ""


@torch.no_grad()
def infer_batch(model: ASRModel, batch_paths: List[str]) -> List[str]:
    signals, lengths = collate_audio(batch_paths, DEVICE)
    encoded, encoded_len = model.forward(
        input_signal=signals,
        input_signal_length=lengths,
    )
    hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
        encoded,
        encoded_len,
        return_hypotheses=True,
    )
    return [_hypothesis_to_text(h) for h in hypotheses]


@torch.no_grad()
def infer_single(model: ASRModel, audio_path: str) -> str:
    try:
        return infer_batch(model, [audio_path])[0]
    except Exception as e:
        logger.warning("Single inference failed for {}: {}", audio_path, e)
        return ""


# -----------------------------------------------------------------------------
# Adaptive batching (same shape as LSTM-memory submission; tune via env)
# -----------------------------------------------------------------------------
def _scale_batch_size(n: int) -> int:
    mul = float(os.environ.get("ADAPTIVE_BS_MULT", "1.0"))
    return max(1, int(round(n * mul)))


def get_batch_size(duration_sec: float) -> int:
    if duration_sec > 20:
        return _scale_batch_size(10)
    if duration_sec > 15:
        return _scale_batch_size(20)
    if duration_sec > 10:
        return _scale_batch_size(20)
    if duration_sec > 7:
        return _scale_batch_size(20)
    if duration_sec > 5:
        return _scale_batch_size(25)
    if duration_sec > 3:
        return _scale_batch_size(25)
    if duration_sec > 2:
        return _scale_batch_size(35)
    return _scale_batch_size(40)


def build_adaptive_batches(items: list) -> list:
    batches = []
    i = 0
    n = len(items)
    while i < n:
        duration = float(items[i].get("audio_duration_sec", 0.0))
        bs = get_batch_size(duration)
        batches.append(items[i : i + bs])
        i += bs
    return batches


def run_submission(model_path: Path, data_manifest: Path) -> None:
    if torch.cuda.is_available():
        logger.info("GPU: {}", torch.cuda.get_device_name(0))

    logger.info("Loading {}", model_path)
    model = load_model_for_inference(model_path)

    data_dir = data_manifest.parent
    format_path = data_dir / "submission_format.jsonl"
    submission_path = Path("submission") / "submission.jsonl"
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    if not format_path.is_file():
        raise FileNotFoundError(
            f"Missing {format_path}; expected benchmark layout with submission_format.jsonl"
        )

    with data_manifest.open("r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    logger.info("Utterances: {}", len(items))
    # Shortest first → larger micro-batches for throughput (same as reference submission)
    items.sort(key=lambda x: x.get("audio_duration_sec", 0.0))
    batches = build_adaptive_batches(items)
    logger.info("Built {} adaptive batches", len(batches))

    predictions: dict = {}
    processed = 0
    total = len(items)

    for batch_idx, batch in enumerate(batches):
        batch_audio = [str(data_dir / item["audio_path"]) for item in batch]
        batch_ids = [item["utterance_id"] for item in batch]
        duration = float(batch[0].get("audio_duration_sec", 0.0))
        batch_size = len(batch)

        if batch_idx % 200 == 0 or batch_idx < 10:
            logger.info(
                "Batch {}/{} | dur≈{:.1f}s | bs={} | done={}/{}",
                batch_idx + 1,
                len(batches),
                duration,
                batch_size,
                processed,
                total,
            )

        try:
            texts = infer_batch(model, batch_audio)
            for uid, text in zip(batch_ids, texts):
                predictions[uid] = text
            processed += batch_size
        except torch.OutOfMemoryError:
            logger.warning(
                "OOM at batch {} (dur≈{:.1f}s, bs={}); single-file fallback",
                batch_idx + 1,
                duration,
                batch_size,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for item in batch:
                p = str(data_dir / item["audio_path"])
                predictions[item["utterance_id"]] = infer_single(model, p)
                processed += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning("Batch {} failed: {}; single-file fallback", batch_idx + 1, e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for item in batch:
                p = str(data_dir / item["audio_path"])
                predictions[item["utterance_id"]] = infer_single(model, p)
                processed += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("Transcribed {} / {} utterances", len(predictions), total)

    missing_ids = []
    with format_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            uid = rec["utterance_id"]
            if uid not in predictions:
                missing_ids.append(uid)
    if missing_ids:
        logger.warning("{} IDs missing from predictions; filling empty string", len(missing_ids))
        for uid in missing_ids:
            predictions[uid] = ""

    logger.info("Writing {}...", submission_path)
    with format_path.open("r", encoding="utf-8") as fr, submission_path.open(
        "w", encoding="utf-8"
    ) as fw:
        for line in fr:
            if not line.strip():
                continue
            item = json.loads(line)
            item["orthographic_text"] = predictions.get(item["utterance_id"], "")
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.success("Done: {}", submission_path)


if __name__ == "__main__":
    _script_dir = Path(__file__).resolve().parent

    if len(sys.argv) > 1:
        _model = Path(sys.argv[1])
    else:
        _env = os.environ.get("MODEL_NEMO", "").strip()
        if _env:
            _model = Path(_env)
        else:
            for candidate in (
                _script_dir / "model.nemo",
                _script_dir / "model_final.nemo",
                _script_dir / "model_epoch6.nemo",
                _script_dir / "ASR-Adapter-best.nemo",
            ):
                if candidate.is_file():
                    _model = candidate
                    break
            else:
                _model = _script_dir / "model.nemo"

    if len(sys.argv) > 2:
        _manifest = Path(sys.argv[2])
    else:
        _manifest = Path("data") / "utterance_metadata.jsonl"

    if not _model.is_file():
        raise SystemExit(
            f"Model not found: {_model}. Place model.nemo next to main.py, pass argv[1], or set MODEL_NEMO."
        )
    if not _manifest.is_file():
        raise SystemExit(f"Manifest not found: {_manifest}")

    logger.info("Torch: {}, CUDA: {}", torch.__version__, torch.cuda.is_available())
    run_submission(_model, _manifest)
