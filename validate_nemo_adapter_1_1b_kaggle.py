"""
One Kaggle cell (or local): Part A pip, Part B write eval worker, Part C subprocess.

Evaluates WER/CER on a val manifest for a fine-tuned Parakeet-TDT-1.1B .nemo produced by
train_nemo_adapter_1.1b (ChainedLinearAdapter + waveform_augmentor + chain state).

Env (optional, inherited by child):
  MODEL_NEMO, VAL_MANIFEST, BATCH_SIZE, NUM_WORKERS, RESULTS_JSON, PREDICTIONS_JSONL
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


# =============================================================================
# PART A: INSTALLATION (aligned with train_nemo_adapter_1.1b.py)
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
    "whisper-normalizer",
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

# =============================================================================
# PART B: WRITE EVAL SCRIPT
# =============================================================================
EVAL_SCRIPT = "/kaggle/working/compute_val_wer_1_1b.py"

eval_code = r'''
from __future__ import annotations

import json
import logging
import os
import random
import time
import types
import warnings

warnings.filterwarnings("ignore", category=Warning, module="numba")
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

import jiwer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil, LinearAdapter

# -----------------------------------------------------------------------------
# Must match training (train_nemo_adapter_1_1b_train.py) for restore_from / weights
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


class WaveformAugmentor(nn.Module):
    """Same class as training; in eval() forward is a no-op."""

    def __init__(
        self,
        speed_min=0.90,
        speed_max=1.10,
        pitch_min=-1.0,
        pitch_max=3.0,
        sample_rate=16000,
        speed_prob=0.5,
        pitch_prob=0.5,
        external_noise_prob=0.6,
        external_snr_min=1.0,
        external_snr_max=7.0,
        noise_file_paths=None,
        gain_prob=0.3,
        gain_db_min=-6.0,
        gain_db_max=6.0,
    ):
        super().__init__()
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.sample_rate = sample_rate
        self.speed_prob = speed_prob
        self.pitch_prob = pitch_prob
        self.external_noise_prob = external_noise_prob
        self.external_snr_min = external_snr_min
        self.external_snr_max = external_snr_max
        self.noise_file_paths = list(noise_file_paths or [])
        self.gain_prob = gain_prob
        self.gain_db_min = gain_db_min
        self.gain_db_max = gain_db_max

    @staticmethod
    def _mix_at_snr(speech, noise, snr_db):
        eps = 1e-8
        alpha = torch.sqrt(
            speech.pow(2).mean().clamp_min(eps)
            / (10 ** (snr_db / 10.0) * noise.pow(2).mean().clamp_min(eps) + eps)
        )
        return speech + alpha * noise

    def _load_noise_segment(self, num_samples, device, dtype):
        if not self.noise_file_paths:
            return None
        path = self.noise_file_paths[random.randint(0, len(self.noise_file_paths) - 1)]
        try:
            wav, sr = torchaudio.load(path)
        except Exception:
            return None
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        n = int(wav.shape[0])
        if n < 1:
            return None
        if n < num_samples:
            wav = wav.repeat((num_samples + n - 1) // n)[:num_samples]
        else:
            start = random.randint(0, n - num_samples)
            wav = wav[start : start + num_samples]
        return wav.to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def forward(self, audio_signal, signal_length):
        if not self.training:
            return audio_signal, signal_length
        B, T = audio_signal.shape
        if torch.rand(1).item() < self.speed_prob:
            sf = self.speed_min + torch.rand(1).item() * (self.speed_max - self.speed_min)
            new_T = max(1, int(T / sf))
            audio_signal = F.interpolate(
                audio_signal.unsqueeze(1), size=new_T, mode="linear", align_corners=False
            ).squeeze(1)
            signal_length = (signal_length.float() / sf).long().clamp(min=1, max=new_T)
        if torch.rand(1).item() < self.pitch_prob:
            n_steps = self.pitch_min + torch.rand(1).item() * (self.pitch_max - self.pitch_min)
            try:
                audio_signal = torchaudio.functional.pitch_shift(
                    audio_signal, self.sample_rate, n_steps
                )
            except Exception:
                pass
        if self.noise_file_paths and torch.rand(1).item() < self.external_noise_prob:
            for b in range(B):
                L = int(min(signal_length[b].item(), audio_signal.shape[1]))
                if L < 1:
                    continue
                seg = self._load_noise_segment(L, audio_signal.device, audio_signal.dtype)
                if seg is None:
                    continue
                snr = self.external_snr_min + torch.rand(1).item() * (
                    self.external_snr_max - self.external_snr_min
                )
                audio_signal[b, :L] = self._mix_at_snr(
                    audio_signal[b, :L].float(), seg, snr
                ).to(audio_signal.dtype)
        if torch.rand(1).item() < self.gain_prob:
            gain_db = self.gain_db_min + torch.rand(1).item() * (self.gain_db_max - self.gain_db_min)
            audio_signal = audio_signal * (10 ** (gain_db / 20.0))
        return audio_signal, signal_length


LinearAdapter.register(ChainedLinearAdapter)

try:
    from whisper_normalizer.english import EnglishTextNormalizer

    _whisper_normalizer = EnglishTextNormalizer()
    HAS_WHISPER_NORM = True
except ImportError:
    _whisper_normalizer = None
    HAS_WHISPER_NORM = False


def normalize_ref(text: str) -> str:
    text = (text or "").strip()
    if HAS_WHISPER_NORM and _whisper_normalizer:
        return _whisper_normalizer(text)
    return text.lower()


# #region agent log
_DBG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug-9056e1.log")


def _dbg(msg: str, data: dict, hypothesis_id: str) -> None:
    import time as _t

    try:
        with open(_DBG_PATH, "a", encoding="utf-8") as _f:
            _f.write(
                json.dumps(
                    {
                        "sessionId": "9056e1",
                        "runId": "val-wer",
                        "hypothesisId": hypothesis_id,
                        "location": "compute_val_wer_1_1b.py",
                        "message": msg,
                        "data": data,
                        "timestamp": int(_t.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass


# #endregion


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
MODEL_PATH = os.environ.get(
    "MODEL_NEMO",
    "/kaggle/working/nemo_adapter_1.1b/ParakeetAdapter1.1B/2026-04-04_02-44-35/checkpoints/model_final.nemo",
)
VAL_MANIFEST = os.environ.get(
    "VAL_MANIFEST",
    "/kaggle/input/datasets/akarshkumarshukla/vel-mani/val_manifest.jsonl",
)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
RESULTS_JSON = os.environ.get("RESULTS_JSON", "/kaggle/working/val_wer_results_1_1b.json")
PREDICTIONS_JSONL = os.environ.get(
    "PREDICTIONS_JSONL", "/kaggle/working/val_predictions_1_1b.jsonl"
)
CHUNK_MULTIPLIER = int(os.environ.get("CHUNK_MULTIPLIER", "10"))


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


def _reattach_chain_state(model: ASRModel) -> None:
    """Training stores adapter_chain_state on the model object; ensure adapters share one state."""
    cs = AdapterChainState()
    model.adapter_chain_state = cs
    n = 0
    for m in model.modules():
        if isinstance(m, ChainedLinearAdapter):
            m.chain_state_ref = cs
            n += 1
    if n:
        print(f"Re-attached AdapterChainState to {n} ChainedLinearAdapter module(s)", flush=True)


def _ensure_inference_forward(model: ASRModel) -> None:
    """Reset chain each forward (matches training); waveform aug only if training."""
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


def main() -> None:
    print("=" * 70)
    print("VALIDATION WER (Parakeet-TDT-1.1B + ChainedLinearAdapter .nemo)")
    print("=" * 70)

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.isfile(VAL_MANIFEST):
        raise FileNotFoundError(f"Val manifest not found: {VAL_MANIFEST}")

    print(f"Model: {MODEL_PATH} ({os.path.getsize(MODEL_PATH) / 1024**3:.2f} GB)")
    print(f"Manifest: {VAL_MANIFEST}")
    print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    t0 = time.time()
    try:
        model = ASRModel.restore_from(MODEL_PATH, map_location=device, strict=False)
    except TypeError:
        model = ASRModel.restore_from(MODEL_PATH, map_location=device)

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
        return _patched_transcribe_dataloader(self, config, NUM_WORKERS)

    model._setup_transcribe_dataloader = types.MethodType(_patched_setup_transcribe, model)

    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    samples = []
    skipped_empty_norm = 0
    skipped_examples: list[dict] = []
    with open(VAL_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                audio_path = (
                    data.get("audio_filepath")
                    or data.get("audio_path")
                    or ""
                ).strip()
                text = (data.get("text") or data.get("orthographic_text") or "").strip()
                duration = float(data.get("duration") or data.get("audio_duration_sec") or 0.0)
                if audio_path and text and os.path.isfile(audio_path):
                    ref_norm = normalize_ref(text)
                    if not (ref_norm or "").strip():
                        skipped_empty_norm += 1
                        if len(skipped_examples) < 10:
                            skipped_examples.append(
                                {
                                    "ref_raw_preview": (text[:120] + "…")
                                    if len(text) > 120
                                    else text,
                                    "ref_norm_repr": repr(ref_norm),
                                }
                            )
                        continue
                    samples.append(
                        {
                            "path": audio_path,
                            "ref_raw": text,
                            "ref_norm": ref_norm,
                            "dur": duration,
                        }
                    )
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

    # #region agent log
    _dbg(
        "manifest_filter_empty_normalized_ref",
        {
            "skipped_empty_norm": skipped_empty_norm,
            "kept_samples": len(samples),
            "examples": skipped_examples,
        },
        "H1",
    )
    # #endregion

    if not samples:
        raise RuntimeError(
            "No valid rows (need audio path + non-empty text after normalization + existing file)."
        )

    references = [s["ref_norm"] for s in samples]
    audio_files = [s["path"] for s in samples]
    total_duration = sum(s["dur"] for s in samples)

    print(
        f"Samples: {len(samples):,}  |  duration: {total_duration / 3600:.2f} h"
        + (
            f"  |  skipped {skipped_empty_norm:,} rows (empty ref after Whisper/lowercase norm)"
            if skipped_empty_norm
            else ""
        ),
        flush=True,
    )

    chunk_size = max(BATCH_SIZE, BATCH_SIZE * CHUNK_MULTIPLIER)
    all_predictions = []
    t1 = time.time()
    for i in tqdm(range(0, len(audio_files), chunk_size), desc="Transcribing"):
        chunk = audio_files[i : i + chunk_size]
        raw = model.transcribe(
            chunk,
            batch_size=BATCH_SIZE,
            channel_selector="average",
            verbose=False,
        )
        if isinstance(raw, tuple):
            raw = raw[0]
        for h in raw:
            txt = getattr(h, "text", None)
            all_predictions.append(txt if isinstance(txt, str) else str(h))

    elapsed = time.time() - t1
    predictions_normalized = [normalize_ref(p) for p in all_predictions]

    # #region agent log
    _empty_post = sum(1 for r in references if not (r or "").strip())
    _dbg(
        "pre_jiwer_ref_nonempty_check",
        {
            "n_refs": len(references),
            "empty_ref_count": _empty_post,
            "n_preds": len(predictions_normalized),
        },
        "H2",
    )
    # #endregion

    if _empty_post:
        raise RuntimeError(
            f"Internal error: {_empty_post} empty references after load filter — report bug."
        )
    if len(predictions_normalized) != len(references):
        raise RuntimeError(
            f"Prediction count {len(predictions_normalized)} != references {len(references)}"
        )

    wer = jiwer.wer(references, predictions_normalized)
    cer = jiwer.cer(references, predictions_normalized)
    measures = jiwer.compute_measures(references, predictions_normalized)
    exact_matches = sum(1 for r, p in zip(references, predictions_normalized) if r == p)

    per_wer = []
    for i, (ref, pred) in enumerate(zip(references, predictions_normalized)):
        try:
            per_wer.append((i, jiwer.wer([ref], [pred]), ref, pred))
        except Exception:
            per_wer.append((i, 1.0, ref, pred))

    worst = sorted(per_wer, key=lambda x: x[1], reverse=True)
    best = sorted(per_wer, key=lambda x: x[1])

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"WER:  {wer:.4f} ({wer * 100:.2f}%)")
    print(f"CER:  {cer:.4f} ({cer * 100:.2f}%)")
    print(f"Exact match: {exact_matches:,}/{len(samples):,} ({100 * exact_matches / len(samples):.1f}%)")
    print(f"Time: {elapsed:.1f}s  |  {len(audio_files) / max(elapsed, 1e-6):.2f} files/s  |  RTF {elapsed / max(total_duration, 1e-6):.3f}x")
    print(f"Substitutions: {measures['substitutions']:,}  Deletions: {measures['deletions']:,}  Insertions: {measures['insertions']:,}")

    print("\n10 worst (by utterance WER):")
    for idx, (i, w, ref, pred) in enumerate(worst[:10]):
        print(f"\n[{idx + 1}] utt={i} WER={w:.2f}")
        print(f"  REF:  {ref[:120]}{'...' if len(ref) > 120 else ''}")
        print(f"  HYP:  {pred[:120]}{'...' if len(pred) > 120 else ''}")

    results = {
        "model_path": MODEL_PATH,
        "val_manifest": VAL_MANIFEST,
        "num_samples": len(samples),
        "skipped_empty_normalized_reference": skipped_empty_norm,
        "total_duration_hours": total_duration / 3600,
        "wer": wer,
        "cer": cer,
        "exact_match_rate": exact_matches / len(samples),
        "substitutions": measures["substitutions"],
        "deletions": measures["deletions"],
        "insertions": measures["insertions"],
        "hits": measures["hits"],
        "inference_time_sec": elapsed,
        "files_per_sec": len(audio_files) / max(elapsed, 1e-6),
    }
    os.makedirs(os.path.dirname(RESULTS_JSON) or ".", exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics: {RESULTS_JSON}")

    with open(PREDICTIONS_JSONL, "w", encoding="utf-8") as f:
        wer_by_idx = {i: w for i, w, _, _ in per_wer}
        for i, s in enumerate(samples):
            f.write(
                json.dumps(
                    {
                        "idx": i,
                        "audio_path": s["path"],
                        "reference_raw": s["ref_raw"],
                        "reference_norm": references[i],
                        "prediction_raw": all_predictions[i],
                        "prediction_norm": predictions_normalized[i],
                        "utterance_wer": wer_by_idx.get(i),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Saved predictions: {PREDICTIONS_JSONL}")
    print("=" * 70)


if __name__ == "__main__":
    main()
'''

os.makedirs("/kaggle/working", exist_ok=True)
with open(EVAL_SCRIPT, "w", encoding="utf-8") as f:
    f.write(eval_code)

print(f"Eval script written to {EVAL_SCRIPT}\n")
print(
    "Env: MODEL_NEMO, VAL_MANIFEST, BATCH_SIZE, NUM_WORKERS, "
    "RESULTS_JSON, PREDICTIONS_JSONL, CHUNK_MULTIPLIER\n"
)

# =============================================================================
# PART C: SUBPROCESS
# =============================================================================
print("=" * 60)
print("Running validation (fresh interpreter)")
print("=" * 60)

_compile = subprocess.run(
    [sys.executable, "-m", "py_compile", EVAL_SCRIPT],
    cwd="/kaggle/working",
    capture_output=True,
    text=True,
)
if _compile.returncode != 0:
    print(_compile.stderr or _compile.stdout)
    raise RuntimeError("eval script py_compile failed")

result = subprocess.run(
    [sys.executable, EVAL_SCRIPT],
    cwd="/kaggle/working",
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
    stdout=None,
    stderr=subprocess.STDOUT,
)

if result.returncode != 0:
    raise RuntimeError(f"Validation failed with exit code {result.returncode}")

print("\nValidation finished successfully.")
