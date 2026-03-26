# =============================================================================
# NeMo ASR Inference Script
# Run this in a NEW cell after training completes
# =============================================================================
# ----lb score 0.2242 with noisy_wer--0.6077


import os
import subprocess
import sys


def _pip(*args):
    cmd = [sys.executable, "-m", "pip", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("\n[pip] command failed:")
        print(" ".join(cmd))
        if proc.stdout:
            print("\n[pip stdout]\n" + proc.stdout[-4000:])
        if proc.stderr:
            print("\n[pip stderr]\n" + proc.stderr[-4000:])
        raise RuntimeError(f"pip failed with exit code {proc.returncode}")


def _install_numpy_scipy_with_fallback():
    # Preferred pins (kept aligned with notebook format).
    try:
        _pip(
            "install",
            "--no-cache-dir",
            "--prefer-binary",
            "numpy>=2.1,<2.3",
            "scipy>=1.14,<1.16",
        )
        return
    except Exception as e:
        print(f"[warn] pinned numpy/scipy install failed: {e}")

    # Fallback for transient index issues / wheel availability on current Python build.
    _pip(
        "install",
        "--no-cache-dir",
        "--prefer-binary",
        "numpy>=2.1",
        "scipy",
    )


# =============================================================================
# PART A: INSTALLATION
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
_pip("install", "--no-cache-dir", "--upgrade", "pip", "setuptools", "wheel")
_install_numpy_scipy_with_fallback()

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

# =======================================================

INFERENCE_SCRIPT = "/kaggle/working/inference_nemo_adapter.py"

inference_code = r'''
# -*- coding: utf-8 -*-
"""
NeMo ASR Inference Script for Trained Adapter Model
Runs inference on validation manifest and computes WER
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=Warning, module="numba")

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================
@dataclass
class InferenceConfig:
    # Model path - change this to your trained model
    model_path: str = "/kaggle/input/models/akarshkumarshukla/chained-adapters-model/pytorch/default/1/model_epoch4.nemo"
    
    # Validation manifest
    val_manifest: str = "/kaggle/input/datasets/akarshkumarshukla/val-data/val_manifest.jsonl"
    
    # Output directory
    output_dir: str = "/kaggle/working/inference_results"
    
    # Inference settings
    batch_size: int = 64
    num_workers: int = 2
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Text normalization (should match training)
    normalize_text: bool = True
    
    # Max samples to process (None = all)
    max_samples: Optional[int] = None


def find_latest_model(base_dir: str = "/kaggle/working/nemo_adapter_1.1b") -> str:
    """Find the most recent model_final.nemo file"""
    import glob
    
    # Look for model_final.nemo
    patterns = [
        os.path.join(base_dir, "**", "model_final.nemo"),
        os.path.join(base_dir, "**", "checkpoints", "model_final.nemo"),
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            # Return most recently modified
            return max(files, key=os.path.getmtime)
    
    # Fallback: look for any .nemo file
    all_nemo = glob.glob(os.path.join(base_dir, "**", "*.nemo"), recursive=True)
    if all_nemo:
        return max(all_nemo, key=os.path.getmtime)
    
    raise FileNotFoundError(f"No .nemo model found in {base_dir}")


# ============================================================================
# Text Normalizer
# ============================================================================
try:
    from whisper_normalizer.english import EnglishTextNormalizer
    _whisper_normalizer = EnglishTextNormalizer()
    HAS_WHISPER_NORM = True
except ImportError:
    HAS_WHISPER_NORM = False
    _whisper_normalizer = None


def normalize_text(text: str, use_whisper: bool = True) -> str:
    """Normalize text for WER computation"""
    if use_whisper and HAS_WHISPER_NORM and _whisper_normalizer:
        return _whisper_normalizer(text)
    return text.lower().strip()


# ============================================================================
# Load Manifest
# ============================================================================
def load_manifest(manifest_path: str, max_samples: Optional[int] = None) -> List[dict]:
    """Load JSONL manifest file"""
    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # Handle different manifest formats
                audio_path = data.get("audio_filepath") or data.get("audio_path", "")
                text = data.get("text") or data.get("orthographic_text", "")
                duration = data.get("duration") or data.get("audio_duration_sec", 0)
                
                if audio_path and os.path.isfile(audio_path):
                    samples.append({
                        "audio_filepath": audio_path,
                        "text": text.strip(),
                        "duration": float(duration),
                    })
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return samples


# ============================================================================
# WER Computation
# ============================================================================
def compute_wer_metrics(references: List[str], hypotheses: List[str]) -> dict:
    """Compute WER and related metrics using jiwer"""
    import jiwer
    
    # Filter empty pairs
    valid_pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip()]
    if not valid_pairs:
        return {"wer": 1.0, "cer": 1.0, "num_samples": 0}
    
    refs, hyps = zip(*valid_pairs)
    refs = list(refs)
    hyps = list(hyps)
    
    # Compute metrics
    wer = jiwer.wer(refs, hyps)
    
    # Word-level stats
    measures = jiwer.compute_measures(refs, hyps)
    
    # Character Error Rate
    try:
        cer = jiwer.cer(refs, hyps)
    except:
        cer = -1.0
    
    return {
        "wer": wer,
        "cer": cer,
        "substitutions": measures["substitutions"],
        "deletions": measures["deletions"],
        "insertions": measures["insertions"],
        "hits": measures["hits"],
        "num_samples": len(refs),
        "num_ref_words": sum(len(r.split()) for r in refs),
        "num_hyp_words": sum(len(h.split()) for h in hyps),
    }


# ============================================================================
# Main Inference
# ============================================================================
def run_inference(cfg: InferenceConfig):
    print("=" * 70)
    print("NeMo ASR Inference")
    print("=" * 70)
    
    # Auto-find model if not specified or doesn't exist
    if not os.path.isfile(cfg.model_path):
        print(f"Model not found at {cfg.model_path}")
        print("Searching for trained model...")
        cfg.model_path = find_latest_model()
        print(f"Found: {cfg.model_path}")
    
    print(f"\nConfiguration:")
    print(f"  Model:        {cfg.model_path}")
    print(f"  Manifest:     {cfg.val_manifest}")
    print(f"  Batch size:   {cfg.batch_size}")
    print(f"  Device:       {cfg.device}")
    print(f"  Normalize:    {cfg.normalize_text}")
    
    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Load manifest
    print(f"\nLoading manifest...")
    samples = load_manifest(cfg.val_manifest, cfg.max_samples)
    print(f"Loaded {len(samples)} samples")
    
    if not samples:
        raise RuntimeError(f"No valid samples found in {cfg.val_manifest}")
    
    # Load model
    print(f"\nLoading model...")
    from nemo.collections.asr.models import ASRModel
    from omegaconf import open_dict
    
    model = ASRModel.restore_from(cfg.model_path, map_location=cfg.device)
    model = model.to(cfg.device)
    model.eval()
    
    # Disable CUDA graph decoder (can cause issues)
    with open_dict(model.cfg):
        if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
            model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    if hasattr(model, "change_decoding_strategy"):
        model.change_decoding_strategy(model.cfg.decoding)
    
    print(f"Model loaded: {type(model).__name__}")
    
    # Run inference
    print(f"\nRunning inference...")
    
    references = []
    hypotheses = []
    results = []
    
    audio_paths = [s["audio_filepath"] for s in samples]
    
    # Use model's transcribe method for batched inference
    start_time = time.time()
    
    try:
        # NeMo's transcribe handles batching internally
        transcriptions = model.transcribe(
            audio_paths,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            return_hypotheses=False,
            verbose=True,
        )
    except Exception as e:
        print(f"Batched transcribe failed: {e}")
        print("Falling back to single-file inference...")
        transcriptions = []
        for path in tqdm(audio_paths, desc="Transcribing"):
            try:
                trans = model.transcribe([path], batch_size=1, verbose=False)
                transcriptions.append(trans[0] if trans else "")
            except Exception as ex:
                print(f"Failed on {path}: {ex}")
                transcriptions.append("")
    
    elapsed = time.time() - start_time
    
    # Process results
    for i, (sample, hyp) in enumerate(zip(samples, transcriptions)):
        ref = sample["text"]
        
        # Normalize if needed
        if cfg.normalize_text:
            ref_norm = normalize_text(ref, use_whisper=True)
            hyp_norm = normalize_text(hyp, use_whisper=True)
        else:
            ref_norm = ref.lower().strip()
            hyp_norm = hyp.lower().strip()
        
        references.append(ref_norm)
        hypotheses.append(hyp_norm)
        
        results.append({
            "audio_filepath": sample["audio_filepath"],
            "reference": ref,
            "reference_normalized": ref_norm,
            "hypothesis": hyp,
            "hypothesis_normalized": hyp_norm,
            "duration": sample["duration"],
        })
    
    # Compute metrics
    print(f"\nComputing metrics...")
    metrics = compute_wer_metrics(references, hypotheses)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Total samples:    {metrics['num_samples']}")
    print(f"  Total ref words:  {metrics['num_ref_words']}")
    print(f"  Total hyp words:  {metrics['num_hyp_words']}")
    print(f"  Inference time:   {elapsed:.1f}s ({len(samples)/elapsed:.1f} samples/sec)")
    print()
    print(f"  WER:              {metrics['wer']*100:.2f}%")
    print(f"  CER:              {metrics['cer']*100:.2f}%")
    print()
    print(f"  Substitutions:    {metrics['substitutions']}")
    print(f"  Deletions:        {metrics['deletions']}")
    print(f"  Insertions:       {metrics['insertions']}")
    print(f"  Hits:             {metrics['hits']}")
    print("=" * 70)
    
    # Save results
    results_path = os.path.join(cfg.output_dir, "predictions.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nPredictions saved to: {results_path}")
    
    # Save metrics
    metrics_path = os.path.join(cfg.output_dir, "metrics.json")
    metrics["inference_time_sec"] = elapsed
    metrics["samples_per_sec"] = len(samples) / elapsed
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Save sample comparisons
    comparison_path = os.path.join(cfg.output_dir, "comparison.txt")
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write(f"WER: {metrics['wer']*100:.2f}%\n")
        f.write(f"CER: {metrics['cer']*100:.2f}%\n")
        f.write(f"Samples: {metrics['num_samples']}\n")
        f.write("=" * 80 + "\n\n")
        
        # Sort by WER (worst first)
        for i, r in enumerate(results[:100]):  # Show first 100
            f.write(f"[{i+1}] {os.path.basename(r['audio_filepath'])}\n")
            f.write(f"  REF: {r['reference_normalized']}\n")
            f.write(f"  HYP: {r['hypothesis_normalized']}\n")
            f.write("\n")
    print(f"Comparison saved to: {comparison_path}")
    
    # Print some examples
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS (first 10)")
    print("=" * 70)
    for i, r in enumerate(results[:10]):
        print(f"\n[{i+1}] {os.path.basename(r['audio_filepath'])}")
        print(f"  REF: {r['reference_normalized']}")
        print(f"  HYP: {r['hypothesis_normalized']}")
    
    return metrics, results


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NeMo ASR Inference")
    parser.add_argument("--model", type=str, default=None, help="Path to .nemo model")
    parser.add_argument("--manifest", type=str, default=None, help="Path to validation manifest")
    parser.add_argument("--output", type=str, default="/kaggle/working/inference_results")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()
    
    cfg = InferenceConfig()
    
    if args.model:
        cfg.model_path = args.model
    if args.manifest:
        cfg.val_manifest = args.manifest
    if args.output:
        cfg.output_dir = args.output
    cfg.batch_size = args.batch_size
    cfg.max_samples = args.max_samples
    cfg.normalize_text = not args.no_normalize
    
    run_inference(cfg)
'''

# Write inference script
with open(INFERENCE_SCRIPT, "w", encoding="utf-8") as f:
    f.write(inference_code)

print(f"Inference script written to: {INFERENCE_SCRIPT}")




#   now the main.py code which i used 


"""
Submission: Children's Speech Recognition Challenge (Word Track)
Model: Parakeet-TDT-1.1B with ChainedLinearAdapter

Fully offline — no HuggingFace downloads.
Uses soundfile + torchaudio.functional.resample (no librosa/numba/torchcodec).
Manual forward + decoding (no model.transcribe / no Lhotse).
OOM-safe adaptive batching with single-file fallback.
"""

import json
import os
import shutil
import tarfile
import tempfile
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

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil, LinearAdapter
from nemo.core.classes import adapter_mixins
from omegaconf import open_dict

# =============================================================================
# PATHS
# =============================================================================
SRC_DIR = Path(__file__).parent.resolve()
DATA_DIR = Path("data")
SUBMISSION_DIR = Path("submission")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = SRC_DIR / "model_epoch4.nemo"
METADATA_PATH = DATA_DIR / "utterance_metadata.jsonl"
FORMAT_PATH = DATA_DIR / "submission_format.jsonl"
OUTPUT_PATH = SUBMISSION_DIR / "submission.jsonl"

ADAPTER_NAME_SHORT = "asr_children_adapter"
TARGET_SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# CUSTOM CLASSES (MUST MATCH TRAINING)
# =============================================================================
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
        assert norm_position == "post"
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


# =============================================================================
# HELPERS
# =============================================================================
def extract_nemo(nemo_path: str, dst_dir: str):
    with tarfile.open(nemo_path, "r:") as tar:
        tar.extractall(dst_dir)


def find_weight_file(extracted_dir: str) -> str:
    for root, _, files in os.walk(extracted_dir):
        for f in files:
            if f in ("model_weights.ckpt", "mp_rank_00_model_states.pt") or f.endswith(".ckpt") or f.endswith(".pt"):
                return os.path.join(root, f)
    raise FileNotFoundError("No weight file found inside extracted .nemo")


# =============================================================================
# MODEL RESTORE (FULLY OFFLINE)
# =============================================================================
def restore_custom_model(nemo_path: str, device=DEVICE):
    """
    Restore model entirely from .nemo file. No internet needed.

    Strategy:
    1. Extract .nemo to get raw state dict
    2. ASRModel.restore_from(.nemo, strict=False)
       - NeMo reads config from .nemo, builds model with LinearAdapter
       - State dict loads base weights; adapter keys mismatch (ignored)
    3. Replace LinearAdapter modules with ChainedLinearAdapter
    4. Reload full state dict — now adapter keys match
    """
    logger.info(f"Restoring model from: {nemo_path}")

    # Step 1: Extract weights for later reload
    tmpdir = tempfile.mkdtemp()
    try:
        extract_nemo(str(nemo_path), tmpdir)
        weight_file = find_weight_file(tmpdir)
        logger.info(f"Extracted weights: {weight_file}")

        full_state = torch.load(weight_file, map_location=device, weights_only=False)
        if isinstance(full_state, dict) and "state_dict" in full_state:
            full_state = full_state["state_dict"]

        logger.info(f"State dict has {len(full_state)} keys")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Step 2: Restore model from .nemo (offline, strict=False)
    logger.info("Restoring model architecture from .nemo (strict=False)...")
    model = ASRModel.restore_from(str(nemo_path), map_location=device, strict=False)

    # Step 3: Configure decoding
    with open_dict(model.cfg):
        if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
            model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    if hasattr(model, "change_decoding_strategy"):
        model.change_decoding_strategy(model.cfg.decoding)

    # Step 4: Replace LinearAdapter with ChainedLinearAdapter
    chain_state = AdapterChainState()
    model.adapter_chain_state = chain_state

    adapter_module_list = []
    for mod_name, module in list(model.named_modules()):
        if isinstance(module, LinearAdapter) and ADAPTER_NAME_SHORT in mod_name:
            adapter_module_list.append((mod_name, module))
    adapter_module_list.sort(key=lambda x: x[0])

    logger.info(f"Replacing {len(adapter_module_list)} LinearAdapter -> ChainedLinearAdapter")

    dev = next(model.parameters()).device
    for i, (mod_name, _orig) in enumerate(adapter_module_list):
        new_adapter = ChainedLinearAdapter(
            in_features=1024,
            dim=128,
            activation="gelu",
            norm_position="post",
            dropout=0.1,
            is_first=(i == 0),
            chain_state_ref=chain_state,
            adapter_strategy=None,
        ).to(dev)

        parts = mod_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child = parts[-1]

        setattr(parent, child, new_adapter)
        if isinstance(parent, nn.ModuleDict):
            parent[child] = new_adapter

        if len(parts) >= 2 and parts[-2] == "adapter_layer":
            layer_mod = model
            for p in parts[:-2]:
                layer_mod = getattr(layer_mod, p)
            if hasattr(layer_mod, "adapter_layer") and ADAPTER_NAME_SHORT in layer_mod.adapter_layer:
                layer_mod.adapter_layer[ADAPTER_NAME_SHORT] = new_adapter

        new_adapter.setup_adapter_strategy(None)

    LinearAdapter.register(ChainedLinearAdapter)

    # Step 5: Reload full state dict (now adapter keys match)
    logger.info("Reloading full state dict with correct adapter architecture...")
    missing, unexpected = model.load_state_dict(full_state, strict=False)
    logger.info(f"Reload done. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    if missing:
        logger.warning(f"Sample missing keys: {missing[:5]}")
    if unexpected:
        logger.warning(f"Sample unexpected keys: {unexpected[:5]}")

    # Step 6: Forward wrapper with chain state reset
    _original_forward = model.forward.__func__

    def _augmented_forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        if hasattr(self, "adapter_chain_state"):
            self.adapter_chain_state.reset()
        return _original_forward(
            self,
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
        )

    model.forward = types.MethodType(_augmented_forward, model)

    model.eval()
    model.freeze()

    logger.info("Model ready (fully offline, no HuggingFace)")
    return model


# =============================================================================
# AUDIO LOADING (soundfile + torchaudio.functional.resample)
# =============================================================================
def load_audio_mono_16k(path: str) -> torch.Tensor:
    wav, sr = sf.read(path, always_2d=True)  # [T, C]
    wav = np.asarray(wav, dtype=np.float32)
    wav = wav.mean(axis=1)  # mono
    wav = torch.from_numpy(wav).float()
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.contiguous()


def collate_audio(batch_paths: List[str], device=DEVICE):
    waves = []
    for p in batch_paths:
        try:
            w = load_audio_mono_16k(p)
            waves.append(w)
        except Exception as e:
            logger.warning(f"Failed to load {p}: {e}, using silence")
            waves.append(torch.zeros(TARGET_SR, dtype=torch.float32))

    lengths = torch.tensor([w.shape[0] for w in waves], dtype=torch.long, device=device)
    max_len = int(lengths.max().item())

    padded = torch.zeros(len(waves), max_len, dtype=torch.float32, device=device)
    for i, w in enumerate(waves):
        padded[i, : w.shape[0]] = w.to(device=device, dtype=torch.float32)

    return padded, lengths


# =============================================================================
# INFERENCE
# =============================================================================
@torch.no_grad()
def infer_batch(model, batch_paths: List[str]) -> List[str]:
    signals, lengths = collate_audio(batch_paths, device=DEVICE)

    encoded, encoded_len = model.forward(
        input_signal=signals,
        input_signal_length=lengths,
    )

    hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
        encoded,
        encoded_len,
        return_hypotheses=True,
    )

    texts = []
    for h in hypotheses:
        if isinstance(h, tuple):
            h = h[0]
        text = getattr(h, "text", None)
        if text is None:
            text = str(h)
        texts.append(text if text else "")
    return texts


@torch.no_grad()
def infer_single(model, audio_path: str) -> str:
    try:
        texts = infer_batch(model, [audio_path])
        return texts[0]
    except Exception as e:
        logger.warning(f"Single inference failed for {audio_path}: {e}")
        return ""


# =============================================================================
# ADAPTIVE BATCHING
# =============================================================================
def get_batch_size(duration_sec: float) -> int:
    if duration_sec > 20:
        return 10
    elif duration_sec > 15:
        return 20
    elif duration_sec > 10:
        return 20
    elif duration_sec > 7:
        return 20
    elif duration_sec > 5:
        return 25
    elif duration_sec > 3:
        return 25
    elif duration_sec > 2:
        return 35
    else:
        return 40


def build_adaptive_batches(items):
    batches = []
    i = 0
    n = len(items)
    while i < n:
        duration = items[i].get("audio_duration_sec", 0.0)
        bs = get_batch_size(duration)
        batch = items[i:i + bs]
        batches.append(batch)
        i += bs
    return batches


# =============================================================================
# MAIN
# =============================================================================
def main():
    logger.info(f"Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model (fully offline)
    model = restore_custom_model(str(MODEL_PATH), device=DEVICE)

    # Load metadata
    with open(METADATA_PATH, "r") as f:
        items = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Utterances: {len(items)}")

    # Sort by duration ascending
    items.sort(key=lambda x: x.get("audio_duration_sec", 0))

    batches = build_adaptive_batches(items)
    logger.info(f"Built {len(batches)} adaptive batches")

    predictions = {}
    processed = 0
    total = len(items)

    for batch_idx, batch in enumerate(batches):
        batch_audio = [str(DATA_DIR / item["audio_path"]) for item in batch]
        batch_ids = [item["utterance_id"] for item in batch]
        duration = batch[0].get("audio_duration_sec", 0.0)
        batch_size = len(batch)

        if batch_idx % 200 == 0 or batch_idx < 10:
            logger.info(
                f"Batch {batch_idx+1}/{len(batches)} | "
                f"dur≈{duration:.1f}s | bs={batch_size} | "
                f"done={processed}/{total}"
            )

        try:
            texts = infer_batch(model, batch_audio)

            for uid, text in zip(batch_ids, texts):
                predictions[uid] = text
            processed += batch_size

        except torch.OutOfMemoryError:
            logger.warning(
                f"OOM at batch {batch_idx+1} (dur≈{duration:.1f}s, bs={batch_size}). "
                f"Falling back to single-file."
            )
            torch.cuda.empty_cache()

            for item in batch:
                audio_path = str(DATA_DIR / item["audio_path"])
                text = infer_single(model, audio_path)
                predictions[item["utterance_id"]] = text
                processed += 1

            torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(
                f"Batch {batch_idx+1} failed: {e}. Falling back to single-file."
            )
            torch.cuda.empty_cache()

            for item in batch:
                audio_path = str(DATA_DIR / item["audio_path"])
                text = infer_single(model, audio_path)
                predictions[item["utterance_id"]] = text
                processed += 1

            torch.cuda.empty_cache()

    logger.info(f"Transcribed {len(predictions)} / {total} utterances")

    # Sanity check: fill missing IDs
    missing_ids = []
    with open(FORMAT_PATH, "r") as f:
        for line in f:
            item = json.loads(line)
            if item["utterance_id"] not in predictions:
                missing_ids.append(item["utterance_id"])

    if missing_ids:
        logger.warning(f"{len(missing_ids)} utterance IDs missing, filling with empty string")
        for uid in missing_ids:
            predictions[uid] = ""

    # Write submission
    logger.info(f"Writing submission to {OUTPUT_PATH}...")
    with open(FORMAT_PATH, "r") as fr, open(OUTPUT_PATH, "w", encoding="utf-8") as fw:
        for line in fr:
            item = json.loads(line)
            item["orthographic_text"] = predictions.get(item["utterance_id"], "")
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.success(f"Done! Submission: {OUTPUT_PATH}")
    logger.info(f"Total predictions: {len(predictions)}")


if __name__ == "__main__":
    main()
