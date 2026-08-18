"""
Inference for Parakeet-TDT-1.1B with Chained Linear Adapters + LSTM Memory (V3)

Paste this entire file into ONE Kaggle notebook cell (or run locally).

  Part A — pip install (clean env + NeMo)
  Part B — write inference script to /kaggle/working
  Part C — subprocess.run (fresh Python interpreter)

Outputs: /kaggle/working/inference_results/
  - predictions.jsonl   (per-utterance ref/hyp)
  - metrics.json         (WER, CER, substitutions, deletions, insertions)
  - comparison.txt       (human-readable side-by-side)

CHECKPOINT SUPPORT:
  - .nemo  (full model saved by model.save_to)
  - .pt    (trainable-only weights saved as adapter_final.pt / adapter_epochN.pt)
  Set CHECKPOINT_PATH env var, or the script auto-discovers the latest checkpoint.
"""

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


# =============================================================================
# PART A: INSTALLATION
# =============================================================================
print("Step 1: Cleaning...")
for _ in range(2):
    subprocess.run(
        [
            sys.executable, "-m", "pip", "uninstall", "-y",
            "numpy", "scipy", "nemo_toolkit", "lightning",
            "pytorch-lightning", "datasets", "diffusers", "gradio",
            "peft", "sentence-transformers", "transformers",
            "huggingface_hub", "torch", "torchaudio", "torchvision", "numba",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

print("Step 2: numpy + scipy...")
_pip("install", "--no-cache-dir", "numpy>=2.1,<2.3", "scipy>=1.14,<1.16")

print("Step 3: PyTorch (CUDA 12.6 wheels)...")
subprocess.check_call(
    [
        sys.executable, "-m", "pip", "install", "--no-cache-dir",
        "--index-url", "https://download.pytorch.org/whl/cu126",
        "torch>=2.9.0", "torchaudio",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
)

print("Step 4: Dependencies...")
_pip(
    "install", "--no-cache-dir",
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
    "install", "--no-cache-dir", "--force-reinstall",
    "numpy>=2.1,<2.3", "scipy>=1.14,<1.16",
)

print("Installation complete.\n")


# =============================================================================
# PART B: WRITE INFERENCE SCRIPT
# =============================================================================
INFERENCE_SCRIPT = "/kaggle/working/inference_chained_memory_v3.py"

inference_code = r'''
from __future__ import annotations

import glob
import json
import logging
import math
import os
import re
import shutil
import tarfile
import tempfile
import time
import types
import warnings
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore", category=Warning, module="numba")
logging.getLogger("nemo.collections.asr.metrics.wer").setLevel(logging.ERROR)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

torch.set_float32_matmul_precision("medium")

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.adapter_modules import LinearAdapter
from nemo.core.classes import adapter_mixins
from nemo.core.classes.mixins import adapter_mixin_strategies
from omegaconf import OmegaConf, open_dict

print(f"torch:  {torch.__version__}")
print(f"CUDA:   {torch.version.cuda}")
print(f"numpy:  {np.__version__}")
print(f"GPU:    {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")


# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

MODEL_ID = "nvidia/parakeet-tdt-1.1b"

# --- Checkpoint path ---
# Supports .nemo (full model) or .pt (trainable weights only).
# Set via env var or edit the default below.
_CKPT_BASE = "/kaggle/input/chained-memory-lstm-v3"
_CKPT_SUBDIR = "nemo_adapter_1.1b/ParakeetAdapter1.1B_Chained_Memory/2026-03-26_22-29-57/checkpoints"
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    os.path.join(_CKPT_BASE, _CKPT_SUBDIR, "model_final.nemo"),
)

# --- Validation manifest ---
VAL_MANIFEST = os.environ.get(
    "VAL_MANIFEST",
    "/kaggle/input/datasets/akarshkumarshukla/val-data/val_manifest.jsonl",
)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/kaggle/working/inference_results")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))

# Must match training hyperparameters exactly for weight-name compatibility.
ADAPTER_DIM = 128
MEMORY_DIM = 64
ADAPTER_ACTIVATION = "gelu"
ADAPTER_DROPOUT = 0.1


# ============================================================================
# SECTION 2: CUSTOM ADAPTER CLASSES
# (Exact copies from training — parameter names must match for weight loading)
# ============================================================================

class AdapterMemoryCell(nn.Module):
    """LSTM-style memory cell operating on the adapter's bottleneck space."""

    def __init__(self, bottleneck_dim: int = 128, memory_dim: int = 64, output_dim: int = 1024):
        super().__init__()
        self.memory_dim = memory_dim
        input_size = bottleneck_dim + memory_dim
        self.gates = nn.Linear(input_size, 4 * memory_dim)
        self.mem_output_proj = nn.Linear(memory_dim, output_dim, bias=False)
        with torch.no_grad():
            self.gates.bias[memory_dim:2 * memory_dim].fill_(2.0)
        nn.init.normal_(self.mem_output_proj.weight, std=0.002)

    def forward(
        self,
        bottleneck: torch.Tensor,
        prev_hidden: torch.Tensor,
        prev_cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.cat([bottleneck, prev_hidden], dim=-1)
        gates = self.gates(z)
        i_gate, f_gate, cell_candidate, o_gate = gates.chunk(4, dim=-1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        cell_candidate = torch.tanh(cell_candidate)
        o_gate = torch.sigmoid(o_gate)
        new_cell = f_gate * prev_cell + i_gate * cell_candidate
        new_hidden = o_gate * torch.tanh(new_cell)
        mem_output = self.mem_output_proj(new_hidden)
        return mem_output, new_hidden, new_cell


class AdapterStreamState:
    """Chain bottleneck + LSTM memory state across encoder layers."""

    def __init__(self, memory_dim: int = 64):
        self.memory_dim = memory_dim
        self.reset()

    def reset(self):
        self.prev_bottleneck: Optional[torch.Tensor] = None
        self.prev_hidden: Optional[torch.Tensor] = None
        self.prev_cell: Optional[torch.Tensor] = None
        self.layer_counter: int = 0

    def get_memory_states(
        self, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.prev_hidden is None or self.prev_cell is None:
            self.prev_hidden = torch.zeros(batch_size, seq_len, self.memory_dim, device=device, dtype=dtype)
            self.prev_cell = torch.zeros(batch_size, seq_len, self.memory_dim, device=device, dtype=dtype)
        return self.prev_hidden, self.prev_cell

    def update(self, bottleneck: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        self.prev_bottleneck = bottleneck.detach()
        self.prev_hidden = hidden
        self.prev_cell = cell
        self.layer_counter += 1


class ChainedLinearAdapterWithMemory(LinearAdapter):
    """Three-stream adapter: bottleneck + cross-layer chain + LSTM memory.

    Subclasses NeMo LinearAdapter so AttentionAdapterModuleMixin accepts it.
    forward() returns the delta only (NeMo's ResidualAddAdapterStrategy adds x + delta).
    """

    def __init__(
        self,
        in_features: int = 1024,
        dim: int = 128,
        memory_dim: int = 64,
        activation: str = "gelu",
        norm_position: str = "post",
        dropout: float = 0.1,
        is_first: bool = False,
        stream_state_ref: Optional[AdapterStreamState] = None,
        adapter_strategy: Optional[adapter_mixin_strategies.AbstractAdapterStrategy] = None,
    ):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.dim = dim
        self.memory_dim = memory_dim
        self.is_first = is_first
        self.stream_state_ref = stream_state_ref

        self.down = nn.Linear(in_features, dim)
        self.up = nn.Linear(dim, in_features)
        self.norm_bottleneck = nn.LayerNorm(dim)
        self.dropout_layer = nn.Dropout(dropout)

        act_map = {"gelu": nn.GELU(), "swish": nn.SiLU(), "relu": nn.ReLU()}
        self.act = act_map.get(activation, nn.GELU())

        if not is_first:
            self.chain_proj = nn.Linear(dim, dim, bias=False)
            nn.init.orthogonal_(self.chain_proj.weight)
            with torch.no_grad():
                self.chain_proj.weight.mul_(0.9)

        self.memory_cell = AdapterMemoryCell(
            bottleneck_dim=dim,
            memory_dim=memory_dim,
            output_dim=in_features,
        )

        nn.init.normal_(self.up.weight, std=0.002)
        nn.init.zeros_(self.up.bias)

        self.setup_adapter_strategy(adapter_strategy)

    def _match_seq_len(self, tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        if tensor.shape[1] == target_len:
            return tensor
        return F.interpolate(
            tensor.transpose(1, 2), size=target_len, mode="nearest"
        ).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.down(x)

        if not self.is_first and self.stream_state_ref is not None:
            prev_b = self.stream_state_ref.prev_bottleneck
            if prev_b is not None:
                prev_b = self._match_seq_len(prev_b, T)
                h = h + self.chain_proj(prev_b)

        bottleneck = self.act(h)
        bottleneck_n = self.norm_bottleneck(bottleneck)

        # LSTM-style memory: when stream_state_ref is set, memory_cell always runs before
        # stream_state_ref.update() below. Each AdapterStreamState.layer_counter increment
        # equals exactly one AdapterMemoryCell.forward() (42 per full encoder pass).

        new_hidden = None
        new_cell = None
        if self.stream_state_ref is not None:
            prev_hidden, prev_cell = self.stream_state_ref.get_memory_states(B, T, x.device, x.dtype)
            prev_hidden = self._match_seq_len(prev_hidden, T)
            prev_cell = self._match_seq_len(prev_cell, T)
            mem_output, new_hidden, new_cell = self.memory_cell(bottleneck_n, prev_hidden, prev_cell)
            h_up = self.up(bottleneck_n) + mem_output
        else:
            h_up = self.up(bottleneck_n)

        h_drop = self.dropout_layer(h_up)

        if self.stream_state_ref is not None:
            self.stream_state_ref.update(bottleneck, new_hidden, new_cell)

        return h_drop


# ============================================================================
# SECTION 3: TEXT NORMALIZATION
# ============================================================================

try:
    from whisper_normalizer.english import EnglishTextNormalizer
    _whisper_normalizer = EnglishTextNormalizer()
    HAS_WHISPER_NORM = True
    print("Whisper text normalizer loaded")
except ImportError:
    HAS_WHISPER_NORM = False
    _whisper_normalizer = None
    print("WARNING: whisper_normalizer not installed, using basic lowercase normalization")


def normalize_text(text: str) -> str:
    text = text.strip()
    if HAS_WHISPER_NORM and _whisper_normalizer:
        return _whisper_normalizer(text)
    return text.lower()


# ============================================================================
# SECTION 4: CHECKPOINT DISCOVERY
# ============================================================================

def find_checkpoint(
    search_dirs: Optional[List[str]] = None,
    prefer_nemo: bool = True,
) -> str:
    """Auto-discover the latest checkpoint under common Kaggle paths."""
    if search_dirs is None:
        search_dirs = [
            "/kaggle/input/chained-memory-lstm-v3/nemo_adapter_1.1b",
            "/kaggle/input/chained-memory-lstm-v3",
            "/kaggle/working/nemo_adapter_1.1b",
            "/kaggle/working",
            "/kaggle/input",
        ]

    nemo_files: List[str] = []
    pt_files: List[str] = []

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for pattern in ["**/model_final.nemo", "**/model_epoch*.nemo"]:
            nemo_files.extend(glob.glob(os.path.join(d, pattern), recursive=True))
        for pattern in ["**/adapter_final.pt", "**/adapter_epoch*.pt"]:
            pt_files.extend(glob.glob(os.path.join(d, pattern), recursive=True))

    if prefer_nemo and nemo_files:
        chosen = max(nemo_files, key=os.path.getmtime)
        print(f"Auto-discovered .nemo checkpoint: {chosen}")
        return chosen
    if pt_files:
        chosen = max(pt_files, key=os.path.getmtime)
        print(f"Auto-discovered .pt checkpoint: {chosen}")
        return chosen
    if nemo_files:
        chosen = max(nemo_files, key=os.path.getmtime)
        print(f"Auto-discovered .nemo checkpoint: {chosen}")
        return chosen

    raise FileNotFoundError(
        "No checkpoint found. Set CHECKPOINT_PATH env var or place "
        "model_final.nemo / adapter_final.pt under /kaggle/working or /kaggle/input."
    )


# ============================================================================
# SECTION 5: EXTRACT STATE DICT FROM .nemo
# ============================================================================

def extract_state_dict_from_nemo(nemo_path: str, device: str = "cpu") -> dict:
    """Extract the model_weights.ckpt state dict from a .nemo tar archive."""
    tmpdir = tempfile.mkdtemp(prefix="nemo_extract_")
    try:
        with tarfile.open(nemo_path, "r:*") as tar:
            tar.extractall(tmpdir)

        weight_file = None
        for root, _, files in os.walk(tmpdir):
            for name in files:
                if name in ("model_weights.ckpt", "mp_rank_00_model_states.pt"):
                    weight_file = os.path.join(root, name)
                    break
                if name.endswith(".ckpt") or name.endswith(".pt"):
                    weight_file = os.path.join(root, name)
            if weight_file:
                break

        if weight_file is None:
            raise FileNotFoundError(f"No weight file found inside {nemo_path}")

        state_dict = torch.load(weight_file, map_location=device, weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        return state_dict
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# SECTION 6: MODEL LOADING WITH ADAPTER RECONSTRUCTION
# ============================================================================

def _encoder_target_key(model_cfg):
    enc = model_cfg.encoder
    if "_target_" in enc:
        return "_target_"
    if "target" in enc:
        return "target"
    raise AttributeError("model_cfg.encoder has neither _target_ nor target")


def _verify_lstm_memory_forward_calls(model: ASRModel, device: str) -> None:
    """Direct proof: count AdapterMemoryCell.forward invocations (expect 42 per utterance)."""
    ctr = [0]

    def _hook(_mod, _inp, _out):
        ctr[0] += 1

    hooks = []
    for mod in model.modules():
        if isinstance(mod, AdapterMemoryCell):
            hooks.append(mod.register_forward_hook(_hook))

    st = getattr(model, "_adapter_stream_state", None)
    if st is not None:
        st.reset()
    dummy_audio = torch.randn(1, 16000, device=device)
    dummy_len = torch.tensor([16000], device=device)
    with torch.no_grad():
        model(input_signal=dummy_audio, input_signal_length=dummy_len)

    for h in hooks:
        h.remove()
    if st is not None:
        st.reset()

    print(
        f"  VERIFY_LSTM_MEMORY: AdapterMemoryCell.forward count={ctr[0]} "
        f"(expect 42; matches encoder adapter layers)"
    )


def load_model(checkpoint_path: str, device: str = "cuda") -> ASRModel:
    """
    Load the base pretrained model, reconstruct custom adapters,
    and load trained weights from checkpoint.
    """
    print(f"\n{'='*60}")
    print(f"Loading base pretrained model: {MODEL_ID}")
    print(f"{'='*60}")

    # --- Step 1: Load base pretrained model with adapter-mixin support ---
    model_cfg = ASRModel.from_pretrained(MODEL_ID, return_config=True)
    enc_key = _encoder_target_key(model_cfg)

    with open_dict(model_cfg):
        adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder[enc_key])
        if adapter_metadata is not None:
            model_cfg.encoder[enc_key] = adapter_metadata.adapter_class_path
            print(f"  Patched encoder target to: {adapter_metadata.adapter_class_path}")

    model = ASRModel.from_pretrained(MODEL_ID, override_config_path=model_cfg)

    # Disable CUDA graph decoder
    with open_dict(model.cfg):
        if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
            model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    if hasattr(model, "change_decoding_strategy"):
        model.change_decoding_strategy(model.cfg.decoding)

    print("  Base model loaded")

    # --- Step 2: Add NeMo adapter infrastructure ---
    print("  Adding NeMo adapter infrastructure...")
    adapter_type_cfg = OmegaConf.create({
        "_target_": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
        "in_features": 1024,
        "dim": ADAPTER_DIM,
        "activation": ADAPTER_ACTIVATION,
        "norm_position": "post",
        "dropout": ADAPTER_DROPOUT,
    })
    nemo_adapter_name = "encoder:asr_children_adapter"
    model.add_adapter(name=nemo_adapter_name, cfg=adapter_type_cfg)
    model.set_enabled_adapters(enabled=False)
    model.set_enabled_adapters(name=nemo_adapter_name, enabled=True)

    # --- Step 3: Create shared stream state ---
    stream_state = AdapterStreamState(memory_dim=MEMORY_DIM)
    model._adapter_stream_state = stream_state

    # --- Step 4: Replace NeMo LinearAdapters with ChainedLinearAdapterWithMemory ---
    print("  Replacing adapters with Chained + Memory adapters...")

    adapter_modules = {}
    for name, module in model.named_modules():
        if hasattr(module, "__class__") and "LinearAdapter" in module.__class__.__name__:
            adapter_modules[name] = module
        elif hasattr(module, "adapter_layer") and hasattr(module.adapter_layer, "asr_children_adapter"):
            sub = module.adapter_layer.asr_children_adapter
            if hasattr(sub, "__class__") and "LinearAdapter" in sub.__class__.__name__:
                adapter_modules[f"{name}.adapter_layer.asr_children_adapter"] = sub

    if len(adapter_modules) == 0:
        for name, module in model.named_modules():
            if "asr_children_adapter" in name and hasattr(module, "module"):
                adapter_modules[name] = module

    sorted_adapter_names = sorted(
        adapter_modules.keys(),
        key=lambda x: int("".join(filter(str.isdigit, x.split("layers.")[-1].split(".")[0])))
        if "layers." in x else 0,
    )

    replaced_count = 0
    for idx, adapter_name_path in enumerate(sorted_adapter_names):
        old_adapter = adapter_modules[adapter_name_path]

        in_features = ADAPTER_DIM
        if hasattr(old_adapter, "module"):
            for sub in old_adapter.module:
                if hasattr(sub, "in_features"):
                    in_features = sub.in_features
                    break
                if hasattr(sub, "weight") and sub.weight.shape[1] > ADAPTER_DIM:
                    in_features = sub.weight.shape[1]
                    break
        if in_features == ADAPTER_DIM:
            in_features = 1024

        _strategy = getattr(old_adapter, "adapter_strategy", None)

        new_adapter = ChainedLinearAdapterWithMemory(
            in_features=in_features,
            dim=ADAPTER_DIM,
            memory_dim=MEMORY_DIM,
            activation=ADAPTER_ACTIVATION,
            dropout=ADAPTER_DROPOUT,
            is_first=(idx == 0),
            stream_state_ref=stream_state,
            adapter_strategy=_strategy,
        )

        parts = adapter_name_path.split(".")
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        setattr(parent, parts[-1], new_adapter)
        replaced_count += 1

    print(f"  Replaced {replaced_count} adapters with ChainedLinearAdapterWithMemory")

    if replaced_count == 0:
        raise RuntimeError(
            "Could not find any NeMo adapter modules to replace. "
            "Check that the adapter config matches training."
        )

    # --- Step 5: Load trained weights ---
    print(f"\n  Loading weights from: {checkpoint_path}")

    if checkpoint_path.endswith(".nemo"):
        state_dict = extract_state_dict_from_nemo(checkpoint_path, device="cpu")
        result = model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded .nemo state dict: {len(state_dict)} tensors")
        if result.missing_keys:
            print(f"  Missing keys: {len(result.missing_keys)}")
            for k in result.missing_keys[:5]:
                print(f"    {k}")
            if len(result.missing_keys) > 5:
                print(f"    ... and {len(result.missing_keys) - 5} more")
        if result.unexpected_keys:
            adapter_unexpected = [k for k in result.unexpected_keys
                                  if any(s in k.lower() for s in ["augment", "waveform", "classroom", "noise"])]
            other_unexpected = [k for k in result.unexpected_keys if k not in adapter_unexpected]
            if adapter_unexpected:
                print(f"  Ignored augmentor keys: {len(adapter_unexpected)} (expected, training-only modules)")
            if other_unexpected:
                print(f"  Other unexpected keys: {len(other_unexpected)}")
                for k in other_unexpected[:5]:
                    print(f"    {k}")

    elif checkpoint_path.endswith(".pt"):
        weights = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        current_sd = model.state_dict()
        loaded, skipped = 0, []
        for k, v in weights.items():
            if k in current_sd:
                if current_sd[k].shape == v.shape:
                    current_sd[k] = v
                    loaded += 1
                else:
                    skipped.append((k, current_sd[k].shape, v.shape))
            else:
                skipped.append((k, None, v.shape))
        model.load_state_dict(current_sd, strict=True)
        print(f"  Loaded {loaded}/{len(weights)} trained parameter tensors from .pt")
        if skipped:
            print(f"  Skipped {len(skipped)} tensors (shape mismatch or missing):")
            for k, expected, got in skipped[:10]:
                print(f"    {k}: expected={expected}, checkpoint={got}")

    else:
        raise ValueError(
            f"Unsupported checkpoint format: {checkpoint_path}\n"
            f"Expected .nemo or .pt file."
        )

    # --- Step 6: Verify adapter weights were loaded ---
    n_nonzero = 0
    n_total = 0
    for name, param in model.named_parameters():
        if "adapter" in name.lower() or "memory" in name.lower() or "chain" in name.lower():
            n_total += 1
            if param.abs().sum().item() > 0:
                n_nonzero += 1
    print(f"  Adapter weight verification: {n_nonzero}/{n_total} tensors non-zero")
    if n_nonzero < n_total * 0.5:
        print("  WARNING: Many adapter tensors are zero — weights may not have loaded correctly!")

    # --- Step 7: Patch forward to reset stream state ---
    _original_forward = model.forward.__func__

    def _inference_forward(self, input_signal=None, input_signal_length=None,
                           processed_signal=None, processed_signal_length=None):
        if hasattr(self, "_adapter_stream_state"):
            self._adapter_stream_state.reset()
        return _original_forward(
            self,
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
        )

    model.forward = types.MethodType(_inference_forward, model)

    # Safety net: also reset on encoder forward (transcribe may call encoder directly)
    def _encoder_pre_hook(module, args):
        if hasattr(model, "_adapter_stream_state"):
            model._adapter_stream_state.reset()
        return args

    model.encoder.register_forward_pre_hook(_encoder_pre_hook)

    # --- Step 8: Move to device and set eval ---
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    adapter_params = sum(
        p.numel() for n, p in model.named_parameters()
        if any(k in n.lower() for k in ["adapter", "memory", "chain"])
    )
    print(f"\n  Model ready on {device}")
    print(f"  Total parameters:   {total_params:,}")
    print(f"  Adapter parameters: {adapter_params:,}")

    if os.environ.get("VERIFY_LSTM_MEMORY", "").strip().lower() in ("1", "true", "yes"):
        _verify_lstm_memory_forward_calls(model, device)

    return model


# ============================================================================
# SECTION 7: MANIFEST LOADING
# ============================================================================

def load_manifest(manifest_path: str, max_samples: Optional[int] = None) -> List[dict]:
    """Load JSONL manifest file, resolving audio paths."""
    samples = []
    skipped = {"no_audio": 0, "missing_file": 0, "no_text": 0, "parse_error": 0}

    with open(manifest_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                audio_path = (
                    data.get("audio_filepath")
                    or data.get("audiofilepath")
                    or data.get("audio_path")
                    or ""
                ).strip()
                text = (data.get("text") or data.get("orthographic_text") or "").strip()
                duration = float(data.get("duration") or data.get("audio_duration_sec") or 0)

                if not audio_path:
                    skipped["no_audio"] += 1
                    continue
                if not os.path.isfile(audio_path):
                    skipped["missing_file"] += 1
                    continue
                if not text:
                    skipped["no_text"] += 1
                    continue

                samples.append({
                    "audio_filepath": audio_path,
                    "text": text,
                    "duration": duration,
                })
            except (json.JSONDecodeError, TypeError, ValueError):
                skipped["parse_error"] += 1
                continue

    print(f"  Loaded {len(samples)} valid samples")
    if any(v > 0 for v in skipped.values()):
        print(f"  Skipped: {skipped}")
    return samples


# ============================================================================
# SECTION 8: WER COMPUTATION
# ============================================================================

def compute_wer_metrics(references: List[str], hypotheses: List[str]) -> dict:
    """Compute WER, CER, and detailed error breakdown using jiwer."""
    import jiwer

    valid_pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip()]
    if not valid_pairs:
        return {"wer": 1.0, "cer": 1.0, "num_samples": 0}

    refs, hyps = zip(*valid_pairs)
    refs, hyps = list(refs), list(hyps)

    wer = jiwer.wer(refs, hyps)
    measures = jiwer.compute_measures(refs, hyps)

    try:
        cer = jiwer.cer(refs, hyps)
    except Exception:
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
# SECTION 9: AUDIO PREPROCESSING + TRANSCRIPTION
# ============================================================================

def _ensure_mono(src_path: str, mono_dir: str) -> str:
    """Return path to a mono 16kHz WAV. If already mono, returns src_path unchanged."""
    import soundfile as sf_lib
    try:
        info = sf_lib.info(src_path)
        if info.channels == 1:
            return src_path
        data, sr = sf_lib.read(src_path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        out_name = os.path.splitext(os.path.basename(src_path))[0] + "_mono.wav"
        out_path = os.path.join(mono_dir, out_name)
        sf_lib.write(out_path, mono, sr)
        return out_path
    except Exception:
        return src_path


def preprocess_audio_to_mono(
    audio_paths: List[str], num_workers: int = 8
) -> Tuple[List[str], str]:
    """Convert any stereo files to mono in a temp directory.

    Returns (mono_paths, temp_dir_path). Caller must clean up temp_dir.
    """
    from concurrent.futures import ThreadPoolExecutor
    mono_dir = tempfile.mkdtemp(prefix="mono_audio_")

    def _process(path):
        return _ensure_mono(path, mono_dir)

    print(f"  Preprocessing audio to mono (workers={num_workers})...")
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        mono_paths = list(tqdm(
            ex.map(_process, audio_paths),
            total=len(audio_paths),
            desc="Mono conversion",
        ))

    converted = sum(1 for orig, mono in zip(audio_paths, mono_paths) if orig != mono)
    print(f"  Converted {converted}/{len(audio_paths)} stereo files to mono")
    return mono_paths, mono_dir


def _to_text(obj) -> str:
    """Extract plain text from NeMo transcribe output (handles various return types)."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    txt = getattr(obj, "text", None)
    if isinstance(txt, str):
        return txt
    if isinstance(obj, (list, tuple)) and obj:
        return _to_text(obj[0])
    return str(obj)


@torch.no_grad()
def run_transcription(
    model: ASRModel,
    audio_paths: List[str],
    batch_size: int = 64,
    num_workers: int = 2,
) -> List[str]:
    """Preprocess audio to mono, then transcribe in batch mode."""

    mono_paths, mono_dir = preprocess_audio_to_mono(audio_paths, num_workers=8)

    print(f"\n  Transcribing {len(mono_paths)} files (batch_size={batch_size})...")

    try:
        raw = model.transcribe(
            mono_paths,
            batch_size=batch_size,
            num_workers=num_workers,
            return_hypotheses=False,
            verbose=True,
        )

        if isinstance(raw, (list, tuple)):
            transcriptions = [_to_text(r) for r in raw]
        else:
            transcriptions = [_to_text(raw)]
        print(f"  Batch transcription complete: {len(transcriptions)} results")

    except Exception as e:
        print(f"  Batch transcribe failed: {e}")
        print("  Falling back to single-file inference...")

        transcriptions = []
        for path in tqdm(mono_paths, desc="Transcribing (single)"):
            try:
                raw = model.transcribe(
                    [path],
                    batch_size=1,
                    num_workers=0,
                    return_hypotheses=False,
                    verbose=False,
                )
                text = _to_text(raw[0]) if isinstance(raw, (list, tuple)) and raw else _to_text(raw)
                transcriptions.append(text)
            except Exception as ex:
                print(f"  Failed on {os.path.basename(path)}: {ex}")
                transcriptions.append("")

    # Clean up temp mono files
    try:
        shutil.rmtree(mono_dir, ignore_errors=True)
        print(f"  Cleaned up temp mono directory")
    except Exception:
        pass

    return transcriptions


# ============================================================================
# SECTION 10: MAIN INFERENCE PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("INFERENCE: Parakeet-TDT-1.1B + Chained Adapters + LSTM Memory (V3)")
    print("=" * 70)

    # Resolve checkpoint path
    ckpt = CHECKPOINT_PATH
    if not ckpt or not os.path.isfile(ckpt):
        if ckpt:
            print(f"Specified checkpoint not found: {ckpt}")
        print("Searching for checkpoint...")
        ckpt = find_checkpoint()
    print(f"\nCheckpoint: {ckpt}")
    print(f"Manifest:   {VAL_MANIFEST}")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Batch size: {BATCH_SIZE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load manifest
    print(f"\nLoading manifest: {VAL_MANIFEST}")
    if not os.path.isfile(VAL_MANIFEST):
        raise FileNotFoundError(f"Validation manifest not found: {VAL_MANIFEST}")
    samples = load_manifest(VAL_MANIFEST)
    if not samples:
        raise RuntimeError(f"No valid samples found in {VAL_MANIFEST}")

    # Load model
    model = load_model(ckpt, device="cuda" if torch.cuda.is_available() else "cpu")

    # Run transcription
    audio_paths = [s["audio_filepath"] for s in samples]
    t0 = time.time()
    transcriptions = run_transcription(model, audio_paths, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    elapsed = time.time() - t0

    # Process results
    references = []
    hypotheses = []
    results = []

    for sample, hyp in zip(samples, transcriptions):
        ref_norm = normalize_text(sample["text"])
        hyp_norm = normalize_text(hyp)
        references.append(ref_norm)
        hypotheses.append(hyp_norm)
        results.append({
            "audio_filepath": sample["audio_filepath"],
            "reference": sample["text"],
            "reference_normalized": ref_norm,
            "hypothesis": hyp,
            "hypothesis_normalized": hyp_norm,
            "duration": sample["duration"],
        })

    # Compute metrics
    print("\nComputing WER metrics...")
    metrics = compute_wer_metrics(references, hypotheses)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Total samples:      {metrics['num_samples']}")
    print(f"  Total ref words:    {metrics['num_ref_words']}")
    print(f"  Total hyp words:    {metrics['num_hyp_words']}")
    print(f"  Inference time:     {elapsed:.1f}s ({len(samples)/max(elapsed,0.01):.1f} samples/sec)")
    print()
    print(f"  WER:                {metrics['wer']*100:.2f}%")
    print(f"  CER:                {metrics['cer']*100:.2f}%")
    print()
    print(f"  Substitutions:      {metrics['substitutions']}")
    print(f"  Deletions:          {metrics['deletions']}")
    print(f"  Insertions:         {metrics['insertions']}")
    print(f"  Hits:               {metrics['hits']}")
    print("=" * 70)

    # Save predictions
    predictions_path = os.path.join(OUTPUT_DIR, "predictions.jsonl")
    with open(predictions_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nPredictions saved to: {predictions_path}")

    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    metrics["checkpoint_path"] = ckpt
    metrics["manifest_path"] = VAL_MANIFEST
    metrics["inference_time_sec"] = elapsed
    metrics["samples_per_sec"] = len(samples) / max(elapsed, 0.01)
    metrics["batch_size"] = BATCH_SIZE
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    # Save human-readable comparison
    comparison_path = os.path.join(OUTPUT_DIR, "comparison.txt")
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {ckpt}\n")
        f.write(f"Manifest:   {VAL_MANIFEST}\n")
        f.write(f"WER: {metrics['wer']*100:.2f}%\n")
        f.write(f"CER: {metrics['cer']*100:.2f}%\n")
        f.write(f"Samples: {metrics['num_samples']}\n")
        f.write("=" * 80 + "\n\n")
        for i, r in enumerate(results):
            f.write(f"[{i+1}] {os.path.basename(r['audio_filepath'])}\n")
            f.write(f"  REF: {r['reference_normalized']}\n")
            f.write(f"  HYP: {r['hypothesis_normalized']}\n\n")
    print(f"Comparison saved to: {comparison_path}")

    # Print sample predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS (first 15)")
    print("=" * 70)
    for i, r in enumerate(results[:15]):
        print(f"\n[{i+1}] {os.path.basename(r['audio_filepath'])}")
        print(f"  REF: {r['reference_normalized']}")
        print(f"  HYP: {r['hypothesis_normalized']}")

    # Clean up GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nDone. Results in {OUTPUT_DIR}/")
    return metrics


if __name__ == "__main__":
    main()
'''

# Write inference script
os.makedirs("/kaggle/working", exist_ok=True)
with open(INFERENCE_SCRIPT, "w", encoding="utf-8") as f:
    f.write(inference_code)
print(f"Inference script written to: {INFERENCE_SCRIPT}")


# =============================================================================
# PART C: LAUNCH INFERENCE
# =============================================================================
print("\nStarting inference subprocess...")

# --- Configure these env vars to override defaults ---
env = {
    **os.environ,
    "PYTHONUNBUFFERED": "1",
    # Uncomment / edit to use a different checkpoint or manifest:
    # "CHECKPOINT_PATH": "/kaggle/input/chained-memory-lstm-v3/nemo_adapter_1.1b/ParakeetAdapter1.1B_Chained_Memory/2026-03-26_22-29-57/checkpoints/adapter_final.pt",
    # "VAL_MANIFEST": "/kaggle/input/datasets/akarshkumarshukla/val-data/val_manifest.jsonl",
    # "BATCH_SIZE": "64",
    # "VERIFY_LSTM_MEMORY": "1",  # prints AdapterMemoryCell.forward count (expect 42)
}

result = subprocess.run(
    [sys.executable, INFERENCE_SCRIPT],
    cwd="/kaggle/working",
    env=env,
    stdout=None,
    stderr=subprocess.STDOUT,
)

if result.returncode != 0:
    raise RuntimeError(f"Inference failed with exit code {result.returncode}")

print("\nInference completed successfully.")
print("Check /kaggle/working/inference_results/ for outputs.")
