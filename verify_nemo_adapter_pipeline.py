#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep architectural survey & verification for NeMo Parakeet-TDT 1.1B adapter training.

Run BEFORE full training (same env as training: NeMo, torch, etc.):
  python verify_nemo_adapter_pipeline.py [--manifest PATH] [--out DIR] [--quick]

On Kaggle:
  - Same “safe” pattern as training (Part C): use a fresh interpreter, not %run in the notebook.
    After Part A–C, set RUN_ADAPTER_VERIFY=1 in the one-cell driver to run Part D (subprocess).
  - Or run manually: !python /kaggle/working/verify_nemo_adapter_pipeline.py ...
  - ImportError: cannot import '_center' from numpy._core.umath → run once:
        python verify_nemo_adapter_pipeline.py --fix-numpy-scipy
    then Kernel → Restart, and run verify again.

Outputs under --out (default: ./adapter_verify_report):
  - VERIFY_REPORT.md
  - plots: lr_schedule.png, spec_augment_mel.png, waveform_aug.png (if matplotlib)

IMPORTANT — Do not paste ``kaggle_one_cell_nemo_adapter_06b_v2.py`` Part A (pip uninstall/install)
into this file. That removes the NeMo/torch imports and causes::

    NameError: name 'AdapterModuleUtil' is not defined

Keep this script as verification-only; run Part A in the notebook, then run this file via
``python verify_nemo_adapter_pipeline.py`` or Part D (subprocess).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=Warning, module="numba")

# ---------------------------------------------------------------------------
# Next: preflight → matplotlib (optional) → torch/nemo imports → adapter classes.
# Never insert the Kaggle one-cell Part A pip loop here (it drops AdapterModuleUtil).
# ---------------------------------------------------------------------------


def _preflight_numpy_scipy_before_lightning_nemo() -> None:
    """Lightning → torchmetrics → scipy → numpy; broken wheels show up as numpy._core errors."""
    try:
        import numpy as _np  # noqa: F401

        _ = _np.__version__
        import scipy.signal  # noqa: F401
    except Exception as err:
        print("=" * 72, file=sys.stderr)
        print(
            "NUMPY/SCIPY import chain failed (Kaggle kernels often break this after mixed pip).",
            file=sys.stderr,
        )
        print(f"Error ({type(err).__name__}): {err}", file=sys.stderr)
        print("", file=sys.stderr)
        print("  1) Run once, then RESTART JUPYTER KERNEL:", file=sys.stderr)
        print(
            "       python verify_nemo_adapter_pipeline.py --fix-numpy-scipy",
            file=sys.stderr,
        )
        print("", file=sys.stderr)
        print('  2) Or: pip install --force-reinstall "numpy>=2.1,<2.3" "scipy>=1.14,<1.16"', file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "  3) Or: run verify in a fresh subprocess after your one-cell training installer.",
            file=sys.stderr,
        )
        print("=" * 72, file=sys.stderr)
        raise SystemExit(2) from err


if "--fix-numpy-scipy" in sys.argv:
    import subprocess

    sys.argv = [a for a in sys.argv if a != "--fix-numpy-scipy"]
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--force-reinstall",
            "numpy>=2.1,<2.3",
            "scipy>=1.14,<1.16",
        ]
    )
    print(
        "pip: numpy/scipy reinstalled. Restart the Jupyter kernel, then run verification again."
    )
    raise SystemExit(0)

_preflight_numpy_scipy_before_lightning_nemo()

# ---------------------------------------------------------------------------
# Optional plotting
# ---------------------------------------------------------------------------
_HAS_MPL = False
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:
    plt = None  # type: ignore

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# ---------------------------------------------------------------------------
# Mirror training script: AdapterChainState + ChainedLinearAdapter + WaveformAugmentor
# (Keep in sync with kaggle_one_cell_nemo_adapter_06b_v2.py train_code.)
# Requires: AdapterModuleUtil + LinearAdapter + nn + F. No pip/install code here.
# ---------------------------------------------------------------------------
from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil, LinearAdapter
from nemo.core.classes import adapter_mixins
from omegaconf import OmegaConf, open_dict


class AdapterChainState:
    def __init__(self):
        self.prev_bottleneck = None
        self.layer_counter = 0
        self._reset_count = 0

    def reset(self):
        self.prev_bottleneck = None
        self.layer_counter = 0
        self._reset_count += 1

    def update(self, bottleneck):
        self.prev_bottleneck = bottleneck
        self.layer_counter += 1


class ChainedLinearAdapter(nn.Module, AdapterModuleUtil):
    """Same as kaggle_one_cell train_code: replace LinearAdapter + sync adapter_layer."""

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
    def __init__(
        self,
        speed_min=0.90,
        speed_max=1.10,
        pitch_min=-1.0,
        pitch_max=3.0,
        sample_rate=16000,
        speed_prob=0.5,
        pitch_prob=0.5,
        classroom_noise_prob=0.5,
        classroom_snr_min=5.0,
        classroom_snr_max=10.0,
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
        self.classroom_noise_prob = classroom_noise_prob
        self.classroom_snr_min = classroom_snr_min
        self.classroom_snr_max = classroom_snr_max
        self.noise_file_paths = list(noise_file_paths or [])
        self.gain_prob = gain_prob
        self.gain_db_min = gain_db_min
        self.gain_db_max = gain_db_max

    @staticmethod
    def _mix_at_snr(speech, noise, snr_db):
        eps = 1e-8
        p_s = speech.pow(2).mean().clamp_min(eps)
        p_n = noise.pow(2).mean().clamp_min(eps)
        snr_lin = 10 ** (snr_db / 10.0)
        alpha = torch.sqrt(p_s / (snr_lin * p_n + eps))
        return speech + alpha * noise

    def _load_noise_segment(self, num_samples, device, dtype):
        paths = self.noise_file_paths
        if not paths:
            return None
        import random

        path = paths[random.randint(0, len(paths) - 1)]
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
            reps = (num_samples + n - 1) // n
            wav = wav.repeat(reps)[:num_samples]
        else:
            import random

            start = random.randint(0, n - num_samples)
            wav = wav[start : start + num_samples]
        return wav.to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def forward(self, audio_signal, signal_length):
        if not self.training:
            return audio_signal, signal_length
        B, T = audio_signal.shape
        if torch.rand(1).item() < self.speed_prob:
            speed_factor = self.speed_min + torch.rand(1).item() * (
                self.speed_max - self.speed_min
            )
            new_T = max(1, int(T / speed_factor))
            audio_signal = F.interpolate(
                audio_signal.unsqueeze(1),
                size=new_T,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            signal_length = (signal_length.float() / speed_factor).long().clamp(
                min=1, max=new_T
            )
        if torch.rand(1).item() < self.pitch_prob:
            n_steps = self.pitch_min + torch.rand(1).item() * (
                self.pitch_max - self.pitch_min
            )
            try:
                audio_signal = torchaudio.functional.pitch_shift(
                    audio_signal, self.sample_rate, n_steps
                )
            except Exception:
                pass
        if self.noise_file_paths and torch.rand(1).item() < self.classroom_noise_prob:
            snr_lo, snr_hi = self.classroom_snr_min, self.classroom_snr_max
            for b in range(B):
                L = int(min(signal_length[b].item(), audio_signal.shape[1]))
                if L < 1:
                    continue
                noise_seg = self._load_noise_segment(
                    L, audio_signal.device, audio_signal.dtype
                )
                if noise_seg is None:
                    continue
                snr = snr_lo + torch.rand(1).item() * (snr_hi - snr_lo)
                seg = audio_signal[b, :L].float()
                mixed = self._mix_at_snr(seg, noise_seg, snr)
                audio_signal[b, :L] = mixed.to(dtype=audio_signal.dtype)
        if torch.rand(1).item() < self.gain_prob:
            gain_db = self.gain_db_min + torch.rand(1).item() * (
                self.gain_db_max - self.gain_db_min
            )
            audio_signal = audio_signal * (10 ** (gain_db / 20))
        return audio_signal, signal_length


# ---------------------------------------------------------------------------
# Report aggregation
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Report:
    checks: List[CheckResult] = field(default_factory=list)
    sections: Dict[str, str] = field(default_factory=dict)

    def add(self, name: str, passed: bool, detail: str = ""):
        self.checks.append(CheckResult(name, passed, detail))

    def summary_md(self) -> str:
        lines = ["# Adapter Pipeline Verification Report", ""]
        lines.append("## Executive Summary")
        for c in self.checks:
            st = "PASS" if c.passed else "FAIL"
            lines.append(f"- **{st}**: {c.name}")
            if c.detail:
                lines.append(f"  - {c.detail}")
        lines.append("")
        for title, body in self.sections.items():
            lines.append(f"## {title}")
            lines.append(body)
            lines.append("")
        return "\n".join(lines)


def _encoder_target_key(model_cfg):
    enc = model_cfg.encoder
    if "_target_" in enc:
        return "_target_"
    if "target" in enc:
        return "target"
    raise AttributeError("encoder target key missing")


def apply_training_setup(
    model: ASRModel,
    cfg_model_adapter_linear: Any,
    chain_state: AdapterChainState,
    device: torch.device,
) -> Tuple[int, List[ChainedLinearAdapter]]:
    adapter_full_name = "encoder:asr_children_adapter"
    adapter_short_name = "asr_children_adapter"
    model.add_adapter(name=adapter_full_name, cfg=cfg_model_adapter_linear)
    model.set_enabled_adapters(enabled=False)
    model.set_enabled_adapters(name=adapter_full_name, enabled=True)
    model.freeze()
    model.unfreeze_enabled_adapters()
    model.adapter_chain_state = chain_state

    adapter_module_list = []
    for mod_name, module in list(model.named_modules()):
        if isinstance(module, LinearAdapter) and adapter_short_name in mod_name:
            adapter_module_list.append((mod_name, module))
    adapter_module_list.sort(key=lambda x: x[0])

    chained_list: List[ChainedLinearAdapter] = []
    for i, (mod_name, _orig) in enumerate(adapter_module_list):
        is_first_layer = i == 0
        new_adapter = ChainedLinearAdapter(
            in_features=1024,
            dim=128,
            activation="gelu",
            norm_position="post",
            dropout=0.1,
            is_first=is_first_layer,
            chain_state_ref=chain_state,
            adapter_strategy=None,
        ).to(device)
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
            if hasattr(layer_mod, "adapter_layer") and adapter_short_name in layer_mod.adapter_layer:
                layer_mod.adapter_layer[adapter_short_name] = new_adapter
        new_adapter.setup_adapter_strategy(None)
        chained_list.append(new_adapter)

    LinearAdapter.register(ChainedLinearAdapter)

    # Joint + decoder (mirror train script)
    if hasattr(model, "joint"):
        for p in model.joint.parameters():
            p.requires_grad = True
    if hasattr(model, "decoder") and hasattr(model.decoder, "prediction"):
        pred = model.decoder.prediction
        if hasattr(pred, "embed"):
            for p in pred.embed.parameters():
                p.requires_grad = True
        for _, mod in pred.named_modules():
            if isinstance(mod, nn.LSTM) and mod.num_layers >= 1:
                li = mod.num_layers - 1
                for pname, p in mod.named_parameters():
                    if f"_l{li}" in pname:
                        p.requires_grad = True

    if hasattr(model, "decoder"):
        model.decoder.train()
    if hasattr(model, "joint"):
        model.joint.train()

    return len(chained_list), chained_list


def module_train_tree(model: nn.Module, max_lines: int = 200) -> str:
    lines = []
    n = 0
    for name, m in model.named_modules():
        if n >= max_lines:
            lines.append("... (truncated)")
            break
        lines.append(f"  {name or '<root>'}: training={m.training}")
        n += 1
    return "\n".join(lines)


def verify_architecture(model, chained_adapters: List[ChainedLinearAdapter], report: Report):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0

    enc_p = sum(
        p.numel() for n, p in model.named_parameters() if n.startswith("encoder.")
    )
    dec_p = sum(
        p.numel() for n, p in model.named_parameters() if n.startswith("decoder.")
    )
    joint_p = sum(
        p.numel() for n, p in model.named_parameters() if n.startswith("joint.")
    )
    ad_p = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and ("adapter" in n.lower() or "chain_proj" in n.lower())
    )

    body = []
    body.append(f"| Metric | Value |\n|---|---|\n")
    body.append(f"| Total params | {total:,} |\n")
    body.append(f"| Trainable | {trainable:,} ({pct:.3f}%) |\n")
    body.append(f"| Encoder (named) | {enc_p:,} |\n")
    body.append(f"| Decoder (named) | {dec_p:,} |\n")
    body.append(f"| Joint (named) | {joint_p:,} |\n")
    body.append(f"| Trainable w/ adapter-like keys | ~{ad_p:,} |\n")

    n_ad = len(chained_adapters)
    first_ok = chained_adapters[0].is_first and not hasattr(
        chained_adapters[0], "chain_proj"
    )
    rest_chain = all(
        hasattr(chained_adapters[i], "chain_proj")
        for i in range(1, len(chained_adapters))
    )

    ortho_scores = []
    for i, m in enumerate(chained_adapters):
        if hasattr(m, "chain_proj"):
            W = m.chain_proj.weight.detach().float()
            if W.shape[0] == W.shape[1]:
                Q = W / 0.9
                I = torch.eye(W.shape[0], device=W.device)
                err = (Q @ Q.T - I).abs().mean().item()
                ortho_scores.append((i, err))

    report.add(
        "Architecture: 42 adapters",
        n_ad == 42,
        f"Found {n_ad} ChainedLinearAdapter modules (expected 42).",
    )
    report.add(
        "Architecture: first adapter no chain_proj",
        first_ok,
        "First layer is_first and no chain_proj" if first_ok else "Mismatch",
    )
    report.add(
        "Architecture: 41x chain_proj",
        rest_chain and n_ad == 42,
        "Layers 1..41 have chain_proj" if rest_chain else "Mismatch",
    )
    report.add(
        "Architecture: orthogonal chain_proj (Q=W/0.9)",
        len(ortho_scores) == 41 and all(s[1] < 0.05 for s in ortho_scores),
        f"max mean|QQ^T-I|: {max(s[1] for s in ortho_scores) if ortho_scores else 'n/a'}",
    )

    body.append("\n### Adapter chain checks\n")
    body.append(f"- Count: {n_ad}\n")
    body.append(f"- First is_first + no chain_proj: {first_ok}\n")
    body.append(f"- Rest have chain_proj: {rest_chain}\n")

    tree = module_train_tree(model, max_lines=80)
    body.append("\n### Module train flags (partial tree)\n```\n" + tree + "\n```\n")

    ascii_diagram = """
    [Audio] -> WaveformAug (train) -> Preprocessor -> Encoder (mostly eval, adapters train)
           -> SpecAug (train) -> ... -> Decoder (train) -> Joint (train) -> Loss
    """
    body.append("\n### ASCII flow\n```" + ascii_diagram + "```\n")

    report.sections["Architecture"] = "".join(body)


def attach_adapter_hooks(
    chained_adapters: List[ChainedLinearAdapter],
    records: List[dict],
):
    def make_hook(idx):
        def hook(_mod, inp, out):
            x = inp[0]
            records.append(
                {
                    "layer_idx": idx,
                    "in_shape": tuple(x.shape),
                    "out_shape": tuple(out.shape) if torch.is_tensor(out) else None,
                    "is_first": _mod.is_first,
                }
            )

        return hook

    handles = []
    for i, m in enumerate(chained_adapters):
        h = m.register_forward_hook(make_hook(i))
        handles.append(h)
    return handles


def _call_asr_preprocessor(preprocessor, input_signal: torch.Tensor, lengths: torch.Tensor):
    """Preprocessor NeuralType ports use `length`; top-level ASR forward uses `input_signal_length`."""
    try:
        return preprocessor(input_signal=input_signal, length=lengths)
    except Exception:
        return preprocessor(input_signal=input_signal, input_signal_length=lengths)


def verify_forward_backward(
    model: ASRModel,
    chained_adapters: List[ChainedLinearAdapter],
    chain_state: AdapterChainState,
    device: torch.device,
    report: Report,
):
    model.train()
    if hasattr(model, "decoder"):
        model.decoder.train()
    if hasattr(model, "joint"):
        model.joint.train()
    if hasattr(model, "waveform_augmentor"):
        model.waveform_augmentor.train()
    if getattr(model, "spec_augmentation", None) is not None:
        model.spec_augmentation.train()

    hook_records: List[dict] = []
    handles = attach_adapter_hooks(chained_adapters, hook_records)
    reset_before = chain_state._reset_count

    B = 2
    T1, T2 = 32000, 24000
    T = max(T1, T2)
    signals = torch.zeros(B, T, device=device, dtype=torch.float32)
    signals[0, :T1] = torch.randn(T1, device=device) * 0.01
    signals[1, :T2] = torch.randn(T2, device=device) * 0.01
    lengths = torch.tensor([T1, T2], device=device, dtype=torch.int64)

    try:
        with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.bfloat16):
            out = model.forward(
                input_signal=signals, input_signal_length=lengths
            )
    except Exception as e:
        for h in handles:
            h.remove()
        report.add("Forward dry-run", False, str(e))
        report.sections["Forward"] = f"Forward failed: {e}"
        return

    for h in handles:
        h.remove()

    reset_after = chain_state._reset_count
    report.add(
        "Forward: chain_state.reset in forward",
        reset_after > reset_before,
        f"reset calls: {reset_after - reset_before} (expected >= 1)",
    )
    report.add(
        "Forward: adapter hooks fired",
        len(hook_records) >= 42,
        f"Hook records: {len(hook_records)}",
    )

    fwd_body = [
        f"- Forward output type: {type(out)}\n",
        f"- Adapter hook records: {len(hook_records)}\n",
        f"- Sample hook[0]: {hook_records[0] if hook_records else 'none'}\n",
    ]

    # Backward via encoder path (always in graph for trainable adapters)
    model.zero_grad(set_to_none=True)
    try:
        pre, pre_len = _call_asr_preprocessor(model.preprocessor, signals, lengths)
        enc_out = model.encoder(audio_signal=pre, length=pre_len)
        enc_tensor = enc_out[0] if isinstance(enc_out, tuple) else enc_out
        loss = enc_tensor.float().sum()
    except Exception as e:
        report.add("Backward setup", False, str(e))
        report.sections["Forward/Backward"] = (
            "".join(fwd_body) + f"\nEncoder backward prep failed: {e}"
        )
        return

    loss.backward()
    report.add("Backward: loss.backward()", True, "Completed")

    grad_stats = defaultdict(list)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            grad_stats["missing_grad"].append(name)
            continue
        g = p.grad.detach().float()
        key = "adapter" if ("adapter" in name.lower() or "chain_proj" in name.lower()) else (
            "joint" if "joint" in name.lower() else (
                "decoder" if "decoder" in name.lower() or "prediction" in name.lower() else "other"
            )
        )
        grad_stats[key].append((name, g.mean().item(), g.std().item(), g.abs().max().item()))

    frozen_enc_grad = [
        n
        for n, p in model.named_parameters()
        if n.startswith("encoder.") and "adapter" not in n.lower() and p.grad is not None and p.grad.abs().max() > 0
    ]
    # Some encoder non-adapter params might get tiny numerical noise; strict: should be empty
    report.add(
        "Backward: frozen encoder (non-adapter) ideally no grad",
        len(frozen_enc_grad) == 0,
        f"Non-adapter encoder params with grad: {len(frozen_enc_grad)}",
    )

    bw_body = fwd_body + [
        "\n### Gradient stats (subset)\n",
    ]
    for k in ["adapter", "joint", "decoder", "other"]:
        if k not in grad_stats or not grad_stats[k]:
            continue
        bw_body.append(f"**{k}** ({len(grad_stats[k])} tensors with grad)\n")
        for item in grad_stats[k][:5]:
            bw_body.append(f"  - {item[0]}: mean={item[1]:.2e} std={item[2]:.2e} max|.|={item[3]:.2e}\n")

    if grad_stats["missing_grad"]:
        bw_body.append(
            f"\nMissing grad (trainable): {len(grad_stats['missing_grad'])} names\n"
        )

    report.sections["Forward/Backward"] = "".join(bw_body)

    # Eval mode: augment off
    model.eval()
    hook_records.clear()
    handles = attach_adapter_hooks(chained_adapters, hook_records)
    with torch.no_grad():
        _ = model.forward(input_signal=signals, input_signal_length=lengths)
    for h in handles:
        h.remove()
    report.add("Eval forward (no grad)", True, f"Adapter hooks in eval: {len(hook_records)}")


def verify_optimizer_scheduler(
    model: ASRModel,
    num_epochs: int,
    steps_per_epoch: int,
    out_dir: Path,
    report: Report,
):
    adapter_params, joint_params, decoder_params = [], [], []
    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        nl = pname.lower()
        if "adapter" in nl or "chain" in nl:
            adapter_params.append(param)
        elif "joint" in nl:
            joint_params.append(param)
        elif "decoder" in nl or "prediction" in nl or "embed" in nl:
            decoder_params.append(param)
        else:
            adapter_params.append(param)

    param_groups = []
    if adapter_params:
        param_groups.append({"params": adapter_params, "lr": 5e-4})
    if joint_params:
        param_groups.append({"params": joint_params, "lr": 1e-4})
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": 5e-5})

    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=0.01)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(0.15 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    lrs = [[] for _ in param_groups]
    for step in range(total_steps):
        for i, pg in enumerate(param_groups):
            base = pg["lr"]
            lrs[i].append(base * lr_lambda(step))
        sched.step()

    report.add(
        "Optimizer: 3 groups LRs",
        len(param_groups) == 3,
        f"Groups: {[pg['lr'] for pg in param_groups]}",
    )
    report.add(
        "Optimizer: AdamW betas/weight_decay",
        True,
        "betas=(0.9,0.999), weight_decay=0.01 (constructed in script)",
    )

    if _HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, pg in enumerate(param_groups):
            ax.plot(lrs[i], label=f"group{i} base_lr={pg['lr']}")
        ax.set_xlabel("step")
        ax.set_ylabel("LR")
        ax.legend()
        ax.set_title("Simulated LambdaLR warmup+cosine (per-group base)")
        p = out_dir / "lr_schedule.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        report.sections["LR schedule plot"] = f"Saved `{p}`"
    else:
        report.sections["LR schedule plot"] = "matplotlib unavailable; skipped plot"

    report.sections["Scheduler note"] = (
        "Plot uses LambdaLR warmup+cosine on group base LRs (matches training script). "
        "NeMo CosineAnnealing also applies min_lr=1e-6 to the *global* schedule in full training — "
        "verify training script if you need exact parity."
    )

    body = "| Group | base_lr | #params |\n|---|---|---|\n"
    for i, pg in enumerate(param_groups):
        n = sum(p.numel() for p in pg["params"])
        body += f"| {i} | {pg['lr']} | {n:,} |\n"
    body += f"\nSimulated total_steps={total_steps}, warmup_steps={warmup_steps}\n"
    report.sections["Optimizer groups"] = body


def verify_spec_augment(model: ASRModel, device: torch.device, out_dir: Path, report: Report):
    sa = getattr(model, "spec_augmentation", None)
    if sa is None:
        report.add("SpecAugment", False, "model.spec_augmentation is None")
        return
    sa.train()
    B, Freq, Time = 2, 80, 100
    x = torch.randn(B, Freq, Time, device=device, requires_grad=False)
    lengths = torch.tensor([Time, Time - 10], device=device)
    with torch.cuda.amp.autocast(enabled=False):
        y = sa(input_spec=x, length=lengths)
    report.add("SpecAugment forward", y.shape == x.shape, f"shape {tuple(y.shape)}")

    if _HAS_MPL:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        ax[0].imshow(x[0].cpu().numpy(), aspect="auto", origin="lower")
        ax[0].set_title("Before")
        ax[1].imshow(y[0].detach().cpu().numpy(), aspect="auto", origin="lower")
        ax[1].set_title("After SpecAug (train)")
        p = out_dir / "spec_augment_mel.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        report.sections["SpecAugment plot"] = f"Saved `{p}`"


def verify_waveform_aug(
    noise_files: List[str], manifest_path: Optional[str], out_dir: Path, report: Report
):
    device = torch.device("cpu")
    paths_3: List[str] = []
    if manifest_path and os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    p = row.get("audio_filepath")
                    if p and os.path.isfile(p):
                        paths_3.append(p)
                except json.JSONDecodeError:
                    continue
                if len(paths_3) >= 3:
                    break

    aug = WaveformAugmentor(
        noise_file_paths=noise_files[:50] if noise_files else [],
        speed_prob=1.0,
        pitch_prob=0.0,
        classroom_noise_prob=0.0,
        gain_prob=0.0,
    )
    aug.train()
    if paths_3:
        w, sr = torchaudio.load(paths_3[0])
        if w.shape[0] > 1:
            w = w.mean(0, keepdim=True)
        w = w[:, : min(16000 * 3, w.shape[1])]
        sig = w.squeeze(0).unsqueeze(0)
        lens = torch.tensor([sig.shape[1]])
        out, _ = aug(sig, lens.clone())
        report.add("Waveform aug (speed-only forced)", True, f"shape {tuple(sig.shape)} -> {tuple(out.shape)}")
        if _HAS_MPL:
            fig, ax = plt.subplots(2, 1, figsize=(10, 4))
            ax[0].plot(sig[0].numpy()[:4000])
            ax[0].set_title("Before")
            ax[1].plot(out[0].detach().numpy()[:4000])
            ax[1].set_title("After speed aug (prob=1)")
            p = out_dir / "waveform_aug.png"
            fig.savefig(p, dpi=120, bbox_inches="tight")
            plt.close(fig)
    else:
        report.add(
            "Waveform aug manifest samples",
            False,
            "Could not load 3 paths from manifest; skipped real-audio aug plot",
        )

    aug.eval()
    sig = torch.randn(1, 8000)
    out_eval, _ = aug(sig, torch.tensor([8000]))
    report.add(
        "Waveform aug disabled in eval",
        torch.allclose(sig, out_eval),
        "eval mode returns input unchanged",
    )


def verify_dataloader(
    model: ASRModel, manifest: str, batch_size: int, num_workers: int, report: Report
):
    if not os.path.isfile(manifest):
        report.add("Dataloader", False, f"Manifest missing: {manifest}")
        return
    try:
        from omegaconf import open_dict

        with open_dict(model.cfg.train_ds):
            model.cfg.train_ds.manifest_filepath = manifest
            model.cfg.train_ds.batch_size = min(batch_size, 4)
            model.cfg.train_ds.num_workers = min(num_workers, 2)
            model.cfg.train_ds.shuffle = False
        model.setup_training_data(model.cfg.train_ds)
        dl = model._train_dl
        it = iter(dl)
        batch = next(it)
        report.add(
            "Dataloader 1 batch",
            True,
            f"keys: {list(batch.keys()) if isinstance(batch, dict) else type(batch)}",
        )
        body = f"Batch structure: { {k: (v.shape if hasattr(v, 'shape') else type(v)) for k, v in (batch.items() if isinstance(batch, dict) else [])} }\n"
        report.sections["Dataloader"] = body
    except Exception as e:
        report.add("Dataloader", False, str(e))


def memory_estimate(model: ASRModel, report: Report):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_p = 4  # fp32 reference
    model_mb = total * bytes_p / 1e6
    grad_mb = trainable * bytes_p / 1e6
    optim_mb = trainable * bytes_p * 2 / 1e6  # Adam m,v rough
    act_mb = 500  # rough order for 1 step 1.1B forward (very approximate)
    peak = model_mb + grad_mb + optim_mb + act_mb
    body = f"""
| Item | MB (approx) |
|---|---|
| Weights fp32 equiv | {model_mb:.0f} |
| Gradients | {grad_mb:.0f} |
| Optimizer state (2x) | {optim_mb:.0f} |
| Activations (rough) | {act_mb} |
| **Sum (rough peak)** | **{peak:.0f}** |

Note: bf16 training uses less memory than this fp32-equivalent table.
"""
    report.sections["Memory estimate"] = body
    report.add("Memory table", True, "See Memory estimate section")


def benchmark_step(model: ASRModel, device: torch.device, report: Report):
    model.train()
    B, T = 2, 32000
    sig = torch.randn(B, T, device=device, dtype=torch.float32) * 0.01
    lens = torch.tensor([T, T - 1000], device=device, dtype=torch.int64)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.bfloat16):
            out = model.forward(input_signal=sig, input_signal_length=lens)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        report.add("Bench forward", True, f"{(t1 - t0) * 1000:.1f} ms")
        report.sections["Benchmark"] = f"Forward: {(t1-t0)*1000:.1f} ms (1 step, B=2, T=32000)\n"
    except Exception as e:
        report.add("Bench forward", False, str(e))


def verify_reinforce_callback(model: ASRModel, report: Report):
    """ReinforceDecoderJointTrainMode: after eval(), callback restores decoder/joint/wa/sa train."""
    from lightning.pytorch.callbacks import Callback

    class _Cb(Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if getattr(pl_module, "decoder", None) is not None:
                pl_module.decoder.train()
            if getattr(pl_module, "joint", None) is not None:
                pl_module.joint.train()
            wa = getattr(pl_module, "waveform_augmentor", None)
            if wa is not None:
                wa.train()
            sa = getattr(pl_module, "spec_augmentation", None)
            if sa is not None:
                sa.train()

    model.eval()
    _Cb().on_train_batch_start(None, model, None, 0)
    ok = model.decoder.training and model.joint.training
    if hasattr(model, "waveform_augmentor") and model.waveform_augmentor is not None:
        ok = ok and model.waveform_augmentor.training
    sa = getattr(model, "spec_augmentation", None)
    if sa is not None:
        ok = ok and sa.training
    report.add(
        "ReinforceDecoderJointTrainMode behavior",
        ok,
        "decoder/joint/wa/sa .training True after callback from eval",
    )


def verify_save_adapters(model: ASRModel, out_dir: Path, report: Report):
    p = out_dir / "verify_adapters.pt"
    try:
        model.save_adapters(str(p))
        report.add("save_adapters", p.is_file(), str(p))
    except Exception as e:
        report.add("save_adapters", False, str(e))


def verify_nemo_version(report: Report):
    try:
        import nemo

        ver = getattr(nemo, "__version__", "unknown")
        report.add("NeMo import", True, f"version {ver}")
        report.sections["NeMo version"] = ver
    except Exception as e:
        report.add("NeMo import", False, str(e))


def verify_residual_strategy(chained: List[ChainedLinearAdapter], report: Report):
    ok = all(
        hasattr(m, "adapter_strategy")
        and m.adapter_strategy is not None
        for m in chained
    )
    names = {
        type(m.adapter_strategy).__name__ for m in chained[:3]
    }
    report.add(
        "adapter_strategy on chained modules",
        ok,
        f"sample types: {names}",
    )


def main():
    ap = argparse.ArgumentParser(
        epilog=(
            "If NeMo import fails with numpy._core.umath / _center errors, run "
            "`python verify_nemo_adapter_pipeline.py --fix-numpy-scipy`, restart the "
            "Jupyter kernel, then run verification again."
        )
    )
    ap.add_argument(
        "--manifest",
        default=os.environ.get(
            "TRAIN_MANIFEST",
            "/kaggle/input/datasets/akarshkumarshukla/train-manifest/train_manifest.jsonl",
        ),
    )
    ap.add_argument("--out", default="./adapter_verify_report")
    ap.add_argument("--model-id", default="nvidia/parakeet-tdt-1.1b")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--num-epochs", type=int, default=6)
    ap.add_argument("--quick", action="store_true", help="Skip dataloader/benchmark if manifest missing")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = Report()
    verify_nemo_version(report)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    MODEL_ID = args.model_id
    noise_dirs = [
        "/kaggle/input/datasets/akarshkumarshukla/noise-1",
        "/kaggle/input/datasets/akarshkumarshukla/asr-data/noise_part_0",
    ]
    noise_files = []
    for d in noise_dirs:
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith((".wav", ".flac", ".mp3")):
                        noise_files.append(os.path.join(root, f))
                if len(noise_files) > 200:
                    break

    cfg_linear = OmegaConf.create(
        {
            "_target_": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
            "in_features": 1024,
            "dim": 128,
            "activation": "gelu",
            "norm_position": "post",
            "dropout": 0.1,
        }
    )
    print(f"Loading {MODEL_ID} on {device}...")
    model_cfg = ASRModel.from_pretrained(MODEL_ID, return_config=True)
    enc_key = _encoder_target_key(model_cfg)
    with open_dict(model_cfg):
        meta = adapter_mixins.get_registered_adapter(model_cfg.encoder[enc_key])
        if meta is not None:
            model_cfg.encoder[enc_key] = meta.adapter_class_path

    model = ASRModel.from_pretrained(MODEL_ID, override_config_path=model_cfg)
    model = model.to(device)
    with open_dict(model.cfg):
        if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
            model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    if hasattr(model, "change_decoding_strategy"):
        model.change_decoding_strategy(model.cfg.decoding)

    # SpecAug + waveform module (forward wrap AFTER adapter_chain_state exists)
    model.spec_augmentation = model.from_config_dict(
        OmegaConf.create(
            {
                "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
                "freq_masks": 2,
                "freq_width": 27,
                "time_masks": 10,
                "time_width": 0.05,
            }
        )
    )
    model.add_module(
        "waveform_augmentor",
        WaveformAugmentor(noise_file_paths=noise_files[:500]).to(device),
    )

    chain_state = AdapterChainState()
    _n, chained = apply_training_setup(model, cfg_linear, chain_state, device)

    _orig = model.forward.__func__

    def _aug(self, input_signal=None, input_signal_length=None, **kw):
        if hasattr(self, "adapter_chain_state"):
            self.adapter_chain_state.reset()
        if self.training and input_signal is not None and input_signal_length is not None:
            input_signal, input_signal_length = self.waveform_augmentor(
                input_signal, input_signal_length
            )
        return _orig(
            self,
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            **kw,
        )

    model.forward = types.MethodType(_aug, model)

    if hasattr(model, "waveform_augmentor"):
        model.waveform_augmentor.train()
    if model.spec_augmentation is not None:
        model.spec_augmentation.train()

    verify_architecture(model, chained, report)
    verify_residual_strategy(chained, report)
    verify_forward_backward(model, chained, chain_state, device, report)
    verify_spec_augment(model, device, out_dir, report)
    verify_waveform_aug(noise_files, args.manifest if os.path.isfile(args.manifest) else None, out_dir, report)

    steps_per_epoch = 1000
    if not args.quick and os.path.isfile(args.manifest):
        verify_dataloader(model, args.manifest, args.batch_size, 2, report)
        try:
            steps_per_epoch = max(1, len(model._train_dl))
        except Exception:
            pass
    else:
        report.add("Dataloader", False, "Skipped (--quick or missing manifest)")

    verify_optimizer_scheduler(model, args.num_epochs, steps_per_epoch, out_dir, report)
    memory_estimate(model, report)
    benchmark_step(model, device, report)
    verify_reinforce_callback(model, report)
    verify_save_adapters(model, out_dir, report)

    report.sections["Limitations / not covered"] = (
        "- Full RNNT `training_step` + transducer loss backward is not exercised; encoder-path "
        "`loss=encoder(...).sum()` validates adapter gradients.\n"
        "- `.nemo` full checkpoint save can be large; only `save_adapters` is verified by default.\n"
        "- Decoder gradient flow is inferred from trainable flags + optional full-step training on your side.\n"
        "- Keep `ChainedLinearAdapter` / `AdapterChainState` / aug hyperparams in sync with "
        "`kaggle_one_cell_nemo_adapter_06b_v2.py` train_code.\n"
    )

    md = report.summary_md()
    (out_dir / "VERIFY_REPORT.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nFull report written to {out_dir / 'VERIFY_REPORT.md'}")

    failed = [c for c in report.checks if not c.passed]
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
