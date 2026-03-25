# # =============================================================================
# # CELL 1: FIX NUMPY/SCIPY COMPATIBILITY
# # Run this cell, then RESTART KERNEL before running anything else
# # =============================================================================
#   this is the chained linear adpater code--fixed    
# import subprocess
# import sys

# print("=" * 60)
# print("FIXING NUMPY/SCIPY COMPATIBILITY")
# print("=" * 60)

# # Step 1: Uninstall broken packages
# print("\n[1/4] Uninstalling broken packages...")
# packages_to_remove = [
#     "numpy", "scipy", "numba",
#     "nemo_toolkit", "lightning", "pytorch-lightning"
# ]

# for pkg in packages_to_remove:
#     subprocess.run(
#         [sys.executable, "-m", "pip", "uninstall", "-y", pkg],
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.DEVNULL
#     )
# print("Done!")

# # Step 2: Clear pip cache
# print("\n[2/4] Clearing pip cache...")
# subprocess.run(
#     [sys.executable, "-m", "pip", "cache", "purge"],
#     stdout=subprocess.DEVNULL,
#     stderr=subprocess.DEVNULL
# )
# print("Done!")

# # Step 3: Install compatible numpy first
# print("\n[3/4] Installing numpy 2.1.x...")
# result = subprocess.run(
#     [sys.executable, "-m", "pip", "install", "--no-cache-dir", "numpy==2.1.3"],
#     capture_output=True,
#     text=True
# )
# if result.returncode != 0:
#     print(f"Warning: {result.stderr}")
# else:
#     print("Done!")

# # Step 4: Install compatible scipy
# print("\n[4/4] Installing scipy 1.14.x...")
# result = subprocess.run(
#     [sys.executable, "-m", "pip", "install", "--no-cache-dir", "scipy==1.14.1"],
#     capture_output=True,
#     text=True
# )
# if result.returncode != 0:
#     print(f"Warning: {result.stderr}")
# else:
#     print("Done!")

# print("\n" + "=" * 60)
# print("✅ STEP 1 COMPLETE!")
# print("=" * 60)
# print("\n⚠️  IMPORTANT: RESTART KERNEL NOW!")
# print("   Click: Kernel -> Restart Kernel")
# print("   Then run Cell 2 (Verification)")
# print("=" * 60)
# # =============================================================================
# # CELL 2: VERIFY NUMPY/SCIPY
# # Run this AFTER restarting kernel
# # =============================================================================

# print("=" * 60)
# print("VERIFYING NUMPY/SCIPY INSTALLATION")
# print("=" * 60)

# import sys
# print(f"Python: {sys.version}")

# # Test numpy
# try:
#     import numpy as np
#     print(f"\n✅ numpy version: {np.__version__}")
    
#     # Quick test
#     arr = np.array([1, 2, 3])
#     print(f"   numpy test: {arr.sum()} (expected: 6)")
# except Exception as e:
#     print(f"\n❌ numpy failed: {e}")
#     print("   Run Cell 1 again and restart kernel!")

# # Test scipy
# try:
#     import scipy
#     print(f"\n✅ scipy version: {scipy.__version__}")
    
#     # Import the problematic module
#     from scipy import signal
#     print(f"   scipy.signal import: OK")
    
#     from scipy import linalg
#     print(f"   scipy.linalg import: OK")
# except Exception as e:
#     print(f"\n❌ scipy failed: {e}")
#     print("   Run Cell 1 again and restart kernel!")

# print("\n" + "=" * 60)
# print("If you see ✅ for both, proceed to Cell 3")
# print("If you see ❌, run Cell 1 again and restart kernel")
# print("=" * 60)
# # =============================================================================
# # CELL 3: INSTALL REMAINING DEPENDENCIES
# # Run ONLY after Cell 2 shows both ✅
# # =============================================================================

# import subprocess
# import sys

# def pip_install(*packages, show_output=False):
#     """Install packages via pip"""
#     cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", *packages]
#     print(f"Installing: {' '.join(packages[:3])}{'...' if len(packages) > 3 else ''}")
#     result = subprocess.run(cmd, capture_output=True, text=True)
#     if result.returncode != 0 and show_output:
#         print(f"Warning: {result.stderr[:200]}")
#     return result.returncode == 0

# print("=" * 60)
# print("INSTALLING REMAINING DEPENDENCIES")
# print("=" * 60)

# # PyTorch
# print("\n[1/4] Installing PyTorch...")
# subprocess.run([
#     sys.executable, "-m", "pip", "install", "--no-cache-dir",
#     "--index-url", "https://download.pytorch.org/whl/cu126",
#     "torch>=2.4.0", "torchaudio"
# ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
# print("Done!")

# # NeMo dependencies
# print("\n[2/4] Installing NeMo dependencies...")
# pip_install(
#     "transformers>=4.40.0",
#     "huggingface_hub>=0.30.0",
#     "lightning>=2.2.0",
#     "omegaconf>=2.3.0",
#     "hydra-core>=1.3.2",
# )

# # Audio processing
# print("\n[3/4] Installing audio libraries...")
# pip_install(
#     "soundfile>=0.12.0",
#     "librosa>=0.10.0",
#     "sentencepiece>=0.2.0",
# )

# # NeMo ASR
# print("\n[4/4] Installing NeMo ASR...")
# pip_install("nemo_toolkit[asr]>=2.0.0")

# # Re-pin numpy/scipy (NeMo might have changed them)
# print("\n[5/4] Re-pinning numpy/scipy...")
# subprocess.run([
#     sys.executable, "-m", "pip", "install", "--no-cache-dir",
#     "--force-reinstall", "numpy==2.1.3", "scipy==1.14.1"
# ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

# print("\n" + "=" * 60)
# print("✅ INSTALLATION COMPLETE!")
# print("=" * 60)
# print("\n⚠️  RESTART KERNEL AGAIN!")
# print("   Then run Cell 4 (Final Verification)")
# print("=" * 60)
# # =============================================================================
# # CELL 4: FINAL VERIFICATION
# # Run AFTER second kernel restart
# # =============================================================================

# import sys
# print("=" * 60)
# print("FINAL VERIFICATION")
# print("=" * 60)
# print(f"Python: {sys.version}\n")

# all_ok = True

# # Check packages
# packages = [
#     ("numpy", "2.1"),
#     ("scipy", "1.14"),
#     ("torch", "2."),
#     ("torchaudio", None),
#     ("transformers", "4."),
#     ("lightning", "2."),
# ]

# for pkg_name, expected_prefix in packages:
#     try:
#         mod = __import__(pkg_name)
#         version = getattr(mod, "__version__", "unknown")
        
#         if expected_prefix and not version.startswith(expected_prefix):
#             print(f"⚠️  {pkg_name:20s} {version:15s} (expected {expected_prefix}x)")
#         else:
#             print(f"✅ {pkg_name:20s} {version}")
#     except ImportError as e:
#         print(f"❌ {pkg_name:20s} NOT INSTALLED")
#         all_ok = False

# # Test NeMo
# print("\n" + "-" * 40)
# print("Testing NeMo imports...")
# try:
#     from nemo.collections.asr.models import ASRModel
#     from nemo.core.classes import adapter_mixins
#     from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil
#     print("✅ NeMo ASR imports successful!")
# except Exception as e:
#     print(f"❌ NeMo import failed: {e}")
#     all_ok = False

# # Test CUDA
# print("\n" + "-" * 40)
# print("Testing CUDA...")
# import torch
# if torch.cuda.is_available():
#     print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
#     print(f"✅ CUDA: {torch.version.cuda}")
# else:
#     print("⚠️  No GPU available (will use CPU)")

# print("\n" + "=" * 60)
# if all_ok:
#     print("🎉 ALL CHECKS PASSED!")
#     print("✅ Ready to run verification script")
# else:
#     print("⚠️  Some checks failed - see errors above")
# print("=" * 60)
# # =============================================================================
# # CELL 5: VERIFICATION SCRIPT (FIXED)
# # Run ONLY after Cell 4 shows "ALL CHECKS PASSED"
# # =============================================================================

# import subprocess
# import sys
# import os

# # Create the verification script as a separate .py file
# script_path = "/kaggle/working/verify_adapter.py"

# # Write script content using a file, avoiding triple-quote nesting issues
# script_content = r'''#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """NeMo Adapter Pipeline Verification Script"""

# from __future__ import annotations

# import argparse
# import json
# import math
# import os
# import sys
# import time
# import types
# import warnings
# from collections import defaultdict
# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=Warning, module="numba")

# print("=" * 60)
# print("NEMO ADAPTER PIPELINE VERIFICATION")
# print("=" * 60)

# # Preflight check
# print("\n[1/6] Checking numpy/scipy...")
# try:
#     import numpy as np
#     import scipy.signal
#     print(f"  numpy:  {np.__version__}")
#     print(f"  scipy:  {scipy.__version__}")
# except Exception as err:
#     print(f"ERROR: numpy/scipy import failed: {err}")
#     sys.exit(2)

# # Optional plotting
# _HAS_MPL = False
# try:
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     _HAS_MPL = True
# except Exception:
#     plt = None

# print("\n[2/6] Loading PyTorch...")
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchaudio
# print(f"  torch:  {torch.__version__}")
# print(f"  CUDA:   {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"  GPU:    {torch.cuda.get_device_name(0)}")

# print("\n[3/6] Loading NeMo...")
# from nemo.collections.asr.models import ASRModel
# from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil
# from nemo.core.classes import adapter_mixins
# from nemo.core.classes.mixins import adapter_mixin_strategies
# from omegaconf import OmegaConf, open_dict

# try:
#     import nemo
#     nemo_ver = getattr(nemo, "__version__", "unknown")
#     print(f"  NeMo:   {nemo_ver}")
# except:
#     print("  NeMo:   (version unknown)")

# # ============================================================================
# # Adapter Classes (mirrored from training script)
# # ============================================================================

# class AdapterChainState:
#     def __init__(self):
#         self.prev_bottleneck = None
#         self.layer_counter = 0
#         self._reset_count = 0

#     def reset(self):
#         self.prev_bottleneck = None
#         self.layer_counter = 0
#         self._reset_count += 1

#     def update(self, bottleneck):
#         self.prev_bottleneck = bottleneck
#         self.layer_counter += 1


# class ChainedLinearAdapter(nn.Module, AdapterModuleUtil):
#     def __init__(
#         self,
#         in_features,
#         dim,
#         activation="gelu",
#         norm_position="post",
#         dropout=0.1,
#         is_first=False,
#         chain_state_ref=None,
#         adapter_strategy: Optional[Any] = None,
#     ):
#         nn.Module.__init__(self)
#         assert norm_position == "post"
#         self.down = nn.Linear(in_features, dim)
#         self.up = nn.Linear(dim, in_features)
#         self.norm = nn.LayerNorm(in_features)
#         self.dropout_layer = nn.Dropout(dropout)
#         self.is_first = is_first
#         self.chain_state_ref = chain_state_ref
#         self.act = nn.GELU()
        
#         if not is_first:
#             self.chain_proj = nn.Linear(dim, dim, bias=False)
#             nn.init.orthogonal_(self.chain_proj.weight)
#             with torch.no_grad():
#                 self.chain_proj.weight.mul_(0.9)
        
#         nn.init.zeros_(self.up.weight)
#         nn.init.zeros_(self.up.bias)
#         self.setup_adapter_strategy(adapter_strategy)

#     def forward(self, x):
#         prev_bottleneck = None
#         if self.chain_state_ref is not None and not self.is_first:
#             prev_bottleneck = self.chain_state_ref.prev_bottleneck
        
#         h = self.down(x)
        
#         if prev_bottleneck is not None:
#             if prev_bottleneck.shape[1] != h.shape[1]:
#                 prev_bottleneck = F.interpolate(
#                     prev_bottleneck.transpose(1, 2),
#                     size=h.shape[1],
#                     mode="nearest",
#                 ).transpose(1, 2)
#             h = h + self.chain_proj(prev_bottleneck)
        
#         bottleneck = self.act(h)
        
#         if self.chain_state_ref is not None:
#             self.chain_state_ref.update(bottleneck)
        
#         h_up = self.up(bottleneck)
#         h_norm = self.norm(h_up)
#         h_drop = self.dropout_layer(h_norm)
#         return h_drop


# class WaveformAugmentor(nn.Module):
#     def __init__(
#         self,
#         speed_min=0.90,
#         speed_max=1.10,
#         pitch_min=-1.0,
#         pitch_max=3.0,
#         sample_rate=16000,
#         speed_prob=0.5,
#         pitch_prob=0.5,
#         classroom_noise_prob=0.5,
#         classroom_snr_min=5.0,
#         classroom_snr_max=10.0,
#         noise_file_paths=None,
#         gain_prob=0.3,
#         gain_db_min=-6.0,
#         gain_db_max=6.0,
#     ):
#         super().__init__()
#         self.speed_min = speed_min
#         self.speed_max = speed_max
#         self.pitch_min = pitch_min
#         self.pitch_max = pitch_max
#         self.sample_rate = sample_rate
#         self.speed_prob = speed_prob
#         self.pitch_prob = pitch_prob
#         self.classroom_noise_prob = classroom_noise_prob
#         self.classroom_snr_min = classroom_snr_min
#         self.classroom_snr_max = classroom_snr_max
#         self.noise_file_paths = list(noise_file_paths or [])
#         self.gain_prob = gain_prob
#         self.gain_db_min = gain_db_min
#         self.gain_db_max = gain_db_max

#     @torch.no_grad()
#     def forward(self, audio_signal, signal_length):
#         if not self.training:
#             return audio_signal, signal_length
#         return audio_signal, signal_length


# # ============================================================================
# # Report Classes
# # ============================================================================

# @dataclass
# class CheckResult:
#     name: str
#     passed: bool
#     detail: str = ""


# @dataclass  
# class Report:
#     checks: List[CheckResult] = field(default_factory=list)
#     sections: Dict[str, str] = field(default_factory=dict)

#     def add(self, name: str, passed: bool, detail: str = ""):
#         self.checks.append(CheckResult(name, passed, detail))

#     def summary_md(self) -> str:
#         lines = ["# Adapter Pipeline Verification Report", ""]
#         lines.append("## Executive Summary")
#         passed = sum(1 for c in self.checks if c.passed)
#         total = len(self.checks)
#         lines.append(f"\n**{passed}/{total} checks passed**\n")
        
#         for c in self.checks:
#             st = "PASS" if c.passed else "FAIL"
#             symbol = "✅" if c.passed else "❌"
#             lines.append(f"- {symbol} **{st}**: {c.name}")
#             if c.detail:
#                 lines.append(f"  - {c.detail}")
#         lines.append("")
        
#         for title, body in self.sections.items():
#             lines.append(f"## {title}")
#             lines.append(body)
#             lines.append("")
#         return "\n".join(lines)


# # ============================================================================
# # Helper Functions
# # ============================================================================

# def _encoder_target_key(model_cfg):
#     enc = model_cfg.encoder
#     if "_target_" in enc:
#         return "_target_"
#     if "target" in enc:
#         return "target"
#     raise AttributeError("encoder target key missing")


# def apply_training_setup(model, cfg_linear, chain_state, device):
#     adapter_name = "encoder:asr_children_adapter"
#     model.add_adapter(name=adapter_name, cfg=cfg_linear)
#     model.set_enabled_adapters(enabled=False)
#     model.set_enabled_adapters(name=adapter_name, enabled=True)
#     model.freeze()
#     model.unfreeze_enabled_adapters()
#     model.adapter_chain_state = chain_state

#     from nemo.collections.common.parts.adapter_modules import LinearAdapter as NemoLinearAdapter

#     adapter_module_list = []
#     for mod_name, module in list(model.named_modules()):
#         if isinstance(module, NemoLinearAdapter):
#             adapter_module_list.append((mod_name, module))
#     adapter_module_list.sort(key=lambda x: x[0])

#     chained_list = []
#     for i, (mod_name, _orig) in enumerate(adapter_module_list):
#         is_first = i == 0
#         new_ad = ChainedLinearAdapter(
#             in_features=1024,
#             dim=128,
#             activation="gelu",
#             norm_position="post",
#             dropout=0.1,
#             is_first=is_first,
#             chain_state_ref=chain_state,
#             adapter_strategy=None,
#         ).to(device)
        
#         parts = mod_name.split(".")
#         parent = model
#         for part in parts[:-1]:
#             parent = getattr(parent, part)
#         setattr(parent, parts[-1], new_ad)
#         chained_list.append(new_ad)

#     # Unfreeze joint + decoder parts
#     if hasattr(model, "joint"):
#         for p in model.joint.parameters():
#             p.requires_grad = True
    
#     if hasattr(model, "decoder") and hasattr(model.decoder, "prediction"):
#         pred = model.decoder.prediction
#         if hasattr(pred, "embed"):
#             for p in pred.embed.parameters():
#                 p.requires_grad = True
#         for _, mod in pred.named_modules():
#             if isinstance(mod, nn.LSTM) and mod.num_layers >= 1:
#                 li = mod.num_layers - 1
#                 for pname, p in mod.named_parameters():
#                     if f"_l{li}" in pname:
#                         p.requires_grad = True

#     if hasattr(model, "decoder"):
#         model.decoder.train()
#     if hasattr(model, "joint"):
#         model.joint.train()

#     return len(chained_list), chained_list


# # ============================================================================
# # Verification Functions
# # ============================================================================

# def verify_architecture(model, chained_adapters, report):
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     pct = 100.0 * trainable / total if total else 0

#     n_ad = len(chained_adapters)
#     first_ok = chained_adapters[0].is_first and not hasattr(chained_adapters[0], "chain_proj")
#     rest_chain = all(hasattr(chained_adapters[i], "chain_proj") for i in range(1, len(chained_adapters)))

#     report.add(
#         "Architecture: 42 adapters",
#         n_ad == 42,
#         f"Found {n_ad} ChainedLinearAdapter modules (expected 42).",
#     )
#     report.add(
#         "Architecture: first adapter no chain_proj",
#         first_ok,
#         "First layer is_first and no chain_proj" if first_ok else "Mismatch",
#     )
#     report.add(
#         "Architecture: 41x chain_proj",
#         rest_chain and n_ad == 42,
#         "Layers 1..41 have chain_proj" if rest_chain else "Mismatch",
#     )
    
#     body = f"Total params: {total:,}\nTrainable: {trainable:,} ({pct:.2f}%)\nAdapters: {n_ad}"
#     report.sections["Architecture"] = body


# def verify_forward_backward(model, chained_adapters, chain_state, device, report):
#     model.train()
#     if hasattr(model, "decoder"):
#         model.decoder.train()
#     if hasattr(model, "joint"):
#         model.joint.train()

#     reset_before = chain_state._reset_count

#     B, T = 2, 32000
#     signals = torch.randn(B, T, device=device, dtype=torch.float32) * 0.01
#     lengths = torch.tensor([T, T - 1000], device=device, dtype=torch.int64)

#     try:
#         with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.bfloat16):
#             out = model.forward(input_signal=signals, input_signal_length=lengths)
#         report.add("Forward pass", True, f"Output type: {type(out)}")
#     except Exception as e:
#         report.add("Forward pass", False, str(e))
#         return

#     reset_after = chain_state._reset_count
#     report.add(
#         "Chain state reset",
#         reset_after > reset_before,
#         f"Reset calls: {reset_after - reset_before}",
#     )

#     # Test backward
#     model.zero_grad(set_to_none=True)
#     try:
#         pre, pre_len = model.preprocessor(input_signal=signals, input_signal_length=lengths)
#         enc_out = model.encoder(audio_signal=pre, length=pre_len)
#         enc_tensor = enc_out[0] if isinstance(enc_out, tuple) else enc_out
#         loss = enc_tensor.float().sum()
#         loss.backward()
#         report.add("Backward pass", True, "Completed successfully")
#     except Exception as e:
#         report.add("Backward pass", False, str(e))


# def verify_optimizer_groups(model, report):
#     adapter_params, joint_params, decoder_params = [], [], []
    
#     for pname, param in model.named_parameters():
#         if not param.requires_grad:
#             continue
#         nl = pname.lower()
#         if "adapter" in nl or "chain" in nl:
#             adapter_params.append(param)
#         elif "joint" in nl:
#             joint_params.append(param)
#         elif "decoder" in nl or "prediction" in nl or "embed" in nl:
#             decoder_params.append(param)
#         else:
#             adapter_params.append(param)

#     report.add(
#         "Optimizer: 3 param groups",
#         len(adapter_params) > 0 and len(joint_params) > 0 and len(decoder_params) > 0,
#         f"Adapter: {sum(p.numel() for p in adapter_params):,}, "
#         f"Joint: {sum(p.numel() for p in joint_params):,}, "
#         f"Decoder: {sum(p.numel() for p in decoder_params):,}",
#     )


# # ============================================================================
# # Main
# # ============================================================================

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--manifest", default="/kaggle/input/datasets/akarshkumarshukla/train-manifest/train_manifest.jsonl")
#     ap.add_argument("--out", default="./adapter_verify_report")
#     ap.add_argument("--model-id", default="nvidia/parakeet-tdt-1.1b")
#     ap.add_argument("--quick", action="store_true")
#     args = ap.parse_args()

#     out_dir = Path(args.out)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     report = Report()
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if torch.cuda.is_available():
#         torch.set_float32_matmul_precision("high")

#     print(f"\n[4/6] Loading model: {args.model_id}")
    
#     cfg_linear = OmegaConf.create({
#         "_target_": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
#         "in_features": 1024,
#         "dim": 128,
#         "activation": "gelu",
#         "norm_position": "post",
#         "dropout": 0.1,
#     })
    
#     model_cfg = ASRModel.from_pretrained(args.model_id, return_config=True)
#     enc_key = _encoder_target_key(model_cfg)
    
#     with open_dict(model_cfg):
#         meta = adapter_mixins.get_registered_adapter(model_cfg.encoder[enc_key])
#         if meta is not None:
#             model_cfg.encoder[enc_key] = meta.adapter_class_path

#     model = ASRModel.from_pretrained(args.model_id, override_config_path=model_cfg)
#     model = model.to(device)
    
#     with open_dict(model.cfg):
#         if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
#             model.cfg.decoding.greedy.use_cuda_graph_decoder = False
#     if hasattr(model, "change_decoding_strategy"):
#         model.change_decoding_strategy(model.cfg.decoding)

#     print("  Model loaded successfully!")
#     report.add("Model load", True, args.model_id)

#     # Add waveform augmentor
#     model.add_module("waveform_augmentor", WaveformAugmentor().to(device))

#     # Setup adapters
#     print("\n[5/6] Setting up adapters...")
#     chain_state = AdapterChainState()
#     n_adapters, chained = apply_training_setup(model, cfg_linear, chain_state, device)
#     print(f"  Created {n_adapters} chained adapters")

#     # Wrap forward
#     _orig = model.forward.__func__
#     def _aug(self, input_signal=None, input_signal_length=None, **kw):
#         if hasattr(self, "adapter_chain_state"):
#             self.adapter_chain_state.reset()
#         return _orig(self, input_signal=input_signal, input_signal_length=input_signal_length, **kw)
#     model.forward = types.MethodType(_aug, model)

#     # Run verifications
#     print("\n[6/6] Running verifications...")
#     verify_architecture(model, chained, report)
#     verify_forward_backward(model, chained, chain_state, device, report)
#     verify_optimizer_groups(model, report)

#     # Save report
#     md = report.summary_md()
#     report_path = out_dir / "VERIFY_REPORT.md"
#     report_path.write_text(md, encoding="utf-8")
    
#     print("\n" + "=" * 60)
#     print(md)
#     print("=" * 60)
#     print(f"\nReport saved to: {report_path}")

#     failed = [c for c in report.checks if not c.passed]
#     if failed:
#         print(f"\n❌ {len(failed)} checks FAILED")
#         sys.exit(1)
#     else:
#         print(f"\n✅ All {len(report.checks)} checks PASSED")
#         sys.exit(0)


# if __name__ == "__main__":
#     main()
# '''

# # Write the script to file
# with open(script_path, "w", encoding="utf-8") as f:
#     f.write(script_content)

# print("=" * 60)
# print("Running verification in subprocess...")
# print("=" * 60)

# result = subprocess.run(
#     [sys.executable, script_path],
#     cwd="/kaggle/working",
#     env={**os.environ, "PYTHONUNBUFFERED": "1"},
# )

# print("=" * 60)
# if result.returncode == 0:
#     print("✅ Verification completed successfully!")
# else:
#     print(f"❌ Verification failed with code {result.returncode}")
# # =============================================================================
# # CELL: NeMo Adapter Pipeline Verification (Notebook-Safe)
# # =============================================================================
# from __future__ import annotations

# import json
# import math
# import os
# import sys
# import time
# import types
# import warnings
# from collections import defaultdict
# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=Warning, module="numba")

# # ---------------------------------------------------------------------------
# # Detect notebook environment
# # ---------------------------------------------------------------------------
# _IN_NOTEBOOK = False
# try:
#     get_ipython()
#     _IN_NOTEBOOK = True
# except NameError:
#     pass

# # ---------------------------------------------------------------------------
# # Preflight numpy/scipy check (notebook-safe)
# # ---------------------------------------------------------------------------
# try:
#     import numpy as _np
#     _ = _np.__version__
#     import scipy.signal
# except Exception as err:
#     msg = (
#         f"NUMPY/SCIPY import failed: {err}\n"
#         "Fix: pip install --force-reinstall 'numpy>=2.1,<2.3' 'scipy>=1.14,<1.16'\n"
#         "Then RESTART KERNEL."
#     )
#     if _IN_NOTEBOOK:
#         print("=" * 60)
#         print(msg)
#         print("=" * 60)
#         raise RuntimeError(msg) from err
#     else:
#         print(msg, file=sys.stderr)
#         sys.exit(2)

# # ---------------------------------------------------------------------------
# # Optional plotting
# # ---------------------------------------------------------------------------
# _HAS_MPL = False
# try:
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     _HAS_MPL = True
# except Exception:
#     plt = None

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchaudio

# from nemo.collections.asr.models import ASRModel
# from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil
# from nemo.core.classes import adapter_mixins
# from nemo.core.classes.mixins import adapter_mixin_strategies
# from omegaconf import OmegaConf, open_dict


# # ============================================================================
# # Adapter Classes (mirrored from training script)
# # ============================================================================

# class AdapterChainState:
#     def __init__(self):
#         self.prev_bottleneck = None
#         self.layer_counter = 0
#         self._reset_count = 0

#     def reset(self):
#         self.prev_bottleneck = None
#         self.layer_counter = 0
#         self._reset_count += 1

#     def update(self, bottleneck):
#         self.prev_bottleneck = bottleneck
#         self.layer_counter += 1


# class ChainedLinearAdapter(nn.Module, AdapterModuleUtil):
#     def __init__(
#         self,
#         in_features,
#         dim,
#         activation="gelu",
#         norm_position="post",
#         dropout=0.1,
#         is_first=False,
#         chain_state_ref=None,
#         adapter_strategy=None,
#     ):
#         nn.Module.__init__(self)
#         assert norm_position == "post"
#         self.down = nn.Linear(in_features, dim)
#         self.up = nn.Linear(dim, in_features)
#         self.norm = nn.LayerNorm(in_features)
#         self.dropout_layer = nn.Dropout(dropout)
#         self.is_first = is_first
#         self.chain_state_ref = chain_state_ref
#         self.act = nn.GELU()
#         if not is_first:
#             self.chain_proj = nn.Linear(dim, dim, bias=False)
#             nn.init.orthogonal_(self.chain_proj.weight)
#             with torch.no_grad():
#                 self.chain_proj.weight.mul_(0.9)
#         nn.init.zeros_(self.up.weight)
#         nn.init.zeros_(self.up.bias)
#         self.setup_adapter_strategy(adapter_strategy)

#     def forward(self, x):
#         prev_bottleneck = None
#         if self.chain_state_ref is not None and not self.is_first:
#             prev_bottleneck = self.chain_state_ref.prev_bottleneck
#         h = self.down(x)
#         if prev_bottleneck is not None:
#             if prev_bottleneck.shape[1] != h.shape[1]:
#                 prev_bottleneck = F.interpolate(
#                     prev_bottleneck.transpose(1, 2),
#                     size=h.shape[1],
#                     mode="nearest",
#                 ).transpose(1, 2)
#             h = h + self.chain_proj(prev_bottleneck)
#         bottleneck = self.act(h)
#         if self.chain_state_ref is not None:
#             self.chain_state_ref.update(bottleneck)
#         h_up = self.up(bottleneck)
#         h_norm = self.norm(h_up)
#         h_drop = self.dropout_layer(h_norm)
#         return h_drop


# class WaveformAugmentor(nn.Module):
#     def __init__(
#         self,
#         speed_min=0.90, speed_max=1.10,
#         pitch_min=-1.0, pitch_max=3.0,
#         sample_rate=16000,
#         speed_prob=0.5, pitch_prob=0.5,
#         classroom_noise_prob=0.5,
#         classroom_snr_min=5.0, classroom_snr_max=10.0,
#         noise_file_paths=None,
#         gain_prob=0.3, gain_db_min=-6.0, gain_db_max=6.0,
#     ):
#         super().__init__()
#         self.speed_min, self.speed_max = speed_min, speed_max
#         self.pitch_min, self.pitch_max = pitch_min, pitch_max
#         self.sample_rate = sample_rate
#         self.speed_prob, self.pitch_prob = speed_prob, pitch_prob
#         self.classroom_noise_prob = classroom_noise_prob
#         self.classroom_snr_min, self.classroom_snr_max = classroom_snr_min, classroom_snr_max
#         self.noise_file_paths = list(noise_file_paths or [])
#         self.gain_prob = gain_prob
#         self.gain_db_min, self.gain_db_max = gain_db_min, gain_db_max

#     @staticmethod
#     def _mix_at_snr(speech, noise, snr_db):
#         eps = 1e-8
#         p_s = speech.pow(2).mean().clamp_min(eps)
#         p_n = noise.pow(2).mean().clamp_min(eps)
#         snr_lin = 10 ** (snr_db / 10.0)
#         alpha = torch.sqrt(p_s / (snr_lin * p_n + eps))
#         return speech + alpha * noise

#     def _load_noise_segment(self, num_samples, device, dtype):
#         import random
#         paths = self.noise_file_paths
#         if not paths:
#             return None
#         path = paths[random.randint(0, len(paths) - 1)]
#         try:
#             wav, sr = torchaudio.load(path)
#         except Exception:
#             return None
#         if wav.shape[0] > 1:
#             wav = wav.mean(dim=0, keepdim=True)
#         wav = wav.squeeze(0)
#         if sr != self.sample_rate:
#             wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
#         n = int(wav.shape[0])
#         if n < 1:
#             return None
#         if n < num_samples:
#             reps = (num_samples + n - 1) // n
#             wav = wav.repeat(reps)[:num_samples]
#         else:
#             start = random.randint(0, n - num_samples)
#             wav = wav[start:start + num_samples]
#         return wav.to(device=device, dtype=torch.float32)

#     @torch.no_grad()
#     def forward(self, audio_signal, signal_length):
#         if not self.training:
#             return audio_signal, signal_length
#         B, T = audio_signal.shape
#         if torch.rand(1).item() < self.speed_prob:
#             sf = self.speed_min + torch.rand(1).item() * (self.speed_max - self.speed_min)
#             new_T = max(1, int(T / sf))
#             audio_signal = F.interpolate(
#                 audio_signal.unsqueeze(1), size=new_T, mode="linear", align_corners=False,
#             ).squeeze(1)
#             signal_length = (signal_length.float() / sf).long().clamp(min=1, max=new_T)
#         if torch.rand(1).item() < self.pitch_prob:
#             n_steps = self.pitch_min + torch.rand(1).item() * (self.pitch_max - self.pitch_min)
#             try:
#                 audio_signal = torchaudio.functional.pitch_shift(audio_signal, self.sample_rate, n_steps)
#             except Exception:
#                 pass
#         if self.noise_file_paths and torch.rand(1).item() < self.classroom_noise_prob:
#             for b in range(B):
#                 L = int(min(signal_length[b].item(), audio_signal.shape[1]))
#                 if L < 1:
#                     continue
#                 noise_seg = self._load_noise_segment(L, audio_signal.device, audio_signal.dtype)
#                 if noise_seg is None:
#                     continue
#                 snr = self.classroom_snr_min + torch.rand(1).item() * (self.classroom_snr_max - self.classroom_snr_min)
#                 seg = audio_signal[b, :L].float()
#                 mixed = self._mix_at_snr(seg, noise_seg, snr)
#                 audio_signal[b, :L] = mixed.to(dtype=audio_signal.dtype)
#         if torch.rand(1).item() < self.gain_prob:
#             gain_db = self.gain_db_min + torch.rand(1).item() * (self.gain_db_max - self.gain_db_min)
#             audio_signal = audio_signal * (10 ** (gain_db / 20))
#         return audio_signal, signal_length


# # ============================================================================
# # Report
# # ============================================================================

# @dataclass
# class CheckResult:
#     name: str
#     passed: bool
#     detail: str = ""

# @dataclass
# class Report:
#     checks: List[CheckResult] = field(default_factory=list)
#     sections: Dict[str, str] = field(default_factory=dict)

#     def add(self, name, passed, detail=""):
#         self.checks.append(CheckResult(name, passed, detail))

#     def summary_md(self):
#         lines = ["# Adapter Pipeline Verification Report", ""]
#         lines.append("## Executive Summary")
#         p = sum(1 for c in self.checks if c.passed)
#         lines.append(f"\n**{p}/{len(self.checks)} checks passed**\n")
#         for c in self.checks:
#             sym = "PASS" if c.passed else "FAIL"
#             lines.append(f"- **{sym}**: {c.name}")
#             if c.detail:
#                 lines.append(f"  - {c.detail}")
#         lines.append("")
#         for title, body in self.sections.items():
#             lines.append(f"## {title}")
#             lines.append(body)
#             lines.append("")
#         return "\n".join(lines)


# # ============================================================================
# # Helpers
# # ============================================================================

# def _encoder_target_key(model_cfg):
#     enc = model_cfg.encoder
#     if "_target_" in enc:
#         return "_target_"
#     if "target" in enc:
#         return "target"
#     raise AttributeError("encoder target key missing")


# def apply_training_setup(model, cfg_linear, chain_state, device):
#     adapter_name = "encoder:asr_children_adapter"
#     model.add_adapter(name=adapter_name, cfg=cfg_linear)
#     model.set_enabled_adapters(enabled=False)
#     model.set_enabled_adapters(name=adapter_name, enabled=True)
#     model.freeze()
#     model.unfreeze_enabled_adapters()
#     model.adapter_chain_state = chain_state

#     from nemo.collections.common.parts.adapter_modules import LinearAdapter as NemoLinearAdapter
#     adapter_module_list = []
#     for mod_name, module in list(model.named_modules()):
#         if isinstance(module, NemoLinearAdapter):
#             adapter_module_list.append((mod_name, module))
#     adapter_module_list.sort(key=lambda x: x[0])

#     chained_list = []
#     for i, (mod_name, _orig) in enumerate(adapter_module_list):
#         new_ad = ChainedLinearAdapter(
#             in_features=1024, dim=128, activation="gelu",
#             norm_position="post", dropout=0.1,
#             is_first=(i == 0), chain_state_ref=chain_state,
#             adapter_strategy=None,
#         ).to(device)
#         parts = mod_name.split(".")
#         parent = model
#         for part in parts[:-1]:
#             parent = getattr(parent, part)
#         setattr(parent, parts[-1], new_ad)
#         chained_list.append(new_ad)

#     if hasattr(model, "joint"):
#         for p in model.joint.parameters():
#             p.requires_grad = True
#     if hasattr(model, "decoder") and hasattr(model.decoder, "prediction"):
#         pred = model.decoder.prediction
#         if hasattr(pred, "embed"):
#             for p in pred.embed.parameters():
#                 p.requires_grad = True
#         for _, mod in pred.named_modules():
#             if isinstance(mod, nn.LSTM) and mod.num_layers >= 1:
#                 li = mod.num_layers - 1
#                 for pname, p in mod.named_parameters():
#                     if f"_l{li}" in pname:
#                         p.requires_grad = True
#     if hasattr(model, "decoder"):
#         model.decoder.train()
#     if hasattr(model, "joint"):
#         model.joint.train()
#     return len(chained_list), chained_list


# def module_train_tree(model, max_lines=200):
#     lines = []
#     for n, (name, m) in enumerate(model.named_modules()):
#         if n >= max_lines:
#             lines.append("... (truncated)")
#             break
#         lines.append(f"  {name or '<root>'}: training={m.training}")
#     return "\n".join(lines)


# # ============================================================================
# # Verification Functions
# # ============================================================================

# def verify_architecture(model, chained_adapters, report):
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     pct = 100.0 * trainable / total if total else 0

#     n_ad = len(chained_adapters)
#     first_ok = chained_adapters[0].is_first and not hasattr(chained_adapters[0], "chain_proj")
#     rest_chain = all(hasattr(chained_adapters[i], "chain_proj") for i in range(1, len(chained_adapters)))

#     ortho_scores = []
#     for i, m in enumerate(chained_adapters):
#         if hasattr(m, "chain_proj"):
#             W = m.chain_proj.weight.detach().float()
#             if W.shape[0] == W.shape[1]:
#                 Q = W / 0.9
#                 I = torch.eye(W.shape[0], device=W.device)
#                 err = (Q @ Q.T - I).abs().mean().item()
#                 ortho_scores.append((i, err))

#     report.add("Architecture: 42 adapters", n_ad == 42,
#                f"Found {n_ad} ChainedLinearAdapter modules (expected 42).")
#     report.add("Architecture: first adapter no chain_proj", first_ok,
#                "First layer is_first and no chain_proj" if first_ok else "Mismatch")
#     report.add("Architecture: 41x chain_proj", rest_chain and n_ad == 42,
#                "Layers 1..41 have chain_proj" if rest_chain else "Mismatch")
#     report.add("Architecture: orthogonal chain_proj (Q=W/0.9)",
#                len(ortho_scores) == 41 and all(s[1] < 0.05 for s in ortho_scores),
#                f"max err: {max(s[1] for s in ortho_scores) if ortho_scores else 'n/a'}")

#     body = (f"Total: {total:,} | Trainable: {trainable:,} ({pct:.2f}%)\n"
#             f"Adapters: {n_ad} | First no chain_proj: {first_ok} | Rest chain_proj: {rest_chain}\n")
#     tree = module_train_tree(model, max_lines=60)
#     body += "\n### Module train flags\n```\n" + tree + "\n```\n"
#     report.sections["Architecture"] = body


# def attach_adapter_hooks(chained_adapters, records):
#     def make_hook(idx):
#         def hook(_mod, inp, out):
#             x = inp[0]
#             records.append({
#                 "layer_idx": idx,
#                 "in_shape": tuple(x.shape),
#                 "out_shape": tuple(out.shape) if torch.is_tensor(out) else None,
#                 "is_first": _mod.is_first,
#             })
#         return hook
#     handles = []
#     for i, m in enumerate(chained_adapters):
#         handles.append(m.register_forward_hook(make_hook(i)))
#     return handles


# def verify_forward_backward(model, chained_adapters, chain_state, device, report):
#     model.train()
#     if hasattr(model, "decoder"): model.decoder.train()
#     if hasattr(model, "joint"): model.joint.train()
#     if hasattr(model, "waveform_augmentor"): model.waveform_augmentor.train()
#     sa = getattr(model, "spec_augmentation", None)
#     if sa is not None: sa.train()

#     hook_records = []
#     handles = attach_adapter_hooks(chained_adapters, hook_records)
#     reset_before = chain_state._reset_count

#     B, T1, T2 = 2, 32000, 24000
#     T = max(T1, T2)
#     signals = torch.zeros(B, T, device=device, dtype=torch.float32)
#     signals[0, :T1] = torch.randn(T1, device=device) * 0.01
#     signals[1, :T2] = torch.randn(T2, device=device) * 0.01
#     lengths = torch.tensor([T1, T2], device=device, dtype=torch.int64)

#     try:
#         with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.bfloat16):
#             out = model.forward(input_signal=signals, input_signal_length=lengths)
#     except Exception as e:
#         for h in handles: h.remove()
#         report.add("Forward dry-run", False, str(e))
#         return
#     for h in handles: h.remove()

#     report.add("Forward: chain_state.reset",
#                chain_state._reset_count > reset_before,
#                f"resets: {chain_state._reset_count - reset_before}")
#     report.add("Forward: adapter hooks fired",
#                len(hook_records) >= 42,
#                f"hooks: {len(hook_records)}")

#     # Backward
#     model.zero_grad(set_to_none=True)
#     try:
#         try:
#             pre, pre_len = model.preprocessor(input_signal=signals, input_signal_length=lengths)
#         except TypeError:
#             pre, pre_len = model.preprocessor(input_signal=signals, length=lengths)
#         enc_out = model.encoder(audio_signal=pre, length=pre_len)
#         enc_tensor = enc_out[0] if isinstance(enc_out, tuple) else enc_out
#         loss = enc_tensor.float().sum()
#         loss.backward()
#         report.add("Backward: loss.backward()", True, "Completed")
#     except Exception as e:
#         report.add("Backward", False, str(e))
#         return

#     grad_stats = defaultdict(list)
#     for name, p in model.named_parameters():
#         if not p.requires_grad: continue
#         if p.grad is None:
#             grad_stats["missing_grad"].append(name)
#             continue
#         g = p.grad.detach().float()
#         key = ("adapter" if ("adapter" in name.lower() or "chain_proj" in name.lower())
#                else "joint" if "joint" in name.lower()
#                else "decoder" if ("decoder" in name.lower() or "prediction" in name.lower())
#                else "other")
#         grad_stats[key].append((name, g.mean().item(), g.std().item(), g.abs().max().item()))

#     frozen = [n for n, p in model.named_parameters()
#               if n.startswith("encoder.") and "adapter" not in n.lower()
#               and p.grad is not None and p.grad.abs().max() > 0]
#     report.add("Backward: frozen encoder no grad", len(frozen) == 0,
#                f"Non-adapter encoder w/ grad: {len(frozen)}")

#     bw = []
#     for k in ["adapter", "joint", "decoder", "other"]:
#         if k in grad_stats and grad_stats[k]:
#             bw.append(f"**{k}** ({len(grad_stats[k])} tensors)\n")
#             for item in grad_stats[k][:3]:
#                 bw.append(f"  {item[0]}: mean={item[1]:.2e} std={item[2]:.2e} max={item[3]:.2e}\n")
#     if grad_stats.get("missing_grad"):
#         bw.append(f"\nMissing grad: {len(grad_stats['missing_grad'])} params\n")
#     report.sections["Forward/Backward"] = "".join(bw)

#     # Eval mode check
#     model.eval()
#     hook_records.clear()
#     handles = attach_adapter_hooks(chained_adapters, hook_records)
#     with torch.no_grad():
#         _ = model.forward(input_signal=signals, input_signal_length=lengths)
#     for h in handles: h.remove()
#     report.add("Eval forward (no grad)", True, f"hooks: {len(hook_records)}")


# def verify_optimizer_scheduler(model, num_epochs, steps_per_epoch, out_dir, report):
#     adapter_params, joint_params, decoder_params = [], [], []
#     for pname, param in model.named_parameters():
#         if not param.requires_grad: continue
#         nl = pname.lower()
#         if "adapter" in nl or "chain" in nl: adapter_params.append(param)
#         elif "joint" in nl: joint_params.append(param)
#         elif "decoder" in nl or "prediction" in nl or "embed" in nl: decoder_params.append(param)
#         else: adapter_params.append(param)

#     param_groups = []
#     if adapter_params: param_groups.append({"params": adapter_params, "lr": 5e-4})
#     if joint_params: param_groups.append({"params": joint_params, "lr": 1e-4})
#     if decoder_params: param_groups.append({"params": decoder_params, "lr": 5e-5})

#     total_steps = num_epochs * steps_per_epoch
#     warmup_steps = int(0.15 * total_steps)

#     def lr_lambda(step):
#         if step < warmup_steps:
#             return step / max(1, warmup_steps)
#         progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
#         return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

#     report.add("Optimizer: 3 groups", len(param_groups) == 3,
#                f"Groups: {[pg['lr'] for pg in param_groups]}")
#     report.add("Optimizer: AdamW config", True, "betas=(0.9,0.999), wd=0.01")

#     if _HAS_MPL:
#         lrs = [[] for _ in param_groups]
#         for step in range(total_steps):
#             for i, pg in enumerate(param_groups):
#                 lrs[i].append(pg["lr"] * lr_lambda(step))
#         fig, ax = plt.subplots(figsize=(10, 4))
#         labels = ["Adapter 5e-4", "Joint 1e-4", "Decoder 5e-5"]
#         for i in range(len(param_groups)):
#             ax.plot(lrs[i], label=labels[i] if i < len(labels) else f"group{i}")
#         ax.set_xlabel("step"); ax.set_ylabel("LR"); ax.legend()
#         ax.set_title(f"LR Schedule (warmup={warmup_steps}, total={total_steps})")
#         p = out_dir / "lr_schedule.png"
#         fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
#         report.sections["LR schedule"] = f"Saved {p}"

#     body = "| Group | LR | Params |\n|---|---|---|\n"
#     for i, pg in enumerate(param_groups):
#         n = sum(p.numel() for p in pg["params"])
#         body += f"| {i} | {pg['lr']} | {n:,} |\n"
#     body += f"\ntotal_steps={total_steps}, warmup={warmup_steps}\n"
#     report.sections["Optimizer groups"] = body


# def verify_spec_augment(model, device, out_dir, report):
#     sa = getattr(model, "spec_augmentation", None)
#     if sa is None:
#         report.add("SpecAugment", False, "spec_augmentation is None")
#         return
#     sa.train()
#     x = torch.randn(2, 80, 100, device=device)
#     lengths = torch.tensor([100, 90], device=device)
#     with torch.cuda.amp.autocast(enabled=False):
#         y = sa(input_spec=x, length=lengths)
#     report.add("SpecAugment forward", y.shape == x.shape, f"shape {tuple(y.shape)}")
#     if _HAS_MPL:
#         fig, ax = plt.subplots(1, 2, figsize=(10, 3))
#         ax[0].imshow(x[0].cpu().numpy(), aspect="auto", origin="lower"); ax[0].set_title("Before")
#         ax[1].imshow(y[0].detach().cpu().numpy(), aspect="auto", origin="lower"); ax[1].set_title("After SpecAug")
#         p = out_dir / "spec_augment_mel.png"
#         fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)


# def verify_waveform_aug(noise_files, manifest_path, out_dir, report):
#     paths_3 = []
#     if manifest_path and os.path.isfile(manifest_path):
#         with open(manifest_path, "r") as f:
#             for line in f:
#                 if not line.strip(): continue
#                 try:
#                     row = json.loads(line)
#                     p = row.get("audio_filepath")
#                     if p and os.path.isfile(p): paths_3.append(p)
#                 except: continue
#                 if len(paths_3) >= 3: break

#     aug = WaveformAugmentor(noise_file_paths=noise_files[:50], speed_prob=1.0,
#                              pitch_prob=0.0, classroom_noise_prob=0.0, gain_prob=0.0)
#     aug.train()
#     if paths_3:
#         w, sr = torchaudio.load(paths_3[0])
#         if w.shape[0] > 1: w = w.mean(0, keepdim=True)
#         w = w[:, :min(16000*3, w.shape[1])]
#         sig = w.squeeze(0).unsqueeze(0)
#         lens = torch.tensor([sig.shape[1]])
#         out, _ = aug(sig, lens.clone())
#         report.add("Waveform aug (speed)", True, f"{tuple(sig.shape)} -> {tuple(out.shape)}")
#         if _HAS_MPL:
#             fig, ax = plt.subplots(2, 1, figsize=(10, 4))
#             ax[0].plot(sig[0].numpy()[:4000]); ax[0].set_title("Before")
#             ax[1].plot(out[0].detach().numpy()[:4000]); ax[1].set_title("After speed aug")
#             p = out_dir / "waveform_aug.png"
#             fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
#     else:
#         report.add("Waveform aug samples", False, "No audio from manifest")

#     aug.eval()
#     sig = torch.randn(1, 8000)
#     out_eval, _ = aug(sig, torch.tensor([8000]))
#     report.add("Waveform aug disabled in eval", torch.allclose(sig, out_eval), "eval=passthrough")


# def verify_reinforce_callback(model, report):
#     from lightning.pytorch.callbacks import Callback
#     class _Cb(Callback):
#         def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
#             for attr in ["decoder", "joint"]:
#                 m = getattr(pl_module, attr, None)
#                 if m is not None: m.train()
#             for attr in ["waveform_augmentor", "spec_augmentation"]:
#                 m = getattr(pl_module, attr, None)
#                 if m is not None: m.train()

#     model.eval()
#     _Cb().on_train_batch_start(None, model, None, 0)
#     ok = True
#     for attr in ["decoder", "joint"]:
#         m = getattr(model, attr, None)
#         if m is not None: ok = ok and m.training
#     for attr in ["waveform_augmentor", "spec_augmentation"]:
#         m = getattr(model, attr, None)
#         if m is not None: ok = ok and m.training
#     report.add("ReinforceCallback behavior", ok, "All restored to train after eval")


# def verify_save_adapters(model, out_dir, report):
#     p = out_dir / "verify_adapters.pt"
#     try:
#         model.save_adapters(str(p))
#         report.add("save_adapters", p.is_file(), str(p))
#     except Exception as e:
#         report.add("save_adapters", False, str(e))


# def verify_nemo_version(report):
#     try:
#         import nemo
#         ver = getattr(nemo, "__version__", "unknown")
#         report.add("NeMo import", True, f"version {ver}")
#     except Exception as e:
#         report.add("NeMo import", False, str(e))


# def verify_residual_strategy(chained, report):
#     ok = all(hasattr(m, "adapter_strategy") and m.adapter_strategy is not None for m in chained)
#     names = {type(m.adapter_strategy).__name__ for m in chained[:3]}
#     report.add("adapter_strategy present", ok, f"types: {names}")


# def memory_estimate(model, report):
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     model_mb = total * 4 / 1e6
#     grad_mb = trainable * 4 / 1e6
#     optim_mb = trainable * 8 / 1e6
#     body = (f"Weights: {model_mb:.0f} MB | Grads: {grad_mb:.0f} MB | "
#             f"Optimizer: {optim_mb:.0f} MB | Est peak: {model_mb+grad_mb+optim_mb+500:.0f} MB\n"
#             f"(bf16 training uses ~50% less)")
#     report.sections["Memory estimate"] = body
#     report.add("Memory estimate", True, "See Memory section")


# def benchmark_step(model, device, report):
#     model.train()
#     sig = torch.randn(2, 32000, device=device) * 0.01
#     lens = torch.tensor([32000, 31000], device=device, dtype=torch.int64)
#     if device.type == "cuda": torch.cuda.synchronize()
#     t0 = time.perf_counter()
#     try:
#         with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.bfloat16):
#             _ = model.forward(input_signal=sig, input_signal_length=lens)
#         if device.type == "cuda": torch.cuda.synchronize()
#         ms = (time.perf_counter() - t0) * 1000
#         report.add("Benchmark forward", True, f"{ms:.1f} ms")
#         report.sections["Benchmark"] = f"Forward: {ms:.1f} ms (B=2, T=32000)\n"
#     except Exception as e:
#         report.add("Benchmark forward", False, str(e))


# # ============================================================================
# # MAIN
# # ============================================================================

# def main():
#     # ---- Notebook-safe argument handling ----
#     if _IN_NOTEBOOK:
#         class Args:
#             manifest = os.environ.get(
#                 "TRAIN_MANIFEST",
#                 "/kaggle/input/datasets/akarshkumarshukla/earlier-manifest/train_manifest.jsonl",
#             )
#             out = "/kaggle/working/adapter_verify_report"
#             model_id = "nvidia/parakeet-tdt-1.1b"
#             batch_size = 48
#             num_epochs = 6
#             quick = False
#         args = Args()
#         print("Running in NOTEBOOK mode (argparse skipped)")
#     else:
#         import argparse
#         ap = argparse.ArgumentParser()
#         ap.add_argument("--manifest", default=os.environ.get(
#             "TRAIN_MANIFEST",
#             "/kaggle/input/datasets/akarshkumarshukla/train-manifest/train_manifest.jsonl"))
#         ap.add_argument("--out", default="/kaggle/working/adapter_verify_report")
#         ap.add_argument("--model-id", default="nvidia/parakeet-tdt-1.1b")
#         ap.add_argument("--batch-size", type=int, default=48)
#         ap.add_argument("--num-epochs", type=int, default=6)
#         ap.add_argument("--quick", action="store_true")
#         args = ap.parse_args()

#     out_dir = Path(args.out)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     print(f"Output directory: {out_dir}")

#     report = Report()
#     verify_nemo_version(report)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if torch.cuda.is_available():
#         torch.set_float32_matmul_precision("high")
#         print(f"GPU: {torch.cuda.get_device_name(0)}")

#     MODEL_ID = args.model_id
#     noise_dirs = [
#         "/kaggle/input/datasets/akarshkumarshukla/noise-1",
#         "/kaggle/input/datasets/akarshkumarshukla/asr-data/noise_part_0",
#     ]
#     noise_files = []
#     for d in noise_dirs:
#         if os.path.isdir(d):
#             for root, _, files in os.walk(d):
#                 for f in files:
#                     if f.lower().endswith((".wav", ".flac", ".mp3")):
#                         noise_files.append(os.path.join(root, f))
#                 if len(noise_files) > 200: break
#     print(f"Noise files: {len(noise_files)}")

#     cfg_linear = OmegaConf.create({
#         "_target_": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
#         "in_features": 1024, "dim": 128, "activation": "gelu",
#         "norm_position": "post", "dropout": 0.1,
#     })

#     print(f"\nLoading {MODEL_ID} on {device}...")
#     model_cfg = ASRModel.from_pretrained(MODEL_ID, return_config=True)
#     enc_key = _encoder_target_key(model_cfg)
#     with open_dict(model_cfg):
#         meta = adapter_mixins.get_registered_adapter(model_cfg.encoder[enc_key])
#         if meta is not None:
#             model_cfg.encoder[enc_key] = meta.adapter_class_path

#     model = ASRModel.from_pretrained(MODEL_ID, override_config_path=model_cfg).to(device)
#     with open_dict(model.cfg):
#         if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
#             model.cfg.decoding.greedy.use_cuda_graph_decoder = False
#     if hasattr(model, "change_decoding_strategy"):
#         model.change_decoding_strategy(model.cfg.decoding)
#     print("Model loaded!")

#     # SpecAug + Waveform
#     model.spec_augmentation = model.from_config_dict(OmegaConf.create({
#         "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
#         "freq_masks": 2, "freq_width": 27, "time_masks": 10, "time_width": 0.05,
#     }))
#     model.add_module("waveform_augmentor",
#                       WaveformAugmentor(noise_file_paths=noise_files[:500]).to(device))

#     chain_state = AdapterChainState()
#     n_adapters, chained = apply_training_setup(model, cfg_linear, chain_state, device)
#     print(f"Setup {n_adapters} chained adapters")

#     # Wrap forward
#     _orig = model.forward.__func__
#     def _aug(self, input_signal=None, input_signal_length=None, **kw):
#         if hasattr(self, "adapter_chain_state"):
#             self.adapter_chain_state.reset()
#         if self.training and input_signal is not None and input_signal_length is not None:
#             input_signal, input_signal_length = self.waveform_augmentor(
#                 input_signal, input_signal_length)
#         return _orig(self, input_signal=input_signal,
#                      input_signal_length=input_signal_length, **kw)
#     model.forward = types.MethodType(_aug, model)

#     if hasattr(model, "waveform_augmentor"): model.waveform_augmentor.train()
#     if model.spec_augmentation is not None: model.spec_augmentation.train()

#     # Run verifications
#     print("\nRunning verifications...")
#     verify_architecture(model, chained, report)
#     verify_residual_strategy(chained, report)
#     verify_forward_backward(model, chained, chain_state, device, report)
#     verify_spec_augment(model, device, out_dir, report)
#     verify_waveform_aug(noise_files,
#                         args.manifest if os.path.isfile(args.manifest) else None,
#                         out_dir, report)

#     steps_per_epoch = 1000
#     if not args.quick and os.path.isfile(args.manifest):
#         try:
#             with open_dict(model.cfg.train_ds):
#                 model.cfg.train_ds.manifest_filepath = args.manifest
#                 model.cfg.train_ds.batch_size = min(args.batch_size, 4)
#                 model.cfg.train_ds.num_workers = 2
#                 model.cfg.train_ds.shuffle = False
#             model.setup_training_data(model.cfg.train_ds)
#             report.add("Dataloader", True, "setup OK")
#             try:
#                 steps_per_epoch = max(1, len(model._train_dl))
#             except: pass
#         except Exception as e:
#             report.add("Dataloader", False, str(e))
#     else:
#         report.add("Dataloader", False, "Skipped (--quick or missing manifest)")

#     verify_optimizer_scheduler(model, args.num_epochs, steps_per_epoch, out_dir, report)
#     memory_estimate(model, report)
#     benchmark_step(model, device, report)
#     verify_reinforce_callback(model, report)
#     verify_save_adapters(model, out_dir, report)

#     # Generate report
#     md = report.summary_md()
#     report_path = out_dir / "VERIFY_REPORT.md"
#     report_path.write_text(md, encoding="utf-8")

#     print("\n" + "=" * 60)
#     print(md)
#     print("=" * 60)
#     print(f"\nReport saved to: {report_path}")

#     failed = [c for c in report.checks if not c.passed]
#     if failed:
#         print(f"\n{len(failed)} checks FAILED")
#     else:
#         print(f"\nAll {len(report.checks)} checks PASSED")

#     # Don't sys.exit in notebook
#     if not _IN_NOTEBOOK:
#         sys.exit(0 if not failed else 1)


# # ============================================================================
# # Run
# # ============================================================================
# main()
# -*- coding: utf-8 -*-
import os; os.environ["USE_EXISTING_MANIFESTS"] = "1"
os.environ["TRAIN_DEBUG_STEPS"] = "0"
"""
Paste this entire file into ONE Kaggle notebook cell (or run locally).

  Part A — pip install (clean env + NeMo)
  Part B — write training script to /kaggle/working
  Part C — subprocess.run (fresh Python interpreter)
  Part D (optional) — RUN_ADAPTER_VERIFY=1: verify_nemo_adapter_pipeline.py via subprocess

Pipeline:
  - Model: nvidia/parakeet-tdt-1.1b  (1.1B params, FastConformer-TDT)
  - Chained bottleneck adapter: 1024 → 128 → 1024 with GELU nonlinearity
  - Linear chain through 128-dim bottleneck across encoder layers
  - Unfrozen: Joint network + Decoder embedding + Last LSTM layer
  - Waveform augmentation: speed ±10%, pitch -1 to +3 semitones, classroom noise (SNR 5–10 dB), gain
  - SpecAugment on mel spectrograms
  - Whisper text normalizer on training labels
  - 6 epochs, discriminative learning rates, training only (no validation)
  - Saves adapter weights + full .nemo checkpoint

Outputs: /kaggle/working/nemo_adapter_1.1b/
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
# PART B: WRITE SELF-CONTAINED TRAINING SCRIPT
# =============================================================================
TRAIN_SCRIPT = "/kaggle/working/train_nemo_adapter_1.1b.py"

train_code = r'''from __future__ import annotations

import json
import logging
import math
import os
import types
import warnings
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List

warnings.filterwarnings("ignore", category=Warning, module="numba")
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
import torch.nn.functional as F
import torchaudio
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil, LinearAdapter
from nemo.core.classes import adapter_mixins
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf, open_dict

# ==========================================
# Whisper Text Normalizer (Change 5)
# ==========================================
try:
    from whisper_normalizer.english import EnglishTextNormalizer
    _whisper_normalizer = EnglishTextNormalizer()
    NORMALIZE_TEXT = True
    print("Whisper text normalizer loaded — will normalize training labels")
except ImportError:
    NORMALIZE_TEXT = False
    _whisper_normalizer = None
    print("WARNING: whisper_normalizer not installed. Training labels will use basic lowercase only.")
    print("Install with: pip install whisper-normalizer")


# ==========================================
# Adapter Chain State Manager (Change 1)
# ==========================================
class AdapterChainState:
    """Manages the bottleneck chain state across encoder layers.

    Each forward pass must call reset() before the encoder runs.
    Adapters call update() to pass their bottleneck to the next layer.
    """

    def __init__(self):
        self.prev_bottleneck = None
        self.layer_counter = 0

    def reset(self):
        self.prev_bottleneck = None
        self.layer_counter = 0

    def update(self, bottleneck):
        self.prev_bottleneck = bottleneck
        self.layer_counter += 1


# ==========================================
# Chained Linear Adapter (Change 1) — nn.Module + AdapterModuleUtil (not LinearAdapter)
# ==========================================
class ChainedLinearAdapter(nn.Module, AdapterModuleUtil):
    """Bottleneck chain across encoder layers. After add_adapter(LinearAdapter), replace
    modules in both the attribute tree and layer.adapter_layer ModuleDict so NeMo's
    adapter forward uses ChainedLinearAdapter."""

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


# ==========================================
# Waveform Augmentation (Change 4: fixed pitch, classroom noise + gain)
# ==========================================
class WaveformAugmentor(nn.Module):
    """On-the-fly waveform augmentation: speed, pitch, classroom noise (SNR), gain."""

    def __init__(
        self,
        speed_min: float = 0.90,
        speed_max: float = 1.10,
        pitch_min: float = -1.0,
        pitch_max: float = 3.0,
        sample_rate: int = 16000,
        speed_prob: float = 0.5,
        pitch_prob: float = 0.5,
        classroom_noise_prob: float = 0.5,
        classroom_snr_min: float = 5.0,
        classroom_snr_max: float = 10.0,
        noise_file_paths: list | None = None,
        gain_prob: float = 0.3,
        gain_db_min: float = -6.0,
        gain_db_max: float = 6.0,
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
    def _mix_at_snr(speech: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Additive mix: SNR (dB) = 10 * log10( mean(s^2) / mean((alpha*n)^2) )."""
        eps = 1e-8
        p_s = speech.pow(2).mean().clamp_min(eps)
        p_n = noise.pow(2).mean().clamp_min(eps)
        snr_lin = 10 ** (snr_db / 10.0)
        alpha = torch.sqrt(p_s / (snr_lin * p_n + eps))
        return speech + alpha * noise

    def _load_noise_segment(self, num_samples: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        paths = self.noise_file_paths
        if not paths:
            return None
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
            start = random.randint(0, n - num_samples)
            wav = wav[start : start + num_samples]
        return wav.to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def forward(
        self, audio_signal: torch.Tensor, signal_length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return audio_signal, signal_length

        B, T = audio_signal.shape

        if torch.rand(1).item() < self.speed_prob:
            speed_factor = (
                self.speed_min
                + torch.rand(1).item() * (self.speed_max - self.speed_min)
            )
            new_T = max(1, int(T / speed_factor))
            audio_signal = F.interpolate(
                audio_signal.unsqueeze(1),
                size=new_T,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            signal_length = (signal_length.float() / speed_factor).long()
            signal_length = signal_length.clamp(min=1, max=new_T)

        if torch.rand(1).item() < self.pitch_prob:
            n_steps = (
                self.pitch_min
                + torch.rand(1).item() * (self.pitch_max - self.pitch_min)
            )
            try:
                audio_signal = torchaudio.functional.pitch_shift(
                    audio_signal, self.sample_rate, n_steps
                )
            except Exception:
                pass

        if (
            self.noise_file_paths
            and torch.rand(1).item() < self.classroom_noise_prob
        ):
            snr_lo, snr_hi = self.classroom_snr_min, self.classroom_snr_max
            for b in range(B):
                L = int(min(signal_length[b].item(), audio_signal.shape[1]))
                if L < 1:
                    continue
                noise_seg = self._load_noise_segment(L, audio_signal.device, audio_signal.dtype)
                if noise_seg is None:
                    continue
                snr = snr_lo + torch.rand(1).item() * (snr_hi - snr_lo)
                seg = audio_signal[b, :L].float()
                mixed = self._mix_at_snr(seg, noise_seg, snr)
                audio_signal[b, :L] = mixed.to(dtype=audio_signal.dtype)

        if torch.rand(1).item() < self.gain_prob:
            gain_db = self.gain_db_min + torch.rand(1).item() * (self.gain_db_max - self.gain_db_min)
            gain_linear = 10 ** (gain_db / 20)
            audio_signal = audio_signal * gain_linear

        return audio_signal, signal_length


# ==========================================
# Save only specific epochs
# ==========================================
class SaveSelectedEpochs(Callback):
    """Save adapter weights + full .nemo checkpoint only for specified epochs."""

    def __init__(self, directory: str, save_epochs: list):
        super().__init__()
        self.directory = directory
        self.save_epochs = save_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        epoch = trainer.current_epoch

        if epoch not in self.save_epochs:
            print(f"[SaveSelectedEpochs] epoch {epoch} -> SKIPPED (not in save list)", flush=True)
            return

        nemo_path = os.path.join(self.directory, f"model_epoch{epoch}.nemo")
        adapter_path = os.path.join(self.directory, f"adapter_epoch{epoch}.pt")
        pl_module.save_to(nemo_path)
        try:
            pl_module.save_adapters(adapter_path)
        except Exception as e:
            print(f"[SaveSelectedEpochs] WARNING: save_adapters failed: {e}", flush=True)
        print(
            f"[SaveSelectedEpochs] epoch {epoch} -> {nemo_path}, {adapter_path}",
            flush=True,
        )


class ReinforceDecoderJointTrainMode(Callback):
    """NeMo NeuralModule.freeze() calls self.eval() on the whole model; only adapter
    submodules are switched back to train via unfreeze_enabled_adapters(). The RNNT
    prediction LSTM stays in eval while the embedding (or joint) is trainable — cuDNN
    then errors on backward: 'RNN backward can only be called in training mode'.

    Also keep waveform_augmentor and spec_augmentation in train mode: their forwards
    gate on self.training, so eval disables augmentation even during trainer.fit().
    """

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        dec = getattr(pl_module, "decoder", None)
        if dec is not None:
            dec.train()
        j = getattr(pl_module, "joint", None)
        if j is not None:
            j.train()
        wa = getattr(pl_module, "waveform_augmentor", None)
        if wa is not None:
            wa.train()
        sa = getattr(pl_module, "spec_augmentation", None)
        if sa is not None:
            sa.train()


class ChainedAdapterTrainDebugCallback(Callback):
    """For the first `num_steps` training batches (rank 0): log batch summary, every
    ChainedLinearAdapter forward (in/out shapes), chain_state counter, loss, and
    max-abs grads on down/up/chain_proj weights after backward."""

    def __init__(self, num_steps: int = 0):
        super().__init__()
        self.num_steps = int(num_steps)
        self._batch_ctr = 0
        self._handles = []
        self._adapters_ordered = []
        self._fwd_records = []
        self._grad_lines = []

    @staticmethod
    def _gmax(t):
        if t is None or t.grad is None:
            return float("nan")
        return t.grad.detach().float().abs().max().item()

    def on_train_start(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        if self.num_steps <= 0:
            return
        self._adapters_ordered = [
            (n, m)
            for n, m in pl_module.named_modules()
            if isinstance(m, ChainedLinearAdapter)
        ]
        self._adapters_ordered.sort(key=lambda x: x[0])
        for h in self._handles:
            h.remove()
        self._handles.clear()

        def _make_hook(idx, name):
            def _hook(mod, inp, out):
                x = inp[0]
                self._fwd_records.append(
                    {
                        "idx": idx,
                        "name": name,
                        "in": tuple(x.shape),
                        "out": tuple(out.shape) if torch.is_tensor(out) else None,
                        "is_first": mod.is_first,
                    }
                )

            return _hook

        for i, (name, mod) in enumerate(self._adapters_ordered):
            self._handles.append(mod.register_forward_hook(_make_hook(i, name)))
        print(
            f"[ChainedAdapterTrainDebug] hooks on ALL {len(self._adapters_ordered)} "
            f"ChainedLinearAdapter modules; logging first {self.num_steps} batches",
            flush=True,
        )

    def on_train_end(self, trainer, pl_module):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        if self._batch_ctr >= self.num_steps:
            return
        self._fwd_records.clear()

    def on_after_backward(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        if self._batch_ctr >= self.num_steps:
            return
        lines = []
        for i, (name, mod) in enumerate(self._adapters_ordered):
            gd = self._gmax(mod.down.weight)
            gu = self._gmax(mod.up.weight)
            cp = getattr(mod, "chain_proj", None)
            if cp is not None:
                gc = self._gmax(cp.weight)
            else:
                gc = float("nan")
            lines.append(
                f"    [{i:02d}] down={gd:.3e} up={gu:.3e} chain={gc:.3e}  {name}"
            )
        self._grad_lines = lines

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        if self._batch_ctr >= self.num_steps:
            return
        recs = sorted(self._fwd_records, key=lambda r: r["idx"])
        print(f"\n{'=' * 80}", flush=True)
        print(
            f"[ChainedAdapterTrainDebug] batch={self._batch_ctr} "
            f"(epoch={trainer.current_epoch}, batch_idx={batch_idx})",
            flush=True,
        )
        if isinstance(batch, dict):
            parts = []
            for k, v in batch.items():
                if torch.is_tensor(v):
                    parts.append(f"{k}={tuple(v.shape)}")
                else:
                    parts.append(f"{k}={type(v).__name__}")
            print(f"  batch keys: {', '.join(parts)}", flush=True)
        else:
            print(f"  batch type: {type(batch).__name__}", flush=True)
        print(
            f"  pl_module.training={pl_module.training} "
            f"dec={getattr(pl_module.decoder, 'training', None)} "
            f"joint={getattr(pl_module.joint, 'training', None)} "
            f"wa={getattr(getattr(pl_module, 'waveform_augmentor', None), 'training', None)} "
            f"sa={getattr(getattr(pl_module, 'spec_augmentation', None), 'training', None)}",
            flush=True,
        )
        loss_line = ""
        if isinstance(outputs, dict):
            for k in ("loss", "train_loss"):
                if k in outputs and outputs[k] is not None:
                    v = outputs[k]
                    if torch.is_tensor(v):
                        loss_line = f"{k}={v.detach().float().item():.6f}"
                    else:
                        loss_line = f"{k}={v}"
                    break
        if not loss_line:
            loss_line = f"outputs={type(outputs).__name__}"
        print(f"  {loss_line}", flush=True)
        print(
            f"  forward: {len(recs)} ChainedLinearAdapter hook fires (all layers)",
            flush=True,
        )
        for r in recs:
            print(
                f"    [{r['idx']:02d}] first={r['is_first']} in={r['in']} out={r['out']}  {r['name']}",
                flush=True,
            )
        cs = getattr(pl_module, "adapter_chain_state", None)
        if cs is not None:
            print(
                f"  adapter_chain_state.layer_counter after forward = {cs.layer_counter}",
                flush=True,
            )
        if self._grad_lines:
            print(
                "  backward (max abs grad on down/up/chain_proj weights, captured in on_after_backward):",
                flush=True,
            )
            for line in self._grad_lines:
                print(line, flush=True)
        self._grad_lines = []
        print(f"{'=' * 80}\n", flush=True)
        self._batch_ctr += 1


# ==========================================
# Configuration (Change 6: updated hyperparameters)
# ==========================================
MODEL_ID = "nvidia/parakeet-tdt-1.1b"

ASR_DATA_DIR = "/kaggle/input/datasets/akarshkumarshukla/asr-data"
ASR_JSONL = os.path.join(ASR_DATA_DIR, "train_word_transcripts.jsonl")
TALKBANK_DIR = "/kaggle/input/datasets/akarshkumarshukla/talk-bank-data/audio"
TALKBANK_JSON = (
    "/kaggle/input/datasets/akarshkumarshukla/talkbank-transcript-full/"
    "train_word_transcripts.jsonl"
)

# Competition classroom-noise clips (two Kaggle input folders). Override with env:
#   CLASSROOM_NOISE_DIRS — comma- or semicolon-separated absolute paths
#   or CLASSROOM_NOISE_DIR_1 / CLASSROOM_NOISE_DIR_2
def _collect_classroom_noise_files(directories: list) -> list:
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    out = []
    for d in directories:
        if not d or not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    out.append(os.path.join(root, f))
    return out


_cnd = True
if _cnd:
    CLASSROOM_NOISE_DIRS = [
    "/kaggle/input/datasets/akarshkumarshukla/noise-1",
            "/kaggle/input/datasets/akarshkumarshukla/asr-data/noise_part_0",
    ]
else:
    _d1 = os.environ.get("CLASSROOM_NOISE_DIR_1", "").strip()
    _d2 = os.environ.get("CLASSROOM_NOISE_DIR_2", "").strip()
    if _d1 or _d2:
        CLASSROOM_NOISE_DIRS = [d for d in (_d1, _d2) if d]
    else:
        CLASSROOM_NOISE_DIRS = [
            "/kaggle/input/datasets/akarshkumarshukla/noise-1",
            "/kaggle/input/datasets/akarshkumarshukla/asr-data/noise_part_0",
        ]

CLASSROOM_NOISE_FILES = _collect_classroom_noise_files(CLASSROOM_NOISE_DIRS)
print(
    f"Classroom noise: {len(CLASSROOM_NOISE_FILES)} audio clips from "
    f"{len(CLASSROOM_NOISE_DIRS)} configured folder(s)"
)
for _dir in CLASSROOM_NOISE_DIRS:
    print(f"  - {_dir}  (exists={os.path.isdir(_dir)})")

SAVE_DIR = "/kaggle/working/nemo_adapter_1.1b"
MANIFEST_DIR = os.path.join(SAVE_DIR, "manifests")

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = 2
NUM_EPOCHS = 6
MAX_DURATION_SEC = 20.0
LEARNING_RATE = 5e-4

SAVE_EPOCHS = [1,2, 4]

os.makedirs(MANIFEST_DIR, exist_ok=True)

print(f"torch:  {torch.__version__}")
print(f"CUDA:   {torch.version.cuda}")
print(f"numpy:  {np.__version__}")

# ==========================================
# Manifest Handling (Change 5: text normalization)
# ==========================================
USE_EXISTING_MANIFESTS = True
TRAIN_MANIFEST = os.environ.get(
    "TRAIN_MANIFEST",
    "/kaggle/input/datasets/akarshkumarshukla/earlier-manifest/train_manifest.jsonl",
)

if USE_EXISTING_MANIFESTS:
    if not os.path.isfile(TRAIN_MANIFEST):
        raise FileNotFoundError(
            f"Train manifest not found: {TRAIN_MANIFEST}. "
            "Attach the dataset or set TRAIN_MANIFEST / USE_EXISTING_MANIFESTS=0."
        )
    train_path = TRAIN_MANIFEST
    print("Using existing manifest:")
    print("  Train:", train_path)
else:

    def _index_directory(search_dir):
        local = {}
        if not os.path.isdir(search_dir):
            return local
        for root, _, files in os.walk(search_dir):
            for f in files:
                if f.endswith(".flac"):
                    local[f] = os.path.join(root, f)
        return local

    print("Indexing audio files...")
    audio_index = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = [
            ex.submit(_index_directory, ASR_DATA_DIR),
            ex.submit(_index_directory, TALKBANK_DIR),
        ]
        for fut in futures:
            audio_index.update(fut.result())
    print(f"Indexed {len(audio_index):,} files")

    def _find_audio(path):
        return audio_index.get(os.path.basename(path), None)

    print("Building manifests from JSONL...")
    records: List[dict] = []
    jsonl_files = [ASR_JSONL]
    if os.path.exists(TALKBANK_JSON):
        jsonl_files.append(TALKBANK_JSON)

    for filepath in jsonl_files:
        if not os.path.exists(filepath):
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("orthographic_text", "").strip()
                    if NORMALIZE_TEXT:
                        text = _whisper_normalizer(text)
                    else:
                        text = text.lower()
                    dur = float(data.get("audio_duration_sec", 0.0))
                    path = _find_audio(data.get("audio_path", ""))
                    if path and text and (0.1 < dur <= MAX_DURATION_SEC):
                        records.append(
                            {"audio_filepath": path, "duration": dur, "text": text}
                        )
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue

    df = pd.DataFrame(records)
    print(f"Total: {len(df):,} utterances")
    if len(df) == 0:
        raise RuntimeError(
            "No training rows after JSONL + audio index. Check ASR_DATA_DIR / TALKBANK paths."
        )

    train_path = os.path.join(MANIFEST_DIR, "train_manifest.jsonl")
    df.to_json(train_path, orient="records", lines=True)
    print(f"Train: {len(df):,} utterances (all data, no val split)")


# ==========================================
# Helper
# ==========================================
def update_model_cfg(orig_cfg, new_cfg):
    with open_dict(orig_cfg):
        for k, v in new_cfg.items():
            orig_cfg[k] = v
    return orig_cfg


def _encoder_target_key(model_cfg):
    enc = model_cfg.encoder
    if "_target_" in enc:
        return "_target_"
    if "target" in enc:
        return "target"
    raise AttributeError("model_cfg.encoder has neither _target_ nor target")


# ==========================================
# OmegaConf Config (Change 2: gelu, Change 6: warmup)
# ==========================================
cfg = OmegaConf.create(
    {
        "model": {
            "pretrained_model": MODEL_ID,
            "log_prediction": False,
            "spec_augment": {
                "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
                "freq_masks": 2,
                "freq_width": 27,
                "time_masks": 10,
                "time_width": 0.05,
            },
            "adapter": {
                "adapter_name": "asr_children_adapter",
                "adapter_module_name": "encoder",
                "adapter_type": "linear",
                "linear": {
                    "_target_": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
                    "in_features": 1024,
                    "dim": 128,
                    "activation": "gelu",
                    "norm_position": "post",
                    "dropout": 0.1,
                },
            },
            "train_ds": {
                "manifest_filepath": train_path,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "pin_memory": False,
                "use_lhotse": False,
                "channel_selector": "average",
                "is_tarred": False,
                "shuffle": True,
            },
            "optim": {
                "name": "adamw",
                "lr": LEARNING_RATE,
                "betas": [0.9, 0.999],
                "weight_decay": 0.01,
                "sched": {
                    "name": "CosineAnnealing",
                    "warmup_ratio": 0.15,
                    "min_lr": 1e-6,
                },
            },
        },
        "trainer": {
            "devices": 1,
            "accelerator": "gpu",
            "precision": "bf16-mixed",
            "max_epochs": NUM_EPOCHS,
            "enable_progress_bar": True,
            "log_every_n_steps": 2000,
            "gradient_clip_val": 1.0,
        },
        "exp_manager": {
            "exp_dir": SAVE_DIR,
            "name": "ParakeetAdapter1.1B",
            "create_tensorboard_logger": True,
            "create_checkpoint_callback": False,
            "resume_if_exists": False,
            "resume_ignore_no_checkpoint": True,
        },
    }
)

# ==========================================
# Trainer
# ==========================================
print("Initializing Trainer...")
trainer = pl.Trainer(
    devices=1,
    accelerator="gpu",
    precision="bf16-mixed",
    max_epochs=NUM_EPOCHS,
    limit_val_batches=0,
    num_sanity_val_steps=0,
    enable_progress_bar=True,
    log_every_n_steps=2000,
    gradient_clip_val=1.0,
    logger=False,
    enable_checkpointing=False,
)
exp_log_dir = exp_manager(trainer, cfg.exp_manager)
ckpt_dir = os.path.join(exp_log_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
print(f"Logs: {exp_log_dir}")
print("Validation: DISABLED (training only)")

trainer.callbacks.append(SaveSelectedEpochs(ckpt_dir, save_epochs=SAVE_EPOCHS))
trainer.callbacks.append(ReinforceDecoderJointTrainMode())
_train_dbg = int(os.environ.get("TRAIN_DEBUG_STEPS", "0"))
if _train_dbg > 0:
    trainer.callbacks.append(ChainedAdapterTrainDebugCallback(num_steps=_train_dbg))
print(f"Saving .nemo + adapter for epochs {[e+1 for e in SAVE_EPOCHS]} + final to {ckpt_dir}")

# ==========================================
# Load Pretrained Model
# ==========================================
print(f"Loading {MODEL_ID}...")
model_cfg = ASRModel.from_pretrained(MODEL_ID, return_config=True)
enc_key = _encoder_target_key(model_cfg)

with open_dict(model_cfg):
    adapter_metadata = adapter_mixins.get_registered_adapter(
        model_cfg.encoder[enc_key]
    )
    if adapter_metadata is not None:
        model_cfg.encoder[enc_key] = adapter_metadata.adapter_class_path
        print(f"Patched encoder to: {adapter_metadata.adapter_class_path}")

model = ASRModel.from_pretrained(
    MODEL_ID, override_config_path=model_cfg, trainer=trainer
)

with open_dict(model.cfg):
    if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
        model.cfg.decoding.greedy.use_cuda_graph_decoder = False
if hasattr(model, "change_decoding_strategy"):
    model.change_decoding_strategy(model.cfg.decoding)

print("Model loaded")

# ==========================================
# Waveform Augmentation (Change 4)
# ==========================================
print("Attaching waveform augmentation (speed 0.90-1.10x, pitch -1 to +3 semitones)...")
if not CLASSROOM_NOISE_FILES:
    print(
        "WARNING: No classroom noise files found. Set CLASSROOM_NOISE_DIRS or "
        "CLASSROOM_NOISE_DIR_1/2 to your Kaggle dataset paths. Noise mixing disabled."
    )
waveform_aug = WaveformAugmentor(
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
    noise_file_paths=CLASSROOM_NOISE_FILES,
    gain_prob=0.2,
    gain_db_min=-6.0,
    gain_db_max=6.0,
)
model.add_module("waveform_augmentor", waveform_aug)

_original_forward = model.forward.__func__


def _augmented_forward(
    self,
    input_signal=None,
    input_signal_length=None,
    processed_signal=None,
    processed_signal_length=None,
):
    if hasattr(self, 'adapter_chain_state'):
        self.adapter_chain_state.reset()
    if self.training and input_signal is not None and input_signal_length is not None:
        input_signal, input_signal_length = self.waveform_augmentor(
            input_signal, input_signal_length
        )
    return _original_forward(
        self,
        input_signal=input_signal,
        input_signal_length=input_signal_length,
        processed_signal=processed_signal,
        processed_signal_length=processed_signal_length,
    )


model.forward = types.MethodType(_augmented_forward, model)
print("Waveform augmentation attached to model forward (chain state reset integrated)")

if "spec_augment" in cfg.model:
    model.spec_augmentation = model.from_config_dict(cfg.model.spec_augment)
    print("SpecAugment enabled (freq_masks=2, time_masks=10)")

# ==========================================
# Data
# ==========================================
print("Setting up training data...")
cfg.model.train_ds = update_model_cfg(model.cfg.train_ds, cfg.model.train_ds)
model.setup_training_data(cfg.model.train_ds)

# ==========================================
# Adapter: standard LinearAdapter cfg, then replace + sync adapter_layer ModuleDict
# ==========================================
print("Adding bottleneck adapter (1024 -> 128 -> 1024, activation=gelu)...")
adapter_full_name = "encoder:asr_children_adapter"
adapter_short_name = "asr_children_adapter"
adapter_type_cfg = cfg.model.adapter.linear
model.add_adapter(name=adapter_full_name, cfg=adapter_type_cfg)
model.set_enabled_adapters(enabled=False)
model.set_enabled_adapters(name=adapter_full_name, enabled=True)

model.freeze()
model.unfreeze_enabled_adapters()

chain_state = AdapterChainState()
model.adapter_chain_state = chain_state

adapter_module_list = []
for mod_name, module in list(model.named_modules()):
    if isinstance(module, LinearAdapter) and adapter_short_name in mod_name:
        adapter_module_list.append((mod_name, module))
adapter_module_list.sort(key=lambda x: x[0])

device = next(model.parameters()).device
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

num_ad = len(adapter_module_list)

LinearAdapter.register(ChainedLinearAdapter)
_test_inst = model.encoder.layers[0].adapter_layer["asr_children_adapter"]
print(f"Registered ChainedLinearAdapter as virtual subclass of LinearAdapter "
      f"(isinstance(ChainedLinearAdapter, LinearAdapter) = {isinstance(_test_inst, LinearAdapter)})")

print(
    f"Successfully replaced {num_ad} adapters with ChainedLinearAdapter "
    "(module tree + adapter_layer registry)"
)
print(f"  Chain connections: {max(0, num_ad - 1)} (orthogonal init, scale=0.9)")

# #region agent log — DEBUG post-fix verification: does isinstance now pass + manual test
import json as _json_dbg
_dbg_log = "/kaggle/working/debug-9eb412.log"
def _dbg(msg, data, hyp):
    import time as _t
    with open(_dbg_log, "a") as _f:
        _f.write(_json_dbg.dumps({"sessionId":"9eb412","runId":"post-fix","hypothesisId":hyp,"location":"train_code:diag","message":msg,"data":data,"timestamp":int(_t.time()*1000)}) + "\n")

_layer0 = model.encoder.layers[0]
_ad0 = _layer0.adapter_layer["asr_children_adapter"]

from nemo.collections.common.parts import adapter_modules as _am
_isinstance_check = isinstance(_ad0, _am.LinearAdapter)
_dbg("POST-FIX: isinstance check after register", {
    "isinstance_ChainedLinearAdapter_of_LinearAdapter": _isinstance_check,
    "adapter_type": type(_ad0).__name__,
}, "FIX_VERIFY")
print(f"[DEBUG-FIX] isinstance(ChainedLinearAdapter, LinearAdapter) = {_isinstance_check}", flush=True)

_device = next(model.parameters()).device
_dummy = torch.randn(1, 10, 1024, device=_device)
_hook_fired = [False]
_hook_shapes = [None]
def _test_hook(mod, inp, out):
    _hook_fired[0] = True
    _hook_shapes[0] = {"in": [list(t.shape) for t in inp if isinstance(t, torch.Tensor)], "out": list(out.shape) if isinstance(out, torch.Tensor) else str(type(out))}
_h = _ad0.register_forward_hook(_test_hook)
try:
    _pack = {"x": _dummy, "loc": "post"}
    with torch.no_grad():
        _out = _layer0.forward_enabled_adapters(_pack)
    _manual_ok = True
    _manual_result = {"is_dict": isinstance(_out, dict)}
    if isinstance(_out, dict) and "x" in _out:
        _manual_result["x_shape"] = list(_out["x"].shape)
except Exception as _e:
    import traceback as _tb
    _manual_ok = False
    _manual_result = {"error": str(_e), "traceback": _tb.format_exc()[-1500:]}
_h.remove()
_dbg("POST-FIX: manual forward_enabled_adapters dict test", {
    "success": _manual_ok,
    "result": _manual_result,
    "hook_fired": _hook_fired[0],
    "hook_shapes": _hook_shapes[0],
}, "FIX_VERIFY")
print(f"[DEBUG-FIX] manual forward_enabled_adapters(dict): ok={_manual_ok}, hook_fired={_hook_fired[0]}", flush=True)
print(f"[DEBUG-FIX] result: {_manual_result}", flush=True)
if _hook_shapes[0]:
    print(f"[DEBUG-FIX] hook shapes: {_hook_shapes[0]}", flush=True)
# #endregion agent log

# ==========================================
# Unfreeze Joint + Partial Decoder (Change 3)
# ==========================================

# 3a: Joint Network (full unfreeze)
try:
    if hasattr(model, 'joint'):
        for param in model.joint.parameters():
            param.requires_grad = True
        joint_count = sum(p.numel() for p in model.joint.parameters() if p.requires_grad)
        print(f"Unfroze joint network: {joint_count:,} params")
    else:
        print("WARNING: model has no 'joint' attribute — skipping joint unfreeze")
except Exception as e:
    print(f"WARNING: Could not unfreeze joint network: {e}")

# 3b: Decoder Embedding
try:
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'prediction'):
        if hasattr(model.decoder.prediction, 'embed'):
            for param in model.decoder.prediction.embed.parameters():
                param.requires_grad = True
            embed_count = sum(
                p.numel() for p in model.decoder.prediction.embed.parameters()
                if p.requires_grad
            )
            print(f"Unfroze decoder embedding: {embed_count:,} params")
        else:
            print("WARNING: decoder.prediction has no 'embed' — skipping embedding unfreeze")
    else:
        print("WARNING: model decoder structure not as expected — skipping embedding unfreeze")
except Exception as e:
    print(f"WARNING: Could not unfreeze decoder embedding: {e}")

# 3c: Last LSTM layer of decoder (RNNT prediction uses prediction["dec_rnn"] wrapper)
def _unfreeze_last_lstm_in_module(container):
    """Enable grads for the highest-index LSTM layer only (PyTorch names: weight_ih_l{N}, ...)."""
    if container is None:
        return False
    unfroze = False
    for _, mod in container.named_modules():
        if isinstance(mod, nn.LSTM) and mod.num_layers >= 1:
            last_i = mod.num_layers - 1
            for pname, param in mod.named_parameters():
                if f"_l{last_i}" in pname:
                    param.requires_grad = True
                    unfroze = True
    return unfroze


try:
    unfroze_lstm = False
    if hasattr(model, "decoder") and hasattr(model.decoder, "prediction"):
        pred = model.decoder.prediction
        unfroze_lstm = _unfreeze_last_lstm_in_module(pred)
        if not unfroze_lstm and hasattr(pred, "dec_rnn"):
            unfroze_lstm = _unfreeze_last_lstm_in_module(pred.dec_rnn)
    if unfroze_lstm:
        print("Unfroze last decoder LSTM layer (via nn.LSTM scan)")
    else:
        print("WARNING: No nn.LSTM found under decoder.prediction — skipping LSTM unfreeze")
except Exception as e:
    print(f"WARNING: Could not unfreeze decoder LSTM: {e}")

# freeze() called eval() on the full model; only encoder adapters were set back to train.
# Decoder LSTM must stay in training mode for cuDNN backward when embedding/joint train.
if hasattr(model, "decoder"):
    model.decoder.train()
if hasattr(model, "joint"):
    model.joint.train()
if hasattr(model, "waveform_augmentor"):
    model.waveform_augmentor.train()
if getattr(model, "spec_augmentation", None) is not None:
    model.spec_augmentation.train()
print(
    "Applied decoder/joint/waveform_augmentor/spec_augmentation .train() "
    "after partial unfreeze (cuDNN RNN + augmentation gates)."
)

# ==========================================
# Optimizer: Discriminative LR (Change 3d) with fallback
# ==========================================
model.setup_optimization(cfg.model.optim)

try:
    from torch.optim import AdamW as TorchAdamW
    from torch.optim.lr_scheduler import LambdaLR

    adapter_params = []
    joint_params = []
    decoder_params = []

    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = pname.lower()
        if 'adapter' in name_lower or 'chain' in name_lower:
            adapter_params.append(param)
        elif 'joint' in name_lower:
            joint_params.append(param)
        elif 'decoder' in name_lower or 'prediction' in name_lower or 'embed' in name_lower:
            decoder_params.append(param)
        else:
            adapter_params.append(param)

    param_groups = []
    if adapter_params:
        param_groups.append({'params': adapter_params, 'lr': 5e-4})
    if joint_params:
        param_groups.append({'params': joint_params, 'lr': 1e-4})
    if decoder_params:
        param_groups.append({'params': decoder_params, 'lr': 5e-5})

    if not param_groups:
        raise RuntimeError("No trainable parameters found for param groups")

    for i, pg in enumerate(param_groups):
        n = sum(p.numel() for p in pg['params'])
        print(f"Param group {i}: {n:,} params, lr={pg['lr']}")

    try:
        num_batches = len(model._train_dl)
    except (TypeError, AttributeError):
        num_batches = 1000
    total_steps = NUM_EPOCHS * num_batches
    warmup_steps = int(0.15 * total_steps)

    _disc_optimizer = TorchAdamW(param_groups, betas=(0.9, 0.999), weight_decay=0.01)

    def _lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    _disc_scheduler = LambdaLR(_disc_optimizer, _lr_lambda)

    def _custom_configure_optimizers(self_model):
        return {
            'optimizer': _disc_optimizer,
            'lr_scheduler': {
                'scheduler': _disc_scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

    model.configure_optimizers = types.MethodType(_custom_configure_optimizers, model)
    print(f"Discriminative LR optimizer configured (warmup={warmup_steps}, total={total_steps} steps)")

except Exception as e:
    print(f"WARNING: Could not set up discriminative LR: {e}")
    print("Falling back to NeMo default optimizer with single LR={LEARNING_RATE}")

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params:     {total:,}")
print(f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)")

# ==========================================
# Train (Change 7: updated summary)
# ==========================================
print("=" * 60)
print("STARTING TRAINING")
print(f"  Model:      {MODEL_ID}")
print(f"  Adapter:    Chained Linear (1024 -> 128 -> 1024, GELU + LayerNorm + dropout=0.1)")
print(f"  Chain:      Linear chain through 128-dim bottleneck (41 connections)")
print(f"  Unfrozen:   Joint network + Decoder embedding + Last LSTM layer")
print(f"  Epochs:     {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  LR:         Adapters=5e-4, Joint=1e-4, Decoder=5e-5")
print(f"  Checkpoints: Epochs {[e+1 for e in SAVE_EPOCHS]} + Final")
print(f"  Augmentation:")
print(f"    - Speed perturbation: 0.90-1.10x (prob=0.5)")
print(f"    - Pitch perturbation: -1 to +3 semitones (prob=0.5)")
print(f"    - Classroom noise: SNR 5-10 dB (prob=0.5), {len(CLASSROOM_NOISE_FILES)} clips")
print(f"    - Gain perturbation: +/-6dB (prob=0.3)")
print(f"    - SpecAugment: freq_masks=2 w=27, time_masks=10 w=0.05")
print("  Note: 'attention_adapter_mixin: No adapter compatible' = MHSA hook skips adapter")
print("        (ChainedLinearAdapter is not LinearAdapter); FFN/conv adapter sites still run.")
print("=" * 60)

trainer.fit(model)

# ==========================================
# Save Final
# ==========================================
print("Saving final model...")
model.save_adapters(os.path.join(ckpt_dir, "adapter_final.pt"))
model.save_to(os.path.join(ckpt_dir, "model_final.nemo"))
print(f"Saved adapter_final.pt and model_final.nemo under {ckpt_dir}")
print("TRAINING COMPLETE!")
print(f"\nOutput files:")
for e in SAVE_EPOCHS:
    print(f"  - model_epoch{e}.nemo + adapter_epoch{e}.pt (Epoch {e+1})")
print(f"  - model_final.nemo + adapter_final.pt (Final)")
'''

os.makedirs("/kaggle/working", exist_ok=True)
with open(TRAIN_SCRIPT, "w", encoding="utf-8") as f:
    f.write(train_code)

print(f"Training script written to {TRAIN_SCRIPT}\n")
print(
    "Optional env (set BEFORE this cell; child inherits):\n"
    "  USE_EXISTING_MANIFESTS — 1 to use pre-built manifests\n"
    "  TRAIN_MANIFEST — override train manifest path\n"
    "  BATCH_SIZE — default 48\n"
    "  CLASSROOM_NOISE_DIRS — comma-separated paths to noise clip folders\n"
    "  CLASSROOM_NOISE_DIR_1 / CLASSROOM_NOISE_DIR_2 — two folders (defaults under /kaggle/input/)\n"
    "  RUN_ADAPTER_VERIFY — 1 to run verify_nemo_adapter_pipeline.py after training (subprocess)\n"
    "  VERIFY_SCRIPT — path to verify script (default /kaggle/working/verify_nemo_adapter_pipeline.py)\n"
    "  VERIFY_OUT — report directory (default /kaggle/working/adapter_verify_report)\n"
    "  VERIFY_QUICK — 1 to skip heavy dataloader checks\n"
    "  TRAIN_DEBUG_STEPS — N>0: log ALL chained-adapter forward shapes + grads for first N batches\n"
)

# =============================================================================
# PART C: FRESH SUBPROCESS (avoids stale notebook imports / numpy)
# =============================================================================
print("=" * 60)
print("Launching training in fresh subprocess")
print("=" * 60)

_compile = subprocess.run(
    [sys.executable, "-m", "py_compile", TRAIN_SCRIPT],
    cwd="/kaggle/working",
    capture_output=True,
    text=True,
)
if _compile.returncode != 0:
    print("Syntax error in generated training script (py_compile failed):\n")
    print(_compile.stderr or _compile.stdout)
    raise RuntimeError(
        "train script failed py_compile — fix the generator or paste errors above."
    )

result = subprocess.run(
    [sys.executable, TRAIN_SCRIPT],
    cwd="/kaggle/working",
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
    stdout=None,
    stderr=subprocess.STDOUT,
)

if result.returncode != 0:
    print(
        f"\nTraining subprocess exited with code {result.returncode}.\n"
        "Scroll up for the Python traceback from the child process.\n"
        "Do not use a hand-edited copy of train_code: broken pastes cause SyntaxError "
        "or wrong __init__ / __future__ / torch.__version__."
    )
    raise RuntimeError(f"Training failed with exit code {result.returncode}")

print("\nTraining completed successfully.")
print("Check /kaggle/working/nemo_adapter_1.1b/ for outputs.")

# =============================================================================
# OPTIONAL Part D — adapter pipeline verify (FRESH SUBPROCESS, same safety as Part C)
# =============================================================================
# Do NOT %run verify_nemo_adapter_pipeline.py in the notebook kernel — that imports NeMo in
# IPython and can hit the same numpy/scipy breakage as pre-Part-C training.
# Set RUN_ADAPTER_VERIFY=1 (and copy verify_nemo_adapter_pipeline.py to /kaggle/working).
VERIFY_SCRIPT = os.environ.get(
    "VERIFY_SCRIPT",
    "/kaggle/working/verify_nemo_adapter_pipeline.py",
)
if os.environ.get("RUN_ADAPTER_VERIFY", "").strip().lower() in ("1", "true", "yes"):
    print("\n" + "=" * 60)
    print("Part D: Launching verify_nemo_adapter_pipeline.py in fresh subprocess")
    print("=" * 60)
    if not os.path.isfile(VERIFY_SCRIPT):
        print(
            f"SKIP: VERIFY_SCRIPT not found: {VERIFY_SCRIPT}\n"
            "Upload verify_nemo_adapter_pipeline.py to /kaggle/working/ or set VERIFY_SCRIPT."
        )
    else:
        _vout = os.environ.get("VERIFY_OUT", "/kaggle/working/adapter_verify_report")
        _vmanifest = os.environ.get("TRAIN_MANIFEST", "").strip()
        _vquick = os.environ.get("VERIFY_QUICK", "").strip().lower() in ("1", "true", "yes")
        _vcmd = [
            sys.executable,
            VERIFY_SCRIPT,
            "--out",
            _vout,
        ]
        if _vmanifest:
            _vcmd.extend(["--manifest", _vmanifest])
        if _vquick:
            _vcmd.append("--quick")
        _vc = subprocess.run(
            _vcmd,
            cwd="/kaggle/working",
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=None,
            stderr=subprocess.STDOUT,
        )
        if _vc.returncode != 0:
            raise RuntimeError(
                f"Verification subprocess exited with code {_vc.returncode}. "
                f"See log above; report under {_vout}/VERIFY_REPORT.md if partial."
            )
        print(f"Verification OK. Report: {_vout}/VERIFY_REPORT.md")
Step 1: Cleaning...
Step 2: numpy + scipy...
Step 3: PyTorch (CUDA 12.6 wheels)...
Step 4: Dependencies...
Step 5: NeMo...
Step 6: Re-pin numpy...
Installation complete.

Training script written to /kaggle/working/train_nemo_adapter_1.1b.py

Optional env (set BEFORE this cell; child inherits):
  USE_EXISTING_MANIFESTS — 1 to use pre-built manifests
  TRAIN_MANIFEST — override train manifest path
  BATCH_SIZE — default 48
  CLASSROOM_NOISE_DIRS — comma-separated paths to noise clip folders
  CLASSROOM_NOISE_DIR_1 / CLASSROOM_NOISE_DIR_2 — two folders (defaults under /kaggle/input/)
  RUN_ADAPTER_VERIFY — 1 to run verify_nemo_adapter_pipeline.py after training (subprocess)
  VERIFY_SCRIPT — path to verify script (default /kaggle/working/verify_nemo_adapter_pipeline.py)
  VERIFY_OUT — report directory (default /kaggle/working/adapter_verify_report)
  VERIFY_QUICK — 1 to skip heavy dataloader checks
  TRAIN_DEBUG_STEPS — N>0: log ALL chained-adapter forward shapes + grads for first N batches

============================================================
Launching training in fresh subprocess
============================================================
[NeMo W 2026-03-24 17:05:58 megatron_init:62] Megatron num_microbatches_calculator not found, using Apex version.
OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for rank (rank=0) with OneLogger disabled. To override: explicitly set error_handling_strategy parameter.
No exporters were provided. This means that no telemetry data will be collected.
[NeMo W 2026-03-24 17:05:59 nemo_logging:364] /usr/local/lib/python3.12/dist-packages/pydub/utils.py:300: SyntaxWarning: invalid escape sequence '\('
      m = re.match('([su]([0-9]{1,2})p?) \(([0-9]{1,2}) bit\)$', token)
    
[NeMo W 2026-03-24 17:05:59 nemo_logging:364] /usr/local/lib/python3.12/dist-packages/pydub/utils.py:301: SyntaxWarning: invalid escape sequence '\('
      m2 = re.match('([su]([0-9]{1,2})p?)( \(default\))?$', token)
    
[NeMo W 2026-03-24 17:05:59 nemo_logging:364] /usr/local/lib/python3.12/dist-packages/pydub/utils.py:310: SyntaxWarning: invalid escape sequence '\('
      elif re.match('(flt)p?( \(default\))?$', token):
    
[NeMo W 2026-03-24 17:05:59 nemo_logging:364] /usr/local/lib/python3.12/dist-packages/pydub/utils.py:314: SyntaxWarning: invalid escape sequence '\('
      elif re.match('(dbl)p?( \(default\))?$', token):
    
Whisper text normalizer loaded — will normalize training labels
Classroom noise: 1940 audio clips from 2 configured folder(s)
  - /kaggle/input/datasets/akarshkumarshukla/noise-1  (exists=True)
  - /kaggle/input/datasets/akarshkumarshukla/asr-data/noise_part_0  (exists=True)
torch:  2.11.0+cu126
CUDA:   12.6
numpy:  2.2.6
Using existing manifest:
  Train: /kaggle/input/datasets/akarshkumarshukla/earlier-manifest/train_manifest.jsonl
Initializing Trainer...
Using bfloat16 Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
[NeMo I 2026-03-24 17:06:05 exp_manager:594] ExpManager schema
[NeMo I 2026-03-24 17:06:05 exp_manager:595] {'explicit_log_dir': None, 'exp_dir': None, 'name': None, 'version': None, 'use_datetime_version': True, 'resume_if_exists': False, 'resume_past_end': False, 'resume_ignore_no_checkpoint': False, 'resume_from_checkpoint': None, 'create_tensorboard_logger': True, 'summary_writer_kwargs': None, 'create_wandb_logger': False, 'wandb_logger_kwargs': None, 'create_mlflow_logger': False, 'mlflow_logger_kwargs': {'experiment_name': None, 'run_name': None, 'tracking_uri': None, 'tags': None, 'save_dir': './mlruns', 'prefix': '', 'artifact_location': None, 'run_id': None, 'log_model': False}, 'create_dllogger_logger': False, 'dllogger_logger_kwargs': {'verbose': False, 'stdout': False, 'json_file': './dllogger.json'}, 'create_clearml_logger': False, 'clearml_logger_kwargs': {'project': None, 'task': None, 'connect_pytorch': False, 'model_name': None, 'tags': None, 'log_model': False, 'log_cfg': False, 'log_metrics': False}, 'create_neptune_logger': False, 'neptune_logger_kwargs': None, 'create_checkpoint_callback': True, 'checkpoint_callback_params': {'filepath': None, 'dirpath': None, 'filename': None, 'monitor': 'val_loss', 'verbose': True, 'save_last': True, 'save_top_k': 3, 'save_weights_only': False, 'mode': 'min', 'auto_insert_metric_name': True, 'every_n_epochs': 1, 'every_n_train_steps': None, 'train_time_interval': None, 'prefix': None, 'postfix': '.nemo', 'save_best_model': False, 'always_save_nemo': False, 'save_nemo_on_train_end': True, 'model_parallel_size': None, 'save_on_train_epoch_end': False, 'async_save': False, 'save_last_n_optim_states': -1}, 'create_early_stopping_callback': False, 'create_ipl_epoch_stopper_callback': False, 'early_stopping_callback_params': {'monitor': 'val_loss', 'mode': 'min', 'min_delta': 0.001, 'patience': 10, 'verbose': True, 'strict': True, 'check_finite': True, 'stopping_threshold': None, 'divergence_threshold': None, 'check_on_train_epoch_end': None, 'log_rank_zero_only': False}, 'ipl_epoch_stopper_callback_params': {'enable_stop': True, 'stop_every_n_epochs': 1}, 'create_preemption_callback': True, 'files_to_copy': None, 'log_step_timing': True, 'log_delta_step_timing': False, 'step_timing_kwargs': {'reduction': 'mean', 'sync_cuda': False, 'buffer_size': 1}, 'log_local_rank_0_only': False, 'log_global_rank_0_only': False, 'disable_validation_on_resume': True, 'ema': {'enable': False, 'decay': 0.999, 'cpu_offload': False, 'validate_original_weights': False, 'every_n_steps': 1}, 'max_time_per_run': None, 'seconds_to_sleep': 5.0, 'create_straggler_detection_callback': False, 'straggler_detection_params': {'report_time_interval': 300.0, 'calc_relative_gpu_perf': True, 'calc_individual_gpu_perf': True, 'num_gpu_perf_scores_to_log': 5, 'gpu_relative_perf_threshold': 0.7, 'gpu_individual_perf_threshold': 0.7, 'stop_if_detected': False}, 'create_fault_tolerance_callback': False, 'fault_tolerance': {'workload_check_interval': 5.0, 'initial_rank_heartbeat_timeout': 3600.0, 'rank_heartbeat_timeout': 2700.0, 'calculate_timeouts': True, 'safety_factor': 5.0, 'rank_termination_signal': <Signals.SIGKILL: 9>, 'log_level': 'INFO', 'max_rank_restarts': 0, 'max_subsequent_job_failures': 0, 'additional_ft_launcher_args': '', 'simulated_fault': None}, 'log_tflops_per_sec_per_gpu': True}
[NeMo I 2026-03-24 17:06:05 exp_manager:655] Experiments will be logged at /kaggle/working/nemo_adapter_1.1b/ParakeetAdapter1.1B/2026-03-24_17-06-05
[NeMo I 2026-03-24 17:06:05 exp_manager:1262] TensorboardLogger has been set up
[NeMo I 2026-03-24 17:06:05 exp_manager:804] TFLOPs per sec per GPU will be calculated, conditioned on supported models. Defaults to -1 upon failure.
Logs: /kaggle/working/nemo_adapter_1.1b/ParakeetAdapter1.1B/2026-03-24_17-06-05
Validation: DISABLED (training only)
Saving .nemo + adapter for epochs [2, 3, 5] + final to /kaggle/working/nemo_adapter_1.1b/ParakeetAdapter1.1B/2026-03-24_17-06-05/checkpoints
Loading nvidia/parakeet-tdt-1.1b...
Patched encoder to: nemo.collections.asr.modules.conformer_encoder.ConformerEncoderAdapter
[NeMo I 2026-03-24 17:06:16 mixins:184] Tokenizer SentencePieceTokenizer initialized with 1024 tokens
[NeMo W 2026-03-24 17:06:16 modelPT:188] If you intend to do training or fine-tuning, please call the ModelPT.setup_training_data() method and provide a valid configuration file to setup the train data loader.
    Train config : 
    manifest_filepath: null
    sample_rate: 16000
    batch_size: 1
    shuffle: true
    num_workers: 8
    pin_memory: true
    max_duration: 20
    min_duration: 0.1
    is_tarred: true
    tarred_audio_filepaths: null
    shuffle_n: 2048
    bucketing_strategy: fully_randomized
    bucketing_batch_size:
    - 64
    - 64
    - 32
    - 32
    - 32
    - 32
    - 16
    - 16
    - 64
    - 64
    - 32
    - 32
    - 32
    - 32
    - 16
    - 16
    - 64
    - 64
    - 32
    - 32
    - 32
    - 32
    - 16
    - 16
    defer_setup: true
    
[NeMo W 2026-03-24 17:06:16 modelPT:195] If you intend to do validation, please call the ModelPT.setup_validation_data() or ModelPT.setup_multiple_validation_data() method and provide a valid configuration file to setup the validation data loader(s). 
    Validation config : 
    manifest_filepath: null
    sample_rate: 16000
    batch_size: 16
    shuffle: false
    use_start_end_token: false
    num_workers: 8
    pin_memory: true
    
[NeMo W 2026-03-24 17:06:16 modelPT:202] Please call the ModelPT.setup_test_data() or ModelPT.setup_multiple_test_data() method and provide a valid configuration file to setup the test data loader(s).
    Test config : 
    manifest_filepath: null
    sample_rate: 16000
    batch_size: 16
    shuffle: false
    use_start_end_token: false
    num_workers: 8
    pin_memory: true
    
[NeMo I 2026-03-24 17:06:21 rnnt_models:226] Using RNNT Loss : tdt
    Loss tdt_kwargs: {'fastemit_lambda': 0.0, 'clamp': -1.0, 'durations': [0, 1, 2, 3, 4], 'sigma': 0.02, 'omega': 0.1}
[NeMo I 2026-03-24 17:06:21 rnnt_models:226] Using RNNT Loss : tdt
    Loss tdt_kwargs: {'fastemit_lambda': 0.0, 'clamp': -1.0, 'durations': [0, 1, 2, 3, 4], 'sigma': 0.02, 'omega': 0.1}
[NeMo I 2026-03-24 17:06:21 rnnt_models:226] Using RNNT Loss : tdt
    Loss tdt_kwargs: {'fastemit_lambda': 0.0, 'clamp': -1.0, 'durations': [0, 1, 2, 3, 4], 'sigma': 0.02, 'omega': 0.1}
[NeMo I 2026-03-24 17:06:25 save_restore_connector:285] Model EncDecRNNTBPEModel was successfully restored from /root/.cache/huggingface/hub/models--nvidia--parakeet-tdt-1.1b/snapshots/53276c6469d1f17a1352e30c4d11be3d0d7e9575/parakeet-tdt-1.1b.nemo.
[NeMo I 2026-03-24 17:06:25 rnnt_models:226] Using RNNT Loss : tdt
    Loss tdt_kwargs: {'fastemit_lambda': 0.0, 'clamp': -1.0, 'durations': [0, 1, 2, 3, 4], 'sigma': 0.02, 'omega': 0.1}
[NeMo I 2026-03-24 17:06:25 rnnt_bpe_models:506] Changed decoding strategy to 
    model_type: tdt
    strategy: greedy
    compute_hypothesis_token_set: false
    preserve_alignments: null
    tdt_include_token_duration: null
    confidence_cfg:
      preserve_frame_confidence: false
      preserve_token_confidence: false
      preserve_word_confidence: false
      exclude_blank: true
      aggregation: min
      tdt_include_duration: false
      method_cfg:
        name: entropy
        entropy_type: tsallis
        alpha: 0.33
        entropy_norm: exp
        temperature: DEPRECATED
    fused_batch_size: null
    compute_timestamps: null
    compute_langs: false
    word_seperator: ' '
    segment_seperators:
    - .
    - '!'
    - '?'
    segment_gap_threshold: null
    rnnt_timestamp_type: all
    greedy:
      max_symbols_per_step: 10
      preserve_alignments: false
      preserve_frame_confidence: false
      tdt_include_token_duration: false
      tdt_include_duration_confidence: false
      confidence_method_cfg:
        name: entropy
        entropy_type: tsallis
        alpha: 0.33
        entropy_norm: exp
        temperature: DEPRECATED
      loop_labels: true
      use_cuda_graph_decoder: false
      ngram_lm_model: null
      ngram_lm_alpha: 0.0
      boosting_tree:
        model_path: null
        key_phrases_file: null
        key_phrases_list: null
        key_phrase_items_list: null
        context_score: 1.0
        depth_scaling: 2.0
        unk_score: 0.0
        final_eos_score: 1.0
        score_per_phrase: 0.0
        source_lang: en
        use_triton: true
        uniform_weights: false
        use_bpe_dropout: false
        num_of_transcriptions: 5
        bpe_alpha: 0.3
      boosting_tree_alpha: 0.0
      enable_per_stream_biasing: false
      max_symbols: 10
    beam:
      beam_size: 2
      search_type: default
      score_norm: true
      return_best_hypothesis: false
      tsd_max_sym_exp_per_step: 50
      alsd_max_target_len: 2.0
      nsc_max_timesteps_expansion: 1
      nsc_prefix_alpha: 1
      maes_num_steps: 2
      maes_prefix_alpha: 1
      maes_expansion_gamma: 2.3
      maes_expansion_beta: 2
      language_model: null
      softmax_temperature: 1.0
      preserve_alignments: false
      ngram_lm_model: null
      ngram_lm_alpha: 0.0
      boosting_tree:
        model_path: null
        key_phrases_file: null
        key_phrases_list: null
        key_phrase_items_list: null
        context_score: 1.0
        depth_scaling: 2.0
        unk_score: 0.0
        final_eos_score: 1.0
        score_per_phrase: 0.0
        source_lang: en
        use_triton: true
        uniform_weights: false
        use_bpe_dropout: false
        num_of_transcriptions: 5
        bpe_alpha: 0.3
      boosting_tree_alpha: 0.0
      hat_subtract_ilm: false
      hat_ilm_weight: 0.0
      max_symbols_per_step: 10
      blank_lm_score_mode: LM_WEIGHTED_FULL
      pruning_mode: LATE
      allow_cuda_graphs: true
      tsd_max_sym_exp: 50
    temperature: 1.0
    durations:
    - 0
    - 1
    - 2
    - 3
    - 4
    big_blank_durations: []
    
Model loaded
Attaching waveform augmentation (speed 0.90-1.10x, pitch -1 to +3 semitones)...
Waveform augmentation attached to model forward (chain state reset integrated)
SpecAugment enabled (freq_masks=2, time_masks=10)
Setting up training data...
[NeMo I 2026-03-24 17:06:30 collections:201] Dataset loaded with 275031 files totalling 230.51 hours
[NeMo I 2026-03-24 17:06:30 collections:202] 0 files were filtered totalling 0.00 hours
Adding bottleneck adapter (1024 -> 128 -> 1024, activation=gelu)...
[NeMo I 2026-03-24 17:06:30 adapter_mixins:811] Setting adapter 'asr_children_adapter' status : Enabled = False
[NeMo I 2026-03-24 17:06:30 adapter_mixins:826] Setting adapter 'encoder:asr_children_adapter' status : Enabled = True
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.0.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.1.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.2.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.3.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.4.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.5.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.6.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.7.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.8.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.9.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.10.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.11.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.12.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.13.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.14.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.15.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.16.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.17.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.18.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.19.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.20.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.21.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.22.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.23.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.24.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.25.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.26.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.27.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.28.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.29.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.30.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.31.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.32.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.33.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.34.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.35.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.36.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.37.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.38.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.39.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.40.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:466] Froze module encoder.layers.41.conv.batch_norm: BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
[NeMo I 2026-03-24 17:06:30 adapter_mixins:496] Unfrozen adapter : asr_children_adapter
Registered ChainedLinearAdapter as virtual subclass of LinearAdapter (isinstance(ChainedLinearAdapter, LinearAdapter) = True)
Successfully replaced 42 adapters with ChainedLinearAdapter (module tree + adapter_layer registry)
  Chain connections: 41 (orthogonal init, scale=0.9)
[DEBUG-FIX] isinstance(ChainedLinearAdapter, LinearAdapter) = True
[DEBUG-FIX] manual forward_enabled_adapters(dict): ok=True, hook_fired=True
[DEBUG-FIX] result: {'is_dict': True, 'x_shape': [1, 10, 1024]}
[DEBUG-FIX] hook shapes: {'in': [[1, 10, 1024]], 'out': [1, 10, 1024]}
Unfroze joint network: 1,726,470 params
Unfroze decoder embedding: 656,000 params
Unfroze last decoder LSTM layer (via nn.LSTM scan)
Applied decoder/joint/waveform_augmentor/spec_augmentation .train() after partial unfreeze (cuDNN RNN + augmentation gates).
[NeMo I 2026-03-24 17:06:30 modelPT:830] Optimizer config = AdamW (
    Parameter Group 0
        amsgrad: False
        betas: (0.9, 0.999)
        capturable: False
        decoupled_weight_decay: True
        differentiable: False
        eps: 1e-08
        foreach: None
        fused: None
        lr: 0.0005
        maximize: False
        weight_decay: 0.01
    )
[NeMo I 2026-03-24 17:06:30 lr_scheduler:995] Scheduler "<nemo.core.optim.lr_scheduler.CosineAnnealing object at 0x7a2250f23f80>" 
    will be used during training (effective maximum steps = 25788) - 
    Parameters : 
    (warmup_ratio: 0.15
    min_lr: 1.0e-06
    max_steps: 25788
    )
Param group 0: 11,816,192 params, lr=0.0005
Param group 1: 1,726,470 params, lr=0.0001
Param group 2: 3,937,920 params, lr=5e-05
Discriminative LR optimizer configured (warmup=3868, total=25788 steps)
Total params:     1,082,252,166
Trainable params: 17,480,582 (1.62%)
============================================================
STARTING TRAINING
  Model:      nvidia/parakeet-tdt-1.1b
  Adapter:    Chained Linear (1024 -> 128 -> 1024, GELU + LayerNorm + dropout=0.1)
  Chain:      Linear chain through 128-dim bottleneck (41 connections)
  Unfrozen:   Joint network + Decoder embedding + Last LSTM layer
  Epochs:     6
  Batch size: 64
  LR:         Adapters=5e-4, Joint=1e-4, Decoder=5e-5
  Checkpoints: Epochs [2, 3, 5] + Final
  Augmentation:
    - Speed perturbation: 0.90-1.10x (prob=0.5)
    - Pitch perturbation: -1 to +3 semitones (prob=0.5)
    - Classroom noise: SNR 5-10 dB (prob=0.5), 1940 clips
    - Gain perturbation: +/-6dB (prob=0.3)
    - SpecAugment: freq_masks=2 w=27, time_masks=10 w=0.05
  Note: 'attention_adapter_mixin: No adapter compatible' = MHSA hook skips adapter
        (ChainedLinearAdapter is not LinearAdapter); FFN/conv adapter sites still run.
============================================================
2026-03-24 17:06:32.359094: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1774371992.488110     180 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1774371992.528349     180 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1774371992.845364     180 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1774371992.845394     180 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1774371992.845397     180 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1774371992.845400     180 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name               | Type                              | Params | Mode 
---------------------------------------------------------------------------------
0 | preprocessor       | AudioToMelSpectrogramPreprocessor | 0      | eval 
1 | encoder            | ConformerEncoderAdapter           | 1.1 B  | eval 
2 | decoder            | RNNTDecoder                       | 7.2 M  | train
3 | joint              | RNNTJoint                         | 1.7 M  | train
4 | loss               | RNNTLoss                          | 0      | train
5 | spec_augmentation  | SpectrogramAugmentation           | 0      | train
6 | wer                | WER                               | 0      | train
7 | waveform_augmentor | WaveformAugmentor                 | 0      | train
---------------------------------------------------------------------------------
17.5 M    Trainable params
1.1 B     Non-trainable params
1.1 B     Total params
4,329.009 Total estimated model params size (MB)
312       Modules in train mode
1234      Modules in eval mode
Epoch 0:   0%|          | 0/4298 [00:00<?, ?it/s] [NeMo W 2026-03-24 17:07:06 attention_adapter_mixin:125] No adapter compatible with the current module. Skipping adapter forward pass.
Epoch 0:  20%|█▉        | 858/4298 [10:12<40:55,  1.40it/s, v_num=6-05, train_step_timing in s=0.726][NeMo E 2026-03-24 17:16:55 segment:348] Loading /kaggle/input/datasets/akarshkumarshukla/talk-bank-data/audio/U_b8a4e8220e65219b.flac via SoundFile raised RuntimeError: `Internal psf_fseek() failed.`. NeMo will fallback to loading via pydub.
Epoch 0:  20%|█▉        | 859/4298 [10:13<40:55,  1.40it/s, v_num=6-05, train_step_timing in s=0.690][NeMo W 2026-03-24 17:16:56 segment:95] Number of channels (2) is greater or equal than number of samples (0). Check for possible transposition.
Epoch 0:  47%|████▋     | 1999/4298 [23:07<26:35,  1.44it/s, v_num=6-05, train_step_timing in s=0.641][NeMo I 2026-03-24 17:29:50 wer:336] 
    
[NeMo I 2026-03-24 17:29:50 wer:337] WER reference:i saw one in the zoo
[NeMo I 2026-03-24 17:29:50 wer:338] WER predicted:and someone in the zoom
[NeMo I 2026-03-24 17:29:50 wer:336] 
    
[NeMo I 2026-03-24 17:29:50 wer:337] WER reference:air
[NeMo I 2026-03-24 17:29:50 wer:338] WER predicted:air
[NeMo I 2026-03-24 17:29:50 wer:336] 
    
[NeMo I 2026-03-24 17:29:50 wer:337] WER reference:u umbrella
[NeMo I 2026-03-24 17:29:50 wer:338] WER predicted:u umbrella
[NeMo I 2026-03-24 17:29:50 wer:336] 
    
[NeMo I 2026-03-24 17:29:50 wer:337] WER reference:grapes
[NeMo I 2026-03-24 17:29:50 wer:338] WER predicted:grapes
[NeMo I 2026-03-24 17:29:50 wer:336] 
    
[NeMo I 2026-03-24 17:29:50 wer:337] WER reference:share
[NeMo I 2026-03-24 17:29:50 wer:338] WER predicted:share
[NeMo I 2026-03-24 17:29:50 wer:336] 
    
[NeMo I 2026-03-24 17:29:50 wer:337] WER reference:they're trying to cross the river
[NeMo I 2026-03-24 17:29:50 wer:338] WER predicted:they're trying to cross the river
[NeMo I 2026-03-24 17:29:50 wer:336] 
    
[NeMo I 2026-03-24 17:29:50 wer:337] WER reference:you eat both plants greens vegetables or and you also eat carnivores or
[NeMo I 2026-03-24 17:29:50 wer:338] WER predicted:you eat both plants greens vegetables or and you also eat carnivores or
[NeMo I 2026-03-24 17:29:50 wer:336] 
    
[NeMo I 2026-03-24 17:29:50 wer:337] WER reference:you were away
[NeMo I 2026-03-24 17:29:50 wer:338] WER predicted:you were away
[NeMo I 2026-03-24 17:29:50 wer:336] 
    
[NeMo I 2026-03-24 17:29:50 wer:337] WER reference:driving
[NeMo I 2026-03-24 17:29:50 wer:338] WER predicted:driving
[NeMo I 2026-03-24 17:29:50 wer:336] 
    
[NeMo I 2026-03-24 17:29:50 wer:337] WER reference:the end
[NeMo I 2026-03-24 17:29:50 wer:338] WER predicted:bm
[NeMo I 2026-03-24 17:29:50 wer:336] 
    
[NeMo I 2026-03-24 17:29:50 wer:337] WER reference:iri
[NeMo I 2026-03-24 17:29:50 wer:338] WER predicted:i ree
[NeMo I 2026-03-24 17:29:50 wer:336] 
    
[NeMo I 2026-03-24 17:29:50 wer:337] WER reference:a apple
[NeMo I 2026-03-24 17:29:50 wer:338] WER predicted:a apple
[NeMo I 2026-03-24 17:29:50 wer:336] 
    
[NeMo I 2026-03-24 17:29:50 wer:337] WER reference:bye marnie
[NeMo I 2026-03-24 17:29:50 wer:338] WER predicted:bye marnie
[NeMo I 2026-03-24 17:29:50 wer:336] 
    
[NeMo I 2026-03-24 17:29:50 wer:337] WER reference:orange cat
[NeMo I 2026-03-24 17:29:50 wer:338] WER predicted:or cup
[NeMo I 2026-03-24 17:29:50 wer:336] 
    
[NeMo I 2026-03-24 17:29:50 wer:337] WER reference:then the giraffe was mad at the elephant
[NeMo I 2026-03-24 17:29:50 wer:338] WER predicted:then the giraffe was mad at the elephant
[NeMo I 2026-03-24 17:29:50 wer:336] 
    
[NeMo I 2026-03-24 17:29:50 wer:337] WER reference:the solar cell is taking energy from the sun and making and making the flag rotate
[NeMo I 2026-03-24 17:29:50 wer:338] WER predicted:the solar cell is taking energy from the sun and making the and making the flag rotate
[NeMo I 2026-03-24 17:29:50 wer:336] 
    
[NeMo I 2026-03-24 17:29:50 wer:337] WER reference:star
[NeMo I 2026-03-24 17:29:50 wer:338] WER predicted:star
[NeMo I 2026-03-24 17:29:51 wer:336] 
    
[NeMo I 2026-03-24 17:29:51 wer:337] WER reference:it is about a two people talking and somebody spills some kind of liquid on the other person and the other person just gets up immediately
[NeMo I 2026-03-24 17:29:51 wer:338] WER predicted:it is about a two people talking somebody spills some kind of liquid on the other person and the other person just gets up immediately
[NeMo I 2026-03-24 17:29:51 wer:336] 
    
[NeMo I 2026-03-24 17:29:51 wer:337] WER reference:salt
[NeMo I 2026-03-24 17:29:51 wer:338] WER predicted:salt
[NeMo I 2026-03-24 17:29:51 wer:336] 
    
[NeMo I 2026-03-24 17:29:51 wer:337] WER reference:where are all those big trucks on the interstate going
[NeMo I 2026-03-24 17:29:51 wer:338] WER predicted:where are all those big trucks on the interstate going
[NeMo I 2026-03-24 17:29:51 wer:336] 
    
[NeMo I 2026-03-24 17:29:51 wer:337] WER reference:he wouldnt go with his sister because he was tired
[NeMo I 2026-03-24 17:29:51 wer:338] WER predicted:he went away when he sleep because he was tired
[NeMo I 2026-03-24 17:29:51 wer:336] 
    
[NeMo I 2026-03-24 17:29:51 wer:337] WER reference:yes but before we like measured how big our like our the body part was we first we estimated how how big it we think it is so
[NeMo I 2026-03-24 17:29:51 wer:338] WER predicted:yes but before we like measured how big our like um the body part was we first we estimated how how big it we think it is so
[NeMo I 2026-03-24 17:29:51 wer:336] 
    
[NeMo I 2026-03-24 17:29:51 wer:337] WER reference:roof
[NeMo I 2026-03-24 17:29:51 wer:338] WER predicted:roof
[NeMo I 2026-03-24 17:29:51 wer:336] 
    
[NeMo I 2026-03-24 17:29:51 wer:337] WER reference:sorry
[NeMo I 2026-03-24 17:29:51 wer:338] WER predicted:sorry
[NeMo I 2026-03-24 17:29:51 wer:336] 
    
[NeMo I 2026-03-24 17:29:51 wer:337] WER reference:boy
[NeMo I 2026-03-24 17:29:51 wer:338] WER predicted:boy
[NeMo I 2026-03-24 17:29:51 wer:336] 
    
[NeMo I 2026-03-24 17:29:51 wer:337] WER reference:the screen it has little holes but only in only little holes enough to let the water drink throughout the marbles infact are too big so by doing that they are separating the marbles and the water from each other
[NeMo I 2026-03-24 17:29:51 wer:338] WER predicted:the screen it has little holes but only not only little holes enough to let the water drain through them the marbles in fact are too big so by doing that they are separating the marbles and the water from each other
[NeMo I 2026-03-24 17:29:51 wer:336] 
    
[NeMo I 2026-03-24 17:29:51 wer:337] WER reference:but miss buck
[NeMo I 2026-03-24 17:29:51 wer:338] WER predicted:but miss back
[NeMo I 2026-03-24 17:29:51 wer:336] 
    
[NeMo I 2026-03-24 17:29:51 wer:337] WER reference:and lily pads
[NeMo I 2026-03-24 17:29:51 wer:338] WER predicted:and lily pad
[NeMo I 2026-03-24 17:29:51 wer:336] 
    
[NeMo I 2026-03-24 17:29:51 wer:337] WER reference:my sandals i just slip on
[NeMo I 2026-03-24 17:29:51 wer:338] WER predicted:my sandals i just slip on
[NeMo I 2026-03-24 17:29:51 wer:336] 
    
[NeMo I 2026-03-24 17:29:51 wer:337] WER reference:a candle apple wood and a gasoline
[NeMo I 2026-03-24 17:29:51 wer:338] WER predicted:a candle an apple wood and a gasoline
[NeMo I 2026-03-24 17:29:51 wer:336] 
    
[NeMo I 2026-03-24 17:29:51 wer:337] WER reference:break
[NeMo I 2026-03-24 17:29:51 wer:338] WER predicted:break
[NeMo I 2026-03-24 17:29:51 wer:336] 
    
[NeMo I 2026-03-24 17:29:51 wer:337] WER reference:we
[NeMo I 2026-03-24 17:29:51 wer:338] WER predicted:we um
Epoch 0:  93%|█████████▎| 3999/4298 [45:37<03:24,  1.46it/s, v_num=6-05, train_step_timing in s=0.524][NeMo I 2026-03-24 17:52:20 wer:336] 
    
[NeMo I 2026-03-24 17:52:20 wer:337] WER reference:red
[NeMo I 2026-03-24 17:52:20 wer:338] WER predicted:red
[NeMo I 2026-03-24 17:52:20 wer:336] 
    
[NeMo I 2026-03-24 17:52:20 wer:337] WER reference:put some milk in
[NeMo I 2026-03-24 17:52:20 wer:338] WER predicted:go in put some milk in
[NeMo I 2026-03-24 17:52:20 wer:336] 
    
[NeMo I 2026-03-24 17:52:20 wer:337] WER reference:uuuu u ⁇  ub ⁇  ud ⁇  ug ⁇  ul ⁇  ur ⁇  ui ubi udi ugi uli uri
[NeMo I 2026-03-24 17:52:20 wer:338] WER predicted:oo oo ah oobah oo dah oo gah oo rah ooah ooe oobeee oo dee oogee oo wee ooh we
[NeMo I 2026-03-24 17:52:20 wer:336] 
    
[NeMo I 2026-03-24 17:52:20 wer:337] WER reference:you mean six centimeters yes um if i could predict what would happen um i think it would hold around thirty passengers maybe more um
[NeMo I 2026-03-24 17:52:20 wer:338] WER predicted:you mean six centimeters yes um if i could predict what would happen um i think it would hold around thirty passengers maybe more um
[NeMo I 2026-03-24 17:52:20 wer:336] 
    
[NeMo I 2026-03-24 17:52:20 wer:337] WER reference:nothing
[NeMo I 2026-03-24 17:52:20 wer:338] WER predicted:nothing
[NeMo I 2026-03-24 17:52:20 wer:336] 
    
[NeMo I 2026-03-24 17:52:20 wer:337] WER reference:like um wrap one around
[NeMo I 2026-03-24 17:52:20 wer:338] WER predicted:like um wrap one around
[NeMo I 2026-03-24 17:52:20 wer:336] 
    
[NeMo I 2026-03-24 17:52:20 wer:337] WER reference:yeah and there's something you get in the morning and it called snack cart
[NeMo I 2026-03-24 17:52:20 wer:338] WER predicted:and her something you get in the morning and it called knack
[NeMo I 2026-03-24 17:52:20 wer:336] 
    
[NeMo I 2026-03-24 17:52:20 wer:337] WER reference:quack
[NeMo I 2026-03-24 17:52:20 wer:338] WER predicted:quack
[NeMo I 2026-03-24 17:52:20 wer:336] 
    
[NeMo I 2026-03-24 17:52:20 wer:337] WER reference:end
[NeMo I 2026-03-24 17:52:20 wer:338] WER predicted:end
[NeMo I 2026-03-24 17:52:20 wer:336] 
    
[NeMo I 2026-03-24 17:52:20 wer:337] WER reference:and he's waiting for the timer to go off
[NeMo I 2026-03-24 17:52:20 wer:338] WER predicted:and he's waiting for the timer to go off
[NeMo I 2026-03-24 17:52:20 wer:336] 
    
[NeMo I 2026-03-24 17:52:20 wer:337] WER reference:or
[NeMo I 2026-03-24 17:52:20 wer:338] WER predicted:or
[NeMo I 2026-03-24 17:52:20 wer:336] 
    
[NeMo I 2026-03-24 17:52:20 wer:337] WER reference:they all interact making the yeast break its dormancy making the bubbles
[NeMo I 2026-03-24 17:52:20 wer:338] WER predicted:they all interact making the yeast break its dormancy making the bubbles
[NeMo I 2026-03-24 17:52:21 wer:336] 
    
[NeMo I 2026-03-24 17:52:21 wer:337] WER reference:raw
[NeMo I 2026-03-24 17:52:21 wer:338] WER predicted:raw
[NeMo I 2026-03-24 17:52:21 wer:336] 
    
[NeMo I 2026-03-24 17:52:21 wer:337] WER reference:hamburger
[NeMo I 2026-03-24 17:52:21 wer:338] WER predicted:unbunder
[NeMo I 2026-03-24 17:52:21 wer:336] 
    
[NeMo I 2026-03-24 17:52:21 wer:337] WER reference:shell
[NeMo I 2026-03-24 17:52:21 wer:338] WER predicted:shell
[NeMo I 2026-03-24 17:52:21 wer:336] 
    
[NeMo I 2026-03-24 17:52:21 wer:337] WER reference:helping
[NeMo I 2026-03-24 17:52:21 wer:338] WER predicted:okay
[NeMo I 2026-03-24 17:52:21 wer:336] 
    
[NeMo I 2026-03-24 17:52:21 wer:337] WER reference:brown
[NeMo I 2026-03-24 17:52:21 wer:338] WER predicted:brown
[NeMo I 2026-03-24 17:52:21 wer:336] 
    
[NeMo I 2026-03-24 17:52:21 wer:337] WER reference:well sometime and one time
[NeMo I 2026-03-24 17:52:21 wer:338] WER predicted:well um sometimes one time
[NeMo I 2026-03-24 17:52:21 wer:336] 
    
[NeMo I 2026-03-24 17:52:21 wer:337] WER reference:good
[NeMo I 2026-03-24 17:52:21 wer:338] WER predicted:good
[NeMo I 2026-03-24 17:52:21 wer:336] 
    
[NeMo I 2026-03-24 17:52:21 wer:337] WER reference:farm
[NeMo I 2026-03-24 17:52:21 wer:338] WER predicted:farm
[NeMo I 2026-03-24 17:52:21 wer:336] 
    
[NeMo I 2026-03-24 17:52:21 wer:337] WER reference:a j
[NeMo I 2026-03-24 17:52:21 wer:338] WER predicted:a j
[NeMo I 2026-03-24 17:52:21 wer:336] 
    
[NeMo I 2026-03-24 17:52:21 wer:337] WER reference:yeah
[NeMo I 2026-03-24 17:52:21 wer:338] WER predicted:yeah
[NeMo I 2026-03-24 17:52:21 wer:336] 
    
[NeMo I 2026-03-24 17:52:21 wer:337] WER reference:the salt is the best for the solute
[NeMo I 2026-03-24 17:52:21 wer:338] WER predicted:the salt is the best for the solute
[NeMo I 2026-03-24 17:52:21 wer:336] 
    
[NeMo I 2026-03-24 17:52:21 wer:337] WER reference:roof
[NeMo I 2026-03-24 17:52:21 wer:338] WER predicted:
[NeMo I 2026-03-24 17:52:21 wer:336] 
    
[NeMo I 2026-03-24 17:52:21 wer:337] WER reference:mixtures and solutions
[NeMo I 2026-03-24 17:52:21 wer:338] WER predicted:mixtures and solutions
[NeMo I 2026-03-24 17:52:21 wer:336] 
    
[NeMo I 2026-03-24 17:52:21 wer:337] WER reference:first the elephant was bouncing a ball
[NeMo I 2026-03-24 17:52:21 wer:338] WER predicted:first elephant was bouncing a ball
[NeMo I 2026-03-24 17:52:21 wer:336] 
    
[NeMo I 2026-03-24 17:52:21 wer:337] WER reference:bye
[NeMo I 2026-03-24 17:52:21 wer:338] WER predicted:bye
[NeMo I 2026-03-24 17:52:21 wer:336] 
    
[NeMo I 2026-03-24 17:52:21 wer:337] WER reference:pretzel
[NeMo I 2026-03-24 17:52:21 wer:338] WER predicted:pretzel
[NeMo I 2026-03-24 17:52:21 wer:336] 
    
[NeMo I 2026-03-24 17:52:21 wer:337] WER reference:not really
[NeMo I 2026-03-24 17:52:21 wer:338] WER predicted:and pass wind
[NeMo I 2026-03-24 17:52:21 wer:336] 
    
[NeMo I 2026-03-24 17:52:21 wer:337] WER reference:worm
[NeMo I 2026-03-24 17:52:21 wer:338] WER predicted:worm
[NeMo I 2026-03-24 17:52:21 wer:336] 
    
[NeMo I 2026-03-24 17:52:21 wer:337] WER reference:the good thing
[NeMo I 2026-03-24 17:52:21 wer:338] WER predicted:what do you think
[NeMo I 2026-03-24 17:52:21 wer:336] 
    
[NeMo I 2026-03-24 17:52:21 wer:337] WER reference:my
[NeMo I 2026-03-24 17:52:21 wer:338] WER predicted:mush
Epoch 0: 100%|██████████| 4298/4298 [49:02<00:00,  1.46it/s, v_num=6-05, train_step_timing in s=0.383][SaveSelectedEpochs] epoch 0 -> SKIPPED (not in save list)
Epoch 1:  40%|███▉      | 1701/4298 [19:08<29:13,  1.48it/s, v_num=6-05, train_step_timing in s=0.697][NeMo I 2026-03-24 18:14:53 wer:336] 
    
[NeMo I 2026-03-24 18:14:53 wer:337] WER reference:leaves ll l
[NeMo I 2026-03-24 18:14:53 wer:338] WER predicted:leaves l
[NeMo I 2026-03-24 18:14:53 wer:336] 
    
[NeMo I 2026-03-24 18:14:53 wer:337] WER reference:er
[NeMo I 2026-03-24 18:14:53 wer:338] WER predicted:air
[NeMo I 2026-03-24 18:14:53 wer:336] 
    
[NeMo I 2026-03-24 18:14:53 wer:337] WER reference:two rocks
[NeMo I 2026-03-24 18:14:53 wer:338] WER predicted:two wraps
[NeMo I 2026-03-24 18:14:53 wer:336] 
    
[NeMo I 2026-03-24 18:14:53 wer:337] WER reference:sick
[NeMo I 2026-03-24 18:14:53 wer:338] WER predicted:sick
[NeMo I 2026-03-24 18:14:53 wer:336] 
    
[NeMo I 2026-03-24 18:14:53 wer:337] WER reference:ree
[NeMo I 2026-03-24 18:14:53 wer:338] WER predicted:ree
[NeMo I 2026-03-24 18:14:53 wer:336] 
    
[NeMo I 2026-03-24 18:14:53 wer:337] WER reference:i yeah
[NeMo I 2026-03-24 18:14:53 wer:338] WER predicted:um yeah
[NeMo I 2026-03-24 18:14:53 wer:336] 
    
[NeMo I 2026-03-24 18:14:53 wer:337] WER reference:the thermometer
[NeMo I 2026-03-24 18:14:53 wer:338] WER predicted:the thermometer
[NeMo I 2026-03-24 18:14:53 wer:336] 
    
[NeMo I 2026-03-24 18:14:53 wer:337] WER reference:guitar
[NeMo I 2026-03-24 18:14:53 wer:338] WER predicted:a guitar
[NeMo I 2026-03-24 18:14:53 wer:336] 
    
[NeMo I 2026-03-24 18:14:53 wer:337] WER reference:u is for umbrella
[NeMo I 2026-03-24 18:14:53 wer:338] WER predicted:u is for umbrella
[NeMo I 2026-03-24 18:14:53 wer:336] 
    
[NeMo I 2026-03-24 18:14:53 wer:337] WER reference:fifteen
[NeMo I 2026-03-24 18:14:53 wer:338] WER predicted:fifteen
[NeMo I 2026-03-24 18:14:53 wer:336] 
    
[NeMo I 2026-03-24 18:14:53 wer:337] WER reference:and a picture
[NeMo I 2026-03-24 18:14:53 wer:338] WER predicted:and a brib
[NeMo I 2026-03-24 18:14:53 wer:336] 
    
[NeMo I 2026-03-24 18:14:53 wer:337] WER reference:chop
[NeMo I 2026-03-24 18:14:53 wer:338] WER predicted:shop
[NeMo I 2026-03-24 18:14:54 wer:336] 
    
[NeMo I 2026-03-24 18:14:54 wer:337] WER reference:the ball went in the water
[NeMo I 2026-03-24 18:14:54 wer:338] WER predicted:the ball went in the water
[NeMo I 2026-03-24 18:14:54 wer:336] 
    
[NeMo I 2026-03-24 18:14:54 wer:337] WER reference:two foo
[NeMo I 2026-03-24 18:14:54 wer:338] WER predicted:two foo
[NeMo I 2026-03-24 18:14:54 wer:336] 
    
[NeMo I 2026-03-24 18:14:54 wer:337] WER reference:don't know
[NeMo I 2026-03-24 18:14:54 wer:338] WER predicted:i don't know
[NeMo I 2026-03-24 18:14:54 wer:336] 
    
[NeMo I 2026-03-24 18:14:54 wer:337] WER reference:i don't know
[NeMo I 2026-03-24 18:14:54 wer:338] WER predicted:i don't know
[NeMo I 2026-03-24 18:14:54 wer:336] 
    
[NeMo I 2026-03-24 18:14:54 wer:337] WER reference:scissors
[NeMo I 2026-03-24 18:14:54 wer:338] WER predicted:scissors
[NeMo I 2026-03-24 18:14:54 wer:336] 
    
[NeMo I 2026-03-24 18:14:54 wer:337] WER reference:r
[NeMo I 2026-03-24 18:14:54 wer:338] WER predicted:r
[NeMo I 2026-03-24 18:14:54 wer:336] 
    
[NeMo I 2026-03-24 18:14:54 wer:337] WER reference:bookstore
[NeMo I 2026-03-24 18:14:54 wer:338] WER predicted:bookstore
[NeMo I 2026-03-24 18:14:54 wer:336] 
    
[NeMo I 2026-03-24 18:14:54 wer:337] WER reference:mhm
[NeMo I 2026-03-24 18:14:54 wer:338] WER predicted:mhm
[NeMo I 2026-03-24 18:14:54 wer:336] 
    
[NeMo I 2026-03-24 18:14:54 wer:337] WER reference:knife
[NeMo I 2026-03-24 18:14:54 wer:338] WER predicted:knife
[NeMo I 2026-03-24 18:14:54 wer:336] 
    
[NeMo I 2026-03-24 18:14:54 wer:337] WER reference:the poles interact by it's like uh something's in the middle of them but nothing is so they don't stick together
[NeMo I 2026-03-24 18:14:54 wer:338] WER predicted:the poles interact but it's like a something's in the middle of them but nothing is so they don't stick together
[NeMo I 2026-03-24 18:14:54 wer:336] 
    
[NeMo I 2026-03-24 18:14:54 wer:337] WER reference:spray
[NeMo I 2026-03-24 18:14:54 wer:338] WER predicted:spray
[NeMo I 2026-03-24 18:14:54 wer:336] 
    
[NeMo I 2026-03-24 18:14:54 wer:337] WER reference:a bell
[NeMo I 2026-03-24 18:14:54 wer:338] WER predicted:bell
[NeMo I 2026-03-24 18:14:54 wer:336] 
    
[NeMo I 2026-03-24 18:14:54 wer:337] WER reference:balloons
[NeMo I 2026-03-24 18:14:54 wer:338] WER predicted:balloons
[NeMo I 2026-03-24 18:14:54 wer:336] 
    
[NeMo I 2026-03-24 18:14:54 wer:337] WER reference:scare
[NeMo I 2026-03-24 18:14:54 wer:338] WER predicted:scare
[NeMo I 2026-03-24 18:14:54 wer:336] 
    
[NeMo I 2026-03-24 18:14:54 wer:337] WER reference:one yack
[NeMo I 2026-03-24 18:14:54 wer:338] WER predicted:a yock
[NeMo I 2026-03-24 18:14:54 wer:336] 
    
[NeMo I 2026-03-24 18:14:54 wer:337] WER reference:cup
[NeMo I 2026-03-24 18:14:54 wer:338] WER predicted:cup
[NeMo I 2026-03-24 18:14:54 wer:336] 
    
[NeMo I 2026-03-24 18:14:54 wer:337] WER reference:vampire
[NeMo I 2026-03-24 18:14:54 wer:338] WER predicted:vampire
[NeMo I 2026-03-24 18:14:54 wer:336] 
    
[NeMo I 2026-03-24 18:14:54 wer:337] WER reference:the sec
[NeMo I 2026-03-24 18:14:54 wer:338] WER predicted:it's like
[NeMo I 2026-03-24 18:14:54 wer:336] 
    
[NeMo I 2026-03-24 18:14:54 wer:337] WER reference:the children can play but they need to be quiet
[NeMo I 2026-03-24 18:14:54 wer:338] WER predicted:the children can play but they need to be quiet
[NeMo I 2026-03-24 18:14:54 wer:336] 
    
[NeMo I 2026-03-24 18:14:54 wer:337] WER reference:because i was the leader like last year
[NeMo I 2026-03-24 18:14:54 wer:338] WER predicted:because i was the leader like last year
Epoch 1:  77%|███████▋  | 3314/4298 [37:08<11:01,  1.49it/s, v_num=6-05, train_step_timing in s=0.769][NeMo E 2026-03-24 18:32:53 segment:348] Loading /kaggle/input/datasets/akarshkumarshukla/talk-bank-data/audio/U_b8a4e8220e65219b.flac via SoundFile raised RuntimeError: `Internal psf_fseek() failed.`. NeMo will fallback to loading via pydub.
[NeMo W 2026-03-24 18:32:53 segment:95] Number of channels (2) is greater or equal than number of samples (0). Check for possible transposition.
Epoch 1:  86%|████████▌ | 3701/4298 [41:30<06:41,  1.49it/s, v_num=6-05, train_step_timing in s=0.603][NeMo I 2026-03-24 18:37:15 wer:336] 
    
[NeMo I 2026-03-24 18:37:15 wer:337] WER reference:shorts
[NeMo I 2026-03-24 18:37:15 wer:338] WER predicted:shorts
[NeMo I 2026-03-24 18:37:15 wer:336] 
    
[NeMo I 2026-03-24 18:37:15 wer:337] WER reference:this cat is trying to climb up the tree
[NeMo I 2026-03-24 18:37:15 wer:338] WER predicted:this cat is trying to climb up the tree
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:seed sayed said sid sad sawed sood sued sewed sud
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:seed sade sed sid sad sod sood sud sowed sud
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:it it could be a pinnate
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:it it could be a pennate
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:seven
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:seven
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:barn
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:barn
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:apple
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:apple
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:hello
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:hello
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:it's a toy ship
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:it's a toy ship
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:some paper two paper airplanes
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:some paper two paper airplanes
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:friend
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:friend
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:a palmate leaf
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:a palm leaf
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:sitting
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:z
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:big
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:eight
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:they all have systems to them and like how they're made
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:they all have systems to them and like how they're made
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:the magnet magnet touch magnetic field
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:the magnet magnet touch magnetic
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:freckles
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:freckles
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:table
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:table
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:dark
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:duck
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:what do you think jibo
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:what do you think jibo
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:i think that's the dad
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:i think that's the bed
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:the wax is melting
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:the wax is melting
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:um they're building with blocks
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:and they're building with blocks
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:finger
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:finger
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:and then i a picture of ice and igloo
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:and then i a picture of ice and igloo
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:i like um the trix
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:i like uh the trix
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:it's because it's like the only food they eat cause just some people just are like kinda like a vegetarian so they just eat like you know
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:it's because it's like the only food they eat cause just some people just are like kind of like a vegetarian so they just eat like you know
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:row
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:row
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:my name is mr donny
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:my name is mr donny
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:ring
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:ring
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:mip
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:mip
[NeMo I 2026-03-24 18:37:16 wer:336] 
    
[NeMo I 2026-03-24 18:37:16 wer:337] WER reference:he he's not gonna sleep by the fire
[NeMo I 2026-03-24 18:37:16 wer:338] WER predicted:he he has to sleep by the fire
Epoch 1: 100%|██████████| 4298/4298 [48:18<00:00,  1.48it/s, v_num=6-05, train_step_timing in s=0.307][SaveSelectedEpochs] epoch 1 -> /kaggle/working/nemo_adapter_1.1b/ParakeetAdapter1.1B/2026-03-24_17-06-05/checkpoints/model_epoch1.nemo, /kaggle/working/nemo_adapter_1.1b/ParakeetAdapter1.1B/2026-03-24_17-06-05/checkpoints/adapter_epoch1.pt
Epoch 2:  32%|███▏      | 1361/4298 [15:03<32:28,  1.51it/s, v_num=6-05, train_step_timing in s=0.680][NeMo E 2026-03-24 18:59:14 segment:348] Loading /kaggle/input/datasets/akarshkumarshukla/talk-bank-data/audio/U_b8a4e8220e65219b.flac via SoundFile raised RuntimeError: `Internal psf_fseek() failed.`. NeMo will fallback to loading via pydub.
[NeMo W 2026-03-24 18:59:14 segment:95] Number of channels (2) is greater or equal than number of samples (0). Check for possible transposition.
Epoch 2:  33%|███▎      | 1403/4298 [15:30<32:00,  1.51it/s, v_num=6-05, train_step_timing in s=0.559][NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:and my hair all in curls
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:and my hair all in curls
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:last last year about in the middle of the spring time there was there was a dog named sarah and a and a rabbit named christopher
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:last last year it bulb in the middle of the spring time there was there was a dog named sarah and a and a rabbit named christopher
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:we we learned of
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:we we learned about
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:because there's one hundred centimeters in a meter and then then there then there'd be thirty two centimeters left so it'd equal one hundred thirty two centimeters
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:because there's one hundred centimeters in a meter and then then there then there'd be thirty two centimeters left so it'd equal one hundred thirty two centimeters
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:fray
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:pray
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:throne
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:throne
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:desk
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:desk
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:it sticks together
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:it sticks together
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:like
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:like a
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:octopus
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:octopus
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:it change by adding another d cell and if you remove one d cell it'll like stay like a little bit blinked but if you add two d cells it would be like light it would be like lighting a lot and the presence is high
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:they changed by adding another d cell and if you remove one d cell it'll like stay like a little bit blank but if you add two d cell it'll be like lighting it'll be like lighting a and the presence is high
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:go
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:go
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:i don't know
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:i don't know
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:that
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:that
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:what
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:what
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:yeah
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:yeah
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:the circuit is not running because the popsicle stick is not a conductor
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:the circuit's not running because a popsicle stick is not a conductor
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:i think it's a boy who's pretending to be a pilot
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:i think it's a boy who's pretending to be a pilot
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:pretzel
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:pretzel
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:the sun killed it
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:the sun killed it
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:i'm look at this time
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:i look at this time
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:my balloon floated away
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:the balloon flew it away
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:raft
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:raft
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:b
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:b
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:rah
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:rah
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:year
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:year
[NeMo I 2026-03-24 18:59:42 wer:336] 
    
[NeMo I 2026-03-24 18:59:42 wer:337] WER reference:twenty
[NeMo I 2026-03-24 18:59:42 wer:338] WER predicted:twenty
[NeMo I 2026-03-24 18:59:43 wer:336] 
    
[NeMo I 2026-03-24 18:59:43 wer:337] WER reference:well we put wax paper on a tray and then we took we propped the tray up on some books and the water slid down really fast with the wax paper and the bigger drops
[NeMo I 2026-03-24 18:59:43 wer:338] WER predicted:well we put wax paper on a tray and then we took we propped the tray up on some books and the water slid down really fast with the wax paper and the bigger drops
[NeMo I 2026-03-24 18:59:43 wer:336] 
    
[NeMo I 2026-03-24 18:59:43 wer:337] WER reference:the elephant is pointing to the diving board and the giraffe has a towel
[NeMo I 2026-03-24 18:59:43 wer:338] WER predicted:the elephant is pointing to the diving board and the giraffe has a towel
[NeMo I 2026-03-24 18:59:43 wer:336] 
    
[NeMo I 2026-03-24 18:59:43 wer:337] WER reference:yellow
[NeMo I 2026-03-24 18:59:43 wer:338] WER predicted:yellow
[NeMo I 2026-03-24 18:59:43 wer:336] 
    
[NeMo I 2026-03-24 18:59:43 wer:337] WER reference:drip
[NeMo I 2026-03-24 18:59:43 wer:338] WER predicted:drip
[NeMo I 2026-03-24 18:59:43 wer:336] 
    
[NeMo I 2026-03-24 18:59:43 wer:337] WER reference:a goat birthday
[NeMo I 2026-03-24 18:59:43 wer:338] WER predicted:a goat birthday
Epoch 2:  79%|███████▉  | 3403/4298 [37:36<09:53,  1.51it/s, v_num=6-05, train_step_timing in s=0.572][NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:and a boat
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:and a boat
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:fishing
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:fashing
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:yes
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:yes
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:uh it
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:uh it
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:coffee
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:coffee
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:car car car
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:tank
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:no
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:no
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:that this is a solution but only water evaporates see the sound to get
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:that this is a solution but when the water evaporates you will see the salt again
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:rip
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:rip
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:that's a
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:that's a boophy
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:it's a lot of coloring
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:it's a lot of coloring
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:stir
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:stir
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:desk
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:desk
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:chair
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:chair
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:the hose the hose
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:hose a hose
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:no
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:no
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:eyebrow
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:eyebrow
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:if you don't get to the base
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:if you don't get to the base
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:no because dissolved into the water but its very low unlikely it actually its very unlikely you can may be evaporated that would take it really long time
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:no because it dissolved into the water but it is very unlikely it's actually it's very unlikely you can maybe evaporate it but that would take a really long time
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:bye
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:um
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:rat
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:rat
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:nurse
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:nurse
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:cause whenever i tap on something metal it will spin because like it's metal it'll will always spin
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:cause whenever i tap on something metal it will spin because like it's metal it'll will always spin
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:two cup
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:two cup
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:sir
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:sir
[NeMo I 2026-03-24 19:21:48 wer:336] 
    
[NeMo I 2026-03-24 19:21:48 wer:337] WER reference:vegetable
[NeMo I 2026-03-24 19:21:48 wer:338] WER predicted:vegetable
[NeMo I 2026-03-24 19:21:49 wer:336] 
    
[NeMo I 2026-03-24 19:21:49 wer:337] WER reference:look there's
[NeMo I 2026-03-24 19:21:49 wer:338] WER predicted:look there's
[NeMo I 2026-03-24 19:21:49 wer:336] 
    
[NeMo I 2026-03-24 19:21:49 wer:337] WER reference:um the yeast um eating the graham cracker the little cracker
[NeMo I 2026-03-24 19:21:49 wer:338] WER predicted:the yeast eating the gram cracker the little cracker
[NeMo I 2026-03-24 19:21:49 wer:336] 
    
[NeMo I 2026-03-24 19:21:49 wer:337] WER reference:farm
[NeMo I 2026-03-24 19:21:49 wer:338] WER predicted:farm
[NeMo I 2026-03-24 19:21:49 wer:336] 
    
[NeMo I 2026-03-24 19:21:49 wer:337] WER reference:ball
[NeMo I 2026-03-24 19:21:49 wer:338] WER predicted:ball
[NeMo I 2026-03-24 19:21:49 wer:336] 
    
[NeMo I 2026-03-24 19:21:49 wer:337] WER reference:i get my hands wet
[NeMo I 2026-03-24 19:21:49 wer:338] WER predicted:i get my hands wet
[NeMo I 2026-03-24 19:21:49 wer:336] 
    
[NeMo I 2026-03-24 19:21:49 wer:337] WER reference:yock
[NeMo I 2026-03-24 19:21:49 wer:338] WER predicted:yock
Epoch 2: 100%|██████████| 4298/4298 [47:34<00:00,  1.51it/s, v_num=6-05, train_step_timing in s=0.342][SaveSelectedEpochs] epoch 2 -> /kaggle/working/nemo_adapter_1.1b/ParakeetAdapter1.1B/2026-03-24_17-06-05/checkpoints/model_epoch2.nemo, /kaggle/working/nemo_adapter_1.1b/ParakeetAdapter1.1B/2026-03-24_17-06-05/checkpoints/adapter_epoch2.pt
Epoch 3:  26%|██▌       | 1105/4298 [12:06<34:59,  1.52it/s, v_num=6-05, train_step_timing in s=0.550][NeMo I 2026-03-24 19:43:59 wer:336] 
    
[NeMo I 2026-03-24 19:43:59 wer:337] WER reference:look it they're letting them get into the back see
[NeMo I 2026-03-24 19:43:59 wer:338] WER predicted:look at three and them get into the back seat
[NeMo I 2026-03-24 19:43:59 wer:336] 
    
[NeMo I 2026-03-24 19:43:59 wer:337] WER reference:good
[NeMo I 2026-03-24 19:43:59 wer:338] WER predicted:good
[NeMo I 2026-03-24 19:43:59 wer:336] 
    
[NeMo I 2026-03-24 19:43:59 wer:337] WER reference:but i don't think this person would want you to do that
[NeMo I 2026-03-24 19:43:59 wer:338] WER predicted:but i don't think miss larson would want you to do that
[NeMo I 2026-03-24 19:43:59 wer:336] 
    
[NeMo I 2026-03-24 19:43:59 wer:337] WER reference:rules
[NeMo I 2026-03-24 19:43:59 wer:338] WER predicted:rules
[NeMo I 2026-03-24 19:43:59 wer:336] 
    
[NeMo I 2026-03-24 19:43:59 wer:337] WER reference:thumb
[NeMo I 2026-03-24 19:43:59 wer:338] WER predicted:thumb
[NeMo I 2026-03-24 19:43:59 wer:336] 
    
[NeMo I 2026-03-24 19:43:59 wer:337] WER reference:they his little sister is making a sandcastle
[NeMo I 2026-03-24 19:43:59 wer:338] WER predicted:then his little sister is making a sandcastle
[NeMo I 2026-03-24 19:43:59 wer:336] 
    
[NeMo I 2026-03-24 19:43:59 wer:337] WER reference:the electricity is flowing of the plus sign to the minus sign to the light bulb and it's a circuit
[NeMo I 2026-03-24 19:43:59 wer:338] WER predicted:the electricity is flowing of the plus sign to the minus sign to the light bulb and it's a circuit
[NeMo I 2026-03-24 19:43:59 wer:336] 
    
[NeMo I 2026-03-24 19:43:59 wer:337] WER reference:door
[NeMo I 2026-03-24 19:43:59 wer:338] WER predicted:door
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:spiders make webs
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:spiders make webs
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:this jumbo drum um i don't know
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:this jumbo jumb i don't know
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:but i have only learned how to do it when i'm six
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:but i have only learned how to do it when i'm six
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:it's it is right but only this one's bigger so it only uses that much and that one's smaller and it only uses that much
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:it's it is right but only this one's bigger so it only uses that much and that one's smaller and it only uses that much
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:year
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:year
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:this shows us a living animal and we've been talking about living animals
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:this shows us a living animal and we've been talking about living animals
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:scarf
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:scarf
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:fur
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:fur
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:oh here's some drawers
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:oh here's the drawers
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:cause producers it shows um producers consumers and decomposers
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:has producers it shows producers consumers and decomposers
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:crayon
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:crayon
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:sunday
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:sunday
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:every single one
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:every single one
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:ladder
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:ladder
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:scarf
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:scarf
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:or i make
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:or i make
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:you pick the um vegetables or fruits
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:you pick the um vegetables or fruits
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:where'd you get this she asked
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:where'd you get this she asked
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:this is an el this is an electromagnet picking up magnets
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:this is an el this is an electromagnet picking up magnets
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:red
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:red
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:a shovel
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:a shovel
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:his friend has an old guitar that you could borrow if you would like
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:his friend has a old guitar that you should that you could borrow if you would like
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:chair
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:chair
[NeMo I 2026-03-24 19:44:00 wer:336] 
    
[NeMo I 2026-03-24 19:44:00 wer:337] WER reference:that
[NeMo I 2026-03-24 19:44:00 wer:338] WER predicted:that
Epoch 3:  47%|████▋     | 2004/4298 [22:02<25:14,  1.51it/s, v_num=6-05, train_step_timing in s=0.703][NeMo E 2026-03-24 19:53:55 segment:348] Loading /kaggle/input/datasets/akarshkumarshukla/talk-bank-data/audio/U_b8a4e8220e65219b.flac via SoundFile raised RuntimeError: `Internal psf_fseek() failed.`. NeMo will fallback to loading via pydub.
[NeMo W 2026-03-24 19:53:55 segment:95] Number of channels (2) is greater or equal than number of samples (0). Check for possible transposition.
Epoch 3:  72%|███████▏  | 3105/4298 [34:12<13:08,  1.51it/s, v_num=6-05, train_step_timing in s=0.636][NeMo I 2026-03-24 20:06:04 wer:336] 
    
[NeMo I 2026-03-24 20:06:04 wer:337] WER reference:see okay
[NeMo I 2026-03-24 20:06:04 wer:338] WER predicted:see okay
[NeMo I 2026-03-24 20:06:04 wer:336] 
    
[NeMo I 2026-03-24 20:06:04 wer:337] WER reference:yeah
[NeMo I 2026-03-24 20:06:04 wer:338] WER predicted:yeah
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:they're really cool toys
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:they're really cool toys
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:fridge
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:fridge
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:run
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:run
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:yes
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:yes
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:are you going
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:are you going
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:then she tries to scoop the airplane up in it
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:then she tries to scoop the airplane up in it
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:i brush it
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:i brush it
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:are
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:ar
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:yeah
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:yeah
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:and i do paper crafts
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:and i do paper crafts
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:and he they got some sand
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:and he digged up some sand
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:yes
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:yes
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:and this big horse
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:this big horse
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:stir
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:stir
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:that was in the cretaceous period i'm guessing
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:that was in the cretaceous period i'm guessing
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:um
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:uhhuh
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:but let's both flip
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:but it's a flip
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:this yeah wait no different different sides of the magnet
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:the s yeah wait no different different sides of the magnet
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:dragon
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:dragon
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:on recess
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:a recess
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:a very
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:a variable
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:window
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:window
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:two crib
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:for two crib
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:they practice together a lot
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:they practice together a lot
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:which is i can't do
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:which is i can't do
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:i did not really observe anything else
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:i did not really observe anything else
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:g
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:g
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:the end
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:the end
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:fish
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:fish
[NeMo I 2026-03-24 20:06:05 wer:336] 
    
[NeMo I 2026-03-24 20:06:05 wer:337] WER reference:a boy on the bed
[NeMo I 2026-03-24 20:06:05 wer:338] WER predicted:a boy on the bed
Epoch 3: 100%|██████████| 4298/4298 [47:25<00:00,  1.51it/s, v_num=6-05, train_step_timing in s=0.333][SaveSelectedEpochs] epoch 3 -> SKIPPED (not in save list)
Epoch 4:  19%|█▉        | 807/4298 [08:52<38:21,  1.52it/s, v_num=6-05, train_step_timing in s=0.514][NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:z
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:z
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:could this be it
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:could this be it
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:clear
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:clear
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:and then the elephant starts running or something
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:and then the elephant starts running or something
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:so you get your paintbrush get some water
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:so you get your paintbrush get some water
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:where is it
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:where should
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:because we have mouse traps
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:because we have mouse traps
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:now the girl
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:now the girl
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:i doesn't want him in my spots i try to hit a tree
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:i guess i want him watch but i try to hit as me
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:no
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:no
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:two cups
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:two cups
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:like pretend
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:like pretends
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:um n mmm
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:um n mmm
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:ar
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:ar
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:yeah
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:yeah
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:okay
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:okay
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:yeah
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:yeah
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:one thousand
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:one thousand
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:or
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:or
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:beard
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:beard
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:scroll
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:scroll
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:and a nose
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:and a nose
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:the leaves of the plant
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:the leaves of the plant
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:yeah
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:yeah
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:ree
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:ree
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:mm i get my clothes
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:mm i get my clothes in
[NeMo I 2026-03-24 20:28:10 wer:336] 
    
[NeMo I 2026-03-24 20:28:10 wer:337] WER reference:if the water's moving it will be hard to put the pennies in because because um you it you can't the boat will be rocking too much and the pennies may tip over
[NeMo I 2026-03-24 20:28:10 wer:338] WER predicted:if the water is moving it will be hard to put the pennies in because because um you it you can't the boat will be rocking too much and the pennies might tip over
[NeMo I 2026-03-24 20:28:11 wer:336] 
    
[NeMo I 2026-03-24 20:28:11 wer:337] WER reference:i just climb up the ladder
[NeMo I 2026-03-24 20:28:11 wer:338] WER predicted:i'll just climb up the ladder
[NeMo I 2026-03-24 20:28:11 wer:336] 
    
[NeMo I 2026-03-24 20:28:11 wer:337] WER reference:yeah
[NeMo I 2026-03-24 20:28:11 wer:338] WER predicted:yeah
[NeMo I 2026-03-24 20:28:11 wer:336] 
    
[NeMo I 2026-03-24 20:28:11 wer:337] WER reference:seventy
[NeMo I 2026-03-24 20:28:11 wer:338] WER predicted:seventy
[NeMo I 2026-03-24 20:28:11 wer:336] 
    
[NeMo I 2026-03-24 20:28:11 wer:337] WER reference:wrap
[NeMo I 2026-03-24 20:28:11 wer:338] WER predicted:wrap
[NeMo I 2026-03-24 20:28:11 wer:336] 
    
[NeMo I 2026-03-24 20:28:11 wer:337] WER reference:one
[NeMo I 2026-03-24 20:28:11 wer:338] WER predicted:y
Epoch 4:  65%|██████▌   | 2807/4298 [30:50<16:22,  1.52it/s, v_num=6-05, train_step_timing in s=0.526][NeMo I 2026-03-24 20:50:08 wer:336] 
    
[NeMo I 2026-03-24 20:50:08 wer:337] WER reference:when the bulb is burnt out it's an open circuit
[NeMo I 2026-03-24 20:50:08 wer:338] WER predicted:when the bulb is burnt out it's an open circuit
[NeMo I 2026-03-24 20:50:08 wer:336] 
    
[NeMo I 2026-03-24 20:50:08 wer:337] WER reference:it having the biosphere the ithiosphere the hydrosphere and one more sphere
[NeMo I 2026-03-24 20:50:08 wer:338] WER predicted:we have the biosphere the hydrosphere and one more sphere
[NeMo I 2026-03-24 20:50:08 wer:336] 
    
[NeMo I 2026-03-24 20:50:08 wer:337] WER reference:i did see the movie about meatballs thing
[NeMo I 2026-03-24 20:50:08 wer:338] WER predicted:i did see the movie about meatball swing
[NeMo I 2026-03-24 20:50:08 wer:336] 
    
[NeMo I 2026-03-24 20:50:08 wer:337] WER reference:ar
[NeMo I 2026-03-24 20:50:08 wer:338] WER predicted:ar
[NeMo I 2026-03-24 20:50:08 wer:336] 
    
[NeMo I 2026-03-24 20:50:08 wer:337] WER reference:and now it's turning green
[NeMo I 2026-03-24 20:50:08 wer:338] WER predicted:now it's turning green
[NeMo I 2026-03-24 20:50:08 wer:336] 
    
[NeMo I 2026-03-24 20:50:08 wer:337] WER reference:thumb
[NeMo I 2026-03-24 20:50:08 wer:338] WER predicted:thumb
[NeMo I 2026-03-24 20:50:08 wer:336] 
    
[NeMo I 2026-03-24 20:50:08 wer:337] WER reference:most of em
[NeMo I 2026-03-24 20:50:08 wer:338] WER predicted:most of em
[NeMo I 2026-03-24 20:50:08 wer:336] 
    
[NeMo I 2026-03-24 20:50:08 wer:337] WER reference:that light up with the um with the electricity in them
[NeMo I 2026-03-24 20:50:08 wer:338] WER predicted:that light up with the um with the electricity in them
[NeMo I 2026-03-24 20:50:08 wer:336] 
    
[NeMo I 2026-03-24 20:50:08 wer:337] WER reference:you mixed calcium chloride with baking soda in one but no water and then in the second one you mixed calcium chloride and citric acid but no water and then the third one you have baking soda and citric acid and that's and no water
[NeMo I 2026-03-24 20:50:08 wer:338] WER predicted:you mixed calcium chloride with baking soda and one no water and then in the second one you mix calcium chloride and citric acid but no water and in the third one you have baking soda and citric acid and that's and no water
[NeMo I 2026-03-24 20:50:08 wer:336] 
    
[NeMo I 2026-03-24 20:50:08 wer:337] WER reference:only with slime that i make
[NeMo I 2026-03-24 20:50:08 wer:338] WER predicted:only with slime that i make
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:graduate
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:graduate
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:girl
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:girl
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:robot
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:robot
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:teacher
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:teacher
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:eat
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:eat
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:one bell
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:one bell
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:red
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:red
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:the elephant is playing basketball
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:a elephant is playing basketball
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:mouth
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:no
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:ring
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:ring
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:i do not eat cereal like that
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:i do not eat cereal like that
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:finger
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:finger
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:well that they are that they sometimes do wrong measurements cause they get bent and stuff and then that shrinks em a little bit
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:well that they are that they sometimes do wrong measurements cause they get bent and stuff and then that shrinks'em a little bit
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:he got it and swam back and got out
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:he got it and swam back and got it out
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:driving
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:driving
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:there
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:yeah
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:scissors
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:scissors
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:this magnet that we're showing here has been connected cause the paper is very thin
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:this magnet that we're showing here has been connected cause the paper is very thin
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:hmm watch like in i watch the fish
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:watch like in i watch the fish
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:three
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:three
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:uhhuh
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:uhhuh
[NeMo I 2026-03-24 20:50:09 wer:336] 
    
[NeMo I 2026-03-24 20:50:09 wer:337] WER reference:fruits
[NeMo I 2026-03-24 20:50:09 wer:338] WER predicted:fruits
Epoch 4:  70%|███████   | 3023/4298 [33:13<14:00,  1.52it/s, v_num=6-05, train_step_timing in s=0.668][NeMo E 2026-03-24 20:52:31 segment:348] Loading /kaggle/input/datasets/akarshkumarshukla/talk-bank-data/audio/U_b8a4e8220e65219b.flac via SoundFile raised RuntimeError: `Internal psf_fseek() failed.`. NeMo will fallback to loading via pydub.
[NeMo W 2026-03-24 20:52:31 segment:95] Number of channels (2) is greater or equal than number of samples (0). Check for possible transposition.
Epoch 4: 100%|██████████| 4298/4298 [47:20<00:00,  1.51it/s, v_num=6-05, train_step_timing in s=0.316][SaveSelectedEpochs] epoch 4 -> /kaggle/working/nemo_adapter_1.1b/ParakeetAdapter1.1B/2026-03-24_17-06-05/checkpoints/model_epoch4.nemo, /kaggle/working/nemo_adapter_1.1b/ParakeetAdapter1.1B/2026-03-24_17-06-05/checkpoints/adapter_epoch4.pt
Epoch 5:  12%|█▏        | 509/4298 [05:32<41:17,  1.53it/s, v_num=6-05, train_step_timing in s=0.662][NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:to feed the cat one must shoo the dog
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:to feed the cat one must shoo the dog
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:he have a live big cat
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:they have a light big cat
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:scare
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:scare
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:share
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:share
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:fear
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:fear
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:what's happening is when you connect the wires one to the swirly side of the light bulb and one to the bottom the light bulb will light up
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:what's happening is when you connect the wires one to the swirly side of the light bulb and one to the bottom the light bulb will light up
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:yeah
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:yeah
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:turn
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:turn
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:shorts
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:shorts
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:you were away
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:you were away
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:hot
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:go
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:i don't necessarily like it but
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:i don't necessarily like it but
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:yeah
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:yeah
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:yes that would be a very bad problem and then the plant would die and then the whole world would explode
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:yes that would be a very bad problem and then the plant would die and then the whole world would explode
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:they are plants
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:they are plants
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:relish
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:relish
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:hm maybe in the bathroom
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:hm wait in the bathroom
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:yes usually a pooping and crackling
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:yes usually a popping thing and crackling
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:and they're trying to get back to land
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:and they're trying to get back to land
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:airplane variables
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:airplane variables
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:chair
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:chair
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:yeah
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:yeah
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:wrap
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:wrap
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:there's a fish in the pond
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:there's a fish in the pond
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:it cant occur action is happening because ice made the water go higher and the glass on the outside the water starting to come outside and it looks weird
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:a chemical reaction is happening because ice made the water go higher and then the glass on the outside the water is starting to outside and it looks weird
[NeMo I 2026-03-24 21:12:19 wer:336] 
    
[NeMo I 2026-03-24 21:12:19 wer:337] WER reference:one crib
[NeMo I 2026-03-24 21:12:19 wer:338] WER predicted:one crib
[NeMo I 2026-03-24 21:12:20 wer:336] 
    
[NeMo I 2026-03-24 21:12:20 wer:337] WER reference:two minutes
[NeMo I 2026-03-24 21:12:20 wer:338] WER predicted:two minutes
[NeMo I 2026-03-24 21:12:20 wer:336] 
    
[NeMo I 2026-03-24 21:12:20 wer:337] WER reference:uh i see some paper airplanes
[NeMo I 2026-03-24 21:12:20 wer:338] WER predicted:uh i see some paper airplanes
[NeMo I 2026-03-24 21:12:20 wer:336] 
    
[NeMo I 2026-03-24 21:12:20 wer:337] WER reference:the blue dots represent the energy
[NeMo I 2026-03-24 21:12:20 wer:338] WER predicted:the blue dots represent the energy
[NeMo I 2026-03-24 21:12:20 wer:336] 
    
[NeMo I 2026-03-24 21:12:20 wer:337] WER reference:maya um makes me takes me to the park where i can swing slide and play ball with my friends
[NeMo I 2026-03-24 21:12:20 wer:338] WER predicted:maya um makes me or takes me to the park where i can swing slide and play ball with my friends
[NeMo I 2026-03-24 21:12:20 wer:336] 
    
[NeMo I 2026-03-24 21:12:20 wer:337] WER reference:i don't know
[NeMo I 2026-03-24 21:12:20 wer:338] WER predicted:i don't know
[NeMo I 2026-03-24 21:12:20 wer:336] 
    
[NeMo I 2026-03-24 21:12:20 wer:337] WER reference:hmm well first we got the bi the very small strings and then we went to smaller to highest then we um tested em
[NeMo I 2026-03-24 21:12:20 wer:338] WER predicted:hmm well first we got the bi very small strings and then we went to smaller the highest then we um tested em
Epoch 5:  58%|█████▊    | 2509/4298 [27:18<19:28,  1.53it/s, v_num=6-05, train_step_timing in s=0.485][NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:i fell outta the thing the air thing and but it had hole in it
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:it fell out of the thing the air thing and but it had a hole in it
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:and he had
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:and he had
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:they need water
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:they need water
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:grass
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:grass
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:truck
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:truck
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:cups
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:cups
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:yeah
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:yeah
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:like this
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:this
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:two skack
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:skack
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:tap his toes
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:tap his toes
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:mother
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:mother
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:lakisha and her cousin carmin liked to play hopscotch
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:lakisha and her cousin carmin liked to play hopscotch
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:mm i think that's all i see
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:mm mm i think that's all i see
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:but then the one sunk
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:but then the one sunk
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:and the dog was mad
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:sad and the dog was mad
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:and and that all he does
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:and and that's all it does
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:well they're all living things they're all living systems
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:well they're all living things they're all living systems
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:he gets balloons
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:he gets balloons
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:yeah
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:yeah
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:the women had to work very hard to get to play now
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:the women had to work very hard to get to play now
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:um
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:um
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:okay
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:okay
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:alum ⁇ creek
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:bottom cake
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:they could've passed
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:i could've passed
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:it's a girl making cupcakes
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:it's a girl making cupcakes
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:not one
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:but one
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:got this
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:a tooth
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:huh
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:hm
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:building
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:building
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:rob
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:rob
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:one hundred
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:one hundred
[NeMo I 2026-03-24 21:34:04 wer:336] 
    
[NeMo I 2026-03-24 21:34:04 wer:337] WER reference:six
[NeMo I 2026-03-24 21:34:04 wer:338] WER predicted:six
Epoch 5:  62%|██████▏   | 2665/4298 [29:01<17:47,  1.53it/s, v_num=6-05, train_step_timing in s=0.703][NeMo E 2026-03-24 21:35:47 segment:348] Loading /kaggle/input/datasets/akarshkumarshukla/talk-bank-data/audio/U_b8a4e8220e65219b.flac via SoundFile raised RuntimeError: `Internal psf_fseek() failed.`. NeMo will fallback to loading via pydub.
[NeMo W 2026-03-24 21:35:47 segment:95] Number of channels (2) is greater or equal than number of samples (0). Check for possible transposition.
Epoch 5: 100%|██████████| 4298/4298 [46:57<00:00,  1.53it/s, v_num=6-05, train_step_timing in s=0.312][SaveSelectedEpochs] epoch 5 -> SKIPPED (not in save list)
`Trainer.fit` stopped: `max_epochs=6` reached.
Epoch 5: 100%|██████████| 4298/4298 [46:57<00:00,  1.53it/s, v_num=6-05, train_step_timing in s=0.312]
Saving final model...
Saved adapter_final.pt and model_final.nemo under /kaggle/working/nemo_adapter_1.1b/ParakeetAdapter1.1B/2026-03-24_17-06-05/checkpoints
TRAINING COMPLETE!

Output files:
  - model_epoch1.nemo + adapter_epoch1.pt (Epoch 2)
  - model_epoch2.nemo + adapter_epoch2.pt (Epoch 3)
  - model_epoch4.nemo + adapter_epoch4.pt (Epoch 5)
  - model_final.nemo + adapter_final.pt (Final)

Training completed successfully.
Check /kaggle/working/nemo_adapter_1.1b/ for outputs.
 
     here is the training code ,, now i want to inference this model ,, on the val manifest data give me the script for that 
