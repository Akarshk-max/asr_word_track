"""
Learnable softmax fusion over NeMo Conformer encoder layer outputs.

Fusion uses ``register_forward_hook`` on each ``encoder.layers[i]`` output so it works under
``transcribe`` (``torch.inference_mode``) and does not depend on InterCTC / AccessMixin
registry population.

Hook tensors match NeMo's Conformer layout ``(B, T, d_model)``. InterCTC registers
``out_proj`` applied on that layout, then ``transpose`` to ``(B, d', T)`` — the same layout
the transducer joint expects. We mirror that so fusion is not left at 1024-d when
``encoder.out_proj`` projects to a smaller ``d'`` (e.g. Parakeet-TDT).

``_collect_interctc_outputs`` remains for debugging or legacy comparisons.
"""

from __future__ import annotations

import json
import os
import time
import types
from typing import List, Literal, Optional

import torch
import torch.nn as nn

from nemo.core.classes.mixins import AccessMixin

InitStrategy = Literal["last_layer", "uniform", "edges"]

_AGENT_FUSION_ENTRY_LOGGED = False
_AGENT_FUSION_REG_LOGGED = False


class LearnableLayerFusion(nn.Module):
    """Softmax over L scalars; weighted sum of L tensors shaped (B, D, T)."""

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


def _layer_tensor_from_module_output(out) -> torch.Tensor:
    return out[0] if isinstance(out, tuple) else out


def _encoder_out_proj_module(encoder: nn.Module) -> Optional[nn.Module]:
    """``out_proj`` may live on a wrapped Conformer (e.g. adapter encoder)."""
    for mod in (encoder, getattr(encoder, "encoder", None)):
        if mod is None:
            continue
        op = getattr(mod, "out_proj", None)
        if isinstance(op, nn.Module):
            return op
    return None


def _conformer_hook_to_fusion_bdt(
    encoder: nn.Module, hook_out: torch.Tensor, encoded_ref: torch.Tensor
) -> torch.Tensor:
    """
    Match NeMo ``ConformerEncoder`` interctc registration: layer output is ``(B, T, d_in)``,
    optional ``out_proj`` on that layout, then ``(B, d_out, T)`` for fusion / joint.
    """
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


def _encoder_forward_with_layer_hooks(
    encoder: nn.Module,
    audio_signal: torch.Tensor,
    length: torch.Tensor,
    num_layers: int,
):
    """Run encoder forward; collect each ``encoder.layers[i]`` output via forward hooks."""
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
            f"Layer fusion hooks captured {len(layer_outs)} outputs, expected {num_layers} "
            f"(len(encoder.layers)). Non-standard encoder layout may need a different hook target."
        )
    return encoded, encoded_len, layer_outs


def _collect_interctc_outputs(encoder: nn.Module, num_layers: int) -> List[torch.Tensor]:
    # InterCTC tensors are registered on ConformerEncoder during forward_internal.
    # AccessMixin.get_module_registry(encoder) merges *every* submodule registry; adapter
    # layers can repeat the same interctc/* keys → false "duplicate" errors. Prefer the
    # encoder's own _registry, then fill gaps with first-wins across submodules.
    total_registry: dict = {}
    enc_reg = getattr(encoder, "_registry", None)
    if isinstance(enc_reg, dict):
        for key, val in enc_reg.items():
            if isinstance(key, str) and key.startswith("interctc/"):
                total_registry[key] = val

    n_out_keys = sum(1 for k in total_registry if k.startswith("interctc/layer_output_"))
    if n_out_keys < num_layers:
        for module_registry in AccessMixin.get_module_registry(encoder).values():
            for key, val in module_registry.items():
                if not (isinstance(key, str) and key.startswith("interctc/")):
                    continue
                if key not in total_registry:
                    total_registry[key] = val

    out: List[torch.Tensor] = []
    for i in range(num_layers):
        key = f"interctc/layer_output_{i}"
        tensors = total_registry.get(key)
        if tensors is None:
            raise RuntimeError(
                f"Missing {key}. Encoder may not support InterCTC capture "
                f"(e.g. unsupported encoder type). Registry keys: {sorted(total_registry)}"
            )
        if len(tensors) != 1:
            raise RuntimeError(f"Expected one tensor for {key}, got {len(tensors)}")
        out.append(tensors[0])
    return out


def attach_layer_fusion_to_model(
    model: nn.Module,
    *,
    init_strategy: InitStrategy = "last_layer",
    module_name: str = "layer_fusion",
) -> LearnableLayerFusion:
    """
    Register ``LearnableLayerFusion`` and replace ``model.forward`` so the encoder
    branch uses a fused multi-layer representation.

    The model must be an EncDecRNNT-style ASR model with ``preprocessor``,
    ``spec_augmentation``, ``encoder``, and ``model_guid`` (NeMo default).

    If ``model`` already has ``module_name`` (e.g. after ``restore_from`` a Stage 1
    checkpoint), only ``forward`` is rebound so loaded weights are kept.
    """
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
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually "
                "exclusive with ``processed_signal`` and ``processed_signal_length``."
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

        global _AGENT_FUSION_ENTRY_LOGGED, _AGENT_FUSION_REG_LOGGED

        guid = getattr(self, "model_guid", None)
        fusion_mod: LearnableLayerFusion = getattr(self, module_name)

        # #region agent log
        def _agent_dbg(hypothesis_id: str, location: str, message: str, data: dict) -> None:
            if os.environ.get("AGENT_DEBUG_LOG", "1").strip().lower() in ("0", "false", "no"):
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

        if not _AGENT_FUSION_ENTRY_LOGGED:
            _agent_dbg(
                "H1",
                "nemo_layer_fusion:_fused_forward:entry",
                "before encoder (hook capture)",
                {
                    "torch_is_inference_mode": torch.is_inference_mode_enabled(),
                    "grad_enabled": torch.is_grad_enabled(),
                    "self_training": bool(self.training),
                    "model_guid_is_none": guid is None,
                },
            )
            _AGENT_FUSION_ENTRY_LOGGED = True

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
            if not _AGENT_FUSION_REG_LOGGED:
                _reg = getattr(self.encoder, "_registry", None)
                _ik = (
                    [k for k in _reg if str(k).startswith("interctc/")]
                    if isinstance(_reg, dict)
                    else []
                )
                _agent_dbg(
                    "H7",
                    "nemo_layer_fusion:after_encoder",
                    "hook layer capture (+ registry snapshot)",
                    {
                        "n_hook_layers": len(layer_tensors),
                        "expected_layers": fusion_mod.num_layers,
                        "n_interctc_keys": len(_ik),
                    },
                )
                _AGENT_FUSION_REG_LOGGED = True
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
