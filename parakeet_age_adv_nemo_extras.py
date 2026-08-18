"""
NeMo Parakeet age-adversarial Kaggle extras: chained bottleneck adapters, waveform aug, noise helpers.

Used by train_parakeet_age_adversarial.py when USE_CHAINED_ADAPTER=1 (notebook-style PEFT + aug).
"""
from __future__ import annotations

import math
import os
import random
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil, LinearAdapter


# -----------------------------------------------------------------------------
# Adapter chain
# -----------------------------------------------------------------------------


class AdapterChainState:
    def __init__(self) -> None:
        self.prev_bottleneck: Optional[torch.Tensor] = None
        self.layer_counter = 0

    def reset(self) -> None:
        self.prev_bottleneck = None
        self.layer_counter = 0

    def update(self, bottleneck: torch.Tensor) -> None:
        self.prev_bottleneck = bottleneck
        self.layer_counter += 1


class ChainedLinearAdapter(nn.Module, AdapterModuleUtil):
    """
    Bottleneck adapter for NeMo encoder layers. Initialized so the branch output is zero before
    training: with the encoder's residual ``x + adapter(x)`` convention, the net acts as identity
    at step zero (up/down/chain_proj zeros; LayerNorm of zeros stays zero).
    """

    def __init__(
        self,
        in_features: int,
        dim: int,
        activation: str = "gelu",
        norm_position: str = "post",
        dropout: float = 0.1,
        is_first: bool = False,
        chain_state_ref: Optional[AdapterChainState] = None,
        adapter_strategy: Any = None,
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
            nn.init.zeros_(self.chain_proj.weight)
        # Identity at init: zero bottleneck contribution so residual path is unchanged.
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.setup_adapter_strategy(adapter_strategy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.dropout_layer(h_norm)


# -----------------------------------------------------------------------------
# Waveform augmentation
# -----------------------------------------------------------------------------


class WaveformAugmentor(nn.Module):
    def __init__(
        self,
        speed_min: float = 0.85,
        speed_max: float = 1.15,
        pitch_min: float = -2.0,
        pitch_max: float = 2.0,
        sample_rate: int = 16000,
        speed_prob: float = 0.5,
        pitch_prob: float = 0.5,
        external_noise_prob: float = 0.6,
        external_snr_min: float = 1.0,
        external_snr_max: float = 7.0,
        noise_file_paths: Optional[Sequence[str]] = None,
        gain_prob: float = 0.3,
        gain_db_min: float = -8.0,
        gain_db_max: float = 8.0,
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
    def _mix_at_snr(speech: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
        eps = 1e-8
        alpha = torch.sqrt(
            speech.pow(2).mean().clamp_min(eps)
            / (10 ** (snr_db / 10.0) * noise.pow(2).mean().clamp_min(eps) + eps)
        )
        return speech + alpha * noise

    def _load_noise_segment(self, num_samples: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
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
            wav = torchaudio.functional.resample(wav, int(sr), self.sample_rate)
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
    def forward(
        self, audio_signal: torch.Tensor, signal_length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return audio_signal, signal_length
        b, t = audio_signal.shape
        if torch.rand(1).item() < self.speed_prob:
            sf = self.speed_min + torch.rand(1).item() * (self.speed_max - self.speed_min)
            new_t = max(1, int(t / sf))
            audio_signal = F.interpolate(
                audio_signal.unsqueeze(1), size=new_t, mode="linear", align_corners=False
            ).squeeze(1)
            signal_length = (signal_length.float() / sf).long().clamp(min=1, max=new_t)
        if torch.rand(1).item() < self.pitch_prob:
            n_steps = self.pitch_min + torch.rand(1).item() * (self.pitch_max - self.pitch_min)
            try:
                audio_signal = torchaudio.functional.pitch_shift(
                    audio_signal, self.sample_rate, float(n_steps)
                )
            except Exception:
                pass
        if self.noise_file_paths and torch.rand(1).item() < self.external_noise_prob:
            for i in range(b):
                L = int(min(signal_length[i].item(), audio_signal.shape[1]))
                if L < 1:
                    continue
                seg = self._load_noise_segment(L, audio_signal.device, audio_signal.dtype)
                if seg is None:
                    continue
                snr = self.external_snr_min + torch.rand(1).item() * (
                    self.external_snr_max - self.external_snr_min
                )
                audio_signal[i, :L] = self._mix_at_snr(
                    audio_signal[i, :L].float(), seg, float(snr)
                ).to(dtype=audio_signal.dtype)
        if torch.rand(1).item() < self.gain_prob:
            gain_db = self.gain_db_min + torch.rand(1).item() * (self.gain_db_max - self.gain_db_min)
            audio_signal = audio_signal * (10 ** (gain_db / 20.0))
        return audio_signal, signal_length


def collect_noise_audio_files(dir_list: Sequence[str]) -> List[str]:
    exts = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
    out: List[str] = []
    for d in dir_list:
        d = (d or "").strip()
        if not d or not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(exts):
                    out.append(os.path.join(root, f))
    return out


def parse_noise_dirs_from_env() -> List[str]:
    raw = os.environ.get("CLASSROOM_NOISE_DIRS", "").strip()
    if raw:
        return [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
    d1 = os.environ.get("CLASSROOM_NOISE_DIR_1", "").strip()
    d2 = os.environ.get("CLASSROOM_NOISE_DIR_2", "").strip()
    return [d for d in (d1, d2) if d]


NOISE_CURRICULUM = os.environ.get("NOISE_CURRICULUM", "1").strip().lower() in ("1", "true", "yes")
NOISE_CURR_E1_2 = (0.3, 5.0, 20.0)
NOISE_CURR_E3_4 = (0.4, 5.0, 15.0)
NOISE_CURR_E5 = (0.45, 3.0, 12.0)


class NoiseCurriculumCallback(Callback):
    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        if not self.enabled:
            return
        wa = getattr(pl_module, "waveform_augmentor", None)
        if wa is None:
            return
        human = int(trainer.current_epoch) + 1
        if human <= 2:
            p, lo, hi, label = *NOISE_CURR_E1_2, "1-2"
        elif human <= 4:
            p, lo, hi, label = *NOISE_CURR_E3_4, "3-4"
        else:
            p, lo, hi, label = *NOISE_CURR_E5, "5+"
        wa.external_noise_prob = p
        wa.external_snr_min = lo
        wa.external_snr_max = hi
        if getattr(trainer, "global_rank", 0) == 0:
            print(
                f"[NoiseCurriculum] human_epoch={human} phase={label} noise_prob={p} SNR=[{lo},{hi}]",
                flush=True,
            )


# -----------------------------------------------------------------------------
# Install chained adapters + aug on model
# -----------------------------------------------------------------------------


def spec_augment_cfg_dict() -> Dict[str, Any]:
    return {
        "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
        "freq_masks": 2,
        "freq_width": 27,
        "time_masks": 10,
        "time_width": 0.05,
    }


def replace_linear_adapters_with_chained(
    model: ASRModel,
    chain_state: AdapterChainState,
    in_features: int,
    bottleneck_dim: int,
    adapter_short_name: str = "asr_children_adapter",
) -> int:
    LinearAdapter.register(ChainedLinearAdapter)
    device = next(model.parameters()).device
    adapter_module_list: List[Tuple[str, LinearAdapter]] = []
    for mod_name, module in list(model.named_modules()):
        if isinstance(module, LinearAdapter) and adapter_short_name in mod_name:
            adapter_module_list.append((mod_name, module))
    adapter_module_list.sort(key=lambda x: x[0])
    for i, (mod_name, _orig) in enumerate(adapter_module_list):
        new_adapter = ChainedLinearAdapter(
            in_features=in_features,
            dim=bottleneck_dim,
            activation="gelu",
            norm_position="post",
            dropout=0.1,
            is_first=(i == 0),
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
    return len(adapter_module_list)


def attach_chained_adapter_stack(
    model: ASRModel,
    bottleneck_dim: int,
    in_features: int = 1024,
    adapter_full_name: str = "encoder:asr_children_adapter",
    adapter_short_name: str = "asr_children_adapter",
) -> AdapterChainState:
    linear_cfg = OmegaConf.create(
        {
            "_target_": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
            "in_features": in_features,
            "dim": bottleneck_dim,
            "activation": "gelu",
            "norm_position": "post",
            "dropout": 0.1,
        }
    )
    model.add_adapter(name=adapter_full_name, cfg=linear_cfg)
    model.set_enabled_adapters(enabled=False)
    model.set_enabled_adapters(name=adapter_full_name, enabled=True)
    model.freeze()
    model.unfreeze_enabled_adapters()
    chain_state = AdapterChainState()
    model.adapter_chain_state = chain_state
    n = replace_linear_adapters_with_chained(
        model, chain_state, in_features, bottleneck_dim, adapter_short_name
    )
    print(f"[age_adv] Replaced {n} LinearAdapter modules with ChainedLinearAdapter (dim={bottleneck_dim})")
    return chain_state


def patch_forward_waveform_and_chain_reset(model: ASRModel, waveform_aug: nn.Module) -> None:
    _orig = model.forward.__func__

    def _augmented_forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
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
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
        )

    model.add_module("waveform_augmentor", waveform_aug)
    model.forward = types.MethodType(_augmented_forward, model)


def apply_spec_augmentation(model: ASRModel) -> None:
    model.spec_augmentation = model.from_config_dict(OmegaConf.create(spec_augment_cfg_dict()))


def partial_unfreeze_joint_decoder_lstm(model: ASRModel) -> None:
    try:
        if hasattr(model, "joint"):
            for p in model.joint.parameters():
                p.requires_grad = True
    except Exception as e:
        print(f"[age_adv] WARN joint unfreeze: {e}")
    try:
        if hasattr(model, "decoder") and hasattr(model.decoder, "prediction"):
            pred = model.decoder.prediction
            if hasattr(pred, "embed"):
                for p in pred.embed.parameters():
                    p.requires_grad = True
    except Exception as e:
        print(f"[age_adv] WARN decoder embed: {e}")

    def _unfreeze_last_lstm(container: Optional[nn.Module]) -> bool:
        if container is None:
            return False
        ok = False
        for _, mod in container.named_modules():
            if isinstance(mod, nn.LSTM) and mod.num_layers >= 1:
                last_i = mod.num_layers - 1
                for pname, param in mod.named_parameters():
                    if f"_l{last_i}" in pname:
                        param.requires_grad = True
                        ok = True
        return ok

    try:
        if hasattr(model, "decoder") and hasattr(model.decoder, "prediction"):
            pred = model.decoder.prediction
            if not _unfreeze_last_lstm(pred) and hasattr(pred, "dec_rnn"):
                _unfreeze_last_lstm(pred.dec_rnn)
    except Exception as e:
        print(f"[age_adv] WARN decoder LSTM: {e}")

    for m in (
        getattr(model, "decoder", None),
        getattr(model, "joint", None),
        getattr(model, "waveform_augmentor", None),
        getattr(model, "spec_augmentation", None),
    ):
        if m is not None:
            m.train()


def build_discriminative_configure_optimizers(
    model: ASRModel,
    age_disc: nn.Module,
    num_epochs: int,
    lr_adapter: float = 5e-4,
    lr_joint: float = 1e-4,
    lr_decoder: float = 5e-5,
    lr_age_disc: float = 1e-3,
    warmup_ratio: float = 0.15,
) -> Any:
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    disc_ids = {id(p) for p in age_disc.parameters()}
    adapter_params: List[torch.nn.Parameter] = []
    joint_params: List[torch.nn.Parameter] = []
    decoder_params: List[torch.nn.Parameter] = []
    age_disc_params: List[torch.nn.Parameter] = []

    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in disc_ids:
            age_disc_params.append(param)
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

    groups = []
    if age_disc_params:
        groups.append({"params": age_disc_params, "lr": lr_age_disc})
    if adapter_params:
        groups.append({"params": adapter_params, "lr": lr_adapter})
    if joint_params:
        groups.append({"params": joint_params, "lr": lr_joint})
    if decoder_params:
        groups.append({"params": decoder_params, "lr": lr_decoder})

    if not groups:
        raise RuntimeError("No trainable parameters for discriminative optimizer")

    try:
        num_batches = len(model._train_dl)
    except (TypeError, AttributeError):
        num_batches = 1000
    total_steps = max(1, num_epochs * num_batches)
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    optimizer = AdamW(groups, betas=(0.9, 0.999), weight_decay=0.01)

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, _lr_lambda)

    def _custom_configure_optimizers(_self) -> Dict[str, Any]:
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    return types.MethodType(_custom_configure_optimizers, model)


def rewire_adapter_chain_state(model: ASRModel) -> AdapterChainState:
    """After ``restore_from``, shared ``AdapterChainState`` may be missing; reattach for ChainedLinearAdapter."""
    state = AdapterChainState()
    model.adapter_chain_state = state
    n = 0
    for mod in model.modules():
        if isinstance(mod, ChainedLinearAdapter):
            mod.chain_state_ref = state
            n += 1
    print(f"[age_adv] Rewired chain_state_ref on {n} ChainedLinearAdapter module(s)", flush=True)
    return state


def stage2_prepare_trainable(model: ASRModel, adapter_full_name: str = "encoder:asr_children_adapter") -> None:
    """
    Full fine-tune mask after Stage 1 (frozen encoder + adapters): encoder backbone, adapters, full joint,
    partial decoder (embed + last LSTM per ``partial_unfreeze_joint_decoder_lstm``), and age head.
    Preprocessor weights stay frozen.
    """
    model.set_enabled_adapters(enabled=False)
    model.set_enabled_adapters(name=adapter_full_name, enabled=True)
    model.freeze()
    model.unfreeze_enabled_adapters()

    enc = getattr(model, "encoder", None)
    if enc is not None:
        for name, p in enc.named_parameters():
            nl = name.lower()
            if ".adapter" in nl or "adapter_layer" in nl:
                continue
            p.requires_grad = True

    partial_unfreeze_joint_decoder_lstm(model)

    head = getattr(model, "parakeet_age_adv", None)
    if head is not None:
        for p in head.parameters():
            p.requires_grad = True

    pre = getattr(model, "preprocessor", None)
    if pre is not None:
        for p in pre.parameters():
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[age_adv] Stage2 trainable: {trainable:,} / {total:,} ({100.0 * trainable / max(1, total):.2f}%)",
        flush=True,
    )


def build_stage2_discriminative_configure_optimizers(
    model: ASRModel,
    age_disc: nn.Module,
    num_epochs: int,
    lr_encoder: float = 5e-6,
    lr_adapter: float = 3e-4,
    lr_joint: float = 1e-4,
    lr_decoder: float = 5e-5,
    lr_age_disc: float = 5e-4,
    warmup_ratio: float = 0.15,
) -> Any:
    """
    Five LR groups: encoder backbone (lowest), bottleneck adapters, joint, partial decoder, age discriminator.
    """
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    disc_ids = {id(p) for p in age_disc.parameters()}
    encoder_p: List[nn.Parameter] = []
    adapter_p: List[nn.Parameter] = []
    joint_p: List[nn.Parameter] = []
    decoder_p: List[nn.Parameter] = []
    age_disc_p: List[nn.Parameter] = []
    other_p: List[nn.Parameter] = []

    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in disc_ids:
            age_disc_p.append(param)
            continue
        nl = pname.lower()
        if ".adapter" in nl or "adapter_layer" in nl:
            adapter_p.append(param)
            continue
        if "joint" in nl:
            joint_p.append(param)
            continue
        if "decoder" in nl or "prediction" in nl:
            decoder_p.append(param)
            continue
        if "encoder" in nl or nl.startswith("encoder."):
            encoder_p.append(param)
            continue
        other_p.append(param)

    if other_p:
        op_ids = {id(p) for p in other_p}
        extra_names = [n for n, p in model.named_parameters() if id(p) in op_ids][:12]
        print(
            f"[age_adv] Stage2 optimizer: {len(other_p)} tensor(s) in 'other' group (lr=lr_joint). "
            f"Examples: {extra_names}",
            flush=True,
        )

    groups: List[Dict[str, Any]] = []
    if age_disc_p:
        groups.append({"params": age_disc_p, "lr": lr_age_disc})
    if encoder_p:
        groups.append({"params": encoder_p, "lr": lr_encoder})
    if adapter_p:
        groups.append({"params": adapter_p, "lr": lr_adapter})
    if joint_p:
        groups.append({"params": joint_p, "lr": lr_joint})
    if decoder_p:
        groups.append({"params": decoder_p, "lr": lr_decoder})
    if other_p:
        groups.append({"params": other_p, "lr": lr_joint})

    if not groups:
        raise RuntimeError("No trainable parameters for stage-2 optimizer")

    try:
        num_batches = len(model._train_dl)
    except (TypeError, AttributeError):
        num_batches = 1000
    total_steps = max(1, num_epochs * num_batches)
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    optimizer = AdamW(groups, betas=(0.9, 0.999), weight_decay=0.01)

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, _lr_lambda)

    def _custom_configure_optimizers(_self) -> Dict[str, Any]:
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    return types.MethodType(_custom_configure_optimizers, model)


class ChainedAdapterTrainDebugCallback(Callback):
    """Log chained adapter forwards + grads for first N batches (rank 0)."""

    def __init__(self, num_steps: int = 0):
        super().__init__()
        self.num_steps = int(num_steps)
        self._batch_ctr = 0
        self._handles: List[Any] = []
        self._adapters_ordered: List[Tuple[str, ChainedLinearAdapter]] = []
        self._fwd_records: List[dict] = []
        self._grad_lines: List[str] = []

    @staticmethod
    def _gmax(t: Optional[torch.Tensor]) -> float:
        if t is None or t.grad is None:
            return float("nan")
        return float(t.grad.detach().float().abs().max().item())

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        if getattr(trainer, "global_rank", 0) != 0 or self.num_steps <= 0:
            return
        self._adapters_ordered = [
            (n, m)
            for n, m in pl_module.named_modules()
            if isinstance(m, ChainedLinearAdapter)
        ]
        self._adapters_ordered.sort(key=lambda x: x[0])

        def _make_hook(idx: int, name: str):
            def _hook(mod: Any, inp: Any, out: Any) -> None:
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
            f"[ChainedAdapterTrainDebug] hooks on {len(self._adapters_ordered)} adapters; "
            f"first {self.num_steps} batches",
            flush=True,
        )

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def on_train_batch_start(self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int) -> None:
        if getattr(trainer, "global_rank", 0) != 0 or self._batch_ctr >= self.num_steps:
            return
        self._fwd_records.clear()

    def on_after_backward(self, trainer: Any, pl_module: Any) -> None:
        if getattr(trainer, "global_rank", 0) != 0 or self._batch_ctr >= self.num_steps:
            return
        lines = []
        for i, (name, mod) in enumerate(self._adapters_ordered):
            gd = self._gmax(mod.down.weight)
            gu = self._gmax(mod.up.weight)
            cp = getattr(mod, "chain_proj", None)
            gc = self._gmax(cp.weight) if cp is not None else float("nan")
            lines.append(f"    [{i:02d}] down={gd:.3e} up={gu:.3e} chain={gc:.3e}  {name}")
        self._grad_lines = lines

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if getattr(trainer, "global_rank", 0) != 0 or self._batch_ctr >= self.num_steps:
            return
        recs = sorted(self._fwd_records, key=lambda r: r["idx"])
        print(f"\n{'=' * 80}", flush=True)
        print(
            f"[ChainedAdapterTrainDebug] batch={self._batch_ctr} epoch={trainer.current_epoch}",
            flush=True,
        )
        if isinstance(outputs, dict):
            for k in ("loss", "train_loss"):
                if k in outputs and outputs[k] is not None and torch.is_tensor(outputs[k]):
                    print(f"  {k}={outputs[k].detach().float().item():.6f}", flush=True)
                    break
        for r in recs:
            print(
                f"    [{r['idx']:02d}] first={r['is_first']} in={r['in']} out={r['out']}  {r['name']}",
                flush=True,
            )
        for line in self._grad_lines:
            print(line, flush=True)
        self._grad_lines = []
        print(f"{'=' * 80}\n", flush=True)
        self._batch_ctr += 1
