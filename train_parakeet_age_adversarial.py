#!/usr/bin/env python3
"""
Adversarial age-invariant ASR training for NeMo Parakeet TDT (paper-style).

Scalar loss (implementation convenience; one Lightning backward):

    L_impl = L_asr + λ_adv * L_adv + s_disc * L_disc

``s_disc`` (env ``DISC_LOSS_SCALE``, alias ``LAMBDA_AGE``) scales the **discriminator-only** supervised
BCE term; it is not an ASR-vs-side-task tradeoff like ``λ_adv``.

Parameter-isolated routing (approximates split optimization: discriminator on L_disc;
encoder + ASR on L_asr + λ_adv * L_adv). Intended zero partials:

    ∂L_disc / ∂θ_enc = 0 ,  ∂L_disc / ∂θ_asr = 0   — z_sup = disc(f.detach()), logits + BCEWithLogits
    ∂L_adv / ∂θ_disc = 0                         — adversarial branch: disc_forward_detached_weights(f)
    ∂L_asr / ∂θ_disc = 0                         — RNNT path has no discriminator

Gradients flow where intended: L_disc → θ_disc only; L_adv → θ_enc (via f); L_asr → θ_enc and θ_asr.

Losses: ``L_disc`` / ``L_adv`` use ``binary_cross_entropy_with_logits`` on discriminator logits vs soft
targets (age target a_i and 0.5); ``L_asr`` = NeMo RNNT/TDT (not CTC).

Age targets (paper-style): youngest child in span → 0; oldest child in corpus span → 0.8 (linear in
between); any age above that span → 1.0 (no adolescent ramp). Knots: ``AGE_CHILD_YOUNGEST``,
``AGE_CHILD_OLDEST_CORPUS`` (defaults 2–12 for a 2–12y child set mixed with Libri adults).

Manifest JSONL: audio_filepath, text, duration; age_years (recommended) or precomputed age_target.
Optional ``age_cohort`` (``child`` / ``adult``) or ``is_child`` (bool) enables **balanced batches** when
``BALANCED_CHILD_ADULT_BATCHES=1``: each batch is half child-cohort and half adult-cohort (oversamples
the smaller group within an epoch). Build LibriSpeech manifests with ``build_librispeech_age_manifest.py``.

**Mixed child + Libri in one run:** comma-separate paths in ``TRAIN_MANIFEST`` (or ``--manifest``) and set
``TRAIN_MANIFEST_COHORTS`` to matching labels, e.g.
``TRAIN_MANIFEST=/path/child.jsonl,/path/libri_clean_age.jsonl`` and ``TRAIN_MANIFEST_COHORTS=child,adult``.
The trainer writes ``SAVE_DIR/merged_train_manifest.jsonl`` (override with ``MERGED_TRAIN_MANIFEST_PATH``).
Alternatively pre-merge offline: ``merge_child_adult_training_manifests.py``.

Run: ``python train_parakeet_age_adversarial.py --manifest /path/to/train.jsonl`` (or set TRAIN_MANIFEST).

**Kaggle / notebook-style PEFT:** set ``USE_CHAINED_ADAPTER=1`` for chained bottleneck adapters (env
``BOTTLENECK_DIM``), waveform + SpecAugment, optional noise dirs (``CLASSROOM_NOISE_DIRS``), noise
curriculum, partial unfreeze (joint + decoder embed + last LSTM), and discriminative AdamW including
a dedicated ``age_disc`` LR (``LR_AGE_DISC``, etc.). Use ``kaggle_run_parakeet_age_adv.py`` for pip
clean + subprocess launch.

For **encoder + full joint + partial decoder** fine-tuning after a frozen-encoder Stage-1 ``.nemo``, use
``train_parakeet_age_adv_stage2.py`` (``INIT_NEMO``, discriminative ``LR_ENCODER``, same aug/noise hooks).

Env: SAVE_DIR, BATCH_SIZE, NUM_EPOCHS, USE_CHAINED_ADAPTER, BOTTLENECK_DIM, CLASSROOM_NOISE_DIRS,
NOISE_CURRICULUM, TRAIN_DEBUG_STEPS, LAMBDA_ADV (λ_max), LAMBDA_ADV_SCHEDULE, LAMBDA_ADV_RAMP_*,
DISC_LOSS_SCALE, LR_AGE_DISC / LR_ADAPTER / LR_JOINT / LR_DECODER, AGE_CHILD_* , WER_PROBE_N,
BALANCED_CHILD_ADULT_BATCHES, TRAIN_MANIFEST (comma-separated + TRAIN_MANIFEST_COHORTS for merge),
MERGED_TRAIN_MANIFEST_PATH, ...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import types
import warnings
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=Warning, module="numba")
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

import jiwer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.data.audio_to_text import AudioToBPEDataset, _speech_collate_fn
from nemo.collections.asr.models import ASRModel
from nemo.core.classes import adapter_mixins
from nemo.core.classes.mixins import AccessMixin
from nemo.utils import logging as nemo_logging

# -----------------------------------------------------------------------------
# Paper-style age target: youngest child → 0, oldest child in span → 0.8, else → 1.0 (no 0.8→1 ramp)
# -----------------------------------------------------------------------------


def paper_age_target(age_years: float, child_youngest: float, child_oldest: float) -> float:
    """
    Soft label in [0, 1] for BCEWithLogits (targets need not be binary).

    - At or below ``child_youngest`` -> 0; at ``child_oldest`` -> 0.8; linear between.
    - Strictly above ``child_oldest`` -> 1.0 (adults and any non-child ages).
    """
    a = float(age_years)
    if a <= child_youngest:
        return 0.0
    if a <= child_oldest:
        if child_oldest <= child_youngest:
            return 0.8
        t = (a - child_youngest) / (child_oldest - child_youngest)
        return float(0.8 * t)
    return 1.0


def scheduled_lambda_adv(
    current_epoch_zero_based: int,
    lambda_max: float,
    ramp_first_epoch_1based: int = 10,
    ramp_last_epoch_1based: int = 40,
) -> float:
    """
    Piecewise λ_adv(e) with e = trainer epoch **1-based** (first epoch has e=1).

    λ_adv = 0 for e < ramp_first; linear from 0 to λ_max for e in [ramp_first, ramp_last];
    λ_max for e > ramp_last. Matches: 0 for e<10, ramp 10–40, full after 40 when defaults are used.
    """
    e = int(current_epoch_zero_based) + 1
    rf = int(ramp_first_epoch_1based)
    rl = int(ramp_last_epoch_1based)
    if e < rf:
        return 0.0
    if e > rl:
        return float(lambda_max)
    denom = float(rl - rf)
    if denom <= 0.0:
        return float(lambda_max)
    return float(lambda_max) * float(e - rf) / denom


def build_age_lookup_from_manifests(
    manifest_filepaths: List[str],
    child_youngest: float,
    child_oldest: float,
    use_precomputed_target: bool = False,
) -> Dict[str, float]:
    """Map normalized absolute audio path -> soft target in [0,1]."""
    out: Dict[str, float] = {}
    for mf in manifest_filepaths:
        if not mf or not os.path.isfile(mf):
            continue
        with open(mf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ap = row.get("audio_filepath") or row.get("audio_file") or row.get("audio_filename")
                if not ap:
                    continue
                abspath = os.path.normpath(os.path.abspath(os.path.expanduser(ap)))
                if use_precomputed_target and "age_target" in row:
                    t = float(row["age_target"])
                elif "age_years" in row:
                    t = paper_age_target(float(row["age_years"]), child_youngest, child_oldest)
                elif "age" in row:
                    # If manifest stores already-normalized soft label in [0,1]
                    t = float(row["age"])
                else:
                    continue
                t = float(min(1.0, max(0.0, t)))
                out[abspath] = t
                out[os.path.basename(abspath)] = t
    return out


def build_cohort_lookup_from_manifests(manifest_filepaths: List[str]) -> Optional[Dict[str, bool]]:
    """
    Map normalized absolute path -> True if child cohort, False if adult, for balanced batching.
    Returns None if no row in any manifest defines a cohort field.
    """
    out: Dict[str, bool] = {}
    any_cohort = False
    for mf in manifest_filepaths:
        if not mf or not os.path.isfile(mf):
            continue
        with open(mf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ap = row.get("audio_filepath") or row.get("audio_file") or row.get("audio_filename")
                if not ap:
                    continue
                is_child: Optional[bool] = None
                if "age_cohort" in row:
                    any_cohort = True
                    v = str(row["age_cohort"]).strip().lower()
                    is_child = v == "child"
                elif "balance_cohort" in row:
                    any_cohort = True
                    v = str(row["balance_cohort"]).strip().lower()
                    is_child = v == "child"
                elif "is_child" in row:
                    any_cohort = True
                    is_child = bool(row["is_child"])
                if is_child is None:
                    continue
                abspath = os.path.normpath(os.path.abspath(os.path.expanduser(ap)))
                out[abspath] = is_child
                out[os.path.basename(abspath)] = is_child
    return out if any_cohort else None


# -----------------------------------------------------------------------------
# Balanced child / adult batches
# -----------------------------------------------------------------------------


class BalancedTwoGroupBatchSampler(torch.utils.data.Sampler[List[int]]):
    """
    Yields batches of indices with n_child = floor(batch_size/2) from the child pool and the rest
    from the adult pool. Pads the shorter pool by cycling so every epoch has the same number of
    batches: max(ceil(|C|/n_c), ceil(|A|/n_a)).
    """

    def __init__(
        self,
        child_indices: List[int],
        adult_indices: List[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if batch_size < 2:
            raise ValueError("BalancedTwoGroupBatchSampler needs batch_size >= 2")
        if not child_indices or not adult_indices:
            raise ValueError("BalancedTwoGroupBatchSampler needs non-empty child and adult index lists")
        self.child_indices = list(child_indices)
        self.adult_indices = list(adult_indices)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _num_batches(self) -> int:
        n_c = self.batch_size // 2
        n_a = self.batch_size - n_c
        n_batches = max(
            (len(self.child_indices) + n_c - 1) // n_c,
            (len(self.adult_indices) + n_a - 1) // n_a,
        )
        if self.drop_last:
            cap = (len(self.child_indices) + len(self.adult_indices)) // self.batch_size
            return min(n_batches, cap)
        return n_batches

    def __len__(self) -> int:
        return self._num_batches()

    def __iter__(self):
        n_c = self.batch_size // 2
        n_a = self.batch_size - n_c
        rng = random.Random(self.seed + self.epoch)
        child_order = list(self.child_indices)
        adult_order = list(self.adult_indices)
        if self.shuffle:
            rng.shuffle(child_order)
            rng.shuffle(adult_order)
        n_batches = self._num_batches()
        ic = ia = 0
        for _ in range(n_batches):
            batch: List[int] = []
            for _ in range(n_c):
                batch.append(child_order[ic % len(child_order)])
                ic += 1
            for _ in range(n_a):
                batch.append(adult_order[ia % len(adult_order)])
                ia += 1
            if self.shuffle:
                rng.shuffle(batch)
            yield batch


class BalancedBatchEpochCallback(Callback):
    """Advance BalancedTwoGroupBatchSampler epoch for reproducible shuffles each training epoch."""

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: Any) -> None:
        dl = getattr(pl_module, "_train_dl", None)
        if dl is None:
            return
        bs = getattr(dl, "batch_sampler", None)
        if bs is not None and hasattr(bs, "set_epoch"):
            bs.set_epoch(int(trainer.current_epoch))


# -----------------------------------------------------------------------------
# f0 normalization hook (integrate paper-specific DSP here)
# -----------------------------------------------------------------------------

_F0_LOGGED = False


def apply_f0_normalization(audio: torch.Tensor, sr: int, enabled: bool) -> torch.Tensor:
    """
    Time-domain waveform [B,T] or [T]. Placeholder: returns input.
    Replace with paper f0 estimation + warping (e.g. librosa/pyworld) when enabled.
    """
    global _F0_LOGGED
    if not enabled:
        return audio
    if not _F0_LOGGED:
        nemo_logging.warning(
            "apply_f0_normalization: enabled but using identity — implement paper f0 warping here."
        )
        _F0_LOGGED = True
    return audio


# -----------------------------------------------------------------------------
# Collate with age
# -----------------------------------------------------------------------------


def collate_age_bpe_batch(batch, pad_id: int):
    """batch items: (sig, sig_len, tok, tok_len, age_scalar_tensor)."""
    ages = torch.stack([b[4] for b in batch]).view(-1, 1)
    core = [b[:4] for b in batch]
    sig, sl, tok, tl = _speech_collate_fn(core, pad_id)
    return sig, sl, tok, tl, ages.to(dtype=torch.float32)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class AudioToBPEDatasetWithAge(AudioToBPEDataset):
    def __init__(
        self,
        *args,
        age_by_path: Optional[Dict[str, float]] = None,
        cohort_by_path: Optional[Dict[str, bool]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.age_by_path = age_by_path or {}
        self.cohort_by_path = cohort_by_path or {}
        self._missing_age = 0
        self._missing_cohort = 0

    def _lookup_age(self, audio_file: str) -> float:
        n = os.path.normpath(audio_file)
        if n in self.age_by_path:
            return float(self.age_by_path[n])
        bn = os.path.basename(n)
        if bn in self.age_by_path:
            return float(self.age_by_path[bn])
        if self._missing_age < 5:
            nemo_logging.warning(f"No age entry for {audio_file}; using target 0.5")
            self._missing_age += 1
        return 0.5

    def _lookup_cohort_is_child(self, audio_file: str) -> Optional[bool]:
        """If cohort dict is empty, balanced batching is disabled. Unknown path -> None (omit from both lists)."""
        if not self.cohort_by_path:
            return None
        n = os.path.normpath(audio_file)
        if n in self.cohort_by_path:
            return bool(self.cohort_by_path[n])
        bn = os.path.basename(n)
        if bn in self.cohort_by_path:
            return bool(self.cohort_by_path[bn])
        return None

    def balance_index_lists(self) -> Tuple[List[int], List[int]]:
        child: List[int] = []
        adult: List[int] = []
        for i in range(len(self.manifest_processor.collection)):
            sample = self.manifest_processor.collection[i]
            v = self._lookup_cohort_is_child(sample.audio_file)
            if v is True:
                child.append(i)
            elif v is False:
                adult.append(i)
            elif self.cohort_by_path:
                if self._missing_cohort < 5:
                    nemo_logging.warning(
                        f"No cohort entry for {sample.audio_file}; treating as adult for batch balance."
                    )
                    self._missing_cohort += 1
                adult.append(i)
        return child, adult

    def _process_sample(self, index):
        sample = self.manifest_processor.collection[index]
        offset = sample.offset
        if offset is None:
            offset = 0
        features = self.featurizer.process(
            sample.audio_file,
            offset=offset,
            duration=sample.duration,
            trim=self.trim,
            orig_sr=sample.orig_sr,
            channel_selector=self.channel_selector,
        )
        f, fl = features, torch.tensor(features.shape[0]).long()
        t, tl = self.manifest_processor.process_text_by_sample(sample=sample)
        age = torch.tensor(self._lookup_age(sample.audio_file), dtype=torch.float32)
        if self.return_sample_id:
            return f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), age, index
        return f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), age

    def _collate_fn(self, batch):
        if self.return_sample_id:
            raise NotImplementedError("return_sample_id + age collate not supported in this script")
        return collate_age_bpe_batch(batch, self.manifest_processor.pad_id)


# -----------------------------------------------------------------------------
# Pooling (encoder layout B, D, T)
# -----------------------------------------------------------------------------


def masked_mean_std_pool_bdt(encoded: torch.Tensor, lengths: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """encoded: [B, D, T], lengths: [B] — returns [B, 2D] concat(mean, std) over valid frames."""
    _, d, t_max = encoded.shape
    idx = torch.arange(t_max, device=encoded.device, dtype=torch.long).unsqueeze(0)
    mask = (idx < lengths.unsqueeze(1)).float().unsqueeze(1)  # [B, 1, T]

    denom = mask.sum(dim=2).clamp(min=1.0)  # [B, 1]
    mean = (encoded * mask).sum(dim=2) / denom  # [B, D]

    var = ((encoded - mean.unsqueeze(2)) ** 2 * mask).sum(dim=2) / denom
    std = torch.sqrt(var + eps)

    return torch.cat([mean, std], dim=1)


# -----------------------------------------------------------------------------
# Discriminator
# -----------------------------------------------------------------------------


class AgeDiscriminator(nn.Module):
    def __init__(self, d_in: int, hidden1: int = 512, hidden2: int = 256, dropout: float = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(d_in, hidden1)
        self.lin2 = nn.Linear(hidden1, hidden2)
        self.lin3 = nn.Linear(hidden2, 1)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.lin1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.lin2(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin3(h)

    def forward_detached_weights(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(F.linear(x, self.lin1.weight.detach(), self.lin1.bias.detach()))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(F.linear(h, self.lin2.weight.detach(), self.lin2.bias.detach()))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.linear(h, self.lin3.weight.detach(), self.lin3.bias.detach())


class ParakeetWithAgeAdversarial(nn.Module):
    """
    Logical head for encoder features + age discriminator. The pretrained ASR body stays on the
    NeMo ``ASRModel`` instance; this module only owns ``age_disc`` so optimizers can include it
    (training also attaches a copy as ``model.age_disc`` for ``training_step``).

    Pooling is mean+std over time (``2 * encoder_dim``) so the discriminator sees temporal spread.
    """

    def __init__(
        self,
        encoder_dim: int,
        hidden1: int = 512,
        hidden2: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.age_disc = AgeDiscriminator(2 * encoder_dim, hidden1, hidden2, dropout)

    def forward(
        self,
        asr: ASRModel,
        input_signal: torch.Tensor,
        input_signal_length: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        encoded, encoded_len = asr.forward(
            input_signal=input_signal, input_signal_length=input_signal_length
        )
        f = masked_mean_std_pool_bdt(encoded, encoded_len)
        age_logits = self.age_disc(f)
        return {
            "encoded": encoded,
            "encoded_len": encoded_len,
            "features": f,
            "age_logits": age_logits,
        }


# -----------------------------------------------------------------------------
# RNNT loss (non-fused + fused), mirrors EncDecRNNTModel.training_step
# -----------------------------------------------------------------------------


def transducer_loss_from_encoded(
    model: ASRModel,
    encoded: torch.Tensor,
    encoded_len: torch.Tensor,
    transcript: torch.Tensor,
    transcript_len: torch.Tensor,
) -> torch.Tensor:
    decoder, target_length, _states = model.decoder(
        targets=transcript, target_length=transcript_len
    )
    if not model.joint.fuse_loss_wer:
        joint = model.joint(encoder_outputs=encoded, decoder_outputs=decoder)
        loss_value = model.loss(
            log_probs=joint,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=target_length,
        )
        return model.add_auxiliary_losses(loss_value)
    loss_value, _wer, _, _ = model.joint(
        encoder_outputs=encoded,
        decoder_outputs=decoder,
        encoder_lengths=encoded_len,
        transcripts=transcript,
        transcript_lengths=transcript_len,
        compute_wer=False,
    )
    return model.add_auxiliary_losses(loss_value)


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------


class ReinforceDecoderJointTrainMode(Callback):
    """Keep decoder/joint (and aug) in train mode for RNN backward + augmentation gates."""

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


class SaveEpochsAndWERProbe(Callback):
    """Save .nemo at human epochs 5,6,7 and run greedy WER on a fixed train subset."""

    def __init__(
        self,
        ckpt_dir: str,
        save_human_epochs: List[int],
        manifest_path: str,
        probe_n: int,
        seed: int = 42,
        save_adapter_weights: bool = False,
    ):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.save_human_epochs = set(save_human_epochs)
        self.manifest_path = manifest_path
        self.probe_n = probe_n
        self.seed = seed
        self.save_adapter_weights = save_adapter_weights
        self._probe_paths: Optional[List[str]] = None
        self._probe_refs: Optional[List[str]] = None
        self.best_wer = float("inf")

    def _ensure_probe_lists(self, model: ASRModel):
        if self._probe_paths is not None:
            return
        if self.probe_n <= 0:
            self._probe_paths = []
            self._probe_refs = []
            nemo_logging.info("WER probe disabled (WER_PROBE_N<=0); only epoch .nemo saves run.")
            return
        rng = random.Random(self.seed)
        rows = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        rng.shuffle(rows)
        paths, refs = [], []
        for r in rows:
            if len(paths) >= self.probe_n:
                break
            ap = r.get("audio_filepath") or r.get("audio_file")
            tx = (r.get("text") or "").strip()
            if not ap or not tx:
                continue
            ap = os.path.normpath(os.path.abspath(os.path.expanduser(ap)))
            if os.path.isfile(ap):
                paths.append(ap)
                refs.append(tx.lower())
        self._probe_paths = paths
        self._probe_refs = refs
        nemo_logging.info(f"WER probe: {len(paths)} utterances from manifest.")

    def on_train_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return
        he = trainer.current_epoch + 1
        model = pl_module

        self._ensure_probe_lists(model)
        if self._probe_paths and self._probe_refs:
            model.eval()
            try:
                preds_raw = model.transcribe(audio=self._probe_paths, batch_size=min(8, len(self._probe_paths)))
            except TypeError:
                preds_raw = model.transcribe(
                    self._probe_paths, batch_size=min(8, len(self._probe_paths))
                )
            if isinstance(preds_raw, tuple):
                preds_raw = preds_raw[0]
            hyps = []
            for h in preds_raw:
                t = getattr(h, "text", None)
                hyps.append((t or str(h)).strip().lower() if t is not None else str(h).strip().lower())
            wer = float(jiwer.wer(self._probe_refs, hyps)) if self._probe_refs else 1.0
            nemo_logging.info(f"Epoch {he} train-subset WER (n={len(self._probe_refs)}): {wer:.4f}")
            log = getattr(trainer, "logger", None)
            if log is not None and callable(getattr(log, "log_metrics", None)):
                log.log_metrics({"train_probe_wer": wer}, step=trainer.global_step)
            if wer < self.best_wer:
                self.best_wer = wer
                best_path = os.path.join(self.ckpt_dir, "best.nemo")
                model.save_to(best_path)
                nemo_logging.info(f"New best WER {wer:.4f} -> {best_path}")
            model.train()

        if he in self.save_human_epochs:
            path = os.path.join(self.ckpt_dir, f"model_epoch{he}.nemo")
            model.save_to(path)
            nemo_logging.info(f"Saved checkpoint epoch {he} -> {path}")
            if self.save_adapter_weights and hasattr(model, "save_adapters"):
                ap = os.path.join(self.ckpt_dir, f"adapter_epoch{he}.pt")
                try:
                    model.save_adapters(ap)
                    nemo_logging.info(f"Saved adapter weights -> {ap}")
                except Exception as e:
                    nemo_logging.warning(f"save_adapters epoch {he}: {e}")


# -----------------------------------------------------------------------------
# Training step factory
# -----------------------------------------------------------------------------


def make_training_step_with_age(
    lambda_adv_max: float,
    disc_loss_scale: float,
    use_f0: bool,
    use_lambda_adv_schedule: bool,
    lambda_adv_ramp_start_1based: int,
    lambda_adv_ramp_end_1based: int,
):
    def training_step(self, batch, batch_nb):
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if len(batch) != 5:
            raise ValueError(f"Expected batch of 5 tensors (with age), got {len(batch)}")
        signal, signal_len, transcript, transcript_len, age_target = batch

        sr = int(getattr(self.preprocessor, "_sample_rate", 16000))
        signal = apply_f0_normalization(signal, sr, use_f0)

        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        f = masked_mean_std_pool_bdt(encoded, encoded_len)
        z_sup = self.age_disc(f.detach())
        z_adv = self.age_disc.forward_detached_weights(f)
        tgt = age_target.to(dtype=z_sup.dtype)
        half = torch.full_like(z_adv, 0.5)
        loss_disc = F.binary_cross_entropy_with_logits(z_sup, tgt, reduction="mean")
        loss_adv = F.binary_cross_entropy_with_logits(z_adv, half, reduction="mean")
        loss_asr = transducer_loss_from_encoded(
            self, encoded, encoded_len, transcript, transcript_len
        )
        if use_lambda_adv_schedule and self.trainer is not None:
            lam_adv = scheduled_lambda_adv(
                int(self.trainer.current_epoch),
                lambda_adv_max,
                ramp_first_epoch_1based=lambda_adv_ramp_start_1based,
                ramp_last_epoch_1based=lambda_adv_ramp_end_1based,
            )
        else:
            lam_adv = float(lambda_adv_max)
        loss = loss_asr + lam_adv * loss_adv + disc_loss_scale * loss_disc

        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        try:
            opt = self.trainer.optimizers[0]
            lr = opt.param_groups[0]["lr"]
        except Exception:
            lr = 0.0

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("loss_asr", loss_asr.detach(), prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss_adv", loss_adv.detach(), prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss_disc", loss_disc.detach(), prog_bar=False, on_step=True, on_epoch=True)
        self.log("lambda_adv", float(lam_adv), prog_bar=False, on_step=True, on_epoch=True)
        self.log("lr", lr, prog_bar=False, on_step=True, on_epoch=True)

        if getattr(self, "_optim_normalize_joint_txu", False):
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {"loss": loss}

    return training_step


def patch_configure_optimizers(model: ASRModel, age_disc: nn.Module):
    """Put age_disc params first in group 0 without duplicating (they are already in model.parameters())."""
    user_conf = model.configure_optimizers

    def _wrapped():
        out = user_conf()
        if isinstance(out, dict):
            opt = out["optimizer"]
        else:
            opt = out[0] if isinstance(out, (list, tuple)) else out
        disc_params = list(age_disc.parameters())
        if not disc_params:
            return out
        disc_ids = {id(p) for p in disc_params}
        g0 = opt.param_groups[0]
        rest = [p for p in g0["params"] if id(p) not in disc_ids]
        g0["params"] = disc_params + rest
        return out

    model.configure_optimizers = types.MethodType(_wrapped, model)


def infer_encoder_dim(model: ASRModel, device: torch.device) -> int:
    sr = int(getattr(model.preprocessor, "_sample_rate", 16000))
    x = torch.zeros(1, int(sr * 0.3), device=device)
    xl = torch.tensor([x.shape[1]], device=device, dtype=torch.long)
    with torch.no_grad():
        enc, el = model.forward(input_signal=x, input_signal_length=xl)
    return int(enc.shape[1])


def build_dataloader_with_age(
    model: ASRModel,
    train_cfg: Any,
    age_map: Dict[str, float],
    cohort_map: Optional[Dict[str, bool]] = None,
    use_balanced_batches: bool = False,
    balanced_sampler_seed: int = 0,
):
    use_start = train_cfg.get("use_start_end_token", True)
    ds = AudioToBPEDatasetWithAge(
        manifest_filepath=train_cfg.manifest_filepath,
        tokenizer=model.tokenizer,
        sample_rate=train_cfg.sample_rate,
        int_values=train_cfg.get("int_values", False),
        augmentor=None,
        max_duration=train_cfg.get("max_duration", None),
        min_duration=train_cfg.get("min_duration", None),
        max_utts=train_cfg.get("max_utts", 0),
        trim=train_cfg.get("trim_silence", False),
        use_start_end_token=use_start,
        return_sample_id=False,
        channel_selector=train_cfg.get("channel_selector", None),
        age_by_path=age_map,
        cohort_by_path=cohort_map if cohort_map is not None else {},
    )

    drop_last = bool(train_cfg.get("drop_last", False))
    nw = int(train_cfg.get("num_workers", 0))
    pin = bool(train_cfg.get("pin_memory", False))
    bs = int(train_cfg.batch_size)
    shuffle = bool(train_cfg.shuffle)

    if use_balanced_batches:
        if not cohort_map:
            nemo_logging.warning(
                "BALANCED_CHILD_ADULT_BATCHES=1 but manifest has no age_cohort / is_child / balance_cohort; "
                "using standard shuffle. Use build_librispeech_age_manifest.py or add cohort fields."
            )
        else:
            ch, ad = ds.balance_index_lists()
            if len(ch) == 0 or len(ad) == 0 or bs < 2:
                nemo_logging.warning(
                    f"Balanced batches disabled (child={len(ch)} adult={len(ad)} batch_size={bs}); "
                    "using standard shuffle."
                )
            else:
                try:
                    batch_sampler = BalancedTwoGroupBatchSampler(
                        ch,
                        ad,
                        batch_size=bs,
                        shuffle=shuffle,
                        seed=balanced_sampler_seed,
                        drop_last=drop_last,
                    )
                except ValueError as e:
                    nemo_logging.warning(f"Balanced batching disabled ({e}); using standard shuffle.")
                else:
                    nemo_logging.info(
                        f"Balanced batches: child_idx={len(ch)} adult_idx={len(ad)} batch_size={bs}"
                    )
                    return torch.utils.data.DataLoader(
                        dataset=ds,
                        batch_sampler=batch_sampler,
                        num_workers=nw,
                        pin_memory=pin,
                        collate_fn=ds.collate_fn,
                    )

    return torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=pin,
        drop_last=drop_last,
        collate_fn=ds.collate_fn,
    )


def main():
    parser = argparse.ArgumentParser(
        description="NeMo Parakeet TDT + age-adversarial BCE head (see module docstring)."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Train JSONL path (overrides TRAIN_MANIFEST env if set)",
    )
    args, _unknown = parser.parse_known_args()

    MODEL_ID = os.environ.get("MODEL_ID", "nvidia/parakeet-tdt-1.1b")
    raw_manifest = (args.manifest or os.environ.get("TRAIN_MANIFEST", "")).strip()
    if not raw_manifest:
        raise FileNotFoundError(
            "Set TRAIN_MANIFEST or pass --manifest to a JSONL with audio_filepath, text, duration, "
            "age_years (or age_target / age). Use comma-separated paths + TRAIN_MANIFEST_COHORTS to merge "
            "child and adult manifests (see module docstring)."
        )

    SAVE_DIR = os.environ.get("SAVE_DIR", "./nemo_parakeet_age_adv")
    os.makedirs(SAVE_DIR, exist_ok=True)

    manifest_paths = [p.strip() for p in raw_manifest.split(",") if p.strip()]
    if not manifest_paths:
        raise FileNotFoundError("TRAIN_MANIFEST is empty after splitting on commas.")

    for _mp in manifest_paths:
        if not os.path.isfile(_mp):
            raise FileNotFoundError(f"Manifest file not found: {_mp}")

    if len(manifest_paths) == 1:
        TRAIN_MANIFEST = manifest_paths[0]
    else:
        cohorts_raw = os.environ.get("TRAIN_MANIFEST_COHORTS", "").strip()
        if not cohorts_raw:
            raise ValueError(
                "Multiple comma-separated manifests require TRAIN_MANIFEST_COHORTS with the same number "
                "of labels (child or adult), e.g. TRAIN_MANIFEST_COHORTS=child,adult matching file order."
            )
        cohort_labels = [c.strip().lower() for c in cohorts_raw.split(",") if c.strip()]
        if len(cohort_labels) != len(manifest_paths):
            raise ValueError(
                f"TRAIN_MANIFEST has {len(manifest_paths)} files but TRAIN_MANIFEST_COHORTS has "
                f"{len(cohort_labels)} labels; counts must match."
            )
        _allowed_c = {"child", "adult"}
        for _c in cohort_labels:
            if _c not in _allowed_c:
                raise ValueError(f"Invalid TRAIN_MANIFEST_COHORTS entry {_c!r}; use child or adult.")

        from merge_child_adult_training_manifests import merge_labeled_manifests

        merged_path = os.environ.get("MERGED_TRAIN_MANIFEST_PATH", "").strip()
        if not merged_path:
            merged_path = os.path.join(SAVE_DIR, "merged_train_manifest.jsonl")
        _pairs = list(zip(manifest_paths, cohort_labels))
        _stats = merge_labeled_manifests(_pairs, merged_path)
        nemo_logging.info(
            f"Merged {len(manifest_paths)} manifests -> {merged_path} ({_stats.written} rows; "
            f"skipped no_age={_stats.skipped_no_age} no_duration={_stats.skipped_no_duration})"
        )
        if _stats.written == 0:
            raise RuntimeError(
                "Merged manifest is empty. Check inputs, age_years/age_target/age, and duration on each row."
            )
        TRAIN_MANIFEST = merged_path
    ckpt_dir = os.path.join(SAVE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
    NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "7"))
    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
    LAMBDA_ADV = float(os.environ.get("LAMBDA_ADV", "0.1"))
    LAMBDA_ADV_SCHEDULE = os.environ.get("LAMBDA_ADV_SCHEDULE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    LAMBDA_ADV_RAMP_START = int(os.environ.get("LAMBDA_ADV_RAMP_START", "10"))
    LAMBDA_ADV_RAMP_END = int(os.environ.get("LAMBDA_ADV_RAMP_END", "40"))
    DISC_LOSS_SCALE = float(
        os.environ.get("DISC_LOSS_SCALE", os.environ.get("LAMBDA_AGE", "1.0"))
    )
    USE_F0 = os.environ.get("USE_F0_NORM", "0").strip().lower() in ("1", "true", "yes")
    WER_PROBE_N = int(os.environ.get("WER_PROBE_N", "500"))
    PRECISION = os.environ.get("PRECISION", "bf16-mixed")

    CHILD_YOUNGEST = float(os.environ.get("AGE_CHILD_YOUNGEST", "2.0"))
    CHILD_OLDEST = float(os.environ.get("AGE_CHILD_OLDEST_CORPUS", "12.0"))
    USE_PRECOMPUTED_AGE_TARGET = os.environ.get("USE_PRECOMPUTED_AGE_TARGET", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    USE_BALANCED_BATCHES = os.environ.get("BALANCED_CHILD_ADULT_BATCHES", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    BALANCED_SAMPLER_SEED = int(os.environ.get("BALANCED_SAMPLER_SEED", "0"))

    USE_CHAINED_ADAPTER = os.environ.get("USE_CHAINED_ADAPTER", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    BOTTLENECK_DIM = int(os.environ.get("BOTTLENECK_DIM", "256"))
    LR_AGE_DISC = float(os.environ.get("LR_AGE_DISC", "1e-3"))
    LR_ADAPTER = float(os.environ.get("LR_ADAPTER", "5e-4"))
    LR_JOINT = float(os.environ.get("LR_JOINT", "1e-4"))
    LR_DECODER = float(os.environ.get("LR_DECODER", "5e-5"))
    _save_ep_raw = os.environ.get("SAVE_HUMAN_EPOCHS", "5,6,7").strip()
    SAVE_HUMAN_EPOCHS: List[int] = []
    for x in _save_ep_raw.split(","):
        x = x.strip()
        if x.isdigit():
            SAVE_HUMAN_EPOCHS.append(int(x))
    if not SAVE_HUMAN_EPOCHS:
        SAVE_HUMAN_EPOCHS = [5, 6, 7]

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    manifest_list = [TRAIN_MANIFEST]
    age_map = build_age_lookup_from_manifests(
        manifest_list,
        CHILD_YOUNGEST,
        CHILD_OLDEST,
        use_precomputed_target=USE_PRECOMPUTED_AGE_TARGET,
    )
    cohort_map = build_cohort_lookup_from_manifests(manifest_list)
    if not age_map:
        raise RuntimeError(
            "No age entries loaded from manifest. Add age_years, age_target, or age fields per row."
        )

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=PRECISION if torch.cuda.is_available() else 32,
        max_epochs=NUM_EPOCHS,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        logger=pl.loggers.TensorBoardLogger(save_dir=SAVE_DIR, name="age_adv"),
        gradient_clip_val=float(os.environ.get("GRAD_CLIP", "1.0")),
    )

    nemo_logging.info(f"Loading {MODEL_ID}...")
    model_cfg = ASRModel.from_pretrained(MODEL_ID, return_config=True)
    enc_key = "_target_" if "_target_" in model_cfg.encoder else "target"
    with open_dict(model_cfg):
        adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder[enc_key])
        if adapter_metadata is not None:
            model_cfg.encoder[enc_key] = adapter_metadata.adapter_class_path

    model = ASRModel.from_pretrained(MODEL_ID, override_config_path=model_cfg, trainer=trainer)

    with open_dict(model.cfg):
        if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
            model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    if hasattr(model, "change_decoding_strategy"):
        model.change_decoding_strategy(model.cfg.decoding)

    device = next(model.parameters()).device

    if USE_CHAINED_ADAPTER:
        from parakeet_age_adv_nemo_extras import (
            NOISE_CURRICULUM,
            ChainedAdapterTrainDebugCallback,
            NoiseCurriculumCallback,
            WaveformAugmentor,
            apply_spec_augmentation,
            attach_chained_adapter_stack,
            build_discriminative_configure_optimizers,
            collect_noise_audio_files,
            parse_noise_dirs_from_env,
            partial_unfreeze_joint_decoder_lstm,
            patch_forward_waveform_and_chain_reset,
        )

        noise_dirs = parse_noise_dirs_from_env()
        if not noise_dirs:
            noise_dirs = [
                "/kaggle/input/datasets/akarshks/noise/noise_part_1",
                "/kaggle/input/datasets/akarshkumarshukla/asr-data/noise_part_0",
            ]
        noise_files = collect_noise_audio_files(noise_dirs)
        nemo_logging.info(
            f"USE_CHAINED_ADAPTER=1: {len(noise_files)} noise clips from {len(noise_dirs)} dir(s)"
        )
        if NOISE_CURRICULUM:
            _nprob, _nsnr0, _nsnr1 = (0.3, 5.0, 20.0)
        else:
            _nprob = float(os.environ.get("NOISE_AUG_PROB", "0.6"))
            _nsnr0 = float(os.environ.get("NOISE_SNR_MIN", "1.0"))
            _nsnr1 = float(os.environ.get("NOISE_SNR_MAX", "7.0"))

        attach_chained_adapter_stack(model, BOTTLENECK_DIM, in_features=1024)
        waug = WaveformAugmentor(
            speed_min=0.85,
            speed_max=1.15,
            pitch_min=-2.0,
            pitch_max=2.0,
            sample_rate=int(getattr(model.preprocessor, "_sample_rate", 16000)),
            speed_prob=0.5,
            pitch_prob=0.5,
            external_noise_prob=_nprob,
            external_snr_min=_nsnr0,
            external_snr_max=_nsnr1,
            noise_file_paths=noise_files if noise_files else None,
            gain_prob=0.3,
            gain_db_min=-8.0,
            gain_db_max=8.0,
        ).to(device)
        patch_forward_waveform_and_chain_reset(model, waug)
        apply_spec_augmentation(model)
        partial_unfreeze_joint_decoder_lstm(model)

    d_enc = infer_encoder_dim(model, device)
    _age_head = ParakeetWithAgeAdversarial(d_enc).to(device)
    model.add_module("parakeet_age_adv", _age_head)
    model.age_disc = model.parakeet_age_adv.age_disc

    model.training_step = types.MethodType(
        make_training_step_with_age(
            LAMBDA_ADV,
            DISC_LOSS_SCALE,
            USE_F0,
            use_lambda_adv_schedule=LAMBDA_ADV_SCHEDULE,
            lambda_adv_ramp_start_1based=LAMBDA_ADV_RAMP_START,
            lambda_adv_ramp_end_1based=LAMBDA_ADV_RAMP_END,
        ),
        model,
    )

    with open_dict(model.cfg.train_ds):
        model.cfg.train_ds.manifest_filepath = TRAIN_MANIFEST
        model.cfg.train_ds.batch_size = BATCH_SIZE
        model.cfg.train_ds.num_workers = NUM_WORKERS
        model.cfg.train_ds.shuffle = True

    model.setup_training_data(model.cfg.train_ds)
    model._train_dl = build_dataloader_with_age(
        model,
        model.cfg.train_ds,
        age_map,
        cohort_map=cohort_map,
        use_balanced_batches=USE_BALANCED_BATCHES,
        balanced_sampler_seed=BALANCED_SAMPLER_SEED,
    )

    if USE_CHAINED_ADAPTER:
        from parakeet_age_adv_nemo_extras import build_discriminative_configure_optimizers

        model.configure_optimizers = build_discriminative_configure_optimizers(
            model,
            model.age_disc,
            NUM_EPOCHS,
            lr_adapter=LR_ADAPTER,
            lr_joint=LR_JOINT,
            lr_decoder=LR_DECODER,
            lr_age_disc=LR_AGE_DISC,
        )
        nemo_logging.info(
            f"Discriminative optimizer: LR_AGE_DISC={LR_AGE_DISC} LR_ADAPTER={LR_ADAPTER} "
            f"LR_JOINT={LR_JOINT} LR_DECODER={LR_DECODER}"
        )
    else:
        patch_configure_optimizers(model, model.age_disc)

    trainer.callbacks.append(BalancedBatchEpochCallback())
    trainer.callbacks.append(ReinforceDecoderJointTrainMode())
    if USE_CHAINED_ADAPTER and NOISE_CURRICULUM:
        trainer.callbacks.append(NoiseCurriculumCallback(enabled=True))
    _dbg = int(os.environ.get("TRAIN_DEBUG_STEPS", "0"))
    if USE_CHAINED_ADAPTER and _dbg > 0:
        trainer.callbacks.append(ChainedAdapterTrainDebugCallback(num_steps=_dbg))
    trainer.callbacks.append(
        SaveEpochsAndWERProbe(
            ckpt_dir,
            save_human_epochs=SAVE_HUMAN_EPOCHS,
            manifest_path=TRAIN_MANIFEST,
            probe_n=WER_PROBE_N,
            save_adapter_weights=USE_CHAINED_ADAPTER,
        )
    )

    if LAMBDA_ADV_SCHEDULE:
        nemo_logging.info(
            f"λ_adv schedule: 0 until epoch<{LAMBDA_ADV_RAMP_START} (1-based), "
            f"linear ramp to λ_max={LAMBDA_ADV} through epoch {LAMBDA_ADV_RAMP_END}, then constant."
        )
    nemo_logging.info(
        f"Training: chained_adapter={USE_CHAINED_ADAPTER} lambda_adv_max={LAMBDA_ADV} "
        f"schedule={LAMBDA_ADV_SCHEDULE} disc_loss_scale={DISC_LOSS_SCALE} epochs={NUM_EPOCHS} "
        f"encoder_dim={d_enc} age_disc_in={2 * d_enc} age_keys={len(age_map)}"
    )
    trainer.fit(model)

    final_path = os.path.join(ckpt_dir, "model_final.nemo")
    model.save_to(final_path)
    nemo_logging.info(f"Saved final model -> {final_path}")
    if USE_CHAINED_ADAPTER and hasattr(model, "save_adapters"):
        try:
            model.save_adapters(os.path.join(ckpt_dir, "adapter_final.pt"))
            nemo_logging.info("Saved adapter_final.pt")
        except Exception as e:
            nemo_logging.warning(f"save_adapters final: {e}")


if __name__ == "__main__":
    main()
