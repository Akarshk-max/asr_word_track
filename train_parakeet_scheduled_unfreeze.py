"""
======================================================
Kaggle Fine-Tuning: Nvidia Parakeet-CTC 0.6B
======================================================
SCHEDULED UNFREEZING VARIANT:
  - Epoch 1: Only CTC head is trainable (encoder fully frozen)
  - Epoch 2+: Last 4 encoder blocks are unfrozen + added to optimizer
  - Differential LR: lower for encoder blocks, higher for CTC head
  - CSV-only split manifests (lightweight)
  - Per-epoch + best-model checkpointing
  - Linear warmup LR scheduler
  - 5 epochs max, no FP16

PREREQUISITES:
!pip install -q transformers datasets evaluate jiwer librosa accelerate soundfile
"""

import os
import csv
import json
import time
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from dataclasses import dataclass
from typing import Dict, List, Union
from datasets import Dataset, load_from_disk
from transformers import AutoProcessor, AutoModelForCTC
import jiwer

# ==========================================
# 1. Configuration
# ==========================================
MODEL_ID = "nvidia/parakeet-ctc-0.6b"

ASR_DATA_DIR  = "/kaggle/input/datasets/akarshkumarshukla/asr-data"
ASR_JSONL     = os.path.join(ASR_DATA_DIR, "train_word_transcripts.jsonl")
TALKBANK_DIR  = "/kaggle/input/datasets/akarshkumarshukla/talk-bank-data/audio"
TALKBANK_JSON = "/kaggle/input/datasets/akarshkumarshukla/talkbank-transcript-full/train_word_transcripts.jsonl"

BATCH_SIZE        = 8
BASE_LR_ENCODER   = 1e-5   # Used from epoch 2 onwards when encoder blocks unfreeze
BASE_LR_HEAD      = 1e-4   # Used from epoch 1 for the CTC head
NUM_EPOCHS        = 5
WARMUP_RATIO      = 0.1
SAVE_DIR          = "/kaggle/working/parakeet_finetuned_scheduled"
CHECKPOINT_DIR    = "/kaggle/working/checkpoints_scheduled"
LOG_EVERY_N_STEPS = 50
NUM_WORKERS       = min(8, os.cpu_count() or 4)   # cap at 8: more workers = more RAM per fork
VAL_SPLIT         = 0.2

# Epoch (1-indexed) at which encoder blocks get unfrozen
UNFREEZE_EPOCH = 2

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ==========================================
# 2. Audio Indexing
# ==========================================
print("Indexing audio files...")
audio_index = {}

def _index_directory(search_dir):
    local = {}
    if not os.path.isdir(search_dir):
        print(f"  Warning: {search_dir} not found, skipping.")
        return local
    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.endswith(".flac"):
                local[f] = os.path.join(root, f)
    return local

with ThreadPoolExecutor(max_workers=2) as ex:
    futures = {ex.submit(_index_directory, d): d for d in [ASR_DATA_DIR, TALKBANK_DIR]}
    for fut in as_completed(futures):
        audio_index.update(fut.result())

print(f"Indexed {len(audio_index)} total audio files")


def _find_audio(audio_path_field):
    return audio_index.get(os.path.basename(audio_path_field), None)


# ==========================================
# 3. Data Loading
# ==========================================
def load_jsonl(jsonl_path, source_tag):
    paths, texts = [], []
    if not os.path.exists(jsonl_path):
        print(f"  Not found: {jsonl_path}")
        return paths, texts
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                text = data.get("orthographic_text", "").strip().lower()
                if not text:
                    continue
                full = _find_audio(data.get("audio_path", ""))
                if full:
                    paths.append(full)
                    texts.append(text)
            except json.JSONDecodeError:
                continue
    print(f"  {source_tag}: {len(paths)} valid samples")
    return paths, texts


def build_combined_dataset():
    print("Loading and combining datasets (parallel)...")
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_asr      = ex.submit(load_jsonl, ASR_JSONL,     "ASR JSONL")
        f_talkbank = ex.submit(load_jsonl, TALKBANK_JSON, "TalkBank JSONL")
        p1, t1 = f_asr.result()
        p2, t2 = f_talkbank.result()
    all_paths = p1 + p2
    all_texts = t1 + t2
    print(f"Combined: {len(all_paths)} total samples")
    # Plain string paths — no Audio() cast (keeps decoding parallel)
    ds = Dataset.from_dict({"audio_path": all_paths, "text": all_texts})
    return ds


# ==========================================
# 4. Model — Start with ONLY head unfrozen
# ==========================================
print(f"Loading Processor and Model ({MODEL_ID})...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCTC.from_pretrained(
    MODEL_ID,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

# Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Unfreeze ONLY the CTC head to start
for param in model.ctc_head.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Epoch 1 trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
model.to(device)


# ==========================================
# 5. Data Collator & Feature Extraction
# ==========================================
@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def prepare_dataset(batch):
    """
    Manually decode audio with soundfile so each worker runs truly in parallel.
    Avoids HuggingFace Audio() single-threaded decoder entirely.
    """
    speech, sr = sf.read(batch["audio_path"])
    # Convert multi-channel (stereo) to mono by averaging channels
    # ParakeetFeatureExtractor only accepts mono; failing to do this crashes workers
    if speech.ndim > 1:
        speech = speech.mean(axis=1)
    if sr != 16000:
        import librosa
        speech = librosa.resample(speech.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
    batch["input_values"] = processor(speech, sampling_rate=sr).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


# ==========================================
# 6. Optimizer Builder (called fresh when unfreezing happens)
# ==========================================
def build_optimizer_and_scheduler(model, warmup_steps, total_steps, encoder_unfrozen=False):
    """
    If encoder_unfrozen=False (epoch 1):  only CTC head param group.
    If encoder_unfrozen=True  (epoch 2+): two groups — encoder (low LR) + head (high LR).
    """
    head_params = list(model.ctc_head.parameters())

    if encoder_unfrozen:
        encoder_params = []
        for layer in model.encoder.layers[-4:]:
            encoder_params += list(layer.parameters())
        param_groups = [
            {"params": encoder_params, "lr": BASE_LR_ENCODER},
            {"params": head_params,    "lr": BASE_LR_HEAD},
        ]
    else:
        param_groups = [
            {"params": head_params, "lr": BASE_LR_HEAD},
        ]

    optimizer = AdamW(param_groups, weight_decay=0.01)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


# ==========================================
# 7. Custom Training Loop with Scheduled Unfreezing
# ==========================================
class ASRTrainerScheduled:
    def __init__(self, model, processor, train_dataset, eval_dataset,
                 batch_size, num_epochs, warmup_ratio,
                 save_dir, checkpoint_dir, num_workers=4, log_every=50):
        self.model          = model
        self.processor      = processor
        self.save_dir       = save_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_every      = log_every
        self.num_epochs     = num_epochs
        self.best_wer       = float("inf")

        collator = DataCollatorCTCWithPadding(processor=processor)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collator, num_workers=num_workers, pin_memory=True
        )
        self.eval_loader = DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collator, num_workers=num_workers, pin_memory=True
        )

        # Steps per epoch (used to resize scheduler when we rebuild optimizer)
        self.steps_per_epoch = len(self.train_loader)
        total_steps  = self.steps_per_epoch * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)

        # Start with head-only optimizer
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            model, warmup_steps, total_steps, encoder_unfrozen=False
        )
        self.encoder_unfrozen = False
        self.global_step      = 0
        self.total_steps      = total_steps
        self.warmup_steps     = warmup_steps

        print(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}")
        print(f"Epoch 1 → CTC head only  (LR: {BASE_LR_HEAD:.1e})")
        print(f"Epoch {UNFREEZE_EPOCH}+ → + last 4 encoder blocks  (Encoder LR: {BASE_LR_ENCODER:.1e})")

    def _unfreeze_encoder(self):
        """Called at the start of UNFREEZE_EPOCH. Re-builds optimizer with both groups."""
        for layer in self.model.encoder.layers[-4:]:
            for param in layer.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())
        print(f"\n  [UNFREEZE] Encoder blocks added. Trainable: {trainable:,} / {total:,}")

        # Rebuild optimizer with two groups; remaining steps from global_step
        remaining_steps = self.total_steps - self.global_step
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            self.model, self.warmup_steps, remaining_steps, encoder_unfrozen=True
        )
        self.encoder_unfrozen = True

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss, num_batches = 0.0, 0

        for step, batch in enumerate(self.train_loader):
            input_values = batch["input_values"].to(device)
            labels       = batch["labels"].to(device)

            self.optimizer.zero_grad()
            outputs = self.model(input_values=input_values, labels=labels)
            loss    = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            total_loss  += loss.item()
            num_batches += 1

            if (step + 1) % self.log_every == 0:
                avg    = total_loss / num_batches
                lrs    = self.scheduler.get_last_lr()
                lr_str = " | ".join([f"LR[{i}]: {v:.2e}" for i, v in enumerate(lrs)])
                print(f"  [Epoch {epoch+1}] Step {step+1}/{len(self.train_loader)} | "
                      f"Loss: {avg:.4f} | {lr_str}")

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        all_preds, all_refs = [], []
        for batch in self.eval_loader:
            input_values = batch["input_values"].to(device)
            labels       = batch["labels"].to(device)
            logits       = self.model(input_values=input_values).logits

            pred_ids  = torch.argmax(logits, dim=-1).cpu().numpy()
            label_ids = labels.cpu().numpy()

            pred_strs = self.processor.batch_decode(pred_ids)
            label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
            ref_strs  = self.processor.batch_decode(label_ids, group_tokens=False)

            all_preds.extend([s.strip().lower() for s in pred_strs])
            all_refs.extend( [s.strip().lower() for s in ref_strs])

        all_preds = [p if p else "<empty>" for p in all_preds]
        return jiwer.wer(all_refs, all_preds)

    def save_checkpoint(self, tag):
        path = os.path.join(self.checkpoint_dir, tag)
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        print(f"  Checkpoint saved → {path}")

    def fit(self):
        print("=" * 60)
        print(f"TRAINING START | Epochs: {self.num_epochs} | "
              f"Train batches: {len(self.train_loader)} | Eval batches: {len(self.eval_loader)}")
        print("=" * 60)

        for epoch in range(self.num_epochs):
            epoch_num = epoch + 1

            # ---- Scheduled Unfreezing ----
            if epoch_num == UNFREEZE_EPOCH and not self.encoder_unfrozen:
                self._unfreeze_encoder()

            t0         = time.time()
            train_loss = self.train_one_epoch(epoch)
            val_wer    = self.validate()
            elapsed    = time.time() - t0

            status = "head only" if not self.encoder_unfrozen else "head + encoder[-4:]"
            print(f"\n{'='*60}")
            print(f"Epoch {epoch_num}/{self.num_epochs}  [{status}]")
            print(f"  Train Loss : {train_loss:.4f}")
            print(f"  Val WER    : {val_wer:.4f}")
            print(f"  Time       : {elapsed:.1f}s")

            self.save_checkpoint(tag=f"epoch_{epoch_num}")

            if val_wer < self.best_wer:
                self.best_wer = val_wer
                self.save_checkpoint(tag="best")
                print(f"  ★ New best WER: {self.best_wer:.4f}")

            print(f"{'='*60}\n")

        print("Training complete! Saving final model...")
        self.model.save_pretrained(self.save_dir)
        self.processor.save_pretrained(self.save_dir)
        print(f"Final model → {self.save_dir}")
        print(f"Best val WER : {self.best_wer:.4f}")


# ==========================================
# 8. Main Entrypoint
# ==========================================
def main():
    combined_ds = build_combined_dataset()

    print(f"Splitting dataset ({1-VAL_SPLIT:.0%} train / {VAL_SPLIT:.0%} val, seed=42)...")
    split    = combined_ds.train_test_split(test_size=VAL_SPLIT, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    def dump_csv(ds, out_path):
        pd.DataFrame({"audio_path": ds["audio_path"], "text": ds["text"]}).to_csv(
            out_path, index=False
        )
        print(f"  Manifest saved → {out_path} ({len(ds)} rows)")

    print("Saving CSV manifests (parallel)...")
    with ThreadPoolExecutor(max_workers=2) as ex:
        ex.submit(dump_csv, train_ds, "/kaggle/working/train_split_scheduled.csv")
        ex.submit(dump_csv, eval_ds,  "/kaggle/working/eval_split_scheduled.csv")

    print("Extracting features...")
    train_vec = train_ds.map(
        prepare_dataset,
        remove_columns=train_ds.column_names,
        num_proc=NUM_WORKERS,
        writer_batch_size=500,
    )
    eval_vec = eval_ds.map(
        prepare_dataset,
        remove_columns=eval_ds.column_names,
        num_proc=NUM_WORKERS,
        writer_batch_size=500,
    )

    trainer = ASRTrainerScheduled(
        model=model,
        processor=processor,
        train_dataset=train_vec,
        eval_dataset=eval_vec,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        save_dir=SAVE_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        num_workers=NUM_WORKERS,
        log_every=LOG_EVERY_N_STEPS,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
