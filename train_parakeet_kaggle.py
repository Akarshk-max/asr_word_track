"""
======================================================
Kaggle Fine-Tuning: Nvidia Parakeet-CTC 0.6B
======================================================
Custom training loop with:
  - Combined dataset: asr_data (JSONL) + talkbank_data (JSONL)
  - CSV-only split manifests (lightweight, no audio data)
  - Per-epoch train loss + validation WER logging
  - Best-model checkpointing (lowest val WER) + per-epoch checkpoints
  - Differential LR: lower for encoder blocks, higher for CTC head
  - Linear warmup LR scheduler (applied per optimizer group)
  - Trains last 4 encoder blocks + full CTC head from epoch 1
  - 5 epochs max
  - On-the-fly feature extraction (no disk caching to prevent Kaggle OOM)

PREREQUISITES (run in a cell before this script):
!pip install -q transformers datasets evaluate jiwer librosa accelerate soundfile
"""

import os
# Prevent OpenMP and Tokenizer deadlocks when forking worker processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
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

# --- Kaggle dataset paths ---
ASR_DATA_DIR  = "/kaggle/input/datasets/akarshkumarshukla/asr-data"
ASR_JSONL     = os.path.join(ASR_DATA_DIR, "train_word_transcripts.jsonl")

TALKBANK_DIR  = "/kaggle/input/datasets/akarshkumarshukla/talk-bank-data/audio"
TALKBANK_JSON = "/kaggle/input/datasets/akarshkumarshukla/talkbank-transcript-full/train_word_transcripts.jsonl"

# Training hyperparameters
BATCH_SIZE        = 8
BASE_LR_ENCODER   = 1e-5   # Lower LR for the last-4 encoder blocks
BASE_LR_HEAD      = 1e-4   # Higher LR for the CTC head
NUM_EPOCHS        = 5
WARMUP_RATIO      = 0.1
SAVE_DIR          = "/kaggle/working/parakeet_finetuned"
CHECKPOINT_DIR    = "/kaggle/working/checkpoints"
CACHE_DIR         = "/kaggle/working/dataset_cache"   # processed features cached here
LOG_EVERY_N_STEPS = 50
NUM_WORKERS       = min(8, os.cpu_count() or 4)   # cap at 8: more workers = more RAM per fork
VAL_SPLIT         = 0.2

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

device = None  # Will be initialized in main() AFTER feature extraction

# ==========================================
# 2. Audio Indexing  (parallel across both dirs)
# ==========================================
print("Indexing audio files (parallel)...")
audio_index = {}

def _index_directory(search_dir):
    """Walk one directory and return {filename: fullpath} for all .flac files."""
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
    """Load both JSONLs in parallel, then create a HuggingFace dataset.
    
    Audio column stores plain string paths only - NO Audio() feature.
    Decoding happens in parallel inside prepare_dataset using soundfile.
    """
    print("Loading and combining datasets (parallel)...")
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_asr      = ex.submit(load_jsonl, ASR_JSONL,     "ASR JSONL")
        f_talkbank = ex.submit(load_jsonl, TALKBANK_JSON, "TalkBank JSONL")
        p1, t1 = f_asr.result()
        p2, t2 = f_talkbank.result()
    all_paths = p1 + p2
    all_texts = t1 + t2
    print(f"Combined: {len(all_paths)} total samples")
    # Keep audio_path as a plain string column — do NOT cast to Audio().
    # Audio() decoding is single-threaded and kills parallelism.
    ds = Dataset.from_dict({"audio_path": all_paths, "text": all_texts})
    return ds


# ==========================================
# 4. Model & Freezing (Moved to main function to avoid multiprocessing deadlocks)
# ==========================================


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


# Worker-local variable for datasets.map
_worker_processor = None

def prepare_dataset(batch):
    """
    On-the-fly feature extraction function for `with_transform()`.
    Takes a dictionary of lists (batch) and returns the processed batch
    ready for the DataCollator.
    """
    global _worker_processor
    if _worker_processor is None:
        _worker_processor = AutoProcessor.from_pretrained(MODEL_ID)
        
    audio_paths = batch["audio_path"]
    texts = batch["text"]
    
    input_values = []
    labels = []
    
    for path, text in zip(audio_paths, texts):
        speech, sr = sf.read(path)
        # Convert multi-channel (stereo) to mono by averaging channels
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            speech = librosa.resample(speech.astype(np.float32), orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Extract features
        iv = _worker_processor(speech, sampling_rate=sr).input_values[0]
        input_values.append(iv)
        
        # Tokenize text
        with _worker_processor.as_target_processor():
            lbl = _worker_processor(text).input_ids
        labels.append(lbl)

    return {
        "input_values": input_values,
        "labels": labels
    }


# ==========================================
# 6. Differential LR Scheduler
# ==========================================
def build_optimizer_and_scheduler(model, warmup_steps, total_steps):
    """
    Two optimizer parameter groups:
      - encoder_blocks: BASE_LR_ENCODER (lower, careful fine-tuning)
      - ctc_head:       BASE_LR_HEAD    (higher, task-specific head)
    A single LambdaLR applies the same warmup+decay multiplier to both groups
    so each group's effective LR scales from its own base.
    """
    encoder_params = []
    for layer in model.encoder.layers[-4:]:
        encoder_params += list(layer.parameters())

    head_params = list(model.ctc_head.parameters())

    optimizer = AdamW(
        [
            {"params": encoder_params, "lr": BASE_LR_ENCODER},
            {"params": head_params,    "lr": BASE_LR_HEAD},
        ],
        weight_decay=0.01,
    )

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
# 7. Custom Training Loop
# ==========================================
class ASRTrainer:
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

        total_steps  = len(self.train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            model, warmup_steps, total_steps
        )
        print(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}")
        print(f"Encoder LR: {BASE_LR_ENCODER:.1e} | Head LR: {BASE_LR_HEAD:.1e}")

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

            total_loss  += loss.item()
            num_batches += 1

            if (step + 1) % self.log_every == 0:
                avg     = total_loss / num_batches
                lr_enc  = self.scheduler.get_last_lr()[0]
                lr_head = self.scheduler.get_last_lr()[1]
                print(f"  [Epoch {epoch+1}] Step {step+1}/{len(self.train_loader)} | "
                      f"Loss: {avg:.4f} | LR_enc: {lr_enc:.2e} | LR_head: {lr_head:.2e}")

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
            t0         = time.time()
            train_loss = self.train_one_epoch(epoch)
            val_wer    = self.validate()
            elapsed    = time.time() - t0

            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"  Train Loss : {train_loss:.4f}")
            print(f"  Val WER    : {val_wer:.4f}")
            print(f"  Time       : {elapsed:.1f}s")

            # Always save per-epoch checkpoint
            self.save_checkpoint(tag=f"epoch_{epoch+1}")

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

    # Save lightweight CSV manifests — audio_path is now a plain string, fast pandas export
    def dump_csv(ds, out_path):
        pd.DataFrame({"audio_path": ds["audio_path"], "text": ds["text"]}).to_csv(
            out_path, index=False
        )
        print(f"  Manifest saved → {out_path} ({len(ds)} rows)")

    print("Saving CSV manifests (parallel)...")
    with ThreadPoolExecutor(max_workers=2) as ex:
        ex.submit(dump_csv, train_ds, "/kaggle/working/train_split.csv")
        ex.submit(dump_csv, eval_ds,  "/kaggle/working/eval_split.csv")

    # ------------------------------------------------------------------
    # On-the-fly feature extraction via transforms (NO disk caching).
    # This prevents the 20-100GB disk OOM issues on Kaggle and runs fast
    # via the DataLoader's num_workers.
    # ------------------------------------------------------------------
    print("Applying on-the-fly transforms to dataset...")
    train_vec = train_ds.with_transform(prepare_dataset)
    eval_vec  = eval_ds.with_transform(prepare_dataset)

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device initialized: {device}")

    print(f"Loading Processor ({MODEL_ID})...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # ==========================================
    # Model Loading & Freezing
    # (Done after feature extraction to prevent multiprocessing CUDA/memory deadlocks)
    # ==========================================
    print(f"Loading Model ({MODEL_ID})...")
    model = AutoModelForCTC.from_pretrained(
        MODEL_ID,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 4 encoder blocks
    for layer in model.encoder.layers[-4:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Unfreeze CTC head
    for param in model.ctc_head.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    model.to(device)

    trainer = ASRTrainer(
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
