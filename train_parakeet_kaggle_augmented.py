"""
======================================================
Kaggle Fine-Tuning Script: Nvidia Parakeet-CTC 0.6B
WITH Classroom Data Augmentation
======================================================
Upload this script to your Kaggle Notebook.

PREREQUISITES (Run in a cell before this script):
!pip install -q transformers datasets evaluate jiwer librosa accelerate soundfile audiomentations

Optional for RoomSimulator (reverberation):
!pip install -q audiomentations[extras]
# or: !pip install -q pyroomacoustics

STRATEGY:
Same as baseline: freeze encoder, train only lm_head. Additionally applies
waveform-level augmentation to simulate classroom acoustic conditions.
"""

import os
import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from datasets import Dataset, Audio
from transformers import (
    AutoProcessor,
    AutoModelForCTC,
    TrainingArguments,
    Trainer
)
import evaluate

# ==========================================
# 1. Configuration
# ==========================================
MODEL_ID = "nvidia/parakeet-ctc-0.6b"
DATASET_BASE_DIR = "."
JSONL_PATH = os.path.join(DATASET_BASE_DIR, "train_word_transcripts.jsonl")

# Path to noise folder (classroom/ambient sounds). Searched under DATASET_BASE_DIR.
NOISE_FOLDER_NAME = "noise_part_0"
NOISE_PATH = os.path.join(DATASET_BASE_DIR, NOISE_FOLDER_NAME)

# Training hyperparameters
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 13
SAVE_DIR = "./parakeet_finetuned_augmented"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Initializing training on: {device}")

# ==========================================
# 2. Augmentation Config (Classroom Simulation)
# ==========================================
AUGMENT_CONFIG = {
    "use_background_noise": True,
    "noise_path": NOISE_PATH,
    "noise_min_snr_db": 5.0,
    "noise_max_snr_db": 20.0,
    "noise_p": 0.75,
    "use_reverb": True,
    "reverb_min_rt60": 0.4,
    "reverb_max_rt60": 0.9,
    "reverb_p": 0.5,
    "use_time_stretch": True,
    "stretch_min_rate": 0.95,
    "stretch_max_rate": 1.05,
    "stretch_p": 0.3,
    "use_gaussian_noise": True,
    "gaussian_min_amplitude": 0.001,
    "gaussian_max_amplitude": 0.01,
    "gaussian_p": 0.3,
    "use_gain": True,
    "gain_min_db": -2.0,
    "gain_max_db": 2.0,
    "gain_p": 0.4,
}

# ==========================================
# 3. Data Loading & Preprocessing
# ==========================================
def _find_audio_path(base_dir: str, audio_path: str) -> Optional[str]:
    """Finds the actual path of an audio file in any audio_part_* folder."""
    full_path = os.path.join(base_dir, audio_path)
    if os.path.exists(full_path):
        return full_path
    filename = os.path.basename(audio_path)
    try:
        for item in sorted(os.listdir(base_dir)):
            if item.startswith("audio_part_"):
                candidate = os.path.join(base_dir, item, filename)
                if os.path.exists(candidate):
                    return candidate
    except OSError:
        pass
    return None


def preprocess_dataset(jsonl_path: str, base_dir: str) -> Dataset:
    """Parses the JSONL and prepares a HuggingFace Dataset."""
    print("Parsing JSONL manifest...")
    audio_paths = []
    texts = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            full_path = _find_audio_path(base_dir, data["audio_path"])
            if full_path is not None:
                audio_paths.append(full_path)
                texts.append(data["orthographic_text"].strip().lower())

    print(f"Found {len(audio_paths)} valid audio files.")

    ds = Dataset.from_dict({"audio": audio_paths, "text": texts})
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


# ==========================================
# 4. Augmentation Pipeline
# ==========================================
def build_augmentation_pipeline(config: Dict[str, Any]):
    """
    Builds an audiomentations Compose pipeline for classroom simulation.
    Returns None if audiomentations is not available; transforms are optional.
    """
    try:
        from audiomentations import (
            Compose,
            AddBackgroundNoise,
            AddGaussianNoise,
            TimeStretch,
            Gain,
        )
    except ImportError:
        print("Warning: audiomentations not installed. Augmentation disabled.")
        return None

    transforms = []

    # Tier 1: Background noise (classroom ambient from noise_part_0)
    if config.get("use_background_noise") and os.path.isdir(config.get("noise_path", "")):
        transforms.append(
            AddBackgroundNoise(
                sounds_path=config["noise_path"],
                min_snr_db=config.get("noise_min_snr_db", 5.0),
                max_snr_db=config.get("noise_max_snr_db", 20.0),
                p=config.get("noise_p", 0.75),
            )
        )
    elif config.get("use_background_noise"):
        print(f"Warning: noise path {config.get('noise_path')} not found. Skipping AddBackgroundNoise.")

    # Tier 1: Reverberation (room acoustics)
    if config.get("use_reverb"):
        try:
            from audiomentations import RoomSimulator

            transforms.append(
                RoomSimulator(
                    calculation_mode="rt60",
                    min_target_rt60=config.get("reverb_min_rt60", 0.4),
                    max_target_rt60=config.get("reverb_max_rt60", 0.9),
                    max_order=3,
                    leave_length_unchanged=True,
                    p=config.get("reverb_p", 0.5),
                )
            )
        except ImportError:
            print("Warning: RoomSimulator requires pyroomacoustics. Skipping reverb. Install: pip install audiomentations[extras]")

    # Tier 2: TimeStretch
    if config.get("use_time_stretch"):
        transforms.append(
            TimeStretch(
                min_rate=config.get("stretch_min_rate", 0.95),
                max_rate=config.get("stretch_max_rate", 1.05),
                p=config.get("stretch_p", 0.3),
            )
        )

    # Tier 2: AddGaussianNoise
    if config.get("use_gaussian_noise"):
        transforms.append(
            AddGaussianNoise(
                min_amplitude=config.get("gaussian_min_amplitude", 0.001),
                max_amplitude=config.get("gaussian_max_amplitude", 0.01),
                p=config.get("gaussian_p", 0.3),
            )
        )

    # Tier 2: Gain (volume perturbation)
    if config.get("use_gain"):
        transforms.append(
            Gain(
                min_gain_db=config.get("gain_min_db", -2.0),
                max_gain_db=config.get("gain_max_db", 2.0),
                p=config.get("gain_p", 0.4),
            )
        )

    if not transforms:
        return None

    pipeline = Compose(transforms)
    print(f"Augmentation pipeline: {len(transforms)} transforms")
    return pipeline


def apply_augmentation(waveform: np.ndarray, sample_rate: int, augment_fn) -> np.ndarray:
    """Apply augmentation if pipeline exists. waveform must be float32, shape (n_samples,)."""
    if augment_fn is None:
        return waveform
    out = augment_fn(samples=waveform.astype(np.float32), sample_rate=sample_rate)
    if out is None:
        return waveform
    return out


# ==========================================
# 5. Model & Feature Extraction Load
# ==========================================
print(f"Loading Processor and Model ({MODEL_ID})...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCTC.from_pretrained(
    MODEL_ID,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

print("Freezing the encoder layers...")
model.freeze_feature_encoder()
for param in model.model.encoder.parameters():
    param.requires_grad = False
for param in model.lm_head.parameters():
    param.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")

# Build augmentation pipeline (used only for training)
augment_pipeline = build_augmentation_pipeline(AUGMENT_CONFIG)

# ==========================================
# 6. Data Collator & Prepare Functions
# ==========================================
@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def make_prepare_dataset(processor, augment_fn, is_training: bool):
    """Factory: returns prepare_dataset function. Augmentation applied only when is_training=True."""

    def prepare_dataset(batch: Dict) -> Dict:
        audio = batch["audio"]
        waveform = audio["array"]
        sample_rate = audio["sampling_rate"]

        if is_training and augment_fn is not None:
            waveform = apply_augmentation(waveform, sample_rate, augment_fn)
            if waveform.dtype != np.float32:
                waveform = waveform.astype(np.float32)

        batch["input_values"] = processor(waveform, sampling_rate=sample_rate).input_values[0]

        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids

        return batch

    return prepare_dataset


# ==========================================
# 7. Training Execution
# ==========================================
def main():
    raw_dataset = preprocess_dataset(JSONL_PATH, DATASET_BASE_DIR)

    print("Splitting dataset into train and validation sets...")
    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split_dataset["train"]
    eval_ds = split_dataset["test"]

    def dump_split(ds, out_path):
        records = []
        for i in range(len(ds)):
            records.append({
                "audio_path": ds[i]["audio"]["path"],
                "orthographic_text": ds[i]["text"],
            })
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    print("Saving split manifests (train_split.json and eval_split.json)...")
    dump_split(train_ds, "train_split.json")
    dump_split(eval_ds, "eval_split.json")

    prepare_train = make_prepare_dataset(processor, augment_pipeline, is_training=True)
    prepare_eval = make_prepare_dataset(processor, None, is_training=False)

    print("Extracting features (training with augmentation, eval without)...")
    train_vectorized = train_ds.map(
        prepare_train,
        remove_columns=train_ds.column_names,
        num_proc=4,
    )
    eval_vectorized = eval_ds.map(
        prepare_eval,
        remove_columns=eval_ds.column_names,
        num_proc=4,
    )

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    training_args = TrainingArguments(
        output_dir=SAVE_DIR,
        group_by_length=True,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        evaluation_strategy="steps",
        num_train_epochs=NUM_EPOCHS,
        fp16=True,
        gradient_checkpointing=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=50,
        learning_rate=LEARNING_RATE,
        warmup_steps=500,
        save_total_limit=2,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_vectorized,
        eval_dataset=eval_vectorized,
        tokenizer=processor.feature_extractor,
    )

    print("Starting training loop (with classroom augmentation)...")
    trainer.train()

    print("Training complete! Saving final model...")
    trainer.save_model(SAVE_DIR)
    processor.save_pretrained(SAVE_DIR)
    print(f"Saved weights to {SAVE_DIR}")


if __name__ == "__main__":
    main()
