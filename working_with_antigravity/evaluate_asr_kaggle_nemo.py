# ==========================================
# Kaggle ASR Evaluation - Nvidia Parakeet (via HuggingFace, NO NeMo required)
# ==========================================
# Run this in a SEPARATE cell above this script, then RESTART the kernel:
# !pip install -q transformers accelerate soundfile librosa jiwer
#
# WHY NO NeMo?
# Kaggle's PyTorch is compiled against numpy 2.x.
# NeMo installs an older lightning that forces numpy 1.x, causing a C ABI crash.
# NVIDIA publishes Parakeet on HuggingFace Hub as a standard CTC model,
# so it can run directly via transformers with zero NeMo involvement.

import os
import json
import time
import torch
import librosa
import jiwer
from transformers import AutoProcessor, AutoModelForCTC

# ==========================================
# Configuration and Paths
# ==========================================
# Replace this with your actual Kaggle dataset path.
DATASET_BASE = "./eval_subset"
JSONL_PATH = os.path.join(DATASET_BASE, "eval_transcripts.jsonl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing on: {DEVICE}")

# Models to evaluate
MODELS_TO_RUN = {
    "Parakeet-CTC-0.6b": {
        "model_id": "nvidia/parakeet-ctc-0.6b"
    },
    # Uncomment to also test the larger 1.1B version:
    # "Parakeet-CTC-1.1b": {
    #     "model_id": "nvidia/parakeet-ctc-1.1b"
    # },
}

# ==========================================
# Utility Functions
# ==========================================
def load_dataset(jsonl_path, base_dir):
    samples = []
    if not os.path.exists(jsonl_path):
        print(f"Dataset not found at {jsonl_path}")
        return samples
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            audio_path = os.path.join(base_dir, data["audio_path"])
            true_text = data.get("orthographic_text", "").strip().lower()
            if os.path.exists(audio_path) and true_text:
                samples.append({"audio_path": audio_path, "text": true_text})
    return samples

def compute_wer(references, predictions):
    return jiwer.wer(references, predictions)

# ==========================================
# Evaluator
# ==========================================
class ParakeetEvaluator:
    def __init__(self, model_id):
        print(f"Loading {model_id} via HuggingFace (direct CTC decode)...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(model_id).to(DEVICE)
        self.model.eval()

    def transcribe(self, audio_paths):
        predictions = []
        for path in audio_paths:
            # Load audio and resample to 16kHz (Parakeet's expected sample rate)
            audio, _ = librosa.load(path, sr=16000)
            
            # Prepare input features
            inputs = self.processor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).to(DEVICE)
            
            # Forward pass through CTC model
            with torch.no_grad():
                logits = self.model(**inputs).logits  # (1, time_steps, vocab_size)
            
            # Greedy CTC decode:
            # 1. argmax picks the most likely token per frame
            # 2. processor.batch_decode collapses consecutive repeated tokens
            #    and removes blank (<pad>) tokens — the two steps of CTC decoding.
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
                group_tokens=True  # This is the key: collapse CTC repeats
            )[0]
            predictions.append(transcription.strip().lower())
        return predictions

# ==========================================
# Main Evaluation Loop
# ==========================================
def run_evaluation():
    samples = load_dataset(JSONL_PATH, DATASET_BASE)
    print(f"Loaded {len(samples)} usable samples for evaluation.")
    if not samples:
        return

    audio_paths = [s["audio_path"] for s in samples]
    references  = [s["text"]       for s in samples]
    results = {}

    for model_name, cfg in MODELS_TO_RUN.items():
        print("=" * 40)
        print(f"Evaluating Model: {model_name}")
        start_time = time.time()
        try:
            evaluator   = ParakeetEvaluator(cfg["model_id"])
            predictions = evaluator.transcribe(audio_paths)

            valid_refs  = []
            valid_preds = []
            for ref, pred in zip(references, predictions):
                if ref.strip():
                    valid_refs.append(ref)
                    valid_preds.append(pred if pred.strip() else "<empty>")

            wer     = compute_wer(valid_refs, valid_preds)
            elapsed = time.time() - start_time

            results[model_name] = {"wer": wer, "time_sec": round(elapsed, 2)}
            print(f"-> WER: {wer:.4f}")
            print(f"-> Time taken: {elapsed:.2f}s")
            print("Examples (Predicted vs True):")
            for i in range(min(5, len(valid_preds))):
                print(f"  P: {valid_preds[i]}")
                print(f"  T: {valid_refs[i]}\n")

        except Exception as e:
            import traceback
            print(f"Failed evaluating {model_name}: {e}")
            traceback.print_exc()

    print("\n==========================================")
    print("FINAL RESULTS")
    print("==========================================")
    for model_name, res in results.items():
        print(f"{model_name:<25} | WER: {res['wer']:.4f} | Time: {res['time_sec']}s")

if __name__ == "__main__":
    run_evaluation()
