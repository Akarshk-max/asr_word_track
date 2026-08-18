# ==========================================
# Kaggle ASR Evaluation - SpeechRecognition Only
# ==========================================
# Run these in a separate cell before running the main script!
# !pip install -q "numpy<2.0.0"
# !pip install -q jiwer SpeechRecognition soundfile librosa
# !apt-get install -y libsndfile1 ffmpeg

import os
import json
import time
import speech_recognition as sr
import jiwer

# ==========================================
# Configuration and Paths
# ==========================================
DATASET_BASE = "./eval_subset" 
JSONL_PATH = os.path.join(DATASET_BASE, "eval_transcripts.jsonl")

# Models to evaluate
MODELS_TO_RUN = {
    # 3. SpeechRecognition Module (Google Web Speech API)
    # Note: Requires internet access.
    "SpeechRecognition-Google": {
        "backend": "google"
    },
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
# Evaluators
# ==========================================
class SpeechRecognitionEvaluator:
    def __init__(self, backend="google"):
        self.recognizer = sr.Recognizer()
        self.backend = backend

    def transcribe(self, audio_paths):
        predictions = []
        for path in audio_paths:
            try:
                with sr.AudioFile(path) as source:
                    audio = self.recognizer.record(source)
                if self.backend == "google":
                    text = self.recognizer.recognize_google(audio)
                else:
                    text = ""
                predictions.append(text.lower())
            except sr.UnknownValueError:
                predictions.append("")
            except sr.RequestError as e:
                predictions.append("")
                print(f"Could not request results from Google Service; {e}")
            except Exception as e:
                predictions.append("")
                print(f"Error processing {path}: {e}")
        return predictions

# ==========================================
# Main Evaluation Loop
# ==========================================
def run_evaluation():
    samples = load_dataset(JSONL_PATH, DATASET_BASE)
    print(f"Loaded {len(samples)} usable samples for evaluation.")
    if len(samples) == 0:
        return

    audio_paths = [s["audio_path"] for s in samples]
    references = [s["text"] for s in samples]
    results = {}

    for model_name, cfg in MODELS_TO_RUN.items():
        print("=" * 40)
        print(f"Evaluating Model: {model_name}")
        
        start_time = time.time()
        try:
            evaluator = SpeechRecognitionEvaluator(cfg["backend"])
            predictions = evaluator.transcribe(audio_paths)
            
            valid_refs = []
            valid_preds = []
            for ref, pred in zip(references, predictions):
                if ref.strip():
                    valid_refs.append(ref)
                    valid_preds.append(pred if pred.strip() else "<empty>")

            wer = compute_wer(valid_refs, valid_preds)
            elapsed = time.time() - start_time
            
            results[model_name] = {
                "wer": wer,
                "time_sec": round(elapsed, 2)
            }
            
            print(f"-> WER: {wer:.4f}")
            print(f"-> Time taken: {elapsed:.2f}s")
            
            print("Examples (Predicted vs True):")
            for i in range(min(3, len(valid_preds))):
                print(f"  P: {valid_preds[i]}")
                print(f"  T: {valid_refs[i]}\n")
            
        except Exception as e:
            print(f"Failed evaluating {model_name}: {e}")

    print("\n==========================================")
    print("FINAL RESULTS")
    print("==========================================")
    for model_name, res in results.items():
        print(f"{model_name:<25} | WER: {res['wer']:.4f} | Time: {res['time_sec']}s")

if __name__ == "__main__":
    run_evaluation()
