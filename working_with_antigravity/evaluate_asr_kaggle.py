import os
import json
import time

# Install requirements if running directly in a new Kaggle kernel:
# Install requirements if running directly in a new Kaggle kernel:
# Run these in a separate cell before running the main script!
# !pip install -q "numpy<2.0.0"
# !pip install -q transformers peft accelerate datasets editdistance evaluate jiwer SpeechRecognition soundfile librosa
# !pip install -q wget
# !apt-get install -y libsndfile1 ffmpeg
# !pip install -q Cython
# !pip install -q nemoguardrails nemo_toolkit['all']

import torch
import librosa
import speech_recognition as sr
from transformers import AutoProcessor, AutoModelForCTC, pipeline
import jiwer

try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    print("NVIDIA NeMo not installed. Skipping Parakeet evaluations. Run `pip install nemo_toolkit['asr']`")
    NEMO_AVAILABLE = False


# ==========================================
# Configuration and Paths
# ==========================================

# Replace this with the actual path to your Kaggle dataset when running on Kaggle.
# Example: "/kaggle/input/my-asr-dataset/"
DATASET_BASE = "./eval_subset" 
JSONL_PATH = os.path.join(DATASET_BASE, "eval_transcripts.jsonl")

# Hardware
DEVICE = "cuda" if torch.cuda.is_axis_available() else "cpu"
print(f"Executing on: {DEVICE}")

# Models to evaluate
MODELS_TO_RUN = {
    # 1. Transformers library: Wav2Vec2
    "Wav2Vec2-Large-960h": {
        "type": "transformers",
        "model_id": "facebook/wav2vec2-large-960h-lv60-self"
    },
    
    # 2. Transformers library: Whisper 
    # Can also test openai/whisper-large-v3, but base is faster for a quick test
    "Whisper-Base": {
       "type": "transformers",
       "model_id": "openai/whisper-base"
    },
    
    # 3. SpeechRecognition Module (Google Web Speech API)
    # Note: Requires internet access, not suitable for massive offline batches
    # but works for small baselines.
    "SpeechRecognition-Google": {
        "type": "speechrecognition",
        "model_id": "google"
    },
}

if NEMO_AVAILABLE:
    # 4. Nvidia NeMo - Parakeet
    # Example versions: 'nvidia/parakeet-rnnt-1.1b', 'nvidia/parakeet-ctc-0.6b'
    MODELS_TO_RUN["NeMo-Parakeet-CTC-0.6b"] = {
        "type": "nemo",
        "model_id": "nvidia/parakeet-ctc-0.6b"
    }


# ==========================================
# Utility Functions
# ==========================================

def load_dataset(jsonl_path, base_dir):
    """Loads transcripts and constructs absolute audio paths."""
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
            # if audio does not exist locally (for example out of subset), skip
            
    return samples

def compute_wer(references, predictions):
    """Computes Word Error Rate (WER) using jiwer."""
    error = jiwer.wer(references, predictions)
    return error

# ==========================================
# Evaluators
# ==========================================

class TransformersEvaluator:
    def __init__(self, model_id):
        self.model_id = model_id
        # We use pipeline for simplicity across architectures (whisper, wav2vec2, etc.)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=DEVICE,
        )

    def transcribe(self, audio_paths):
        # We process seq-wise here, `pipeline` can take batches but requires more setup
        predictions = []
        for path in audio_paths:
            # We assume the user's FLAC files load correctly into transformers
            result = self.pipe(path)
            predictions.append(result["text"].strip().lower())
        return predictions

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
                    # Using the free web api here requires an internet connection
                    text = self.recognizer.recognize_google(audio)
                else:
                    text = ""
                predictions.append(text.lower())
            except sr.UnknownValueError:
                predictions.append("") # Unintelligible
            except sr.RequestError as e:
                predictions.append("")
                print(f"Could not request results from Google Speech Recognition service; {e}")
            except Exception as e:
                predictions.append("")
                print(f"Error processing {path}: {e}")
        return predictions

class NeMoEvaluator:
    def __init__(self, model_id):
        self.model_id = model_id
        print(f"Loading NeMo model {model_id}...")
        self.model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=model_id)
        if DEVICE == "cuda":
            self.model = self.model.cuda()
        self.model.eval()

    def transcribe(self, audio_paths):
        # NeMo has a powerful batch transcription feature `transcribe()` natively
        print(f"NeMo transcribing {len(audio_paths)} files in batch...")
        
        # Depending on the NeMo version and model, it natively returns a tuple or just texts
        transcriptions = self.model.transcribe(paths2audio_files=audio_paths, batch_size=8)
        
        predictions = []
        if type(transcriptions) == tuple and len(transcriptions) > 0:
            texts = transcriptions[0]
        else:
            texts = transcriptions
            
        for text in texts:
             predictions.append(str(text).strip().lower())
        
        return predictions


# ==========================================
# Main Evaluation Loop
# ==========================================

def run_evaluation():
    samples = load_dataset(JSONL_PATH, DATASET_BASE)
    print(f"Loaded {len(samples)} usable samples for evaluation.")
    if len(samples) == 0:
        return

    # Extract isolated lists for batched apis
    audio_paths = [s["audio_path"] for s in samples]
    references = [s["text"] for s in samples]

    results = {}

    for model_name, cfg in MODELS_TO_RUN.items():
        print("=" * 40)
        print(f"Evaluating Model: {model_name}")
        
        # Initialize
        start_time = time.time()
        
        try:
            if cfg["type"] == "transformers":
                evaluator = TransformersEvaluator(cfg["model_id"])
            elif cfg["type"] == "speechrecognition":
                evaluator = SpeechRecognitionEvaluator(cfg["model_id"])
            elif cfg["type"] == "nemo":
                evaluator = NeMoEvaluator(cfg["model_id"])
            else:
                print(f"Unknown evaluator type {cfg['type']}")
                continue
                
            predictions = evaluator.transcribe(audio_paths)
            
            # Compute WER comparing non-empty pairs to avoid jiwer crashing on empty strings
            valid_refs = []
            valid_preds = []
            for ref, pred in zip(references, predictions):
                if ref.strip(): # JiWer cannot handle entirely empty references
                    valid_refs.append(ref)
                    valid_preds.append(pred if pred.strip() else "<empty>")

            wer = compute_wer(valid_refs, valid_preds)
            elapsed = time.time() - start_time
            
            # Store summary
            results[model_name] = {
                "wer": wer,
                "time_sec": round(elapsed, 2)
            }
            
            print(f"-> WER: {wer:.4f}")
            print(f"-> Time taken: {elapsed:.2f}s")
            
            # Show a few examples
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
