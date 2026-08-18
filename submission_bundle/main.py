import os
import json
from pathlib import Path
import torch
import librosa
from transformers import AutoProcessor, AutoModelForCTC

# ---------------------------------------------------------
# Step 2: Main DrivenData Entrypoint
# ---------------------------------------------------------
# This script runs INSIDE the container. It has:
# - NO internet access
# - Access to /code_execution/data (read-only audio + metadata)
# - Access to /code_execution/submission (write output here)
# - Its current working directory is /code_execution.

def transcribe():
    # 1. Setup paths according to container rules
    data_dir = Path("data")
    manifest_path = data_dir / "utterance_metadata.jsonl"
    submission_path = Path("submission") / "submission.jsonl"
    
    # We load our bundled model from the exact folder name we zipped
    # Because main.py is in the root of the zip, it unzips to /code_execution/src/
    # wait, the prompt says: "Your code will be unzipped into ./src/"
    # If main.py is at the root of the zip, it unzips to ./src/main.py
    # So our bundled weights exist at ./src/model_weights/
    src_dir = Path(__file__).parent.resolve()
    model_weights_dir = src_dir / "model_weights"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Container executing on device: {device}")
    
    # 2. Load model fully offline from the downloaded folder
    print(f"Loading offline weights from {model_weights_dir}...")
    processor = AutoProcessor.from_pretrained(model_weights_dir)
    model = AutoModelForCTC.from_pretrained(model_weights_dir).to(device)
    model.eval()

    # 3. Read metadata to find all tests
    print(f"Loading manifest: {manifest_path}")
    test_items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            test_items.append(json.loads(line))
            
    # Sort by duration to optimize batched inference
    test_items.sort(key=lambda x: x.get("audio_duration_sec", 0.0), reverse=True)
    
    print(f"Processing {len(test_items)} audio files...")
    
    # 4. Open submission file for writing
    with open(submission_path, "w", encoding="utf-8") as fw:
        
        # 5. Process inferences
        # For simplicity in this baseline, we process 1x1. You can batch this 
        # heavily (e.g., BATCH_SIZE=8) to improve A100 throughput.
        for item in test_items:
            # Container path is strictly "data/audio/whatever.flac"
            audio_file = data_dir / item["audio_path"]
            
            # Load and resample to 16kHz
            audio, _ = librosa.load(audio_file, sr=16000)
            
            # Extract features
            inputs = processor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).to(device)
            
            # Infer
            with torch.no_grad():
                logits = model(**inputs).logits
            
            # CTC Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True, 
                group_tokens=True
            )[0].strip().lower()
            
            # Create valid output JSON array line
            out_obj = {
                "utterance_id": item["utterance_id"],
                "orthographic_text": transcription
            }
            fw.write(json.dumps(out_obj) + "\n")

    print(f"Successfully wrote {len(test_items)} predictions to {submission_path}")

if __name__ == "__main__":
    transcribe()
