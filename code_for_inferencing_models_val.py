# ==========================================
# VALIDATION WER EVALUATION
# ==========================================
# Computes WER on your val_manifest.jsonl file
# Run this BEFORE making a competition submission

import subprocess
import sys
import os

# ======================
# PART A: INSTALLATION
# ======================
def _pip(*args):
    subprocess.check_call(
        [sys.executable, "-m", "pip", *args],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )

print("🧹 Step 1: Cleaning...")
for _ in range(2):
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y",
         "numpy", "scipy", "nemo_toolkit", "lightning",
         "pytorch-lightning", "datasets", "diffusers", "gradio",
         "peft", "sentence-transformers", "transformers",
         "huggingface_hub", "torch", "torchaudio", "torchvision", "numba"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

print("🔧 Step 2: numpy + scipy...")
_pip("install", "--no-cache-dir", "numpy>=2.1,<2.3", "scipy>=1.14,<1.16")

print("⚡ Step 3: PyTorch...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--no-cache-dir",
    "--index-url", "https://download.pytorch.org/whl/cu126",
    "torch>=2.9.0", "torchaudio"
], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

print("📦 Step 4: Dependencies...")
_pip("install", "--no-cache-dir",
     "transformers>=4.57.6,<4.58", "huggingface_hub>=0.30.0",
     "lightning>=2.2.0", "omegaconf>=2.3.0", "hydra-core>=1.3.2",
     "soundfile>=0.12.0", "librosa>=0.10.0", "sentencepiece>=0.2.0",
     "datasets>=2.18.0", "pandas>=2.0.0", "scikit-learn>=1.4.0",
     "loguru>=0.7.0", "jiwer>=3.0.0", "tqdm>=4.60.0",
     "webdataset>=0.2.80", "braceexpand>=0.1.7", "editdistance>=0.6.0")

print("🔥 Step 5: NeMo...")
_pip("install", "--no-cache-dir", "nemo_toolkit[asr]>=2.5.0")

print("🔒 Step 6: Re-pin numpy...")
_pip("install", "--no-cache-dir", "--force-reinstall",
     "numpy>=2.1,<2.3", "scipy>=1.14,<1.16")

print("✅ Installation complete!\n")

# ======================
# PART B: WRITE EVAL SCRIPT
# ======================
EVAL_SCRIPT = "/kaggle/working/compute_val_wer.py"

eval_code = r'''
import warnings, logging
warnings.filterwarnings("ignore", module="numba")
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

import json
import os
import time
from tqdm import tqdm
import torch
import jiwer
from nemo.collections.asr.models import ASRModel
from omegaconf import open_dict, DictConfig

# ==========================================
# CONFIGURATION - UPDATE THESE PATHS
# ==========================================

# Path to your trained .nemo model
MODEL_PATH = "/kaggle/input/notebooks/akarshkumarshukla/baseline-nemo-training0-2293/nemo_adapter_1.1b/ParakeetAdapter1.1B/2026-03-21_19-35-48/checkpoints/model_final.nemo"

# Path to validation manifest (JSONL format with audio_filepath, text, duration)
VAL_MANIFEST = "/kaggle/input/datasets/akarshkumarshukla/vel-mani/val_manifest.jsonl"

# Inference settings
BATCH_SIZE = 60  # Adjust based on GPU memory
NUM_WORKERS = 4

# ==========================================
# LOAD MODEL
# ==========================================
print("=" * 70)
print("VALIDATION WER EVALUATION")
print("=" * 70)

print(f"\nModel: {os.path.basename(MODEL_PATH)}")
if os.path.exists(MODEL_PATH):
    print(f"Size: {os.path.getsize(MODEL_PATH) / 1024**3:.2f} GB")
else:
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print(f"Val manifest: {VAL_MANIFEST}")
if not os.path.exists(VAL_MANIFEST):
    raise FileNotFoundError(f"Validation manifest not found: {VAL_MANIFEST}")

print(f"\ntorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nLoading model...")
t0 = time.time()
model = ASRModel.restore_from(MODEL_PATH, map_location="cuda")

# Disable CUDA graph decoder (can cause issues)
with open_dict(model.cfg):
    if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
        model.cfg.decoding.greedy.use_cuda_graph_decoder = False
if hasattr(model, "change_decoding_strategy"):
    model.change_decoding_strategy(model.cfg.decoding)

model.eval()
model.freeze()
print(f"Model loaded in {time.time()-t0:.1f}s")

# ==========================================
# PATCH TRANSCRIBE TO DISABLE LHOTSE
# ==========================================
def _patched_transcribe_dataloader(config):
    """Disable lhotse in transcribe dataloader to avoid channel_selector issues"""
    if "manifest_filepath" in config:
        manifest_filepath = config["manifest_filepath"]
        batch_size = config["batch_size"]
    else:
        manifest_filepath = os.path.join(config["temp_dir"], "manifest.json")
        batch_size = min(config["batch_size"], len(config["paths2audio_files"]))

    dl_config = {
        "use_lhotse": False,
        "manifest_filepath": manifest_filepath,
        "sample_rate": model.preprocessor._sample_rate,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": min(batch_size, NUM_WORKERS),
        "pin_memory": False,
        "channel_selector": config.get("channel_selector", None),
        "use_start_end_token": model.cfg.validation_ds.get("use_start_end_token", False),
    }
    if config.get("augmentor"):
        dl_config["augmentor"] = config["augmentor"]
    return model._setup_dataloader_from_config(config=DictConfig(dl_config))

model._setup_transcribe_dataloader = _patched_transcribe_dataloader
print("Patched transcribe to disable Lhotse")

# ==========================================
# LOAD VALIDATION DATA
# ==========================================
print("\nLoading validation manifest...")

samples = []
with open(VAL_MANIFEST, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            # Handle different manifest formats
            audio_path = data.get("audio_filepath") or data.get("audio_path", "")
            text = data.get("text") or data.get("orthographic_text", "")
            text = text.strip().lower()
            duration = float(data.get("duration") or data.get("audio_duration_sec", 0))
            
            if audio_path and text and os.path.exists(audio_path):
                samples.append({
                    "path": audio_path,
                    "ref": text,
                    "dur": duration
                })
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            continue

print(f"Loaded {len(samples):,} validation samples")
if len(samples) == 0:
    raise RuntimeError("No valid samples found in validation manifest!")

total_duration = sum(s["dur"] for s in samples)
print(f"Total duration: {total_duration/3600:.2f} hours")

# ==========================================
# TRANSCRIBE IN BATCHES
# ==========================================
audio_files = [s["path"] for s in samples]
references = [s["ref"] for s in samples]

print(f"\nTranscribing {len(audio_files):,} files (batch_size={BATCH_SIZE})...")
t0 = time.time()

# Process in chunks for better progress tracking
all_predictions = []
chunk_size = BATCH_SIZE * 10  # Process 10 batches at a time for progress updates

for i in tqdm(range(0, len(audio_files), chunk_size), desc="Transcribing"):
    chunk_files = audio_files[i:i+chunk_size]
    
    raw = model.transcribe(
        chunk_files,
        batch_size=BATCH_SIZE,
        channel_selector="average",
        verbose=False,
    )
    
    if isinstance(raw, tuple):
        raw = raw[0]
    
    chunk_preds = [h.text if hasattr(h, "text") else str(h) for h in raw]
    all_predictions.extend(chunk_preds)

predictions = all_predictions
elapsed = time.time() - t0

print(f"\nTranscription complete!")
print(f"Time: {elapsed:.1f}s ({len(audio_files)/elapsed:.1f} files/sec)")
print(f"RTF: {elapsed/total_duration:.3f}x")

# ==========================================
# COMPUTE WER
# ==========================================
print("\n" + "=" * 70)
print("COMPUTING WER")
print("=" * 70)

# Normalize predictions
predictions_normalized = [p.strip().lower() for p in predictions]

# Compute overall WER
wer = jiwer.wer(references, predictions_normalized)
cer = jiwer.cer(references, predictions_normalized)

# Compute detailed metrics
measures = jiwer.compute_measures(references, predictions_normalized)

print(f"\n📊 VALIDATION RESULTS:")
print(f"   WER:  {wer:.4f} ({wer*100:.2f}%)")
print(f"   CER:  {cer:.4f} ({cer*100:.2f}%)")
print(f"\n📈 DETAILED METRICS:")
print(f"   Substitutions: {measures['substitutions']:,}")
print(f"   Deletions:     {measures['deletions']:,}")
print(f"   Insertions:    {measures['insertions']:,}")
print(f"   Hits:          {measures['hits']:,}")

# Exact match accuracy
exact_matches = sum(1 for r, p in zip(references, predictions_normalized) if r == p)
print(f"\n✅ Exact matches: {exact_matches:,}/{len(samples):,} ({100*exact_matches/len(samples):.1f}%)")

# ==========================================
# ERROR ANALYSIS (Sample of worst predictions)
# ==========================================
print("\n" + "=" * 70)
print("ERROR ANALYSIS (10 worst predictions)")
print("=" * 70)

# Calculate per-sample WER
sample_wers = []
for i, (ref, pred) in enumerate(zip(references, predictions_normalized)):
    try:
        sample_wer = jiwer.wer([ref], [pred])
    except:
        sample_wer = 1.0
    sample_wers.append((i, sample_wer, ref, pred))

# Sort by WER (worst first)
sample_wers.sort(key=lambda x: x[1], reverse=True)

for idx, (i, swer, ref, pred) in enumerate(sample_wers[:10]):
    print(f"\n[{idx+1}] WER: {swer:.2f} (sample #{i})")
    print(f"  REF:  {ref[:100]}{'...' if len(ref) > 100 else ''}")
    print(f"  PRED: {pred[:100]}{'...' if len(pred) > 100 else ''}")

# ==========================================
# SAMPLE GOOD PREDICTIONS
# ==========================================
print("\n" + "=" * 70)
print("SAMPLE GOOD PREDICTIONS (10 best)")
print("=" * 70)

# Sort by WER (best first)
sample_wers.sort(key=lambda x: x[1])

for idx, (i, swer, ref, pred) in enumerate(sample_wers[:10]):
    print(f"\n[{idx+1}] WER: {swer:.2f}")
    print(f"  REF:  {ref[:100]}{'...' if len(ref) > 100 else ''}")
    print(f"  PRED: {pred[:100]}{'...' if len(pred) > 100 else ''}")

# ==========================================
# GPU MEMORY & SPEED STATS
# ==========================================
if torch.cuda.is_available():
    peak = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n💾 GPU Memory: {peak:.1f}/{total:.0f} GB ({peak/total*100:.0f}%)")

files_per_sec = len(audio_files) / elapsed
print(f"⚡ Speed: {files_per_sec:.1f} files/sec")
print(f"⏱️  Est. 100k files: {100000/files_per_sec/60:.0f} min")

# ==========================================
# SAVE RESULTS
# ==========================================
results_path = "/kaggle/working/val_wer_results.json"
results = {
    "model_path": MODEL_PATH,
    "val_manifest": VAL_MANIFEST,
    "num_samples": len(samples),
    "total_duration_hours": total_duration / 3600,
    "wer": wer,
    "cer": cer,
    "exact_match_rate": exact_matches / len(samples),
    "substitutions": measures["substitutions"],
    "deletions": measures["deletions"],
    "insertions": measures["insertions"],
    "hits": measures["hits"],
    "inference_time_sec": elapsed,
    "files_per_sec": files_per_sec,
}

with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n📁 Results saved to: {results_path}")

# Save predictions for analysis
predictions_path = "/kaggle/working/val_predictions.jsonl"
with open(predictions_path, "w") as f:
    for i, (ref, pred) in enumerate(zip(references, predictions_normalized)):
        f.write(json.dumps({
            "idx": i,
            "audio_path": samples[i]["path"],
            "reference": ref,
            "prediction": pred,
            "wer": sample_wers[i][1] if i < len(sample_wers) else None
        }) + "\n")
print(f"📁 Predictions saved to: {predictions_path}")

# ==========================================
# SUMMARY
# ==========================================
print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)
print(f"   Model:         {os.path.basename(MODEL_PATH)}")
print(f"   Val samples:   {len(samples):,}")
print(f"   Val duration:  {total_duration/3600:.2f} hours")
print(f"   WER:           {wer*100:.2f}%")
print(f"   CER:           {cer*100:.2f}%")
print(f"   Exact match:   {100*exact_matches/len(samples):.1f}%")
print("=" * 70)
print("✅ VALIDATION COMPLETE!")
print("=" * 70)
'''

with open(EVAL_SCRIPT, "w") as f:
    f.write(eval_code)

print("=" * 60)
print("🧪 Running validation WER evaluation...")
print("=" * 60)

result = subprocess.run(
    [sys.executable, EVAL_SCRIPT],
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
    stdout=None, stderr=subprocess.STDOUT,
)

if result.returncode != 0:
    print(f"\n❌ Evaluation failed with code {result.returncode}")
else:
    print(f"\n✅ Evaluation complete!")
