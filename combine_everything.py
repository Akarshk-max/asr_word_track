import os
import json
import shutil
import pandas as pd

# Paths
ASR_DATA_DIR = r"C:\Users\Amogh Shukla\Downloads\ASR\asr_data"
ASR_JSONL = os.path.join(ASR_DATA_DIR, "train_word_transcripts.jsonl")

TALKBANK_DIR = r"C:\Users\Amogh Shukla\Downloads\ASR\talkbank_data"
TALKBANK_JSON = r"C:\Users\Amogh Shukla\Downloads\ASR\talkbank_data\train_word_transcripts.jsonl"

# Output Paths
OUTPUT_DIR = "./combined_asr_dataset"
OUTPUT_AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
OUTPUT_METADATA = os.path.join(OUTPUT_DIR, "metadata.csv")

os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

# 1. Index all audio files
print("Indexing audio files across all directories...")
audio_index = {}
for search_dir in [ASR_DATA_DIR, TALKBANK_DIR]:
    if not os.path.isdir(search_dir):
        continue
    for root, dirs, files in os.walk(search_dir):
        for f in files:
            if f.endswith(".flac"):
                audio_index[f] = os.path.join(root, f)

print(f"Indexed {len(audio_index)} total audio files.")

def find_audio(audio_path_field):
    filename = os.path.basename(audio_path_field)
    return filename, audio_index.get(filename, None)

metadata_records = []

# 2. Parse ASR JSONL
print("Parsing ASR JSONL data...")
if os.path.exists(ASR_JSONL):
    with open(ASR_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            data = json.loads(line)
            
            filename, full_path = find_audio(data.get("audio_path", ""))
            if full_path:
                text = data.get("orthographic_text", "").strip().lower()
                age = data.get("age_bucket", "unknown")
                metadata_records.append({
                    "file_name": os.path.join("audio", filename),
                    "text": text,
                    "age_bucket": age,
                    "source": "asr",
                    "original_path": full_path
                })
print(f"  ASR records: {len([r for r in metadata_records if r['source'] == 'asr'])}")

# 3. Parse TalkBank JSONL
print("Parsing TalkBank JSONL data...")
if os.path.exists(TALKBANK_JSON):
    with open(TALKBANK_JSON, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                filename, full_path = find_audio(data.get("audio_path", ""))
                if full_path:
                    text = data.get("orthographic_text", "").strip().lower()
                    age = data.get("age_bucket", "unknown")
                    if text:
                        metadata_records.append({
                            "file_name": os.path.join("audio", filename),
                            "text": text,
                            "age_bucket": age,
                            "source": "talkbank",
                            "original_path": full_path
                        })
            except json.JSONDecodeError:
                pass
print(f"  TalkBank records: {len([r for r in metadata_records if r['source'] == 'talkbank'])}")

# 4. Copy Audio Files and Save Metadata
print(f"\nFound {len(metadata_records)} valid audio-text pairs.")
print("Copying audio files to combined directory (this may take a while)...")

final_metadata = []
for i, record in enumerate(metadata_records):
    orig_path = record["original_path"]
    dest_path = os.path.join(OUTPUT_DIR, record["file_name"])
    
    if not os.path.exists(dest_path):
        shutil.copy2(orig_path, dest_path)
    
    final_metadata.append({
        "file_name": record["file_name"],
        "text": record["text"],
        "age_bucket": record["age_bucket"],
        "source": record["source"]
    })
    
    if (i + 1) % 10000 == 0:
        print(f"  Processed {i + 1}/{len(metadata_records)} files...")

# Save HuggingFace Imagefolder format metadata
df = pd.DataFrame(final_metadata)
df.to_csv(OUTPUT_METADATA, index=False)

print("\nDone! Dataset structure prepared at:")
print(f"  {OUTPUT_DIR}/")
print(f"    - audio/       ({len(os.listdir(OUTPUT_AUDIO_DIR))} files)")
print(f"    - metadata.csv ({len(df)} rows)")
print("\nYou can now download or directly use this folder in Kaggle.")
