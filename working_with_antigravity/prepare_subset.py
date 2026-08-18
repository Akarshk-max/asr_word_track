import json
import os
import shutil

NUM_SAMPLES = 100
JSONL_PATH = "train_word_transcripts.jsonl"
OUT_DIR = "eval_subset"
OUT_JSONL = os.path.join(OUT_DIR, "eval_transcripts.jsonl")

# The user mentioned audio_part_0 and audio_part_1 folders.
# Let's check both for the required audio files.
AUDIO_DIRS = ["audio_part_0/audio", "audio_part_1/audio"]

def prep_eval_subset():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        
    out_audio_dir = os.path.join(OUT_DIR, "audio")
    if not os.path.exists(out_audio_dir):
        os.makedirs(out_audio_dir)

    found_samples = 0
    with open(JSONL_PATH, "r", encoding="utf-8") as f_in, \
         open(OUT_JSONL, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            if found_samples >= NUM_SAMPLES:
                break
                
            data = json.loads(line)
            # audio_path in jsonl looks like "audio/U_00003c3ae1c35c6f.flac"
            rel_audio_path = data["audio_path"]
            filename = os.path.basename(rel_audio_path)
            
            # Find the actual file in the unzipped parts
            src_path = None
            for adir in AUDIO_DIRS:
                # The zip seems to contain files directly in audio/, or maybe without it.
                # Let's try both.
                candidate_1 = os.path.join(adir, filename)
                # If they unzipped and the folder structure is audio_part_0/audio/...
                if os.path.exists(candidate_1):
                    src_path = candidate_1
                    break
            
            if src_path:
                # Copy file
                dest_path = os.path.join(out_audio_dir, filename)
                shutil.copy2(src_path, dest_path)
                
                # Update jsonl path and write
                data["audio_path"] = f"audio/{filename}"  # Relative to OUT_DIR
                f_out.write(json.dumps(data) + "\n")
                found_samples += 1
            else:
                # File not found in the unzipped directories we checked, skip for now.
                pass

    print(f"Prepared {found_samples} samples in {OUT_DIR}/")

if __name__ == "__main__":
    prep_eval_subset()
