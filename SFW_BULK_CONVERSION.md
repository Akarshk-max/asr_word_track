# Full LibriSpeech clean-100 + clean-360 SFW conversion

`build_sfw_manifest.py` supports `--librispeech-data-root` and `--librispeech-splits` (see `--help`).

## Kaggle command

Point `--librispeech-data-root` at the folder that **contains** `train-clean-100` and `train-clean-360` (your tree: `.../LibriSpeech`).

```bash
pip install -q soundfile whisper-normalizer librosa

cd /kaggle/working   # or wherever sfw_augment.py + build_sfw_manifest.py live

python build_sfw_manifest.py \
  --librispeech-data-root "/kaggle/input/YOUR_DATASET/LibriSpeech" \
  --librispeech-splits train-clean-100 train-clean-360 \
  --output-wav-dir /kaggle/working/sfw_audio \
  --output-manifest /kaggle/working/sfw_train_clean_manifest.jsonl \
  --num-workers 4 \
  --skip-existing \
  --seed 0
```

- **Whisper-normalized** `text` is written for every row (same as before).
- **`--skip-existing`**: resume after interrupt; rewrites manifest for existing wavs.
- **`--num-workers`**: parallel CPU workers (try 4–8). Use `1` if multiprocessing fails on your platform.
- **Runtime**: clean-100 ≈ 100 h + clean-360 ≈ 360 h of speech is **very large**; Griffin–Lim per utterance is slow—expect **days** of CPU time unless you shard across jobs or reduce corpus.

## Alternative: two runs + merge (same as single `--librispeech-data-root`)

You can also run per split and concatenate:

```bash
python build_sfw_manifest.py \
  --librispeech-root "/kaggle/input/.../LibriSpeech/train-clean-100" \
  --output-wav-dir /kaggle/working/sfw_audio \
  --output-manifest /kaggle/working/sfw_part100.jsonl \
  --skip-existing

python build_sfw_manifest.py \
  --librispeech-root "/kaggle/input/.../LibriSpeech/train-clean-360" \
  --output-wav-dir /kaggle/working/sfw_audio \
  --output-manifest /kaggle/working/sfw_part360.jsonl \
  --skip-existing

cat sfw_part100.jsonl sfw_part360.jsonl > sfw_train_clean_manifest.jsonl
```

Same output directory keeps **unique** utterance IDs across splits (LibriSpeech IDs are global).
