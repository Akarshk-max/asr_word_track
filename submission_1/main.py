"""
Inference: Children's Speech Recognition Challenge (Word Track)
Model: Parakeet-TDT-1.1B with trained adapter for children's speech

OOM-safe adaptive batching + 4 waveform variants per utterance and jiwer-based
consensus (mean pairwise WER with each candidate as reference).

Requires: torch, torchaudio, soundfile, jiwer, loguru, nemo_toolkit (typical NeMo env).
"""

import json
import os
import tempfile
import uuid
from pathlib import Path

import jiwer
import soundfile as sf
import torch
import torchaudio
from loguru import logger

SRC_DIR = Path(__file__).parent.resolve()
DATA_DIR = Path("data")
SUBMISSION_DIR = Path("submission")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = SRC_DIR / "ParakeetAdapter.nemo"
METADATA_PATH = DATA_DIR / "utterance_metadata.jsonl"
FORMAT_PATH = DATA_DIR / "submission_format.jsonl"
OUTPUT_PATH = SUBMISSION_DIR / "submission.jsonl"

NUM_VARIANTS = 4
SPEED_FACTOR_UP = 1.05
SPEED_FACTOR_DOWN = 0.95
PITCH_SEMITONES_VARIANT = -2


def get_batch_size(duration_sec: float) -> int:
    """
    Adaptive batch size by clip duration (utterance count before ×4 variants).
    Conservative to avoid OOM on long utterances.
    """
    if duration_sec > 20:
        return 4
    elif duration_sec > 15:
        return 10
    elif duration_sec > 10:
        return 16
    elif duration_sec > 7:
        return 18
    elif duration_sec > 5:
        return 22
    elif duration_sec > 3:
        return 24
    elif duration_sec > 2:
        return 28
    else:
        return 30


def build_adaptive_batches(items):
    """
    Items must already be sorted by duration ascending.
    Utterance batch size is get_batch_size // 4 so 4 variants fit ~prior GPU budget.
    """
    batches = []
    i = 0
    n = len(items)

    while i < n:
        duration = items[i].get("audio_duration_sec", 0.0)
        bs = max(1, get_batch_size(duration) // NUM_VARIANTS)
        batch = items[i : i + bs]
        batches.append(batch)
        i += bs

    return batches


def _normalize_peak(waveform: torch.Tensor) -> torch.Tensor:
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak * 0.95
    return waveform


def load_audio_mono(path: str) -> tuple[torch.Tensor, int]:
    data, sr = sf.read(path, dtype="float32")
    t = torch.from_numpy(data)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    else:
        t = t.T
        t = t.mean(dim=0, keepdim=True)
    return t, sr


def pitch_shift(
    waveform: torch.Tensor, sr: int, n_steps: int, device: str
) -> torch.Tensor:
    if n_steps == 0:
        return waveform
    return torchaudio.functional.pitch_shift(
        waveform.to(device), sr, n_steps=n_steps
    ).cpu()


def speed_change(waveform: torch.Tensor, sr: int, factor: float) -> torch.Tensor:
    if abs(factor - 1.0) < 0.005:
        return waveform
    orig_freq = int(sr * factor)
    resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=sr)
    return resampler(waveform)


def write_temp_wav(waveform: torch.Tensor, sr: int, tmpdir: str) -> str:
    path = os.path.join(tmpdir, f"{uuid.uuid4().hex}.wav")
    arr = waveform.squeeze(0).numpy()
    sf.write(path, arr, sr, subtype="FLOAT")
    return path


def build_four_variant_paths(src_path: str, tmpdir: str, device: str) -> list[str]:
    w, sr = load_audio_mono(src_path)
    paths: list[str] = []
    w0 = _normalize_peak(w.clone())
    paths.append(write_temp_wav(w0, sr, tmpdir))
    w1 = _normalize_peak(speed_change(w.clone(), sr, SPEED_FACTOR_UP))
    paths.append(write_temp_wav(w1, sr, tmpdir))
    w2 = _normalize_peak(speed_change(w.clone(), sr, SPEED_FACTOR_DOWN))
    paths.append(write_temp_wav(w2, sr, tmpdir))
    w3 = pitch_shift(w.clone(), sr, PITCH_SEMITONES_VARIANT, device)
    w3 = _normalize_peak(w3)
    paths.append(write_temp_wav(w3, sr, tmpdir))
    return paths


def consensus_pick(hyps: list[str]) -> str:
    """Pick hypothesis minimizing mean WER(h_i, h_j) over j != i (h_i as reference)."""
    hyps = [h or "" for h in hyps]
    n = len(hyps)
    if n <= 1:
        return hyps[0] if hyps else ""
    best_i = 0
    best_score = float("inf")
    for i in range(n):
        others = [hyps[j] for j in range(n) if j != i]
        scores = [jiwer.wer([hyps[i]], [o]) for o in others]
        s = sum(scores) / len(scores)
        if s < best_score:
            best_score = s
            best_i = i
    return hyps[best_i]


def hyp_to_text(hyp) -> str:
    text = hyp.text if hasattr(hyp, "text") else str(hyp)
    return text if text else ""


def transcribe_paths(model, paths: list[str], batch_size: int):
    raw = model.transcribe(
        paths,
        batch_size=batch_size,
        channel_selector="average",
        verbose=False,
    )
    if isinstance(raw, tuple):
        raw = raw[0]
    return raw


def transcribe_with_consensus(
    model, batch: list[dict], data_dir: Path, tmpdir: str, device: str
) -> dict[str, str]:
    """Returns utterance_id -> chosen text for all items in batch."""
    batch_ids = [item["utterance_id"] for item in batch]
    all_paths: list[str] = []

    for item in batch:
        src = str(data_dir / item["audio_path"])
        try:
            paths = build_four_variant_paths(src, tmpdir, device)
            all_paths.extend(paths)
        except Exception as e:
            logger.warning(
                f"Variant build failed for {item['utterance_id']} ({src}): {e}. "
                "Using original path ×4."
            )
            all_paths.extend([src] * NUM_VARIANTS)

    audio_bs = len(all_paths)
    raw = transcribe_paths(model, all_paths, audio_bs)
    out: dict[str, str] = {}
    for row, uid in enumerate(batch_ids):
        hyps = [hyp_to_text(raw[row * NUM_VARIANTS + c]) for c in range(NUM_VARIANTS)]
        out[uid] = consensus_pick(hyps)
    return out


def fallback_one_utterance(
    model, item: dict, data_dir: Path, device: str
) -> str:
    """OOM / error fallback: 4 variants in temp dir, then per-variant if needed."""
    src = str(data_dir / item["audio_path"])
    uid = item["utterance_id"]

    with tempfile.TemporaryDirectory(prefix="asr_fb_") as tmpdir:
        try:
            paths = build_four_variant_paths(src, tmpdir, device)
        except Exception as e:
            logger.warning(f"Fallback variant build failed for {uid}: {e}")
            paths = None

        if paths is None:
            try:
                raw = transcribe_paths(model, [src], 1)
                return hyp_to_text(raw[0])
            except Exception as e2:
                logger.warning(f"Single-file fallback failed for {uid}: {e2}")
                return ""

        try:
            raw = transcribe_paths(model, paths, NUM_VARIANTS)
            hyps = [hyp_to_text(raw[c]) for c in range(NUM_VARIANTS)]
            return consensus_pick(hyps)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning(
                f"OOM on 4-variant transcribe for {uid}; trying per-variant."
            )
        except Exception as e:
            logger.warning(f"4-variant transcribe failed for {uid}: {e}")

        texts: list[str] = []
        for vp in paths:
            try:
                raw = transcribe_paths(model, [vp], 1)
                texts.append(hyp_to_text(raw[0]))
            except Exception as e:
                logger.warning(f"Single-variant failed for {uid} ({vp}): {e}")
                texts.append("")

        if not any(t.strip() for t in texts):
            return ""
        while len(texts) < NUM_VARIANTS:
            texts.append(texts[-1])
        return consensus_pick(texts[:NUM_VARIANTS])


def main():
    logger.info(f"Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    logger.info(f"Loading model from {MODEL_PATH}...")
    from nemo.collections.asr.models import ASRModel
    from omegaconf import open_dict, DictConfig

    model = ASRModel.restore_from(str(MODEL_PATH), map_location="cuda")

    with open_dict(model.cfg):
        if hasattr(model.cfg, "decoding") and hasattr(model.cfg.decoding, "greedy"):
            model.cfg.decoding.greedy.use_cuda_graph_decoder = False

    if hasattr(model, "change_decoding_strategy"):
        model.change_decoding_strategy(model.cfg.decoding)

    model.eval()
    model.freeze()

    # Disable lhotse in transcribe
    def _patched_transcribe_dataloader(config):
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
            "num_workers": min(batch_size, 8),
            "pin_memory": False,
            "channel_selector": config.get("channel_selector", None),
            "use_start_end_token": model.cfg.validation_ds.get(
                "use_start_end_token", False
            ),
        }
        if config.get("augmentor"):
            dl_config["augmentor"] = config["augmentor"]
        return model._setup_dataloader_from_config(config=DictConfig(dl_config))

    model._setup_transcribe_dataloader = _patched_transcribe_dataloader
    logger.info("Model ready (Lhotse patched, multi-variant + consensus)")

    # Load metadata
    with open(METADATA_PATH, "r") as f:
        items = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Utterances: {len(items)}")

    items.sort(key=lambda x: x.get("audio_duration_sec", 0))

    batches = build_adaptive_batches(items)
    logger.info(f"Built {len(batches)} adaptive batches (×{NUM_VARIANTS} paths each)")

    predictions = {}
    processed = 0
    total = len(items)

    for batch_idx, batch in enumerate(batches):
        batch_ids = [item["utterance_id"] for item in batch]
        duration = batch[0].get("audio_duration_sec", 0.0)
        batch_size = len(batch)

        if batch_idx % 200 == 0 or batch_idx < 20:
            logger.info(
                f"Batch {batch_idx+1}/{len(batches)} | "
                f"duration≈{duration:.2f}s | utterances={batch_size} | "
                f"processed={processed}/{total}"
            )

        try:
            with tempfile.TemporaryDirectory(prefix="asr_var_") as tmpdir:
                preds = transcribe_with_consensus(
                    model, batch, DATA_DIR, tmpdir, device
                )
            for uid in batch_ids:
                predictions[uid] = preds.get(uid, "")
            processed += batch_size

        except torch.OutOfMemoryError:
            logger.warning(
                f"OOM at batch {batch_idx+1} (duration≈{duration:.2f}s, "
                f"utterances={batch_size}). Retrying per-utterance 4-variant / single."
            )
            torch.cuda.empty_cache()
            for item in batch:
                predictions[item["utterance_id"]] = fallback_one_utterance(
                    model, item, DATA_DIR, device
                )
                processed += 1
            torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(
                f"Batch {batch_idx+1} failed unexpectedly: {e}. "
                f"Falling back per-utterance."
            )
            torch.cuda.empty_cache()
            for item in batch:
                predictions[item["utterance_id"]] = fallback_one_utterance(
                    model, item, DATA_DIR, device
                )
                processed += 1
            torch.cuda.empty_cache()

    logger.info(f"Transcribed {len(predictions)} utterances")

    logger.info(f"Writing to {OUTPUT_PATH}...")
    with open(FORMAT_PATH, "r") as fr, open(OUTPUT_PATH, "w", encoding="utf-8") as fw:
        for line in fr:
            item = json.loads(line)
            item["orthographic_text"] = predictions.get(item["utterance_id"], "")
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.success("Done!")


if __name__ == "__main__":
    main()
