from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from jiwer import cer, wer
from nemo.collections.asr.models import ASRModel


def _load_manifest(path: str, max_samples: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            audio = item.get("audio_filepath")
            text = item.get("text", "")
            if not audio:
                continue
            rows.append(
                {
                    "audio_filepath": str(audio),
                    "text": str(text),
                    "duration": float(item.get("duration", 0.0) or 0.0),
                }
            )
            if max_samples > 0 and len(rows) >= max_samples:
                break
    if not rows:
        raise RuntimeError(f"No valid rows found in manifest: {path}")
    return rows


def _maybe_normalize(s: str, normalize: bool) -> str:
    s = str(s)
    if not normalize:
        return s
    try:
        from whisper_normalizer.english import EnglishTextNormalizer

        normalizer = EnglishTextNormalizer()
        return normalizer(s).strip()
    except Exception:
        return s.lower().strip()


def _safe_load_adapters(model: ASRModel, adapter_path: str | None) -> None:
    if not adapter_path:
        return
    if not os.path.isfile(adapter_path):
        print(f"[warn] adapter file not found, skipping: {adapter_path}")
        return
    try:
        model.load_adapters(adapter_path)
        print(f"Loaded adapters from: {adapter_path}")
    except Exception as e:
        print(f"[warn] Could not load adapters from {adapter_path}: {e}")
        print("[warn] Continuing with .nemo weights only.")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model: {args.model_nemo}")
    model = ASRModel.restore_from(args.model_nemo, map_location=device)
    model.eval()
    _safe_load_adapters(model, args.adapter_pt)

    rows = _load_manifest(args.val_manifest, max_samples=args.max_samples)
    print(f"Loaded {len(rows):,} validation samples")

    audio_paths = [r["audio_filepath"] for r in rows]
    refs_raw = [r["text"] for r in rows]

    print("Running transcription...")
    hyps = model.transcribe(audio_paths, batch_size=args.batch_size)
    hyps = [h if isinstance(h, str) else str(h) for h in hyps]

    refs = [_maybe_normalize(x, args.normalize_text) for x in refs_raw]
    preds = [_maybe_normalize(x, args.normalize_text) for x in hyps]

    score_wer = float(wer(refs, preds))
    score_cer = float(cer(refs, preds))

    pred_path = out_dir / "val_predictions.csv"
    metrics_path = out_dir / "val_metrics.json"

    df = pd.DataFrame(
        {
            "audio_filepath": audio_paths,
            "reference": refs_raw,
            "prediction": hyps,
            "reference_norm": refs,
            "prediction_norm": preds,
        }
    )
    df.to_csv(pred_path, index=False)

    metrics = {
        "model_nemo": args.model_nemo,
        "adapter_pt": args.adapter_pt or "",
        "val_manifest": args.val_manifest,
        "num_samples": len(rows),
        "batch_size": args.batch_size,
        "normalize_text": bool(args.normalize_text),
        "wer": score_wer,
        "cer": score_cer,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nValidation complete")
    print(f"WER: {score_wer:.4f}")
    print(f"CER: {score_cer:.4f}")
    print(f"Predictions: {pred_path}")
    print(f"Metrics:     {metrics_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeMo ASR inference on val manifest JSONL")
    p.add_argument(
        "--model_nemo",
        type=str,
        default="/kaggle/working/nemo_adapter_1.1b/checkpoints/model_final.nemo",
        help="Path to saved .nemo model",
    )
    p.add_argument(
        "--adapter_pt",
        type=str,
        default="",
        help="Optional adapter .pt path (if you want to explicitly load adapters)",
    )
    p.add_argument(
        "--val_manifest",
        type=str,
        required=True,
        help="Path to val manifest .jsonl with audio_filepath + text",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="/kaggle/working/val_infer_report",
        help="Directory for outputs",
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="0 = all samples, otherwise evaluate first N rows",
    )
    p.add_argument(
        "--normalize_text",
        action="store_true",
        help="Apply whisper_normalizer (or lowercase fallback) to ref/pred before WER/CER",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
