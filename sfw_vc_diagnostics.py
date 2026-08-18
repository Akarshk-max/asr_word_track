#!/usr/bin/env python3
"""
SFW vs LibriSpeech VC-style diagnostics (mel plots, envelope proxy, F0 stats).

Kaggle: attach LibriSpeech + your SFW wav folder, then e.g.
  python sfw_vc_diagnostics.py \\
    --librispeech-root /kaggle/input/.../train-clean-100 \\
    --sfw-wav-dir /kaggle/working/sfw_audio \\
    --out-dir /kaggle/working/sfw_vc_plots \\
    --n 10

Interpretation: high mel_mae_db / mel_rmse_db suggests envelope / formant path or
phase reconstruction issues; small median F0 shift with large mel error points at
spectral smearing more than gross pitch. Optional --with-pyworld adds WORLD log-SP
distance (not paper MCD; swap in mgc+DTW for VERSA-style MCD if you install tools).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import librosa
    import librosa.display
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install librosa") from e

from sfw_before_after_preview import find_flac, load_before_after
from sfw_augment import SFW_SAMPLE_RATE

# Match SFW frontend-ish framing for apples-to-apples mel comparison
MEL_KW = dict(
    sr=SFW_SAMPLE_RATE,
    n_fft=512,
    hop_length=160,
    n_mels=128,
    fmin=0.0,
    fmax=SFW_SAMPLE_RATE / 2.0,
    power=2.0,
)

PYIN_KW = dict(
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C7"),
    sr=SFW_SAMPLE_RATE,
    frame_length=2048,
    hop_length=160,
)


def collect_pairs(
    librispeech_root: Path,
    sfw_wav_dir: Path,
    n: int,
    utterance: str | None,
) -> list[tuple[str, Path, Path]]:
    wav_dir = Path(sfw_wav_dir).resolve()
    root = Path(librispeech_root).resolve()
    wavs = sorted(wav_dir.glob("*.wav"))
    if utterance:
        st = utterance.replace(".wav", "")
        cand = wav_dir / f"{st}.wav"
        wavs = [cand] if cand.is_file() else []
    out: list[tuple[str, Path, Path]] = []
    for wav in wavs:
        stem = wav.stem
        flac = find_flac(root, stem)
        if flac is None:
            print(f"[skip] no flac for {stem}", file=sys.stderr)
            continue
        out.append((stem, flac, wav))
        if len(out) >= n and not utterance:
            break
    return out


def log_mels(y: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, **MEL_KW)
    return librosa.power_to_db(S, ref=np.max)


def mel_envelope_metrics(S_db_a: np.ndarray, S_db_b: np.ndarray) -> dict[str, float]:
    """Frame-wise mel log-power mismatch after length alignment (min time frames)."""
    t = min(S_db_a.shape[1], S_db_b.shape[1])
    if t <= 0:
        return {"mel_mae_db": float("nan"), "mel_rmse_db": float("nan")}
    A = S_db_a[:, :t]
    B = S_db_b[:, :t]
    diff = A - B
    return {
        "mel_mae_db": float(np.mean(np.abs(diff))),
        "mel_rmse_db": float(np.sqrt(np.mean(diff**2))),
    }


def smooth_mel_db(S_db: np.ndarray, win: int = 3) -> np.ndarray:
    if win <= 1:
        return S_db
    k = np.ones(win, dtype=np.float64) / win
    out = np.empty_like(S_db, dtype=np.float64)
    for i in range(S_db.shape[0]):
        out[i] = np.convolve(S_db[i].astype(np.float64), k, mode="same")
    return out


def f0_stats(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """pyin F0 (Hz), median over voiced frames, voicing rate."""
    f0, voiced_flag, _ = librosa.pyin(y.astype(np.float32), **PYIN_KW)
    voiced = f0 > 0
    med = float(np.nanmedian(f0[voiced])) if np.any(voiced) else float("nan")
    vrate = float(np.mean(voiced)) if voiced.size else 0.0
    return f0, med, vrate


def f0_rmse_voiced_both(f0_a: np.ndarray, f0_b: np.ndarray) -> float:
    """RMSE log-F0 where both frames voiced (caveat: contours often misalign after VC)."""
    t = min(len(f0_a), len(f0_b))
    a, b = f0_a[:t], f0_b[:t]
    m = (a > 0) & (b > 0)
    if not np.any(m):
        return float("nan")
    la, lb = np.log(a[m]), np.log(b[m])
    return float(np.sqrt(np.mean((la - lb) ** 2)))


def plot_mel_triple(
    y_orig: np.ndarray,
    y_conv: np.ndarray,
    out_png: Path,
    stem: str,
) -> None:
    S0 = log_mels(y_orig)
    S1 = log_mels(y_conv)
    t = min(S0.shape[1], S1.shape[1])
    D = np.abs(S0[:, :t] - S1[:, :t])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    _hop = int(MEL_KW["hop_length"])
    for ax, S, title in zip(
        axes[:2],
        (S0, S1),
        ("Original (Libri)", "SFW converted"),
    ):
        librosa.display.specshow(
            S,
            x_axis="time",
            y_axis="mel",
            sr=SFW_SAMPLE_RATE,
            hop_length=_hop,
            ax=ax,
        )
        ax.set_title(f"{stem} — {title}")
    im = librosa.display.specshow(
        D,
        x_axis="time",
        y_axis="mel",
        sr=SFW_SAMPLE_RATE,
        hop_length=_hop,
        ax=axes[2],
    )
    axes[2].set_title(f"{stem} — |Δ mel (dB)|")
    fig.colorbar(im, ax=axes[2], format="%+2.0f dB")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def world_logsp_distance(y_orig: np.ndarray, y_conv: np.ndarray, sr: int) -> float:
    """
    Optional: mean |log SP_orig - log SP_conv| over frames (cheaptrick envelopes).
    This is NOT mel-cepstral MCD (VERSA); for that install pyworld + mgc/DTW pipeline.
    """
    try:
        import pyworld as pw
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install pyworld for --with-pyworld") from e

    def sp_log(yw: np.ndarray) -> np.ndarray:
        x = np.asarray(yw, dtype=np.float64).ravel()
        x = x / (np.max(np.abs(x)) + 1e-9)
        f0, t = pw.harvest(x, sr)
        f0 = pw.stonemask(x, f0, t, sr)
        sp = pw.cheaptrick(x, f0, t, sr)
        return np.log(np.maximum(sp, 1e-12))

    L0 = sp_log(y_orig)
    L1 = sp_log(y_conv)
    T = min(L0.shape[0], L1.shape[0])
    if T <= 0:
        return float("nan")
    return float(np.mean(np.abs(L0[:T] - L1[:T])))


def main() -> int:
    ap = argparse.ArgumentParser(description="SFW vs Libri VC diagnostics (mel + F0 + optional pyworld).")
    ap.add_argument("--librispeech-root", type=str, required=True)
    ap.add_argument("--sfw-wav-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--utterance", type=str, default="")
    ap.add_argument("--normalize", action="store_true", help="Peak-normalize both clips (plots/metrics only)")
    ap.add_argument(
        "--smooth-mel",
        type=int,
        default=3,
        help="Moving-average window along time per mel bin for mel_mae (0=off)",
    )
    ap.add_argument(
        "--with-pyworld",
        action="store_true",
        help="WORLD cheaptrick log-SP mean abs diff (requires pyworld; not paper MCD)",
    )
    args = ap.parse_args()

    pairs = collect_pairs(
        Path(args.librispeech_root),
        Path(args.sfw_wav_dir),
        args.n,
        args.utterance.strip() or None,
    )
    if not pairs:
        print("No pairs found.", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir).resolve()
    rows: list[dict[str, float | str]] = []

    for stem, flac, wav in pairs:
        yo, yc, sr = load_before_after(flac, wav, target_sr=SFW_SAMPLE_RATE, normalize=args.normalize)
        S0 = log_mels(yo)
        S1 = log_mels(yc)
        if args.smooth_mel and args.smooth_mel > 1:
            S0 = smooth_mel_db(S0, args.smooth_mel)
            S1 = smooth_mel_db(S1, args.smooth_mel)
        m = mel_envelope_metrics(S0, S1)
        f0_o, med_o, vr_o = f0_stats(yo)
        f0_c, med_c, vr_c = f0_stats(yc)
        f0_rmse = f0_rmse_voiced_both(f0_o, f0_c)

        row: dict[str, float | str] = {
            "stem": stem,
            "mel_mae_db": m["mel_mae_db"],
            "mel_rmse_db": m["mel_rmse_db"],
            "f0_median_hz_orig": med_o,
            "f0_median_hz_sfw": med_c,
            "voicing_rate_orig": vr_o,
            "voicing_rate_sfw": vr_c,
            "f0_log_rmse_both_voiced": f0_rmse,
        }
        if args.with_pyworld:
            row["world_logsp_mae"] = world_logsp_distance(yo, yc, sr)

        rows.append(row)
        plot_mel_triple(yo, yc, out_dir / f"{stem}_mel_compare.png", stem)

    # Print table
    keys = [k for k in rows[0].keys() if k != "stem"]
    print(f"{'stem':<22} " + " ".join(f"{k:>18}" for k in keys))
    for r in rows:
        parts = []
        for k in keys:
            v = r[k]
            parts.append(f"{float(v):>18.4g}" if v == v else f"{'nan':>18}")  # NaN != NaN
        print(f"{str(r['stem']):<22} " + " ".join(parts))

    print(f"\nSaved mel PNGs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
