"""
Source–Filter Warping (SFW) for adult→child-like spectrogram augmentation.

Implements the locked spec: power-domain envelope (gamma=0.2, min of forward/backward),
independent linear frequency warps on S and V, recombine, sqrt, Griffin–Lim.
STFT: 16 kHz, n_fft=512, win_length=400 (25 ms), hop_length=160 (10 ms), Hann.
"""

from __future__ import annotations

import numpy as np

try:
    import librosa
except ImportError as e:  # pragma: no cover
    raise ImportError("sfw_augment requires librosa") from e

SFW_SAMPLE_RATE = 16_000
SFW_N_FFT = 512
SFW_WIN_LENGTH = 400
SFW_HOP_LENGTH = 160
SFW_WINDOW = "hann"
SFW_GAMMA = 0.2
SFW_GRIFFIN_ITER = 8
SFW_ALPHA_MIN = 1.0
SFW_ALPHA_MAX = 1.3
SFW_BETA_MIN = 1.0
SFW_BETA_MAX = 1.3

SFW_STFT_KW = dict(
    n_fft=SFW_N_FFT,
    hop_length=SFW_HOP_LENGTH,
    win_length=SFW_WIN_LENGTH,
    window=SFW_WINDOW,
    center=True,
)


def envelope_forward(Y: np.ndarray, gamma: float = SFW_GAMMA) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float64)
    V = np.empty_like(Y)
    V[0] = Y[0]
    for i in range(1, len(Y)):
        V[i] = max(Y[i], V[i - 1] + gamma * (Y[i] - V[i - 1]))
    return V


def envelope_backward(Y: np.ndarray, gamma: float = SFW_GAMMA) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float64)
    n = len(Y)
    V = np.empty_like(Y)
    V[n - 1] = Y[n - 1]
    for i in range(n - 2, -1, -1):
        V[i] = max(Y[i], V[i + 1] + gamma * (Y[i] - V[i + 1]))
    return V


def envelope_bidirectional(Y: np.ndarray, gamma: float = SFW_GAMMA) -> np.ndarray:
    vf = envelope_forward(Y, gamma)
    vb = envelope_backward(Y, gamma)
    return np.minimum(vf, vb)


def warp_1d(F: np.ndarray, lam: float) -> np.ndarray:
    """Linear frequency warp: output bin i reads F at continuous index i/lam."""
    F = np.asarray(F, dtype=np.float64)
    n = F.shape[0]
    top_k = max(1, int(np.ceil(0.02 * n)))
    fill_value = float(np.mean(np.partition(F, -top_k)[-top_k:]))

    i = np.arange(n, dtype=np.float64)
    u = i / lam
    j = np.floor(u).astype(np.int64)
    d = u - j
    valid = j < n - 1
    out = np.full(n, fill_value, dtype=np.float64)
    jv = j[valid]
    dv = d[valid]
    out[valid] = (1.0 - dv) * F[jv] + dv * F[jv + 1]
    return out


def apply_sfw_to_power_spectrogram(
    Y: np.ndarray,
    alpha: float,
    beta: float,
    *,
    gamma: float = SFW_GAMMA,
) -> np.ndarray:
    """
    Y: power spectrogram (n_freq, n_frames).
    Returns Y' of same shape, non-negative.
    """
    Y = np.asarray(Y, dtype=np.float64)
    n_bins, n_frames = Y.shape
    Yp = np.empty_like(Y)
    for t in range(n_frames):
        yt = Y[:, t]
        vt = envelope_bidirectional(yt, gamma)
        eps = max(1e-10, 1e-8 * float(np.max(yt)))
        st = yt / (vt + eps)
        sw = warp_1d(st, alpha)
        vw = warp_1d(vt, beta)
        ypt = sw * vw
        Yp[:, t] = np.maximum(ypt, 0.0)
    return Yp


def apply_sfw_to_waveform(
    y: np.ndarray,
    sr: int,
    *,
    alpha: float | None = None,
    beta: float | None = None,
    rng: np.random.Generator | None = None,
    griffin_iter: int = SFW_GRIFFIN_ITER,
) -> tuple[np.ndarray, float, float]:
    """
    Load-style mono float waveform -> SFW waveform at SFW_SAMPLE_RATE.

    Returns (y_out_1d_float32, alpha, beta).
    """
    rng = rng if rng is not None else np.random.default_rng()
    if alpha is None:
        alpha = float(rng.uniform(SFW_ALPHA_MIN, SFW_ALPHA_MAX))
    if beta is None:
        beta = float(rng.uniform(SFW_BETA_MIN, SFW_BETA_MAX))

    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        # librosa.to_mono expects (n_ch, n_samples)
        y = librosa.to_mono(y.T if y.shape[0] > y.shape[1] else y)
    y = np.ascontiguousarray(np.squeeze(y), dtype=np.float32)

    if sr != SFW_SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SFW_SAMPLE_RATE).astype(np.float32)
        sr = SFW_SAMPLE_RATE

    if y.size == 0:
        return y.astype(np.float32), alpha, beta

    if y.size < SFW_WIN_LENGTH:
        y = np.pad(y, (0, SFW_WIN_LENGTH - y.size), mode="constant")

    S = librosa.stft(y, **SFW_STFT_KW)
    Y = (np.abs(S) ** 2).astype(np.float64)
    Yp = apply_sfw_to_power_spectrogram(Y, alpha, beta)
    mag = np.sqrt(np.maximum(Yp, 0.0))

    y_rec = librosa.griffinlim(
        mag,
        n_iter=int(griffin_iter),
        hop_length=SFW_HOP_LENGTH,
        win_length=SFW_WIN_LENGTH,
        n_fft=SFW_N_FFT,
        window=SFW_WINDOW,
    )
    return y_rec.astype(np.float32), alpha, beta


__all__ = [
    "SFW_SAMPLE_RATE",
    "SFW_STFT_KW",
    "SFW_GRIFFIN_ITER",
    "envelope_forward",
    "envelope_backward",
    "envelope_bidirectional",
    "warp_1d",
    "apply_sfw_to_power_spectrogram",
    "apply_sfw_to_waveform",
]
