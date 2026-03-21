# ═══════════════════════════════════════════════════════════════════════
# 🎛️  INTERACTIVE AUDIO TRANSFORM PLAYGROUND (torchaudio)
# ═══════════════════════════════════════════════════════════════════════

# ──── KNOBS (tweak these!) ────
PITCH_SEMITONES = -6      # negative = lower pitch  (try: -2, -4, -6)
SPEED_FACTOR    = 1.08      # >1 = faster speech      (try: 1.0, 1.05, 1.10, 1.15)
NUM_CLIPS       = 5
MANIFEST        = "/kaggle/input/datasets/akarshks/val-meta/val_manifest.jsonl"

# ──── CODE ────
import json, os, torch, torchaudio
import soundfile as sf
import IPython.display as ipd
from IPython.display import display, HTML

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_audio(path):
    """Load audio with soundfile → torch tensor (bypasses torchcodec)."""
    data, sr = sf.read(path, dtype="float32")   # numpy array
    t = torch.from_numpy(data)
    if t.ndim == 1:
        t = t.unsqueeze(0)                      # [samples] → [1, samples]
    else:
        t = t.T                                 # [samples, channels] → [channels, samples]
        t = t.mean(dim=0, keepdim=True)          # stereo → mono
    return t, sr

def pitch_shift(waveform, sr, n_steps):
    if n_steps == 0:
        return waveform
    return torchaudio.functional.pitch_shift(
        waveform.to(device), sr, n_steps=n_steps
    ).cpu()

def speed_change(waveform, sr, factor):
    if abs(factor - 1.0) < 0.005:
        return waveform
    orig_freq = int(sr * factor)
    resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=sr)
    return resampler(waveform)

def transform(waveform, sr, pitch_st, speed):
    w = pitch_shift(waveform, sr, pitch_st)
    w = speed_change(w, sr, speed)
    peak = w.abs().max()
    if peak > 0:
        w = w / peak * 0.95
    return w

# ── Load manifest ──
rows = []
with open(MANIFEST) as f:
    for line in f:
        d = json.loads(line.strip())
        ap = d.get("audio_filepath") or d.get("audio_path", "")
        text = (d.get("text") or "").strip()
        if ap and os.path.isfile(ap) and text:
            rows.append((ap, text))
        if len(rows) >= NUM_CLIPS:
            break

display(HTML(f"""
<div style="background:#1a1a2e; color:#e0e0e0; padding:15px; border-radius:10px; margin-bottom:20px;">
  <h2 style="color:#00d4ff; margin:0;">🎛️ Audio Transform Playground</h2>
  <table style="color:#e0e0e0; margin-top:10px;">
    <tr><td style="padding:4px 12px;"><b>Pitch:</b></td><td>{PITCH_SEMITONES:+d} semitones</td></tr>
    <tr><td style="padding:4px 12px;"><b>Speed:</b></td><td>{SPEED_FACTOR}x</td></tr>
    <tr><td style="padding:4px 12px;"><b>Clips:</b></td><td>{len(rows)}</td></tr>
  </table>
</div>
"""))

for i, (orig_path, ref_text) in enumerate(rows):
    waveform, sr = load_audio(orig_path)
    waveform_t = transform(waveform, sr, PITCH_SEMITONES, SPEED_FACTOR)

    display(HTML(f'<div style="background:#16213e; padding:12px; border-radius:8px; margin:10px 0;">'
                 f'<h3 style="color:#e94560; margin:0;">Clip {i+1}: "{ref_text}"</h3></div>'))

    display(HTML("<b>🔵 Original (child):</b>"))
    display(ipd.Audio(waveform.squeeze().numpy(), rate=sr))

    display(HTML(f"<b>🟢 Transformed (pitch {PITCH_SEMITONES:+d}st, speed {SPEED_FACTOR}x):</b>"))
    display(ipd.Audio(waveform_t.squeeze().numpy(), rate=sr))

    display(HTML("<hr>"))

display(HTML("""
<div style="background:#0f3460; color:#e0e0e0; padding:12px; border-radius:8px;">
  <b>💡 Try these:</b><br>
  <code>PITCH = -2, SPEED = 1.0</code>  → mild pitch only<br>
  <code>PITCH = -4, SPEED = 1.08</code> → moderate both<br>
  <code>PITCH = -6, SPEED = 1.15</code> → aggressive<br>
  <code>PITCH = 0,  SPEED = 1.10</code> → speed only<br>
  <code>PITCH = -4, SPEED = 1.0</code>  → pitch only
</div>
"""))
