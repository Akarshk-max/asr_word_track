# asr_word_track

Parameter-efficient adaptation of **NVIDIA NeMo Parakeet-TDT** to children's speech, for the
[DrivenData "On Top of Pasketti: Children's Speech Recognition Challenge"](https://kidsasr.drivendata.org/) — **Word (orthographic) track**.

The challenge is a *code-execution* competition: you submit model weights plus a `main.py`, and it runs
offline in a fixed container (Python 3.11 / CUDA 12, no internet). The metric is **WER** after Whisper's
`EnglishTextNormalizer`.

This repository holds the training, augmentation, evaluation and inference code. **Model weights, audio
and manifests are not included** — see [Data & weights](#data--weights).

---

## What's here

### Backbone

`nvidia/parakeet-tdt-1.1b` — FastConformer-Transducer, **42 encoder layers x d_model 1024**, 8 heads,
depthwise-striding subsampling x8, RNN-T decoder (pred_hidden 640, 2 LSTM layers), joint 640,
**TDT loss with 5 durations**, 1024-token BPE. `nvidia/parakeet-tdt-0.6b-v2` (24 layers) and
`nvidia/parakeet-ctc-0.6b` were used in earlier runs.

The encoder backbone stays **frozen**; only adapters — plus, per experiment, the joint / decoder
embedding / last decoder LSTM — are trained.

### Adapter architectures

| Module | File | Idea |
|---|---|---|
| Bottleneck `LinearAdapter` | `0.2307_model.py` | 1024 -> r -> 1024, GELU, post-LayerNorm, dropout 0.1, on all 42 layers |
| **`ChainedLinearAdapter`** | `train_nemo_adapter_1.1b_worker.py`, `main.py` | Each layer's adapter bottleneck receives a learned `r -> r` projection of the **previous** layer's bottleneck, carried through a shared state object and **not detached** — so gradients propagate across the full encoder depth inside a low-dimensional subspace. `up` is zero-initialised (the adapter starts as an exact no-op); `chain_proj` is orthogonally initialised and scaled by 0.9. |
| Chained + **cross-layer LSTM memory** | `chained_memory_v3.py` | Adds a gated `AdapterMemoryCell` recurring across encoder *depth* (not time), carrying `(h, c)` between layers |
| **vNext**: two-regime adapters + LoRA | `train_vnext_kaggle_cell.py`, `train_nemo_adapter_parakeet_vnext.py` | Layers 0-7 two-bottleneck 1024 -> 256 -> 256, layers >= 8 single 1024 -> 256, continuous chain across the boundary; residual delta = `0.7*memory + 0.3*up-projection`; **LoRA r=16, alpha=32, dropout 0.05 on layers 0-20**, targeting attention/FFN linears **and pointwise `Conv1d`** (k=1, groups=1 — equivalent to a per-timestep linear map; depthwise convs deliberately untouched) |
| **Learnable layer fusion** | `nemo_layer_fusion.py` | ELMo-style softmax weighting over all 42 encoder layer outputs, captured with `register_forward_hook` so it survives `torch.inference_mode` during `transcribe()`. `eval_nemo_layer_fusion_stage1.py` prints the learned per-layer importance ranking. |
| **TPA** — two parallel FFN adapters | `kaggle_nemo_parakeet_tdt_1.1b_tpa_single_cell.py` | Two parallel residual adapters per layer, hooked onto the macaron FFN1/FFN2 blocks |
| **Age-adversarial head** | `train_parakeet_age_adversarial.py` | Domain-adversarial age invariance (see below) |

> **A NeMo gotcha worth knowing.** `AttentionAdapterModuleMixin` dispatches adapters behind
> `isinstance(module, LinearAdapter)`. A custom `nn.Module` fails that check and **every adapter forward
> is silently skipped** — no error, and the loss still decreases (from the unfrozen joint/decoder) while
> the adapters never learn. Fixed with `LinearAdapter.register(ChainedLinearAdapter)` (ABC virtual
> subclass), and guarded by an observable invariant: 42 hook fires per batch, and `layer_counter == 42`
> after each encoder pass. Written up in `parakeet_tdt_chained_adapter_architecture.tex`.

### Augmentation

- **`WaveformAugmentor`** — on-the-fly, applied inside a wrapped `model.forward` before the mel
  front-end: speed x0.85-1.15 (p=0.5), pitch -2..+2 semitones (p=0.5), additive noise at controlled
  SNR, random gain +/-8 dB (p=0.3). SNR mixing uses `alpha = sqrt(P_speech / (10^(SNR/10) * P_noise))`
  per utterance.
- **Noise curriculum** (`NoiseCurriculumCallback`) — mix probability rises and SNR tightens across
  epochs: `(0.3, 5-20 dB) -> (0.4, 5-15 dB) -> (0.45-0.5, 3-12 dB)`.
- **SpecAugment** — `freq_masks=2 (w=27)`, `time_masks=10 (w=0.05)`.
- **Cohort-aware probability** — real child audio gets *less* pitch perturbation (p=0.2) than
  adult/synthetic audio (p=0.5), since the child F0 range is the target distribution.

### Adult-to-child voice conversion (corpus augmentation)

- **`childrenize_librispeech.py`** — WORLD-vocoder childrenization, re-implemented from
  [zhao-shuyang/childrenize](https://github.com/zhao-shuyang/childrenize) with multiprocessing, resume,
  batch sharding and error handling added. Gender-conditional spectral warping (mean F0 > 160 Hz gives
  a 3-band piecewise warp, alpha in [1.10, 1.25]; otherwise linear, alpha in [1.2, 1.4]), F0 retargeted
  to **240-300 Hz**, vowel-length stretching x[1.1, 1.4], WORLD resynthesis at a 5 ms frame period.
- **`sfw_augment.py`** — Source-Filter Warping, from-scratch DSP: power-domain envelope by bidirectional
  first-order tracking (gamma=0.2, `min(forward, backward)`), source `S = Y/(V+eps)`, **independent**
  linear frequency warps on `S` (alpha) and `V` (beta), both in U[1.0, 1.3], recombine, then Griffin-Lim
  (8 iterations). STFT: 16 kHz, `n_fft=512`, `win=400` (25 ms), `hop=160` (10 ms), Hann.
  Tooling: `build_sfw_manifest.py` (multiprocess corpus build, resumable),
  `sfw_vc_diagnostics.py` (mel MAE/RMSE in dB, F0 statistics, optional WORLD log-SP distance),
  `sfw_before_after_preview.py` (A/B listening pairs).

### Domain-adversarial age invariance

`train_parakeet_age_adversarial.py`:

```
L = L_asr + lambda_adv * L_adv + s_disc * L_disc
```

A **confusion-loss** formulation of domain-adversarial training, implemented by *parameter-isolated
routing* rather than a gradient-reversal layer, so it stays a single Lightning backward pass:

- `L_disc = BCEWithLogits(disc(f.detach()), age_target)` — gradients reach only the discriminator
- `L_adv  = BCEWithLogits(disc_detached_weights(f), 0.5)` — gradients reach only the encoder, pushing
  features toward age-uninformative

`f` is masked **mean+std** pooling over time of the encoder output (2 x 1024). Soft age targets
(youngest child -> 0, oldest child in corpus -> 0.8, adult -> 1.0), a linear `lambda_adv` ramp (0 until
epoch 10, rising to the maximum through epoch 40), and optional balanced child/adult batches.

### Inference under the competition runtime

`main.py` and `submission_1/main.py`:

- Duration-sorted **adaptive batching** (batch size 4 -> 40 as a function of clip length)
- **Three-level OOM fallback**: batch -> per-utterance -> per-variant
- Bypasses `model.transcribe` for a manual `forward` + `rnnt_decoder_predictions_tensor`, avoiding
  Lhotse and temp-manifest round-trips; `_setup_transcribe_dataloader` is patched to disable Lhotse
- `use_cuda_graph_decoder=False` (incompatible with PyTorch 2.10)
- soundfile-first audio loading (`safe_load_audio_soundfile_first.py`) to avoid a torchaudio/TorchCodec
  path that hangs on FLAC, plus silence substitution so a single unreadable file cannot fail a
  205k-utterance run
- **4-variant test-time augmentation with medoid consensus** (`submission_1/main.py`): original,
  speed x1.05, speed x0.95, pitch -2 semitones; pick `argmin_i mean_{j != i} WER(h_i, h_j)`. Batch sizes
  are divided by the variant count so GPU cost is unchanged.

Decoding is **greedy RNN-T** throughout — no beam search, no LM fusion.

### Verification and validation

- `verify_nemo_adapter_pipeline.py` — a pre-flight harness run *before* committing GPU hours:
  architecture survey, adapter-chain integrity, module train-flag tree, forward/backward with
  per-parameter gradient statistics, optimizer/scheduler check with a rendered LR-schedule plot,
  SpecAugment and waveform-augmentation plots, dataloader check, memory estimate, per-step benchmark,
  `save_adapters` round-trip, NeMo version and residual-strategy checks. Emits `VERIFY_REPORT.md`.
- `validate_nemo_adapter_1_1b_kaggle.py`, `validate_nemo_parakeet_vnext_kaggle.py` and
  `inference_chained_adapter_lstm_v3.py` — WER/CER plus a **substitution / deletion / insertion**
  breakdown and the 10 worst utterances by per-utterance WER.
- `chained_memory_v3.py` includes a dummy-tensor unit test and a runtime probe that *empirically
  determines* whether NeMo expects a full residual or a delta, rather than assuming the API contract.

### Data preparation

`create_combined_dataset.py`, `prepare_librispeech_manifest.py`, `build_librispeech_age_manifest.py`,
`merge_child_adult_training_manifests.py`, `enrich_submission_train_manifest_age.py`.

Training labels are normalised with Whisper's `EnglishTextNormalizer`, and the LibriSpeech (uppercase)
manifests go through the **same** normaliser — `build_librispeech_age_manifest.py` exposes
`--require-whisper-normalizer` so cohorts cannot be silently mismatched when mixed in one batch.

---

## Typical training configuration

| | |
|---|---|
| Precision | `bf16-mixed`, single GPU |
| Optimizer | AdamW, betas (0.9, 0.999), weight decay 0.01 |
| Schedule | linear warmup (ratio 0.15) then cosine, `min_lr` 1e-6 |
| Discriminative LRs | adapters 5e-4, LoRA/memory 3e-4, joint 1e-4, decoder 5e-5, **encoder 5e-6** (stage 2) |
| Gradient clipping | 1.0 (0.5 in stage 2) |
| Batch size / epochs | 32-64 / 3-7 (+1 stage-2) |
| Max clip duration | 20 s |

Most Kaggle-facing scripts follow the same one-cell pattern: purge conflicting packages, pin
numpy/scipy, install torch from the cu126 index, install dependencies, install NeMo, **re-pin numpy**,
write a self-contained training script to disk, `py_compile` it, then run it in a **fresh subprocess**
so no stale import survives. `kaggle_wheel_downloader.py` builds an offline wheelhouse matching the
competition runtime spec (Python 3.11 / CUDA 12.6 / manylinux2014).

---

## Results

| | WER |
|---|---:|
| Organizers' reference implementation (parakeet-tdt-0.6b-v2 + 32-d adapter, 5,000 steps) | 0.1546 |
| This work, on a held-out 68,758-utterance split | **0.1269** |

Both use the competition metric (Whisper-normalized WER). **Caveats, stated plainly:** the two splits
are not identical (different seeds and duration filters), and the split used here is **utterance-level,
not speaker-level** — the same child can appear in both train and validation, so this figure is
optimistic relative to the held-out competition test set. A `GroupShuffleSplit` on `child_id` is the
correct fix.

Error profile at 0.1269 (351,582 reference words): **substitutions 6.97%**, deletions 2.94%,
insertions 2.78%; 69.96% of utterances transcribed exactly. Substitutions dominate at roughly 2.4x
deletions — the acoustic-confusability signature expected from child pronunciation variation, rather
than dropped or hallucinated speech.

Leaderboard placement is not recorded in this repository.

---

## Data & weights

Deliberately not included:

- **Competition audio and transcripts** — redistribution is not mine to grant. Get them from the
  [Word track data page](https://www.drivendata.org/competitions/308/childrens-word-asr/data/).
- **TalkBank** child-speech corpus — see [talkbank.org](https://talkbank.org/).
- **LibriSpeech** — see [openslr.org/12](https://www.openslr.org/12/).
- **Model checkpoints** — the `.nemo` files are ~4.3 GB each, well past GitHub's limits.

Upstream repositories this code targets:
[runtime](https://github.com/drivendataorg/childrens-speech-recognition-runtime) and
[reference implementation](https://github.com/drivendataorg/childrens-speech-recognition-benchmark-pub).

## Branches

`nemo_finetune_no_aug` holds an earlier iteration of this work (Mar-Apr 2026) and is kept as-is.
