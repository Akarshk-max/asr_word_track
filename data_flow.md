Of course. Here’s a mathematical and dimensional breakdown of what happens to a 4-second audio sample during training.

---

# Data Flow of a 4-Second Audio Sample

## Assumptions
- **Sample rate:** `16,000 Hz`
- **Audio duration:** `4.0 seconds`
- **Encoder:** Parakeet FastConformer (8x downsampling)
- **Adapter dim:** `64`
- **Hidden dim:** `d_model = 1024`

---

## 1. Input Audio

A 4-second audio clip at 16kHz is a 1D tensor:

```text
x_audio ∈ ℝ^(1, 64000)
```
where `64000 = 4s * 16000 Hz`.

---

## 2. Preprocessor: Mel Spectrogram

The preprocessor converts the waveform into a 2D feature map.

- **Window size:** `25 ms` (400 samples)
- **Hop size:** `10 ms` (160 samples)
- **Mel bins:** `80`

Number of frames (`T_frames`):
```text
T_frames = ⌊(64000 - 400) / 160⌋ + 1 = 398 frames
```

Output is a log-mel spectrogram `X_spec`:
```text
X_spec ∈ ℝ^(1, 80, 398)
```
This is a `(batch, mel_bins, time_frames)` tensor.

---

## 3. SpecAugment (During Training)

Masks are applied to `X_spec`:

- **Time masking:** Zeros out `N` random time steps
- **Frequency masking:** Zeros out `M` random frequency bins

`X_aug` remains the same shape:
```text
X_aug ∈ ℝ^(1, 80, 398)
```

---

## 4. Encoder Input & Subsampling

The augmented spectrogram enters the FastConformer encoder. The first layers are convolutional subsampling with a factor of **8x**.

Input to encoder:
```text
X_in = X_aug ∈ ℝ^(1, 80, 398)
```

After subsampling, the time dimension is reduced by 8x and features are projected to `d_model = 1024`:
```text
T_sub = T_frames / 8 = 398 / 8 ≈ 49
```

Subsampled output `Z_sub`:
```text
Z_sub ∈ ℝ^(1, 49, 1024)
```

---

## 5. Inside a Conformer Block (with Adapter)

The tensor `Z_sub` now goes through **42 layers** of Conformer blocks. Let's trace one block `L_i`.

Let the input to a block be `h_i ∈ ℝ^(1, 49, 1024)`.

The Conformer block has 4 main parts: Feed-forward, Attention, Convolution, and another Feed-forward. This is where the adapter is injected.

### Standard Conformer Path (Frozen)
The main path is:
```text
h_ffn1 = FFN1(LayerNorm(h_i)) + h_i
h_attn = MHA(LayerNorm(h_ffn1)) + h_ffn1
h_conv = ConvModule(LayerNorm(h_attn)) + h_attn
h_ffn2 = FFN2(LayerNorm(h_conv)) + h_conv
h_out  = LayerNorm(h_ffn2)
```

### Adapter Path (Trainable)
The adapter is a small feed-forward network inserted after the main block transformation.

Let `h' = h_out` be the output of the frozen Conformer block. The adapter performs:
1. **Down-projection:** `W_down ∈ ℝ^(1024, 64)`
2. **Activation:** (e.g., SiLU/Swish)
3. **Dropout:**
4. **Up-projection:** `W_up ∈ ℝ^(64, 1024)`

The mathematical operation of the adapter `A(h')` is:
```text
A(h') = Dropout(SiLU(h' * W_down)) * W_up
```

The final output of the block `h_(i+1)` is a residual connection:
```text
h_(i+1) = h' + A(h')
```

**Dimensionality inside the adapter:**
```text
h' ∈ ℝ^(1, 49, 1024)
↓
h' * W_down ∈ ℝ^(1, 49, 64)
↓
SiLU(h' * W_down) ∈ ℝ^(1, 49, 64)
↓
Dropout(...) * W_up ∈ ℝ^(1, 49, 1024)
↓
A(h') ∈ ℝ^(1, 49, 1024)
```

**Final block output `h_(i+1)` is still `ℝ^(1, 49, 1024)`**, ready for the next layer.

---

## 6. Encoder Output

After 42 Conformer layers, the final encoder output `Z_final` is:
```text
Z_final ∈ ℝ^(1, 49, 1024)
```
This is the **acoustic representation**.

---

## 7. Decoder & Joint Network (TDT)

Parakeet-TDT uses an RNNT-style decoder.

- **Predictor Network (Text-based):** An autoregressive RNN (e.g., 2-layer LSTM) takes the previously predicted token `y_(j-1)` and produces a text representation `h_pred ∈ ℝ^(640)`.

- **Joint Network:** A feed-forward network combines the acoustic and text representations:
  - Takes `Z_final` (acoustic) and `h_pred` (text)
  - `joint_input = Concat(Z_final, h_pred)`
  - `h_joint = FFN(joint_input)`

- **Output Projection:** The joint network output is projected to the vocabulary size (1024) + blank/duration tokens.

---

## 8. Loss Calculation (TDT Loss)

The model outputs probabilities for:
- Vocabulary tokens (0-1023)
- Duration tokens (e.g., skip 0, 1, 2, 3, 4 frames)

The TDT loss function finds the optimal path through this grid of acoustic frames and token/duration predictions to match the ground truth text.

---

## 9. Backpropagation & Gradient Update

Gradients are calculated with respect to the TDT loss.

```text
d(Loss) / d(W_adapter) ≠ 0   ← Adapter weights are updated

d(Loss) / d(W_conformer) = 0   ← Base Conformer weights are frozen
d(Loss) / d(W_decoder) = 0     ← Decoder weights are frozen
```

Only the weights of the adapters (`W_down`, `W_up` and LayerNorm in each of the 42 blocks) are updated by the optimizer.

---

## **Summary of the Journey**

```
x_audio ∈ ℝ^(64000)
     │
     ▼ (Preprocessor)
X_spec ∈ ℝ^(1, 80, 398)
     │
     ▼ (SpecAugment)
X_aug ∈ ℝ^(1, 80, 398)
     │
     ▼ (Subsampling)
Z_sub ∈ ℝ^(1, 49, 1024)
     │
     ▼ (Conformer Block i)
h_i ∈ ℝ^(1, 49, 1024)
     │
     ├─ [Frozen Path: MHA, Conv, FFN]
     │
     └─ [Trainable Path: Adapter(h')]
     │       └─ ℝ^(1024) → ℝ^(64) → ℝ^(1024)
     │
     ▼ (Residual Sum)
h_(i+1) ∈ ℝ^(1, 49, 1024)
     │
   (Repeat 42 times)
     │
     ▼ (Final Encoder Output)
Z_final ∈ ℝ^(1, 49, 1024)
     │
     ▼ (TDT Decoder + Joint Network)
Probabilities(Tokens, Durations)
     │
     ▼ (TDT Loss)
Scalar Loss
     │
     ▼ (Backpropagation)
Gradient updates ONLY for Adapter weights
```

# A different prespective 
Okay, let's trace the data flow for a **single 4-second audio test sample** through your Parakeet-TDT-1.1B adapter architecture, with a focus on the mathematical transformations and dimensions at each stage.

We'll assume:
- **Audio sample rate:** 16 kHz
- **Conformer Encoder `d_model`:** 1024
- **Conformer Encoder layers:** 42 (for the 1.1B model)
- **Conformer subsampling factor:** 8
- **Mel spectrogram features (`n_mels`):** 80
- **Adapter `dim` (bottleneck):** 64
- **Vocabulary size:** 1024 (BPE tokens) + 1 (blank token) = 1025

---

## Data Flow for a Single 4-Second Audio Sample

### 1. Raw Audio Input

*   **Description:** The initial unprocessed audio signal.
*   **Mathematical Concept:** A discrete time-series sequence of amplitude values.
*   **Calculation:** `N_samples = duration (s) * sample_rate (Hz)`
    *   `N_samples = 4 * 16000 = 64000`
*   **Shape:** `(64000,)`
*   **State:** Raw, float array.

---

### 2. Audio Preprocessing (Mel Spectrogram)

*   **Description:** The raw audio is transformed into a sequence of log-mel spectrogram features, which are more perceptually relevant for speech.
*   **Parameters:**
    *   `window_size = 0.025 s` (400 samples)
    *   `window_stride = 0.010 s` (160 samples)
    *   `n_mels = 80`
*   **Mathematical Concept:** Short-Time Fourier Transform (STFT) followed by mel-filter banks and log scaling. The number of frames is determined by the stride.
*   **Calculation (Approximate frames `T_mel`):** `T_mel = (N_samples - window_size_samples) / window_stride_samples + 1`
    *   `T_mel = (64000 - 400) / 160 + 1 = 397.5 + 1 ≈ 399` frames. (For simplicity, we often round to `duration / window_stride = 4 / 0.01 = 400` frames.)
*   **Shape:** `(Batch_size, T_mel, n_mels)`
    *   Since it's a single sample, `Batch_size = 1`.
    *   `(1, 399, 80)` or `(1, 400, 80)`
*   **State:** Float tensor, representing acoustic features.
*   **Note:** For *test* samples, no spectrogram augmentation (like time/frequency masking) is applied here. That's only for training.

---

### 3. Conformer Encoder (ConformerEncoderAdapter)

This is the core of the model, processing the spectrogram. The `ConformerEncoderAdapter` includes an adapter in each of its 42 layers.

*   **Input:** Log-mel spectrogram `F_mel`
    *   **Shape:** `(1, 399, 80)`

#### a. Convolutional Subsampling Layers (Input to Encoder)

*   **Description:** Typically a few `Conv2d` layers with stride 2 and `LayerNorm` to downsample the temporal dimension and project to `d_model`. Parakeet's encoder has `subsampling_factor: 8`.
*   **Mathematical Concept:** Convolutional layers progressively reduce the temporal resolution.
*   **Calculation:** `T_subsampled = T_mel / subsampling_factor`
    *   `T_subsampled = 399 / 8 ≈ 49` frames (this will be floor or ceil depending on padding, let's say 50 for evenness).
*   **Shape:** `(1, T_subsampled, d_model)`
    *   `(1, 50, 1024)`
*   **State:** Float tensor.

#### b. Conformer Blocks (42 Layers)

Each of the 42 Conformer layers takes `(1, 50, 1024)` and outputs `(1, 50, 1024)`.

*   **Inside each Conformer block `k` (from `k=0` to `41`):**
    *   **Input `x_k_in`:** `(1, 50, 1024)`
    *   **LayerNorm:** `(1, 50, 1024)`
    *   **Feed-Forward Module (FFN) 1 (Half-step):**
        *   `Linear(1024, 4096) → SiLU → Dropout → Linear(4096, 1024)`
        *   **Intermediate Shape:** `(1, 50, 4096)`
        *   **Output Shape:** `(1, 50, 1024)`
    *   **Multi-Head Self-Attention (MHSA):**
        *   `LayerNorm` applied to FFN1 output.
        *   Attention mechanism with `d_model=1024`, `n_heads=8`. Computes `Q, K, V` matrices.
        *   `Q,K,V = Linear(1024, 1024)` (for each head, then concatenated)
        *   Relative positional encodings (`rel_pos`) are added.
        *   **Output Shape:** `(1, 50, 1024)`
    *   **Convolution Module:**
        *   `LayerNorm` applied to MHSA output.
        *   `Pointwise Conv (1024, 2048) → GLU → Depthwise Conv (kernel=9) → BatchNorm → SiLU → Pointwise Conv (2048, 1024)`
        *   **Output Shape:** `(1, 50, 1024)`
    *   **Feed-Forward Module (FFN) 2 (Half-step):** Same as FFN1.
        *   **Output Shape `x_k_out_conformer`:** `(1, 50, 1024)`
    *   **Adapter Layer (LinearAdapter):**
        *   **Description:** This small trainable module sits `post` the Conformer block.
        *   **Mathematical Concept:** `x_adapted = x_k_out_conformer + Linear(d_model, dim_adapter) → Activation → Linear(dim_adapter, d_model)` (Residual connection)
        *   **Calculation:**
            *   `Linear(1024, 64)` projects input to bottleneck dimension.
            *   `Linear(64, 1024)` projects back.
        *   **Input Shape to Adapter:** `(1, 50, 1024)`
        *   **Intermediate Shape within Adapter:** `(1, 50, 64)`
        *   **Output Shape of Adapter:** `(1, 50, 1024)`
        *   **State:** Adapter parameters are **TRAINED**, base Conformer parameters are **FROZEN**.
    *   **LayerNorm:** Final `LayerNorm` for the block.
    *   **Output `x_k_out`:** `(1, 50, 1024)`

*   **Output of Encoder (Z):** The final `(1, 50, 1024)` tensor after 42 layers.

---

### 4. TDT Decoder (RNNTDecoder + RNNTJoint)

The TDT decoder takes the encoder output and iteratively predicts tokens.

*   **Input:**
    *   Encoder output `Z`: `(1, T_enc, d_model)` = `(1, 50, 1024)` (acoustic context)
    *   Previously predicted tokens `y_prev`: Sequence of vocabulary indices.
*   **Process (iterative, for each time step `t` in `T_enc` and each potential token `u`):**

    #### a. Prediction Network (Predictor)
    *   **Input `y_prev`:** An embedding of the last predicted token.
    *   **Mathematical Concept:** An RNN (often LSTM) or Transformer that maintains a state based on the sequence of previously predicted tokens.
    *   **Shape:** `(1, d_pred_hidden)` (e.g., `(1, 640)`)
    *   **State:** Internal state of the prediction network.

    #### b. Joint Network
    *   **Input:**
        *   Current encoder frame `z_t`: `(1, d_model)` = `(1, 1024)`
        *   Prediction network output `h_pred`: `(1, d_pred_hidden)` = `(1, 640)`
    *   **Mathematical Concept:** Combines the acoustic (`z_t`) and linguistic (`h_pred`) contexts through a feed-forward network to produce token logits.
    *   **Calculation:**
        *   Concatenation: `[z_t, h_pred]` -> `(1, d_model + d_pred_hidden)` = `(1, 1024 + 640)` = `(1, 1664)`
        *   `Linear(1664, d_joint_hidden) → Activation → Linear(d_joint_hidden, (vocab_size + 1))`
        *   `vocab_size + 1` includes the blank token for TDT/RNNT alignment.
*   **Output (logits):** `(1, vocab_size + 1)` = `(1, 1025)`
    *   This represents the probability distribution over all possible output tokens (including blank) for the current step.

*   **Decoding Strategy:** During inference, typically a greedy search (or beam search) is applied to these logits over time steps (`T_enc`) to find the most probable sequence of tokens. TDT specifically also predicts a duration for each token, allowing for more efficient decoding.

---

### 5. Final Output

*   **Description:** The decoded sequence of subword tokens, which are then concatenated to form the final text transcription.
*   **Mathematical Concept:** A sequence of integers representing BPE tokens.
*   **Shape:** `(N_predicted_tokens,)`
*   **Example:** `['▁hell', 'o', '▁wor', 'ld']`
*   **Final step:** The `processor.batch_decode()` method converts these token IDs back into readable text.

---

## Summary of Dimensions for a 4-Second Sample

| Stage                                  | Input Shape     | Output Shape          | Operations (Key)                            | Trained/Frozen |
| :------------------------------------- | :-------------- | :-------------------- | :------------------------------------------ | :------------- |
| **1. Raw Audio**                       | N/A             | `(64000,)`            | Sampled waveform                            | -              |
| **2. Mel Spectrogram**                 | `(64000,)`      | `(1, 399, 80)`        | STFT, Mel-filter banks, Log-scaling         | -              |
| **3. Conformer Encoder (Subsampling)** | `(1, 399, 80)`  | `(1, 50, 1024)`       | Conv2d layers (8x temporal subsampling)     | Frozen         |
| **4. Conformer Blocks (42x)**          | `(1, 50, 1024)` | `(1, 50, 1024)`       | FFN, MHSA, Conv, FFN                        | Frozen         |
| **5. Adapter Layers (42x)**            | `(1, 50, 1024)` | `(1, 50, 1024)`       | `Linear(1024,64) → Act → Linear(64,1024)`   | **Trained**    |
| **6. Prediction Network**              | `(1, 1)` (token)| `(1, 640)`            | LSTM/Transformer (iterative)                | Frozen         |
| **7. Joint Network**                   | `(1, 50, 1024)` + `(1, 640)` | `(1, 1025)` (logits)  | Concat, Linear, Activation, Linear (iterative) | Frozen         |
| **8. Decoding**                        | `(1, 1025)`     | `(N_predicted_tokens,)` | Greedy/Beam Search, Tokenization            | -              |

This detailed breakdown shows how the 4-second audio transforms, where the adapters contribute their learned modifications, and the dimensions at each critical step.
