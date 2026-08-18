# NeMo Parakeet TDT: linear adapters + learnable layer fusion

This adds a **softmax-weighted sum of all Conformer encoder layer outputs** (via NeMo’s InterCTC / `AccessMixin` capture) before the RNNT decoder, on top of your existing **encoder linear adapters** recipe.

## Files

| File | Role |
|------|------|
| [nemo_layer_fusion.py](nemo_layer_fusion.py) | `LearnableLayerFusion` + `attach_layer_fusion_to_model()` |
| [train_nemo_layer_fusion_stage1.py](train_nemo_layer_fusion_stage1.py) | Frozen encoder + adapters + fusion |
| [train_nemo_layer_fusion_stage2.py](train_nemo_layer_fusion_stage2.py) | Restore Stage 1 `.nemo`, unfreeze last-K layers + adapters + fusion |
| [eval_nemo_layer_fusion_stage1.py](eval_nemo_layer_fusion_stage1.py) | Restore Stage 1 `.nemo`, print **layer-fusion** softmax weights, **WER** on `val_manifest.jsonl` |

## Kaggle: one cell (recommended)

Use **[kaggle_one_cell_nemo_layer_fusion.py](kaggle_one_cell_nemo_layer_fusion.py)**: paste the **entire file** into a single notebook cell. It installs packages, writes **`/kaggle/working/train_nemo_layer_fusion_stage1_standalone.py`** (fusion **inlined** — no second file, no import conflicts), and runs it with **`subprocess.run`** in a fresh interpreter.

**Checkpoints:** By default, one mid-run full **`model_stage1_step6000.nemo`** is written under `.../checkpoints/` (optimizer `global_step`). Override with env **`NEMO_SAVE_AT_STEPS`** (comma-separated), e.g. `2000,6000`, or set empty to disable mid-train `.nemo` saves. **`exp_manager.create_checkpoint_callback`** is **off** so you are not duplicating huge artifacts as both `.ckpt` and `.nemo`. Training end still saves **`model_stage1_fusion.nemo`** plus `adapter_final.pt` and `layer_fusion_weights.pt`.

**Decoder + joint:** With fused encoder activations, pretrained decoder/joint logits are mis-matched; by default **`TRAIN_DECODER_JOINT=1`** enables training **`model.decoder`** and **`model.joint`** while the encoder backbone stays frozen (adapters + fusion still train). Set **`TRAIN_DECODER_JOINT=0`** for the old adapter-only behavior. **`setup_optimization`** runs *after* all `requires_grad` flags so those parameters are included in AdamW.

## Kaggle: two files (alternative)

1. Copy **`nemo_layer_fusion.py`** and **`train_nemo_layer_fusion_stage1.py`** into `/kaggle/working/`.
2. Run the training script in a subprocess:

```python
import subprocess, sys, os
subprocess.run(
    [sys.executable, "/kaggle/working/train_nemo_layer_fusion_stage1.py"],
    cwd="/kaggle/working",
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
)
```

3. After Stage 1, either set `STAGE1_NEMO` to the full path of `model_stage1_fusion.nemo`, or keep the default glob under `nemo_layer_fusion_stage1/`. Run Stage 2 the same way with `train_nemo_layer_fusion_stage2.py`.

**Evaluate Stage 1 (local):** `eval_nemo_layer_fusion_stage1.py` — set **`VAL_MANIFEST`**, optional **`STAGE1_NEMO`**, **`BATCH_SIZE`**, **`MAX_SAMPLES`**.

**Evaluate Stage 1 (Kaggle):**

| File | Use when |
|------|----------|
| [kaggle_one_cell_nemo_layer_fusion_infer.py](kaggle_one_cell_nemo_layer_fusion_infer.py) | One cell: same pip pins as training (or **`SKIP_PIP=1`** if the training install cell already ran), writes standalone infer script, **`subprocess`** — no notebook helper imports. |
| [kaggle_notebook_infer_cell_using_helpers.py](kaggle_notebook_infer_cell_using_helpers.py) | Second cell after your pasted **fusion helper** cell; uses global **`attach_layer_fusion_to_model`**. |

Both print a **layer-fusion weight report** and **WER** on a NeMo manifest (`audio_filepath`, `text`).

**Finding the Stage 1 `.nemo`:** `exp_manager` saves under nested folders, e.g. `nemo_layer_fusion_stage1/ParakeetAdapterLayerFusion/<timestamp>/checkpoints/model_stage1_fusion.nemo`. The infer script auto-searches for that file and, if missing, the newest **`model_stage1_step*.nemo`**. Set **`STAGE1_NEMO`** explicitly if needed. Re-paste the infer driver from the repo if a hand-edited cell broke the embedded script (`from __future__`, `__init__`, `if __name__`).

**NeMo 2.x inference:** `transcribe()` expects **`audio=`** (list of paths), not `paths2audio_files=`. Fusion weights are loaded into **`model.layer_fusion`** only (not `model.load_state_dict`), so you do not get a giant spurious `missing_keys` list for the full encoder.

## Path tweaks

In both training scripts, adjust dataset roots if your Kaggle dataset slug differs (`ASR_DATA_DIR`, `TALKBANK_DIR`, `TALKBANK_JSON`).

## Requirements

- NeMo ASR encoder that registers `interctc/layer_output_{i}` (standard **ConformerEncoder** / **ConformerEncoderAdapter**). If you see `Missing interctc/layer_output_0`, your checkpoint may use a different encoder class; open an issue with `type(model.encoder)`.

## Notes

- **`forward` patching** is not stored inside `.nemo` metadata; **Stage 2 / inference always call `attach_layer_fusion_to_model` again** so fusion is active. The base NeMo class graph does not declare `layer_fusion`, so **`restore_from(..., strict=False)`** is used and **`layer_fusion.*` tensors are loaded from the `.nemo` tar** after attaching the module. Without this, you get `Unexpected key(s) in state_dict: "layer_fusion.raw_weights"`.
- Stage 2 uses a **single AdamW LR** for all trainable parameters; for importance-scaled per-layer LRs, extend `setup_optimization` or build parameter groups in a small subclass.
