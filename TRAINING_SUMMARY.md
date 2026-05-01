# Training Pipeline Summary

## ✓ Data Distribution

### NuImages 5% Stratified Subset
Location: `/home/tamoghno/datasets/nuimages/v1.0-train-5pct/`

| Split | Samples | Purpose | Images Location |
|-------|---------|---------|-----------------|
| **Train** | 2,344 samples | Primary training data | `/datasets/nuimages/samples/` |
| **Validation** | 491 samples | Per-epoch validation, early stopping | `/datasets/nuimages/samples/` |
| **Test (5% split)** | 488 samples | Final model evaluation | `/datasets/nuimages/samples/` |
| **Full Eval (bonus)** | ~14,000 samples | Generalization testing on full val set | `/home/tamoghno/datasets/nuimages/v1.0-val/` |

**Total Training Data: 2,344 samples (5% of original nuImages training set)**
**Validation: 491 samples (20% of 5% subset)**
**Test: 488 samples (20% of 5% subset)**

### Semantic Classes (11 total, 6 relevant for mIoU)
- **Relevant (counted in mIoU):** car, truck, bus, bicycle, motorcycle, pedestrian
- **Surface/Ignored (weighted but not in mIoU):** driveable_surface, other_flat, terrain, manmade, vegetation

---

## ✓ Model Architectures

All models use **UPerNet decoder** with task-specific encoders:

| Model | Encoder | Backbone | Params | Batch Size | Status |
|-------|---------|----------|--------|------------|--------|
| **SegFormer UPerNet** | MiT-B0 (custom) | Hierarchical transformer | ~4M + 3M (decoder) | 4 | Training |
| **ResNet101 UPerNet** | ResNet101 | Residual CNN | ~44M + 3M (decoder) | 6 | Pending |
| **Swin-B UPerNet** | Swin-B | Shifted window transformer | ~88M + 3M (decoder) | 6 | Pending |
| **ConvNeXt-B UPerNet** | ConvNeXt-B | Modern CNN | ~89M + 3M (decoder) | 4 (OOM fix applied) | Pending |

---

## ✓ Training Techniques

### 1. **Loss Function**
- **Type:** Cross-Entropy with class weighting
- **Class Weights:** Inverse frequency (rare classes weighted higher)
- **Ignore Index:** 255 (for unlabeled pixels)
- **Purpose:** Handle class imbalance (car: 141M pixels vs pedestrian: 8.2M pixels)

### 2. **Optimizer**
- **Type:** AdamW (Adam with weight decay)
- **Learning Rate:** 6e-5 (0.00006)
- **Weight Decay:** 0.01
- **Beta1/Beta2:** Default (0.9, 0.999)
- **Purpose:** Adaptive learning with regularization to prevent overfitting

### 3. **Learning Rate Scheduler**
- **Type:** WarmupThenSlash (custom)
- **Warmup:** Linear warmup for 5 epochs (0 → 6e-5)
- **Then:** Exponential decay every 10 epochs (multiply by 0.5)
- **Schedule:**
  - Epochs 1-5: Linear warmup (6e-5/5 = 1.2e-5 per epoch)
  - Epoch 6: 6e-5
  - Epoch 16: 3e-5
  - Epoch 26: 1.5e-5
  - Epoch 36: 7.5e-6
  - ...continues until epoch 100
- **Purpose:** Smooth convergence with gradual learning rate reduction

### 4. **Early Stopping**
- **Metric:** Validation loss
- **Patience:** 20 epochs
- **Purpose:** Stop training if no improvement for 20 consecutive epochs
- **Previous Issue:** Patience=5 was too aggressive, stopping too early
- **Fixed:** Increased to 20 to allow proper convergence

### 5. **Mixed Precision Training (AMP)**
- **Type:** CUDA Automatic Mixed Precision
- **GradScaler:** Prevents gradient underflow
- **Purpose:** 30-40% faster training with minimal accuracy loss

### 6. **Data Augmentation**
- **Image Resizing:** 512x512 (fixed size)
- **Normalization:** ImageNet statistics (mean, std)
- **Seed:** 42 (reproducibility)
- **Purpose:** Consistent preprocessing across batches

### 7. **Validation & Evaluation**
- **Per-Epoch Validation:** Compute mIoU on 491 validation samples
- **Metric:** Mean Intersection-over-Union (IoU) on 6 relevant classes only
- **Best Model Save:** Save when validation loss improves
- **Test Evaluation:** Run final evaluation on test set (488 samples)
- **Full Dataset Evaluation:** Optional evaluation on full 14K val set

---

## ✓ Checkpoint Management (CRITICAL)

### Checkpoint Policy: NEVER DELETE CHECKPOINTS

**Why?**
- Training neural networks is extremely expensive (40-50 GPU hours per model)
- Checkpoints are the only way to recover from issues without retraining
- Accidental deletion = loss of weeks of compute

### Checkpoint Types Saved

For each model, training saves **3 checkpoint files**:

1. **`checkpoint_best.pth` (109 MB per model)**
   - Full checkpoint with COMPLETE training state
   - Contains:
     - Model weights
     - Optimizer state (momentum, RMSprop cache)
     - Learning rate scheduler state
     - Current epoch number
     - Training history (all metrics)
     - Best mIoU achieved
   - **Use case:** Resume training from exact checkpoint
   - **Size:** ~109 MB

2. **`best.pth` (37 MB per model)**
   - Weights-only checkpoint
   - Contains: Model weights only
   - **Use case:** Inference, model deployment
   - **Size:** ~37 MB

3. **`checkpoint_epoch_X.pth` (109 MB per model)**
   - Periodic checkpoint every N epochs
   - Full state like `checkpoint_best.pth`
   - **Use case:** Manual resume if early stopping is too aggressive
   - **Size:** ~109 MB

### Checkpoint Backup
- **Location:** `outputs/checkpoints_backup/` 
- **Purpose:** Secondary backup of best checkpoints
- **Contains:** All successful model checkpoints from previous runs
- **Update Policy:** Copy new best checkpoints here regularly

### Saved Metadata
- **`results.json`** (per model)
  - Training history: loss, mIoU per epoch
  - Best validation metrics
  - Test set results
  - Full dataset evaluation (if run)

### Directory Structure
```
outputs/
├── checkpoints_backup/          # ← PROTECTED BACKUP
│   ├── best.pth
│   ├── checkpoint_best.pth
│   ├── checkpoint_epoch_20.pth
│   └── segformer_results.json
├── segformer_upernet/           # ← Model 1
│   ├── checkpoint_best.pth      # Full state
│   ├── best.pth                 # Weights only
│   ├── checkpoint_epoch_X.pth   # Periodic
│   ├── results.json             # Metrics
│   └── training_curves.png      # Plots
├── resnet101/                   # ← Model 2
│   ├── checkpoint_best.pth
│   ├── best.pth
│   ├── ...
├── swin_b/                      # ← Model 3
├── convnext_b/                  # ← Model 4
└── summary.json                 # Overall summary
```

### Checkpoint Recovery Procedure
If training is interrupted:
```bash
# Training automatically resumes if checkpoint exists
python train_nuimages.py --model swin_b --resume outputs/swin_b/checkpoint_best.pth

# The code will:
# 1. Load model weights
# 2. Load optimizer state
# 3. Load scheduler state
# 4. Continue from epoch N+1
# 5. Use same history and best metrics
```

---

## ✓ Training Configuration

```
Device:             CUDA (RTX 4090, 24GB VRAM)
Epochs:             100
Batch Size:         4-6 (depends on model)
Image Size:         512x512
Precision:          Mixed (FP16 + FP32)
Seed:               42 (reproducible)
Num Workers:        4 (data loading)
Pin Memory:         True (GPU optimization)
```

---

## ✓ Expected Performance

Based on previous training runs:

| Model | Baseline mIoU | Expected Target |
|-------|---------------|-----------------|
| SegFormer UPerNet | 0.0600 | > 0.30 |
| ResNet101 UPerNet | 0.3055 | > 0.35 |
| Swin-B UPerNet | 0.4290 | > 0.45 |
| ConvNeXt-B UPerNet | 0.3774 | > 0.40 |

---

## ✓ Previous Issues (FIXED)

### Issue 1: ✓ Scheduler Per-Batch Bug (FIXED)
- **Problem:** `lr_scheduler.step()` was called inside batch loop
- **Effect:** Learning rate collapsed to 7e-40, model froze
- **Fix:** Moved to once-per-epoch after validation

### Issue 2: ✓ Checkpoint Loading Reset (FIXED)
- **Problem:** Loaded weights-only, trained from epoch 0
- **Effect:** Optimizer momentum/scheduler lost
- **Fix:** Load full checkpoint, resume from correct epoch

### Issue 3: ✓ Early Stopping Too Aggressive (FIXED)
- **Problem:** Patience=5 stopped after 5-7 epochs
- **Effect:** Never reached convergence
- **Fix:** Increased patience to 20 epochs

### Issue 4: ✓ ConvNeXt-B OOM (FIXED)
- **Problem:** Batch size 6 exceeded 24GB VRAM
- **Effect:** Training crashed with OOM
- **Fix:** Reduced to batch size 4

---

## ✓ Monitoring Training

### Check Current Status
```bash
# Attach to tmux session
tmux attach-session -t model-training

# View without attaching
tmux capture-pane -t model-training -p | tail -50

# Kill if needed
tmux kill-session -t model-training
```

### View Training Curves
- Plots saved every 20 epochs: `outputs/MODEL_NAME/training_curves.png`
- Shows: train_loss, val_loss, val_mIoU over time

### View Metrics
```bash
cat outputs/MODEL_NAME/results.json | python -m json.tool
```

---

## ✓ Expected Training Timeline

| Model | Hours | Data | Status |
|-------|-------|------|--------|
| SegFormer UPerNet | ~20h | 2,344 train samples | Currently training |
| ResNet101 UPerNet | ~8h | 2,344 train samples | Pending |
| Swin-B UPerNet | ~12h | 2,344 train samples | Pending |
| ConvNeXt-B UPerNet | ~15h | 2,344 train samples | Pending |
| **TOTAL** | **~55h** | **Sequential (one model at a time)** | In progress |

---

## ✓ Output Files

After training completes:

```
outputs/
├── MODEL_NAME/
│   ├── checkpoint_best.pth          # ✓ Save to backup
│   ├── checkpoint_final_best.pth    # ✓ Save to backup
│   ├── checkpoint_epoch_*.pth       # ✓ Save to backup
│   ├── best.pth                     # ✓ Save to backup
│   ├── results.json                 # Training metrics
│   └── training_curves.png          # Visualization
```

**ALL CHECKPOINT FILES MUST BE BACKED UP TO `checkpoints_backup/`**

---

## ✓ Reproducibility Guarantee

- Seed: 42 (all random processes)
- PyTorch deterministic: True
- CUDA benchmark: False
- Same data ordering: Guaranteed by seed

→ **Same code + same seed = identical results**

---

## ✓ Important Notes

1. **First training run with fixed code** - expecting improvements
2. **All models train sequentially** - one at a time
3. **Checkpoints are precious** - never delete manually
4. **Early stopping is tuned** - patience=20 is final setting
5. **Class weighting applied** - handles data imbalance
6. **Validation every epoch** - monitor convergence in real-time
