# 📊 Training Pipeline - Complete Summary

## ✅ Data Distribution Overview

### Dataset: NuImages 5% Stratified Subset
```
Total Samples: 3,323 (5% of full nuImages dataset)
├─ Training:   2,344 samples (70.5%)  ← Primary training data
├─ Validation:   491 samples (14.8%)  ← Per-epoch validation
└─ Test:         488 samples (14.7%)  ← Final evaluation
```

### Data Characteristics
- **Image Resolution:** 512×512 pixels
- **Semantic Classes:** 11 total
  - **Relevant (mIoU counted):** car, truck, bus, bicycle, motorcycle, pedestrian (6 classes)
  - **Surface/Ignored:** driveable_surface, other_flat, terrain, manmade, vegetation (5 classes)
- **Dataset:** Autonomous driving scene understanding (nuImages)
- **Location:** `/home/tamoghno/datasets/nuimages/`

### Class Distribution (Training Data)
| Class | Pixels | Weight | Notes |
|-------|--------|--------|-------|
| car | 141.0M | 0.082 | Most common |
| truck | 31.9M | 0.363 | Medium freq |
| bus | 11.0M | 1.054 | Uncommon |
| bicycle | 2.6M | 4.483 | **Rare** (highest weight) |
| motorcycle | 4.3M | 2.719 | **Rare** |
| pedestrian | 8.2M | 1.414 | Uncommon |
| driveable_surface | 676.0M | 0.017 | Dominant (ignored in mIoU) |
| other_flat | 26.1M | 0.444 | - |
| terrain | 0.0M | 0.141 | Zero-shot class |
| manmade | 0.0M | 0.141 | Zero-shot class |
| vegetation | 0.0M | 0.141 | Zero-shot class |

**Total Training Pixels:** ~900M

---

## 🏗️ Model Architectures

All models use **UPerNet (Unified Perceptron) decoder** with task-specific backbones:

### 1. SegFormer UPerNet
- **Backbone:** MiT-B0 (Hierarchical Transformer, custom implementation)
- **Parameters:** ~4M encoder + ~3M decoder = ~7M total
- **Batch Size:** 4 (memory-optimized)
- **GPU Memory:** ~18-20 GB
- **Strengths:** Efficient, modern transformer architecture
- **Status:** 🔄 Currently training (Epoch 1/100)

### 2. ResNet101 UPerNet
- **Backbone:** ResNet101 (Standard CNN)
- **Parameters:** ~44M encoder + ~3M decoder = ~47M total
- **Batch Size:** 6
- **GPU Memory:** ~20 GB
- **Pretrained Weights:** ImageNet1K-V2 (torchvision) ✓
- **Strengths:** Proven, standard baseline
- **Status:** ⏳ Pending (starts after SegFormer finishes)

### 3. Swin-B UPerNet
- **Backbone:** Swin-B (Shifted Window Transformer)
- **Parameters:** ~88M encoder + ~3M decoder = ~91M total
- **Batch Size:** 6
- **GPU Memory:** ~22 GB
- **Strengths:** State-of-the-art transformer, previously: 0.4290 mIoU
- **Status:** ⏳ Pending

### 4. ConvNeXt-B UPerNet
- **Backbone:** ConvNeXt-B (Modern CNN with attention)
- **Parameters:** ~89M encoder + ~3M decoder = ~92M total
- **Batch Size:** 4 (OOM issue at 6, now fixed)
- **GPU Memory:** ~20 GB
- **Strengths:** Modern architecture, efficient
- **Status:** ⏳ Pending

---

## 🎯 Training Techniques

### 1. **Loss Function: Cross-Entropy with Class Weighting**
```
Loss = CrossEntropyLoss(weight=class_weights, ignore_index=255)
```
- **Purpose:** Handle extreme class imbalance
  - Driveable surface: 676M pixels
  - Bicycle: 2.6M pixels (260× difference!)
- **Weights:** Inverse frequency normalization
  - Rare classes (bicycle, motorcycle): weight ≈ 2-4
  - Common classes (car, surface): weight ≈ 0.01-0.08
- **Ignore Index:** 255 (unmapped/void pixels ignored in loss)

### 2. **Optimizer: AdamW (Adaptive Moment Estimation with Weight Decay)**
```
AdamW(lr=6e-5, weight_decay=0.01)
```
- **Learning Rate:** 6e-5 (0.00006)
- **Beta1/Beta2:** 0.9, 0.999 (momentum/RMSprop)
- **Weight Decay:** 0.01 (L2 regularization)
- **Purpose:** Adaptive learning with regularization
  - Faster convergence than SGD
  - Better generalization than Adam (due to decoupled weight decay)

### 3. **Learning Rate Schedule: WarmupThenSlash**
```
Warmup:  Epochs 1-5   (linear 0 → 6e-5)
Slash:   Every 10 epochs (multiply by 0.5)
```
**Schedule Over 100 Epochs:**
| Epoch Range | LR | Note |
|-------------|-----|------|
| 1-5 | Linear warmup → 6e-5 | Gradual start |
| 6-15 | 6e-5 | Full learning rate |
| 16-25 | 3e-5 | First reduction |
| 26-35 | 1.5e-5 | Second reduction |
| 36-45 | 7.5e-6 | Third reduction |
| 46-55 | 3.75e-6 | Fine-tuning range |
| 56+ | Continues halving | Ultra fine-tuning |

**Why?** Gradual learning rate decay helps navigate loss landscape efficiently.

### 4. **Early Stopping: Patience=20 Epochs**
```
Monitor: Validation Loss
Patience: 20 epochs (no improvement before stopping)
```
- **Previous Issue:** Patience=5 was too aggressive
  - Stopped at epoch 5-7 before convergence
  - Lost 30-40 hours of potential training
- **Fixed:** Patience=20 (allows proper convergence)
  - Typical convergence: 20-35 epochs
  - Backup buffer: 20 epochs after best

### 5. **Mixed Precision Training (Automatic Mixed Precision)**
```
with autocast():
    outputs = model(images)
    loss = criterion(outputs, masks)
scaler.scale(loss).backward()
```
- **Benefits:**
  - 30-40% faster training
  - 20-30% less VRAM usage
  - Minimal accuracy loss (<0.1%)
- **How:** Uses FP16 for fast ops, FP32 for sensitive operations
- **GradScaler:** Prevents gradient underflow in FP16

### 6. **Data Normalization & Consistency**
```
ImageNet Statistics:
  mean = [0.485, 0.456, 0.406]
  std  = [0.229, 0.224, 0.225]
```
- **Image Resizing:** 512×512 (fixed)
- **Seed:** 42 (reproducible random operations)
- **Purpose:** Consistent preprocessing across all batches

### 7. **Per-Epoch Validation**
```
Every epoch:
1. Run inference on 491 validation samples
2. Compute mIoU (mean Intersection-over-Union)
3. Save if validation loss improves
4. Update training history
```
- **Metric:** mIoU on 6 relevant classes only
- **Frequency:** Every epoch
- **Action:** Save checkpoint if loss decreases

### 8. **Batch Normalization Statistics**
```
BatchNorm(momentum=0.1, eps=1e-5)
```
- Tracks running statistics during training
- Used for inference with exponential moving average
- Stabilizes training, reduces internal covariate shift

---

## 📁 Checkpoint Management (CRITICAL ⚠️)

### Policy: **NEVER DELETE CHECKPOINTS**
This is a hard rule due to computational cost (40-50 GPU hours per model).

### Checkpoint Files Saved Per Model

| File | Size | State | Purpose | Keep |
|------|------|-------|---------|------|
| `checkpoint_best.pth` | 109 MB | **Full** (model, optimizer, scheduler, epoch, history) | Resume training | **YES** |
| `checkpoint_final_best.pth` | 109 MB | **Full** | Final best when training stops | **YES** |
| `checkpoint_epoch_X.pth` | 109 MB | **Full** | Periodic backup every N epochs | **YES** |
| `best.pth` | 37 MB | Weights only | Inference/deployment | **YES** |
| `results.json` | <1 MB | Metrics only | Training history, final scores | **YES** |

### Checkpoint Backup Structure
```
outputs/
├── checkpoints_backup/                    ← PRIMARY BACKUP (Read-only)
│   ├── segformer_upernet_20260424_1306/
│   │   ├── checkpoint_best.pth
│   │   ├── best.pth
│   │   ├── checkpoint_epoch_20.pth
│   │   └── results.json
│   ├── resnet101_20260424_1400/
│   ├── swin_b_20260424_1500/
│   └── convnext_b_20260424_1600/
│
├── segformer_upernet/                     ← Active training dir
│   ├── checkpoint_best.pth
│   ├── checkpoint_final_best.pth
│   ├── best.pth
│   ├── checkpoint_epoch_*.pth
│   ├── results.json
│   └── training_curves.png
│
├── resnet101/
├── swin_b/
└── convnext_b/
```

### Backup Automation
```bash
# After each model completes:
bash backup_checkpoints.sh

# This will:
# 1. Copy all .pth files to backups directory
# 2. Create timestamped subdirectory
# 3. Verify file integrity
# 4. Make backups read-only (prevent deletion)
```

### Resume Training
```bash
# Training automatically detects and resumes from checkpoint:
python train_nuimages.py --model swin_b

# Or explicitly:
python train_nuimages.py --model swin_b \
    --resume outputs/swin_b/checkpoint_best.pth
```

When resuming:
- ✓ Loads model weights
- ✓ Restores optimizer state (momentum)
- ✓ Restores scheduler state
- ✓ Continues from correct epoch
- ✓ Keeps training history

---

## ⚙️ Hardware & Configuration

```
GPU:                RTX 4090 (24 GB VRAM)
CUDA Version:       12.4
PyTorch Version:    2.5.1
Python Version:     3.11
Conda Environment:  encoders

Training Mode:      Sequential (one model at a time)
Parallelization:    None (single GPU)
Mixed Precision:    CUDA AMP (FP16 + FP32)
Deterministic:      Yes (seed=42)
Benchmark:          Off (for reproducibility)
```

---

## 📈 Expected Timeline & Performance

### Training Duration Per Model
| Model | Duration | GPU Hours | Notes |
|-------|----------|-----------|-------|
| SegFormer UPerNet | ~20 hours | ~20 | Smallest model |
| ResNet101 UPerNet | ~8 hours | ~8 | Standard baseline |
| Swin-B UPerNet | ~12 hours | ~12 | Large transformer |
| ConvNeXt-B UPerNet | ~15 hours | ~15 | Large CNN |
| **TOTAL** | **~55 hours** | **~55 GPU hours** | Sequential |

### Expected Performance Targets
| Model | Previous Best | Target | Improvement |
|-------|---------------|--------|-------------|
| SegFormer | 0.0600 | >0.30 | 5× |
| ResNet101 | 0.3055 | >0.35 | 1.1× |
| Swin-B | 0.4290 | >0.45 | 1.05× |
| ConvNeXt-B | 0.3774 | >0.40 | 1.06× |

---

## 🔧 Key Fixes Applied Before Training

### ✓ Fix 1: Scheduler Per-Batch Bug
- **Problem:** LR scheduler step() called inside batch loop
- **Effect:** LR collapsed to 7e-40, model froze at 0.06 mIoU
- **Fix:** Moved to once-per-epoch after validation
- **Status:** ✅ FIXED

### ✓ Fix 2: Checkpoint Loading Reset
- **Problem:** Loaded weights-only, trained from epoch 0
- **Effect:** Optimizer/scheduler reset, lost 30+ hours of training
- **Fix:** Load full checkpoint, resume from correct epoch
- **Status:** ✅ FIXED

### ✓ Fix 3: Early Stopping Too Aggressive
- **Problem:** Patience=5 stopped at epoch 5-7
- **Effect:** Never reached convergence
- **Fix:** Increased patience to 20 epochs
- **Status:** ✅ FIXED

### ✓ Fix 4: ConvNeXt-B OOM
- **Problem:** Batch size 6 exceeded 24GB VRAM
- **Effect:** CUDA out of memory on epoch 1
- **Fix:** Reduced to batch size 4
- **Status:** ✅ FIXED

---

## 🚀 Training Monitoring

### Current Status
```
Model:              segformer_upernet
Epoch:              1/100
Batch Size:         4
Training Samples:   2,344 (586 batches)
Validation Samples: 491 (123 batches)

Status:             Training in progress
Session:            tmux (model-training)
```

### View Training Live
```bash
# Attach to tmux session
tmux attach-session -t model-training

# View without attaching
tmux capture-pane -t model-training -p | tail -50

# Kill session (only if needed)
tmux kill-session -t model-training
```

### View Results After Each Epoch
```bash
# After training:
cat outputs/MODEL_NAME/results.json | python -m json.tool

# Shows:
# - best_miou
# - test_results (if evaluated)
# - full_dataset_results (if evaluated)
# - training history (loss, mIoU per epoch)
```

### Visualize Training Curves
```bash
# Every 20 epochs:
cat outputs/MODEL_NAME/training_curves.png
```
Shows: train_loss, val_loss, val_mIoU over time

---

## 📝 Important Notes

1. **First fresh training with all fixes applied**
   - Scheduler fixed: LR won't collapse
   - Checkpoint loading fixed: can resume properly
   - Early stopping fixed: won't stop too early
   
2. **Sequential training:**
   - SegFormer starts now (~20 hours)
   - ResNet101 starts after SegFormer finishes
   - Swin-B and ConvNeXt-B follow
   
3. **Checkpoint policy is STRICT:**
   - NEVER delete .pth files manually
   - Use backup_checkpoints.sh for management
   - Backups are read-only by default
   
4. **Expected convergence:**
   - Most models converge within 20-35 epochs
   - Early stopping with patience=20 provides buffer
   - May reach 100 epochs if improvement continues

5. **Class weighting handles imbalance:**
   - Bicycle: 4.5× weight (only 2.6M pixels)
   - Car: 0.08× weight (141M pixels)
   - Prevents model from ignoring rare classes

---

## ✅ Quick Reference

| Command | Purpose |
|---------|---------|
| `tmux attach-session -t model-training` | Monitor training |
| `tmux capture-pane -t model-training -p \| tail -50` | View last 50 lines |
| `bash backup_checkpoints.sh` | Backup all checkpoints |
| `bash backup_checkpoints.sh --list` | Show backup contents |
| `bash backup_checkpoints.sh --verify` | Verify checkpoint integrity |
| `cat outputs/MODEL/results.json \| python -m json.tool` | View training metrics |

---

**Last Updated:** April 24, 2026  
**Training Status:** In Progress ✓  
**Checkpoint Policy:** Active - Never Delete ⚠️
