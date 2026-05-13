# Training Guide: Reproducibility & Checkpoint Management

## Overview

The updated `train_nuimages.py` script now includes:
- **Random seed management** for reproducible results
- **Checkpoint saving** for resuming interrupted training
- **Full training state preservation** (model, optimizer, scheduler, history)

---

## Random Seed & Reproducibility

### Why This Matters

Without random seeds, every training run produces different results:
```
Run 1: Swin-B mIoU = 0.4431
Run 2: Swin-B mIoU = 0.4387  (different initialization!)
Run 3: Swin-B mIoU = 0.4456  (can't tell if changes are real improvements)
```

With seeds, results are deterministic and comparable.

### How It Works

```python
SEED = 42  # Global constant

def set_seed(seed=SEED):
    """Sets all random sources for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Affected randomness:**
- Python's `random` module
- NumPy random operations
- PyTorch tensor initialization
- GPU operations (CUDA)
- cuDNN algorithms

### Using Custom Seeds

```bash
# Default seed (42)
python train_nuimages.py --model swin_b --data_root /home/tamoghno/datasets/nuimages

# Custom seed
python train_nuimages.py --model swin_b --data_root /home/tamoghno/datasets/nuimages --seed 123

# Another run with same seed (should produce identical results)
python train_nuimages.py --model swin_b --data_root /home/tamoghno/datasets/nuimages --seed 123
```

### Deterministic Mode Trade-offs

```
✓ Advantages:
  - Reproducible results
  - Fair comparison between models
  - Meaningful improvement tracking

✗ Disadvantages:
  - Slightly slower training (~1-5% slower)
  - Can't use some GPU optimizations
  - Larger memory footprint
```

**Recommendation:** Keep seeds enabled for research/experimentation. You can disable for production:

```python
# To disable deterministic mode (faster but non-reproducible):
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
```

---

## Checkpoint Management

### Checkpoint Types

#### 1. **Best Model** (`best.pth`)
- Saved when validation mIoU improves
- **Only contains model weights** (lightweight, 428 MB)
- Use for: Inference, final evaluation

#### 2. **Best Checkpoint** (`checkpoint_best.pth`)
- Saved when validation mIoU improves
- **Contains full training state** (model, optimizer, scheduler, history)
- Use for: Resuming training from best point during interrupted runs

#### 3. **Final Best Checkpoint** (`checkpoint_final_best.pth`) ⭐ **NEW**
- Saved **automatically when training completes**
- **Contains full training state** from the epoch with the best validation mIoU
- **This is the recommended checkpoint to use for resuming or transfer learning**
- Guaranteed to be the best checkpoint from this training run
- Use for: Resuming training, transfer learning, guaranteed best state

#### 4. **Periodic Checkpoints** (`checkpoint_epoch_*.pth`)
- Saved every 5 epochs (epoch 5, 10, 15, ...)
- **Contains full training state**
- Use for: Analyzing training curves, recovery from crashes

### Checkpoint Contents

```json
{
  "epoch": 19,
  "model_state": {...},              // Model weights
  "optimizer_state": {...},          // Optimizer momentum/state
  "scheduler_state": {...},          // Learning rate scheduler state
  "best_miou": 0.4431,              // Best validation mIoU so far
  "history": [...]                   // All previous epochs' metrics
}
```

### File Locations

```
outputs/
├── swin_b/
│   ├── best.pth                    (428 MB, weights only)
│   ├── checkpoint_best.pth         (~500 MB, full state, updated during training)
│   ├── checkpoint_final_best.pth   (~500 MB, full state, FINAL BEST ⭐)
│   ├── checkpoint_epoch_5.pth      (~500 MB)
│   ├── checkpoint_epoch_10.pth     (~500 MB)
│   ├── checkpoint_epoch_15.pth     (~500 MB)
│   └── results.json
├── convnext_b/
│   ├── best.pth
│   ├── checkpoint_best.pth
│   ├── checkpoint_final_best.pth   ⭐ Use this for resuming!
│   └── results.json
└── resnet101/
    ├── best.pth
    ├── checkpoint_best.pth
    ├── checkpoint_final_best.pth
    └── ...
```

**Key Point:** `checkpoint_final_best.pth` is automatically created when training completes and is guaranteed to be the checkpoint with the best validation mIoU from that run.

---

## Resuming Training

### ⭐ Recommended: Resume from Final Best Checkpoint

```bash
# After training completes, use checkpoint_final_best.pth
# This is GUARANTEED to be the best checkpoint from training

python train_nuimages.py \
  --model swin_b \
  --data_root /home/tamoghno/datasets/nuimages \
  --epochs 100 \
  --resume ./outputs/swin_b/checkpoint_final_best.pth
```

**What happens:**
- Model weights from best epoch restored
- Optimizer state restored (momentum, etc.)
- Scheduler state restored (learning rate resets)
- Training continues from epoch where it left off
- Full history preserved

### Scenario 1: Resume from Last Checkpoint

```bash
# Training was interrupted at epoch 17/50
# Use checkpoint_best.pth or periodic checkpoint

python train_nuimages.py \
  --model swin_b \
  --data_root /home/tamoghno/datasets/nuimages \
  --resume ./outputs/swin_b/checkpoint_best.pth
```

**Difference from final_best:**
- `checkpoint_best.pth`: Updated during training, may not be the final best
- `checkpoint_final_best.pth`: Only created at the END, guaranteed to be the best

### Scenario 2: Resume from Specific Epoch

```bash
# Want to resume from epoch 10 specifically

python train_nuimages.py \
  --model swin_b \
  --data_root /home/tamoghno/datasets/nuimages \
  --resume ./outputs/swin_b/checkpoint_epoch_10.pth
```

### Scenario 3: Transfer Learning (Load Weights Only)

```bash
# Load best.pth (model weights only) and restart training
# This resets optimizer and scheduler

python train_nuimages.py \
  --model swin_b \
  --data_root /home/tamoghno/datasets/nuimages
  
# Then manually copy best.pth to checkpoint_best.pth for next resume
```

---

## Common Workflows

### Workflow 1: Standard Training (Fresh)

```bash
python train_nuimages.py \
  --model swin_b \
  --data_root /home/tamoghno/datasets/nuimages \
  --epochs 50 \
  --batch_size 6

# Automatic outputs:
# - outputs/swin_b/best.pth
# - outputs/swin_b/checkpoint_best.pth
# - outputs/swin_b/checkpoint_epoch_5.pth
# - outputs/swin_b/checkpoint_epoch_10.pth
# - outputs/swin_b/results.json
```

### Workflow 2: Interrupted Training (Resume)

```bash
# Original command (interrupted at epoch 17/50)
# python train_nuimages.py --model swin_b ... --epochs 50

# Resume using the final best checkpoint (RECOMMENDED)
python train_nuimages.py \
  --model swin_b \
  --data_root /home/tamoghno/datasets/nuimages \
  --epochs 100 \
  --resume ./outputs/swin_b/checkpoint_final_best.pth

# This will continue training from epoch ~17 to 100 total epochs
```

**Why use `checkpoint_final_best.pth`?**
- ✓ Guaranteed to be the best checkpoint from training
- ✓ Created automatically when training completes
- ✓ No ambiguity about which checkpoint is best
- ✓ Perfect for extending training or hyperparameter tuning

### Workflow 3: Hyperparameter Tuning (Different Random Seeds)

```bash
# Run 1: seed 42
python train_nuimages.py \
  --model swin_b \
  --data_root /home/tamoghno/datasets/nuimages \
  --seed 42 \
  --epochs 50

# Run 2: seed 123
python train_nuimages.py \
  --model swin_b \
  --data_root /home/tamoghno/datasets/nuimages \
  --seed 123 \
  --epochs 50

# Run 3: seed 456
python train_nuimages.py \
  --model swin_b \
  --data_root /home/tamoghno/datasets/nuimages \
  --seed 456 \
  --epochs 50

# Average results across 3 runs for robust evaluation
```

### Workflow 4: Multi-Model Training with Reproducibility

```bash
# Train all models with same seed (fair comparison)
python train_nuimages.py \
  --model all \
  --data_root /home/tamoghno/datasets/nuimages \
  --seed 42 \
  --epochs 50

# Produces:
# outputs/swin_b/
# outputs/convnext_b/
# outputs/resnet101/
# outputs/segformer_upernet/
```

---

## Debugging & Verification

### Check if Seed is Applied

The script prints when seed is set:

```
✓ Random seed set to 42 for reproducibility
```

### Verify Reproducibility

```bash
# Run 1
python train_nuimages.py --model swin_b --data_root /path --epochs 10 --seed 42
# Output: best_miou = 0.1234

# Run 2 (identical parameters)
python train_nuimages.py --model swin_b --data_root /path --epochs 10 --seed 42
# Output: best_miou = 0.1234  ✓ SAME!

# Run 3 (different seed)
python train_nuimages.py --model swin_b --data_root /path --epochs 10 --seed 123
# Output: best_miou = 0.1198  ✓ DIFFERENT (expected)
```

### Resume Checkpoint Validation

After resuming:

```bash
# Resume from epoch 15 checkpoint
python train_nuimages.py --resume outputs/swin_b/checkpoint_epoch_15.pth ...

# Should print:
# ⟳ Resuming from checkpoint: outputs/swin_b/checkpoint_epoch_15.pth
# ✓ Model weights loaded
# ✓ Optimizer state loaded
# ✓ Learning rate scheduler state loaded
# ✓ Resuming from epoch 16
# ✓ Best mIoU: 0.4431
# ✓ Training history loaded (15 epochs)
```

---

## Tips & Best Practices

### ✓ Do's

- **Always use a fixed seed** when comparing models or hyperparameters
- **Save checkpoints** before making large changes to code
- **Resume from `checkpoint_best.pth`** (not `best.pth`) to maintain optimizer state
- **Use different seeds** (42, 123, 456) for ensemble training
- **Document seed values** when reporting results

### ✗ Don'ts

- Don't change batch size when resuming (breaks scheduler)
- Don't change data splits when resuming (inconsistent training)
- Don't mix seeds in ensemble (loses reproducibility value)
- Don't use `best.pth` for resuming if you need optimizer state
- Don't rely on non-seeded runs for research conclusions

### Storage Management

```bash
# Best.pth files are small (428 MB each)
du -sh outputs/*/best.pth

# Checkpoints are larger
du -sh outputs/*/checkpoint_*.pth

# Clean up old checkpoints to save space
rm outputs/swin_b/checkpoint_epoch_5.pth
rm outputs/swin_b/checkpoint_epoch_10.pth
# Keep checkpoint_best.pth and final best.pth
```

---

## Command Reference

```bash
# Training with custom seed
--seed VALUE                    Default: 42

# Resume training
--resume PATH/TO/CHECKPOINT     Default: None (fresh training)

# Load pretrained weights
--load_pretrained              Default: False (not yet implemented)

# All parameters together
python train_nuimages.py \
  --model swin_b \
  --data_root /path/to/nuimages \
  --epochs 50 \
  --batch_size 6 \
  --img_size 512 \
  --lr 6e-5 \
  --seed 42 \
  --resume ./outputs/swin_b/checkpoint_best.pth
```

---

## Summary

| Feature | Benefit |
|---------|---------|
| Random Seeding | Reproducible, fair model comparison |
| Full Checkpoints | Resume training from interruptions |
| Final Best Checkpoint | Guaranteed best checkpoint after training completes ⭐ |
| Periodic Saves | Recovery from crashes every 5 epochs |
| History Tracking | Monitor training progress across resumes |
| Optimizer State | Proper learning rate continuation |

Your training is now **production-ready** with proper reproducibility and fault-tolerance! 🚀
