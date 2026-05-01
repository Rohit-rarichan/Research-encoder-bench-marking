# 🎯 QUICK REFERENCE CARD

## 📊 Data at a Glance
```
Training:   2,344 samples
Validation:   491 samples  
Test:         488 samples
Classes:       11 total (6 counted in mIoU)
```

## 🏗️ 4 Models Training
```
1. SegFormer UPerNet    (7M params)   Batch 4   ~20 hours
2. ResNet101 UPerNet   (47M params)   Batch 6   ~8 hours
3. Swin-B UPerNet      (91M params)   Batch 6   ~12 hours
4. ConvNeXt-B UPerNet  (92M params)   Batch 4   ~15 hours
                                            Total: ~55 hours
```

## 🔧 Techniques Used
```
✓ Cross-Entropy Loss + Class Weighting
✓ AdamW Optimizer (lr=6e-5, weight_decay=0.01)
✓ WarmupThenSlash Scheduler (warmup 5ep, slash every 10ep)
✓ Early Stopping (patience=20 epochs)
✓ Mixed Precision Training (AMP)
✓ Per-Epoch Validation
✓ ImageNet Normalization
```

## 💾 Checkpoint Policy: **NEVER DELETE**
```
Saved per model:
  - checkpoint_best.pth      (109 MB, full state)
  - checkpoint_final_best.pth (109 MB, full state)
  - best.pth                 (37 MB, weights only)
  - checkpoint_epoch_*.pth   (109 MB periodic)

Backup location:
  outputs/checkpoints_backup/  ← Read-only protection
```

## 🚀 Live Training Commands
```bash
# Monitor
tmux capture-pane -t model-training -p | tail -50

# Backup after each model finishes
bash backup_checkpoints.sh

# View results
cat outputs/MODEL_NAME/results.json | python -m json.tool

# Verify checkpoints
bash backup_checkpoints.sh --verify
```

## ⚠️ RULES (DO NOT BREAK)
```
1. NEVER DELETE any .pth checkpoint files
2. ALWAYS run backup_checkpoints.sh after training
3. KEEP outputs/checkpoints_backup/ read-only
4. VERIFY checksums after each backup
```

## 📈 Expected Results
```
SegFormer:   0.06  → > 0.30  (5× improvement)
ResNet101:   0.31  → > 0.35  (1.1× improvement)
Swin-B:      0.43  → > 0.45  (1.05× improvement)
ConvNeXt-B:  0.38  → > 0.40  (1.06× improvement)
```

## 🛑 If Training Crashes
```bash
# 1. Training auto-resumes from checkpoint
python train_nuimages.py --model MODEL_NAME

# 2. Or explicitly resume:
python train_nuimages.py --model MODEL_NAME \
    --resume outputs/MODEL_NAME/checkpoint_best.pth

# 3. No data loss - continues from epoch N+1
```

---

**Training Started:** April 24, 2026  
**Session:** tmux (model-training)  
**Status:** SegFormer training now... → ResNet101 → Swin-B → ConvNeXt-B  
**Est. Completion:** April 28, 2026 (~55 hours)
