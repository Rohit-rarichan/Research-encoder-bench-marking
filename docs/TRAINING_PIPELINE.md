# Sequential Training Pipeline

## Quick Start

Run all 4 models back-to-back with a single command:

```bash
bash /home/tamoghno/rohit-encoders/start_training.sh
```

## What It Does

The training pipeline will:

1. **Activate conda environment** `encoders`
2. **Run models sequentially:**
   - SegFormer → ResNet101 → Swin-B → ConvNeXt-B
3. **Resume from checkpoints** if available (continues training)
4. **Configuration:**
   - **Epochs:** 100
   - **Batch Size:** 6
   - **Seed:** 42 (reproducible)
   - **Data:** 5% subset for training
   - **Full Dataset Eval:** 95% evaluation for generalization
5. **Auto-detect hardware** and manage memory
6. **Run in tmux** for persistent execution

## Usage

### Option 1: Launch in tmux (Recommended)

```bash
bash /home/tamoghno/rohit-encoders/start_training.sh
```

Then attach to monitor:

```bash
tmux attach-session -t model-training
```

### Option 2: Direct execution

```bash
bash /home/tamoghno/rohit-encoders/train_all_models.sh
```

## Monitoring

### While training (tmux attached)

Watch real-time output:
- Train/Val loss decreasing
- mIoU values improving
- Learning rate changes (warmup → slash phases)
- Checkpoint saves every 20 epochs
- Plot generation every 20 epochs
- Early stopping status

### Detach without stopping

Press `Ctrl+B` then `D` (training continues)

### Check progress from terminal

```bash
# View current status
tmux capture-pane -t model-training -p

# Monitor GPU memory
watch -n 1 nvidia-smi

# View logs for specific model
tail -f outputs/segformer_upernet/training.log
```

## Expected Runtime

```
SegFormer:    ~4-6 hours   (lighter model)
ResNet101:    ~6-8 hours   (medium model)
Swin-B:       ~8-12 hours  (large model)
ConvNeXt-B:   ~10-14 hours (largest model)
─────────────────────────
Total:        ~35-50 hours (with early stopping, likely ~25-35 hours)
```

## Output Files

After each model completes:

```
outputs/
├── segformer_upernet/
│   ├── best.pth                          (weights only)
│   ├── checkpoint_final_best.pth         (full state)
│   ├── checkpoint_epoch_20.pth           (periodic saves)
│   ├── checkpoint_epoch_40.pth
│   ├── checkpoint_epoch_60.pth
│   ├── checkpoint_epoch_80.pth
│   ├── checkpoint_epoch_100.pth          (if not early stopped)
│   ├── training_curves.png               (at epochs 20, 40, 60, 80, 100)
│   ├── training_curves_final.png         (at end)
│   └── results.json                      (mIoU on 5% + full 95% dataset)
│
├── resnet101/                            (same structure)
├── swin_b/                               (same structure)
└── convnext_b/                           (same structure)
```

## Results Interpretation

Each `results.json` contains:

```json
{
  "best_miou": 0.45,                           // Best mIoU on 5% test set
  "val_results": {"miou": 0.44, ...},          // 5% validation set
  "test_results": {"miou": 0.45, ...},         // 5% test set
  "full_dataset_results": {"miou": 0.38, ...}, // 95% remaining dataset
  "history": [...]                            // Epoch-by-epoch metrics
}
```

**Key insight:** Compare `test_results.miou` vs `full_dataset_results.miou`
- If similar: Good generalization
- If test much higher: Possible overfitting to 5% subset

## Checkpoint Resume

Scripts automatically detect and resume from best checkpoints. To reset and train fresh:

```bash
# Remove checkpoints
rm -rf outputs/segformer_upernet/checkpoint*.pth

# Training will start fresh
bash /home/tamoghno/rohit-encoders/start_training.sh
```

## Troubleshooting

### CUDA Out of Memory

If you see "CUDA out of memory":
1. Stop training: `tmux kill-session -t model-training`
2. Edit `train_all_models.sh` and reduce `BATCH_SIZE` from 6 to 4
3. Restart: `bash start_training.sh`

### Training too slow

- Check `nvidia-smi` to verify GPU is being used
- If GPU utilization low, increase batch size to 8 (if memory allows)

### Early stopping too aggressive

Edit `train_nuimages.py` line 157:
```python
EarlyStopping(patience=5)  # Change 5 to 10 for more patience
```

### Want to train only specific models

Edit `train_all_models.sh` line 59:
```bash
# Change this line to pick which models to train
MODELS=("segformer_upernet" "resnet101")  # Skip swin_b and convnext_b
```

## Benefits of This Pipeline

✅ **Hands-off:** Start and forget, runs all 4 models automatically
✅ **Checkpoint aware:** Resumes from best checkpoints, not wasted effort
✅ **Reproducible:** Fixed seed (42), same results every run
✅ **Monitored:** Watch progress in real-time with tmux
✅ **Full eval:** Automatically tests on remaining 95% dataset
✅ **Production-ready:** All optimizations enabled (class weighting, LR scheduling, early stopping)

## After Training

Compare results across models:

```bash
python3 << 'EOF'
import json
import os

models = ["segformer_upernet", "resnet101", "swin_b", "convnext_b"]

print("\n" + "="*70)
print("Model Performance Comparison (100 epochs, batch_size 6)")
print("="*70 + "\n")

for model in models:
    results_path = f"outputs/{model}/results.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
            
        test_miou = results.get('test_results', {}).get('miou', 0)
        full_miou = results.get('full_dataset_results', {}).get('miou', 0)
        
        print(f"{model:20} | 5% Test: {test_miou:.4f} | Full Dataset: {full_miou:.4f}")
    else:
        print(f"{model:20} | Not trained yet")

print("\n" + "="*70 + "\n")
EOF
```

---

**Ready to train!** 🚀 Run: `bash /home/tamoghno/rohit-encoders/start_training.sh`
