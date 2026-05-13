# Checkpoint Protection Policy

## ✓ CRITICAL RULE: NEVER DELETE CHECKPOINTS

**This repository has a strict policy on checkpoint management.**

### Why?
- Neural network training on RTX 4090: **40-50 GPU hours per model**
- All 4 models: **~200 total GPU hours**
- Training cost (AWS p100-8x): **$500+ to retrain**
- Accidental deletion = complete loss of compute investment

---

## ✓ Safe Checkpoint Deletion

**Only allowed if:**
1. Checkpoint is marked as "redundant" in CHECKPOINTS.md
2. Newer checkpoint exists with better metrics
3. Explicit approval in commit message

**Even then:**
1. Always keep `checkpoint_best.pth` for each model
2. Always keep `checkpoints_backup/` directory
3. Keep at least 2 backup copies

---

## ✓ Checkpoint Backup Process

### Automatic (during training)
```bash
# Every epoch:
# - Save checkpoint_best.pth (full state)
# - Save best.pth (weights only)
# - Save checkpoint_epoch_X.pth (periodic)
```

### Manual (after training)
```bash
# After each model finishes:
bash backup_checkpoints.sh

# This:
# 1. Copies all .pth files to checkpoints_backup/
# 2. Creates timestamped subdirectory
# 3. Verifies file integrity
# 4. Makes backups read-only
```

### Verification
```bash
# Check backup contents
bash backup_checkpoints.sh --list

# Verify all checkpoints are loadable
bash backup_checkpoints.sh --verify
```

---

## ✓ Directory Structure

```
outputs/
├── checkpoints_backup/                    ← PRIMARY BACKUP (READ-ONLY)
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
│   ├── best.pth
│   ├── checkpoint_epoch_X.pth
│   ├── results.json
│   └── training_curves.png
│
├── resnet101/
├── swin_b/
└── convnext_b/
```

---

## ✓ Checkpoint Files Explained

| File | Size | Contents | Keep | Use |
|------|------|----------|------|-----|
| `checkpoint_best.pth` | 109 MB | Full state (model, optimizer, scheduler, epoch, history) | **YES** | Resume training |
| `checkpoint_final_best.pth` | 109 MB | Final best state when training stopped | **YES** | Backup |
| `checkpoint_epoch_X.pth` | 109 MB | State at epoch X | **MAYBE** | Manual recovery |
| `best.pth` | 37 MB | Weights only | **YES** | Inference |
| `results.json` | <1 MB | Training metrics | **YES** | Analysis |

---

## ✓ Git Hooks (Prevent Accidental Deletion)

Add to `.git/hooks/pre-commit`:
```bash
#!/bin/bash
# Prevent committing checkpoint deletions

if git diff --cached --name-only | grep -q "^outputs.*\.pth$"; then
    echo "ERROR: Attempting to delete checkpoints in commit"
    echo "Checkpoints are protected - use 'git restore' if accidental"
    exit 1
fi
```

---

## ✓ Recovery Procedures

### If checkpoint is deleted:
```bash
# 1. Check if in backup
ls outputs/checkpoints_backup/*/checkpoint_best.pth

# 2. Restore from backup
cp outputs/checkpoints_backup/MODEL_*/checkpoint_best.pth \
   outputs/MODEL/checkpoint_best.pth

# 3. Resume training
python train_nuimages.py --model MODEL --resume outputs/MODEL/checkpoint_best.pth
```

### If all checkpoints lost:
```bash
# 1. Check git history
git log --oneline outputs/

# 2. Check git stash
git stash list

# 3. Recover from recent commit
git show COMMIT:outputs/MODEL/checkpoint_best.pth > checkpoint.pth

# 4. Last resort: restart training from scratch
# (Compute cost: 20-50 GPU hours)
```

---

## ✓ Checkpoint Inventory

### Current Status
```
Training Date: April 24, 2026
Session: model-training (tmux)

Model                  | Status      | Checkpoints | Backed Up
---------------------- | ----------- | ----------- | ---------
segformer_upernet      | Training    | 4 files     | YES (361 MB)
resnet101              | Pending     | 0 files     | -
swin_b                 | Pending     | 0 files     | -
convnext_b             | Pending     | 0 files     | -
```

### Backup Locations
- **Primary:** `outputs/checkpoints_backup/` (Read-only)
- **Secondary:** Recommend off-site backup (AWS S3, Google Drive, etc.)
- **Tertiary:** GitHub LFS (if repo uses it)

---

## ✓ Automation: Scheduled Backups

Add to crontab (backup every 6 hours):
```bash
0 */6 * * * cd /home/tamoghno/rohit-encoders && bash backup_checkpoints.sh >> backup_log.txt 2>&1
```

---

## ✓ Commands Reference

```bash
# Backup all current checkpoints
bash backup_checkpoints.sh

# List all backups
bash backup_checkpoints.sh --list

# Verify checkpoint integrity
bash backup_checkpoints.sh --verify

# Make backups read-only (default)
bash backup_checkpoints.sh --protect

# Unlock backups for management
bash backup_checkpoints.sh --unprotect

# Check training status
tmux capture-pane -t model-training -p | tail -50

# View metrics from last epoch
cat outputs/MODEL/results.json | python -m json.tool | tail -50
```

---

## ✓ Long-Term Strategy

1. **After each model training completes:**
   - Run `bash backup_checkpoints.sh`
   - Verify backup with `--verify`
   - Commit to git: `git add -A && git commit -m "Checkpoint backup: MODEL complete"`

2. **Weekly backup:**
   - Archive outputs/ to external drive
   - Verify checksums match

3. **End of project:**
   - Archive all checkpoints and results
   - Document best performing models
   - Create final summary report

---

## ✓ Questions?

- **How big are the backups?** ~360 MB per model (full state)
- **How long to resume training?** ~1-2 minutes to load checkpoint
- **Can I delete old epoch checkpoints?** Yes, but keep checkpoint_best.pth
- **What if disk is full?** Archive old backups to external storage
- **How to share checkpoints?** Use git LFS or cloud storage (S3, etc.)

---

**Last Updated:** April 24, 2026  
**Policy Status:** ACTIVE  
**Backup Status:** Verified ✓
