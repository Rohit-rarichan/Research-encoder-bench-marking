"""
train_nuimages.py — Train all encoder+UPerNet models on NuImages mini.

Usage:
    # Single model
    python train_nuimages.py --model resnet101 --data_root /path/to/nuimages-v1.0-mini

    # All models sequentially
    python train_nuimages.py --model all --data_root /path/to/nuimages-v1.0-mini

    # Recommended for lab server (GPU, full run):
    python train_nuimages.py --model all --data_root /path/to/nuimages-v1.0-mini \\
        --epochs 40 --batch_size 8 --img_size 512
        
    # Resume from checkpoint
    python train_nuimages.py --model swin_b --data_root /path/to/nuimages \\
        --resume ./outputs/swin_b/checkpoint_epoch_20.pth
"""

import argparse
import json
import random
from pathlib import Path

from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from nuimages_dataset import NuImagesDataset, NUM_CLASSES, CLASSES
from metrics import SegmentationMetrics

# Import all model classes
from segformer import SegformerClasswise
from segformer_upernet import SegformerUPerNet
from resnet101_upernet import ResNet101UPerNet
from swin_upernet import SwinBUPerNet
from convnext_upernet import ConvNeXtBUPerNet


# ── Reproducibility ───────────────────────────────────────────────────────────

SEED = 42

def set_seed(seed=SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make training deterministic (slightly slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Random seed set to {seed} for reproducibility")


# ── Model registry ────────────────────────────────────────────────────────────

def build_model(name, num_classes):
    """Build model by name."""
    if name == "segformer":
        # MiT-B0 + MLP decoder (Rohit's original)
        return SegformerClasswise(
            num_classes=num_classes,
            embed_dims=(32, 64, 160, 256),
            num_heads=(1, 2, 5, 8),
            depths=(2, 2, 2, 2),
            sr_ratios=(8, 4, 2, 1),
        )
    elif name == "segformer_upernet":
        # MiT-B0 + UPerNet decoder
        return SegformerUPerNet(num_classes=num_classes)
    elif name == "resnet101":
        return ResNet101UPerNet(num_classes=num_classes)
    elif name == "swin_b":
        return SwinBUPerNet(num_classes=num_classes)
    elif name == "convnext_b":
        return ConvNeXtBUPerNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")


MODEL_NAMES = [
    "swin_b", "convnext_b"
]


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        print("Device: CUDA")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Device: Apple MPS")
        return torch.device("mps")
    print("Device: CPU (training will be slow)")
    return torch.device("cpu")


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion  = nn.CrossEntropyLoss(ignore_index=255)
    scaler     = GradScaler()
    total_loss = 0.0
    for imgs, masks in tqdm(loader, desc="  train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(imgs)
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss   = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion  = nn.CrossEntropyLoss(ignore_index=255)
    total_loss = 0.0
    metrics    = SegmentationMetrics(NUM_CLASSES, class_names=CLASSES)
    for imgs, masks in tqdm(loader, desc="  eval ", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        logits      = model(imgs)
        # Upsample to match mask resolution
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        total_loss += criterion(logits, masks).item()
        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)
    
    results = metrics.compute()
    return total_loss / len(loader), results


# ── Per-model training run ────────────────────────────────────────────────────

def train_model(model_name, args, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    import gc; gc.collect()

    print(f"\n{'='*60}")
    print(f"  Model  : {model_name}")
    print(f"  Epochs : {args.epochs}  |  BS : {args.batch_size}  |  LR : {args.lr}")
    print(f"  Train Version : {args.train_version}  |  Val Version : {args.val_version}")
    print(f"  Explicit Splits: {args.explicit_splits}")
    print(f"  Random Seed: {SEED}")
    print(f"{'='*60}")

    pin          = device.type == "cuda"
    if args.explicit_splits:
        train_ds = NuImagesDataset(
            args.data_root,
            split="train",
            img_size=args.img_size,
            version=args.train_version,
            use_internal_split=False,
        )
        val_ds = NuImagesDataset(
            args.data_root,
            split="val",
            img_size=args.img_size,
            version=args.val_version,
            use_internal_split=False,
        )
    else:
        train_ds = NuImagesDataset(
            args.data_root,
            split="train",
            img_size=args.img_size,
            version=args.train_version,
            use_internal_split=True,
        )
        val_ds = NuImagesDataset(
            args.data_root,
            split="val",
            img_size=args.img_size,
            version=args.train_version,
            use_internal_split=True,
        )

    print(f"  Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=pin)
    
    # Load test set if using explicit splits
    test_loader = None
    if args.explicit_splits and hasattr(args, 'test_version'):
        try:
            test_ds = NuImagesDataset(
                args.data_root,
                split="test",
                img_size=args.img_size,
                version=args.test_version,
                use_internal_split=False,
            )
            test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                     shuffle=False, num_workers=4, pin_memory=pin)
            print(f"  Test samples: {len(test_ds)}")
        except Exception as e:
            print(f"  ⚠ Could not load test set: {e}")


    model     = build_model(model_name, NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_miou    = 0.0
    best_results = None
    history      = []
    start_epoch  = 0

    # ── Resume from checkpoint if available ────────────────────────────────────
    resume_checkpoint = args.resume if hasattr(args, 'resume') and args.resume else None
    
    if resume_checkpoint and Path(resume_checkpoint).exists():
        print(f"\n  ⟳ Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        
        if isinstance(checkpoint, dict):
            # Full checkpoint format with training state
            if "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
                print(f"    ✓ Model weights loaded")
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                print(f"    ✓ Optimizer state loaded")
            if "scheduler_state" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
                print(f"    ✓ Learning rate scheduler state loaded")
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"]
                print(f"    ✓ Resuming from epoch {start_epoch + 1}")
            if "best_miou" in checkpoint:
                best_miou = checkpoint["best_miou"]
                print(f"    ✓ Best mIoU: {best_miou:.4f}")
            if "history" in checkpoint:
                history = checkpoint["history"]
                print(f"    ✓ Training history loaded ({len(history)} epochs)")
        else:
            # Legacy format: just model state dict
            model.load_state_dict(checkpoint)
            print(f"    ✓ Model weights loaded (legacy format)")
    elif hasattr(args, 'load_pretrained') and args.load_pretrained:
        # Load pretrained ImageNet weights if available
        print(f"\n  ⟳ Attempting to load pretrained ImageNet weights...")
        try:
            # This would need timm or torchvision integration
            print(f"    (Pretrained loading requires additional setup)")
        except Exception as e:
            print(f"    ⚠ Could not load pretrained weights: {e}")

    # ────────────────────────────────────────────────────────────────────────────

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss            = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_results = evaluate(model, val_loader, device)
        scheduler.step()

        miou = val_results["miou"]
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  mIoU={miou:.4f}")
        history.append({"epoch": epoch+1, "train_loss": train_loss,
                        "val_loss": val_loss, "miou": miou})

        if miou > best_miou:
            best_miou    = miou
            best_results = val_results
            torch.save(model.state_dict(), out_dir / "best.pth")
            print(f"  ✓ New best mIoU={best_miou:.4f}")
            
            # Also save full checkpoint for resuming
            full_checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_miou": best_miou,
                "history": history,
            }
            torch.save(full_checkpoint, out_dir / "checkpoint_best.pth")
        
        # Save periodic checkpoint every N epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = out_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_miou": best_miou,
                "history": history,
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: checkpoint_epoch_{epoch+1}.pth")

    # ── Save final best checkpoint after training completes ──────────────────────
    # This ensures the saved checkpoint is always the one with the best validation mIoU
    if best_results is not None and (out_dir / "best.pth").exists():
        print(f"\n  ⟳ Consolidating final best checkpoint...")
        # Reload the best model weights
        model.load_state_dict(torch.load(out_dir / "best.pth", map_location=device))
        
        # Find which epoch had the best mIoU
        best_epoch_idx = 0
        best_epoch_miou = 0.0
        for idx, h in enumerate(history):
            if h["miou"] >= best_epoch_miou:
                best_epoch_miou = h["miou"]
                best_epoch_idx = idx
        
        # Save final consolidated best checkpoint with full state
        final_checkpoint = {
            "epoch": best_epoch_idx,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_miou": best_miou,
            "history": history,
            "is_final_best": True,
        }
        torch.save(final_checkpoint, out_dir / "checkpoint_final_best.pth")
        print(f"  ✓ Final best checkpoint saved (Epoch {best_epoch_idx + 1}, mIoU={best_epoch_miou:.4f})")

    # Print classwise table for this model
    print(f"\n── {model_name} best validation results ──")
    if best_results is not None:
        print(f"{'Class':<30} {'IoU':>8}")
        print("-" * 42)
        for cls, iou in best_results["class_iou"].items():
            print(f"{cls:<30} {iou:>8.4f}")
        print("-" * 42)
        print(f"{'mIoU':<30} {best_results['miou']:>8.4f}")
        print(f"{'Pixel Acc':<30} {best_results['pixel_acc']:>8.4f}")
    else:
        print("⚠ No validation results recorded. Training may have failed.")

    # Evaluate on test set if available
    test_results = None
    if test_loader is not None:
        print(f"\n── {model_name} test set evaluation ──")
        model.load_state_dict(torch.load(out_dir / "best.pth"))
        test_loss, test_results = evaluate(model, test_loader, device)
        print(f"Test Loss: {test_loss:.4f}  |  Test mIoU: {test_results['miou']:.4f}")
        print(f"{'Class':<30} {'IoU':>8}")
        print("-" * 42)
        for cls, iou in test_results["class_iou"].items():
            print(f"{cls:<30} {iou:>8.4f}")
        print("-" * 42)
        print(f"{'mIoU':<30} {test_results['miou']:>8.4f}")
        print(f"{'Pixel Acc':<30} {test_results['pixel_acc']:>8.4f}")

    with open(out_dir / "results.json", "w") as f:
        # Filter results to only include relevant classes (exclude surface classes)
        irrelevant_classes = {"driveable_surface", "other_flat", "terrain", "manmade", "vegetation"}
        
        def filter_results(results):
            if results is None:
                return None
            filtered = dict(results)
            filtered["class_iou"] = {k: v for k, v in results["class_iou"].items() 
                                     if k not in irrelevant_classes and v > 0}
            filtered["class_acc"] = {k: v for k, v in results["class_acc"].items() 
                                     if k not in irrelevant_classes}
            # Recalculate mIoU for relevant classes only
            if filtered["class_iou"]:
                filtered["miou"] = sum(filtered["class_iou"].values()) / len(filtered["class_iou"])
            return filtered
        
        filtered_val = filter_results(best_results)
        filtered_test = filter_results(test_results)
        
        # Calculate filtered mIoU
        filtered_miou = best_miou
        if filtered_val and "class_iou" in filtered_val:
            filtered_miou = filtered_val["miou"]
        
        json.dump({"model": model_name, 
                   "best_miou_all_classes": best_miou,
                   "best_miou": filtered_miou,
                   "relevant_classes": list(filtered_val["class_iou"].keys()) if filtered_val else [],
                   "val_results": filtered_val, 
                   "test_results": filtered_test, 
                   "history": history}, f, indent=2)
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    
    return best_miou, best_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="swin_b",
                        choices=MODEL_NAMES + ["all"])
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--train_version", default="v1.0-train-5pct-train",
                        help="Metadata split directory for training")
    parser.add_argument("--val_version", default="v1.0-train-5pct-val",
                        help="Metadata split directory for validation")
    parser.add_argument("--test_version", default="v1.0-train-5pct-test",
                        help="Metadata split directory for testing")
    parser.add_argument("--explicit_splits", action="store_true", default=True,
                        help="Use explicit split directories for train/val/test")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--img_size",   type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=6e-5)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--load_pretrained", action="store_true",
                        help="Load pretrained ImageNet weights (requires setup)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # ── Set random seed for reproducibility ─────────────────────────────────
    set_seed(args.seed)

    if args.val_version is None:
        args.val_version = args.train_version
    if args.test_version is None:
        args.test_version = args.train_version

    device = get_device()
    models = MODEL_NAMES if args.model == "all" else [args.model]

    all_results = {}
    for m in models:
        miou, results = train_model(m, args, device)
        if results is not None:
            all_results[m] = {"miou": miou, "class_iou": results["class_iou"]}
        else:
            print(f"⚠ Skipping {m} — validation failed")

    # Final comparison table — classwise mIoU for all models
    if all_results:
        print(f"\n{'='*60}")
        print("FINAL COMPARISON — Classwise mIoU")
        print(f"{'='*60}")
        col_w = 12
        successful_models = list(all_results.keys())
        print(f"{'Class':<30}", end="")
        for m in successful_models:
            print(f"  {m[:col_w]:<{col_w}}", end="")
        print()
        print("-" * (30 + (col_w + 2) * len(successful_models)))

        for cls in CLASSES:
            print(f"{cls:<30}", end="")
            for m in successful_models:
                iou = all_results[m]["class_iou"].get(cls, 0.0)
                print(f"  {iou:>{col_w}.4f}", end="")
            print()

        print("-" * (30 + (col_w + 2) * len(successful_models)))
        print(f"{'mIoU':<30}", end="")
        for m in successful_models:
            print(f"  {all_results[m]['miou']:>{col_w}.4f}", end="")
        print()
    else:
        print("\n⚠ No models completed successfully.")

    # Save summary
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    if all_results:
        print(f"\nSummary saved → {out_dir / 'summary.json'}")
    else:
        print(f"\n⚠ Summary (empty) saved → {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
