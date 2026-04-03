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
"""

import argparse
import json
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
    # "segformer",  # Commented out - using UPerNet variants only
    "segformer_upernet", "resnet101", "swin_b", "convnext_b"
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
    print(f"{'='*60}")

    pin          = device.type == "cuda"
    train_ds     = NuImagesDataset(args.data_root, split="train", img_size=args.img_size)
    val_ds       = NuImagesDataset(args.data_root, split="val",   img_size=args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=pin)

    model     = build_model(model_name, NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_miou    = 0.0
    best_results = None
    history      = []

    for epoch in range(args.epochs):
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

    # Print classwise table for this model
    print(f"\n── {model_name} best results ──")
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

    with open(out_dir / "results.json", "w") as f:
        json.dump({"model": model_name, "best_miou": best_miou,
                   "results": best_results, "history": history}, f, indent=2)
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    
    return best_miou, best_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="resnet101",
                        choices=MODEL_NAMES + ["all"])
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--img_size",   type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=6e-5)
    args = parser.parse_args()

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
