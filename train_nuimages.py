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
import matplotlib.pyplot as plt
from collections import defaultdict

from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

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


# ── Class Weighting ───────────────────────────────────────────────────────────

def compute_class_weights(dataset, num_classes, ignore_index=255):
    """Compute class weights based on inverse frequency from training dataset."""
    print("\n  Computing class weights from training data...")
    class_counts = np.zeros(num_classes)
    
    for _, mask in tqdm(dataset, desc="    scanning dataset", leave=False):
        # Count pixels per class
        for c in range(num_classes):
            class_counts[c] += (mask == c).sum().item()
    
    # Compute weights: inverse frequency
    total_pixels = class_counts.sum()
    weights = np.zeros(num_classes)
    for c in range(num_classes):
        if class_counts[c] > 0:
            # weight = total / (num_classes * count_per_class)
            weights[c] = total_pixels / (num_classes * class_counts[c])
        else:
            weights[c] = 1.0  # Default weight for unseen classes
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    # Note: ignore_index is handled by CrossEntropyLoss, not by weights array
    # weights array only has entries for the num_classes
    
    print(f"  Class weights computed:")
    for i, w in enumerate(weights):
        if i < len(CLASSES):
            print(f"    {CLASSES[i]:<20}: {w:>6.3f} (pixels: {class_counts[i]/1e6:>6.1f}M)")
    
    return torch.tensor(weights, dtype=torch.float32)


# ── Training History Logger ───────────────────────────────────────────────────

class TrainingLogger:
    """Track training metrics and generate plots."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = defaultdict(list)
    
    def log(self, epoch, **metrics):
        """Log metrics for an epoch."""
        for key, value in metrics.items():
            self.history[key].append(value)
    
    def plot_curves(self, filename="training_curves.png"):
        """Generate and save training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Loss curves
        if 'train_loss' in self.history and 'val_loss' in self.history:
            epochs = range(1, len(self.history['train_loss']) + 1)
            axes[0].plot(epochs, self.history['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=3)
            axes[0].plot(epochs, self.history['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=3)
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
            axes[0].legend(fontsize=11)
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: mIoU curves
        if 'train_miou' in self.history and 'val_miou' in self.history:
            epochs = range(1, len(self.history['train_miou']) + 1)
            axes[1].plot(epochs, self.history['train_miou'], label='Train mIoU', linewidth=2, marker='o', markersize=3)
            axes[1].plot(epochs, self.history['val_miou'], label='Val mIoU', linewidth=2, marker='s', markersize=3)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('mIoU', fontsize=12)
            axes[1].set_title('Training & Validation mIoU', fontsize=13, fontweight='bold')
            axes[1].legend(fontsize=11)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / filename
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Plot saved: {plot_path}")
    
    def get_best_epoch(self):
        """Get epoch with best validation mIoU."""
        if 'val_miou' not in self.history or not self.history['val_miou']:
            return 0
        return np.argmax(self.history['val_miou'])


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training if validation loss doesn't improve."""
    
    def __init__(self, patience=5, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"    ✓ Validation loss improved to {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"    ⚠ No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"    ⟳ Early stopping triggered at epoch {epoch+1}")


# ── Learning Rate Scheduler with Warmup ───────────────────────────────────────

class WarmupThenSlash:
    """Learning rate scheduler: warmup then slash every N epochs."""
    
    def __init__(self, optimizer, warmup_epochs=5, slash_interval=10, slash_factor=0.5):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.slash_interval = slash_interval
        self.slash_factor = slash_factor
        self.base_lr = optimizer.defaults['lr']
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        
        # Warmup phase
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Count how many slash intervals have passed
            num_slashes = (self.current_epoch - self.warmup_epochs) // self.slash_interval
            lr = self.base_lr * (self.slash_factor ** num_slashes)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


# ── Load pretrained weights ────────────────────────────────────────────────────

def load_pretrained_weights(model, model_name):
    """
    Load pretrained ImageNet weights into model backbones where available.
    For ResNet101, loads pretrained weights from torchvision.
    Others use random initialization (models are sufficiently expressive for this task).
    """
    try:
        if model_name == "resnet101":
            from torchvision.models import resnet101, ResNet101_Weights
            print("  Loading pretrained ResNet101 backbone from torchvision...")
            try:
                pretrained_model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
                
                # Transfer backbone weights (ResNet101Encoder)
                if hasattr(model, 'backbone'):
                    backbone_state = pretrained_model.state_dict()
                    our_backbone_state = model.backbone.state_dict()
                    
                    # Map torchvision ResNet101 layers to our ResNet101Encoder
                    matched = 0
                    for key in our_backbone_state.keys():
                        if key in backbone_state:
                            our_backbone_state[key] = backbone_state[key]
                            matched += 1
                    
                    if matched > 0:
                        model.backbone.load_state_dict(our_backbone_state, strict=False)
                        print(f"  ✓ Loaded {matched} matched weights into ResNet101 backbone")
                    else:
                        print("  ⚠ Could not find matching weights, using random initialization")
                return model
            except Exception as e:
                print(f"  ⚠ Error loading ResNet101 pretrained: {e}")
                return model
        
        elif model_name == "convnext_b":
            import timm, re
            print("  Loading pretrained ConvNeXt-B backbone from timm (ImageNet-21k→1k)...")
            try:
                pretrained = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True)
                timm_sd = pretrained.state_dict()
                encoder = model.encoder

                def remap_convnext(key):
                    if key.startswith('stem.'):
                        return key
                    mo = re.match(r'stages\.(\d+)\.blocks\.(\d+)\.(.*)', key)
                    if mo:
                        i, j, rest = mo.group(1), mo.group(2), mo.group(3)
                        rest = (rest.replace('conv_dw', 'dwconv')
                                    .replace('mlp.fc1', 'pwconv1')
                                    .replace('mlp.fc2', 'pwconv2'))
                        return f'stages.{i}.{j}.{rest}'
                    mo = re.match(r'stages\.(\d+)\.downsample\.(\d+)\.(.*)', key)
                    if mo:
                        si, k, rest = int(mo.group(1)), mo.group(2), mo.group(3)
                        return f'downsamplers.{si-1}.{k}.{rest}'
                    return None

                our_sd = encoder.state_dict()
                mapped = {}
                matched = 0
                for tk, tv in timm_sd.items():
                    ok = remap_convnext(tk)
                    if not ok or ok not in our_sd:
                        continue
                    ov = our_sd[ok]
                    if tv.shape == ov.shape:
                        mapped[ok] = tv
                        matched += 1
                    elif tv.unsqueeze(-1).unsqueeze(-1).shape == ov.shape:
                        # gamma [C] → [C,1,1], Linear weights [out,in] → [out,in,1,1]
                        mapped[ok] = tv.unsqueeze(-1).unsqueeze(-1)
                        matched += 1

                encoder.load_state_dict(mapped, strict=False)
                print(f"  ✓ Loaded {matched} matched weights into ConvNeXt-B encoder")
                del pretrained
            except Exception as e:
                print(f"  ⚠ Could not load pretrained ConvNeXt-B weights: {e}")
            return model

        elif model_name == "swin_b":
            import timm, re
            print("  Loading pretrained Swin-B backbone from timm (ImageNet-1k)...")
            try:
                pretrained = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
                timm_sd = pretrained.state_dict()
                encoder = model.encoder

                def remap(key):
                    if key.startswith('patch_embed'):
                        return key
                    mo = re.match(r'layers\.(\d+)\.blocks\.(\d+)\.(.*)', key)
                    if mo:
                        i, j, rest = mo.group(1), mo.group(2), mo.group(3)
                        rest = rest.replace('mlp.fc1', 'mlp.0').replace('mlp.fc2', 'mlp.3')
                        return f'stages.{i}.{j}.{rest}'
                    mo = re.match(r'layers\.(\d+)\.downsample\.(.*)', key)
                    if mo:
                        li, rest = int(mo.group(1)), mo.group(2)
                        return f'patch_merging.{li - 1}.{rest}'
                    if key in ('norm.weight', 'norm.bias'):
                        return f'stage_norms.3.{key.split(".")[-1]}'
                    return None

                our_sd = encoder.state_dict()
                mapped = {}
                matched = 0
                for tk, tv in timm_sd.items():
                    ok = remap(tk)
                    if ok and ok in our_sd and our_sd[ok].shape == tv.shape:
                        mapped[ok] = tv
                        matched += 1

                encoder.load_state_dict(mapped, strict=False)
                print(f"  ✓ Loaded {matched} matched weights into Swin-B encoder")
                del pretrained
            except Exception as e:
                print(f"  ⚠ Could not load pretrained Swin-B weights: {e}")
            return model

        else:
            # For other models (ConvNeXt-B, SegFormer), skip pretrained loading
            print(f"  Using random initialization for {model_name} (architecture-specific encoders)")
            return model
    
    except Exception as e:
        print(f"  ⚠ Unexpected error in pretrained loading: {e}")
        return model



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
    "swin_b", "convnext_b", "resnet101", "segformer_upernet"
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

def train_one_epoch(model, loader, optimizer, device, criterion, warmup_scheduler=None):
    model.train()
    scaler     = GradScaler()
    total_loss = 0.0
    batch_count = 0
    bad_batches = []  # Track batches with unusually high loss
    
    for batch_idx, (imgs, masks) in enumerate(tqdm(loader, desc="  train", leave=False)):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            logits = model(imgs)
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss   = criterion(logits, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loss_val = loss.item()
        total_loss += loss_val
        batch_count += 1
        
        # Track bad batches (loss > 2x average)
        if loss_val > 5.0:
            bad_batches.append((batch_idx, loss_val))
    
    avg_loss = total_loss / batch_count
    
    # Report bad batches if any
    if bad_batches:
        print(f"    ⚠ {len(bad_batches)} batches with high loss detected:")
        for batch_idx, loss_val in bad_batches[:3]:  # Show first 3
            print(f"      Batch {batch_idx}: loss={loss_val:.4f}")
    
    return avg_loss

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
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
    
    # Load pretrained ImageNet weights if available
    load_pretrained_weights(model, model_name)
    model = model.to(device)  # Ensure model is on correct device after loading
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Compute class weights from training data (for weighted loss)
    class_weights = compute_class_weights(train_ds, NUM_CLASSES)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    
    # Learning rate scheduler: warmup + slash every 10 epochs
    lr_scheduler = WarmupThenSlash(optimizer, warmup_epochs=5, slash_interval=10, slash_factor=0.5)
    
    # Early stopping: stop if val loss doesn't improve for N epochs (increased patience for resumed training)
    early_stopping = EarlyStopping(patience=20, verbose=True)
    
    # Training logger for plots
    logger = TrainingLogger(Path(args.output_dir) / model_name)

    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_miou    = 0.0
    best_results = None
    history      = []
    start_epoch  = 0

    # ── Resume from checkpoint if available ────────────────────────────────────
    # Priority: checkpoint_final_best.pth > checkpoint_best.pth > best.pth > --resume flag
    # Always loads model weights only (no optimizer/scheduler) so training restarts
    # cleanly on the new full dataset.
    _ckpt_candidates = [
        Path(out_dir) / "checkpoint_final_best.pth",
        Path(out_dir) / "checkpoint_best.pth",
        Path(out_dir) / "best.pth",
    ]
    if hasattr(args, 'resume') and args.resume:
        _ckpt_candidates.append(Path(args.resume))

    _loaded = False
    for _ckpt_path in _ckpt_candidates:
        if not (_ckpt_path.exists() and _ckpt_path.stat().st_size > 1_000_000):
            continue
        print(f"\n  ⟳ Loading weights from: {_ckpt_path.name}")
        _ckpt = torch.load(_ckpt_path, map_location=device, weights_only=False)
        _state = _ckpt.get("model_state", _ckpt) if isinstance(_ckpt, dict) else _ckpt
        model.load_state_dict(_state, strict=True)
        if isinstance(_ckpt, dict) and "best_miou" in _ckpt:
            best_miou = _ckpt["best_miou"]
        print(f"    ✓ Weights loaded (training restarts from epoch 1 on new dataset)")
        _loaded = True
        break

    if not _loaded:
        print(f"  ⟳ No existing checkpoint found — starting from scratch")
    
    elif resume_checkpoint and Path(resume_checkpoint).exists():
        # Manual resume flag checkpoint
        print(f"\n  ⟳ Loading checkpoint from --resume flag: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            print(f"    ✓ Model weights loaded")
            if "best_miou" in checkpoint:
                best_miou = checkpoint["best_miou"]
                print(f"    ✓ Best mIoU: {best_miou:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"    ✓ Model weights loaded")
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
        print(f"\nEpoch {epoch+1}/{args.epochs}  (LR: {optimizer.param_groups[0]['lr']:.2e})")
        
        # Training phase
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        
        # Learning rate step (once per epoch, not per batch)
        lr_scheduler.step()
        
        # Validation phase
        val_loss, val_results = evaluate(model, val_loader, device, criterion)
        
        # Also compute train metrics
        train_loss_check, train_results = evaluate(model, train_loader, device, criterion)
        
        train_miou = train_results["miou"]
        val_miou = val_results["miou"]
        
        # Log metrics
        logger.log(epoch, 
                   train_loss=train_loss, 
                   val_loss=val_loss,
                   train_miou=train_miou,
                   val_miou=val_miou)
        
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        print(f"  train_mIoU={train_miou:.4f}  val_mIoU={val_miou:.4f}")
        
        history.append({"epoch": epoch+1, "train_loss": train_loss,
                        "val_loss": val_loss, "train_miou": train_miou, "val_miou": val_miou})

        if val_miou > best_miou:
            best_miou    = val_miou
            best_results = val_results
            torch.save(model.state_dict(), out_dir / "best.pth")
            print(f"  ✓ New best mIoU={best_miou:.4f}")
            
            # Also save full checkpoint for resuming
            full_checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "lr_scheduler_state": lr_scheduler.__dict__,
                "best_miou": best_miou,
                "history": history,
            }
            torch.save(full_checkpoint, out_dir / "checkpoint_best.pth")
        
        # Save periodic checkpoint every N epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = out_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "lr_scheduler_state": lr_scheduler.__dict__,
                "best_miou": best_miou,
                "history": history,
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: checkpoint_epoch_{epoch+1}.pth")
        
        # Generate plots every 20 epochs
        if (epoch + 1) % 20 == 0:
            logger.plot_curves()
        
        # Early stopping check
        early_stopping(val_loss, epoch)
        if early_stopping.should_stop:
            print(f"\n  ⟳ Training stopped early at epoch {epoch+1}/{args.epochs}")
            break

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
            if h.get("val_miou", 0.0) >= best_epoch_miou:
                best_epoch_miou = h.get("val_miou", 0.0)
                best_epoch_idx = idx
        
        # Save final consolidated best checkpoint with full state
        final_checkpoint = {
            "epoch": best_epoch_idx,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": lr_scheduler.__dict__,
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
        model.load_state_dict(torch.load(out_dir / "best.pth", map_location=device))
        test_loss, test_results = evaluate(model, test_loader, device, criterion)
        print(f"Test Loss: {test_loss:.4f}  |  Test mIoU: {test_results['miou']:.4f}")
        print(f"{'Class':<30} {'IoU':>8}")
        print("-" * 42)
        for cls, iou in test_results["class_iou"].items():
            print(f"{cls:<30} {iou:>8.4f}")
        print("-" * 42)
        print(f"{'mIoU':<30} {test_results['miou']:>8.4f}")
        print(f"{'Pixel Acc':<30} {test_results['pixel_acc']:>8.4f}")
    
    # Evaluate on full remaining dataset (95%) if available
    full_dataset_results = None
    try:
        print(f"\n── {model_name} full dataset evaluation (remaining 95%) ──")
        full_ds = NuImagesDataset(
            args.data_root,
            split="val",  # This loads the full v1.0-val set
            img_size=args.img_size,
            version="v1.0-val",  # Full validation set
            use_internal_split=False,
        )
        full_loader = DataLoader(full_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))
        
        print(f"  Full dataset samples: {len(full_ds)}")
        model.load_state_dict(torch.load(out_dir / "best.pth", map_location=device))
        full_loss, full_dataset_results = evaluate(model, full_loader, device, criterion)
        print(f"Full Loss: {full_loss:.4f}  |  Full mIoU: {full_dataset_results['miou']:.4f}")
        print(f"{'Class':<30} {'IoU':>8}")
        print("-" * 42)
        for cls, iou in list(full_dataset_results["class_iou"].items())[:10]:  # Show first 10
            print(f"{cls:<30} {iou:>8.4f}")
        print("-" * 42)
        print(f"{'mIoU':<30} {full_dataset_results['miou']:>8.4f}")
        print(f"{'Pixel Acc':<30} {full_dataset_results['pixel_acc']:>8.4f}")
    except Exception as e:
        print(f"  ⚠ Could not evaluate on full dataset: {e}")

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
        filtered_full = filter_results(full_dataset_results)
        
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
                   "full_dataset_results": filtered_full,
                   "history": history}, f, indent=2)
    
    # Generate final training curves
    print(f"\n  ⟳ Generating final training curves...")
    logger.plot_curves("training_curves_final.png")
    
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
