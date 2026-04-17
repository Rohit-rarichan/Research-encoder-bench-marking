import torch
import json
from pathlib import Path
from nuimages_dataset import NuImagesDataset, NUM_CLASSES, CLASSES
from metrics import SegmentationMetrics
from train_nuimages import build_model, evaluate, get_device
from torch.utils.data import DataLoader

device = get_device()

# Load best models and evaluate on test set
for model_name in ["segformer_upernet", "resnet101"]:
    print(f"\n{'='*60}")
    print(f"  {model_name.upper()}")
    print(f"{'='*60}")
    
    model = build_model(model_name, NUM_CLASSES).to(device)
    best_pth = Path("outputs") / model_name / "best.pth"
    model.load_state_dict(torch.load(best_pth, map_location=device))
    
    # Load test set
    test_ds = NuImagesDataset(
        "/home/tamoghno/datasets/nuimages",
        split="test",
        img_size=512,
        version="v1.0-train-5pct-test",
        use_internal_split=False,
    )
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    
    # Evaluate
    test_loss, test_results = evaluate(model, test_loader, device)
    
    print(f"\nTest Results (excluding driveable_surface, other_flat, terrain, manmade, vegetation):")
    print(f"  mIoU: {test_results['miou']:.4f}")
    print(f"  Pixel Acc: {test_results['pixel_acc']:.4f}")
    print(f"\n{'Class':<30} {'IoU':>8}")
    print("-" * 42)
    for cls, iou in test_results["class_iou"].items():
        if cls not in {"driveable_surface", "other_flat", "terrain", "manmade", "vegetation"}:
            print(f"{cls:<30} {iou:>8.4f}")
    
    del model
    torch.cuda.empty_cache()
