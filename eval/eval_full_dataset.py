"""Run full annotated dataset inference on a saved checkpoint.
Usage: python eval/eval_full_dataset.py --model swin_b
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.nuimages_dataset import NuImagesMiniDataset, NUM_CLASSES
from train_nuimages import evaluate

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODELS = {
    "swin_b":          ("swin_upernet",   "SwinBUPerNet"),
    "segformer_upernet": ("segformer_upernet", "SegformerUPerNet"),
    "resnet101":       ("resnet_upernet", "ResNet101UPerNet"),
    "convnext_b":      ("convnext_upernet", "ConvNeXtBUPerNet"),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True, choices=list(MODELS.keys()))
    parser.add_argument("--data_root",  default="/home/tamoghno/datasets/nuimages")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--train_version", default="v1.0-train-5pct-train",
                        help="Version used for training — those tokens are excluded from eval")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    module_name, class_name = MODELS[args.model]
    import importlib
    mod = importlib.import_module(module_name)
    model = getattr(mod, class_name)(num_classes=NUM_CLASSES).to(device)

    ckpt_path = f"{args.output_dir}/{args.model}/best.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded: {ckpt_path}")

    # Exclude training tokens
    train_tokens = {sd["token"] for sd in NuImagesMiniDataset(args.data_root, version=args.train_version).items}
    print(f"Excluding {len(train_tokens)} training samples")

    ds_tr = NuImagesMiniDataset(args.data_root, version="v1.0-train")
    ds_va = NuImagesMiniDataset(args.data_root, version="v1.0-val")
    ds_tr.items = [sd for sd in ds_tr.items if sd["token"] not in train_tokens]

    class Combined(torch.utils.data.Dataset):
        def __init__(self, a, b): self.a, self.b = a, b
        def __len__(self): return len(self.a) + len(self.b)
        def __getitem__(self, i):
            if i < len(self.a): return self.a[i]
            return self.b[i - len(self.a)]

    full_ds = Combined(ds_tr, ds_va)
    print(f"Total eval samples: {len(full_ds)}  "
          f"(v1.0-train excl. train={len(ds_tr)}, v1.0-val={len(ds_va)})")

    loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=(device.type == "cuda"))

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    loss, results = evaluate(model, loader, device, criterion)

    print(f"\nmIoU  : {results['miou']:.4f}")
    print(f"Loss  : {loss:.4f}")
    print(f"Px Acc: {results['pixel_acc']:.4f}")
    print(f"\n{'Class':<25} {'IoU':>8}")
    print("-" * 36)
    for cls, iou in results["class_iou"].items():
        print(f"{cls:<25} {iou:>8.4f}")

    out = f"{args.output_dir}/{args.model}/full_dataset_results_83k.json"
    with open(out, "w") as f:
        json.dump({"miou": results["miou"], "class_iou": results["class_iou"],
                   "pixel_acc": results["pixel_acc"], "loss": loss,
                   "eval_samples": len(full_ds)}, f, indent=2)
    print(f"\nSaved → {out}")

if __name__ == "__main__":
    main()