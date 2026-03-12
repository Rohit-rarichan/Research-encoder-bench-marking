import torch
from segformer import SegformerClasswise
import glob
from PIL import Image
import numpy as np
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig
from load_pretrained import load_pretrained_hf
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── My model (random weights) ──────────────────────────────────────────────
model = SegformerClasswise(num_classes=7)    #change back to 150 after nuscenes is done
load_pretrained_hf(model, "nvidia/segformer-b0-finetuned-ade-512-512")
model = model.to(device).eval()

# Quick shape sanity check
with torch.no_grad():
    _x = torch.randn(1, 3, 512, 512, device=device)
    _logits = model(_x)
    print("Classwise logits shape:", _logits.shape)  # expect [1, 150, 128, 128]

# ── HuggingFace model (random weights, same config as pretrained) ──────────
HF_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
cfg = SegformerConfig.from_pretrained(HF_ID)
hf_pretrained = SegformerForSemanticSegmentation.from_pretrained(HF_ID).to(device).eval()

# ── Load one image and preprocess with HF processor (shared for both) ──────
img_dir = "./v2x-subset/image_1"
img_path = sorted(glob.glob(img_dir + "/*.jpg"))[0]
img = Image.open(img_path).convert("RGB")

processor = SegformerImageProcessor.from_pretrained(HF_ID)
inputs = processor(images=img, return_tensors="pt")
x_shared = inputs["pixel_values"].to(device)  # [1, 3, 512, 512]

# ── Forward pass ───────────────────────────────────────────────────────────
with torch.no_grad():
    logits_classwise = model(x_shared)                          # [1, 150, h, w]
    logits_hf = hf_pretrained(pixel_values=x_shared).logits

# ── Upsample logits FIRST, then argmax (same for both) ────────────────────
logits_classwise_up = F.interpolate(logits_classwise, size=(512, 512), mode="bilinear", align_corners=False)
logits_hf_up        = F.interpolate(logits_hf,        size=(512, 512), mode="bilinear", align_corners=False)

pred_classwise = logits_classwise_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
pred_hf        = logits_hf_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

# ── Stats ──────────────────────────────────────────────────────────────────
def print_stats(name, logits_up, pred):
    lg = logits_up
    mean, std = lg.mean().item(), lg.std().item()
    mn, mx    = lg.min().item(),  lg.max().item()
    top2      = torch.topk(lg, k=2, dim=1).values
    margin    = (top2[:, 0] - top2[:, 1]).mean().item()
    vals, counts = np.unique(pred, return_counts=True)
    top = sorted(zip(counts, vals), reverse=True)[:10]
    print(f"\n=== {name} ===")
    print(f"logits mean/std : {mean:.4f} / {std:.4f}")
    print(f"logits min/max  : {mn:.4f} / {mx:.4f}")
    print(f"avg top1-top2 margin: {margin:.4f}")
    print(f"unique classes  : {len(vals)}")
    print(f"top classes (count, id): {top}")

print_stats("MY model",  logits_classwise_up, pred_classwise)
print_stats("HF random", logits_hf_up,        pred_hf)

# ── Visualisation ──────────────────────────────────────────────────────────
rng     = np.random.default_rng(0)
palette = rng.integers(0, 256, size=(256, 3), dtype=np.uint8)
palette[0] = [0, 0, 0]

orig_np = np.array(img.resize((512, 512)))

def save_color_and_overlay(pred_uint8, prefix):
    color_mask = palette[pred_uint8]
    Image.fromarray(color_mask).save(f"{prefix}_pred_color.png")
    overlay = (orig_np * 0.5 + color_mask * 0.5).astype(np.uint8)
    Image.fromarray(overlay).save(f"{prefix}_overlay.png")

save_color_and_overlay(pred_classwise, "my_shared")
save_color_and_overlay(pred_hf,        "hf_shared")
print("\nSaved my_shared_pred_color.png / my_shared_overlay.png")
print("Saved hf_shared_pred_color.png / hf_shared_overlay.png")