import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor

from segformer import SegformerClasswise
from load_pretrained import load_pretrained_hf
from nuimages_dataset import NuImagesMiniDataset, CLASS_NAMES
from metrics import update_confusion_matrix, compute_iou

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATAROOT = os.path.expanduser("~/dev/research/datasets/nuImages")
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255

HF_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
processor = SegformerImageProcessor.from_pretrained(HF_ID)

def load_model():
    model = SegformerClasswise(num_classes=7)
    model.load_state_dict(torch.load("segformer_nuimages_7cls.pth", map_location=DEVICE))
    model = model.to(DEVICE).eval()
    return model

def predict_mask(model, img_path, out_h, out_w):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    x = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():
        logits = model(x)   # [1, 150, h, w]

        logits = F.interpolate(
            logits,
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False
        )

        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    return pred

def main():
    dataset = NuImagesMiniDataset(DATAROOT)
    model = load_model()

    confmat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for i in range(len(dataset)):
        sample = dataset[i]
        gt_mask = sample["mask"]
        pred_mask = predict_mask(model, sample["img_path"], gt_mask.shape[0], gt_mask.shape[1])

        # TEMP: skip if predicted labels exceed nuImages class range
        pred_mask = np.where(pred_mask < NUM_CLASSES, pred_mask, IGNORE_INDEX)

        update_confusion_matrix(confmat, pred_mask, gt_mask, NUM_CLASSES, IGNORE_INDEX)
        print(f"[{i+1}/{len(dataset)}] done")

    iou, miou = compute_iou(confmat)

    # count GT pixels per class
    gt_pixels = confmat.sum(axis=1)

    # only evaluate classes that appear in GT
    valid = gt_pixels > 0
    miou = iou[valid].mean()

    print("\nPer-class IoU:")
    for name, score in zip(CLASS_NAMES, iou):
        print(f"{name:20s}: {score:.4f}")

    print(f"\nmIoU (valid classes only): {miou:.4f}")
    
    gt_pixels = confmat.sum(axis=1)
    pred_pixels = confmat.sum(axis=0)

    print("\nGT pixels per class:")
    for name, n in zip(CLASS_NAMES, gt_pixels):
        print(f"{name:20s}: {n}")

    print("\nPred pixels per class:")
    for name, n in zip(CLASS_NAMES, pred_pixels):
        print(f"{name:20s}: {n}")

if __name__ == "__main__":
    main()