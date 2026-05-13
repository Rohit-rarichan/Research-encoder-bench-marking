"""
metrics.py — Segmentation evaluation metrics.

Provides:
    - per-class IoU
    - mean IoU (mIoU)
    - confusion matrix accumulation
"""

import numpy as np
import torch


class SegmentationMetrics:
    """
    Accumulates predictions across batches and computes classwise IoU.

    Usage:
        metrics = SegmentationMetrics(num_classes=12)
        for preds, targets in loader:
            metrics.update(preds, targets)
        results = metrics.compute()
        print(results["miou"])
        print(results["class_iou"])
    """

    def __init__(self, num_classes, ignore_index=255, class_names=None):
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.class_names   = class_names or [str(i) for i in range(num_classes)]
        self.confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_mat[:] = 0

    def update(self, preds, targets):
        """
        Args:
            preds:   LongTensor [B, H, W] — predicted class indices
            targets: LongTensor [B, H, W] — ground truth class indices
        """
        if isinstance(preds, torch.Tensor):
            preds   = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        preds   = preds.flatten()
        targets = targets.flatten()

        # Remove ignored pixels
        valid   = targets != self.ignore_index
        preds   = preds[valid]
        targets = targets[valid]

        # Accumulate confusion matrix
        np.add.at(self.confusion_mat,
                  (targets.astype(np.int64), preds.astype(np.int64)),
                  1)

    def compute(self):
        """
        Returns dict with:
            miou       : float — mean IoU across classes present in GT
            class_iou  : dict  — {class_name: iou} for each class
            class_acc  : dict  — {class_name: accuracy}
            pixel_acc  : float — overall pixel accuracy
        """
        cm = self.confusion_mat.astype(np.float64)

        # Per-class IoU = TP / (TP + FP + FN)
        tp    = np.diag(cm)
        fp    = cm.sum(axis=0) - tp   # predicted as class but not GT
        fn    = cm.sum(axis=1) - tp   # GT class but not predicted

        iou_per_class = np.zeros(self.num_classes)
        acc_per_class = np.zeros(self.num_classes)

        for i in range(self.num_classes):
            denom = tp[i] + fp[i] + fn[i]
            if denom > 0:
                iou_per_class[i] = tp[i] / denom
            gt_count = cm[i].sum()
            if gt_count > 0:
                acc_per_class[i] = tp[i] / gt_count

        # mIoU only over classes that appear in GT, excluding surface/background classes
        present = cm.sum(axis=1) > 0
        # Exclude classes without meaningful training: driveable_surface, other_flat, terrain, manmade, vegetation
        exclude_classes = {"driveable_surface", "other_flat", "terrain", "manmade", "vegetation"}
        exclude_mask = np.ones(self.num_classes, dtype=bool)
        for i, name in enumerate(self.class_names):
            if name in exclude_classes:
                exclude_mask[i] = False
        
        present_and_not_excluded = present & exclude_mask
        miou    = iou_per_class[present_and_not_excluded].mean() if present_and_not_excluded.any() else 0.0

        # Pixel accuracy
        pixel_acc = tp.sum() / cm.sum() if cm.sum() > 0 else 0.0

        return {
            "miou":       float(miou),
            "pixel_acc":  float(pixel_acc),
            "class_iou":  {self.class_names[i]: float(iou_per_class[i])
                           for i in range(self.num_classes)},
            "class_acc":  {self.class_names[i]: float(acc_per_class[i])
                           for i in range(self.num_classes)},
        }

    def print_table(self, results=None):
        """Print a formatted per-class IoU table."""
        if results is None:
            results = self.compute()
        print(f"\n{'Class':<30} {'IoU':>8} {'Acc':>8}")
        print("-" * 50)
        for cls in self.class_names:
            iou = results["class_iou"][cls]
            acc = results["class_acc"][cls]
            print(f"{cls:<30} {iou:>8.4f} {acc:>8.4f}")
        print("-" * 50)
        print(f"{'mIoU':<30} {results['miou']:>8.4f}")
        print(f"{'Pixel Acc':<30} {results['pixel_acc']:>8.4f}")
