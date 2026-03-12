import numpy as np

def update_confusion_matrix(confmat, pred, target, num_classes, ignore_index=255):
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    inds = num_classes * target + pred
    bincount = np.bincount(inds, minlength=num_classes * num_classes)
    confmat += bincount.reshape(num_classes, num_classes)

def compute_iou(confmat):
    tp = np.diag(confmat)
    fp = confmat.sum(axis=0) - tp
    fn = confmat.sum(axis=1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    miou = iou.mean()
    return iou, miou