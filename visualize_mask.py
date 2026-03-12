import cv2
import numpy as np
from nuimages_dataset import NuImagesMiniDataset

ds = NuImagesMiniDataset("~/dev/research/datasets/nuImages")
sample = ds[0]

img = cv2.imread(sample["img_path"])
mask = sample["mask"]

color_mask = np.zeros_like(img)
colors = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
    6: (128, 128, 255),
}

for k, color in colors.items():
    color_mask[mask == k] = color

overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)

cv2.imwrite("debug_overlay.jpg", overlay)
cv2.imwrite("debug_mask.png", mask)
print("saved debug_overlay.jpg and debug_mask.png")
print("unique:", np.unique(mask))