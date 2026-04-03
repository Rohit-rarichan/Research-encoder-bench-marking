# nuimages_dataset.py

import os
import json
import cv2
import numpy as np
from pycocotools import mask as mask_utils
import torch
from torchvision import transforms as T


def mask_decode(rle_mask):
    """Decode RLE encoded mask using pycocotools.
    
    Args:
        rle_mask: Dict with 'size' and 'counts' keys (nuimages RLE format)
    
    Returns:
        Decoded binary mask as numpy array
    """
    if not isinstance(rle_mask, dict) or 'size' not in rle_mask or 'counts' not in rle_mask:
        # Return empty mask if format is invalid
        return np.zeros((rle_mask.get('size', [0, 0])[0], rle_mask.get('size', [0, 0])[1]), dtype=bool)
    
    try:
        # pycocotools.decode expects a list of RLE objects
        decoded = mask_utils.decode([rle_mask])
        return decoded[:, :, 0]  # Return the first (and only) channel
    except Exception as e:
        # If decoding fails, return empty mask with the correct size
        size = rle_mask.get('size', [0, 0])
        return np.zeros((size[0], size[1]), dtype=bool)

IGNORE_INDEX = 255

CLASS_NAMES = [
    "vehicle",
    "pedestrian",
    "driveable_surface",
    "other_flat",
    "terrain",
    "manmade",
    "vegetation",
]
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

def map_category_name(category_name: str):
    name = category_name.lower()

    if any(x in name for x in ["vehicle", "car", "truck", "bus", "trailer", "construction", "bicycle", "motorcycle"]):
        return CLASS_NAME_TO_ID["vehicle"]
    if "pedestrian" in name:
        return CLASS_NAME_TO_ID["pedestrian"]
    if "driveable_surface" in name or "driveable surface" in name:
        return CLASS_NAME_TO_ID["driveable_surface"]
    if "other_flat" in name or "other flat" in name:
        return CLASS_NAME_TO_ID["other_flat"]
    if "terrain" in name:
        return CLASS_NAME_TO_ID["terrain"]
    if "manmade" in name:
        return CLASS_NAME_TO_ID["manmade"]
    if "vegetation" in name:
        return CLASS_NAME_TO_ID["vegetation"]

    return None


class NuImagesMiniDataset:
    def __init__(self, dataroot, version="v1.0-mini"):
        self.dataroot = os.path.expanduser(dataroot)
        self.meta_dir = os.path.join(self.dataroot, version)

        with open(os.path.join(self.meta_dir, "sample_data.json")) as f:
            self.sample_data = json.load(f)

        with open(os.path.join(self.meta_dir, "object_ann.json")) as f:
            self.object_ann = json.load(f)

        with open(os.path.join(self.meta_dir, "surface_ann.json")) as f:
            self.surface_ann = json.load(f)

        with open(os.path.join(self.meta_dir, "category.json")) as f:
            categories = json.load(f)

        self.category_token_to_name = {
            c["token"]: c["name"] for c in categories
        }

        self.obj_by_sd = {}
        for ann in self.object_ann:
            self.obj_by_sd.setdefault(ann["sample_data_token"], []).append(ann)

        self.surf_by_sd = {}
        for ann in self.surface_ann:
            self.surf_by_sd.setdefault(ann["sample_data_token"], []).append(ann)

        self.items = [
            sd for sd in self.sample_data
            if sd.get("is_key_frame", False) and sd.get("filename", "").startswith("samples/")
        ]

    def __len__(self):
        return len(self.items)

    def _decode_rle(self, ann_mask):
        import base64
        
        # Convert base64-encoded counts to bytes
        rle_dict = dict(ann_mask)  # Make a copy
        if isinstance(rle_dict.get('counts'), str):
            # Counts is base64-encoded string, decode it to bytes
            rle_dict['counts'] = base64.b64decode(rle_dict['counts'])
        
        decoded = mask_decode(rle_dict).astype(bool)
        return decoded

    def build_mask(self, sd_record):
        img_path = os.path.join(self.dataroot, sd_record["filename"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)

        h, w = img.shape[:2]
        mask = np.full((h, w), IGNORE_INDEX, dtype=np.uint8)
        sd_token = sd_record["token"]

        for ann in self.surf_by_sd.get(sd_token, []):
            category_name = self.category_token_to_name.get(ann["category_token"], "")
            class_id = map_category_name(category_name)
            if class_id is None:
                continue
            if ann.get("mask") is None:
                continue
            binary_mask = self._decode_rle(ann["mask"])
            mask[binary_mask] = class_id

        for ann in self.obj_by_sd.get(sd_token, []):
            category_name = self.category_token_to_name.get(ann["category_token"], "")
            class_id = map_category_name(category_name)
            if class_id is None:
                continue
            if ann.get("mask") is None:
                continue
            binary_mask = self._decode_rle(ann["mask"])
            mask[binary_mask] = class_id

        return img_path, mask

    def __getitem__(self, idx):
        sd = self.items[idx]
        img_path, mask = self.build_mask(sd)
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize image to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        
        # Convert mask to tensor
        mask_tensor = torch.from_numpy(mask).long()
        
        return img_tensor, mask_tensor


# Aliases and exports for compatibility
NUM_CLASSES = len(CLASS_NAMES)
CLASSES = CLASS_NAMES


class NuImagesDataset(NuImagesMiniDataset):
    """Extended dataset class with split support and img_size parameter."""
    
    def __init__(self, dataroot, split="train", img_size=512, version="v1.0-mini"):
        super().__init__(dataroot, version=version)
        self.split = split
        self.img_size = img_size
        
        # Train/val split (80/20)
        split_idx = int(0.8 * len(self.items))
        if split == "train":
            self.items = self.items[:split_idx]
        elif split == "val":
            self.items = self.items[split_idx:]
        else:
            raise ValueError(f"Unknown split: {split}")