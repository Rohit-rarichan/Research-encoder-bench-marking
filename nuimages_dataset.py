# nuimages_dataset.py

import os
import json
import cv2
import numpy as np
from pycocotools import mask as mask_utils
from nuimages.utils.utils import mask_decode

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
        return mask_decode(ann_mask).astype(bool)

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
        return {
            "img_path": img_path,
            "mask": mask,
            "sample_data_token": sd["token"],
        }