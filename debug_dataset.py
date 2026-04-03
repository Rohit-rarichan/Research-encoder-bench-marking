import json
import os

data_root = "./nuImages"
version = "v1.0-mini"
meta_dir = os.path.join(data_root, version)

# Load data
with open(os.path.join(meta_dir, "sample_data.json")) as f:
    sample_data = json.load(f)

with open(os.path.join(meta_dir, "object_ann.json")) as f:
    object_ann = json.load(f)

with open(os.path.join(meta_dir, "surface_ann.json")) as f:
    surface_ann = json.load(f)

print(f"Total samples: {len(sample_data)}")
print(f"Total object annotations: {len(object_ann)}")
print(f"Total surface annotations: {len(surface_ann)}")

# Check structure
if sample_data:
    print(f"\nFirst sample keys: {sample_data[0].keys()}")
    print(f"First sample: {sample_data[0]}")

if object_ann:
    print(f"\nFirst object annotation keys: {object_ann[0].keys()}")
    print(f"Has 'sample_data_token': {'sample_data_token' in object_ann[0]}")

if surface_ann:
    print(f"\nFirst surface annotation keys: {surface_ann[0].keys()}")
    print(f"Has 'sample_data_token': {'sample_data_token' in surface_ann[0]}")

# Count annotations per sample
obj_by_sd = {}
for ann in object_ann:
    obj_by_sd.setdefault(ann.get("sample_data_token"), []).append(ann)

surf_by_sd = {}
for ann in surface_ann:
    surf_by_sd.setdefault(ann.get("sample_data_token"), []).append(ann)

# Check key frames with samples/ prefix
keyframe_samples = [
    sd for sd in sample_data
    if sd.get("is_key_frame", False) and sd.get("filename", "").startswith("samples/")
]

print(f"\nKey frame samples with 'samples/' prefix: {len(keyframe_samples)}")

if keyframe_samples:
    sample = keyframe_samples[0]
    token = sample.get("token")
    print(f"First keyframe token: {token}")
    print(f"Annotations for this token - obj: {len(obj_by_sd.get(token, []))}, surf: {len(surf_by_sd.get(token, []))}")

# List all unique tokens in object_ann to see what tokens have annotations
sample_tokens_with_obj = set(ann.get("sample_data_token") for ann in object_ann)
sample_tokens_with_surf = set(ann.get("sample_data_token") for ann in surface_ann)

print(f"\nUnique sample_data_tokens in object_ann: {len(sample_tokens_with_obj)}")
print(f"Unique sample_data_tokens in surface_ann: {len(sample_tokens_with_surf)}")

# Check if keyframe tokens exist in annotations
keyframe_tokens = set(sd.get("token") for sd in keyframe_samples)
overlap_obj = keyframe_tokens & sample_tokens_with_obj
overlap_surf = keyframe_tokens & sample_tokens_with_surf
overlap_either = keyframe_tokens & (sample_tokens_with_obj | sample_tokens_with_surf)

print(f"\nKeyframe tokens in object_ann: {len(overlap_obj)}")
print(f"Keyframe tokens in surface_ann: {len(overlap_surf)}")
print(f"Keyframe tokens in either: {len(overlap_either)}")
