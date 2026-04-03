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

# Get keyframe samples
keyframe_samples = [
    sd for sd in sample_data
    if sd.get("is_key_frame", False) and sd.get("filename", "").startswith("samples/")
]

# Get annotation keys
obj_tokens = set(ann["sample_data_token"] for ann in object_ann)

# Get dataset tokens
dataset_tokens = set(sd["token"] for sd in keyframe_samples)

print(f"Keyframe samples: {len(keyframe_samples)}")
print(f"Unique tokens in sample_data: {len(dataset_tokens)}")
print(f"Unique tokens in object_ann: {len(obj_tokens)}")

# Check intersection
overlap= dataset_tokens & obj_tokens
print(f"Tokens that overlap: {len(overlap)}")

# Print mismatches
in_dataset_not_in_ann = dataset_tokens - obj_tokens
in_ann_not_in_dataset = obj_tokens - dataset_tokens

print(f"\nTokens in dataset but NOT in object_ann: {len(in_dataset_not_in_ann)}")
if in_dataset_not_in_ann:
    print(f"Examples: {list(in_dataset_not_in_ann)[:5]}")

print(f"\nTokens in object_ann but NOT in dataset: {len(in_ann_not_in_dataset)}")
if in_ann_not_in_dataset:
    print(f"Examples: {list(in_ann_not_in_dataset)[:5]}")

# Check what tokens the first few dataset items have
print("\nFirst 5 dataset tokens:")
for i, sd in enumerate(keyframe_samples[:5]):
   print(f"  {i}: {sd['token']} (has obj anns: {sd['token'] in obj_tokens})")
