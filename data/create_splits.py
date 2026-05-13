"""
Create stratified train/test/val splits from the 5% dataset.
"""
import json
import os
import random
from collections import defaultdict, Counter
import numpy as np
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)

# Load the 5% subset
subset_dir = "/home/tamoghno/datasets/nuimages/v1.0-train-5pct"
output_base = "/home/tamoghno/datasets/nuimages"

print("Loading 5% subset...")
with open(os.path.join(subset_dir, "sample.json")) as f:
    samples = json.load(f)
with open(os.path.join(subset_dir, "sample_data.json")) as f:
    sample_data = json.load(f)
with open(os.path.join(subset_dir, "object_ann.json")) as f:
    annotations = json.load(f)
with open(os.path.join(subset_dir, "ego_pose.json")) as f:
    ego_poses = json.load(f)
with open(os.path.join(subset_dir, "calibrated_sensor.json")) as f:
    calibrated_sensors = json.load(f)
with open(os.path.join(subset_dir, "log.json")) as f:
    logs = json.load(f)
with open(os.path.join(subset_dir, "sensor.json")) as f:
    sensors = json.load(f)
with open(os.path.join(subset_dir, "attribute.json")) as f:
    attributes = json.load(f)
with open(os.path.join(subset_dir, "category.json")) as f:
    categories = json.load(f)
with open(os.path.join(subset_dir, "surface_ann.json")) as f:
    surface_anns = json.load(f)

print(f"Total samples: {len(samples)}")

# Build mapping from sample_token to annotations
sample_to_annotations = defaultdict(list)
for ann in annotations:
    sample_data_token = ann.get('sample_data_token')
    sample_to_annotations[sample_data_token].append(ann)

# Build category token to name mapping
cat_token_to_name = {cat['token']: cat['name'] for cat in categories}

# Create stratification feature: most frequent category in each sample
print("Creating stratification feature...")
sample_features = []
for sample in samples:
    key_camera_token = sample.get('key_camera_token')
    anns = sample_to_annotations.get(key_camera_token, [])
    
    if anns:
        cat_tokens = [ann.get('category_token') for ann in anns]
        cat_counts = Counter(cat_tokens)
        primary_cat = cat_counts.most_common(1)[0][0] if cat_counts else None
    else:
        primary_cat = None
    
    sample_features.append({
        'token': sample['token'],
        'primary_category': primary_cat,
        'num_annotations': len(anns)
    })

# Create stratification groups
print("Creating strata...")
strata = defaultdict(list)
for feature in sample_features:
    primary_cat = feature['primary_category']
    num_anns = feature['num_annotations']
    ann_stratum = 'no_obj' if num_anns == 0 else ('few' if num_anns <= 5 else ('medium' if num_anns <= 20 else 'many'))
    stratum_key = (primary_cat, ann_stratum)
    strata[stratum_key].append(feature)

print(f"Created {len(strata)} strata")

# Stratified split: 70% train, 15% val, 15% test
train_tokens = set()
val_tokens = set()
test_tokens = set()

for stratum_key, members in strata.items():
    stratum_size = len(members)
    train_size = max(1, int(stratum_size * 0.70))
    val_size = max(1, int(stratum_size * 0.15))
    test_size = max(1, int(stratum_size * 0.15))
    
    random.shuffle(members)
    
    for member in members[:train_size]:
        train_tokens.add(member['token'])
    for member in members[train_size:train_size+val_size]:
        val_tokens.add(member['token'])
    for member in members[train_size+val_size:train_size+val_size+test_size]:
        test_tokens.add(member['token'])

print(f"\nSplit sizes:")
print(f"  Train: {len(train_tokens)} samples")
print(f"  Val:   {len(val_tokens)} samples")
print(f"  Test:  {len(test_tokens)} samples")
print(f"  Total: {len(train_tokens) + len(val_tokens) + len(test_tokens)}")

# Helper function to filter data
def filter_by_samples(data, sample_tokens, key='token'):
    return [item for item in data if item.get(key) in sample_tokens]

# Create sample_token to sample mapping
sample_token_map = {s['token']: s for s in samples}

# Helper to create dataset
def create_split(split_tokens, split_name):
    output_dir = os.path.join(output_base, f"v1.0-train-5pct-{split_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get samples for this split
    split_samples = [s for s in samples if s['token'] in split_tokens]
    split_sample_tokens = {s['token'] for s in split_samples}
    
    # Get sample_data tokens for this split
    split_sample_data_tokens = set()
    for s in split_samples:
        key_camera_token = s.get('key_camera_token')
        if key_camera_token:
            split_sample_data_tokens.add(key_camera_token)
    
    split_sample_data = [s for s in sample_data if s.get('token') in split_sample_data_tokens]
    split_annotations = [a for a in annotations if a.get('sample_data_token') in split_sample_data_tokens]
    
    # Get related tokens
    ego_pose_tokens = {s.get('ego_pose_token') for s in split_sample_data if s.get('ego_pose_token')}
    calibrated_sensor_tokens = {s.get('calibrated_sensor_token') for s in split_sample_data if s.get('calibrated_sensor_token')}
    log_tokens = {s.get('log_token') for s in split_samples}
    
    split_ego_poses = [e for e in ego_poses if e.get('token') in ego_pose_tokens]
    split_calibrated_sensors = [c for c in calibrated_sensors if c.get('token') in calibrated_sensor_tokens]
    split_logs = [l for l in logs if l.get('token') in log_tokens]
    split_surface_anns = [s for s in surface_anns if s.get('sample_data_token') in split_sample_data_tokens]
    
    # Write split
    print(f"\nWriting {split_name} split to {output_dir}...")
    with open(os.path.join(output_dir, "sample.json"), 'w') as f:
        json.dump(split_samples, f)
    with open(os.path.join(output_dir, "sample_data.json"), 'w') as f:
        json.dump(split_sample_data, f)
    with open(os.path.join(output_dir, "object_ann.json"), 'w') as f:
        json.dump(split_annotations, f)
    with open(os.path.join(output_dir, "ego_pose.json"), 'w') as f:
        json.dump(split_ego_poses, f)
    with open(os.path.join(output_dir, "calibrated_sensor.json"), 'w') as f:
        json.dump(split_calibrated_sensors, f)
    with open(os.path.join(output_dir, "log.json"), 'w') as f:
        json.dump(split_logs, f)
    with open(os.path.join(output_dir, "sensor.json"), 'w') as f:
        json.dump(sensors, f)
    with open(os.path.join(output_dir, "surface_ann.json"), 'w') as f:
        json.dump(split_surface_anns, f)
    with open(os.path.join(output_dir, "category.json"), 'w') as f:
        json.dump(categories, f)
    with open(os.path.join(output_dir, "attribute.json"), 'w') as f:
        json.dump(attributes, f)
    
    # Print class distribution
    split_cat_dist = Counter([ann.get('category_token') for ann in split_annotations])
    orig_cat_dist = Counter([ann.get('category_token') for ann in annotations])
    
    print(f"  Samples: {len(split_samples)} | Annotations: {len(split_annotations)}")
    print(f"  Class distribution (top 5):")
    for cat_token, count in split_cat_dist.most_common(5):
        cat_name = cat_token_to_name.get(cat_token, 'unknown')
        ratio = (count / orig_cat_dist.get(cat_token, 1) * 100)
        print(f"    {cat_name:<30} {count:>6} ({ratio:.2f}%)")

# Create all three splits
create_split(train_tokens, "train")
create_split(val_tokens, "val")
create_split(test_tokens, "test")

print("\n✓ All splits created successfully!")
