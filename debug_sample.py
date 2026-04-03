import sys
sys.path.insert(0, '/home/tamoghno/image_encoders/bench-marking')

from nuimages_dataset import NuImagesDataset

# Load dataset
print("Loading dataset...")
ds = NuImagesDataset("./nuImages", split="train", img_size=512)

print(f"Dataset size: {len(ds)}")
print(f"Object annotations dict size: {len(ds.obj_by_sd)}")
print(f"Surface annotations dict size: {len(ds.surf_by_sd)}")

# Try to load first sample
print("\nLoading first sample...")
img, mask = ds[0]

print(f"Image shape: {img.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Mask unique values: {mask.unique()}")
print(f"Valid pixels (not 255): {(mask != 255).sum()}")
