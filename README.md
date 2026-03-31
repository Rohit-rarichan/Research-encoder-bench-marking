# Quick test locally (CPU/MPS, small run)
python train_nuimages.py --model resnet101 --data_root /Users/aashiryarai/Downloads/nuimages-v1.0-mini --epochs 2 --batch_size 2

# Full run on lab server once GPU is sorted
python train_nuimages.py --model all --data_root /path/to/nuimages-v1.0-mini --epochs 40 --batch_size 8
