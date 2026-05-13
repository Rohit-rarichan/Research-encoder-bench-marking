import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

pred = torch.randint(0, 5, (2, 256, 256), device=device)
target = torch.randint(0, 5, (2, 256, 256), device=device)

print(pred.device, target.device)
