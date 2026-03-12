import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import SegformerImageProcessor

from segformer import SegformerClasswise
from load_pretrained import load_pretrained_hf
from nuimages_dataset import NuImagesMiniDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATAROOT = os.path.expanduser("~/dev/research/datasets/nuImages")
HF_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
NUM_CLASSES = 7
IGNORE_INDEX = 255
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4

processor = SegformerImageProcessor.from_pretrained(HF_ID)

class NuImagesTorchDataset(Dataset):
    def __init__(self, dataroot):
        self.base = NuImagesMiniDataset(dataroot)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        image = Image.open(sample["img_path"]).convert("RGB")
        mask = sample["mask"]

        proc = processor(images=image, return_tensors="pt")
        pixel_values = proc["pixel_values"].squeeze(0)   # [3, H, W]

        mask = torch.tensor(mask, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "labels": mask
        }

def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    return {"pixel_values": pixel_values, "labels": labels}

def main():
    dataset = NuImagesTorchDataset(DATAROOT)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = SegformerClasswise(num_classes=7)
    load_pretrained_hf(model, HF_ID)   # classifier stays random because we skipped it
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in loader:
            x = batch["pixel_values"].to(DEVICE)
            y = batch["labels"].to(DEVICE)

            logits = model(x)   # [B, 7, h, w]

            logits = F.interpolate(
                logits,
                size=y.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            loss = F.cross_entropy(logits, y, ignore_index=IGNORE_INDEX)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), "segformer_nuimages_7cls.pth")

if __name__ == "__main__":
    main()