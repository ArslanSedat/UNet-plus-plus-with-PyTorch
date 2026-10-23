"""Entry point for UNet++
"""

import torch
from torch import nn, optim
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from unetplusplus import UNetPlusPlus
from dataset import SegmentationDataset

def train_model(dataset, model_save_path, in_channels=3, num_classes=3,
                learning_rate=3e-4, batch_size=8, epochs=2, validation_fraction=0.2
                ):
    val_size = max(1, int(len(dataset) * validation_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetPlusPlus(in_channels=in_channels, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, masks = images.float().to(device), masks.long().to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.float().to(device), masks.long().to(device)
                val_loss += criterion(model(images), masks).item()

        print(f"Epoch {epoch + 1}: train={train_loss / len(train_loader):.4f}, " f"validation={val_loss / len(val_loader):.4f}")

    torch.save(model.state_dict(), model_save_path)
    return model

if __name__ == "__main__":
    DATASET_ROOT = Path("./dataset")
    MODEL_SAVE_PATH = Path("./models/unetplusplus.pth")
    dataset = SegmentationDataset(DATASET_ROOT, split="train", num_classes=2)
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_model(dataset, MODEL_SAVE_PATH, num_classes=3)