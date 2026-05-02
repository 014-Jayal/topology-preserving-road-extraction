import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import RoadDataset
from src.training.loss import HybridLoss
from experiments.cnn_baseline.model import SimpleCNN


def train_cnn(
    train_img_dir,
    train_mask_dir,
    val_img_dir,
    val_mask_dir,
    epochs=20,
    batch_size=8,
    device="cuda",
    save_path="checkpoints/cnn.pth"
):

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = SimpleCNN(in_channels=3).to(device)

    train_dataset = RoadDataset(train_img_dir, train_mask_dir)
    val_dataset = RoadDataset(val_img_dir, val_mask_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = HybridLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        model.train()
        train_loss = 0

        for images, masks in tqdm(train_loader, leave=False):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print("Saved best model")


if __name__ == "__main__":
    train_cnn(
        train_img_dir="data/train/images",
        train_mask_dir="data/train/masks",
        val_img_dir="data/val/images",
        val_mask_dir="data/val/masks"
    )