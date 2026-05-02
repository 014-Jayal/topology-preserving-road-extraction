import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import RoadDataset
from src.training.loss import HybridLoss
from src.models.decoder import RoadSegmentationModel


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    loop = tqdm(loader, leave=False)

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_description(f"loss: {loss.item():.4f}")

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()

    return total_loss / len(loader)


def get_optimizer(model, backbone_lr=1e-5, decoder_lr=5e-4):
    """
    Split LR:
    - backbone → low LR
    - decoder → higher LR
    """

    return torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": backbone_lr},
        {"params": model.decoder.parameters(), "lr": decoder_lr},
    ])


def train(
    train_img_dir,
    train_mask_dir,
    val_img_dir,
    val_mask_dir,
    model,
    epochs=20,
    batch_size=8,
    device="cuda",
    save_path="checkpoints/model.pth"
):

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # datasets
    train_dataset = RoadDataset(train_img_dir, train_mask_dir)
    val_dataset = RoadDataset(val_img_dir, val_mask_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # loss + optimizer
    criterion = HybridLoss()
    optimizer = get_optimizer(model)

    best_loss = float("inf")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        # save best model
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print("Saved best model")


if __name__ == "__main__":
    # placeholder — we’ll wire actual model later

    print("Training script ready. Connect model + paths.")