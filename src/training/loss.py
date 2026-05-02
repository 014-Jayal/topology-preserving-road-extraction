import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for overlap and structure preservation
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: (B, 1, H, W)
        targets: (B, H, W)
        """

        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss to handle class imbalance
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits: (B, 1, H, W)
        targets: (B, H, W)
        """

        probs = torch.sigmoid(logits)

        targets = targets.unsqueeze(1)  # match shape

        bce = F.binary_cross_entropy(probs, targets, reduction="none")

        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce

        return focal.mean()


class HybridLoss(nn.Module):
    """
    Combination of Focal + Dice
    """

    def __init__(self, alpha=0.5):
        super().__init__()

        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.alpha = alpha  # balance between two losses

    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)

        loss = self.alpha * focal_loss + (1 - self.alpha) * dice_loss

        return loss