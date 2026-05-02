import torch
import torch.nn as nn


class PrithviAdapter(nn.Module):
    """
    Adapter over Prithvi (ViT-based) backbone.

    Key idea:
    - Uses full forward pass (no masking)
    - Extracts dense spatial features for segmentation
    """

    def __init__(self, backbone, embed_dim=768):
        """
        backbone: pretrained Prithvi model
        embed_dim: transformer embedding size
        """
        super().__init__()

        self.backbone = backbone
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        x: (B, C, H, W)

        Returns:
        feature map (B, D, H', W')
        """

        # Prithvi usually expects patches → internally handled
        # We bypass masking and use full encoder

        features = self._forward_backbone(x)

        return features

    def _forward_backbone(self, x):
        """
        Handles forward pass through ViT encoder
        """

        # Case 1: backbone has custom forward_features (common in ViT)
        if hasattr(self.backbone, "forward_features"):
            tokens = self.backbone.forward_features(x)

        else:
            # fallback
            tokens = self.backbone(x)

        # tokens shape: (B, N, D)
        # remove cls token if present
        if tokens.dim() == 3 and tokens.shape[1] > 1:
            tokens = tokens[:, 1:, :]

        B, N, D = tokens.shape

        # convert tokens → spatial map
        H = W = int(N ** 0.5)

        features = tokens.transpose(1, 2).contiguous()
        features = features.view(B, D, H, W)

        return features


def load_prithvi_backbone(model_name="prithvi_eo_v2", pretrained=True):
    """
    Placeholder loader.

    You can replace this with actual Prithvi loading logic
    (from HuggingFace / internal weights / checkpoint)
    """

    # Example stub — replace later
    print(f"[INFO] Loading backbone: {model_name}")

    backbone = None

    if pretrained:
        print("[INFO] Using pretrained weights (implement this)")

    return backbone