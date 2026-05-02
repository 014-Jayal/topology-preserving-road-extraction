import os
import argparse
import numpy as np
import torch
import cv2
import rasterio

from src.models.prithvi_adapter import PrithviAdapter, load_prithvi_backbone
from src.models.decoder import SimpleDecoder, RoadSegmentationModel
from src.inference.sliding_window import sliding_window_inference


def load_image(path):
    """
    Supports .tif and normal images
    """
    if path.endswith(".tif"):
        with rasterio.open(path) as src:
            image = src.read()
            image = np.transpose(image, (1, 2, 0))
    else:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype(np.float32) / 255.0
    return image


def save_mask(mask, path):
    """
    Saves binary mask
    """
    mask = (mask > 0.5).astype(np.uint8) * 255
    cv2.imwrite(path, mask)


def build_model(checkpoint_path, device):
    """
    Loads full model
    """

    backbone = load_prithvi_backbone()
    backbone = PrithviAdapter(backbone)

    decoder = SimpleDecoder(in_channels=768)
    model = RoadSegmentationModel(backbone, decoder)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    return model


def run_inference(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading model...")
    model = build_model(args.checkpoint, device)

    print("[INFO] Loading image...")
    image = load_image(args.input)

    print("[INFO] Running sliding window inference...")
    pred = sliding_window_inference(
        model,
        image,
        device=device,
        patch_size=args.patch_size,
        stride=args.stride
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("[INFO] Saving output...")
    save_mask(pred, args.output)

    print("[DONE] Inference complete")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="outputs/pred.png")
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)

    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)