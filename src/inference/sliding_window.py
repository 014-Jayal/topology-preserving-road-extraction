import numpy as np
import torch


def generate_patches(image, patch_size=224, stride=112):
    """
    Splits image into overlapping patches.

    image: (H, W, C)
    """
    H, W = image.shape[:2]
    patches = []

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append((patch, i, j))

    return patches


def reconstruct_from_patches(preds, coords, image_shape, patch_size=224):
    """
    Reconstruct full image from overlapping predictions.

    preds: list of (patch_pred) → (1, H, W)
    coords: list of (i, j)
    image_shape: (H, W)
    """

    H, W = image_shape
    output = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for pred, (i, j) in zip(preds, coords):
        pred = pred.squeeze()  # (H, W)

        output[i:i + patch_size, j:j + patch_size] += pred
        count_map[i:i + patch_size, j:j + patch_size] += 1.0

    # avoid division by zero
    count_map[count_map == 0] = 1.0

    output = output / count_map

    return output


def sliding_window_inference(model, image, device="cuda",
                             patch_size=224, stride=112):
    """
    Runs inference on large image using sliding window.

    image: numpy array (H, W, C)
    """

    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    patches = generate_patches(image, patch_size, stride)

    preds = []
    coords = []

    with torch.no_grad():
        for patch, i, j in patches:
            # preprocess
            patch_tensor = torch.tensor(patch, dtype=torch.float32)
            patch_tensor = patch_tensor.permute(2, 0, 1).unsqueeze(0)
            patch_tensor = patch_tensor.to(device)

            # forward
            output = model(patch_tensor)
            output = torch.sigmoid(output)

            preds.append(output.cpu().numpy())
            coords.append((i, j))

    full_pred = reconstruct_from_patches(
        preds,
        coords,
        image.shape[:2],
        patch_size
    )

    return full_pred