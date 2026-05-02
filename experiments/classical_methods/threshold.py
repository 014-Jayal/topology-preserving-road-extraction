import os
import cv2
import numpy as np
from tqdm import tqdm


def apply_threshold(image):
    """
    Simple global thresholding.
    Works on grayscale or single-channel intensity.
    """

    if len(image.shape) == 3:
        # convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Otsu threshold (auto)
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return mask


def load_image(path):
    if path.endswith(".tif"):
        import rasterio
        with rasterio.open(path) as src:
            image = src.read()
            image = np.transpose(image, (1, 2, 0))
    else:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def run_threshold(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(os.listdir(input_dir))

    for file in tqdm(files, desc="Thresholding"):
        img_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, file)

        image = load_image(img_path)

        mask = apply_threshold(image)

        cv2.imwrite(out_path, mask)


if __name__ == "__main__":
    input_dir = "data/images"
    output_dir = "results/threshold"

    run_threshold(input_dir, output_dir)