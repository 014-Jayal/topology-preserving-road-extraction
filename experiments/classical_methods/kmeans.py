import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans


def apply_kmeans(image, k=2):
    """
    K-Means clustering for segmentation.

    Clusters pixels into k groups and returns binary mask.
    """

    h, w, c = image.shape

    # reshape to (num_pixels, channels)
    pixels = image.reshape(-1, c).astype(np.float32)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)

    labels = labels.reshape(h, w)

    # choose cluster with higher intensity as road (simple heuristic)
    cluster_means = []
    for i in range(k):
        cluster_means.append(np.mean(pixels[labels.reshape(-1) == i]))

    road_cluster = int(np.argmax(cluster_means))

    mask = (labels == road_cluster).astype(np.uint8) * 255

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


def run_kmeans(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(os.listdir(input_dir))

    for file in tqdm(files, desc="KMeans"):
        img_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, file)

        image = load_image(img_path)

        mask = apply_kmeans(image)

        cv2.imwrite(out_path, mask)


if __name__ == "__main__":
    input_dir = "data/images"
    output_dir = "results/kmeans"

    run_kmeans(input_dir, output_dir)