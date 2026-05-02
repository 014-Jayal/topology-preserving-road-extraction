import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC


def extract_features(image):
    """
    Same feature setup as RF for consistency
    """

    h, w, c = image.shape

    pixels = image.reshape(-1, c).astype(np.float32)

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.stack([xs, ys], axis=-1).reshape(-1, 2)

    features = np.concatenate([pixels, coords], axis=1)

    return features


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


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.uint8)
    return mask


def train_svm(image_paths, mask_paths, sample_size=200000):
    """
    Train SVM (sampling needed because SVM is slow)
    """

    X = []
    y = []

    for img_path, mask_path in zip(image_paths, mask_paths):
        image = load_image(img_path)
        mask = load_mask(mask_path)

        features = extract_features(image)
        labels = mask.reshape(-1)

        X.append(features)
        y.append(labels)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    # random sampling (important)
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X = X[idx]
        y = y[idx]

    print(f"[INFO] Training SVM on {X.shape[0]} samples")

    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    )

    model.fit(X, y)

    return model


def predict_svm(model, image):
    h, w, _ = image.shape

    features = extract_features(image)
    preds = model.predict(features)

    mask = preds.reshape(h, w).astype(np.uint8) * 255

    return mask


def run_svm(train_img_dir, train_mask_dir, test_img_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    train_imgs = sorted(os.listdir(train_img_dir))
    train_masks = sorted(os.listdir(train_mask_dir))

    train_img_paths = [os.path.join(train_img_dir, f) for f in train_imgs]
    train_mask_paths = [os.path.join(train_mask_dir, f) for f in train_masks]

    # train
    model = train_svm(train_img_paths, train_mask_paths)

    # inference
    test_files = sorted(os.listdir(test_img_dir))

    for file in tqdm(test_files, desc="SVM Inference"):
        img_path = os.path.join(test_img_dir, file)
        out_path = os.path.join(output_dir, file)

        image = load_image(img_path)

        mask = predict_svm(model, image)

        cv2.imwrite(out_path, mask)


if __name__ == "__main__":
    run_svm(
        train_img_dir="data/train/images",
        train_mask_dir="data/train/masks",
        test_img_dir="data/test/images",
        output_dir="results/svm"
    )