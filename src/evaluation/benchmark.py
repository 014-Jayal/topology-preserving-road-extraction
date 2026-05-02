import os
import numpy as np
import cv2
from tqdm import tqdm

from src.evaluation.metrics import evaluate_all


def load_mask(path):
    """
    Loads binary mask
    """
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.uint8)
    return mask


def evaluate_model(pred_dir, gt_dir):
    """
    Evaluates a model given prediction folder and ground truth folder

    Assumes:
    - same filenames in both dirs
    """

    files = sorted(os.listdir(gt_dir))

    all_metrics = {
        "precision": [],
        "recall": [],
        "f1": [],
        "iou": [],
        "accuracy": []
    }

    for file in tqdm(files, desc="Evaluating"):
        gt_path = os.path.join(gt_dir, file)
        pred_path = os.path.join(pred_dir, file)

        if not os.path.exists(pred_path):
            continue

        gt = load_mask(gt_path)
        pred = load_mask(pred_path)

        metrics = evaluate_all(pred, gt)

        for k in all_metrics:
            all_metrics[k].append(metrics[k])

    # average metrics
    results = {k: np.mean(v) for k, v in all_metrics.items()}

    return results


def print_results(results_dict):
    """
    Prints clean comparison table
    """

    print("\n" + "=" * 70)
    print(f"{'Model':<20} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'IoU':<10}")
    print("-" * 70)

    for model_name, metrics in results_dict.items():
        print(
            f"{model_name:<20} | "
            f"{metrics['precision']:.4f}     | "
            f"{metrics['recall']:.4f}     | "
            f"{metrics['f1']:.4f}     | "
            f"{metrics['iou']:.4f}"
        )

    print("=" * 70)


def run_benchmark(models_dict, gt_dir):
    """
    models_dict:
        {
            "model_name": "path/to/predictions"
        }
    """

    results = {}

    for model_name, pred_dir in models_dict.items():
        print(f"\nRunning evaluation for: {model_name}")
        results[model_name] = evaluate_model(pred_dir, gt_dir)

    print_results(results)

    return results


if __name__ == "__main__":
    # Example usage

    models = {
        "threshold": "results/threshold",
        "kmeans": "results/kmeans",
        "random_forest": "results/rf",
        "svm": "results/svm",
        "cnn": "results/cnn",
        "prithvi": "results/prithvi"
    }

    gt_dir = "data/gt_masks"

    run_benchmark(models, gt_dir)