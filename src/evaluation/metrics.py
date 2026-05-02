import numpy as np


def _binarize(pred, threshold=0.5):
    """
    Converts probabilities → binary mask
    """
    return (pred >= threshold).astype(np.uint8)


def precision(pred, target, eps=1e-6):
    pred = _binarize(pred)
    target = target.astype(np.uint8)

    tp = np.sum((pred == 1) & (target == 1))
    fp = np.sum((pred == 1) & (target == 0))

    return tp / (tp + fp + eps)


def recall(pred, target, eps=1e-6):
    pred = _binarize(pred)
    target = target.astype(np.uint8)

    tp = np.sum((pred == 1) & (target == 1))
    fn = np.sum((pred == 0) & (target == 1))

    return tp / (tp + fn + eps)


def f1_score(pred, target, eps=1e-6):
    p = precision(pred, target)
    r = recall(pred, target)

    return 2 * p * r / (p + r + eps)


def iou_score(pred, target, eps=1e-6):
    pred = _binarize(pred)
    target = target.astype(np.uint8)

    intersection = np.sum((pred == 1) & (target == 1))
    union = np.sum((pred == 1) | (target == 1))

    return intersection / (union + eps)


def accuracy(pred, target):
    pred = _binarize(pred)
    target = target.astype(np.uint8)

    correct = np.sum(pred == target)
    total = target.size

    return correct / total


def evaluate_all(pred, target):
    """
    Returns all metrics in one call
    """

    return {
        "precision": precision(pred, target),
        "recall": recall(pred, target),
        "f1": f1_score(pred, target),
        "iou": iou_score(pred, target),
        "accuracy": accuracy(pred, target),
    }