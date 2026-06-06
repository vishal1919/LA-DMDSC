"""
Utilities for creating imbalanced datasets dynamically.
Geometric long-tailed imbalance based on target IR.
Works for ASC (6 classes) and any other dataset.

Key idea:
  IR = N_maj / N_min

If exact_ir=True:
  N_min = floor(max_samples / IR)  (>=1)
  N_maj = N_min * IR               (<= max_samples)
This guarantees exact IR with integer counts (when possible under max_samples).

Example: max_samples=320, IR=100, n_classes=6
  N_min = 320//100 = 3
  N_maj = 3*100 = 300
  => [300, 119, 47, 19, 7, 3]
"""

import numpy as np
from collections import Counter
from typing import Dict, Optional, List


def create_imbalanced_indices(
    labels: np.ndarray,
    imbalance_ratio: int = 10,
    random_state: int = 42,
    exact_ir: bool = True,
    shuffle_class_order: bool = False,
) -> np.ndarray:
    """
    Create indices for an imbalanced dataset using geometric long-tailed decay.

    Parameters
    ----------
    labels : np.ndarray, shape (n_samples,)
        Original labels.
    imbalance_ratio : int
        Target IR = max_samples / min_samples (e.g., 2, 5, 10, 50, 100).
    random_state : int
        Random seed for reproducibility.
    exact_ir : bool
        If True, slightly reduces the effective majority class so that IR is exact
        with integer counts (under max_samples constraint).
        If False, keeps majority at max_samples and IR may be approximate.
    shuffle_class_order : bool
        If True, randomly assigns which class becomes majority/minority, etc.
        (Useful to avoid "class 0 is always majority" bias.)

    Returns
    -------
    np.ndarray
        Indices to sample from the original dataset.
    """
    rng = np.random.default_rng(random_state)

    labels = np.array(labels).flatten()
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)

    # Original class counts
    class_counts = Counter(labels)
    max_samples = max(class_counts.values())

    # Compute target samples per class id (0..n_classes-1 in sorted label order)
    target_samples = get_ir_based_samples(
        n_classes=n_classes,
        max_samples=max_samples,
        imbalance_ratio=imbalance_ratio,
        exact_ir=exact_ir,
        random_state=random_state,
        shuffle_class_order=shuffle_class_order,
        class_labels=sorted(unique_classes.tolist()),
    )

    imbalanced_indices: List[int] = []

    for class_label in sorted(unique_classes):
        class_indices = np.where(labels == class_label)[0]
        n_target = target_samples[class_label]

        # Ensure we don't sample more than available
        n_target = min(n_target, len(class_indices))

        if n_target > 0:
            sampled = rng.choice(class_indices, size=n_target, replace=False)
            imbalanced_indices.extend(sampled.tolist())

    imbalanced_indices = np.array(imbalanced_indices, dtype=int)
    rng.shuffle(imbalanced_indices)
    return imbalanced_indices


def _get_geometric_samples(
    n_classes: int,
    max_samples: int,
    ir: int,
    exact_ir: bool = True,
) -> List[int]:
    """
    Generate geometric long-tailed samples (monotone non-increasing) from max to min.

    If exact_ir=True:
      min_s = floor(max_samples / ir)  (>=1)
      max_eff = min_s * ir            (<= max_samples)
      so max_eff / min_s == ir exactly.

    If exact_ir=False:
      max_eff = max_samples
      min_s = floor(max_samples / ir) (>=1)
      so achieved IR may be approximate due to integer rounding.

    Returns a list of length n_classes: [N1, N2, ..., NC]
    """
    if n_classes <= 0:
        return []
    if n_classes == 1:
        return [max_samples]
    if ir < 1:
        raise ValueError("imbalance_ratio (ir) must be >= 1")

    min_s = max(1, max_samples // ir)  # floor
    if exact_ir:
        max_eff = min_s * ir           # ensures exact IR, <= max_samples
    else:
        max_eff = max_samples

    # geometric ratio r so that: max_eff * r^(n_classes-1) = min_s
    r = (min_s / max_eff) ** (1.0 / (n_classes - 1))

    samples: List[int] = []
    for k in range(n_classes):
        val = int(np.round(max_eff * (r ** k)))
        samples.append(val)

    # enforce endpoints exactly
    samples[0] = max_eff
    samples[-1] = min_s

    # fix rounding to be monotone non-increasing and >= 1
    for i in range(1, n_classes):
        samples[i] = max(1, min(samples[i], samples[i - 1]))

    # never exceed available max_samples
    samples[0] = min(samples[0], max_samples)
    for i in range(1, n_classes):
        samples[i] = min(samples[i], samples[i - 1])

    return samples


def get_ir_based_samples(
    n_classes: int,
    max_samples: int,
    imbalance_ratio: int,
    exact_ir: bool = True,
    random_state: int = 42,
    shuffle_class_order: bool = False,
    class_labels: Optional[List[int]] = None,
) -> Dict[int, int]:
    """
    Calculate target samples per class using geometric decay.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    max_samples : int
        Maximum samples available per class (assumes roughly balanced start).
    imbalance_ratio : int
        Target IR.
    exact_ir : bool
        Make IR exact by reducing effective majority if needed.
    random_state : int
        RNG seed if shuffle_class_order=True.
    shuffle_class_order : bool
        If True, randomly permute which labels get which sample counts.
    class_labels : Optional[List[int]]
        Actual label ids in sorted order. If provided, returned dict keys are these labels.
        If None, keys are 0..n_classes-1.

    Returns
    -------
    Dict[int, int]
        Mapping: class_label -> target_sample_count
    """
    samples = _get_geometric_samples(
        n_classes=n_classes,
        max_samples=max_samples,
        ir=imbalance_ratio,
        exact_ir=exact_ir,
    )
    samples = [max(1, int(s)) for s in samples]

    if class_labels is None:
        class_labels = list(range(n_classes))

    if len(class_labels) != n_classes:
        raise ValueError("class_labels length must match n_classes")

    if shuffle_class_order:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(n_classes)
        # assign largest to a random label, etc.
        shuffled = [0] * n_classes
        for i, p in enumerate(perm):
            shuffled[p] = samples[i]
        samples = shuffled

    return {class_labels[i]: samples[i] for i in range(n_classes)}


def get_imbalance_info(
    labels: np.ndarray,
    indices: Optional[np.ndarray] = None
) -> Dict:
    """
    Get information about dataset imbalance.
    """
    if indices is not None:
        labels = labels[indices]

    labels = np.array(labels).flatten()
    class_counts = Counter(labels)

    max_count = max(class_counts.values()) if class_counts else 0
    min_count = min(class_counts.values()) if class_counts else 0
    ir = (max_count / min_count) if min_count > 0 else float("inf")

    return {
        "class_counts": dict(class_counts),
        "total_samples": len(labels),
        "max_samples": max_count,
        "min_samples": min_count,
        "imbalance_ratio": ir
    }


def print_imbalance_summary(
    labels: np.ndarray,
    indices: Optional[np.ndarray] = None,
    dataset_name: str = "Dataset"
):
    """
    Print formatted imbalance summary.
    """
    info = get_imbalance_info(labels, indices)

    print(f"\n{'='*60}")
    print(f"{dataset_name} Imbalance Summary")
    print(f"{'='*60}")
    print(f"Total samples: {info['total_samples']}")
    print(f"Imbalance Ratio (IR): {info['imbalance_ratio']:.2f}")
    print(f"\nClass distribution:")

    for class_id in sorted(info["class_counts"].keys()):
        count = info["class_counts"][class_id]
        percentage = (count / info["total_samples"]) * 100 if info["total_samples"] > 0 else 0
        print(f"  Class {class_id}: {count:4d} samples ({percentage:5.2f}%)")

    print(f"{'='*60}\n")


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    """
    Compute inverse frequency class weights for imbalanced data.

    Returns
    -------
    np.ndarray
        Weight for each class (shape: num_classes) assuming class ids are 0..C-1.
        If your labels are not 0..C-1, remap them before using this.
    """
    labels = np.array(labels).flatten()
    class_counts = Counter(labels)
    n_samples = len(labels)
    n_classes = len(class_counts)

    # If labels are not contiguous 0..C-1, this will not align; remap in that case.
    max_label = max(class_counts.keys()) if class_counts else -1
    weights = np.zeros(max_label + 1, dtype=float)

    for class_id, count in class_counts.items():
        weights[class_id] = n_samples / (n_classes * count)

    return weights
