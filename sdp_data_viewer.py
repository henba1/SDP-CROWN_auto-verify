"""Utility functions to load and view SDP dataset files (.npy format)."""

import numpy as np
from pathlib import Path


def load_sdp_data(dataset_dir: str | Path, dataset_name: str = "cifar") -> tuple[np.ndarray, np.ndarray]:
    """
    Load SDP dataset from .npy files.

    Args:
        dataset_dir: Path to the datasets/sdp directory
        dataset_name: Name of the dataset subdirectory ('cifar' or 'mnist')

    Returns:
        Tuple of (X, y) where X is the feature array and y is the label array
    """
    dataset_path = Path(dataset_dir) / dataset_name
    X_path = dataset_path / "X_sdp.npy"
    y_path = dataset_path / "y_sdp.npy"

    if not X_path.exists():
        raise FileNotFoundError(f"X_sdp.npy not found at {X_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"y_sdp.npy not found at {y_path}")

    X = np.load(X_path)
    y = np.load(y_path)

    return X, y


def view_sdp_data(
    dataset_dir: str | Path,
    dataset_name: str = "cifar",
    show_sample: bool = True,
    sample_idx: int = 0,
) -> None:
    """
    Load and display information about SDP dataset files.

    Args:
        dataset_dir: Path to the datasets/sdp directory
        dataset_name: Name of the dataset subdirectory ('cifar' or 'mnist')
        show_sample: Whether to display sample statistics
        sample_idx: Index of the sample to display (if show_sample=True)
    """
    X, y = load_sdp_data(dataset_dir, dataset_name)

    print(f"\n{'='*60}")
    print(f"SDP Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")

    print(f"\nFeatures (X_sdp.npy):")
    print(f"  Shape: {X.shape}")
    print(f"  Dtype: {X.dtype}")
    print(f"  Min: {X.min():.6f}, Max: {X.max():.6f}, Mean: {X.mean():.6f}, Std: {X.std():.6f}")

    print(f"\nLabels (y_sdp.npy):")
    print(f"  Shape: {y.shape}")
    print(f"  Dtype: {y.dtype}")
    if y.size > 0:
        print(f"  Min: {y.min()}, Max: {y.max()}")
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"  Unique labels: {len(unique_labels)}")
        print(f"  Label distribution: {dict(zip(unique_labels, counts))}")

    print(f"\nDataset Summary:")
    print(f"  Number of samples: {X.shape[0] if X.ndim > 1 else 1}")
    if X.ndim > 1:
        print(f"  Feature dimensions: {X.shape[1:]}")
    if X.shape[0] == y.shape[0] or (y.ndim == 0 and X.ndim > 0):
        print(f"  Samples match labels: ✓")

    if show_sample and X.shape[0] > 0:
        idx = min(sample_idx, X.shape[0] - 1)
        print(f"\nSample {idx}:")
        print(f"  Features shape: {X[idx].shape}")
        print(f"  Label: {y[idx] if y.ndim > 0 else y}")
        print(f"  Feature range: [{X[idx].min():.6f}, {X[idx].max():.6f}]")
        print(f"  Feature mean: {X[idx].mean():.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="View SDP dataset files")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./datasets/sdp",
        help="Path to datasets/sdp directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar", "mnist"],
        default="cifar",
        help="Dataset name to view",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of sample to display",
    )
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Don't show sample statistics",
    )

    args = parser.parse_args()
    view_sdp_data(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset,
        show_sample=not args.no_sample,
        sample_idx=args.sample_idx,
    )

