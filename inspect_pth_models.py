import argparse
from pathlib import Path

import torch

from models import CONV_BIG, CIFAR10_ConvLarge


def print_state_dict_summary(state, title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"Total tensors: {len(state)}")
    for name, tensor in state.items():
        shape = "x".join(str(d) for d in tensor.shape)
        print(f"  {name:40s} shape=({shape})")


def investigate_mismatch(state_a: dict, state_b: dict, title_a: str, title_b: str) -> None:
    print_state_dict_summary(state_a, title_a)
    print_state_dict_summary(state_b, title_b)

    keys_a = list(state_a.keys())
    keys_b = list(state_b.keys())

    print("\n=== Key count ===")
    print(f"{title_a}: {len(keys_a)} tensors")
    print(f"{title_b}: {len(keys_b)} tensors")

    print("\n=== First few keys side-by-side (name, shape) ===")
    max_len = max(len(keys_a), len(keys_b))
    for i in range(max_len):
        a_name = keys_a[i] if i < len(keys_a) else "<missing>"
        b_name = keys_b[i] if i < len(keys_b) else "<missing>"
        a_shape = tuple(state_a[a_name].shape) if a_name in state_a else ()
        b_shape = tuple(state_b[b_name].shape) if b_name in state_b else ()
        print(f"{i:2d}: A {a_name:35s} {a_shape}   |   B {b_name:35s} {b_shape}")

    print(f"\n=== {title_a} params with shapes not present anywhere in {title_b} ===")
    shapes_b = {tuple(t.shape) for t in state_b.values()}
    for name, tensor in state_a.items():
        if tuple(tensor.shape) not in shapes_b:
            print(f"  {name:40s} shape={tuple(tensor.shape)}  (no matching shape in {title_b})")

    print(f"\n=== {title_b} params with shapes not present anywhere in {title_a} ===")
    shapes_a = {tuple(t.shape) for t in state_a.values()}
    for name, tensor in state_b.items():
        if tuple(tensor.shape) not in shapes_a:
            print(f"  {name:40s} shape={tuple(tensor.shape)}  (no matching shape in {title_a})")


def load_arch_for_pth(path: Path, device: torch.device) -> torch.nn.Module:
    """Instantiate a suitable architecture and load a .pth checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, torch.nn.Module):
        return ckpt.to(device)

    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint type at '{path}': {type(ckpt)}")

    name = path.stem.lower()
    if "conv_big" in name:
        model = CONV_BIG().to(device)
    elif "convlarge" in name or "conv_large" in name:
        model = CIFAR10_ConvLarge().to(device)
    else:
        raise ValueError(
            f"Cannot infer architecture for '{path.name}'. "
            "Add a case in load_arch_for_pth for this file."
        )

    model.load_state_dict(ckpt)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect mismatch between two PyTorch .pth models (e.g. conv_big_from_onnx vs cifar10_convlarge)."
    )
    parser.add_argument(
        "--model_a",
        type=str,
        required=True,
        help="Path to first .pth model (e.g. conv_big_from_onnx.pth).",
    )
    parser.add_argument(
        "--model_b",
        type=str,
        required=True,
        help="Path to second .pth model (e.g. cifar10_convlarge.pth).",
    )
    args = parser.parse_args()

    path_a = Path(args.model_a)
    path_b = Path(args.model_b)
    if not path_a.is_file():
        raise FileNotFoundError(f"Model A not found: {path_a}")
    if not path_b.is_file():
        raise FileNotFoundError(f"Model B not found: {path_b}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model A from: {path_a}")
    model_a = load_arch_for_pth(path_a, device)
    print(f"Loading model B from: {path_b}")
    model_b = load_arch_for_pth(path_b, device)

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    investigate_mismatch(state_a, state_b, title_a=path_a.name, title_b=path_b.name)


if __name__ == "__main__":
    main()