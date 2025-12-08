import argparse
from pathlib import Path

import torch
from onnx2torch import convert

from models import CONV_BIG


def print_state_dict_summary(state, title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"Total tensors: {len(state)}")
    for name, tensor in state.items():
        shape = "x".join(str(d) for d in tensor.shape)
        print(f"  {name:40s} shape=({shape})")


def investigate_mismatch(onnx_model: torch.nn.Module, conv_big: torch.nn.Module) -> None:
    onnx_state = onnx_model.state_dict()
    conv_state = conv_big.state_dict()

    print_state_dict_summary(onnx_state, "ONNX→Torch model state_dict")
    print_state_dict_summary(conv_state, "CONV_BIG state_dict")

    onnx_keys = list(onnx_state.keys())
    conv_keys = list(conv_state.keys())

    print("\n=== Key count ===")
    print(f"ONNX→Torch: {len(onnx_keys)} tensors")
    print(f"CONV_BIG  : {len(conv_keys)} tensors")

    # Compare key-by-key where possible.
    print("\n=== First few keys side-by-side (name, shape) ===")
    for i, (ok, ck) in enumerate(zip(onnx_keys, conv_keys, strict=False)):
        if i >= max(len(onnx_keys), len(conv_keys)):
            break
        o_name = onnx_keys[i] if i < len(onnx_keys) else "<missing>"
        c_name = conv_keys[i] if i < len(conv_keys) else "<missing>"
        o_shape = tuple(onnx_state[o_name].shape) if o_name in onnx_state else ()
        c_shape = tuple(conv_state[c_name].shape) if c_name in conv_state else ()
        print(f"{i:2d}: ONNX {o_name:35s} {o_shape}   |   CONV_BIG {c_name:35s} {c_shape}")

    # Show any ONNX keys that have no shape match in CONV_BIG.
    print("\n=== ONNX params with shapes not present anywhere in CONV_BIG ===")
    conv_shapes = {tuple(t.shape) for t in conv_state.values()}
    for name, tensor in onnx_state.items():
        if tuple(tensor.shape) not in conv_shapes:
            print(f"  {name:40s} shape={tuple(tensor.shape)}  (no matching shape in CONV_BIG)")

    # Show any CONV_BIG params with shapes not present in ONNX.
    print("\n=== CONV_BIG params with shapes not present anywhere in ONNX→Torch ===")
    onnx_shapes = {tuple(t.shape) for t in onnx_state.values()}
    for name, tensor in conv_state.items():
        if tuple(tensor.shape) not in onnx_shapes:
            print(f"  {name:40s} shape={tuple(tensor.shape)}  (no matching shape in ONNX)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect mismatch between ONNX ConvBig and hand-written CONV_BIG.")
    parser.add_argument(
        "--onnx_model",
        type=str,
        required=True,
        help="Path to the conv_big ONNX model (e.g. conv_big_standard.onnx).",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx_model)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading ONNX model from: {onnx_path}")
    onnx_model = convert(str(onnx_path)).to(device)
    onnx_model.eval()

    conv_big = CONV_BIG().to(device)
    conv_big.eval()

    investigate_mismatch(onnx_model, conv_big)


if __name__ == "__main__":
    main()