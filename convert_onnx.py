import argparse
from pathlib import Path

import torch
from onnx2torch import convert

from models import CONV_BIG


def copy_weights_greedy_by_shape(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    """
    Copy parameters from src → dst by walking through src state_dict and
    assigning the next tensor with a matching shape to each dst parameter.

    This tolerates extra tensors in src (e.g. from ONNX reshape / fused ops):
    - We iterate over dst params in order.
    - For each dst param, we advance through src params until we find a shape match.
    - Any unmatched src params are effectively skipped.
    """
    src_items = list(src.state_dict().items())
    dst_items = list(dst.state_dict().items())

    new_state: dict[str, torch.Tensor] = {}
    src_idx = 0

    for dst_name, dst_tensor in dst_items:
        dst_shape = tuple(dst_tensor.shape)
        matched = False

        while src_idx < len(src_items):
            src_name, src_tensor = src_items[src_idx]
            src_idx += 1
            src_shape = tuple(src_tensor.shape)

            if src_shape == dst_shape:
                new_state[dst_name] = src_tensor.detach().clone()
                matched = True
                break

        if not matched:
            raise ValueError(
                f"Could not find a matching source tensor for dst param '{dst_name}' "
                f"with shape {dst_shape}. Reached end of src state_dict."
            )

    # Load the constructed state_dict into dst
    dst.load_state_dict(new_state)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ONNX ConvBig model to a CONV_BIG-compatible .pth (state_dict)."
    )
    parser.add_argument(
        "--onnx_model",
        type=str,
        required=True,
        help="Path to the conv_big ONNX model (e.g. conv_big_standard.onnx).",
    )
    parser.add_argument(
        "--output_pth",
        type=str,
        required=True,
        help="Output path for the converted .pth (state_dict) file.",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx_model)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    output_path = Path(args.output_pth)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load ONNX as a PyTorch module via onnx2torch.
    print(f"Loading ONNX model from: {onnx_path}")
    onnx_model = convert(str(onnx_path)).to(device)
    onnx_model.eval()

    # 2) Instantiate hand-written CONV_BIG architecture.
    conv_big = CONV_BIG().to(device)
    conv_big.eval()

    # Optional: print param counts for sanity.
    print(f"src (onnx2torch) param tensors: {len(onnx_model.state_dict())}")
    print(f"dst (CONV_BIG)      param tensors: {len(conv_big.state_dict())}")

    # 3) Copy weights greedily by matching shapes.
    print("Copying weights from ONNX-converted model into CONV_BIG (shape-greedy)...")
    copy_weights_greedy_by_shape(src=onnx_model, dst=conv_big)

    # 4) Save CONV_BIG state_dict.
    state_dict = conv_big.state_dict()
    torch.save(state_dict, output_path)
    print(f"Saved CONV_BIG-compatible state_dict to: {output_path}")


if __name__ == "__main__":
    main()