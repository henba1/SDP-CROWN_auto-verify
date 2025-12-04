import torch
import torch.nn as nn
import numpy as np
import gc
import onnx

from pathlib import Path
from models import *
from fractions import Fraction
from ada_verona import ONNXNetwork

def parse_float_or_fraction(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return float(Fraction(x))

# Helper function to generate C matrix for calculate the margins.
def build_C(label, classes):
    """
    label: shape (B,). Each label[b] in [0..classes-1].
    Return:
        C: shape (B, classes-1, classes).
        For each sample b, each row is a “negative class” among [0..classes-1]\{label[b]}.
        Puts +1 at column=label[b], -1 at each negative class column.
    """
    device = label.device
    batch_size = label.size(0)
    
    # 1) Initialize
    C = torch.zeros((batch_size, classes-1, classes), device=device)
    
    # 2) All class indices
    # shape: (1, K) -> (B, K)
    all_cls = torch.arange(classes, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # 3) Negative classes only, shape (B, K-1)
    # mask out the ground-truth
    mask = all_cls != label.unsqueeze(1)
    neg_cls = all_cls[mask].view(batch_size, -1)
    
    # 4) Scatter +1 at each sample’s ground-truth label
    #    shape needed: (B, K-1, 1)
    pos_idx = label.unsqueeze(1).expand(-1, classes-1).unsqueeze(-1)
    C.scatter_(dim=2, index=pos_idx, value=1.0)
    
    # 5) Scatter -1 at each row’s negative label
    #    We have (B, K-1) negative labels. For row j in each sample b, neg_cls[b, j] is that row’s negative label
    row_idx = torch.arange(classes-1, device=device).unsqueeze(0).expand(batch_size, -1)
    # shape: (B, K-1)
    
    # We can do advanced indexing:
    C[torch.arange(batch_size).unsqueeze(1), row_idx, neg_cls] = -1.0
    
    return C

def preprocess_cifar(image, inception_preprocess=False, perturbation=False):
    """
    Preprocess images and perturbations.Preprocessing used by the SDP paper.
    """
    MEANS = np.array([125.3, 123.0, 113.9], dtype=np.float32)/255
    STD = np.array([0.225, 0.225, 0.225], dtype=np.float32)
    if inception_preprocess:
        # Use 2x - 1 to get [-1, 1]-scaled images
        rescaled_devs = 0.5
        rescaled_means = 0.5
    else:
        rescaled_means = MEANS
        rescaled_devs = STD
    if perturbation:
        return image / rescaled_devs
    else:
        return (image - rescaled_means) / rescaled_devs

def log_onnx_metadata(onnx_net: ONNXNetwork, log_path: Path) -> None:
    """
    Extract and log key metadata from an ONNX model to a file.

    Args:
        onnx_net: The ONNXNetwork instance to extract metadata from.
        log_path: Path to the log file where metadata will be written.
    """
    onnx_model = onnx_net.load_onnx_model()
    input_shape = onnx_net.get_input_shape()

    # Extract input information
    input_info = []
    for inp in onnx_model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
        # Use the recommended helper function instead of deprecated mapping
        try:
            np_dtype = onnx.helper.tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type)
            dtype = str(np_dtype)  # Convert numpy dtype to string
        except (KeyError, AttributeError, TypeError) as e:
            dtype = f"Unknown (error: {e})"
        input_info.append({
            "name": inp.name,
            "shape": shape,
            "dtype": dtype,
        })

    # Extract output information
    output_info = []
    for out in onnx_model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
        # Use the recommended helper function instead of deprecated mapping
        try:
            np_dtype = onnx.helper.tensor_dtype_to_np_dtype(out.type.tensor_type.elem_type)
            dtype = str(np_dtype)  # Convert numpy dtype to string
        except (KeyError, AttributeError, TypeError) as e:
            dtype = f"Unknown (error: {e})"
        output_info.append({
            "name": out.name,
            "shape": shape,
            "dtype": dtype,
        })

    # Extract model metadata
    model_producer = onnx_model.producer_name if onnx_model.producer_name else "Unknown"
    model_version = onnx_model.producer_version if onnx_model.producer_version else "Unknown"
    opset_version = onnx_model.opset_import[0].version if onnx_model.opset_import else "Unknown"
    ir_version = onnx_model.ir_version

    # Count nodes and initializers (weights)
    num_nodes = len(onnx_model.graph.node)
    num_initializers = len(onnx_model.graph.initializer)

    # Get wrapper input shape (what TorchModelWrapper will reshape to)
    wrapper_input_shape = onnx_net.get_input_shape()

    # Write to log file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ONNX Model Metadata\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Model Path: {onnx_net.path}\n")
        f.write(f"Model Name: {onnx_net.name}\n\n")

        f.write(f"Input Shape: {input_shape}\n")

        f.write("Model Information:\n")
        f.write(f"  Producer: {model_producer}\n")
        f.write(f"  Producer Version: {model_version}\n")
        f.write(f"  IR Version: {ir_version}\n")
        f.write(f"  Opset Version: {opset_version}\n")
        f.write(f"  Number of Nodes: {num_nodes}\n")
        f.write(f"  Number of Initializers (Weights): {num_initializers}\n\n")

        f.write("Input Information:\n")
        for i, inp in enumerate(input_info):
            f.write(f"  Input {i}:\n")
            f.write(f"    Name: {inp['name']}\n")
            f.write(f"    Shape: {inp['shape']}\n")
            f.write(f"    Data Type: {inp['dtype']}\n")
        f.write(f"\n  TorchModelWrapper Input Shape: {wrapper_input_shape}\n\n")

        f.write("Output Information:\n")
        for i, out in enumerate(output_info):
            f.write(f"  Output {i}:\n")
            f.write(f"    Name: {out['name']}\n")
            f.write(f"    Shape: {out['shape']}\n")
            f.write(f"    Data Type: {out['dtype']}\n")
        f.write("\n")

        f.write("Expected Input Format:\n")
        f.write(f"  PyTorch format: (batch, channels, height, width)\n")
        if len(input_info) > 0 and len(input_info[0]["shape"]) == 4:
            shape = input_info[0]["shape"]
            f.write(f"  Expected shape: {shape}\n")
            if shape[0] == 1 or shape[0] == -1:
                f.write(f"  Batch size: {shape[0]} (dynamic: {shape[0] == -1})\n")
                f.write(f"  Channels: {shape[1]}\n")
                f.write(f"  Height: {shape[2]}\n")
                f.write(f"  Width: {shape[3]}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")


def load_model_and_dataset(args, device, image: np.ndarray):
    """
    Load a PyTorch model from a checkpoint path and wrap a single image/label
    instance into tensors usable by SDP-CROWN.

    Args:
        args: Argument namespace, with args.model (path to .pth or model id)
              and args.radius already set.
        device: Torch device.
        image: Numpy array representing a single input instance (flattened or shaped).
        label: Integer class label for the instance.

    Returns:
        model: nn.Module on the correct device, in eval mode.
        dataset: Tensor of shape (1, ...) containing the image.
        labels: Tensor of shape (1,) containing the label.
        radius_rescale: Float radius used for the perturbation.
        classes: Integer number of output classes inferred from the model.
    """

    model_path = Path(args.model)
    # Use VERONA's ONNX to Torch conversion (new code)
    onnx_net = ONNXNetwork.from_file(model_path)

    # Log ONNX metadata if log path is specified
    onnx_metadata_log_path = "/gpfs/work2/0/prjs1681/runs/results/SDP-crown_testing"
    if onnx_metadata_log_path is not None:
        log_path = Path(onnx_metadata_log_path) / "onnx_model_metadata.log"
        log_onnx_metadata(onnx_net, log_path)
        print(f"ONNX metadata logged to: {log_path}")
    # Alternatively, log to log directory if available
    elif hasattr(args, "logpath") and args.logpath:
        log_subdir = getattr(args, "log_subdir", "default")
        log_dir = Path(args.logpath) / log_subdir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "onnx_model_metadata.log"
        log_onnx_metadata(onnx_net, log_path)
        print(f"ONNX metadata logged to: {log_path}")

    torch_model_wrapper = onnx_net.load_pytorch_model() 
    model = torch_model_wrapper.to(device)
    model.eval()
    
    # Process single image: ensure it's in HWC format, add batch dimension, convert to tensor, permute to CHW
    image_arr = image.copy()
    
    # Handle different input shapes
    if image_arr.ndim == 1:
        # Flattened image - reshape based on size
        if image_arr.size == 3072:  # CIFAR-10: 3*32*32
            image_arr = image_arr.reshape(32, 32, 3)
        else:
            raise ValueError(f"Unexpected flattened image size: {image_arr.size}")
    # Add batch dimension: (H, W, C) -> (1, H, W, C)
    if image_arr.ndim == 3:
        image_arr = image_arr[np.newaxis, ...]

    #image_arr = preprocess_cifar(image_arr)
    # Convert to tensor and permute to (1, C, H, W)
    image_tensor = torch.from_numpy(image_arr).permute(0, 3, 1, 2)
    
    #radius_rescale = args.radius / 0.225
    radius_rescale = args.radius
    classes = 10 #hardcoded for now

    return model, image_tensor, radius_rescale, classes


#GPU memory management utility functions
def get_gpu_memory_info(device):
    """
    Get current GPU memory usage in GB and percentage.

    Args:
        device: CUDA device

    Returns:
        dict: Contains memory_allocated_gb, memory_reserved_gb, total_memory_gb, memory_percent
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_allocated = (
            torch.cuda.memory_allocated(device) / 1024**3
        )  # Convert to GB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # Convert to GB
        total_memory = (
            torch.cuda.get_device_properties(device).total_memory / 1024**3
        )  # Convert to GB
        memory_percent = (memory_allocated / total_memory) * 100
        return {
            "memory_allocated_gb": memory_allocated,
            "memory_reserved_gb": memory_reserved,
            "total_memory_gb": total_memory,
            "memory_percent": memory_percent,
        }
    return None


def cleanup_gpu_memory(model):
    """Clear GPU memory after each sample."""
    if torch.cuda.is_available():
        # Clear gradients from model
        for param in model.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad = None

        gc.collect()

        # Clear CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()