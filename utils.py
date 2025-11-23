import torch
import torch.nn as nn
import numpy as np
import gc

from pathlib import Path
from models import *
from fractions import Fraction

def parse_float_or_fraction(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return float(Fraction(x))

# Help function to generate C matrix for calculate the margins.
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
    model_arg = args.model
    model_path = Path(model_arg)

    if model_path.suffix == ".pth":
        # Generic PyTorch checkpoint path. We assume it stores an nn.Module
        # or a state_dict compatible with one of the architectures in models.py.
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, nn.Module):
            model = checkpoint.to(device)
        elif isinstance(checkpoint, dict):
            # User must pass in a compatible architecture via model id.
            # Fall back to simple MNIST/CIFAR-10 architectures based on name.
            name = model_path.stem.lower()
            if "mnist" in name and "mlp" in name:
                model = MNIST_MLP().to(device)
            elif "mnist" in name and "convsmall" in name:
                model = MNIST_ConvSmall().to(device)
            elif "mnist" in name:
                model = MNIST_ConvLarge().to(device)
            elif "cifar" in name and "cnn_a" in name:
                model = CIFAR10_CNN_A().to(device)
            elif "cifar" in name and "cnn_b" in name:
                model = CIFAR10_CNN_B().to(device)
            elif "cifar" in name and "cnn_c" in name:
                model = CIFAR10_CNN_C().to(device)
            elif "cifar" in name and "convsmall" in name:
                model = CIFAR10_ConvSmall().to(device)
            elif "cifar" in name and "convdeep" in name:
                model = CIFAR10_ConvDeep().to(device)
            elif "cifar" in name:
                model = CIFAR10_ConvLarge().to(device)
            else:
                raise ValueError(
                    f"Could not infer architecture from checkpoint name '{model_path.name}'. "
                    "Please use one of the known SDP-CROWN architectures or adapt the loader."
                )
            model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Unsupported checkpoint format at '{model_path}'.")
    elif model_path.suffix == ".onnx":
        # ONNX frontends are not yet supported in this SDP-CROWN wrapper.
        raise NotImplementedError(
            "Loading ONNX models directly into SDP-CROWN is not supported yet. "
            "Please provide a compatible PyTorch .pth checkpoint."
        )
    else:
        # Fallback to legacy string identifiers (mnist_mlp, cifar10_cnn_a, ...)
        name = str(model_arg).lower()
        if name == "mnist_mlp":
            model = MNIST_MLP().to(device)
        elif name == "mnist_convsmall":
            model = MNIST_ConvSmall().to(device)
        elif name == "mnist_convlarge":
            model = MNIST_ConvLarge().to(device)
        elif name == "cifar10_cnn_a":
            model = CIFAR10_CNN_A().to(device)
        elif name == "cifar10_cnn_b":
            model = CIFAR10_CNN_B().to(device)
        elif name == "cifar10_cnn_c":
            model = CIFAR10_CNN_C().to(device)
        elif name == "cifar10_convsmall":
            model = CIFAR10_ConvSmall().to(device)
        elif name == "cifar10_convdeep":
            model = CIFAR10_ConvDeep().to(device)
        elif name == "cifar10_convlarge":
            model = CIFAR10_ConvLarge().to(device)
        else:
            raise ValueError(f"Unexpected model identifier: {args.model}")

    model.eval()

    # Wrap single image and label into tensors.
    x = torch.from_numpy(image).float().to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    dataset = x  # shape: (1, ...)


    # Infer number of classes from a forward pass.
    with torch.no_grad():
        logits = model(dataset)
    classes = int(logits.shape[-1])

    radius_rescale = float(getattr(args, "radius", 0.0))
    return model, dataset, radius_rescale, classes


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