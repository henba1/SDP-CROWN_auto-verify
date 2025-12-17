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

def preprocess_cifar(image: np.ndarray, inception_preprocess: bool = False, perturbation: bool = False) -> np.ndarray:
    """
    Preprocess images and perturbations. Preprocessing used by the SDP paper.

    This version supports both channel-last (HWC / NHWC) and channel-first (CHW / NCHW)
    layouts, which allows us to use it directly on the CHW images used in this verifier.
    """
    image = image.astype(np.float32, copy=False)

    means = np.array([125.3, 123.0, 113.9], dtype=np.float32) / 255.0
    std = np.array([0.225, 0.225, 0.225], dtype=np.float32)

    if inception_preprocess:
        # Use 2x - 1 to get [-1, 1]-scaled images
        rescaled_devs = 0.5
        rescaled_means = 0.5
    else:
        if image.ndim == 3:
            # Single image: HWC or CHW
            if image.shape[-1] == 3:
                # HWC
                rescaled_means = means
                rescaled_devs = std
            elif image.shape[0] == 3:
                # CHW
                rescaled_means = means[:, None, None]
                rescaled_devs = std[:, None, None]
            else:
                raise ValueError(f"Unexpected 3D image shape for preprocessing: {image.shape}")
        elif image.ndim == 4:
            # Batched images: NHWC or NCHW
            if image.shape[-1] == 3:
                # NHWC
                rescaled_means = means
                rescaled_devs = std
            elif image.shape[1] == 3:
                # NCHW
                rescaled_means = means[None, :, None, None]
                rescaled_devs = std[None, :, None, None]
            else:
                raise ValueError(f"Unexpected 4D image shape for preprocessing: {image.shape}")
        else:
            raise ValueError(f"preprocess_cifar expects a 3D or 4D array, got ndim={image.ndim}")

    if perturbation:
        return image / rescaled_devs
    return (image - rescaled_means) / rescaled_devs

def load_model_and_dataset(args, device, image: np.ndarray):
    """
    Load a PyTorch model from a checkpoint path and wrap a single image
    instance into tensors usable by SDP-CROWN.

    Args:
        args: Argument namespace, with args.model (path to a .pth checkpoint)
              and args.radius already set.
        device: Torch device.
        image: Numpy array representing a single input instance (flattened or shaped).

    Returns:
        model: nn.Module on the correct device, in eval mode.
        image_tensor: Tensor of shape (1, C, H, W) containing the image.
        radius_rescale: Float radius used for the perturbation.
        classes: Integer number of output classes inferred from the model.
    """

    model_path = Path(args.model)
  
    loaded = torch.load(model_path, map_location=device, weights_only=False)
    model = loaded.to(device)
    model.eval()

    # Process single image. Verona stores CIFAR-10 images in CHW format (C,H,W),
    # while the original SDP-CROWN utilities assumed HWC. To avoid channel
    # mismatches like [1, 32, 3, 32] (seen in conv2d error), we explicitly
    # normalize to (1, 3, 32, 32) here.
    image_arr = image.copy()

    # Handle flattened images.
    if image_arr.ndim == 1:
        if image_arr.size == 3072:  # CIFAR-10: 3*32*32
            # Interpret as CHW: (3, 32, 32)
            image_arr = image_arr.reshape(3, 32, 32)
        else:
            raise ValueError(f"Unexpected flattened image size: {image_arr.size}")

    # Handle 3D images: either CHW (3,32,32) or HWC (32,32,3).
    if image_arr.ndim == 3:
        if image_arr.shape == (3, 32, 32):
            # Already CHW, nothing to do.
            pass
        elif image_arr.shape == (32, 32, 3):
            # HWC -> CHW
            image_arr = np.transpose(image_arr, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected 3D image shape: {image_arr.shape}")

    # At this point, image_arr is in CHW format (3, 32, 32).
    # Normalize it using the same preprocessing as during training
    # (non-inception CIFAR preprocessing).
    image_arr = preprocess_cifar(image_arr, inception_preprocess=False, perturbation=False)

    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    if image_arr.ndim == 3:
        image_arr = image_arr[np.newaxis, ...]

    radius_rescale = args.radius/0.225

    # Convert to tensor; already in (1, C, H, W).
    image_tensor = torch.from_numpy(image_arr).float().to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
    classes = int(logits.shape[-1])

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