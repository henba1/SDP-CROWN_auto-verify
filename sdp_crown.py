import argparse
import datetime
import os
import time

from pathlib import Path

import numpy as np
import torch
import yaml

from models import *
from utils import *
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


def verified_sdp_crown(
    image,
    label,
    model,
    radius,
    device,
    classes,
    args,
):
    """Run SDP-CROWN on the data instance passed by VERONA's epsilon_value_estimator for the specified epsilon."""
    total_time = 0
    verifiction_result = "UNSAT"
    # image is expected to already have batch dimension of 1
    image = image.to(device)
    # label is a scalar tensor; make it batch of size 1
    if label.dim() == 0:
        label = label.unsqueeze(0)
    label = label.to(device)
    norm = 2.0
    method = "CROWN-Optimized"
    C = build_C(label, classes)

    logits_min = None
    logits_max = None
    exact_margins_min = None
    exact_margins_max = None
    exact_margins_vector = None
    with torch.no_grad():
        logits = model(image.to(device))  # shape: (1, K)
        logits_min = logits.min().item()
        logits_max = logits.max().item()
        # Sanity: print basic stats
        print(f"[DEBUG] logits min/max: {logits.min().item():.3e} / {logits.max().item():.3e}")

        # Exact margins via C matrix that SDP-CROWN uses
        exact_margins = torch.bmm(C, logits.unsqueeze(-1)).squeeze(-1)  # shape: (1, K-1)
        print(f"[DEBUG] exact margins (eps=0) min/max: "
            f"{exact_margins.min().item():.3e} / {exact_margins.max().item():.3e}")
        print(f"[DEBUG] exact margins vector: {exact_margins[0].tolist()}")
        exact_margins_min = exact_margins.min().item()
        exact_margins_max = exact_margins.max().item()
        exact_margins_vector = exact_margins[0].tolist()
            
    # Global box constraint for image datasets whose inputs live in [0, 1] (e.g. MNIST, CIFAR-10 JAIR models).
    # This enforces that the L2-ball perturbation never leaves the valid pixel range, i.e., we verify robustness
    # over the intersection of the L2-ball and the [0, 1]^n input domain.
    x_L, x_U = None, None
    box_lower = getattr(args, "input_box_lower", None)
    box_upper = getattr(args, "input_box_upper", None)

    if box_lower is not None and box_upper is not None:
        if box_lower > box_upper:
            raise ValueError(
                f"Invalid input box constraint: lower bound {box_lower} is greater than upper bound {box_upper}."
            )
        x_L = torch.full_like(image, box_lower)
        x_U = torch.full_like(image, box_upper)

    ptb = PerturbationLpNorm(norm=norm, eps=radius, x_U=x_U, x_L=x_L)
    image = BoundedTensor(image, ptb)
    lirpa_model = BoundedModule(model, image, device=image.device, verbose=0)
    lirpa_model.set_bound_opts(
        {
            "optimize_bound_args": {
                "iteration": args.iteration,
                "lr_alpha": args.lr_alpha,
                "early_stop_patience": args.early_stop_patience,
                "fix_interm_bounds": args.fix_interm_bounds,
                "enable_opt_interm_bounds": args.enable_opt_interm_bounds,
                "enable_SDP_crown": args.enable_SDP_crown,
                "lr_lambda": args.lr_lambda,
            }
        }
    )

    # Run SDP-CROWN
    start_time = time.time()

    crown_lb, _ = lirpa_model.compute_bounds(
        x=(image,),
        method=method.split()[0],
        C=C,
        bound_lower=True,
        bound_upper=False,
    )
    end_time = time.time()
    
    # Log GPU memory after processing
    gpu_mem_after = get_gpu_memory_info(device)

    with torch.no_grad():
        if torch.any(crown_lb < 0):
            verifiction_result = "SAT" 

        elapsed_time = end_time - start_time
        total_time += elapsed_time

        # For auto-verify we only need SAT/UNSAT on stdout; we deliberately do not
        # write to the results_file so that no counterexample string is parsed -
        # Keep a local log for manual inspection
        true_label_value = (
            int(label.item()) if isinstance(label, torch.Tensor) else int(label)
        )
        sample_log = {
            "true_label": true_label_value,
            "radius": float(radius),
            "margins": crown_lb.cpu().tolist()[0],
            "verification_result": verifiction_result,
            "elapsed_time": elapsed_time,
            "gpu_mem_after": gpu_mem_after['memory_percent'],
            "logits_min": logits_min,
            "logits_max": logits_max,
        }
        if args.radius == 0:
            sample_log["exact_margins_min"] = exact_margins_min
            sample_log["exact_margins_max"] = exact_margins_max
            sample_log["exact_margins_vector"] = exact_margins_vector
        sample_log_path = args.property_path.with_suffix(".log")
        with open(sample_log_path, "w", encoding="utf-8") as f:
            for key, val in sample_log.items():
                f.write(f"{key}: {val}\n")
    
        # Cleanup GPU memory to prevent mem accumulation
        cleanup_gpu_memory(lirpa_model)
        del lirpa_model, image, label, C, ptb, crown_lb

    # Result line that auto-verify parses for verification result.
    print(f"Result: {verifiction_result.lower()}")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or identifier of the network",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the SDP-CROWN YAML configuration file",
    )
    parser.add_argument(
        "--vnnlib_property",
        type=str,
        required=True,
        help="Path to the VNNLIB property file for the instance to verify",
    )

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load SDP-CROWN parameters and log path from YAML config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    sdp_cfg = yaml.safe_load(config_path.read_text()) or {}

    opt_bounds_cfg = sdp_cfg.get("optimize_bound_args", {})
    args.iteration = int(opt_bounds_cfg.get("iteration", 300))
    args.lr_alpha = float(opt_bounds_cfg.get("lr_alpha", 0.5))
    args.lr_lambda = float(opt_bounds_cfg.get("lr_lambda", 0.05))
    args.early_stop_patience = int(opt_bounds_cfg.get("early_stop_patience", 20))
    args.fix_interm_bounds = bool(opt_bounds_cfg.get("fix_interm_bounds", False))
    args.enable_opt_interm_bounds = bool(opt_bounds_cfg.get("enable_opt_interm_bounds", True))
    args.enable_SDP_crown = bool(opt_bounds_cfg.get("enable_SDP_crown", True))


    box_cfg = sdp_cfg.get("input_box_constraint", None)
    if box_cfg is None:
        args.input_box_lower = None
        args.input_box_upper = None
    else:
        if not (isinstance(box_cfg, (list, tuple)) and len(box_cfg) == 2):
            raise ValueError(
                "SDP-CROWN config error: 'input_box_constraint' must be a list or tuple of length 2: "
                "[lower_bound, upper_bound]."
            )
        args.input_box_lower = float(box_cfg[0])
        args.input_box_upper = float(box_cfg[1])


    property_path = Path(args.vnnlib_property)
    args.property_path = property_path
    
    meta_path = property_path.with_suffix(".npz")
    if meta_path.exists():
        data = np.load(meta_path)
        epsilon = float(data.get("epsilon", -1.0))

        image_np = data["image"]

        try:
            image_class = int(data.get("image_class", -1))
        except Exception as e:
            raise KeyError("SDP-CROWN VNNLib parsing error: Failed to load 'image_class' from npz file") from e

        if epsilon < 0.0:
            raise ValueError("SDP-CROWN VNNLib parsing error: Epsilon / radius must be non-negative")

        args.radius = epsilon

        if image_class == -1:
            raise ValueError(
                "SDP-CROWN VNNLib parsing error: image_class must be provided for SDP-CROWN "
                "and be non-negative in the metadata .npz file"
            )
        label_int = image_class
        image_range = image_np.max() - image_np.min()

    else:
        raise FileNotFoundError(
            f"SDP-CROWN VNNLib parsing error: Metadata sidecar not found for property: {property_path}"
        )

    model, image_tensor, radius_rescale, classes = load_model_and_dataset(
        args, device, image=image_np
    )
    
    label_tensor = torch.tensor(label_int, dtype=torch.long, device=device)

    verified_sdp_crown(
        image=image_tensor,
        label=label_tensor,
        model=model,
        radius=radius_rescale,
        device=device,
        classes=classes,
        args=args,
    )