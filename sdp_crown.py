import comet_ml
import os
import gc
import torch
import time
import argparse
import datetime
from models import *
from utils import *
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


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
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # Convert to GB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # Convert to GB
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # Convert to GB
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
    
def verified_sdp_crown(
    dataset, labels, model, radius, clean_output, device, classes, args, experiment=None
):
    samples = dataset.shape[0]
    verification_fail = samples - len(clean_output)
    verification_fail_idx = []
    total_time = 0
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'./logs/sdp_crown/{args.model.lower()}/{args.radius}/{timestamp}'
    os.makedirs(log_dir, exist_ok=True)

    sample_count = 0
    for idx, (image, label) in enumerate(zip(dataset, labels)):
        if idx not in clean_output:
            continue
        sample_idx = args.start + idx
        verifiction_status = "Success"
        
        # Log GPU memory before processing
        gpu_mem_before = get_gpu_memory_info(device)
        
        image = image.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        norm = 2.0
        method = 'CROWN-Optimized'
        C = build_C(label, classes)
        x_L, x_U = None, None
        if "mnist" in args.model.lower():
            x_U = torch.ones_like(image)
            x_L = torch.zeros_like(image)
          
        ptb = PerturbationLpNorm(norm=norm, eps=radius, x_U=x_U, x_L=x_L)
        image = BoundedTensor(image, ptb)
        lirpa_model = BoundedModule(model, image, device=image.device, verbose=0)
        lirpa_model.set_bound_opts({
            'optimize_bound_args': {
                'iteration': 300,
                'lr_alpha': args.lr_alpha,
                'early_stop_patience': 20,
                'fix_interm_bounds': False,
                'enable_opt_interm_bounds': True,
                'enable_SDP_crown': True,
                'lr_lambda': args.lr_lambda,
            }
        })

        # Run SDP-CROWN
        start_time = time.time()
        crown_lb, _ = lirpa_model.compute_bounds(
            x=(image,), method=method.split()[0], C=C, bound_lower=True, bound_upper=False
        )
        end_time = time.time()

        # Log GPU memory after processing
        gpu_mem_after = get_gpu_memory_info(device)

        with torch.no_grad():
            if torch.any(crown_lb < 0):
                verification_fail += 1
                verifiction_status = "Fail"
                verification_fail_idx.append(sample_idx)
            
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            sample_log = {
                'sample_idx': sample_idx,
                'true_label': label.item() if isinstance(label, torch.Tensor) else label,
                'margins': crown_lb.cpu().tolist()[0],     
                'verifiction_status': verifiction_status,
                'elapsed_time': elapsed_time,
            }
            sample_log_path = f'{log_dir}/sample_{sample_idx}.log'
            with open(sample_log_path, "w", encoding='utf-8') as f:
                for key, val in sample_log.items():
                    f.write(f"{key}: {val}\n")
            
            # Log sample file to Comet ML
            if experiment:
                experiment.log_asset(sample_log_path) 
            
            # Log to Comet ML if experiment is available
            if experiment:
                experiment.log_metrics({
                    "sample_index": sample_idx,
                    "verification_status": 1 if verifiction_status == "Success" else 0,
                    "elapsed_time_seconds": elapsed_time,
                    "margin_value": crown_lb.cpu().tolist()[0][0] if isinstance(crown_lb.cpu().tolist()[0], list) else crown_lb.cpu().tolist()[0],
                    "gpu_memory_allocated_gb_before": gpu_mem_before["memory_allocated_gb"],
                    "gpu_memory_percent_before": gpu_mem_before["memory_percent"],
                    "gpu_memory_allocated_gb_after": gpu_mem_after["memory_allocated_gb"],
                    "gpu_memory_percent_after": gpu_mem_after["memory_percent"],
                    "gpu_memory_peak_gb": max(gpu_mem_before["memory_allocated_gb"], gpu_mem_after["memory_allocated_gb"]),
                }, step=sample_count)
                
                experiment.log_other(f"sample_{sample_idx}_result", {
                    "index": sample_idx,
                    "true_label": label.item() if isinstance(label, torch.Tensor) else label,
                    "verification_status": verifiction_status,
                    "margins": crown_lb.cpu().tolist()[0],
                    "elapsed_time": elapsed_time,
                    "gpu_memory_allocated_gb": gpu_mem_after["memory_allocated_gb"],
                    "gpu_memory_reserved_gb": gpu_mem_after["memory_reserved_gb"],
                })
            
            print(
                f'Sample {sample_idx}, verifiction_status: {verifiction_status}, '
                f'elapsed_time: {elapsed_time}s, gpu_mem: {gpu_mem_after["memory_percent"]:.1f}%'
            )
            sample_count += 1
            
            # Cleanup GPU memory between samples to prevent accumulation
            cleanup_gpu_memory(lirpa_model)
            del lirpa_model, image, label, C, ptb, crown_lb
    
    verified_accuracy = (samples - verification_fail) / samples * 100
    average_time = total_time / len(clean_output)
    final_log = {
        'verification_fail_idx': verification_fail_idx,
        'verification_fail': verification_fail,
        'verified_accuracy': verified_accuracy,
        'average_time': average_time,
    }
    final_log_path = f'{log_dir}/final_results.log'
    with open(final_log_path, "w", encoding='utf-8') as f:
        for key, val in final_log.items():
            f.write(f"{key}: {val}\n")
    
    # Log final results file to Comet ML
    if experiment:
        experiment.log_asset(final_log_path)
    
    # Log final results to Comet ML
    if experiment:
        final_gpu_mem = get_gpu_memory_info(device)
        experiment.log_metrics({
            "final_accuracy": verified_accuracy,
            "total_samples_processed": sample_count,
            "total_verification_fails": verification_fail,
            "average_time_per_sample": average_time,
            "final_gpu_memory_percent": final_gpu_mem["memory_percent"],
        })
        
        experiment.log_other("experiment_summary", {
            "total_samples": samples,
            "samples_processed": sample_count,
            "verification_fails": verification_fail,
            "verified_accuracy": verified_accuracy,
            "average_time_per_sample": average_time,
            "gpu_device": torch.cuda.get_device_name(device),
            "gpu_total_memory_gb": final_gpu_mem["total_memory_gb"],
        })
    
    print(
        f'Total Verification Fail: {verification_fail}, verified_accuracy: {verified_accuracy}%, '
        f'average_time: {average_time}s'
    )
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', default=1, type=parse_float_or_fraction, help='L2 norm perturbation')
    parser.add_argument('--lr_alpha', default=0.5, type=float, help='alpha learning rate')
    parser.add_argument('--lr_lambda', default=0.05, type=float, help='lambda learning rate')
    parser.add_argument('--start', default=0, type=int, help='start index for the dataset')
    parser.add_argument('--end', default=200, type=int, help='end index for the dataset')
    parser.add_argument('--model', default='mnist_mlp',
    choices=[
        'mnist_mlp',
        'mnist_convsmall',
        'mnist_convlarge',
        'cifar10_cnn_a',
        'cifar10_cnn_b',
        'cifar10_cnn_c',
        'cifar10_convsmall',
        'cifar10_convdeep',
        'cifar10_convlarge',
        ])
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, dataset, labels, radius_rescale, classes = load_model_and_dataset(args, device)

    # Create Comet ML experiment
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"SDP-CROWN_{args.model}_{args.radius}_{timestamp}"
    
    comet_ml.login()
    
    experiment_config = comet_ml.ExperimentConfig(
        name=experiment_name, tags=["SDP-CROWN-verification"]
    )
    experiment = comet_ml.start(
        project_name="rs-rd",
        experiment_config=experiment_config,
    )
    
    
    # Log experiment parameters
    experiment.log_parameters({
        "model": args.model,
        "radius": float(radius_rescale),
        "lr_alpha": args.lr_alpha,
        "lr_lambda": args.lr_lambda,
        "start": args.start,
        "end": args.end,
        "num_samples": args.end - args.start,
        "gpu_device": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU",
    })

    # Run original model for clean accuracy.
    with torch.no_grad():
        labels_tensor = labels.to(device)
        dataset_tensor = dataset.to(device)
        output = model(dataset_tensor)
        clean_output = torch.sum((output.max(1)[1] == labels_tensor).float()).cpu()
        predictions = output.argmax(dim=1)
        correct_indices = (predictions == labels_tensor).nonzero(as_tuple=True)[0]
    
    clean_accuracy = clean_output / (args.end - args.start) * 100
    print(f'perturbation: {radius_rescale}')
    print(f'The clean output for the {args.end-args.start} samples is {clean_accuracy}%')
    
    experiment.log_metrics({
        "clean_accuracy": clean_accuracy,
        "correctly_classified_samples": len(correct_indices),
    })
    
    verified_sdp_crown(
        dataset=dataset,
        labels=labels,
        model=model,
        radius=radius_rescale,
        clean_output=correct_indices,
        device=device,
        classes=classes,
        args=args,
        experiment=experiment,
    )
    
    experiment.end()
    print(f"Experiment completed. View results at: {experiment.url}")
