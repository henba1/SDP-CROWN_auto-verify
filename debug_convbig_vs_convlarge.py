import torch

from models import CIFAR10_ConvLarge, CONV_BIG


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def summarize_model(model: torch.nn.Module, name: str) -> None:
    print(f"\n=== {name} summary ===")
    print(f"Total parameters: {count_parameters(model):,}")
    print("Layers and weight shapes:")
    for param_name, param in model.named_parameters():
        shape = "x".join(str(d) for d in param.shape)
        print(f"  {param_name:30s}  shape=({shape})  requires_grad={param.requires_grad}")


def compare_state_dicts(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    reference_name: str,
    candidate_name: str,
) -> None:
    print(f"\n=== Comparing state_dict: {candidate_name} vs {reference_name} ===")

    ref_keys = set(reference.keys())
    cand_keys = set(candidate.keys())

    missing_in_candidate = sorted(ref_keys - cand_keys)
    extra_in_candidate = sorted(cand_keys - ref_keys)

    if missing_in_candidate:
        print(f"Missing in {candidate_name} (present in {reference_name}):")
        for k in missing_in_candidate:
            print(f"  {k}")
    else:
        print(f"No keys missing in {candidate_name} compared to {reference_name}.")

    if extra_in_candidate:
        print(f"Extra in {candidate_name} (not in {reference_name}):")
        for k in extra_in_candidate:
            print(f"  {k}")
    else:
        print(f"No extra keys in {candidate_name} compared to {reference_name}.")

    print("Shape mismatches for common keys:")
    for k in sorted(ref_keys & cand_keys):
        ref_shape = tuple(reference[k].shape)
        cand_shape = tuple(candidate[k].shape)
        if ref_shape != cand_shape:
            print(f"  {k}: {candidate_name} {cand_shape} vs {reference_name} {ref_shape}")


def forward_debug(model: torch.nn.Module, name: str, device: torch.device) -> None:
    model.eval()
    x = torch.randn(1, 3, 32, 32, device=device)
    with torch.no_grad():
        logits = model(x)
    print(f"\n=== Forward pass for {name} on random input ===")
    print(f"Output shape: {tuple(logits.shape)}")
    print(
        f"logits min/max: {logits.min().item():.4f} / {logits.max().item():.4f}, "
        f"mean: {logits.mean().item():.4f}"
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CIFAR10_ConvLarge and its checkpoint (original SDP-CROWN model).
    conv_large = CIFAR10_ConvLarge().to(device)
    conv_large_ckpt = torch.load("./models/cifar10_convlarge.pth", map_location=device)
    conv_large.load_state_dict(conv_large_ckpt)

    loaded = torch.load("./models/conv_big_best.pth", map_location=device, weights_only=False)
    if isinstance(loaded, dict):
        # Checkpoint is a pure state_dict
        conv_big = CONV_BIG().to(device)
        conv_big.load_state_dict(loaded)
        conv_big_ckpt = loaded
    else:
        # Checkpoint is a full model instance (e.g. adversarial_training_box.models.conv_big.CONV_BIG)
        conv_big = loaded.to(device)
        conv_big_ckpt = conv_big.state_dict()

    
    # Summaries.
    summarize_model(conv_large, "CIFAR10_ConvLarge")
    summarize_model(conv_big, "CONV_BIG")

    # Compare checkpoints directly (same keys and shapes?).
    compare_state_dicts(
        reference=conv_big.state_dict(),
        candidate=conv_big_ckpt,
        reference_name="CONV_BIG (architecture)",
        candidate_name="conv_big_best.pth (checkpoint)",
    )

    # Quick sanity check: forward pass on random input for both models.
    forward_debug(conv_large, "CIFAR10_ConvLarge", device)
    forward_debug(conv_big, "CONV_BIG", device)


if __name__ == "__main__":
    main()


