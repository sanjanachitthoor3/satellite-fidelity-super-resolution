import torch

from uncertainty.estimation import estimate_uncertainty


def dummy_sr(lr):
    """
    Cheap fake SR function for testing.

    It upsamples LR by 4x and applies a simple
    nonlinear transformation.
    """
    return torch.nn.functional.interpolate(
        lr,
        scale_factor=4,
        mode="bilinear",
        align_corners=False,
    )


# --------------------------------------------------
# Test input
# --------------------------------------------------

torch.manual_seed(123)

lr = torch.rand(1, 4, 128, 128)

print("LR input:", lr.shape)


# --------------------------------------------------
# Run uncertainty estimation
# --------------------------------------------------

predictions, variance, uncertainty_map, normalized_map = (
    estimate_uncertainty(
        lr=lr,
        sr_fn=dummy_sr,
        num_samples=5,
        noise_std=0.005,
        seed=42,
    )
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\nSUCCESS!")

print("Predictions:", predictions.shape)
print("Variance:", variance.shape)
print("Uncertainty map:", uncertainty_map.shape)

print(
    "Variance range:",
    variance.min().item(),
    "→",
    variance.max().item(),
)

print(
    "Uncertainty range:",
    uncertainty_map.min().item(),
    "→",
    uncertainty_map.max().item(),
)

print(
    "Normalized range:",
    normalized_map.min().item(),
    "→",
    normalized_map.max().item(),
)


# --------------------------------------------------
# Basic checks
# --------------------------------------------------

assert predictions.shape == (
    5,
    1,
    4,
    512,
    512,
)

assert variance.shape == (
    1,
    4,
    512,
    512,
)

assert uncertainty_map.shape == (
    1,
    512,
    512,
)

assert normalized_map.shape == (
    1,
    512,
    512,
)

assert torch.isfinite(predictions).all()
assert torch.isfinite(variance).all()
assert torch.isfinite(uncertainty_map).all()

assert normalized_map.min() >= 0
assert normalized_map.max() <= 1

print("\nAll uncertainty tests passed!")