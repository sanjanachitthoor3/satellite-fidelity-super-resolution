import torch


def perturb_input(
    lr: torch.Tensor,
    noise_std: float = 0.005,
) -> torch.Tensor:
    """
    Add small Gaussian noise to an LR image.

    Args:
        lr: LR tensor [C,H,W] or [B,C,H,W]
        noise_std: Standard deviation of Gaussian noise.

    Returns:
        Perturbed LR tensor with the same shape.
    """

    noise = torch.randn_like(lr) * noise_std

    return lr + noise


def estimate_uncertainty(
    lr: torch.Tensor,
    sr_fn,
    num_samples: int = 5,
    noise_std: float = 0.005,
    seed: int = 42,
):
    """
    Estimate SR prediction uncertainty using input perturbations.

    Args:
        lr:
            Original LR tensor [C,H,W] or [B,C,H,W].

        sr_fn:
            Function that accepts an LR tensor and returns an SR tensor.

        num_samples:
            Number of perturbed SR predictions.

        noise_std:
            Standard deviation of Gaussian input perturbation.

        seed:
            Random seed for reproducibility.

    Returns:
        predictions:
            All SR predictions stacked together.

        variance:
            Pixel-wise variance across predictions.

        uncertainty_map:
            Channel-reduced spatial uncertainty map.

        normalized_map:
            Uncertainty map normalized to [0,1].
    """

    if num_samples < 2:
        raise ValueError(
            "num_samples must be at least 2."
        )

    if not torch.isfinite(lr).all():
        raise ValueError(
            "LR input contains NaN or Inf."
        )

    torch.manual_seed(seed)

    predictions = []

    for i in range(num_samples):

        perturbed_lr = perturb_input(
            lr,
            noise_std=noise_std,
        )

        sr = sr_fn(perturbed_lr)

        if not torch.isfinite(sr).all():
            raise ValueError(
                f"SR prediction {i} contains NaN or Inf."
            )

        predictions.append(sr)

    predictions = torch.stack(
        predictions,
        dim=0,
    )

    # Pixel-wise variance across predictions.
    variance = torch.var(
        predictions,
        dim=0,
        unbiased=False,
    )

    # Reduce channels to obtain a spatial uncertainty map.
    if variance.ndim == 3:
        uncertainty_map = variance.mean(dim=0)

    elif variance.ndim == 4:
        uncertainty_map = variance.mean(dim=1)

    else:
        raise ValueError(
            f"Unexpected variance shape: {variance.shape}"
        )

    # Normalize to [0,1].
    min_value = uncertainty_map.min()
    max_value = uncertainty_map.max()

    if torch.isclose(
        max_value,
        min_value,
    ):
        normalized_map = torch.zeros_like(
            uncertainty_map
        )

    else:
        normalized_map = (
            uncertainty_map - min_value
        ) / (
            max_value - min_value
        )

    return (
        predictions,
        variance,
        uncertainty_map,
        normalized_map,
    )
