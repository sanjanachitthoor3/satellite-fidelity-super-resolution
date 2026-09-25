import torch
import torch.nn.functional as F


def prepare_lr_input(lr: torch.Tensor, target_size: int = 128) -> torch.Tensor:
    """
    Prepare an OpenSR-degraded LR tensor for LDSR-S2.

    Input:
        [C, H, W] or [B, C, H, W]

    Output:
        [C, target_size, target_size]
        or
        [B, C, target_size, target_size]

    For our current prototype:
        4 × 121 × 121 -> 4 × 128 × 128

    Padding is used rather than interpolation so that we do not
    alter the spatial sampling of the degraded observation.
    """

    if lr.ndim == 3:
        lr = lr.unsqueeze(0)
        squeeze_batch = True
    elif lr.ndim == 4:
        squeeze_batch = False
    else:
        raise ValueError(
            f"Expected [C,H,W] or [B,C,H,W], got {lr.shape}"
        )

    _, _, height, width = lr.shape

    if height > target_size or width > target_size:
        raise ValueError(
            f"Input {height}x{width} is larger than "
            f"target size {target_size}x{target_size}"
        )

    pad_height = target_size - height
    pad_width = target_size - width

    lr = F.pad(
        lr,
        (0, pad_width, 0, pad_height),
        mode="reflect"
    )

    if squeeze_batch:
        lr = lr.squeeze(0)

    return lr


def compute_fidelity_error(
    lr: torch.Tensor,
    lr_prime: torch.Tensor
) -> torch.Tensor:
    """
    Compute pixel-wise fidelity error between LR and LR'.

    Both tensors must have matching dimensions.

    Returns:
        Per-channel absolute error with the same shape as the inputs.
    """

    if lr.shape != lr_prime.shape:
        raise ValueError(
            f"Shape mismatch: LR={lr.shape}, "
            f"LR'={lr_prime.shape}"
        )

    if not torch.isfinite(lr).all():
        raise ValueError("LR contains NaN or Inf values.")

    if not torch.isfinite(lr_prime).all():
        raise ValueError("LR' contains NaN or Inf values.")

    return torch.abs(lr - lr_prime)


def spatial_fidelity_map(
    error: torch.Tensor
) -> torch.Tensor:
    """
    Reduce channel-wise fidelity error into a spatial map.

    For:
        [C,H,W] -> [H,W]
        [B,C,H,W] -> [B,H,W]

    Channel reduction uses mean absolute error.
    """

    if error.ndim == 3:
        return error.mean(dim=0)

    if error.ndim == 4:
        return error.mean(dim=1)

    raise ValueError(
        f"Expected [C,H,W] or [B,C,H,W], got {error.shape}"
    )


def normalize_fidelity_map(
    fidelity_map: torch.Tensor
) -> torch.Tensor:
    """
    Normalize fidelity error to [0,1].

    Handles constant maps without producing NaNs.
    """

    min_value = fidelity_map.min()
    max_value = fidelity_map.max()

    if torch.isclose(max_value, min_value):
        return torch.zeros_like(fidelity_map)

    return (fidelity_map - min_value) / (
        max_value - min_value
    )

def compute_fidelity_from_sr(
    lr_input: torch.Tensor,
    sr: torch.Tensor,
    degradation_model,
    lr_prime_index: int = 2,
):
    """
    Compute LR-SR-LR fidelity error.

    lr_input:
        [C,H,W] or [1,C,H,W]
        The exact LR tensor given to LDSR-S2.

    sr:
        [C,H,W] or [1,C,H,W]
        LDSR-S2 super-resolved output.

    Returns:
        lr_prime: degraded SR image [C,H,W]
        error: per-channel absolute error [C,H,W]
        fidelity_map: spatial error map [H,W]
        normalized_map: normalized fidelity map [H,W]
    """

    from degradation.opensr_degradation import degrade_sr_to_lr

    # Degrade SR back to LR
    lr_prime_all = degrade_sr_to_lr(
        sr,
        degradation_model,
    )

    # OpenSR returns 5 degradation variants
    if lr_prime_all.ndim != 4:
        raise ValueError(
            f"Expected 5 LR variants [N,C,H,W], got {lr_prime_all.shape}"
        )

    if lr_prime_index >= lr_prime_all.shape[0]:
        raise ValueError(
            f"Invalid LR prime index {lr_prime_index}"
        )

    lr_prime = lr_prime_all[lr_prime_index]

    # Make sure LR input has no batch dimension
    if lr_input.ndim == 4:
        lr_input = lr_input.squeeze(0)

    if lr_input.shape != lr_prime.shape:
        raise ValueError(
            f"Shape mismatch: LR input {lr_input.shape}, "
            f"LR prime {lr_prime.shape}"
        )

    # Pixel-wise fidelity error
    error = compute_fidelity_error(
        lr_input,
        lr_prime,
    )

    # Reduce channels -> spatial map
    fidelity_map = spatial_fidelity_map(error)

    # Normalize to [0,1]
    normalized_map = normalize_fidelity_map(
        fidelity_map
    )

    return (
        lr_prime,
        error,
        fidelity_map,
        normalized_map,
    )