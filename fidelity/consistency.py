import torch

def reconstruct_lr_from_sr(sr, degradation_fn):
    """
    Reconstruct LR' from a super-resolved image.

    Parameters
    ----------
    sr : torch.Tensor
        Super-resolved image. Supports CHW or BCHW format.
    degradation_fn : callable
        The SAME degradation pipeline used to generate the original LR.

    Returns
    -------
    torch.Tensor
        Reconstructed LR image (LR').
    """
    return degradation_fn(sr)


def l1_error_map(lr, reconstructed_lr):
    """
    Per-pixel absolute error between LR and reconstructed LR'.
    Works with any number of channels.
    """
    if lr.shape != reconstructed_lr.shape:
        raise ValueError(
            f"Shape mismatch: LR {lr.shape} vs LR' {reconstructed_lr.shape}"
        )

    return torch.abs(lr - reconstructed_lr)


def mse_error_map(lr, reconstructed_lr):
    """
    Per-pixel squared error between LR and reconstructed LR'.
    Works with any number of channels.
    """
    if lr.shape != reconstructed_lr.shape:
        raise ValueError(
            f"Shape mismatch: LR {lr.shape} vs LR' {reconstructed_lr.shape}"
        )

    return (lr - reconstructed_lr) ** 2


def channel_reduce(error_map, reduction="mean"):
    """
    Convert a channel-wise error map into a spatial error map.

    Input:
        BCHW -> BHW
        CHW  -> HW

    reduction:
        'mean' or 'max'
    """
    if error_map.ndim not in (3, 4):
        raise ValueError(
            f"Expected CHW or BCHW tensor, got shape {error_map.shape}"
        )

    channel_dim = 1 if error_map.ndim == 4 else 0

    if reduction == "mean":
        return error_map.mean(dim=channel_dim)

    if reduction == "max":
        return error_map.max(dim=channel_dim).values

    raise ValueError("reduction must be 'mean' or 'max'")


def consistency_score(lr, reconstructed_lr, metric="l1"):
    """
    Calculate a single global LR-LR' consistency error.

    Lower = more consistent.
    Higher = less consistent.
    """
    if lr.shape != reconstructed_lr.shape:
        raise ValueError(
            f"Shape mismatch: LR {lr.shape} vs LR' {reconstructed_lr.shape}"
        )

    if metric == "l1":
        return torch.mean(torch.abs(lr - reconstructed_lr))

    if metric == "mse":
        return torch.mean((lr - reconstructed_lr) ** 2)

    raise ValueError("metric must be 'l1' or 'mse'")


def fidelity_check(lr, sr, degradation_fn, metric="l1"):
    """
    Complete LR -> SR -> LR' consistency check.

    Parameters
    ----------
    lr : torch.Tensor
        Original low-resolution image.

    sr : torch.Tensor
        Super-resolved image.

    degradation_fn : callable
        Same degradation used to create LR from HR.

    metric : str
        'l1' or 'mse'.

    Returns
    -------
    dict
        Contains reconstructed LR', error map, spatial map,
        and global consistency score.
    """
    reconstructed_lr = reconstruct_lr_from_sr(sr, degradation_fn)

    error_map = (
        l1_error_map(lr, reconstructed_lr)
        if metric == "l1"
        else mse_error_map(lr, reconstructed_lr)
    )

    spatial_error_map = channel_reduce(error_map, reduction="mean")

    score = consistency_score(
        lr,
        reconstructed_lr,
        metric=metric,
    )

    return {
        "reconstructed_lr": reconstructed_lr,
        "error_map": error_map,
        "spatial_error_map": spatial_error_map,
        "score": score,
    }