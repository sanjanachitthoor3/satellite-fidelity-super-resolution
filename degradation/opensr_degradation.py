import torch
import opensr_degradation


def create_naip_degradation_model(device="cpu", seed=42):
    """
    Create the same OpenSR NAIP-D degradation pipeline
    used to generate the original LR images.
    """

    return opensr_degradation.pipe(
        sensor="naip_d",
        add_noise=True,
        params={
            "reflectance_method": [
                "identity",
                "gamma_lognormal",
                "gamma_multivariate_normal",
                "unet_histogram_matching",
                "vae_histogram_matching",
            ],
            "noise_method": "gaussian_noise",
            "device": device,
            "seed": seed,
            "percentiles": [50],
            "vae_reflectance_model": opensr_degradation.naip_vae_model(device),
            "unet_reflectance_model": opensr_degradation.naip_unet_model(device),
        },
    )


def degrade_sr_to_lr(sr, degradation_model):
    """
    Degrade an SR image back to LR using OpenSR.

    Input:
        sr: [C,H,W] or [B,C,H,W]

    Returns:
        lr_prime: [C,H/4,W/4]
    """

    if sr.ndim == 4:
        if sr.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1, got {sr.shape}"
            )
        sr = sr.squeeze(0)

    if sr.ndim != 3:
        raise ValueError(
            f"Expected SR shape [C,H,W] or [1,C,H,W], got {sr.shape}"
        )

    sr = sr.float()

    lr_prime, _ = degradation_model.forward(sr)

    if lr_prime.ndim == 4:
        lr_prime = lr_prime.squeeze(0)

    return lr_prime