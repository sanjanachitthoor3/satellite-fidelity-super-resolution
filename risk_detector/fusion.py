import torch
import torch.nn.functional as F


def load_maps(
    fidelity_path,
    uncertainty_path,
):
    fidelity = torch.load(
        fidelity_path,
        map_location="cpu",
        weights_only=False,
    )

    uncertainty = torch.load(
        uncertainty_path,
        map_location="cpu",
        weights_only=False,
    )

    if fidelity.ndim != 2:
        raise ValueError(
            f"Expected fidelity map [H,W], got {fidelity.shape}"
        )

    if uncertainty.ndim != 2:
        raise ValueError(
            f"Expected uncertainty map [H,W], got {uncertainty.shape}"
        )

    return fidelity.float(), uncertainty.float()


def robust_normalize_p99(
    uncertainty_map,
):
    """
    Normalize uncertainty using the 99th percentile
    instead of the absolute maximum.
    """

    p99 = torch.quantile(
        uncertainty_map.flatten(),
        0.99,
    )

    if p99 <= 0:
        return torch.zeros_like(uncertainty_map)

    return torch.clamp(
        uncertainty_map,
        min=0.0,
        max=p99,
    ) / p99


def align_fidelity_to_sr(
    fidelity_map,
    sr_size=484,
):
    fidelity = fidelity_map.unsqueeze(0).unsqueeze(0)

    fidelity_sr = F.interpolate(
        fidelity,
        size=(sr_size, sr_size),
        mode="bilinear",
        align_corners=False,
    )

    return fidelity_sr.squeeze(0).squeeze(0)


def create_risk_map(
    fidelity_map,
    uncertainty_map,
    fidelity_weight=0.5,
    uncertainty_weight=0.5,
):
    if abs(
        fidelity_weight + uncertainty_weight - 1.0
    ) > 1e-6:
        raise ValueError(
            "Fusion weights must sum to 1."
        )

    # Align fidelity to valid SR region
    fidelity_sr = align_fidelity_to_sr(
        fidelity_map,
        sr_size=484,
    )

    # Use only the matching valid SR region
    uncertainty_valid = uncertainty_map[
        :484,
        :484,
    ]

    if fidelity_sr.shape != uncertainty_valid.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"{fidelity_sr.shape} vs "
            f"{uncertainty_valid.shape}"
        )

    # Robustly normalize uncertainty
    uncertainty_normalized = robust_normalize_p99(
        uncertainty_valid
    )

    # Fuse
    risk = (
        fidelity_weight * fidelity_sr
        +
        uncertainty_weight * uncertainty_normalized
    )

    # Normalize combined risk
    min_value = risk.min()
    max_value = risk.max()

    if torch.isclose(
        min_value,
        max_value,
    ):
        risk_normalized = torch.zeros_like(risk)

    else:
        risk_normalized = (
            risk - min_value
        ) / (
            max_value - min_value
        )

    return (
        fidelity_sr,
        uncertainty_normalized,
        risk,
        risk_normalized,
    )