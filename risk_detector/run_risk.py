import os
import sys

import torch
import matplotlib.pyplot as plt


# ============================================================
# Make project root importable
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


from risk_detector.fusion import (
    load_maps,
    create_risk_map,
)


# ============================================================
# Paths
# ============================================================

FIDELITY_PATH = (
    "outputs/fidelity/"
    "fidelity_map_normalized_valid.pt"
)

UNCERTAINTY_PATH = (
    "outputs/uncertainty/"
    "excess_uncertainty_normalized.pt"
)

OUTPUT_DIR = "outputs/risk"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# Load maps
# ============================================================

print("Loading fidelity map...")

fidelity, uncertainty = load_maps(
    FIDELITY_PATH,
    UNCERTAINTY_PATH,
)

print(
    f"Fidelity shape: {fidelity.shape}"
)

print(
    f"Uncertainty shape: {uncertainty.shape}"
)


# ============================================================
# Create combined risk map
# ============================================================

print()
print("Creating combined risk map...")

(
    fidelity_sr,
    uncertainty_valid,
    risk,
    risk_normalized,
) = create_risk_map(
    fidelity,
    uncertainty,
    fidelity_weight=0.5,
    uncertainty_weight=0.5,
)


# ============================================================
# Statistics
# ============================================================

flat = risk.flatten()

print()
print("========================================")
print("RISK MAP STATISTICS")
print("========================================")

print(
    f"Mean   : {flat.mean().item():.6f}"
)

print(
    f"Median : {flat.median().item():.6f}"
)

print(
    f"P90    : "
    f"{torch.quantile(flat, 0.90).item():.6f}"
)

print(
    f"P95    : "
    f"{torch.quantile(flat, 0.95).item():.6f}"
)

print(
    f"P99    : "
    f"{torch.quantile(flat, 0.99).item():.6f}"
)


# ============================================================
# Save tensors
# ============================================================

torch.save(
    fidelity_sr,
    os.path.join(
        OUTPUT_DIR,
        "fidelity_aligned_sr.pt",
    ),
)

torch.save(
    uncertainty_valid,
    os.path.join(
        OUTPUT_DIR,
        "uncertainty_aligned_sr.pt",
    ),
)

torch.save(
    risk,
    os.path.join(
        OUTPUT_DIR,
        "risk_map.pt",
    ),
)

torch.save(
    risk_normalized,
    os.path.join(
        OUTPUT_DIR,
        "risk_map_normalized.pt",
    ),
)


# ============================================================
# P99 visualization
# ============================================================

p99 = torch.quantile(
    risk.flatten(),
    0.99,
)

if p99 > 0:

    visual_map = (
        torch.clamp(
            risk,
            max=p99,
        )
        / p99
    )

else:

    visual_map = torch.zeros_like(
        risk
    )


plt.figure(
    figsize=(8, 8)
)

plt.imshow(
    visual_map.numpy(),
    cmap="inferno",
)

plt.colorbar(
    label="Combined risk (P99-clipped)"
)

plt.title(
    "Fidelity + Uncertainty Risk Map"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "risk_map.png",
    ),
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Done
# ============================================================

print()
print("========================================")
print("SUCCESS!")
print("========================================")

print(
    "Risk map saved to: "
    "outputs/risk/risk_map.png"
)