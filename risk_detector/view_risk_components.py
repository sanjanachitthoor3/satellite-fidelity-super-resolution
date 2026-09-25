import os

import torch
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

OUTPUT_DIR = "outputs/risk"

FIDELITY_PATH = os.path.join(
    OUTPUT_DIR,
    "fidelity_aligned_sr.pt",
)

UNCERTAINTY_PATH = os.path.join(
    OUTPUT_DIR,
    "uncertainty_aligned_sr.pt",
)

RISK_PATH = os.path.join(
    OUTPUT_DIR,
    "risk_map_normalized.pt",
)


# ============================================================
# Load maps
# ============================================================

fidelity = torch.load(
    FIDELITY_PATH,
    map_location="cpu",
    weights_only=False,
)

uncertainty = torch.load(
    UNCERTAINTY_PATH,
    map_location="cpu",
    weights_only=False,
)

risk = torch.load(
    RISK_PATH,
    map_location="cpu",
    weights_only=False,
)


# ============================================================
# Validate
# ============================================================

print("Fidelity:", fidelity.shape)
print("Uncertainty:", uncertainty.shape)
print("Risk:", risk.shape)


if not (
    fidelity.shape
    == uncertainty.shape
    == risk.shape
):
    raise ValueError(
        "Risk components do not have matching shapes."
    )


# ============================================================
# Print statistics
# ============================================================

print()
print("========================================")
print("COMPONENT STATISTICS")
print("========================================")

print(
    f"Fidelity     mean: "
    f"{fidelity.mean().item():.6f}"
)

print(
    f"Uncertainty  mean: "
    f"{uncertainty.mean().item():.6f}"
)

print(
    f"Risk         mean: "
    f"{risk.mean().item():.6f}"
)


# ============================================================
# Create diagnostic visualization
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6),
)


# ------------------------------------------------------------
# Fidelity
# ------------------------------------------------------------

im1 = axes[0].imshow(
    fidelity.numpy(),
    cmap="inferno",
    vmin=0,
    vmax=1,
)

axes[0].set_title(
    "Fidelity Error"
)

axes[0].axis("off")

fig.colorbar(
    im1,
    ax=axes[0],
    fraction=0.046,
    pad=0.04,
)


# ------------------------------------------------------------
# Excess uncertainty
# ------------------------------------------------------------

im2 = axes[1].imshow(
    uncertainty.numpy(),
    cmap="inferno",
    vmin=0,
    vmax=1,
)

axes[1].set_title(
    "Excess Uncertainty"
)

axes[1].axis("off")

fig.colorbar(
    im2,
    ax=axes[1],
    fraction=0.046,
    pad=0.04,
)


# ------------------------------------------------------------
# Combined risk
# ------------------------------------------------------------

im3 = axes[2].imshow(
    risk.numpy(),
    cmap="inferno",
    vmin=0,
    vmax=1,
)

axes[2].set_title(
    "Combined Risk"
)

axes[2].axis("off")

fig.colorbar(
    im3,
    ax=axes[2],
    fraction=0.046,
    pad=0.04,
)


# ============================================================
# Save
# ============================================================

plt.suptitle(
    "Hallucination-Risk Components",
    fontsize=16,
)

plt.tight_layout()

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "risk_components.png",
)

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


print()
print("========================================")
print("SUCCESS!")
print("========================================")

print(
    f"Saved diagnostic to: {OUTPUT_PATH}"
)