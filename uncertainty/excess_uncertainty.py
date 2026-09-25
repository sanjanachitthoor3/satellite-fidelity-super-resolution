import os

import torch
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PERTURBED_VARIANCE_PATH = (
    "outputs/uncertainty/variance.pt"
)

CONTROL_VARIANCE_PATH = (
    "outputs/uncertainty/control_variance.pt"
)

OUTPUT_DIR = "outputs/uncertainty"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Load variance maps
# ============================================================

print("Loading perturbed variance...")

perturbed_variance = torch.load(
    PERTURBED_VARIANCE_PATH,
    map_location="cpu",
    weights_only=False
)

print(
    f"Perturbed variance shape: "
    f"{perturbed_variance.shape}"
)


print("Loading control variance...")

control_variance = torch.load(
    CONTROL_VARIANCE_PATH,
    map_location="cpu",
    weights_only=False
)

print(
    f"Control variance shape: "
    f"{control_variance.shape}"
)


# ============================================================
# Validate
# ============================================================

if perturbed_variance.shape != control_variance.shape:
    raise ValueError(
        "Perturbed and control variance shapes do not match."
    )

if not torch.isfinite(perturbed_variance).all():
    raise ValueError(
        "Perturbed variance contains NaN or Inf."
    )

if not torch.isfinite(control_variance).all():
    raise ValueError(
        "Control variance contains NaN or Inf."
    )


# ============================================================
# Convert 4-channel variance to spatial maps
# ============================================================

# Shape:
#
# [1, 4, 512, 512]
#
# → [512, 512]

perturbed_map = perturbed_variance.mean(
    dim=1
).squeeze(0)

control_map = control_variance.mean(
    dim=1
).squeeze(0)


# ============================================================
# Compute excess uncertainty
# ============================================================

print()
print("Computing excess uncertainty...")

excess_uncertainty = (
    perturbed_map - control_map
)

# Negative values mean the perturbed run was
# actually less variable than the baseline.
#
# We only want the additional uncertainty caused
# by LR perturbation.

excess_uncertainty = torch.clamp(
    excess_uncertainty,
    min=0.0
)


# ============================================================
# Statistics
# ============================================================

flat = excess_uncertainty.flatten()

mean_value = flat.mean().item()
median_value = flat.median().item()

p90 = torch.quantile(
    flat,
    0.90
).item()

p95 = torch.quantile(
    flat,
    0.95
).item()

p99 = torch.quantile(
    flat,
    0.99
).item()


print()
print("========================================")
print("EXCESS UNCERTAINTY STATISTICS")
print("========================================")

print(
    f"Mean   : {mean_value:.8f}"
)

print(
    f"Median : {median_value:.8f}"
)

print(
    f"P90    : {p90:.8f}"
)

print(
    f"P95    : {p95:.8f}"
)

print(
    f"P99    : {p99:.8f}"
)


# ============================================================
# Normalize to 0-1
# ============================================================

min_value = excess_uncertainty.min()
max_value = excess_uncertainty.max()

if torch.isclose(
    max_value,
    min_value
):

    normalized_map = torch.zeros_like(
        excess_uncertainty
    )

else:

    normalized_map = (
        excess_uncertainty - min_value
    ) / (
        max_value - min_value
    )


# ============================================================
# Save tensors
# ============================================================

torch.save(
    excess_uncertainty,
    os.path.join(
        OUTPUT_DIR,
        "excess_uncertainty.pt"
    )
)

torch.save(
    normalized_map,
    os.path.join(
        OUTPUT_DIR,
        "excess_uncertainty_normalized.pt"
    )
)


# ============================================================
# P99-clipped visualization
# ============================================================

p99_tensor = torch.quantile(
    excess_uncertainty.flatten(),
    0.99
)

if p99_tensor > 0:

    visual_map = torch.clamp(
        excess_uncertainty,
        max=p99_tensor
    ) / p99_tensor

else:

    visual_map = torch.zeros_like(
        excess_uncertainty
    )


plt.figure(
    figsize=(8, 8)
)

plt.imshow(
    visual_map.numpy(),
    cmap="inferno"
)

plt.colorbar(
    label="Excess uncertainty (P99-clipped)"
)

plt.title(
    "LDSR-S2 Excess Prediction Uncertainty"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "excess_uncertainty.png"
    ),
    dpi=200,
    bbox_inches="tight"
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
    "Saved: "
    "outputs/uncertainty/excess_uncertainty.pt"
)

print(
    "Saved: "
    "outputs/uncertainty/excess_uncertainty_normalized.pt"
)

print(
    "Saved: "
    "outputs/uncertainty/excess_uncertainty.png"
)
