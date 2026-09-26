import torch


# ============================================================
# Load uncertainty maps
# ============================================================

perturbed = torch.load(
    "outputs/uncertainty/uncertainty_map.pt",
    map_location="cpu",
    weights_only=False,
).float()

control = torch.load(
    "outputs/uncertainty/control_uncertainty_map.pt",
    map_location="cpu",
    weights_only=False,
).float()

excess = torch.load(
    "outputs/uncertainty/excess_uncertainty.pt",
    map_location="cpu",
    weights_only=False,
).float()


# ============================================================
# Remove batch dimensions if present
# ============================================================

if perturbed.ndim == 3:
    perturbed = perturbed.squeeze(0)

if control.ndim == 3:
    control = control.squeeze(0)

if excess.ndim == 3:
    excess = excess.squeeze(0)


# ============================================================
# Validate
# ============================================================

print("Perturbed shape:", perturbed.shape)
print("Control shape  :", control.shape)
print("Excess shape   :", excess.shape)

if not (
    perturbed.shape
    == control.shape
    == excess.shape
):
    raise ValueError("Map shapes do not match.")


# ============================================================
# Basic statistics
# ============================================================

print()
print("========================================")
print("UNCERTAINTY SANITY CHECK")
print("========================================")

print(
    f"Perturbed mean : {perturbed.mean().item():.8f}"
)

print(
    f"Control mean   : {control.mean().item():.8f}"
)

print(
    f"Excess mean    : {excess.mean().item():.8f}"
)

print(
    f"Perturbed P99  : "
    f"{torch.quantile(perturbed.flatten(), 0.99).item():.8f}"
)

print(
    f"Control P99    : "
    f"{torch.quantile(control.flatten(), 0.99).item():.8f}"
)

print(
    f"Excess P99     : "
    f"{torch.quantile(excess.flatten(), 0.99).item():.8f}"
)


# ============================================================
# Check how often perturbation increases uncertainty
# ============================================================

positive_excess = (
    excess > 0
).float().mean()

negative_excess = (
    excess < 0
).float().mean()

zero_excess = (
    excess == 0
).float().mean()

print()
print("Fraction of pixels:")
print(
    f"Excess > 0 : {positive_excess.item() * 100:.2f}%"
)

print(
    f"Excess < 0 : {negative_excess.item() * 100:.2f}%"
)

print(
    f"Excess = 0 : {zero_excess.item() * 100:.2f}%"
)


# ============================================================
# Correlation between perturbed uncertainty and excess
# ============================================================

x = perturbed.flatten()
y = excess.flatten()

x_centered = x - x.mean()
y_centered = y - y.mean()

correlation = (
    (x_centered * y_centered).mean()
    /
    (
        x_centered.std()
        * y_centered.std()
        + 1e-12
    )
)

print()
print(
    f"Correlation "
    f"(perturbed vs excess): "
    f"{correlation.item():.6f}"
)


# ============================================================
# Increase ratio
# ============================================================

perturbed_mean = perturbed.mean()
control_mean = control.mean()

increase_ratio = (
    perturbed_mean / (control_mean + 1e-12)
)

increase_percent = (
    (perturbed_mean - control_mean)
    / (control_mean + 1e-12)
) * 100

print()
print(
    f"Perturbed / control ratio: "
    f"{increase_ratio.item():.3f}x"
)

print(
    f"Increase over control: "
    f"{increase_percent.item():.2f}%"
)


# ============================================================
# Final interpretation
# ============================================================

print()
print("========================================")
print("SANITY CHECK COMPLETE")
print("========================================")

if perturbed_mean > control_mean:
    print(
        "PASS: Perturbed predictions show "
        "higher average variability than control."
    )
else:
    print(
        "WARNING: Perturbation did not increase "
        "average variability."
    )

if positive_excess > 0.5:
    print(
        "PASS: Positive excess uncertainty "
        "occurs across more than half the pixels."
    )
else:
    print(
        "NOTE: Excess uncertainty is positive "
        "in less than half the pixels."
    )