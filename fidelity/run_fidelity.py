import os
import torch
import matplotlib.pyplot as plt

from fidelity.consistency import compute_fidelity_from_sr
from degradation.opensr_degradation import create_naip_degradation_model


# --------------------------------------------------
# Paths
# --------------------------------------------------

LR_PATH = "outputs/sr/lr_input.pt"
SR_PATH = "outputs/sr/sr_demo.pt"

OUTPUT_DIR = "outputs/fidelity"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Load actual LDSR-S2 input/output
# --------------------------------------------------

print("Loading LR input...")
lr_input = torch.load(LR_PATH, weights_only=False)

print("Loading SR output...")
sr = torch.load(SR_PATH, weights_only=False)

print("LR input:", lr_input.shape)
print("SR output:", sr.shape)


# --------------------------------------------------
# Create OpenSR degradation model
# --------------------------------------------------

print("\nCreating OpenSR degradation model...")

degradation_model = create_naip_degradation_model(
    device="cpu",
    seed=42,
)

print("Degradation model ready.")


# --------------------------------------------------
# Compute fidelity
# --------------------------------------------------

print("\nComputing LR-SR-LR consistency...")

lr_prime, error, fidelity_map, normalized_map = (
    compute_fidelity_from_sr(
        lr_input=lr_input,
        sr=sr,
        degradation_model=degradation_model,
        lr_prime_index=2,
    )
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\nSUCCESS!")

print("LR input:", lr_input.shape)
print("LR prime:", lr_prime.shape)
print("Error:", error.shape)
print("Fidelity map:", fidelity_map.shape)

print(
    "Full fidelity error:",
    error.min().item(),
    "→",
    error.max().item(),
)


# --------------------------------------------------
# Crop to valid original LR region
# --------------------------------------------------

valid_size = 121

valid_error = error[:, :valid_size, :valid_size]
valid_fidelity_map = fidelity_map[:valid_size, :valid_size]

print("\nValid region:")
print("Valid error:", valid_error.shape)
print("Valid fidelity map:", valid_fidelity_map.shape)

print(
    "Valid mean:",
    valid_fidelity_map.mean().item(),
)

print(
    "Valid median:",
    valid_fidelity_map.median().item(),
)

p90 = torch.quantile(
    valid_fidelity_map.flatten(),
    0.90,
).item()

p95 = torch.quantile(
    valid_fidelity_map.flatten(),
    0.95,
).item()

p99 = torch.quantile(
    valid_fidelity_map.flatten(),
    0.99,
).item()

print("Valid P90:", p90)
print("Valid P95:", p95)
print("Valid P99:", p99)


# --------------------------------------------------
# Robust normalization for valid region
# --------------------------------------------------

if p99 > 0:
    valid_normalized_map = torch.clamp(
        valid_fidelity_map / p99,
        0.0,
        1.0,
    )
else:
    valid_normalized_map = torch.zeros_like(
        valid_fidelity_map
    )

print(
    "Robust normalized range:",
    valid_normalized_map.min().item(),
    "→",
    valid_normalized_map.max().item(),
)


# --------------------------------------------------
# Save tensors
# --------------------------------------------------

torch.save(
    lr_prime,
    f"{OUTPUT_DIR}/lr_prime.pt",
)

torch.save(
    error,
    f"{OUTPUT_DIR}/fidelity_error.pt",
)

torch.save(
    valid_error,
    f"{OUTPUT_DIR}/fidelity_error_valid.pt",
)

torch.save(
    valid_fidelity_map,
    f"{OUTPUT_DIR}/fidelity_map_valid.pt",
)

torch.save(
    valid_normalized_map,
    f"{OUTPUT_DIR}/fidelity_map_normalized_valid.pt",
)


# --------------------------------------------------
# Save visualization
# --------------------------------------------------

plt.figure(figsize=(7, 7))

plt.imshow(
    valid_normalized_map.cpu().numpy(),
    cmap="hot",
    vmin=0,
    vmax=1,
)

plt.colorbar(
    label="Normalized Fidelity Error"
)

plt.title(
    "LR-SR-LR Fidelity Error Map"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/fidelity_map_valid.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# Done
# --------------------------------------------------

print(
    f"\nSaved fidelity outputs to: {OUTPUT_DIR}"
)

print(
    f"Visualization saved to: "
    f"{OUTPUT_DIR}/fidelity_map_valid.png"
)