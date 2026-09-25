import os
from io import StringIO

import requests
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import opensr_model


# ============================================================
# Settings
# ============================================================

LR_PATH = "outputs/sr/lr_input.pt"
CHECKPOINT_PATH = "weights/opensr-ldsrs2_v1_0_0.ckpt"

OUTPUT_DIR = "outputs/uncertainty"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 5
SAMPLING_STEPS = 100
SEED = 42


# ============================================================
# Device
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


# ============================================================
# Load official LDSR-S2 configuration
# ============================================================

config_url = (
    "https://raw.githubusercontent.com/ESAOpenSR/"
    "opensr-model/refs/heads/main/"
    "opensr_model/configs/config_10m.yaml"
)

print("Downloading official LDSR-S2 configuration...")

response = requests.get(config_url, timeout=30)
response.raise_for_status()

config = OmegaConf.load(
    StringIO(response.text)
)

print("LDSR-S2 configuration loaded.")


# ============================================================
# Create model
# ============================================================

print("Creating LDSR-S2 model...")

model = opensr_model.SRLatentDiffusion(
    config,
    device=device
)

print("Loading pretrained LDSR-S2 weights...")

model.load_pretrained(
    CHECKPOINT_PATH
)

model.eval()

print("LDSR-S2 model loaded successfully.")


# ============================================================
# Load LR
# ============================================================

lr = torch.load(
    LR_PATH,
    map_location="cpu",
    weights_only=False
)

if lr.shape != (1, 4, 128, 128):
    raise ValueError(
        f"Expected LR shape (1, 4, 128, 128), "
        f"got {lr.shape}"
    )

lr = lr.float().to(device)

print(f"LR shape: {lr.shape}")


# ============================================================
# Run same LR repeatedly
# ============================================================

print()
print("========================================")
print("UNCERTAINTY CONTROL")
print("SAME LR — NO PERTURBATION")
print("========================================")
print()

torch.manual_seed(SEED)

if device == "cuda":
    torch.cuda.manual_seed_all(SEED)


predictions = []


for i in range(NUM_SAMPLES):

    print(
        f"Running control sample "
        f"{i + 1}/{NUM_SAMPLES}..."
    )

    with torch.no_grad():

        sr = model.forward(
            lr,
            sampling_steps=SAMPLING_STEPS
        )

    print(f"  SR shape: {sr.shape}")

    predictions.append(
        sr.detach().cpu()
    )


# ============================================================
# Stack predictions
# ============================================================

predictions = torch.stack(
    predictions,
    dim=0
)

print()
print(
    f"Predictions shape: {predictions.shape}"
)


# ============================================================
# Pixel-wise variance
# ============================================================

variance = torch.var(
    predictions,
    dim=0,
    unbiased=False
)

print(
    f"Variance shape: {variance.shape}"
)


# ============================================================
# Channel-average uncertainty
# ============================================================

uncertainty_map = variance.mean(
    dim=1
).squeeze(0)

print(
    f"Uncertainty map shape: "
    f"{uncertainty_map.shape}"
)


# ============================================================
# Statistics
# ============================================================

flat = uncertainty_map.flatten()

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
print("CONTROL STATISTICS")
print("========================================")

print(f"Mean   : {mean_value:.8f}")
print(f"Median : {median_value:.8f}")
print(f"P90    : {p90:.8f}")
print(f"P95    : {p95:.8f}")
print(f"P99    : {p99:.8f}")


# ============================================================
# Save
# ============================================================

torch.save(
    predictions,
    os.path.join(
        OUTPUT_DIR,
        "control_predictions.pt"
    )
)

torch.save(
    variance,
    os.path.join(
        OUTPUT_DIR,
        "control_variance.pt"
    )
)

torch.save(
    uncertainty_map,
    os.path.join(
        OUTPUT_DIR,
        "control_uncertainty_map.pt"
    )
)


# ============================================================
# P99 visualization
# ============================================================

p99_tensor = torch.quantile(
    flat,
    0.99
)

visual_map = torch.clamp(
    uncertainty_map,
    max=p99_tensor
) / p99_tensor


plt.figure(figsize=(8, 8))

plt.imshow(
    visual_map.numpy(),
    cmap="inferno"
)

plt.colorbar(
    label="Control uncertainty (P99-clipped)"
)

plt.title(
    "LDSR-S2 Control Uncertainty\nSame LR, No Perturbation"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "control_uncertainty_map.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print()
print("========================================")
print("CONTROL SUCCESS!")
print("========================================")

print(
    "Saved: outputs/uncertainty/"
    "control_uncertainty_map.png"
)