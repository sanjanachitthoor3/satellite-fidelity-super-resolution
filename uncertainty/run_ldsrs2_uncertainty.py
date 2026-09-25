import os
from io import StringIO

import requests
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import opensr_model


# ============================================================
# 1. Paths
# ============================================================

LR_PATH = "outputs/sr/lr_input.pt"
CHECKPOINT_PATH = "weights/opensr-ldsrs2_v1_0_0.ckpt"

OUTPUT_DIR = "outputs/uncertainty"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Uncertainty settings
# ============================================================

NUM_SAMPLES = 5
NOISE_STD = 0.005
SEED = 42
SAMPLING_STEPS = 100


# ============================================================
# 3. Device
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


# ============================================================
# 4. Load official LDSR-S2 configuration
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
# 5. Create LDSR-S2 model
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
# 6. Load LR input
# ============================================================

print(f"Loading LR input from: {LR_PATH}")

lr = torch.load(
    LR_PATH,
    map_location="cpu",
    weights_only=False
)

print(f"LR shape: {lr.shape}")

if lr.shape != (1, 4, 128, 128):
    raise ValueError(
        f"Expected LR shape (1, 4, 128, 128), "
        f"got {lr.shape}"
    )

lr = lr.float()


# ============================================================
# 7. Move LR to GPU
# ============================================================

lr = lr.to(device)

print(
    f"LR range: "
    f"{lr.min().item():.6f} -> "
    f"{lr.max().item():.6f}"
)


# ============================================================
# 8. Generate perturbed SR predictions
# ============================================================

print()
print("========================================")
print("RUNNING UNCERTAINTY ESTIMATION")
print("========================================")
print()

print(f"Number of samples : {NUM_SAMPLES}")
print(f"Noise std         : {NOISE_STD}")
print(f"Sampling steps    : {SAMPLING_STEPS}")
print()

torch.manual_seed(SEED)

if device == "cuda":
    torch.cuda.manual_seed_all(SEED)


predictions = []


for i in range(NUM_SAMPLES):

    print(
        f"Running LDSR-S2 sample "
        f"{i + 1}/{NUM_SAMPLES}..."
    )

    # --------------------------------------------------------
    # Add small Gaussian perturbation to LR
    # --------------------------------------------------------

    noise = torch.randn_like(lr) * NOISE_STD

    perturbed_lr = lr + noise

    # --------------------------------------------------------
    # Run LDSR-S2
    # --------------------------------------------------------

    with torch.no_grad():

        sr = model.forward(
            perturbed_lr,
            sampling_steps=SAMPLING_STEPS
        )

    print(
        f"  SR shape: {sr.shape}"
    )

    predictions.append(
        sr.detach().cpu()
    )


# ============================================================
# 9. Stack predictions
# ============================================================

predictions = torch.stack(
    predictions,
    dim=0
)

print()
print(
    f"Predictions tensor shape: "
    f"{predictions.shape}"
)

# Expected:
#
# 5 × 1 × 4 × 512 × 512
#


# ============================================================
# 10. Pixel-wise prediction variance
# ============================================================

print()
print("Computing pixel-wise variance...")

variance = torch.var(
    predictions,
    dim=0,
    unbiased=False
)

print(
    f"Variance shape: {variance.shape}"
)


# ============================================================
# 11. Convert 4-channel variance to spatial map
# ============================================================

# variance:
#
# 1 × 4 × 512 × 512
#
# Average the four spectral channels.

uncertainty_map = variance.mean(
    dim=1
)

print(
    f"Uncertainty map shape: "
    f"{uncertainty_map.shape}"
)


# ============================================================
# 12. Remove batch dimension
# ============================================================

uncertainty_map = uncertainty_map.squeeze(0)


# ============================================================
# 13. Normalize uncertainty map to 0-1
# ============================================================

min_value = uncertainty_map.min()
max_value = uncertainty_map.max()

if torch.isclose(
    max_value,
    min_value
):

    normalized_map = torch.zeros_like(
        uncertainty_map
    )

else:

    normalized_map = (
        uncertainty_map - min_value
    ) / (
        max_value - min_value
    )


# ============================================================
# 14. Statistics
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
print("UNCERTAINTY STATISTICS")
print("========================================")

print(f"Mean   : {mean_value:.8f}")
print(f"Median : {median_value:.8f}")
print(f"P90    : {p90:.8f}")
print(f"P95    : {p95:.8f}")
print(f"P99    : {p99:.8f}")


# ============================================================
# 15. Save prediction tensors
# ============================================================

torch.save(
    predictions,
    os.path.join(
        OUTPUT_DIR,
        "predictions.pt"
    )
)

torch.save(
    variance,
    os.path.join(
        OUTPUT_DIR,
        "variance.pt"
    )
)

torch.save(
    uncertainty_map,
    os.path.join(
        OUTPUT_DIR,
        "uncertainty_map.pt"
    )
)

torch.save(
    normalized_map,
    os.path.join(
        OUTPUT_DIR,
        "uncertainty_map_normalized.pt"
    )
)


# ============================================================
# 16. Save visualization
# ============================================================

plt.figure(
    figsize=(8, 8)
)

plt.imshow(
    normalized_map.numpy(),
    cmap="inferno"
)

plt.colorbar(
    label="Normalized uncertainty"
)

plt.title(
    "LDSR-S2 Prediction Uncertainty"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "uncertainty_map.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. Done
# ============================================================

print()
print("========================================")
print("SUCCESS!")
print("========================================")

print(
    f"Predictions saved to: "
    f"{OUTPUT_DIR}/predictions.pt"
)

print(
    f"Variance saved to: "
    f"{OUTPUT_DIR}/variance.pt"
)

print(
    f"Uncertainty map saved to: "
    f"{OUTPUT_DIR}/uncertainty_map.pt"
)

print(
    f"Visualization saved to: "
    f"{OUTPUT_DIR}/uncertainty_map.png"
)