import os
import sys
import requests
import torch
import torch.nn.functional as F
from io import StringIO
from omegaconf import OmegaConf
import opensr_model


# ============================================================
# Configuration
# ============================================================

ROI_ID = sys.argv[1] if len(sys.argv) > 1 else "ROI_2319"

OUTPUT_DIR = "outputs/baselines/ldsr_s2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    f"{ROI_ID}_sr.pt"
)

CHECKPOINT = "weights/opensr-ldsrs2_v1_0_0.ckpt"


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

response = requests.get(config_url)
response.raise_for_status()

config = OmegaConf.load(
    StringIO(response.text)
)

print("LDSR-S2 configuration loaded.")


# ============================================================
# Create LDSR-S2 model
# ============================================================

print("Creating LDSR-S2 model...")

model = opensr_model.SRLatentDiffusion(
    config,
    device=device
)

print("Loading pretrained LDSR-S2 weights...")

model.load_pretrained(CHECKPOINT)

model.eval()

print("LDSR-S2 model loaded successfully.")


# ============================================================
# Load exact benchmark LR input
# ============================================================

LR_PATH = f"baseline_inputs/{ROI_ID}/lr.pt"

if not os.path.exists(LR_PATH):
    raise FileNotFoundError(LR_PATH)

lr = torch.load(
    LR_PATH,
    map_location="cpu",
    weights_only=True,
).float()

print()
print(f"ROI: {ROI_ID}")
print(f"LR shape: {lr.shape}")
print(
    f"LR range: "
    f"{lr.min().item():.6f} -> {lr.max().item():.6f}"
)


# ============================================================
# Validate LR input
# ============================================================

if tuple(lr.shape) != (4, 121, 121):
    raise ValueError(
        f"Expected LR shape (4, 121, 121), "
        f"got {tuple(lr.shape)}"
    )


# ============================================================
# Add batch dimension
# ============================================================

lr = lr.unsqueeze(0).float()

print(f"Before padding: {lr.shape}")


# ============================================================
# Pad 121 × 121 → 128 × 128
# ============================================================

height = lr.shape[-2]
width = lr.shape[-1]

pad_height = 128 - height
pad_width = 128 - width

if pad_height < 0 or pad_width < 0:
    raise ValueError(
        f"Input is larger than 128×128: {lr.shape}"
    )

lr = F.pad(
    lr,
    (0, pad_width, 0, pad_height),
    mode="reflect"
)

print(f"After padding: {lr.shape}")


# ============================================================
# LDSR-S2 inference
# ============================================================

lr = lr.to(device)

print()
print("========================================")
print("RUNNING LDSR-S2 BASELINE")
print("========================================")
print()

with torch.no_grad():

    sr = model.forward(
        lr,
        sampling_steps=100
    )


# ============================================================
# Crop padded SR output
# ============================================================

print(f"Raw SR output shape: {sr.shape}")

expected_size = 484

if sr.shape[-2] < expected_size or sr.shape[-1] < expected_size:
    raise ValueError(
        f"SR output is too small: {sr.shape}"
    )

sr = sr[..., :expected_size, :expected_size]

print(f"Valid SR shape: {sr.shape}")


# ============================================================
# Remove batch dimension
# ============================================================

if sr.ndim == 4 and sr.shape[0] == 1:
    sr = sr.squeeze(0)

print(f"Final SR shape: {sr.shape}")


# ============================================================
# Validate SR output
# ============================================================

if tuple(sr.shape) != (4, 484, 484):
    raise ValueError(
        f"Expected SR shape (4, 484, 484), "
        f"got {tuple(sr.shape)}"
    )


# ============================================================
# Save
# ============================================================

torch.save(
    sr.cpu(),
    OUTPUT_PATH
)

print()
print("SUCCESS!")
print(f"SR saved to: {OUTPUT_PATH}")
print(
    f"SR range: "
    f"{sr.min().item():.6f} -> {sr.max().item():.6f}"
)