import os
import requests
import torch
import torch.nn.functional as F
from io import StringIO
from omegaconf import OmegaConf
import opensr_model



# ============================================================
# Configuration
# ============================================================

import sys

ROI_ID = sys.argv[1] if len(sys.argv) > 1 else "ROI_1732"

OUTPUT_DIR = "outputs/sr"
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
# Load one real SEN2NAIP test ROI
# ============================================================

LR_PATH = f"data/processed/{ROI_ID}/lr.pt"
HR_PATH = f"data/processed/{ROI_ID}/hr.pt"

if not os.path.exists(LR_PATH):
    raise FileNotFoundError(LR_PATH)

if not os.path.exists(HR_PATH):
    raise FileNotFoundError(HR_PATH)

lr = torch.load(
    LR_PATH,
    map_location="cpu",
    weights_only=True,
).float()

hr = torch.load(
    HR_PATH,
    map_location="cpu",
    weights_only=True,
).float()

print()
print(f"ROI: {ROI_ID}")
print(f"LR shape: {lr.shape}")
print(f"HR shape: {hr.shape}")
print(
    f"LR range: "
    f"{lr.min().item():.6f} -> {lr.max().item():.6f}"
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
print("RUNNING LDSR-S2 SUPER-RESOLUTION")
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