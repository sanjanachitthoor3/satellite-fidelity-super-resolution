import os
from io import StringIO

import requests
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import opensr_model


# ============================================================
# 1. Paths
# ============================================================

LR_PATH = "data/lr_demo.pt"

OUTPUT_DIR = "outputs/sr"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "sr_demo.pt"
)


# ============================================================
# 2. Device
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


# ============================================================
# 3. Download official LDSR-S2 configuration
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
# 4. Create LDSR-S2 model
# ============================================================

print("Creating LDSR-S2 model...")

model = opensr_model.SRLatentDiffusion(
    config,
    device=device
)

print("Loading pretrained LDSR-S2 weights...")

model.load_pretrained(
    "weights/opensr-ldsrs2_v1_0_0.ckpt"
)

model.eval()

print("LDSR-S2 model loaded successfully.")


# ============================================================
# 5. Load our LR data
# ============================================================

print(f"Loading LR data from: {LR_PATH}")

lr_all = torch.load(
    LR_PATH,
    map_location="cpu",
    weights_only=False
)

print(f"All LR data shape: {lr_all.shape}")

# We have 5 degradation variants.
# For the first prototype, use variant 3.
lr = lr_all[2]

print(f"Selected LR shape: {lr.shape}")


# ============================================================
# 6. Add batch dimension
# ============================================================

# Currently:
#
# 4 × 121 × 121
#
# We need:
#
# 1 × 4 × 121 × 121

lr = lr.unsqueeze(0).float()

print(f"Before padding: {lr.shape}")


# ============================================================
# 7. PAD 121 × 121 → 128 × 128
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
# 8. Convert reflectance to approximately 0–1
# ============================================================

if lr.max() > 1:
    lr = lr / 10000.0

print(
    f"LR range before model: "
    f"{lr.min().item():.4f} → {lr.max().item():.4f}"
)

lr = lr.to(device)


# ============================================================
# 9. ACTUAL SUPER-RESOLUTION
# ============================================================

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
# 10. Save SR
# ============================================================

print(f"SR output shape: {sr.shape}")

torch.save(
    sr.cpu(),
    OUTPUT_PATH
)

print()
print("SUCCESS!")
print(f"SR saved to: {OUTPUT_PATH}")