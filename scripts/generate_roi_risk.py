import os
import sys
from io import StringIO

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import requests
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import opensr_model

from degradation.opensr_degradation import (
    create_naip_degradation_model,
    degrade_sr_to_lr,
)


# ============================================================
# Configuration
# ============================================================

ROI_ID = sys.argv[1] if len(sys.argv) > 1 else "ROI_1732"

LR_PATH = f"data/processed/{ROI_ID}/lr.pt"
SR_PATH = f"outputs/sr/{ROI_ID}_sr.pt"

CHECKPOINT_PATH = "weights/opensr-ldsrs2_v1_0_0.ckpt"

OUTPUT_DIR = "outputs/risk"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 5
NOISE_STD = 0.005
SAMPLING_STEPS = 100
SEED = 42

VALID_SIZE = 484
LR_SIZE = 121


# ============================================================
# Helpers
# ============================================================

def normalize_p99(x):
    x = x.float()

    p99 = torch.quantile(
        x.flatten(),
        0.99,
    )

    if p99 <= 0:
        return torch.zeros_like(x)

    return torch.clamp(
        x,
        min=0.0,
        max=p99,
    ) / p99


def pad_lr(lr):
    if lr.shape != (4, LR_SIZE, LR_SIZE):
        raise ValueError(
            f"Expected LR [4,121,121], got {lr.shape}"
        )

    lr = lr.unsqueeze(0)

    return F.pad(
        lr,
        (0, 7, 0, 7),
        mode="reflect",
    )


# ============================================================
# Load data
# ============================================================

print("========================================")
print(f"GENERATING RISK MAP FOR {ROI_ID}")
print("========================================")

lr = torch.load(
    LR_PATH,
    map_location="cpu",
    weights_only=True,
).float()

sr = torch.load(
    SR_PATH,
    map_location="cpu",
    weights_only=True,
).float()

if sr.ndim == 4:
    sr = sr.squeeze(0)

if lr.shape != (4, 121, 121):
    raise ValueError(
        f"Expected LR [4,121,121], got {lr.shape}"
    )

if sr.shape != (4, 484, 484):
    raise ValueError(
        f"Expected SR [4,484,484], got {sr.shape}"
    )

print(f"LR: {lr.shape}")
print(f"SR: {sr.shape}")


# ============================================================
# PART 1 — Fidelity
# ============================================================

print()
print("========================================")
print("1. FIDELITY")
print("========================================")

degradation_model = create_naip_degradation_model(
    device="cpu",
    seed=SEED,
    add_noise=True,
)

with torch.no_grad():
    lr_prime_all = degrade_sr_to_lr(
        sr,
        degradation_model,
    )

# Verified OpenSR branch mapping:
# index 2 = gamma_multivariate_normal
lr_prime = lr_prime_all[2]

fidelity_error = (
    lr_prime - lr
).abs()

fidelity_map_lr = fidelity_error.mean(
    dim=0
)

fidelity_map = F.interpolate(
    fidelity_map_lr.unsqueeze(0).unsqueeze(0),
    size=(VALID_SIZE, VALID_SIZE),
    mode="bilinear",
    align_corners=False,
).squeeze()

fidelity_risk = normalize_p99(
    fidelity_map
)

print(
    f"Fidelity mean: "
    f"{fidelity_map.mean().item():.8f}"
)

print(
    f"Fidelity P99: "
    f"{torch.quantile(fidelity_map.flatten(), 0.99).item():.8f}"
)


# ============================================================
# PART 2 — LDSR-S2 uncertainty
# ============================================================

print()
print("========================================")
print("2. LDSR-S2 UNCERTAINTY")
print("========================================")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")

config_url = (
    "https://raw.githubusercontent.com/ESAOpenSR/"
    "opensr-model/refs/heads/main/"
    "opensr_model/configs/config_10m.yaml"
)

response = requests.get(
    config_url,
    timeout=30,
)

response.raise_for_status()

config = OmegaConf.load(
    StringIO(response.text)
)

model = opensr_model.SRLatentDiffusion(
    config,
    device=device,
)

model.load_pretrained(
    CHECKPOINT_PATH
)

model.eval()

print("LDSR-S2 model loaded.")


# ============================================================
# Prepare LR
# ============================================================

lr_input = pad_lr(lr).to(device)

print(
    f"Padded LR shape: "
    f"{lr_input.shape}"
)


# ============================================================
# Control uncertainty
# ============================================================

print()
print("Running control samples...")

torch.manual_seed(SEED)

if device == "cuda":
    torch.cuda.manual_seed_all(SEED)

control_predictions = []

for i in range(NUM_SAMPLES):

    print(
        f"  Control sample "
        f"{i + 1}/{NUM_SAMPLES}"
    )

    with torch.no_grad():
        prediction = model.forward(
            lr_input,
            sampling_steps=SAMPLING_STEPS,
        )

    control_predictions.append(
        prediction.detach().cpu()
    )

control_predictions = torch.stack(
    control_predictions,
    dim=0,
)

control_variance = torch.var(
    control_predictions,
    dim=0,
    unbiased=False,
)

control_map = control_variance.mean(
    dim=1
).squeeze(0)


# ============================================================
# Perturbed uncertainty
# ============================================================

print()
print("Running perturbed samples...")

torch.manual_seed(SEED)

if device == "cuda":
    torch.cuda.manual_seed_all(SEED)

perturbed_predictions = []

for i in range(NUM_SAMPLES):

    print(
        f"  Perturbed sample "
        f"{i + 1}/{NUM_SAMPLES}"
    )

    noise = (
        torch.randn_like(lr_input)
        * NOISE_STD
    )

    perturbed_lr = (
        lr_input + noise
    )

    with torch.no_grad():
        prediction = model.forward(
            perturbed_lr,
            sampling_steps=SAMPLING_STEPS,
        )

    perturbed_predictions.append(
        prediction.detach().cpu()
    )

perturbed_predictions = torch.stack(
    perturbed_predictions,
    dim=0,
)

perturbed_variance = torch.var(
    perturbed_predictions,
    dim=0,
    unbiased=False,
)

perturbed_map = perturbed_variance.mean(
    dim=1
).squeeze(0)


# ============================================================
# Excess uncertainty
# ============================================================

excess_uncertainty = torch.clamp(
    perturbed_map - control_map,
    min=0.0,
)

# 512 -> 484
excess_uncertainty = excess_uncertainty[
    :VALID_SIZE,
    :VALID_SIZE,
]

uncertainty_risk = normalize_p99(
    excess_uncertainty
)

print(
    f"Excess uncertainty mean: "
    f"{excess_uncertainty.mean().item():.8f}"
)

print(
    f"Excess uncertainty P99: "
    f"{torch.quantile(excess_uncertainty.flatten(), 0.99).item():.8f}"
)


# ============================================================
# PART 3 — Fuse risk signals
# ============================================================

print()
print("========================================")
print("3. RISK FUSION")
print("========================================")

if fidelity_risk.shape != uncertainty_risk.shape:
    raise ValueError(
        f"Risk shape mismatch: "
        f"{fidelity_risk.shape} vs "
        f"{uncertainty_risk.shape}"
    )

risk_map = (
    0.5 * fidelity_risk
    + 0.5 * uncertainty_risk
)

risk_map = torch.clamp(
    risk_map,
    0.0,
    1.0,
)

print(f"Risk shape: {risk_map.shape}")

print(
    f"Risk mean: "
    f"{risk_map.mean().item():.8f}"
)

print(
    f"Risk median: "
    f"{risk_map.median().item():.8f}"
)

print(
    f"Risk P90: "
    f"{torch.quantile(risk_map.flatten(), 0.90).item():.8f}"
)

print(
    f"Risk P95: "
    f"{torch.quantile(risk_map.flatten(), 0.95).item():.8f}"
)

print(
    f"Risk P99: "
    f"{torch.quantile(risk_map.flatten(), 0.99).item():.8f}"
)


# ============================================================
# Save ROI-specific outputs
# ============================================================

torch.save(
    fidelity_risk,
    os.path.join(
        OUTPUT_DIR,
        f"{ROI_ID}_fidelity_risk.pt",
    ),
)

torch.save(
    uncertainty_risk,
    os.path.join(
        OUTPUT_DIR,
        f"{ROI_ID}_uncertainty_risk.pt",
    ),
)

torch.save(
    excess_uncertainty,
    os.path.join(
        OUTPUT_DIR,
        f"{ROI_ID}_excess_uncertainty.pt",
    ),
)

torch.save(
    risk_map,
    os.path.join(
        OUTPUT_DIR,
        f"{ROI_ID}_risk_map_normalized.pt",
    ),
)


# ============================================================
# Complete
# ============================================================

print()
print("========================================")
print("RISK GENERATION COMPLETE")
print("========================================")

print(
    f"Saved: "
    f"{OUTPUT_DIR}/{ROI_ID}_fidelity_risk.pt"
)

print(
    f"Saved: "
    f"{OUTPUT_DIR}/{ROI_ID}_uncertainty_risk.pt"
)

print(
    f"Saved: "
    f"{OUTPUT_DIR}/{ROI_ID}_excess_uncertainty.pt"
)

print(
    f"Saved: "
    f"{OUTPUT_DIR}/{ROI_ID}_risk_map_normalized.pt"
)