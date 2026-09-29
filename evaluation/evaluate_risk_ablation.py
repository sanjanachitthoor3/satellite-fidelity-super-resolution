import os
import json
import torch
import numpy as np

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from degradation.opensr_degradation import create_naip_degradation_model
from skimage.metrics import structural_similarity as skimage_ssim


# ============================================================
# TEST 6 — RISK COMPONENT ABLATION EVALUATION
# ============================================================

ROIS = ["ROI_2319", "ROI_0630"]

MODES = {
    "Fidelity-only": "fidelity",
    "Uncertainty-only": "uncertainty",
    "Fused": "fused",
}

OUTPUT_DIR = "outputs/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Loading
# ------------------------------------------------------------

def load_tensor(path):
    x = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    ).float()

    return x


def clamp_image(x):
    return torch.clamp(x, 0.0, 1.0)


# ------------------------------------------------------------
# Metrics — matching final_evaluation.py
# ------------------------------------------------------------

def psnr(prediction, target):
    mse = torch.mean((prediction - target) ** 2).item()

    if mse == 0:
        return float("inf")

    return float(10.0 * np.log10(1.0 / mse))


def ssim(prediction, target):
    prediction = prediction.squeeze(0).cpu().numpy()
    target = target.squeeze(0).cpu().numpy()

    # [C,H,W] -> [H,W,C]
    prediction = np.transpose(prediction, (1, 2, 0))
    target = np.transpose(target, (1, 2, 0))

    return float(
        skimage_ssim(
            target,
            prediction,
            data_range=1.0,
            channel_axis=2,
        )
    )


def opensr_lr_consistency(sr, lr, degradation_model):
    """Match the OpenSR consistency calculation used by final_evaluation.py."""

    sr_input = sr.squeeze(0)

    with torch.no_grad():
        degraded_lr, _ = degradation_model.forward(sr_input)

    # OpenSR NAIP-D returns multiple LR variants:
    # [5, 4, 121, 121]
    # The project uses the same branch as the degradation pipeline:
    # LR variant index 2.
    if degraded_lr.ndim == 4:
        degraded_lr = degraded_lr[2]

    if degraded_lr.ndim == 3:
        degraded_lr = degraded_lr.unsqueeze(0)

    if lr.ndim == 3:
        lr = lr.unsqueeze(0)

    if degraded_lr.shape != lr.shape:
        raise ValueError(
            "OpenSR degraded LR shape mismatch: "
            f"{degraded_lr.shape} vs {lr.shape}"
        )

    return float(
        torch.mean(
            torch.abs(degraded_lr - lr)
        )
    )

def evaluate_image(prediction, hr, lr, degradation_model):

    prediction = clamp_image(prediction)
    hr = clamp_image(hr)

    return {
        "PSNR_dB": psnr(prediction, hr),
        "SSIM": ssim(prediction, hr),
        "OpenSR_consistency_L1": opensr_lr_consistency(
            prediction,
            lr,
            degradation_model,
        ),
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("TEST 6 — RISK COMPONENT ABLATION EVALUATION")
print("=" * 70)

print("\nValidation ROIs:")
for roi in ROIS:
    print(f"  {roi}")

print("\nVariants:")
for name in MODES:
    print(f"  {name}")


# ------------------------------------------------------------
# OpenSR model
# ------------------------------------------------------------

print("\nCreating OpenSR degradation model...")

degradation_model = create_naip_degradation_model(
    device="cpu"
)

print("OpenSR evaluation model ready.")


results = {}


# ============================================================
# PER-ROI EVALUATION
# ============================================================

for roi in ROIS:

    print("\n" + "=" * 70)
    print(f"EVALUATING {roi}")
    print("=" * 70)

    hr_path = os.path.join(
        "data",
        "processed",
        roi,
        "hr.pt",
    )

    lr_path = os.path.join(
        "data",
        "processed",
        roi,
        "lr.pt",
    )

    hr = load_tensor(hr_path)
    lr = load_tensor(lr_path)

    print(f"HR shape: {tuple(hr.shape)}")
    print(f"LR shape: {tuple(lr.shape)}")

    results[roi] = {}

    for display_name, mode in MODES.items():

        sr_path = os.path.join(
            "outputs",
            "refinement",
            "ablations",
            f"{roi}_{mode}_safe_sr.pt",
        )

        print(f"\n{display_name}")
        print(f"  File: {sr_path}")

        if not os.path.exists(sr_path):
            print("  ERROR: file not found")
            continue

        prediction = load_tensor(sr_path)

        print(
            f"  SR shape: {tuple(prediction.shape)}"
        )

        metrics = evaluate_image(
            prediction,
            hr,
            lr,
            degradation_model,
        )

        results[roi][mode] = metrics

        print(
            f"  PSNR: "
            f"{metrics['PSNR_dB']:.4f} dB"
        )

        print(
            f"  SSIM: "
            f"{metrics['SSIM']:.6f}"
        )

        print(
            f"  OpenSR consistency: "
            f"{metrics['OpenSR_consistency_L1']:.6f}"
        )


# ============================================================
# AVERAGE
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST 6 RESULTS")
print("=" * 70)

average = {}

for display_name, mode in MODES.items():

    values = [
        results[roi][mode]
        for roi in ROIS
        if mode in results[roi]
    ]

    if not values:
        continue

    avg_psnr = np.mean(
        [v["PSNR_dB"] for v in values]
    )

    avg_ssim = np.mean(
        [v["SSIM"] for v in values]
    )

    avg_consistency = np.mean(
        [v["OpenSR_consistency_L1"] for v in values]
    )

    average[mode] = {
        "PSNR_dB": float(avg_psnr),
        "SSIM": float(avg_ssim),
        "OpenSR_consistency_L1": float(avg_consistency),
    }

    print(f"\n{display_name}")
    print(f"  PSNR: {avg_psnr:.4f} dB")
    print(f"  SSIM: {avg_ssim:.6f}")
    print(f"  OpenSR consistency: {avg_consistency:.6f}")


# ============================================================
# SAVE
# ============================================================

output = {
    "test": "Test 6 - Risk Component Ablation",
    "validation_rois": ROIS,
    "per_roi": results,
    "average": average,
}

output_path = os.path.join(
    OUTPUT_DIR,
    "risk_component_ablation.json",
)

with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print("\n" + "=" * 70)
print("RESULTS SAVED")
print("=" * 70)
print(output_path)
print("=" * 70)