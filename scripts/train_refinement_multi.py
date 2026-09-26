import os
import sys
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Adam


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from refinement.safe_reconstruction import RiskAwareResidualNet
from degradation.opensr_degradation import (
    create_naip_degradation_model,
)


# ============================================================
# Configuration
# ============================================================

TRAIN_SPLIT = "data/splits/train.txt"

# Smoke test first.
# After this works, change to None for the full training split.
MAX_TRAIN_ROIS = 10

OUTPUT_DIR = "outputs/refinement"
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "multi_roi_risk_aware_residual_net.pt",
)

EPOCHS = 100
LEARNING_RATE = 1e-4

LAMBDA_HR = 1.0
LAMBDA_LR = 1.0
LAMBDA_RESIDUAL = 0.01

SEED = 42


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")


    # ========================================================
    # Load training ROI IDs
    # ========================================================

    with open(TRAIN_SPLIT, "r") as f:
        roi_ids = [
            line.strip()
            for line in f
            if line.strip()
        ]

    if MAX_TRAIN_ROIS is not None:
        roi_ids = roi_ids[:MAX_TRAIN_ROIS]

    if len(roi_ids) == 0:
        raise ValueError(
            "No ROI IDs found in train split."
        )

    print(f"Training ROIs: {len(roi_ids)}")

    print()
    print("ROIs:")
    for roi_id in roi_ids:
        print(f"  {roi_id}")


    # ========================================================
    # Create differentiable OpenSR degradation model
    # ========================================================

    print()
    print("Creating OpenSR degradation model...")

    degradation_model = create_naip_degradation_model(
        device=device,
        seed=SEED,
        add_noise=True,
        reflectance_methods=[
            "gamma_multivariate_normal"
        ],
    )

    print(
        "OpenSR degradation model ready."
    )

    print(
        "Refinement fidelity branch: "
        "gamma_multivariate_normal"
    )


    # ========================================================
    # Create refinement model
    # ========================================================

    model = RiskAwareResidualNet(
        in_channels=9,
        out_channels=4,
        features=64,
    ).to(device)

    model.train()


    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )


    # ========================================================
    # Training
    # ========================================================

    print()
    print("========================================")
    print("STARTING MULTI-ROI SAFE-SR TRAINING")
    print("========================================")

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        epoch_total = 0.0
        epoch_hr = 0.0
        epoch_lr = 0.0
        epoch_residual = 0.0

        for roi_id in roi_ids:

            # ------------------------------------------------
            # Paths
            # ------------------------------------------------

            hr_path = (
                f"data/processed/{roi_id}/hr.pt"
            )

            lr_path = (
                f"data/processed/{roi_id}/lr.pt"
            )

            sr_path = (
                f"outputs/sr/{roi_id}_sr.pt"
            )

            risk_path = (
                f"outputs/risk/"
                f"{roi_id}_risk_map_normalized.pt"
            )


            # ------------------------------------------------
            # Check required files
            # ------------------------------------------------

            required_files = [
                hr_path,
                lr_path,
                sr_path,
                risk_path,
            ]

            for path in required_files:
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"{roi_id}: missing file:\n{path}"
                    )


            # ------------------------------------------------
            # Load HR
            # ------------------------------------------------

            hr = torch.load(
                hr_path,
                map_location="cpu",
                weights_only=True,
            ).float()

            if hr.ndim == 3:
                hr = hr.unsqueeze(0)

            if hr.shape != (
                1,
                4,
                484,
                484,
            ):
                raise ValueError(
                    f"{roi_id}: expected HR "
                    f"[1,4,484,484], got {hr.shape}"
                )


            # ------------------------------------------------
            # Load LR
            # ------------------------------------------------

            lr = torch.load(
                lr_path,
                map_location="cpu",
                weights_only=True,
            ).float()

            if lr.ndim == 3:
                lr = lr.unsqueeze(0)

            if lr.shape != (
                1,
                4,
                121,
                121,
            ):
                raise ValueError(
                    f"{roi_id}: expected LR "
                    f"[1,4,121,121], got {lr.shape}"
                )


            # ------------------------------------------------
            # Load LDSR-S2 SR
            # ------------------------------------------------

            sr = torch.load(
                sr_path,
                map_location="cpu",
                weights_only=True,
            ).float()

            if sr.ndim == 3:
                sr = sr.unsqueeze(0)

            if sr.shape != (
                1,
                4,
                484,
                484,
            ):
                raise ValueError(
                    f"{roi_id}: expected SR "
                    f"[1,4,484,484], got {sr.shape}"
                )


            # ------------------------------------------------
            # Load risk map
            # ------------------------------------------------

            risk = torch.load(
                risk_path,
                map_location="cpu",
                weights_only=True,
            ).float()

            if risk.shape != (
                484,
                484,
            ):
                raise ValueError(
                    f"{roi_id}: expected risk "
                    f"[484,484], got {risk.shape}"
                )

            risk = (
                risk
                .unsqueeze(0)
                .unsqueeze(0)
            )


            # ------------------------------------------------
            # Move tensors to device
            # ------------------------------------------------

            hr = hr.to(device)
            lr = lr.to(device)
            sr = sr.to(device)
            risk = risk.to(device)


            # ------------------------------------------------
            # Upsample LR for network input
            # ------------------------------------------------

            lr_up = F.interpolate(
                lr,
                size=(484, 484),
                mode="bilinear",
                align_corners=False,
            )


            # ------------------------------------------------
            # Training step
            # ------------------------------------------------

            optimizer.zero_grad()


            # Predict residual
            residual = model(
                sr,
                lr_up,
                risk,
            )


            # Refined SR
            refined_sr = sr + residual


            # HR reconstruction loss
            hr_loss = F.l1_loss(
                refined_sr,
                hr,
            )


            # OpenSR LR consistency loss
            degraded = degradation_model.forward(
                refined_sr.squeeze(0),
            )

            # With one reflectance branch:
            # degraded[0] = LR output
            # shape = [1,4,121,121]

            refined_lr = degraded[0]

            if refined_lr.shape != lr.shape:
                raise ValueError(
                    f"{roi_id}: OpenSR degraded LR "
                    f"shape mismatch: "
                    f"{refined_lr.shape} vs {lr.shape}"
                )

            lr_loss = F.l1_loss(
                refined_lr,
                lr,
            )


            # Residual regularization
            residual_loss = torch.mean(
                torch.abs(residual)
            )


            # Total loss
            loss = (
                LAMBDA_HR * hr_loss
                + LAMBDA_LR * lr_loss
                + LAMBDA_RESIDUAL * residual_loss
            )


            # Backpropagation
            loss.backward()

            optimizer.step()


            # Accumulate epoch statistics
            epoch_total += loss.item()
            epoch_hr += hr_loss.item()
            epoch_lr += lr_loss.item()
            epoch_residual += residual_loss.item()


        # ====================================================
        # Epoch logging
        # ====================================================

        n = len(roi_ids)

        avg_total = epoch_total / n
        avg_hr = epoch_hr / n
        avg_lr = epoch_lr / n
        avg_residual = epoch_residual / n

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Total: {avg_total:.6f} | "
                f"HR: {avg_hr:.6f} | "
                f"LR: {avg_lr:.6f} | "
                f"Residual: {avg_residual:.6f}"
            )


    # ========================================================
    # Save trained model
    # ========================================================

    torch.save(
        model.state_dict(),
        MODEL_PATH,
    )

    print()
    print(
        f"Saved model: {MODEL_PATH}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()