import os
import sys

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

ROI_ID = "ROI_1732"

HR_PATH = f"data/processed/{ROI_ID}/hr.pt"
LR_PATH = f"data/processed/{ROI_ID}/lr.pt"
SR_PATH = f"outputs/sr/{ROI_ID}_sr.pt"
RISK_PATH = (
    f"outputs/risk/{ROI_ID}_risk_map_normalized.pt"
)

OUTPUT_DIR = "outputs/refinement"
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    f"{ROI_ID}_no_risk_residual_net.pt",
)

SAFE_SR_PATH = os.path.join(
    OUTPUT_DIR,
    f"{ROI_ID}_no_risk_sr.pt",
)

RESIDUAL_PATH = os.path.join(
    OUTPUT_DIR,
    f"{ROI_ID}_no_risk_residual.pt",
)


# ============================================================
# Training configuration
# ============================================================

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

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"ROI: {ROI_ID}")
    print("Experiment: NO-RISK ABLATION")


    # ========================================================
    # Load HR
    # ========================================================

    hr = torch.load(
        HR_PATH,
        map_location="cpu",
        weights_only=True,
    ).float()

    if hr.ndim == 3:
        hr = hr.unsqueeze(0)

    if hr.shape != (1, 4, 484, 484):
        raise ValueError(
            f"Expected HR [1,4,484,484], got {hr.shape}"
        )


    # ========================================================
    # Load LR
    # ========================================================

    lr = torch.load(
        LR_PATH,
        map_location="cpu",
        weights_only=True,
    ).float()

    if lr.ndim == 3:
        lr = lr.unsqueeze(0)

    if lr.shape != (1, 4, 121, 121):
        raise ValueError(
            f"Expected LR [1,4,121,121], got {lr.shape}"
        )


    # ========================================================
    # Load LDSR-S2 SR
    # ========================================================

    sr = torch.load(
        SR_PATH,
        map_location="cpu",
        weights_only=True,
    ).float()

    if sr.ndim == 3:
        sr = sr.unsqueeze(0)

    if sr.shape != (1, 4, 484, 484):
        raise ValueError(
            f"Expected SR [1,4,484,484], got {sr.shape}"
        )


    # ========================================================
    # Load risk map only for shape verification
    # ========================================================

    risk = torch.load(
        RISK_PATH,
        map_location="cpu",
        weights_only=True,
    ).float()

    if risk.shape != (484, 484):
        raise ValueError(
            f"Expected risk [484,484], got {risk.shape}"
        )

    risk = risk.unsqueeze(0).unsqueeze(0)


    # ========================================================
    # Print loaded tensors
    # ========================================================

    print()
    print("Loaded tensors:")
    print(f"  HR:   {hr.shape}")
    print(f"  LR:   {lr.shape}")
    print(f"  SR:   {sr.shape}")
    print(f"  Risk: {risk.shape}")


    # ========================================================
    # Replace risk map with zeros
    # ========================================================

    risk = torch.zeros_like(risk)

    print()
    print("Risk input replaced with zeros.")
    print(
        f"  Zero-risk mean: {risk.mean().item():.6f}"
    )
    print(
        f"  Zero-risk max:  {risk.max().item():.6f}"
    )


    # ========================================================
    # Move tensors to device
    # ========================================================

    hr = hr.to(device)
    lr = lr.to(device)
    sr = sr.to(device)
    risk = risk.to(device)


    # ========================================================
    # Upsample LR for network input
    # ========================================================

    lr_up = F.interpolate(
        lr,
        size=(484, 484),
        mode="bilinear",
        align_corners=False,
    )


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
    print("STARTING NO-RISK SAFE-SR TRAINING")
    print("========================================")

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Predict residual
        # ----------------------------------------------------

        residual = model(
            sr,
            lr_up,
            risk,
        )


        # ----------------------------------------------------
        # Refined SR
        # ----------------------------------------------------

        refined_sr = sr + residual


        # ----------------------------------------------------
        # HR reconstruction loss
        # ----------------------------------------------------

        hr_loss = F.l1_loss(
            refined_sr,
            hr,
        )


        # ----------------------------------------------------
        # OpenSR LR consistency loss
        # ----------------------------------------------------

        degraded = degradation_model.forward(
            refined_sr.squeeze(0),
        )

        # With one reflectance branch:
        #
        # degraded[0] = LR output
        # shape = [1, 4, 121, 121]

        refined_lr = degraded[0]

        if refined_lr.shape != lr.shape:
            raise ValueError(
                "OpenSR degraded LR shape mismatch: "
                f"{refined_lr.shape} vs {lr.shape}"
            )

        lr_loss = F.l1_loss(
            refined_lr,
            lr,
        )


        # ----------------------------------------------------
        # Residual regularization
        # ----------------------------------------------------

        residual_loss = torch.mean(
            torch.abs(residual)
        )


        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------

        loss = (
            LAMBDA_HR * hr_loss
            + LAMBDA_LR * lr_loss
            + LAMBDA_RESIDUAL * residual_loss
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Total: {loss.item():.6f} | "
                f"HR: {hr_loss.item():.6f} | "
                f"LR: {lr_loss.item():.6f} | "
                f"Residual: "
                f"{residual_loss.item():.6f}"
            )


    # ========================================================
    # Save model
    # ========================================================

    torch.save(
        model.state_dict(),
        MODEL_PATH,
    )

    print()
    print(
        f"Saved model: {MODEL_PATH}"
    )


    # ========================================================
    # Generate No-Risk SR
    # ========================================================

    model.eval()

    with torch.no_grad():

        residual = model(
            sr,
            lr_up,
            risk,
        )

        no_risk_sr = sr + residual


    # ========================================================
    # Validate
    # ========================================================

    if not torch.isfinite(
        no_risk_sr
    ).all():

        raise ValueError(
            "No-risk SR contains NaN or Inf."
        )

    print()
    print(
        f"No-risk SR shape: {no_risk_sr.shape}"
    )

    print(
        f"No-risk SR range: "
        f"{no_risk_sr.min().item():.6f} -> "
        f"{no_risk_sr.max().item():.6f}"
    )


    # ========================================================
    # Save outputs
    # ========================================================

    torch.save(
        no_risk_sr.cpu(),
        SAFE_SR_PATH,
    )

    torch.save(
        residual.cpu(),
        RESIDUAL_PATH,
    )

    print()
    print("Saved:")
    print(
        f"  {SAFE_SR_PATH}"
    )

    print(
        f"  {RESIDUAL_PATH}"
    )


if __name__ == "__main__":
    main()