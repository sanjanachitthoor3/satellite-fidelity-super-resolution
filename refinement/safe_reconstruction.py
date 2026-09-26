import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
SR_PATH = "outputs/sr/sr_demo.pt"
LR_PATH = "outputs/sr/lr_input.pt"
RISK_PATH = "outputs/risk/risk_map_normalized.pt"

OUTPUT_DIR = "outputs/refinement"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "safe_sr.pt",
)


# ---------------------------------------------------------
# Risk-Aware Residual Network
# ---------------------------------------------------------
class RiskAwareResidualNet(nn.Module):
    """
    Residual refinement network.

    Risk-aware mode:
        SR (4) + LR_up (4) + risk (1) = 9 channels

    No-risk mode:
        SR (4) + LR_up (4) = 8 channels
    """

    def __init__(self, in_channels=9, out_channels=4, features=64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                features,
                features,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                features,
                features,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                features,
                features,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                features,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, sr, lr_up, risk=None):

        if risk is None:
            x = torch.cat(
                [sr, lr_up],
                dim=1,
            )
        else:
            x = torch.cat(
                [sr, lr_up, risk],
                dim=1,
            )

        return self.network(x)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # -----------------------------------------------------
    # Load SR
    # -----------------------------------------------------
    sr = torch.load(
        SR_PATH,
        map_location="cpu",
        weights_only=False,
    ).float()

    # Expected:
    # [1, 4, 512, 512]

    if sr.ndim != 4:
        raise ValueError(
            f"Expected SR to be 4D, got {sr.shape}"
        )

    if sr.shape[1] != 4:
        raise ValueError(
            f"Expected 4 SR channels, got {sr.shape[1]}"
        )

    print(f"SR shape: {sr.shape}")

    # -----------------------------------------------------
    # Load LR
    # -----------------------------------------------------
    lr = torch.load(
        LR_PATH,
        map_location="cpu",
        weights_only=False,
    ).float()

    # Expected:
    # [1, 4, 128, 128]

    if lr.ndim != 4:
        raise ValueError(
            f"Expected LR to be 4D, got {lr.shape}"
        )

    if lr.shape[1] != 4:
        raise ValueError(
            f"Expected 4 LR channels, got {lr.shape[1]}"
        )

    print(f"LR shape: {lr.shape}")

    # -----------------------------------------------------
    # Upsample LR to SR resolution
    # -----------------------------------------------------
    lr_up = F.interpolate(
        lr,
        size=(512, 512),
        mode="bilinear",
        align_corners=False,
    )

    print(f"Upsampled LR shape: {lr_up.shape}")

    # -----------------------------------------------------
    # Load risk map
    # -----------------------------------------------------
    risk = torch.load(
        RISK_PATH,
        map_location="cpu",
        weights_only=False,
    ).float()

    # Expected:
    # [484, 484]

    if risk.ndim != 2:
        raise ValueError(
            f"Expected risk map to be 2D, got {risk.shape}"
        )

    if risk.shape != (484, 484):
        raise ValueError(
            f"Expected risk map [484, 484], got {risk.shape}"
        )

    print(f"Risk shape: {risk.shape}")

    # -----------------------------------------------------
    # Construct full-resolution risk map
    # -----------------------------------------------------
    #
    # The 484x484 risk map corresponds to the valid
    # non-padded region of the SR image.
    #
    # LDSR-S2 operates on a padded 128x128 LR input:
    #
    #     121x121 valid LR
    #     + 7 pixels padding
    #
    # After 4x SR:
    #
    #     484x484 valid region
    #     + 28 pixel padded border
    #
    # Therefore we do NOT invent risk values for the
    # 28-pixel border.
    #
    # Border risk = 0
    # -----------------------------------------------------

    risk_full = torch.zeros(
        (
            1,
            1,
            sr.shape[2],
            sr.shape[3],
        ),
        dtype=sr.dtype,
    )

    risk_full[
        :,
        :,
        :risk.shape[0],
        :risk.shape[1],
    ] = risk.unsqueeze(0).unsqueeze(0)

    print(f"Full risk shape: {risk_full.shape}")

    # -----------------------------------------------------
    # Move tensors to device
    # -----------------------------------------------------
    sr = sr.to(device)
    lr_up = lr_up.to(device)
    risk_full = risk_full.to(device)

    # -----------------------------------------------------
    # Create residual network
    # -----------------------------------------------------
    model = RiskAwareResidualNet(
        in_channels=9,
        out_channels=4,
        features=64,
    ).to(device)

    model.eval()

    # -----------------------------------------------------
    # Residual prediction
    # -----------------------------------------------------
    #
    # IMPORTANT:
    # This network is currently untrained.
    #
    # We are creating the architecture and verifying
    # the complete forward pass first.
    #
    # The learned version will later load trained
    # residual-network weights.
    # -----------------------------------------------------

    with torch.no_grad():

        residual = model(
            sr,
            lr_up,
            risk_full,
        )

        print(
            f"Residual shape: {residual.shape}"
        )

        # -------------------------------------------------
        # Safe reconstruction
        # -------------------------------------------------
        safe_sr = sr + residual

    # -----------------------------------------------------
    # Verify output
    # -----------------------------------------------------
    if not torch.isfinite(safe_sr).all():
        raise ValueError(
            "Safe-SR contains NaN or Inf values."
        )

    print(
        f"Safe-SR shape: {safe_sr.shape}"
    )

    print(
        f"SR range: "
        f"{sr.min().item():.6f} → "
        f"{sr.max().item():.6f}"
    )

    print(
        f"Residual range: "
        f"{residual.min().item():.6f} → "
        f"{residual.max().item():.6f}"
    )

    print(
        f"Safe-SR range: "
        f"{safe_sr.min().item():.6f} → "
        f"{safe_sr.max().item():.6f}"
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------
    torch.save(
        safe_sr.cpu(),
        OUTPUT_PATH,
    )

    torch.save(
        residual.cpu(),
        os.path.join(
            OUTPUT_DIR,
            "predicted_residual.pt",
        ),
    )

    torch.save(
        risk_full.cpu(),
        os.path.join(
            OUTPUT_DIR,
            "risk_full_512.pt",
        ),
    )

    print()
    print("Saved:")
    print(f"  {OUTPUT_PATH}")
    print(
        f"  {OUTPUT_DIR}/predicted_residual.pt"
    )
    print(
        f"  {OUTPUT_DIR}/risk_full_512.pt"
    )


if __name__ == "__main__":
    main()