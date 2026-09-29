import os
import sys
import random
import argparse

import torch
import torch.nn.functional as F
from torch.optim import Adam

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from refinement.safe_reconstruction import RiskAwareResidualNet
from refinement.patches import build_patch_index, crop_hr, crop_lr, augment
from degradation.opensr_degradation import create_naip_degradation_model


# ============================================================
# CONTROLLED 10-ROI ABLATION CONFIGURATION
# ============================================================

ROI_IDS = [
    "ROI_2319",
    "ROI_0630",
    "ROI_1897",
    "ROI_2224",
    "ROI_1570",
    "ROI_0141",
    "ROI_2317",
    "ROI_0091",
    "ROI_1884",
    "ROI_0582",
]

OUTPUT_DIR = "outputs/refinement/ablations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
PATCH_SIZE = 96
PATCH_STRIDE = 32

LAMBDA_HR = 1.0
LAMBDA_LR = 1.0
LAMBDA_RESIDUAL = 0.01
LAMBDA_RISK = 1.0

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# RISK MAP LOADING
# ============================================================

def load_risk_map(roi_id, risk_mode):
    """
    risk_mode:
        no-risk
        fidelity
        uncertainty
        fused
    """

    if risk_mode == "no-risk":
        return torch.zeros((1, 1, 484, 484), dtype=torch.float32)

    if risk_mode == "fidelity":
        path = f"outputs/risk/{roi_id}_fidelity_risk.pt"
        risk = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        ).float()

    elif risk_mode == "uncertainty":
        path = f"outputs/risk/{roi_id}_excess_uncertainty.pt"
        risk = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        ).float()

    elif risk_mode == "fused":
        fidelity_path = f"outputs/risk/{roi_id}_fidelity_risk.pt"
        uncertainty_path = f"outputs/risk/{roi_id}_excess_uncertainty.pt"

        fidelity = torch.load(
            fidelity_path,
            map_location="cpu",
            weights_only=True,
        ).float()

        uncertainty = torch.load(
            uncertainty_path,
            map_location="cpu",
            weights_only=True,
        ).float()

        risk = 0.5 * fidelity + 0.5 * uncertainty

        risk = torch.clamp(risk, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown risk mode: {risk_mode}")

    # Accept [H,W], [1,H,W], or [1,1,H,W].
    if risk.ndim == 2:
        risk = risk.unsqueeze(0).unsqueeze(0)

    elif risk.ndim == 3:
        risk = risk.unsqueeze(0)

    elif risk.ndim != 4:
        raise ValueError(
            f"{roi_id}: unexpected risk shape {risk.shape}"
        )

    if risk.shape != (1, 1, 484, 484):
        raise ValueError(
            f"{roi_id}: expected risk [1,1,484,484], "
            f"got {risk.shape}"
        )

    if not torch.isfinite(risk).all():
        raise ValueError(
            f"{roi_id}: risk contains NaN or Inf"
        )

    risk = torch.clamp(risk, 0.0, 1.0)

    return risk


# ============================================================
# LOAD ONE ROI
# ============================================================

def load_roi(roi_id, risk_mode, device):

    hr_path = f"data/processed/{roi_id}/hr.pt"
    lr_path = f"data/processed/{roi_id}/lr.pt"
    sr_path = f"outputs/sr/{roi_id}_sr.pt"

    for path in [hr_path, lr_path, sr_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{roi_id}: missing file:\n{path}"
            )

    hr = torch.load(
        hr_path,
        map_location="cpu",
        weights_only=True,
    ).float()

    lr = torch.load(
        lr_path,
        map_location="cpu",
        weights_only=True,
    ).float()

    sr = torch.load(
        sr_path,
        map_location="cpu",
        weights_only=True,
    ).float()

    if hr.ndim == 3:
        hr = hr.unsqueeze(0)

    if lr.ndim == 3:
        lr = lr.unsqueeze(0)

    if sr.ndim == 3:
        sr = sr.unsqueeze(0)

    if hr.shape != (1, 4, 484, 484):
        raise ValueError(
            f"{roi_id}: expected HR [1,4,484,484], "
            f"got {hr.shape}"
        )

    if lr.shape != (1, 4, 121, 121):
        raise ValueError(
            f"{roi_id}: expected LR [1,4,121,121], "
            f"got {lr.shape}"
        )

    if sr.shape != (1, 4, 484, 484):
        raise ValueError(
            f"{roi_id}: expected SR [1,4,484,484], "
            f"got {sr.shape}"
        )

    risk = load_risk_map(roi_id, risk_mode)

    lr_up = F.interpolate(
        lr,
        size=(484, 484),
        mode="bilinear",
        align_corners=False,
    )

    return {
        "roi_id": roi_id,
        "hr": hr.to(device),
        "lr": lr.to(device),
        "sr": sr.to(device),
        "lr_up": lr_up.to(device),
        "risk": risk.to(device),
    }


# ============================================================
# PATCH BATCH
# ============================================================

def gather_patch_batch(
    roi_data,
    batch_coords,
    patch_size,
    train,
):

    hr_list = []
    sr_list = []
    lr_up_list = []
    risk_list = []
    lr_list = []

    for roi_id, row, col in batch_coords:

        data = roi_data[roi_id]

        hr_p = crop_hr(
            data["hr"],
            row,
            col,
            patch_size,
        )

        sr_p = crop_hr(
            data["sr"],
            row,
            col,
            patch_size,
        )

        lr_up_p = crop_hr(
            data["lr_up"],
            row,
            col,
            patch_size,
        )

        risk_p = crop_hr(
            data["risk"],
            row,
            col,
            patch_size,
        )

        lr_p = crop_lr(
            data["lr"],
            row,
            col,
            patch_size,
        )

        if train:

            lr_p_up = F.interpolate(
                lr_p,
                size=(patch_size, patch_size),
                mode="nearest",
            )

            hr_p, sr_p, lr_up_p, risk_p, lr_p_up = augment(
                hr_p,
                sr_p,
                lr_up_p,
                risk_p,
                lr_p_up,
            )

            lr_p = F.avg_pool2d(
                lr_p_up,
                kernel_size=4,
            )

        hr_list.append(hr_p)
        sr_list.append(sr_p)
        lr_up_list.append(lr_up_p)
        risk_list.append(risk_p)
        lr_list.append(lr_p)

    return (
        torch.cat(hr_list, dim=0),
        torch.cat(sr_list, dim=0),
        torch.cat(lr_up_list, dim=0),
        torch.cat(risk_list, dim=0),
        torch.cat(lr_list, dim=0),
    )


# ============================================================
# DEGRADATION
# ============================================================

def apply_degradation_batch(
    refined_sr,
    degradation_model,
):

    outputs = []

    for i in range(refined_sr.shape[0]):

        single = refined_sr[i]

        degraded_lr, _ = degradation_model.forward(
            single
        )

        if degraded_lr.ndim == 3:
            degraded_lr = degraded_lr.unsqueeze(0)

        outputs.append(degraded_lr)

    return torch.cat(outputs, dim=0)


# ============================================================
# LOSS
# ============================================================

def compute_loss(
    model,
    hr,
    sr,
    lr_up,
    risk,
    lr,
    degradation_model,
):

    residual = model(
        sr,
        lr_up,
        risk,
    )

    refined_sr = sr + residual

    # Uniform HR reconstruction loss
    hr_loss = F.l1_loss(
        refined_sr,
        hr,
    )

    # Risk-weighted HR reconstruction loss
    risk_weighted_loss = torch.mean(
        risk * torch.abs(refined_sr - hr)
    )

    # LR consistency
    refined_lr = apply_degradation_batch(
        refined_sr,
        degradation_model,
    )

    if refined_lr.shape != lr.shape:
        raise ValueError(
            f"OpenSR LR shape mismatch: "
            f"{refined_lr.shape} vs {lr.shape}"
        )

    lr_loss = F.l1_loss(
        refined_lr,
        lr,
    )

    # Risk-scaled residual regularization
    residual_loss = torch.mean(
        (1.0 - risk) * torch.abs(residual)
    )

    total_loss = (
        LAMBDA_HR * hr_loss
        + LAMBDA_LR * lr_loss
        + LAMBDA_RESIDUAL * residual_loss
        + LAMBDA_RISK * risk_weighted_loss
    )

    return (
        total_loss,
        hr_loss,
        lr_loss,
        residual_loss,
        risk_weighted_loss,
        refined_sr,
    )


# ============================================================
# FULL-IMAGE VALIDATION
# ============================================================

def evaluate_full_images(
    model,
    roi_data,
    roi_ids,
):

    model.eval()

    total_hr = 0.0

    with torch.no_grad():

        for roi_id in roi_ids:

            data = roi_data[roi_id]

            residual = model(
                data["sr"],
                data["lr_up"],
                data["risk"],
            )

            refined_sr = data["sr"] + residual

            total_hr += F.l1_loss(
                refined_sr,
                data["hr"],
            ).item()

    return total_hr / len(roi_ids)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--risk-mode",
        choices=[
            "no-risk",
            "fidelity",
            "uncertainty",
            "fused",
        ],
        required=True,
    )

    parser.add_argument(
        "--sanity-only",
        action="store_true",
    )

    args = parser.parse_args()

    set_seed(SEED)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("SAFE-SR RISK COMPONENT ABLATION")
    print("=" * 60)

    print(f"Risk mode: {args.risk_mode}")
    print(f"Device: {device}")
    print(f"ROIs: {len(ROI_IDS)}")

    # --------------------------------------------------------
    # Deterministic 80/20 split
    # --------------------------------------------------------

    shuffled = ROI_IDS.copy()

    rng = random.Random(SEED)
    rng.shuffle(shuffled)

    split_index = int(
        0.8 * len(shuffled)
    )

    train_ids = shuffled[:split_index]
    val_ids = shuffled[split_index:]

    print("\nTraining ROIs:")
    for roi in train_ids:
        print(f"  {roi}")

    print("\nValidation ROIs:")
    for roi in val_ids:
        print(f"  {roi}")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading ROI data...")

    roi_data = {}

    for roi_id in ROI_IDS:

        print(f"  Loading {roi_id}...")

        roi_data[roi_id] = load_roi(
            roi_id,
            args.risk_mode,
            device,
        )

        risk = roi_data[roi_id]["risk"]

        print(
            f"    risk shape: {tuple(risk.shape)}"
        )

        print(
            f"    risk min: {risk.min().item():.6f}"
        )

        print(
            f"    risk max: {risk.max().item():.6f}"
        )

        print(
            f"    risk mean: {risk.mean().item():.6f}"
        )

    # --------------------------------------------------------
    # Sanity-only mode
    # --------------------------------------------------------

    if args.sanity_only:

        print("\n" + "=" * 60)
        print("SANITY CHECK PASSED")
        print("=" * 60)

        print(
            f"Risk mode '{args.risk_mode}' loaded successfully "
            f"for all {len(ROI_IDS)} ROIs."
        )

        return

    # --------------------------------------------------------
    # Patch index
    # --------------------------------------------------------

    train_patch_index = build_patch_index(
        train_ids,
        image_size=484,
        patch_size=PATCH_SIZE,
        stride=PATCH_STRIDE,
    )

    print(
        f"\nTraining patches per epoch: "
        f"{len(train_patch_index)}"
    )

    # --------------------------------------------------------
    # OpenSR degradation
    # --------------------------------------------------------

    print(
        "\nCreating OpenSR degradation model..."
    )

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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = RiskAwareResidualNet(
        in_channels=9,
        out_channels=4,
        features=64,
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    model_path = os.path.join(
        OUTPUT_DIR,
        f"multi_roi_{args.risk_mode}_best.pt",
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    best_val_hr = float("inf")

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        random.shuffle(
            train_patch_index
        )

        epoch_loss = 0.0
        epoch_hr = 0.0
        epoch_lr = 0.0
        epoch_res = 0.0
        epoch_risk = 0.0

        n_batches = 0

        for start in range(
            0,
            len(train_patch_index),
            BATCH_SIZE,
        ):

            batch_coords = train_patch_index[
                start:start + BATCH_SIZE
            ]

            (
                hr_b,
                sr_b,
                lr_up_b,
                risk_b,
                lr_b,
            ) = gather_patch_batch(
                roi_data,
                batch_coords,
                PATCH_SIZE,
                train=True,
            )

            optimizer.zero_grad()

            (
                loss,
                hr_loss,
                lr_loss,
                residual_loss,
                risk_loss,
                _,
            ) = compute_loss(
                model,
                hr_b,
                sr_b,
                lr_up_b,
                risk_b,
                lr_b,
                degradation_model,
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_hr += hr_loss.item()
            epoch_lr += lr_loss.item()
            epoch_res += residual_loss.item()
            epoch_risk += risk_loss.item()

            n_batches += 1

        epoch_loss /= n_batches
        epoch_hr /= n_batches
        epoch_lr /= n_batches
        epoch_res /= n_batches
        epoch_risk /= n_batches

        val_hr = evaluate_full_images(
            model,
            roi_data,
            val_ids,
        )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Train: {epoch_loss:.6f} | "
                f"HR: {epoch_hr:.6f} | "
                f"LR: {epoch_lr:.6f} | "
                f"Res: {epoch_res:.6f} | "
                f"RiskW: {epoch_risk:.6f} | "
                f"Val HR: {val_hr:.6f}"
            )

        if val_hr < best_val_hr:

            best_val_hr = val_hr

            torch.save(
                model.state_dict(),
                model_path,
            )

    # --------------------------------------------------------
    # Generate validation outputs
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(
        f"Best validation HR L1: "
        f"{best_val_hr:.6f}"
    )

    print(
        f"Saved: {model_path}"
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()

    print(
        "\nGenerating validation Safe-SR..."
    )

    with torch.no_grad():

        for roi_id in val_ids:

            data = roi_data[roi_id]

            residual = model(
                data["sr"],
                data["lr_up"],
                data["risk"],
            )

            safe_sr = data["sr"] + residual

            if not torch.isfinite(
                safe_sr
            ).all():

                raise ValueError(
                    f"{roi_id}: Safe-SR contains "
                    "NaN or Inf."
                )

            output_path = os.path.join(
                OUTPUT_DIR,
                f"{roi_id}_{args.risk_mode}_safe_sr.pt",
            )

            torch.save(
                safe_sr.cpu(),
                output_path,
            )

            print(
                f"  Saved {roi_id}: "
                f"{output_path}"
            )

    print("\n" + "=" * 60)
    print("ABLATION TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()