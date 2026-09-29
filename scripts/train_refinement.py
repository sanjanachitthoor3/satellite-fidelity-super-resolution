import os
import sys
import argparse
import random

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
# Configuration
# ============================================================

ROI_IDS = [
    "ROI_2319", "ROI_0630", "ROI_1897", "ROI_2224", "ROI_1570",
    "ROI_0141", "ROI_2317", "ROI_0091", "ROI_1884", "ROI_0582",
]

OUTPUT_DIR = "outputs/refinement"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 16          # patches per step -- adjust down if VRAM-limited
PATCH_SIZE = 96
PATCH_STRIDE = 32        # train-time stride; smaller = more (correlated) patches

LAMBDA_HR = 1.0
LAMBDA_LR = 1.0
LAMBDA_RESIDUAL = 0.01
LAMBDA_RISK = 1.0        # weight on the risk-weighted HR loss term

SEED = 42

# ============================================================
# Reproducibility
# ============================================================


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Load one ROI (full image, kept resident for patch cropping)
# ============================================================


def load_roi(roi_id, mode, device):
    hr_path = f"data/processed/{roi_id}/hr.pt"
    lr_path = f"data/processed/{roi_id}/lr.pt"
    sr_path = f"outputs/sr/{roi_id}_sr.pt"

    hr = torch.load(hr_path, map_location="cpu", weights_only=True).float()
    lr = torch.load(lr_path, map_location="cpu", weights_only=True).float()
    sr = torch.load(sr_path, map_location="cpu", weights_only=True).float()

    if hr.ndim == 3:
        hr = hr.unsqueeze(0)
    if lr.ndim == 3:
        lr = lr.unsqueeze(0)
    if sr.ndim == 3:
        sr = sr.unsqueeze(0)

    if hr.shape != (1, 4, 484, 484):
        raise ValueError(f"{roi_id}: Expected HR [1,4,484,484], got {hr.shape}")
    if lr.shape != (1, 4, 121, 121):
        raise ValueError(f"{roi_id}: Expected LR [1,4,121,121], got {lr.shape}")
    if sr.shape != (1, 4, 484, 484):
        raise ValueError(f"{roi_id}: Expected SR [1,4,484,484], got {sr.shape}")

    lr_up = F.interpolate(lr, size=(484, 484), mode="bilinear", align_corners=False)

    # ALWAYS load a risk tensor. In no-risk mode it's zeros, but the
    # model still sees a real 9th channel -- see safe_reconstruction.py
    # for why this matters for a clean ablation.
    if mode == "risk":
        risk_path = f"outputs/risk/{roi_id}_risk_map_normalized.pt"
        risk = torch.load(risk_path, map_location="cpu", weights_only=True).float()
        if risk.shape != (484, 484):
            raise ValueError(f"{roi_id}: Expected risk [484,484], got {risk.shape}")
        risk = risk.unsqueeze(0).unsqueeze(0)
    else:
        risk = torch.zeros((1, 1, 484, 484), dtype=hr.dtype)

    return {
        "roi_id": roi_id,
        "hr": hr.to(device),
        "lr": lr.to(device),
        "sr": sr.to(device),
        "lr_up": lr_up.to(device),
        "risk": risk.to(device),
    }


# ============================================================
# Patch batch assembly
# ============================================================


def gather_patch_batch(roi_data, batch_coords, patch_size, train):
    """
    roi_data: dict roi_id -> loaded full-image tensors (from load_roi)
    batch_coords: list of (roi_id, row, col)
    train: if True, apply one shared random flip/rotation per patch to
           hr/sr/lr_up/risk/lr together, so the low-res LR crop stays
           spatially aligned with the high-res crops.
    Returns stacked [B,4,P,P] hr/sr/lr_up, [B,1,P,P] risk, [B,4,P/4,P/4] lr
    """
    hr_list, sr_list, lr_up_list, risk_list, lr_list = [], [], [], [], []

    for roi_id, row, col in batch_coords:
        data = roi_data[roi_id]

        hr_p = crop_hr(data["hr"], row, col, patch_size)
        sr_p = crop_hr(data["sr"], row, col, patch_size)
        lr_up_p = crop_hr(data["lr_up"], row, col, patch_size)
        risk_p = crop_hr(data["risk"], row, col, patch_size)
        lr_p = crop_lr(data["lr"], row, col, patch_size)

        if train:
            # lr_p is 1/4 the spatial size of the rest, but flips and
            # 90-degree rotations commute with downsampling by an
            # integer factor, so applying the *same random choices* to
            # both groups keeps them aligned even though augment()
            # re-rolls its random choices per call. We work around that
            # by temporarily upsampling lr_p to patch_size, augmenting
            # everything together, then downsampling back.
            lr_p_up = F.interpolate(
                lr_p, size=(patch_size, patch_size), mode="nearest"
            )
            hr_p, sr_p, lr_up_p, risk_p, lr_p_up = augment(
                hr_p, sr_p, lr_up_p, risk_p, lr_p_up
            )
            # Downsample the (now-augmented) upsampled LR back to P/4.
            lr_p = F.avg_pool2d(lr_p_up, kernel_size=4)

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
# Degradation-model call (per-sample -- see justification below)
# ============================================================


def apply_degradation_batch(refined_sr, degradation_model):
    """
    Apply `degradation_model.forward()` one image at a time and stack
    the results back into a batch.

    Why not a single batched call: every existing call site in the
    code you provided -- the original train_refinement.py
    (`degraded = degradation_model.forward(refined_sr.squeeze(0));
    refined_lr = degraded[0]`) and final_evaluation.py
    (`degraded_lr, _ = degradation_model.forward(sr_input)`) -- passes
    an UNBATCHED [C,H,W] tensor and unpacks the same 2-tuple return.
    Neither is a counterexample to the other; they're the same
    convention written two ways. Nothing in the code given to us
    calls this with a batch dimension, so treating it as
    batch-capable would be an assumption, not a verified fact.

    This loops over the batch using the exact single-image convention
    that IS proven correct elsewhere in this codebase. It costs a
    Python loop over `degradation_model.forward()` (cheap relative to
    the LDSR-S2 diffusion model -- this is just the blur/downsample/
    noise degradation, not the SR network), not a shape gamble.

    If your actual opensr_degradation wrapper has a verified batched
    entry point (e.g. a `.pipe(...).forward()` that explicitly
    documents/tests [B,C,H,W] support), replace this loop with a
    single vectorized call -- but do that with the wrapper's source
    in hand, not by assumption.
    """
    outputs = []
    for i in range(refined_sr.shape[0]):
        single = refined_sr[i]  # [C,H,W] -- matches the proven convention
        degraded_lr, _ = degradation_model.forward(single)
        if degraded_lr.ndim == 3:
            degraded_lr = degraded_lr.unsqueeze(0)
        outputs.append(degraded_lr)
    return torch.cat(outputs, dim=0)


# ============================================================
# Loss
# ============================================================


def compute_loss(model, hr, sr, lr_up, risk, lr, degradation_model):
    residual = model(sr, lr_up, risk)
    refined_sr = sr + residual

    # --- Uniform HR reconstruction loss (unchanged) ---
    hr_loss = F.l1_loss(refined_sr, hr)

    # --- Risk-weighted HR loss: NEW. Explicitly rewards getting
    # high-risk pixels right, instead of hoping the network infers
    # that purely from risk being an input channel.
    #
    # Scale check: with LAMBDA_HR=1.0 and LAMBDA_RISK=1.0, the two
    # HR-related terms algebraically combine to
    #     LAMBDA_HR*hr_loss + LAMBDA_RISK*risk_weighted_loss
    #   = mean(|e|) + mean(risk*|e|) = mean((1 + risk) * |e|)
    # where e = refined_sr - hr and risk in [0,1] (clamped at
    # generation time). So LAMBDA_RISK=1.0 means a risk=1 pixel gets
    # exactly 2x the reconstruction weight of a risk=0 pixel, and a
    # risk=0 pixel is untouched (weight stays 1x). This is a bounded,
    # interpretable multiplier -- it cannot make this term dominate
    # or destabilize the loss, since lr_loss carries the same weight
    # (1.0) and is the same order of magnitude as hr_loss. Lower
    # LAMBDA_RISK (e.g. 0.3-0.5) for a milder max 1.3x-1.5x weighting
    # if you want a more conservative first run. ---
    risk_weighted_loss = torch.mean(risk * torch.abs(refined_sr - hr))

    # --- OpenSR LR consistency, per-sample (see apply_degradation_batch
    # docstring for why this loops rather than assumes batching). This
    # is computed per-patch, so the degradation model's receptive field
    # can bleed slightly across a patch boundary (e.g. from its blur
    # kernel) -- acceptable for training gradients, but trust the
    # full-image consistency number in final_evaluation.py as the
    # reported metric, not this patch-level one. ---
    refined_lr = apply_degradation_batch(refined_sr, degradation_model)
    if refined_lr.shape != lr.shape:
        raise ValueError(
            f"OpenSR LR shape mismatch: {refined_lr.shape} vs {lr.shape}"
        )
    lr_loss = F.l1_loss(refined_lr, lr)

    # --- Residual regularization, risk-scaled: NEW, narrower claim
    # than before. The gradient of this term w.r.t. residual is
    #     -LAMBDA_RESIDUAL * (1 - risk) * sign(residual)
    # which VANISHES as risk -> 1. This does NOT create upward
    # pressure on residual magnitude anywhere -- an L1 penalty only
    # ever pulls toward zero. What it does do: it removes that
    # pull-toward-zero specifically at high-risk pixels, so it no
    # longer works AGAINST whatever correction magnitude hr_loss /
    # risk_weighted_loss otherwise call for there. At risk=0 pixels,
    # behavior is identical to the original uniform penalty. ---
    residual_loss = torch.mean((1.0 - risk) * torch.abs(residual))

    total_loss = (
        LAMBDA_HR * hr_loss
        + LAMBDA_LR * lr_loss
        + LAMBDA_RESIDUAL * residual_loss
        + LAMBDA_RISK * risk_weighted_loss
    )

    return total_loss, hr_loss, lr_loss, residual_loss, risk_weighted_loss, refined_sr


# ============================================================
# Full-image evaluation (unchanged in spirit -- used for model
# selection and for writing out full safe_sr.pt files consumed by
# evaluation/final_evaluation.py)
# ============================================================


def evaluate_full_images(model, roi_data, roi_ids, degradation_model):
    model.eval()
    total_hr = 0.0
    with torch.no_grad():
        for roi_id in roi_ids:
            data = roi_data[roi_id]
            residual = model(data["sr"], data["lr_up"], data["risk"])
            refined_sr = data["sr"] + residual
            total_hr += F.l1_loss(refined_sr, data["hr"]).item()
    return total_hr / len(roi_ids)


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["risk", "no-risk"], required=True)
    args = parser.parse_args()

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("========================================")
    print("MULTI-ROI SAFE-SR TRAINING (patch-based)")
    print("========================================")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Total ROIs: {len(ROI_IDS)}")
    print(f"Patch size: {PATCH_SIZE}  stride: {PATCH_STRIDE}  batch: {BATCH_SIZE}")

    # --------------------------------------------------------
    # Deterministic 80/20 split (same split logic as before)
    # --------------------------------------------------------
    shuffled = ROI_IDS.copy()
    rng = random.Random(SEED)
    rng.shuffle(shuffled)
    split_index = int(0.8 * len(shuffled))
    train_ids = shuffled[:split_index]
    val_ids = shuffled[split_index:]

    print("\nTraining ROIs:", train_ids)
    print("Validation ROIs:", val_ids)

    # --------------------------------------------------------
    # Load all ROI full images once (kept resident for cropping)
    # --------------------------------------------------------
    print("\nLoading ROI data...")
    roi_data = {roi: load_roi(roi, args.mode, device) for roi in ROI_IDS}

    # --------------------------------------------------------
    # Build train patch index (many patches, small stride)
    # Validation still uses full images -- see evaluate_full_images.
    # --------------------------------------------------------
    train_patch_index = build_patch_index(
        train_ids, image_size=484, patch_size=PATCH_SIZE, stride=PATCH_STRIDE
    )
    print(f"Train patches per epoch: {len(train_patch_index)}")

    # --------------------------------------------------------
    # OpenSR degradation model
    # --------------------------------------------------------
    degradation_model = create_naip_degradation_model(
        device=device, seed=SEED, add_noise=True,
        reflectance_methods=["gamma_multivariate_normal"],
    )

    # --------------------------------------------------------
    # Model -- always in_channels=9 (see safe_reconstruction.py)
    # --------------------------------------------------------
    model = RiskAwareResidualNet(in_channels=9, out_channels=4, features=64).to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    model_path = os.path.join(OUTPUT_DIR, f"multi_roi_{args.mode}_best.pt")

    print("\n========================================")
    print("STARTING TRAINING")
    print("========================================")

    best_val_hr = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        random.shuffle(train_patch_index)

        epoch_loss = epoch_hr = epoch_lr = epoch_res = epoch_risk = 0.0
        n_batches = 0

        for start in range(0, len(train_patch_index), BATCH_SIZE):
            batch_coords = train_patch_index[start:start + BATCH_SIZE]

            hr_b, sr_b, lr_up_b, risk_b, lr_b = gather_patch_batch(
                roi_data, batch_coords, PATCH_SIZE, train=True
            )

            optimizer.zero_grad()
            (loss, hr_loss, lr_loss, residual_loss, risk_loss, _) = compute_loss(
                model, hr_b, sr_b, lr_up_b, risk_b, lr_b, degradation_model
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

        val_hr = evaluate_full_images(model, roi_data, val_ids, degradation_model)

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Train: {epoch_loss:.6f} | HR: {epoch_hr:.6f} | "
                f"LR: {epoch_lr:.6f} | Res: {epoch_res:.6f} | "
                f"RiskW: {epoch_risk:.6f} | Val HR: {val_hr:.6f}"
            )

        if val_hr < best_val_hr:
            best_val_hr = val_hr
            torch.save(model.state_dict(), model_path)

    print("\n========================================")
    print("BEST MODEL")
    print("========================================")
    print(f"Best validation HR L1: {best_val_hr:.6f}")
    print(f"Saved: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    print("\nGenerating validation Safe-SR...")
    with torch.no_grad():
        for roi_id in val_ids:
            data = roi_data[roi_id]
            residual = model(data["sr"], data["lr_up"], data["risk"])
            safe_sr = data["sr"] + residual

            if not torch.isfinite(safe_sr).all():
                raise ValueError(f"{roi_id}: Safe-SR contains NaN or Inf.")

            output_path = os.path.join(OUTPUT_DIR, f"{roi_id}_{args.mode}_safe_sr.pt")
            torch.save(safe_sr.cpu(), output_path)
            print(f"  Saved {roi_id}: {output_path}")

    print("\n========================================")
    print("TRAINING COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()