import os
import sys
import argparse
import random

import torch
import torch.nn.functional as F
from torch.optim import Adam

PROJECT_ROOT = os.path.dirname(
os.path.dirname(os.path.abspath(file))
)

if PROJECT_ROOT not in sys.path:
sys.path.insert(0, PROJECT_ROOT)

from refinement.safe_reconstruction import RiskAwareResidualNet
from degradation.opensr_degradation import (
create_naip_degradation_model,
)

============================================================

Configuration

============================================================

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

OUTPUT_DIR = "outputs/refinement"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 100
LEARNING_RATE = 1e-4

LAMBDA_HR = 1.0
LAMBDA_LR = 1.0
LAMBDA_RESIDUAL = 0.01

SEED = 42

============================================================

Reproducibility

============================================================

def set_seed(seed):
random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

============================================================

Load one ROI

============================================================

def load_roi(roi_id, mode, device):

hr_path = f"data/processed/{roi_id}/hr.pt"
lr_path = f"data/processed/{roi_id}/lr.pt"
sr_path = f"outputs/sr/{roi_id}_sr.pt"

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
        f"{roi_id}: Expected HR [1,4,484,484], "
        f"got {hr.shape}"
    )

if lr.shape != (1, 4, 121, 121):
    raise ValueError(
        f"{roi_id}: Expected LR [1,4,121,121], "
        f"got {lr.shape}"
    )

if sr.shape != (1, 4, 484, 484):
    raise ValueError(
        f"{roi_id}: Expected SR [1,4,484,484], "
        f"got {sr.shape}"
    )

lr_up = F.interpolate(
    lr,
    size=(484, 484),
    mode="bilinear",
    align_corners=False,
)

data = {
    "roi_id": roi_id,
    "hr": hr.to(device),
    "lr": lr.to(device),
    "sr": sr.to(device),
    "lr_up": lr_up.to(device),
}

if mode == "risk":

    risk_path = (
        f"outputs/risk/"
        f"{roi_id}_risk_map_normalized.pt"
    )

    risk = torch.load(
        risk_path,
        map_location="cpu",
        weights_only=True,
    ).float()

    if risk.shape != (484, 484):
        raise ValueError(
            f"{roi_id}: Expected risk [484,484], "
            f"got {risk.shape}"
        )

    risk = (
        risk
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    data["risk"] = risk

return data

============================================================

Create model

============================================================

def create_model(mode, device):

if mode == "risk":
    in_channels = 9
else:
    in_channels = 8

model = RiskAwareResidualNet(
    in_channels=in_channels,
    out_channels=4,
    features=64,
).to(device)

return model

============================================================

Forward pass

============================================================

def forward_roi(
model,
data,
mode,
degradation_model,
):

sr = data["sr"]
lr_up = data["lr_up"]
lr = data["lr"]
hr = data["hr"]

if mode == "risk":

    residual = model(
        sr,
        lr_up,
        data["risk"],
    )

else:

    # No-risk baseline.
    # Risk channel is completely absent.
    residual = model(
        sr,
        lr_up,
    )

refined_sr = sr + residual

# --------------------------------------------------------
# HR reconstruction loss
# --------------------------------------------------------

hr_loss = F.l1_loss(
    refined_sr,
    hr,
)

# --------------------------------------------------------
# OpenSR LR consistency
# --------------------------------------------------------

degraded = degradation_model.forward(
    refined_sr.squeeze(0),
)

refined_lr = degraded[0]

if refined_lr.shape != lr.shape:
    raise ValueError(
        f"OpenSR LR shape mismatch: "
        f"{refined_lr.shape} vs {lr.shape}"
    )

lr_loss = F.l1_loss(
    refined_lr,
    lr,
)

# --------------------------------------------------------
# Residual regularization
# --------------------------------------------------------

residual_loss = torch.mean(
    torch.abs(residual)
)

# --------------------------------------------------------
# Total loss
# --------------------------------------------------------

total_loss = (
    LAMBDA_HR * hr_loss
    + LAMBDA_LR * lr_loss
    + LAMBDA_RESIDUAL * residual_loss
)

return (
    total_loss,
    hr_loss,
    lr_loss,
    residual_loss,
    refined_sr,
)

============================================================

Evaluate

============================================================

def evaluate(
model,
validation_data,
mode,
degradation_model,
):

model.eval()

total_hr = 0.0
total_lr = 0.0
total_loss = 0.0

with torch.no_grad():

    for data in validation_data:

        (
            loss,
            hr_loss,
            lr_loss,
            residual_loss,
            refined_sr,
        ) = forward_roi(
            model,
            data,
            mode,
            degradation_model,
        )

        total_loss += loss.item()
        total_hr += hr_loss.item()
        total_lr += lr_loss.item()

n = len(validation_data)

return (
    total_loss / n,
    total_hr / n,
    total_lr / n,
)

============================================================

Main

============================================================

def main():

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    choices=["risk", "no-risk"],
    required=True,
)

args = parser.parse_args()

set_seed(SEED)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("========================================")
print("MULTI-ROI SAFE-SR TRAINING")
print("========================================")
print(f"Mode: {args.mode}")
print(f"Device: {device}")
print(f"Total ROIs: {len(ROI_IDS)}")

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

print()
print("Training ROIs:")
for roi in train_ids:
    print(f"  {roi}")

print()
print("Validation ROIs:")
for roi in val_ids:
    print(f"  {roi}")

# --------------------------------------------------------
# Load data
# --------------------------------------------------------

print()
print("Loading training data...")

train_data = [
    load_roi(
        roi,
        args.mode,
        device,
    )
    for roi in train_ids
]

print("Loading validation data...")

val_data = [
    load_roi(
        roi,
        args.mode,
        device,
    )
    for roi in val_ids
]

# --------------------------------------------------------
# OpenSR degradation
# --------------------------------------------------------

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
    "OpenSR degradation branch: "
    "gamma_multivariate_normal"
)

# --------------------------------------------------------
# Model
# --------------------------------------------------------

model = create_model(
    args.mode,
    device,
)

model.train()

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

# --------------------------------------------------------
# Output paths
# --------------------------------------------------------

model_path = os.path.join(
    OUTPUT_DIR,
    f"multi_roi_{args.mode}_best.pt",
)

# --------------------------------------------------------
# Training
# --------------------------------------------------------

print()
print("========================================")
print("STARTING TRAINING")
print("========================================")

best_val_hr = float("inf")

for epoch in range(
    1,
    EPOCHS + 1,
):

    model.train()

    epoch_loss = 0.0
    epoch_hr = 0.0
    epoch_lr = 0.0
    epoch_residual = 0.0

    for data in train_data:

        optimizer.zero_grad()

        (
            loss,
            hr_loss,
            lr_loss,
            residual_loss,
            refined_sr,
        ) = forward_roi(
            model,
            data,
            args.mode,
            degradation_model,
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_hr += hr_loss.item()
        epoch_lr += lr_loss.item()
        epoch_residual += residual_loss.item()

    n_train = len(train_data)

    epoch_loss /= n_train
    epoch_hr /= n_train
    epoch_lr /= n_train
    epoch_residual /= n_train

    # ----------------------------------------------------
    # Validation
    # ----------------------------------------------------

    (
        val_loss,
        val_hr,
        val_lr,
    ) = evaluate(
        model,
        val_data,
        args.mode,
        degradation_model,
    )

    if (
        epoch == 1
        or epoch % 10 == 0
        or epoch == EPOCHS
    ):

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train: {epoch_loss:.6f} | "
            f"Train HR: {epoch_hr:.6f} | "
            f"Train LR: {epoch_lr:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"Val HR: {val_hr:.6f} | "
            f"Val LR: {val_lr:.6f}"
        )

    # ----------------------------------------------------
    # Save best model
    # ----------------------------------------------------

    if val_hr < best_val_hr:

        best_val_hr = val_hr

        torch.save(
            model.state_dict(),
            model_path,
        )

# ========================================================
# Load best model
# ========================================================

print()
print("========================================")
print("BEST MODEL")
print("========================================")

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

# ========================================================
# Generate validation Safe-SR
# ========================================================

print()
print("Generating validation Safe-SR...")

with torch.no_grad():

    for data in val_data:

        roi_id = data["roi_id"]

        if args.mode == "risk":

            residual = model(
                data["sr"],
                data["lr_up"],
                data["risk"],
            )

        else:

            residual = model(
                data["sr"],
                data["lr_up"],
            )

        safe_sr = (
            data["sr"] + residual
        )

        if not torch.isfinite(
            safe_sr
        ).all():

            raise ValueError(
                f"{roi_id}: "
                "Safe-SR contains NaN or Inf."
            )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{roi_id}_{args.mode}_safe_sr.pt",
        )

        torch.save(
            safe_sr.cpu(),
            output_path,
        )

        print(
            f"  Saved {roi_id}: "
            f"{output_path}"
        )

print()
print("========================================")
print("TRAINING COMPLETE")
print("========================================")

if name == "main":
main()