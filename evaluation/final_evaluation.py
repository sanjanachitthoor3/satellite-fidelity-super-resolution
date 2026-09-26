import os
import json
import sys

import torch
from skimage.metrics import structural_similarity

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from degradation.opensr_degradation import (
    create_naip_degradation_model,
)


# ============================================================
# Configuration
# ============================================================

VAL_ROIS = [
    "ROI_2319",
    "ROI_0630",
]

OUTPUT_DIR = "outputs/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42


# ============================================================
# Helpers
# ============================================================

def load_tensor(path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    ).float()


def standardize_batch(x):
    if x.ndim == 3:
        x = x.unsqueeze(0)

    return x


def clamp_image(x):
    return torch.clamp(
        x,
        0.0,
        1.0,
    )


def psnr(pred, target):

    mse = torch.mean(
        (pred - target) ** 2
    )

    if mse <= 0:
        return float("inf")

    return float(
        10.0 * torch.log10(
            1.0 / mse
        )
    )


def ssim(pred, target):

    pred = pred.squeeze(0)
    target = target.squeeze(0)

    pred_np = pred.permute(
        1, 2, 0
    ).numpy()

    target_np = target.permute(
        1, 2, 0
    ).numpy()

    values = []

    for c in range(
        pred_np.shape[-1]
    ):

        values.append(
            structural_similarity(
                target_np[:, :, c],
                pred_np[:, :, c],
                data_range=1.0,
            )
        )

    return float(
        sum(values) / len(values)
    )


def opensr_lr_consistency(
    sr,
    lr,
    degradation_model,
):

    sr_input = sr.squeeze(0)

    degraded_lr, _ = (
        degradation_model.forward(
            sr_input
        )
    )

    if degraded_lr.ndim == 3:
        degraded_lr = (
            degraded_lr.unsqueeze(0)
        )

    if degraded_lr.shape != lr.shape:

        raise ValueError(
            "OpenSR degraded LR shape mismatch: "
            f"{degraded_lr.shape} vs "
            f"{lr.shape}"
        )

    return float(
        torch.mean(
            torch.abs(
                degraded_lr - lr
            )
        )
    )


def evaluate_image(
    prediction,
    hr,
    lr,
    degradation_model,
):

    prediction = clamp_image(
        prediction
    )

    hr = clamp_image(hr)

    return {
        "PSNR_dB": psnr(
            prediction,
            hr,
        ),
        "SSIM": ssim(
            prediction,
            hr,
        ),
        "OpenSR_consistency_L1":
            opensr_lr_consistency(
                prediction,
                lr,
                degradation_model,
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("========================================")
    print("MULTI-ROI SAFE-SR EVALUATION")
    print("========================================")

    print()
    print("Validation ROIs:")

    for roi in VAL_ROIS:
        print(f"  {roi}")


    # ========================================================
    # Create OpenSR evaluation model
    # ========================================================

    print()
    print(
        "Creating OpenSR evaluation model..."
    )

    degradation_model = (
        create_naip_degradation_model(
            device="cpu",
            seed=SEED,
            add_noise=True,
            reflectance_methods=[
                "gamma_multivariate_normal"
            ],
        )
    )

    print(
        "OpenSR evaluation model ready."
    )


    # ========================================================
    # Results storage
    # ========================================================

    all_results = {}

    baseline_values = {
        "PSNR_dB": [],
        "SSIM": [],
        "OpenSR_consistency_L1": [],
    }

    no_risk_values = {
        "PSNR_dB": [],
        "SSIM": [],
        "OpenSR_consistency_L1": [],
    }

    risk_values = {
        "PSNR_dB": [],
        "SSIM": [],
        "OpenSR_consistency_L1": [],
    }


    # ========================================================
    # Evaluate each validation ROI
    # ========================================================

    for roi in VAL_ROIS:

        print()
        print("========================================")
        print(f"EVALUATING {roi}")
        print("========================================")

        hr_path = (
            f"data/processed/"
            f"{roi}/hr.pt"
        )

        lr_path = (
            f"data/processed/"
            f"{roi}/lr.pt"
        )

        sr_path = (
            f"outputs/sr/"
            f"{roi}_sr.pt"
        )

        no_risk_path = (
            f"outputs/refinement/"
            f"{roi}_no-risk_safe_sr.pt"
        )

        risk_path = (
            f"outputs/refinement/"
            f"{roi}_risk_safe_sr.pt"
        )


        # ----------------------------------------------------
        # Load tensors
        # ----------------------------------------------------

        hr = standardize_batch(
            load_tensor(hr_path)
        )

        lr = standardize_batch(
            load_tensor(lr_path)
        )

        sr = standardize_batch(
            load_tensor(sr_path)
        )

        no_risk_sr = standardize_batch(
            load_tensor(no_risk_path)
        )

        risk_sr = standardize_batch(
            load_tensor(risk_path)
        )


        # ----------------------------------------------------
        # Validate shapes
        # ----------------------------------------------------

        expected_hr = (
            1,
            4,
            484,
            484,
        )

        expected_lr = (
            1,
            4,
            121,
            121,
        )

        if hr.shape != expected_hr:
            raise ValueError(
                f"{roi}: HR shape "
                f"{hr.shape}, expected "
                f"{expected_hr}"
            )

        if lr.shape != expected_lr:
            raise ValueError(
                f"{roi}: LR shape "
                f"{lr.shape}, expected "
                f"{expected_lr}"
            )

        if sr.shape != expected_hr:
            raise ValueError(
                f"{roi}: SR shape "
                f"{sr.shape}, expected "
                f"{expected_hr}"
            )

        if no_risk_sr.shape != expected_hr:
            raise ValueError(
                f"{roi}: No-risk Safe-SR shape "
                f"{no_risk_sr.shape}"
            )

        if risk_sr.shape != expected_hr:
            raise ValueError(
                f"{roi}: Risk-aware Safe-SR shape "
                f"{risk_sr.shape}"
            )


        # ----------------------------------------------------
        # Check finite
        # ----------------------------------------------------

        tensors = {
            "HR": hr,
            "LR": lr,
            "SR": sr,
            "No-risk Safe-SR": no_risk_sr,
            "Risk-aware Safe-SR": risk_sr,
        }

        for name, tensor in tensors.items():

            if not torch.isfinite(
                tensor
            ).all():

                raise ValueError(
                    f"{roi}: {name} contains "
                    "NaN or Inf."
                )


        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        baseline = evaluate_image(
            sr,
            hr,
            lr,
            degradation_model,
        )

        no_risk = evaluate_image(
            no_risk_sr,
            hr,
            lr,
            degradation_model,
        )

        risk = evaluate_image(
            risk_sr,
            hr,
            lr,
            degradation_model,
        )


        all_results[roi] = {
            "baseline_LDSR_S2": baseline,
            "no_risk_Safe_SR": no_risk,
            "risk_aware_Safe_SR": risk,
        }


        # ----------------------------------------------------
        # Collect values
        # ----------------------------------------------------

        for metric in baseline_values:
            baseline_values[metric].append(
                baseline[metric]
            )

            no_risk_values[metric].append(
                no_risk[metric]
            )

            risk_values[metric].append(
                risk[metric]
            )


        # ----------------------------------------------------
        # Print ROI results
        # ----------------------------------------------------

        print()
        print("LDSR-S2")
        print(
            f"  PSNR: "
            f"{baseline['PSNR_dB']:.4f} dB"
        )
        print(
            f"  SSIM: "
            f"{baseline['SSIM']:.6f}"
        )
        print(
            f"  OpenSR consistency: "
            f"{baseline['OpenSR_consistency_L1']:.6f}"
        )

        print()
        print("No-risk Safe-SR")
        print(
            f"  PSNR: "
            f"{no_risk['PSNR_dB']:.4f} dB"
        )
        print(
            f"  SSIM: "
            f"{no_risk['SSIM']:.6f}"
        )
        print(
            f"  OpenSR consistency: "
            f"{no_risk['OpenSR_consistency_L1']:.6f}"
        )

        print()
        print("Risk-aware Safe-SR")
        print(
            f"  PSNR: "
            f"{risk['PSNR_dB']:.4f} dB"
        )
        print(
            f"  SSIM: "
            f"{risk['SSIM']:.6f}"
        )
        print(
            f"  OpenSR consistency: "
            f"{risk['OpenSR_consistency_L1']:.6f}"
        )


    # ========================================================
    # Average metrics
    # ========================================================

    def average(values):
        return sum(values) / len(values)


    baseline_avg = {
        metric: average(values)
        for metric, values
        in baseline_values.items()
    }

    no_risk_avg = {
        metric: average(values)
        for metric, values
        in no_risk_values.items()
    }

    risk_avg = {
        metric: average(values)
        for metric, values
        in risk_values.items()
    }


    # ========================================================
    # Relative changes
    # ========================================================

    def relative_change(
        new,
        old,
    ):

        if old == 0:
            return None

        return (
            (new - old)
            / abs(old)
        ) * 100.0


    no_risk_changes = {
        "PSNR_percent":
            relative_change(
                no_risk_avg["PSNR_dB"],
                baseline_avg["PSNR_dB"],
            ),
        "SSIM_percent":
            relative_change(
                no_risk_avg["SSIM"],
                baseline_avg["SSIM"],
            ),
        "OpenSR_consistency_percent":
            relative_change(
                no_risk_avg[
                    "OpenSR_consistency_L1"
                ],
                baseline_avg[
                    "OpenSR_consistency_L1"
                ],
            ),
    }

    risk_changes = {
        "PSNR_percent":
            relative_change(
                risk_avg["PSNR_dB"],
                baseline_avg["PSNR_dB"],
            ),
        "SSIM_percent":
            relative_change(
                risk_avg["SSIM"],
                baseline_avg["SSIM"],
            ),
        "OpenSR_consistency_percent":
            relative_change(
                risk_avg[
                    "OpenSR_consistency_L1"
                ],
                baseline_avg[
                    "OpenSR_consistency_L1"
                ],
            ),
    }


    # ========================================================
    # Print final summary
    # ========================================================

    print()
    print("========================================")
    print("FINAL VALIDATION RESULTS")
    print("========================================")

    print()
    print("AVERAGE OVER VALIDATION ROIs")
    print("----------------------------------------")

    print()
    print("LDSR-S2")
    print(
        f"  PSNR: "
        f"{baseline_avg['PSNR_dB']:.4f} dB"
    )
    print(
        f"  SSIM: "
        f"{baseline_avg['SSIM']:.6f}"
    )
    print(
        f"  OpenSR consistency: "
        f"{baseline_avg['OpenSR_consistency_L1']:.6f}"
    )

    print()
    print("No-risk Safe-SR")
    print(
        f"  PSNR: "
        f"{no_risk_avg['PSNR_dB']:.4f} dB"
    )
    print(
        f"  SSIM: "
        f"{no_risk_avg['SSIM']:.6f}"
    )
    print(
        f"  OpenSR consistency: "
        f"{no_risk_avg['OpenSR_consistency_L1']:.6f}"
    )

    print()
    print("Risk-aware Safe-SR")
    print(
        f"  PSNR: "
        f"{risk_avg['PSNR_dB']:.4f} dB"
    )
    print(
        f"  SSIM: "
        f"{risk_avg['SSIM']:.6f}"
    )
    print(
        f"  OpenSR consistency: "
        f"{risk_avg['OpenSR_consistency_L1']:.6f}"
    )


    print()
    print("CHANGE FROM LDSR-S2")
    print("----------------------------------------")

    print()
    print("No-risk Safe-SR:")
    print(
        f"  PSNR: "
        f"{no_risk_changes['PSNR_percent']:+.2f}%"
    )
    print(
        f"  SSIM: "
        f"{no_risk_changes['SSIM_percent']:+.2f}%"
    )
    print(
        f"  Consistency: "
        f"{no_risk_changes['OpenSR_consistency_percent']:+.2f}%"
    )

    print()
    print("Risk-aware Safe-SR:")
    print(
        f"  PSNR: "
        f"{risk_changes['PSNR_percent']:+.2f}%"
    )
    print(
        f"  SSIM: "
        f"{risk_changes['SSIM_percent']:+.2f}%"
    )
    print(
        f"  Consistency: "
        f"{risk_changes['OpenSR_consistency_percent']:+.2f}%"
    )


    # ========================================================
    # Save JSON
    # ========================================================

    results = {
        "experiment": (
            "10-ROI Safe-SR validation "
            "ablation"
        ),
        "validation_rois": VAL_ROIS,

        "per_roi": all_results,

        "average": {
            "baseline_LDSR_S2":
                baseline_avg,
            "no_risk_Safe_SR":
                no_risk_avg,
            "risk_aware_Safe_SR":
                risk_avg,
        },

        "change_from_LDSR_S2": {
            "no_risk_Safe_SR":
                no_risk_changes,
            "risk_aware_Safe_SR":
                risk_changes,
        },
    }


    output_path = os.path.join(
        OUTPUT_DIR,
        "multi_roi_validation_metrics.json",
    )

    with open(
        output_path,
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )


    print()
    print(
        f"Saved results: {output_path}"
    )

    print()
    print("========================================")
    print("EVALUATION COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()