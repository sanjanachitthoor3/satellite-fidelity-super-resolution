import os
import json
import sys
import math

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity


# ============================================================
# Project root
# ============================================================

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

# These are the same validation ROIs used by the previous evaluation.
VAL_ROIS = [
    "ROI_2319",
    "ROI_0630",
]

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "evaluation",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42

EXPECTED_HR_SHAPE = (1, 4, 484, 484)
EXPECTED_LR_SHAPE = (1, 4, 121, 121)

# Risk-region analysis: compare the highest-risk 10% of pixels
# against the remaining 90%.
HIGH_RISK_FRACTION = 0.10


# ============================================================
# Helpers
# ============================================================

def load_tensor(path):
    """Load a tensor safely onto CPU as float32."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required file not found:\n  {path}"
        )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    ).float()


def standardize_batch(x):
    """Convert [C,H,W] tensors to [1,C,H,W]."""
    if x.ndim == 3:
        x = x.unsqueeze(0)

    return x


def standardize_risk_map(x):
    """Convert a saved risk map to [H,W]."""
    x = x.float().squeeze()

    if x.ndim != 2:
        raise ValueError(
            "Risk map must reduce to [H,W], but got "
            f"shape {tuple(x.shape)}."
        )

    return x


def clamp_image(x):
    return torch.clamp(x, 0.0, 1.0)


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)

    if mse <= 0:
        return float("inf")

    return float(
        10.0 * torch.log10(1.0 / mse)
    )


def ssim(pred, target):
    pred = pred.squeeze(0)
    target = target.squeeze(0)

    pred_np = pred.permute(1, 2, 0).numpy()
    target_np = target.permute(1, 2, 0).numpy()

    values = []

    for c in range(pred_np.shape[-1]):
        values.append(
            structural_similarity(
                target_np[:, :, c],
                pred_np[:, :, c],
                data_range=1.0,
            )
        )

    return float(sum(values) / len(values))


def opensr_lr_consistency(sr, lr, degradation_model):
    """Degrade SR back to LR and measure mean absolute error."""
    sr_input = sr.squeeze(0)

    with torch.no_grad():
        degraded_lr, _ = degradation_model.forward(
            sr_input
        )

    if degraded_lr.ndim == 3:
        degraded_lr = degraded_lr.unsqueeze(0)

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
    """Compute the three global image-quality metrics."""
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


def error_map(prediction, hr):
    """
    Per-pixel reconstruction error averaged across the 4 bands.
    Returns [H,W].
    """
    prediction = clamp_image(prediction)
    hr = clamp_image(hr)

    return torch.mean(
        torch.abs(prediction - hr),
        dim=1,
    ).squeeze(0)


def pearson_correlation(x, y):
    """Pearson correlation with a safe zero-variance check."""
    x = x.flatten().double()
    y = y.flatten().double()

    x_centered = x - x.mean()
    y_centered = y - y.mean()

    denominator = torch.sqrt(
        torch.sum(x_centered ** 2)
        * torch.sum(y_centered ** 2)
    )

    if denominator <= 0:
        return None

    return float(
        torch.sum(x_centered * y_centered)
        / denominator
    )


def risk_region_metrics(risk_map, prediction, hr):
    """
    Compare reconstruction error in the highest-risk 10% of pixels
    against the remaining 90%.
    """
    risk_map = standardize_risk_map(risk_map)
    error = error_map(prediction, hr)

    if risk_map.shape != error.shape:
        raise ValueError(
            "Risk/error spatial shape mismatch: "
            f"risk={tuple(risk_map.shape)}, "
            f"error={tuple(error.shape)}"
        )

    risk_flat = risk_map.flatten()
    error_flat = error.flatten()

    n = risk_flat.numel()
    high_count = max(
        1,
        int(math.ceil(n * HIGH_RISK_FRACTION)),
    )

    # Select exactly the highest-risk pixels rather than using a
    # quantile threshold, which can include more than 10% when values tie.
    _, high_indices = torch.topk(
        risk_flat,
        k=high_count,
        largest=True,
        sorted=False,
    )

    high_mask = torch.zeros(
        n,
        dtype=torch.bool,
    )
    high_mask[high_indices] = True
    low_mask = ~high_mask

    high_error = float(error_flat[high_mask].mean())
    low_error = float(error_flat[low_mask].mean())

    return {
        "Pearson_risk_error_correlation": pearson_correlation(
            risk_map,
            error,
        ),
        "high_risk_fraction": HIGH_RISK_FRACTION,
        "high_risk_mean_error": high_error,
        "low_risk_mean_error": low_error,
        "high_to_low_error_ratio": (
            high_error / low_error
            if low_error > 0
            else None
        ),
    }


def relative_change(new, old):
    if old == 0:
        return None

    return ((new - old) / abs(old)) * 100.0


def metric_changes(new, baseline):
    """Return raw and relative changes from LDSR-S2."""
    return {
        "PSNR_dB_delta": (
            new["PSNR_dB"] - baseline["PSNR_dB"]
        ),
        "PSNR_percent": relative_change(
            new["PSNR_dB"],
            baseline["PSNR_dB"],
        ),
        "SSIM_delta": (
            new["SSIM"] - baseline["SSIM"]
        ),
        "SSIM_percent": relative_change(
            new["SSIM"],
            baseline["SSIM"],
        ),
        "OpenSR_consistency_L1_delta": (
            new["OpenSR_consistency_L1"]
            - baseline["OpenSR_consistency_L1"]
        ),
        "OpenSR_consistency_percent": relative_change(
            new["OpenSR_consistency_L1"],
            baseline["OpenSR_consistency_L1"],
        ),
    }


def compare_risk_and_no_risk(risk, no_risk):
    """
    Direct controlled comparison.
    Positive error reduction means risk-aware has lower error.
    """
    return {
        "PSNR_dB_delta_risk_minus_no_risk": (
            risk["PSNR_dB"] - no_risk["PSNR_dB"]
        ),
        "SSIM_delta_risk_minus_no_risk": (
            risk["SSIM"] - no_risk["SSIM"]
        ),
        "OpenSR_consistency_L1_delta_risk_minus_no_risk": (
            risk["OpenSR_consistency_L1"]
            - no_risk["OpenSR_consistency_L1"]
        ),
    }


def print_metric_block(name, metrics):
    print()
    print(name)
    print(
        f"  PSNR: {metrics['PSNR_dB']:.4f} dB"
    )
    print(
        f"  SSIM: {metrics['SSIM']:.6f}"
    )
    print(
        "  OpenSR consistency: "
        f"{metrics['OpenSR_consistency_L1']:.6f}"
    )


def print_region_block(name, metrics):
    print()
    print(name)
    corr = metrics["Pearson_risk_error_correlation"]
    corr_text = (
        f"{corr:.6f}"
        if corr is not None
        else "undefined"
    )
    print(
        "  Risk-error correlation: "
        f"{corr_text}"
    )
    print(
        "  High-risk 10% mean error: "
        f"{metrics['high_risk_mean_error']:.6f}"
    )
    print(
        "  Low-risk 90% mean error: "
        f"{metrics['low_risk_mean_error']:.6f}"
    )
    print(
        "  High/low error ratio: "
        f"{metrics['high_to_low_error_ratio']:.6f}"
    )


# ============================================================
# Main
# ============================================================

def main():
    print("========================================")
    print("PATCH-BASED CONTROLLED SAFE-SR EVALUATION")
    print("========================================")
    print()
    print("Validation ROIs:")
    for roi in VAL_ROIS:
        print(f"  {roi}")

    print()
    print("Models being compared:")
    print("  1. LDSR-S2 baseline")
    print("  2. No-risk Safe-SR")
    print("  3. Risk-aware Safe-SR")
    print()
    print(
        "Risk-region analysis: highest-risk 10% "
        "vs remaining 90%"
    )

    # ========================================================
    # Create OpenSR evaluation model
    # ========================================================

    print()
    print("Creating OpenSR evaluation model...")

    degradation_model = create_naip_degradation_model(
        device="cpu",
        seed=SEED,
        add_noise=True,
        reflectance_methods=[
            "gamma_multivariate_normal"
        ],
    )

    print("OpenSR evaluation model ready.")

    # ========================================================
    # Results storage
    # ========================================================

    all_results = {}

    metric_names = [
        "PSNR_dB",
        "SSIM",
        "OpenSR_consistency_L1",
    ]

    baseline_values = {
        metric: [] for metric in metric_names
    }
    no_risk_values = {
        metric: [] for metric in metric_names
    }
    risk_values = {
        metric: [] for metric in metric_names
    }

    # ========================================================
    # Evaluate each validation ROI
    # ========================================================

    for roi in VAL_ROIS:
        print()
        print("========================================")
        print(f"EVALUATING {roi}")
        print("========================================")

        hr_path = os.path.join(
            PROJECT_ROOT,
            "data",
            "processed",
            roi,
            "hr.pt",
        )
        lr_path = os.path.join(
            PROJECT_ROOT,
            "data",
            "processed",
            roi,
            "lr.pt",
        )
        sr_path = os.path.join(
            PROJECT_ROOT,
            "outputs",
            "sr",
            f"{roi}_sr.pt",
        )
        no_risk_path = os.path.join(
            PROJECT_ROOT,
            "outputs",
            "refinement",
            f"{roi}_no-risk_safe_sr.pt",
        )
        risk_path = os.path.join(
            PROJECT_ROOT,
            "outputs",
            "refinement",
            f"{roi}_risk_safe_sr.pt",
        )
        risk_map_path = os.path.join(
            PROJECT_ROOT,
            "outputs",
            "risk",
            f"{roi}_risk_map_normalized.pt",
        )

        # ----------------------------------------------------
        # Load tensors
        # ----------------------------------------------------

        hr = standardize_batch(load_tensor(hr_path))
        lr = standardize_batch(load_tensor(lr_path))
        sr = standardize_batch(load_tensor(sr_path))
        no_risk_sr = standardize_batch(
            load_tensor(no_risk_path)
        )
        risk_sr = standardize_batch(
            load_tensor(risk_path)
        )
        risk_map = standardize_risk_map(
            load_tensor(risk_map_path)
        )

        # ----------------------------------------------------
        # Validate shapes
        # ----------------------------------------------------

        shape_checks = {
            "HR": (hr, EXPECTED_HR_SHAPE),
            "LR": (lr, EXPECTED_LR_SHAPE),
            "LDSR-S2 SR": (sr, EXPECTED_HR_SHAPE),
            "No-risk Safe-SR": (
                no_risk_sr,
                EXPECTED_HR_SHAPE,
            ),
            "Risk-aware Safe-SR": (
                risk_sr,
                EXPECTED_HR_SHAPE,
            ),
        }

        for name, (tensor, expected) in shape_checks.items():
            if tensor.shape != expected:
                raise ValueError(
                    f"{roi}: {name} shape "
                    f"{tuple(tensor.shape)}, expected "
                    f"{expected}"
                )

        if risk_map.shape != EXPECTED_HR_SHAPE[2:]:
            raise ValueError(
                f"{roi}: risk map shape "
                f"{tuple(risk_map.shape)}, expected "
                f"{EXPECTED_HR_SHAPE[2:]}"
            )

        # ----------------------------------------------------
        # Check finite values
        # ----------------------------------------------------

        tensors = {
            "HR": hr,
            "LR": lr,
            "LDSR-S2 SR": sr,
            "No-risk Safe-SR": no_risk_sr,
            "Risk-aware Safe-SR": risk_sr,
            "Risk map": risk_map,
        }

        for name, tensor in tensors.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"{roi}: {name} contains NaN or Inf."
                )

        # ----------------------------------------------------
        # Global metrics
        # ----------------------------------------------------

        with torch.no_grad():
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

            # Spatial error in each risk region.
            baseline_region = risk_region_metrics(
                risk_map,
                sr,
                hr,
            )
            no_risk_region = risk_region_metrics(
                risk_map,
                no_risk_sr,
                hr,
            )
            risk_region = risk_region_metrics(
                risk_map,
                risk_sr,
                hr,
            )

        # Direct controlled comparison: risk-aware minus no-risk.
        direct_comparison = compare_risk_and_no_risk(
            risk,
            no_risk,
        )

        # Region-specific error differences.
        direct_region_comparison = {
            "high_risk_error_risk_minus_no_risk": (
                risk_region["high_risk_mean_error"]
                - no_risk_region["high_risk_mean_error"]
            ),
            "low_risk_error_risk_minus_no_risk": (
                risk_region["low_risk_mean_error"]
                - no_risk_region["low_risk_mean_error"]
            ),
            "high_risk_error_reduction_risk_vs_no_risk": (
                no_risk_region["high_risk_mean_error"]
                - risk_region["high_risk_mean_error"]
            ),
            "low_risk_error_reduction_risk_vs_no_risk": (
                no_risk_region["low_risk_mean_error"]
                - risk_region["low_risk_mean_error"]
            ),
        }

        all_results[roi] = {
            "global_metrics": {
                "baseline_LDSR_S2": baseline,
                "no_risk_Safe_SR": no_risk,
                "risk_aware_Safe_SR": risk,
            },
            "change_from_LDSR_S2": {
                "no_risk_Safe_SR": metric_changes(
                    no_risk,
                    baseline,
                ),
                "risk_aware_Safe_SR": metric_changes(
                    risk,
                    baseline,
                ),
            },
            "direct_risk_vs_no_risk": direct_comparison,
            "risk_region_analysis": {
                "LDSR_S2": baseline_region,
                "no_risk_Safe_SR": no_risk_region,
                "risk_aware_Safe_SR": risk_region,
                "risk_aware_minus_no_risk": direct_region_comparison,
            },
        }

        for metric in metric_names:
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

        print_metric_block("LDSR-S2", baseline)
        print_metric_block("No-risk Safe-SR", no_risk)
        print_metric_block("Risk-aware Safe-SR", risk)

        print()
        print("Risk-aware minus No-risk")
        print(
            f"  PSNR delta: "
            f"{direct_comparison['PSNR_dB_delta_risk_minus_no_risk']:+.4f} dB"
        )
        print(
            f"  SSIM delta: "
            f"{direct_comparison['SSIM_delta_risk_minus_no_risk']:+.6f}"
        )
        print(
            "  Consistency delta: "
            f"{direct_comparison['OpenSR_consistency_L1_delta_risk_minus_no_risk']:+.6f}"
        )

        print_region_block(
            "Risk regions — LDSR-S2",
            baseline_region,
        )
        print_region_block(
            "Risk regions — No-risk Safe-SR",
            no_risk_region,
        )
        print_region_block(
            "Risk regions — Risk-aware Safe-SR",
            risk_region,
        )

        print()
        print("Risk-aware vs No-risk region differences")
        print(
            "  High-risk error reduction: "
            f"{direct_region_comparison['high_risk_error_reduction_risk_vs_no_risk']:+.6f}"
        )
        print(
            "  Low-risk error reduction: "
            f"{direct_region_comparison['low_risk_error_reduction_risk_vs_no_risk']:+.6f}"
        )

    # ========================================================
    # Average global metrics
    # ========================================================

    def average(values):
        if not values:
            raise ValueError("No validation values were collected.")
        return sum(values) / len(values)

    baseline_avg = {
        metric: average(values)
        for metric, values in baseline_values.items()
    }
    no_risk_avg = {
        metric: average(values)
        for metric, values in no_risk_values.items()
    }
    risk_avg = {
        metric: average(values)
        for metric, values in risk_values.items()
    }

    no_risk_changes = metric_changes(
        no_risk_avg,
        baseline_avg,
    )
    risk_changes = metric_changes(
        risk_avg,
        baseline_avg,
    )
    direct_average_comparison = compare_risk_and_no_risk(
        risk_avg,
        no_risk_avg,
    )

    # ========================================================
    # Average region metrics
    # ========================================================

    region_keys = [
        "high_risk_mean_error",
        "low_risk_mean_error",
        "high_to_low_error_ratio",
    ]

    region_average = {
        "LDSR_S2": {},
        "no_risk_Safe_SR": {},
        "risk_aware_Safe_SR": {},
    }

    for model_key in region_average:
        for metric in region_keys:
            values = []
            for roi in VAL_ROIS:
                values.append(
                    all_results[roi][
                        "risk_region_analysis"
                    ][model_key][metric]
                )
            region_average[model_key][metric] = average(values)

    correlations = []
    for roi in VAL_ROIS:
        corr = all_results[roi][
            "risk_region_analysis"
        ]["LDSR_S2"][
            "Pearson_risk_error_correlation"
        ]
        if corr is not None:
            correlations.append(corr)

    region_average["LDSR_S2"][
        "Pearson_risk_error_correlation"
    ] = (
        average(correlations)
        if correlations
        else None
    )

    # Average high/low errors for the two refined models.
    for model_key in [
        "no_risk_Safe_SR",
        "risk_aware_Safe_SR",
    ]:
        model_correlations = []
        for roi in VAL_ROIS:
            corr = all_results[roi][
                "risk_region_analysis"
            ][model_key][
                "Pearson_risk_error_correlation"
            ]
            if corr is not None:
                model_correlations.append(corr)

        region_average[model_key][
            "Pearson_risk_error_correlation"
        ] = (
            average(model_correlations)
            if model_correlations
            else None
        )

    region_average_direct = {
        "high_risk_error_reduction_risk_vs_no_risk": (
            region_average["no_risk_Safe_SR"][
                "high_risk_mean_error"
            ]
            - region_average["risk_aware_Safe_SR"][
                "high_risk_mean_error"
            ]
        ),
        "low_risk_error_reduction_risk_vs_no_risk": (
            region_average["no_risk_Safe_SR"][
                "low_risk_mean_error"
            ]
            - region_average["risk_aware_Safe_SR"][
                "low_risk_mean_error"
            ]
        ),
    }

    # ========================================================
    # Print final summary
    # ========================================================

    print()
    print("========================================")
    print("FINAL PATCH-BASED VALIDATION RESULTS")
    print("========================================")

    print()
    print("AVERAGE OVER VALIDATION ROIs")
    print("----------------------------------------")

    print_metric_block("LDSR-S2", baseline_avg)
    print_metric_block("No-risk Safe-SR", no_risk_avg)
    print_metric_block("Risk-aware Safe-SR", risk_avg)

    print()
    print("CHANGE FROM LDSR-S2")
    print("----------------------------------------")
    print(
        f"No-risk PSNR: "
        f"{no_risk_changes['PSNR_percent']:+.2f}%"
    )
    print(
        f"No-risk SSIM: "
        f"{no_risk_changes['SSIM_percent']:+.2f}%"
    )
    print(
        "No-risk consistency: "
        f"{no_risk_changes['OpenSR_consistency_percent']:+.2f}%"
    )
    print()
    print(
        f"Risk-aware PSNR: "
        f"{risk_changes['PSNR_percent']:+.2f}%"
    )
    print(
        f"Risk-aware SSIM: "
        f"{risk_changes['SSIM_percent']:+.2f}%"
    )
    print(
        "Risk-aware consistency: "
        f"{risk_changes['OpenSR_consistency_percent']:+.2f}%"
    )

    print()
    print("DIRECT CONTROLLED COMPARISON")
    print("----------------------------------------")
    print(
        "Risk-aware minus No-risk PSNR: "
        f"{direct_average_comparison['PSNR_dB_delta_risk_minus_no_risk']:+.4f} dB"
    )
    print(
        "Risk-aware minus No-risk SSIM: "
        f"{direct_average_comparison['SSIM_delta_risk_minus_no_risk']:+.6f}"
    )
    print(
        "Risk-aware minus No-risk consistency: "
        f"{direct_average_comparison['OpenSR_consistency_L1_delta_risk_minus_no_risk']:+.6f}"
    )

    print()
    print("RISK-REGION ANALYSIS")
    print("----------------------------------------")
    for model_key, display_name in [
        ("LDSR_S2", "LDSR-S2"),
        ("no_risk_Safe_SR", "No-risk Safe-SR"),
        ("risk_aware_Safe_SR", "Risk-aware Safe-SR"),
    ]:
        metrics = region_average[model_key]
        print()
        print(display_name)
        corr = metrics[
            "Pearson_risk_error_correlation"
        ]
        print(
            "  Average risk-error correlation: "
            + (
                f"{corr:.6f}"
                if corr is not None
                else "undefined"
            )
        )
        print(
            "  High-risk 10% mean error: "
            f"{metrics['high_risk_mean_error']:.6f}"
        )
        print(
            "  Low-risk 90% mean error: "
            f"{metrics['low_risk_mean_error']:.6f}"
        )
        print(
            "  High/low error ratio: "
            f"{metrics['high_to_low_error_ratio']:.6f}"
        )

    print()
    print("RISK-AWARE VS NO-RISK IN RISK REGIONS")
    print("----------------------------------------")
    print(
        "High-risk error reduction: "
        f"{region_average_direct['high_risk_error_reduction_risk_vs_no_risk']:+.6f}"
    )
    print(
        "Low-risk error reduction: "
        f"{region_average_direct['low_risk_error_reduction_risk_vs_no_risk']:+.6f}"
    )

    # ========================================================
    # Save JSON
    # ========================================================

    results = {
        "experiment": (
            "Patch-based controlled Safe-SR validation "
            "ablation"
        ),
        "validation_rois": VAL_ROIS,
        "high_risk_fraction": HIGH_RISK_FRACTION,
        "risk_map_source": (
            "outputs/risk/{roi}_risk_map_normalized.pt"
        ),
        "per_roi": all_results,
        "average": {
            "baseline_LDSR_S2": baseline_avg,
            "no_risk_Safe_SR": no_risk_avg,
            "risk_aware_Safe_SR": risk_avg,
        },
        "change_from_LDSR_S2": {
            "no_risk_Safe_SR": no_risk_changes,
            "risk_aware_Safe_SR": risk_changes,
        },
        "direct_risk_vs_no_risk": direct_average_comparison,
        "average_risk_region_analysis": region_average,
        "average_risk_region_direct_comparison": (
            region_average_direct
        ),
    }

    output_path = os.path.join(
        OUTPUT_DIR,
        "patch_validation_metrics.json",
    )

    with open(output_path, "w") as f:
        json.dump(
            results,
            f,
            indent=4,
        )

    print()
    print(f"Saved results: {output_path}")
    print()
    print("========================================")
    print("EVALUATION COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()
