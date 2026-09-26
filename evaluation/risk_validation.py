import os
import json

import torch
import torch.nn.functional as F


VAL_ROIS = [
    "ROI_2319",
    "ROI_0630",
]

OUTPUT_DIR = "outputs/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_tensor(path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    ).float()


def pearson_correlation(x, y):
    x = x.flatten()
    y = y.flatten()

    x = x - x.mean()
    y = y - y.mean()

    denominator = torch.sqrt(
        torch.sum(x ** 2) *
        torch.sum(y ** 2)
    )

    if denominator == 0:
        return 0.0

    return float(
        torch.sum(x * y) / denominator
    )


def prepare_risk_map(risk):
    """
    Convert risk map to [1, 1, 121, 121].

    The saved ROI risk maps are 484x484 HR-resolution maps,
    while the reconstruction-error analysis is performed
    at LR resolution (121x121).
    """

    if risk.ndim == 2:
        # [484, 484]
        risk = risk.unsqueeze(0).unsqueeze(0)

    elif risk.ndim == 3:
        # Could be [C,H,W] or [1,H,W]
        risk = risk.unsqueeze(0)

    elif risk.ndim == 4:
        pass

    else:
        raise ValueError(
            f"Unexpected risk-map shape: {risk.shape}"
        )

    # If multiple channels exist, average them.
    if risk.shape[1] > 1:
        risk = risk.mean(
            dim=1,
            keepdim=True,
        )

    # Convert HR risk map (484x484)
    # to LR resolution (121x121).
    if risk.shape[-2:] == (484, 484):
        risk = F.avg_pool2d(
            risk,
            kernel_size=4,
            stride=4,
        )

    elif risk.shape[-2:] != (121, 121):
        risk = F.interpolate(
            risk,
            size=(121, 121),
            mode="bilinear",
            align_corners=False,
        )

    return risk


def analyze_roi(roi):

    hr = load_tensor(
        f"data/processed/{roi}/hr.pt"
    )

    sr = load_tensor(
        f"outputs/sr/{roi}_sr.pt"
    )

    risk = load_tensor(
        f"outputs/risk/{roi}_risk_map_normalized.pt"
    )

    # --------------------------------------------------
    # Standardize HR and SR
    # --------------------------------------------------

    if hr.ndim == 3:
        hr = hr.unsqueeze(0)

    if sr.ndim == 3:
        sr = sr.unsqueeze(0)

    # --------------------------------------------------
    # Prepare risk map
    # --------------------------------------------------

    risk = prepare_risk_map(risk)

    print(
        f"  HR shape: {hr.shape}"
    )

    print(
        f"  SR shape: {sr.shape}"
    )

    print(
        f"  Risk shape after alignment: "
        f"{risk.shape}"
    )

    # --------------------------------------------------
    # Compute LDSR-S2 reconstruction error
    # --------------------------------------------------

    pixel_error_hr = torch.mean(
        torch.abs(sr - hr),
        dim=1,
        keepdim=True,
    )

    # HR error: 484x484
    # Downsample to 121x121 so it matches
    # the aligned risk map.

    pixel_error = F.avg_pool2d(
        pixel_error_hr,
        kernel_size=4,
        stride=4,
    )

    if pixel_error.shape != risk.shape:
        raise ValueError(
            f"{roi}: shape mismatch after alignment: "
            f"error={pixel_error.shape}, "
            f"risk={risk.shape}"
        )

    error = pixel_error.flatten()
    risk_values = risk.flatten()

    # --------------------------------------------------
    # Pearson correlation
    # --------------------------------------------------

    correlation = pearson_correlation(
        risk_values,
        error,
    )

    # --------------------------------------------------
    # High-risk vs low-risk regions
    # --------------------------------------------------

    k = max(
        1,
        int(0.10 * risk_values.numel())
    )

    sorted_indices = torch.argsort(
        risk_values,
        descending=True,
    )

    high_risk_indices = sorted_indices[:k]
    low_risk_indices = sorted_indices[k:]

    high_risk_error = float(
        error[high_risk_indices].mean()
    )

    low_risk_error = float(
        error[low_risk_indices].mean()
    )

    error_ratio = (
        high_risk_error /
        (low_risk_error + 1e-8)
    )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    result = {
        "pearson_risk_error_correlation": correlation,
        "high_risk_top_10_percent_error": high_risk_error,
        "remaining_90_percent_error": low_risk_error,
        "high_vs_low_error_ratio": error_ratio,
        "mean_risk": float(risk_values.mean()),
        "mean_error": float(error.mean()),
        "num_pixels": int(risk_values.numel()),
    }

    return result


def main():

    print("========================================")
    print("RISK MAP VALIDATION")
    print("========================================")

    all_results = {}

    for roi in VAL_ROIS:

        print()
        print(f"Analyzing {roi}...")

        result = analyze_roi(roi)

        all_results[roi] = result

        print()
        print(
            f"Pearson risk-error correlation: "
            f"{result['pearson_risk_error_correlation']:.6f}"
        )

        print(
            f"Top 10% risk error: "
            f"{result['high_risk_top_10_percent_error']:.6f}"
        )

        print(
            f"Remaining 90% error: "
            f"{result['remaining_90_percent_error']:.6f}"
        )

        print(
            f"High/low error ratio: "
            f"{result['high_vs_low_error_ratio']:.4f}"
        )

    # --------------------------------------------------
    # Average validation result
    # --------------------------------------------------

    avg_correlation = sum(
        r["pearson_risk_error_correlation"]
        for r in all_results.values()
    ) / len(all_results)

    avg_ratio = sum(
        r["high_vs_low_error_ratio"]
        for r in all_results.values()
    ) / len(all_results)

    print()
    print("========================================")
    print("FINAL RISK VALIDATION")
    print("========================================")

    print()
    print(
        f"Average Pearson correlation: "
        f"{avg_correlation:.6f}"
    )

    print(
        f"Average high/low error ratio: "
        f"{avg_ratio:.4f}"
    )

    results = {
        "validation_rois": VAL_ROIS,
        "per_roi": all_results,
        "average_pearson_correlation":
            avg_correlation,
        "average_high_low_error_ratio":
            avg_ratio,
    }

    output_path = os.path.join(
        OUTPUT_DIR,
        "risk_validation.json",
    )

    with open(output_path, "w") as f:
        json.dump(
            results,
            f,
            indent=4,
        )

    print()
    print(f"Saved: {output_path}")

    print()
    print("========================================")
    print("VALIDATION COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()