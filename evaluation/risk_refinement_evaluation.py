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


def prepare_risk(risk):

    if risk.ndim == 2:
        risk = risk.unsqueeze(0).unsqueeze(0)

    elif risk.ndim == 3:
        risk = risk.unsqueeze(0)

    if risk.shape[1] > 1:
        risk = risk.mean(
            dim=1,
            keepdim=True,
        )

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

    sr = load_tensor(
        f"outputs/sr/{roi}_sr.pt"
    )

    no_risk = load_tensor(
        f"outputs/refinement/"
        f"{roi}_no-risk_safe_sr.pt"
    )

    risk_aware = load_tensor(
        f"outputs/refinement/"
        f"{roi}_risk_safe_sr.pt"
    )

    risk = load_tensor(
        f"outputs/risk/"
        f"{roi}_risk_map_normalized.pt"
    )

    if sr.ndim == 3:
        sr = sr.unsqueeze(0)

    if no_risk.ndim == 3:
        no_risk = no_risk.unsqueeze(0)

    if risk_aware.ndim == 3:
        risk_aware = risk_aware.unsqueeze(0)

    risk = prepare_risk(risk)

    # --------------------------------------------------
    # Magnitude of refinement
    # --------------------------------------------------

    no_risk_change_hr = torch.mean(
        torch.abs(no_risk - sr),
        dim=1,
        keepdim=True,
    )

    risk_change_hr = torch.mean(
        torch.abs(risk_aware - sr),
        dim=1,
        keepdim=True,
    )

    # Match risk resolution.
    no_risk_change = F.avg_pool2d(
        no_risk_change_hr,
        kernel_size=4,
        stride=4,
    )

    risk_change = F.avg_pool2d(
        risk_change_hr,
        kernel_size=4,
        stride=4,
    )

    risk_values = risk.flatten()

    no_risk_values = no_risk_change.flatten()
    risk_values_change = risk_change.flatten()

    # --------------------------------------------------
    # Does refinement magnitude depend on risk?
    # --------------------------------------------------

    no_risk_corr = pearson_correlation(
        risk_values,
        no_risk_values,
    )

    risk_aware_corr = pearson_correlation(
        risk_values,
        risk_values_change,
    )

    # --------------------------------------------------
    # High-risk regions
    # --------------------------------------------------

    k = max(
        1,
        int(0.10 * risk_values.numel())
    )

    indices = torch.argsort(
        risk_values,
        descending=True,
    )

    high = indices[:k]
    low = indices[k:]

    no_risk_high = float(
        no_risk_values[high].mean()
    )

    no_risk_low = float(
        no_risk_values[low].mean()
    )

    risk_high = float(
        risk_values_change[high].mean()
    )

    risk_low = float(
        risk_values_change[low].mean()
    )

    return {
        "risk_vs_no_risk_refinement_correlation":
            no_risk_corr,

        "risk_vs_risk_aware_refinement_correlation":
            risk_aware_corr,

        "no_risk_high_risk_region_change":
            no_risk_high,

        "no_risk_low_risk_region_change":
            no_risk_low,

        "risk_aware_high_risk_region_change":
            risk_high,

        "risk_aware_low_risk_region_change":
            risk_low,

        "no_risk_high_low_change_ratio":
            no_risk_high / (no_risk_low + 1e-8),

        "risk_aware_high_low_change_ratio":
            risk_high / (risk_low + 1e-8),
    }


def main():

    print("========================================")
    print("RISK-AWARE REFINEMENT ANALYSIS")
    print("========================================")

    results = {}

    for roi in VAL_ROIS:

        print()
        print(f"Analyzing {roi}...")

        result = analyze_roi(roi)

        results[roi] = result

        print()
        print(
            "Risk vs No-risk refinement "
            f"correlation: "
            f"{result['risk_vs_no_risk_refinement_correlation']:.6f}"
        )

        print(
            "Risk vs Risk-aware refinement "
            f"correlation: "
            f"{result['risk_vs_risk_aware_refinement_correlation']:.6f}"
        )

        print()
        print(
            "No-risk high/low change ratio: "
            f"{result['no_risk_high_low_change_ratio']:.4f}"
        )

        print(
            "Risk-aware high/low change ratio: "
            f"{result['risk_aware_high_low_change_ratio']:.4f}"
        )

    avg_no_risk_ratio = sum(
        r["no_risk_high_low_change_ratio"]
        for r in results.values()
    ) / len(results)

    avg_risk_ratio = sum(
        r["risk_aware_high_low_change_ratio"]
        for r in results.values()
    ) / len(results)

    print()
    print("========================================")
    print("FINAL ANALYSIS")
    print("========================================")

    print(
        f"Average no-risk high/low ratio: "
        f"{avg_no_risk_ratio:.4f}"
    )

    print(
        f"Average risk-aware high/low ratio: "
        f"{avg_risk_ratio:.4f}"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "risk_refinement_analysis.json",
    )

    with open(output_path, "w") as f:
        json.dump(
            {
                "validation_rois": VAL_ROIS,
                "per_roi": results,
                "average_no_risk_high_low_ratio":
                    avg_no_risk_ratio,
                "average_risk_aware_high_low_ratio":
                    avg_risk_ratio,
            },
            f,
            indent=4,
        )

    print()
    print(f"Saved: {output_path}")

    print()
    print("========================================")
    print("ANALYSIS COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()