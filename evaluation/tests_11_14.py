import os
import json
import torch

ROIS = ["ROI_2319", "ROI_0630"]
OUT = "outputs/evaluation"
os.makedirs(OUT, exist_ok=True)


def load(path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=True
    ).float()


def clean(x):
    x = x.squeeze()

    if x.ndim == 3:
        x = x.mean(dim=0)

    return x


def pearson(a, b):
    a = a.flatten().double()
    b = b.flatten().double()

    a = a - a.mean()
    b = b - b.mean()

    denom = torch.sqrt(
        (a * a).sum() * (b * b).sum()
    )

    return (a * b).sum().item() / denom.item()


def risk_analysis(roi, risk_name, risk_file, hr, sr):

    # Load risk map
    risk = clean(
        load(f"outputs/risk/{roi}_{risk_file}.pt")
    )

    # Remove batch dimension from SR if present
    # [1, 4, 484, 484] -> [4, 484, 484]
    if sr.ndim == 4:
        sr = sr.squeeze(0)

    # Remove batch dimension from HR if present
    if hr.ndim == 4:
        hr = hr.squeeze(0)

    # Ensure channel dimension exists
    if hr.ndim == 2:
        hr = hr.unsqueeze(0)

    if sr.ndim == 2:
        sr = sr.unsqueeze(0)

    # Pixel-wise reconstruction error
    # Average across the 4 spectral channels
    error = torch.abs(sr - hr).mean(dim=0)

    if risk.shape != error.shape:
        raise ValueError(
            f"{roi} {risk_name}: "
            f"risk {risk.shape} != error {error.shape}"
        )

    # Pearson correlation
    r = pearson(risk, error)

    flat_risk = risk.flatten()
    flat_error = error.flatten()

    # Highest-risk 10%
    threshold = torch.quantile(
        flat_risk,
        0.90
    )

    high = flat_error[
        flat_risk >= threshold
    ]

    low = flat_error[
        flat_risk < threshold
    ]

    high_mean = high.mean().item()
    low_mean = low.mean().item()

    return {
        "pearson_correlation": r,
        "high_risk_10pct_mean_error": high_mean,
        "remaining_90pct_mean_error": low_mean,
        "high_to_low_error_ratio": (
            high_mean / low_mean
        )
    }


print("=" * 70)
print("TESTS 11-14 - COMPLETE RISK ANALYSIS")
print("=" * 70)


results = {
    "test_11_fidelity_risk_error_correlation": {},
    "test_12_uncertainty_risk_error_correlation": {},
    "test_13_fused_risk_error_correlation": {},
    "test_14_control_vs_perturbed_uncertainty": {}
}


# ============================================================
# TESTS 11-13
# ============================================================

all_corr = {
    "fidelity": [],
    "uncertainty": [],
    "fused": []
}


for roi in ROIS:

    print()
    print("=" * 50)
    print(roi)
    print("=" * 50)

    # Keep original HR/SR dimensions
    hr = load(
        f"data/processed/{roi}/hr.pt"
    )

    sr = load(
        f"outputs/sr/{roi}_sr.pt"
    )

    print(f"HR: {hr.shape}")
    print(f"SR: {sr.shape}")

    tests = [
        ("fidelity", "fidelity_risk"),
        ("uncertainty", "uncertainty_risk"),
        ("fused", "risk_map_normalized")
    ]

    for name, filename in tests:

        result = risk_analysis(
            roi,
            name,
            filename,
            hr,
            sr
        )

        print()
        print(f"{name.upper()} RISK")
        print(
            f"Pearson correlation : "
            f"{result['pearson_correlation']:.4f}"
        )
        print(
            f"High-risk 10% error : "
            f"{result['high_risk_10pct_mean_error']:.6f}"
        )
        print(
            f"Other 90% error     : "
            f"{result['remaining_90pct_mean_error']:.6f}"
        )
        print(
            f"High/low ratio      : "
            f"{result['high_to_low_error_ratio']:.4f}"
        )

        if name == "fidelity":

            results[
                "test_11_fidelity_risk_error_correlation"
            ][roi] = result

        elif name == "uncertainty":

            results[
                "test_12_uncertainty_risk_error_correlation"
            ][roi] = result

        else:

            results[
                "test_13_fused_risk_error_correlation"
            ][roi] = result

        all_corr[name].append(
            result["pearson_correlation"]
        )


# ============================================================
# MEAN CORRELATIONS
# ============================================================

results[
    "test_11_fidelity_risk_error_correlation"
]["mean_correlation"] = (
    sum(all_corr["fidelity"])
    / len(all_corr["fidelity"])
)

results[
    "test_12_uncertainty_risk_error_correlation"
]["mean_correlation"] = (
    sum(all_corr["uncertainty"])
    / len(all_corr["uncertainty"])
)

results[
    "test_13_fused_risk_error_correlation"
]["mean_correlation"] = (
    sum(all_corr["fused"])
    / len(all_corr["fused"])
)


# ============================================================
# TEST 14 - CONTROL VS PERTURBED UNCERTAINTY
# ============================================================

print()
print("=" * 70)
print("TEST 14 - CONTROL VS PERTURBED UNCERTAINTY")
print("=" * 70)


perturbed = clean(
    load(
        "outputs/uncertainty/"
        "uncertainty_map.pt"
    )
)

control = clean(
    load(
        "outputs/uncertainty/"
        "control_uncertainty_map.pt"
    )
)


if perturbed.shape != control.shape:

    raise ValueError(
        f"Perturbed {perturbed.shape} "
        f"!= control {control.shape}"
    )


# Excess uncertainty
excess = torch.clamp(
    perturbed - control,
    min=0
)


def stats(x):

    x = x.flatten()

    return {
        "mean": x.mean().item(),
        "median": x.median().item(),
        "p90": torch.quantile(
            x, 0.90
        ).item(),
        "p95": torch.quantile(
            x, 0.95
        ).item(),
        "p99": torch.quantile(
            x, 0.99
        ).item()
    }


pert_stats = stats(perturbed)
control_stats = stats(control)
excess_stats = stats(excess)


ratio = (
    pert_stats["mean"]
    / control_stats["mean"]
)


positive_excess = (
    (excess > 0)
    .float()
    .mean()
    .item()
    * 100
)


results[
    "test_14_control_vs_perturbed_uncertainty"
] = {

    "perturbed": pert_stats,

    "control": control_stats,

    "excess": excess_stats,

    "perturbed_to_control_mean_ratio": ratio,

    "positive_excess_percent": positive_excess
}


print(
    f"Perturbed mean : "
    f"{pert_stats['mean']:.8f}"
)

print(
    f"Control mean   : "
    f"{control_stats['mean']:.8f}"
)

print(
    f"Mean ratio     : "
    f"{ratio:.4f}x"
)

print(
    f"Excess mean    : "
    f"{excess_stats['mean']:.8f}"
)

print(
    f"Positive excess: "
    f"{positive_excess:.2f}%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (
    f"{OUT}/tests_11_14_results.json"
)


with open(output_path, "w") as f:

    json.dump(
        results,
        f,
        indent=2
    )


print()
print("=" * 70)
print("ALL FOUR TESTS COMPLETE")
print("=" * 70)
print(f"Saved: {output_path}")