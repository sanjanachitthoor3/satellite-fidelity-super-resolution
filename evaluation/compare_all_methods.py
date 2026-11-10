from pathlib import Path
import sys
import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from skimage.metrics import structural_similarity
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity


# ============================================================
# CONFIG
# ============================================================

ROIS = ["ROI_2319", "ROI_0630"]

BASELINE_DIR = Path("outputs/baselines")
REFINEMENT_DIR = Path("outputs/refinement")
DATA_DIR = Path("baseline_inputs")

METHODS = {
    "Bicubic": BASELINE_DIR / "bicubic",
    "LDSR-S2": BASELINE_DIR / "ldsr_s2",
    "Safe-SR (no risk)": REFINEMENT_DIR,
    "Safe-SR (risk-aware)": REFINEMENT_DIR,
}

# OpenSR consistency settings used by your existing evaluation pipeline
CONSISTENCY_SEED = 42


# ============================================================
# LOADERS
# ============================================================

def load_tensor(path):
    x = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{path} did not contain a tensor")

    x = x.float()

    if x.ndim == 4 and x.shape[0] == 1:
        x = x.squeeze(0)

    if x.shape != (4, 484, 484):
        raise ValueError(
            f"{path}: expected (4,484,484), got {tuple(x.shape)}"
        )

    return x


def load_hr(roi):
    return load_tensor(DATA_DIR / roi / "hr.pt")


def prediction_path(method, roi):
    if method == "Bicubic":
        return BASELINE_DIR / "bicubic" / f"{roi}_sr.pt"

    if method == "LDSR-S2":
        return BASELINE_DIR / "ldsr_s2" / f"{roi}_sr.pt"

    if method == "Safe-SR (no risk)":
        return REFINEMENT_DIR / f"{roi}_no-risk_safe_sr.pt"

    if method == "Safe-SR (risk-aware)":
        return REFINEMENT_DIR / f"{roi}_risk_safe_sr.pt"

    raise ValueError(method)


# ============================================================
# METRICS
# ============================================================

def psnr(pred, target):
    """
    PSNR over all four spectral bands.

    Assumes tensors are normalized to [0,1].
    """
    mse = torch.mean((pred - target) ** 2).item()

    if mse == 0:
        return float("inf")

    return 10.0 * np.log10(1.0 / mse)


def ssim_4band(pred, target):
    """
    Compute SSIM independently for each of the four bands,
    then average.
    """
    pred_np = pred.numpy()
    target_np = target.numpy()

    values = []

    for c in range(4):
        value = structural_similarity(
            target_np[c],
            pred_np[c],
            data_range=1.0,
        )
        values.append(value)

    return float(np.mean(values))


def sam(pred, target):
    """
    Mean Spectral Angle Mapper (SAM), in degrees.

    Each pixel is treated as a 4-dimensional spectral vector.
    """
    p = pred.permute(1, 2, 0).reshape(-1, 4)
    t = target.permute(1, 2, 0).reshape(-1, 4)

    dot = torch.sum(p * t, dim=1)

    p_norm = torch.linalg.vector_norm(p, dim=1)
    t_norm = torch.linalg.vector_norm(t, dim=1)

    denominator = p_norm * t_norm

    valid = denominator > 1e-12

    cosine = torch.zeros_like(dot)
    cosine[valid] = dot[valid] / denominator[valid]

    cosine = torch.clamp(cosine, -1.0, 1.0)

    angles = torch.acos(cosine)

    angles = angles[valid]

    return torch.rad2deg(angles).mean().item()


def lpips_rgb(pred, target, lpips_metric):
    """
    LPIPS on RGB channels only.

    Tensor order is RGB-NIR, so channels 0,1,2 are RGB.

    Inputs are [4,H,W] and are converted to [1,3,H,W].
    normalize=True in TorchMetrics handles [0,1] -> [-1,1].
    """

    pred_rgb = pred[:3].unsqueeze(0)
    target_rgb = target[:3].unsqueeze(0)

    # Reset accumulated TorchMetrics state so each ROI
    # produces an independent LPIPS value.
    lpips_metric.reset()

    with torch.no_grad():
        value = lpips_metric(
            pred_rgb,
            target_rgb,
        )

    return value.item()


# ============================================================
# LR CONSISTENCY
# ============================================================

def create_opensr_degradation_model():
    """
    Uses the same OpenSR degradation configuration as the
    existing final_evaluation.py.
    """

    try:
        from degradation.opensr_degradation import (
            create_naip_degradation_model,
        )
    except ImportError:
        raise ImportError(
            "Could not import create_naip_degradation_model "
            "from degradation.opensr_degradation.py"
        )

    return create_naip_degradation_model(
        device="cpu",
        seed=CONSISTENCY_SEED,
        add_noise=True,
        reflectance_methods=["gamma_multivariate_normal"],
    )


def lr_consistency(pred, lr, degradation_model):
    """
    Degrade SR back to LR using the exact OpenSR evaluation
    procedure already used in final_evaluation.py.
    """

    # OpenSR expects [4, 484, 484]
    sr_input = pred

    with torch.no_grad():
        degraded_lr, _ = degradation_model.forward(sr_input)

    if degraded_lr.ndim == 3:
        degraded_lr = degraded_lr.unsqueeze(0)

    # LR should be [1, 4, 121, 121]
    lr_batch = lr.unsqueeze(0)

    if degraded_lr.shape != lr_batch.shape:
        raise ValueError(
            "OpenSR degraded LR shape mismatch: "
            f"{degraded_lr.shape} vs {lr_batch.shape}"
        )

    return torch.mean(
        torch.abs(degraded_lr - lr_batch)
    ).item()


# ============================================================
# EVALUATE ONE METHOD
# ============================================================

def evaluate_method(
    method,
    degradation_model,
    lpips_metric,
):
    results = []

    for roi in ROIS:

        hr = load_hr(roi)

        lr = torch.load(
            DATA_DIR / roi / "lr.pt",
            map_location="cpu",
            weights_only=True,
        ).float()

        pred_path = prediction_path(method, roi)

        if not pred_path.exists():
            print(f"[SKIP] Missing: {pred_path}")
            continue

        # Clamp exactly as the existing evaluation does.
        pred = torch.clamp(
            load_tensor(pred_path),
            0.0,
            1.0,
        )

        hr = torch.clamp(
            hr,
            0.0,
            1.0,
        )

        # ----------------------------------------------------
        # Sanity checks
        # ----------------------------------------------------

        if pred.shape != hr.shape:
            raise ValueError(
                f"{method} / {roi}: "
                f"prediction {pred.shape} != HR {hr.shape}"
            )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        p = psnr(pred, hr)
        s = ssim_4band(pred, hr)
        a = sam(pred, hr)
        l = lpips_rgb(
            pred,
            hr,
            lpips_metric,
        )
        c = lr_consistency(
            pred,
            lr,
            degradation_model,
        )

        results.append({
            "roi": roi,
            "PSNR": p,
            "SSIM": s,
            "SAM": a,
            "LPIPS": l,
            "Consistency": c,
        })

        print(
            f"{method:22s} | {roi} | "
            f"PSNR={p:.4f} | "
            f"SSIM={s:.6f} | "
            f"SAM={a:.6f} | "
            f"LPIPS={l:.6f} | "
            f"Consistency={c:.6f}"
        )

    if not results:
        return None

    return {
        "PSNR": np.mean(
            [x["PSNR"] for x in results]
        ),
        "SSIM": np.mean(
            [x["SSIM"] for x in results]
        ),
        "SAM": np.mean(
            [x["SAM"] for x in results]
        ),
        "LPIPS": np.mean(
            [x["LPIPS"] for x in results]
        ),
        "Consistency": np.mean(
            [x["Consistency"] for x in results]
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("SAFE-SR COMMON EVALUATION")
    print("=" * 100)

    print("\nROIs:")
    for roi in ROIS:
        print(f"  {roi}")

    print("\nMethods:")
    for method in METHODS:
        print(f"  {method}")

    # --------------------------------------------------------
    # OpenSR degradation model
    # --------------------------------------------------------

    print("\nCreating OpenSR degradation model...")

    degradation_model = create_opensr_degradation_model()

    # --------------------------------------------------------
    # LPIPS model
    # --------------------------------------------------------

    print("Creating LPIPS model...")

    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex",
        normalize=True,
    )

    lpips_metric.eval()

    print("LPIPS model ready.")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    all_results = {}

    print("\n" + "=" * 100)
    print("PER-ROI RESULTS")
    print("=" * 100)

    for method in METHODS:

        print(f"\n--- {method} ---")

        result = evaluate_method(
            method,
            degradation_model,
            lpips_metric,
        )

        if result is not None:
            all_results[method] = result

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print("\n\n" + "=" * 115)
    print("FINAL COMPARISON TABLE")
    print("=" * 115)

    print(
        f"{'Method':24s}"
        f"{'PSNR ↑':>12s}"
        f"{'SSIM ↑':>12s}"
        f"{'SAM ↓':>14s}"
        f"{'LPIPS ↓':>14s}"
        f"{'Consistency ↓':>18s}"
    )

    print("-" * 115)

    for method, result in all_results.items():

        print(
            f"{method:24s}"
            f"{result['PSNR']:12.4f}"
            f"{result['SSIM']:12.6f}"
            f"{result['SAM']:14.6f}"
            f"{result['LPIPS']:14.6f}"
            f"{result['Consistency']:18.6f}"
        )

    print("=" * 115)


if __name__ == "__main__":
    main()