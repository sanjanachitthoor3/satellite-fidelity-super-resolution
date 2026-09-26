import os
import sys

import torch
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


VAL_ROIS = [
    "ROI_2319",
    "ROI_0630",
]

OUTPUT_DIR = "outputs/evaluation/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_tensor(path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    ).float()


def to_rgb(x):
    """
    Convert a 4-band tensor [1,4,H,W] or [4,H,W]
    to a displayable RGB image using bands 0,1,2.
    """
    if x.ndim == 4:
        x = x.squeeze(0)

    rgb = x[:3].permute(1, 2, 0)

    low = torch.quantile(rgb, 0.01)
    high = torch.quantile(rgb, 0.99)

    rgb = (rgb - low) / (high - low + 1e-8)
    rgb = torch.clamp(rgb, 0.0, 1.0)

    return rgb.numpy()


def load_risk_map(roi):
    path = (
        f"outputs/risk/"
        f"{roi}_risk_map_normalized.pt"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Risk map not found: {path}"
        )

    return load_tensor(path)


def main():

    print("========================================")
    print("SAFE-SR VISUALIZATION")
    print("========================================")

    for roi in VAL_ROIS:

        print()
        print(f"Processing {roi}...")

        hr = load_tensor(
            f"data/processed/{roi}/hr.pt"
        )

        lr = load_tensor(
            f"data/processed/{roi}/lr.pt"
        )

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

        risk_map = load_risk_map(roi)

        lr_rgb = to_rgb(lr)
        sr_rgb = to_rgb(sr)
        no_risk_rgb = to_rgb(no_risk)
        risk_aware_rgb = to_rgb(risk_aware)
        hr_rgb = to_rgb(hr)

        if risk_map.ndim == 4:
            risk_map = risk_map.squeeze(0)

        if risk_map.ndim == 3:
            risk_map = risk_map.mean(dim=0)

        risk_map = torch.clamp(
            risk_map,
            0.0,
            1.0,
        ).numpy()

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(15, 9),
        )

        axes[0, 0].imshow(lr_rgb)
        axes[0, 0].set_title("LR Input")

        axes[0, 1].imshow(sr_rgb)
        axes[0, 1].set_title("LDSR-S2")

        axes[0, 2].imshow(no_risk_rgb)
        axes[0, 2].set_title("No-risk Safe-SR")

        axes[1, 0].imshow(risk_aware_rgb)
        axes[1, 0].set_title("Risk-aware Safe-SR")

        axes[1, 1].imshow(hr_rgb)
        axes[1, 1].set_title("HR Ground Truth")

        im = axes[1, 2].imshow(
            risk_map,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        axes[1, 2].set_title("Fused Risk Map")

        fig.colorbar(
            im,
            ax=axes[1, 2],
            fraction=0.046,
            pad=0.04,
        )

        for ax in axes.flat:
            ax.axis("off")

        fig.suptitle(
            f"{roi} — Safe-SR Comparison",
            fontsize=16,
        )

        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{roi}_comparison.png",
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved: {output_path}")

    print()
    print("========================================")
    print("VISUALIZATION COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()