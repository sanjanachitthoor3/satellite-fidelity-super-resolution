import os
import json
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "outputs/paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def save_figure(fig, name):
    png_path = os.path.join(
        OUTPUT_DIR,
        f"{name}.png"
    )

    pdf_path = os.path.join(
        OUTPUT_DIR,
        f"{name}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# ============================================================
# FIGURE 1
# PSNR COMPARISON
# ============================================================

def figure_1_psnr():

    methods = [
        "LDSR-S2",
        "No-risk\nSafe-SR",
        "Fidelity-only",
        "Uncertainty-only",
        "Fused"
    ]

    psnr = [
        10.1011,
        22.4666,
        20.4406,
        23.0448,
        22.5312
    ]

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    bars = ax.bar(
        methods,
        psnr
    )

    ax.set_ylabel("PSNR (dB)")
    ax.set_title(
        "PSNR Comparison of Super-Resolution Variants"
    )

    ax.set_ylim(
        0,
        max(psnr) * 1.15
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    for bar, value in zip(bars, psnr):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.25,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    fig.tight_layout()

    save_figure(
        fig,
        "fig1_psnr_comparison"
    )


# ============================================================
# FIGURE 2
# RISK-ERROR CORRELATION
# ============================================================

def figure_2_correlation():

    risk_types = [
        "Fidelity",
        "Uncertainty",
        "Fused"
    ]

    roi_2319 = [
        0.6970,
        0.1058,
        0.3378
    ]

    roi_0630 = [
        0.3001,
        -0.0179,
        0.0492
    ]

    x = np.arange(
        len(risk_types)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    bars1 = ax.bar(
        x - width / 2,
        roi_2319,
        width,
        label="ROI_2319"
    )

    bars2 = ax.bar(
        x + width / 2,
        roi_0630,
        width,
        label="ROI_0630"
    )

    ax.set_ylabel(
        "Pearson correlation"
    )

    ax.set_title(
        "Risk-Error Correlation"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        risk_types
    )

    ax.axhline(
        0,
        linewidth=0.8
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.legend()

    for bars in [bars1, bars2]:

        for bar in bars:

            value = bar.get_height()

            offset = (
                0.025
                if value >= 0
                else -0.06
            )

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + offset,
                f"{value:.3f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8
            )

    fig.tight_layout()

    save_figure(
        fig,
        "fig2_risk_error_correlation"
    )


# ============================================================
# FIGURE 3
# CONTROL VS PERTURBED UNCERTAINTY
# ============================================================

def figure_3_uncertainty():

    conditions = [
        "Control",
        "Perturbed"
    ]

    values = [
        0.00003513,
        0.00012250
    ]

    fig, ax = plt.subplots(
        figsize=(5.5, 4.5)
    )

    bars = ax.bar(
        conditions,
        values
    )

    ax.set_ylabel(
        "Mean uncertainty"
    )

    ax.set_title(
        "Control vs Perturbed Uncertainty"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    for bar, value in zip(
        bars,
        values
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2e}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ratio = values[1] / values[0]

    ax.text(
        0.5,
        0.92,
        f"Perturbed / Control = {ratio:.2f}×",
        transform=ax.transAxes,
        ha="center",
        fontsize=10
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig3_control_vs_perturbed_uncertainty"
    )


# ============================================================
# FIGURE 4
# HIGH-RISK 10% VS REMAINING 90%
# ============================================================

def figure_4_high_risk_error():

    labels = [
        "Fidelity",
        "Uncertainty",
        "Fused"
    ]

    roi_2319_high = [
        0.374214,
        0.319082,
        0.333687
    ]

    roi_2319_low = [
        0.300947,
        0.307073,
        0.305450
    ]

    roi_0630_high = [
        0.315659,
        0.302659,
        0.304680
    ]

    roi_0630_low = [
        0.303109,
        0.304553,
        0.304329
    ]

    x = np.arange(
        len(labels)
    )

    width = 0.2

    fig, ax = plt.subplots(
        figsize=(8, 4.8)
    )

    ax.bar(
        x - 1.5 * width,
        roi_2319_high,
        width,
        label="ROI_2319 high-risk 10%"
    )

    ax.bar(
        x - 0.5 * width,
        roi_2319_low,
        width,
        label="ROI_2319 remaining 90%"
    )

    ax.bar(
        x + 0.5 * width,
        roi_0630_high,
        width,
        label="ROI_0630 high-risk 10%"
    )

    ax.bar(
        x + 1.5 * width,
        roi_0630_low,
        width,
        label="ROI_0630 remaining 90%"
    )

    ax.set_ylabel(
        "Mean reconstruction error"
    )

    ax.set_title(
        "Reconstruction Error in High-Risk and Remaining Regions"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.legend(
        loc="upper right",
        fontsize=8
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig4_high_risk_vs_remaining_error"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)

    figure_1_psnr()
    figure_2_correlation()
    figure_3_uncertainty()
    figure_4_high_risk_error()

    print()
    print("=" * 60)
    print("ALL FIGURES GENERATED")
    print("=" * 60)