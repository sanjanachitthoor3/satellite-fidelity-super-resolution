import torch
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Load saved tensors
# -----------------------------
hr = torch.load("data/hr_demo.pt", weights_only=False)
lr = torch.load("data/lr_demo.pt", weights_only=False)

print("HR:", hr.shape)
print("LR:", lr.shape)

# Create output directory
output_dir = Path("outputs/degradation")
output_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helper: convert 4-band image
# to RGB for visualization
# -----------------------------
def to_rgb(img):
    """
    Assumes channel order:
    [R, G, B, NIR]

    img shape:
        [4, H, W]
    """

    rgb = img[:3].permute(1, 2, 0).numpy()

    # Normalize for visualization
    rgb_min = rgb.min()
    rgb_max = rgb.max()

    rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)

    return rgb


# -----------------------------
# HR image
# -----------------------------
hr_rgb = to_rgb(hr)

plt.figure(figsize=(8, 8))
plt.imshow(hr_rgb)
plt.title("Original HR Image")
plt.axis("off")
plt.tight_layout()
plt.savefig(
    output_dir / "HR.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()


# -----------------------------
# Each degraded LR image
# -----------------------------
for i in range(lr.shape[0]):

    lr_rgb = to_rgb(lr[i])

    plt.figure(figsize=(6, 6))
    plt.imshow(lr_rgb)
    plt.title(f"Degraded LR - Method {i+1}")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        output_dir / f"LR{i+1}.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# -----------------------------
# Comparison figure
# -----------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(hr_rgb)
axes[0, 0].set_title("Original HR")

for i in range(5):
    lr_rgb = to_rgb(lr[i])

    row = (i + 1) // 3
    col = (i + 1) % 3

    axes[row, col].imshow(lr_rgb)
    axes[row, col].set_title(f"LR Method {i+1}")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    output_dir / "HR_vs_all_LR.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\nSaved visualizations to:")
print(output_dir)