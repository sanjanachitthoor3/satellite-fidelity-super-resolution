import torch
import rioxarray
import opensr_degradation

print("Loading official OpenSR HR image...")

HR_URL = (
    "https://huggingface.co/datasets/isp-uv-es/SEN2NAIP/"
    "resolve/main/demo/cross-sensor/ROI_0000/hr.tif"
)

hr = rioxarray.open_rasterio(HR_URL)

print("HR shape:", hr.shape)
print("HR dtype:", hr.dtype)

# Convert to tensor and normalize
hr_tensor = torch.from_numpy(hr.to_numpy()).float() / 255.0

print("Tensor shape:", hr_tensor.shape)

# Use CPU for this FIRST test.
# Your RTX 2050 only has 4 GB VRAM.
device = "cpu"

print("Creating OpenSR degradation model...")

degradation_model = opensr_degradation.pipe(
    sensor="naip_d",
    add_noise=True,
    params={
        "reflectance_method": [
            "identity",
            "gamma_lognormal",
            "gamma_multivariate_normal",
            "unet_histogram_matching",
            "vae_histogram_matching",
        ],
        "noise_method": "gaussian_noise",
        "device": device,
        "seed": 42,
        "percentiles": [50],
        "vae_reflectance_model": opensr_degradation.naip_vae_model(device),
        "unet_reflectance_model": opensr_degradation.naip_unet_model(device),
    },
)

print("Running degradation...")

lr, hr_harmonized = degradation_model.forward(hr_tensor)

print("SUCCESS!")
print("Original HR:", hr_tensor.shape)
print("Harmonized HR:", hr_harmonized.shape)
print("LR:", lr.shape)

# Save tensors so we can inspect them later
torch.save(hr_tensor, "data/hr_demo.pt")
torch.save(hr_harmonized, "data/hr_harmonized.pt")
torch.save(lr, "data/lr_demo.pt")

print("Saved:")
print("  data/hr_demo.pt")
print("  data/hr_harmonized.pt")
print("  data/lr_demo.pt")