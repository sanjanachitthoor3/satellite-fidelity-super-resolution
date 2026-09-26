import torch

from degradation.opensr_degradation import (
    create_naip_degradation_model,
    degrade_sr_to_lr,
)


ROI_ID = "ROI_1732"

LR_PATH = f"data/processed/{ROI_ID}/lr.pt"
SR_PATH = f"outputs/sr/{ROI_ID}_sr.pt"

lr = torch.load(LR_PATH, map_location="cpu", weights_only=True)
sr = torch.load(SR_PATH, map_location="cpu", weights_only=True)

if sr.ndim == 4:
    sr = sr.squeeze(0)

print("LR shape:", lr.shape)
print("SR shape:", sr.shape)


for add_noise in [False, True]:

    model = create_naip_degradation_model(
        device="cpu",
        seed=42,
        add_noise=add_noise,
    )

    lr_prime = degrade_sr_to_lr(sr, model)

    # Select the verified gamma_multivariate_normal branch
    lr_prime = lr_prime[2]

    error = (lr_prime - lr).abs()

    print()
    print("add_noise =", add_noise)
    print("LR' shape:", lr_prime.shape)
    print("Mean error:", float(error.mean()))
    print("Median error:", float(error.median()))
    print("P90 error:", float(torch.quantile(error.flatten(), 0.90)))
    print("P95 error:", float(torch.quantile(error.flatten(), 0.95)))
    print("P99 error:", float(torch.quantile(error.flatten(), 0.99)))