import torch
import opensr_degradation


class OpenSRNAIPDegradation:
    """
    ESA OpenSR NAIP degradation adapter.

    Expected input:
        CHW tensor with 4 channels: R, G, B, NIR

    For the benchmark setup:
        HR: 484 x 484
        LR: 121 x 121
        Scale: 4x
    """

    def __init__(
        self,
        add_noise=False,
        device="cuda",
        seed=42,
    ):
        self.device = device

        self.params = {
            "reflectance_method": ["identity"],
            "device": device,
            "seed": seed,
        }

        self.model = opensr_degradation.pipe(
            sensor="naip_d",
            add_noise=add_noise,
            params=self.params,
        )

    @torch.no_grad()
    def __call__(self, image):
        """
        Degrade a 4-channel HR/SR image to LR.

        Parameters
        ----------
        image : torch.Tensor
            Shape: [4, H, W]

        Returns
        -------
        torch.Tensor
            LR image.
        """

        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")

        if image.ndim != 3:
            raise ValueError(
                f"Expected CHW tensor, got shape {tuple(image.shape)}"
            )

        if image.shape[0] != 4:
            raise ValueError(
                f"Expected 4 channels (RGB+NIR), got {image.shape[0]}"
            )

        image = image.float().to(self.device)

        lr, _ = self.model.forward(image)

        if lr.ndim == 4 and lr.shape[0] == 1:
            lr = lr.squeeze(0)

        if lr.shape != (4, 121, 121):
            raise ValueError(
                f"Expected OpenSR output shape (4, 121, 121), got {tuple(lr.shape)}"
            )

        return lr