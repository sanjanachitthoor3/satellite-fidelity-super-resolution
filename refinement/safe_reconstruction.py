import torch
import torch.nn as nn


class RiskAwareResidualNet(nn.Module):
    """
    Residual refinement network.

    IMPORTANT CHANGE from the original version:
    This network ALWAYS takes 9 input channels — SR (4) + LR_up (4) + risk (1).

    There is no more `in_channels=8` / risk=None path. For the "no-risk"
    ablation, pass a zero tensor of shape [B,1,H,W] as `risk` instead of
    omitting the channel entirely.

    Why: the original code gave the risk-aware model in_channels=9 and the
    no-risk baseline in_channels=8. That means the two models had different
    first-layer parameter counts and different random initializations —
    so any PSNR/SSIM gap between them was partly measuring init variance
    and capacity, not just whether the risk information helps. Keeping the
    architecture identical and only zeroing the risk channel isolates the
    one thing you actually want to test: does the *information* in the
    risk map improve reconstruction, holding everything else constant.
    """

    def __init__(self, in_channels=9, out_channels=4, features=64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, sr, lr_up, risk):
        """
        sr:     [B, 4, H, W]
        lr_up:  [B, 4, H, W]
        risk:   [B, 1, H, W]  -- pass torch.zeros_like(...) for the
                                  no-risk ablation, never None.
        """
        x = torch.cat([sr, lr_up, risk], dim=1)
        return self.network(x)


def make_zero_risk(reference_tensor):
    """
    Build a zero risk channel with the same batch size, spatial dims,
    device and dtype as `reference_tensor` (pass in `sr` or `lr_up`).
    Use this for the no-risk ablation so the two models see literally
    identical shapes and only differ in whether the risk channel
    carries information.
    """
    b, _, h, w = reference_tensor.shape
    return torch.zeros(
        (b, 1, h, w),
        dtype=reference_tensor.dtype,
        device=reference_tensor.device,
    )