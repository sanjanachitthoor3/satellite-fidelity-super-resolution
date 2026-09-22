import torch

from degradation.opensr_degradation import OpenSRNAIPDegradation


def test_opensr_naip_output_shape():
    hr = torch.rand(4, 484, 484)

    degradation = OpenSRNAIPDegradation(
        add_noise=False,
        device="cuda",
    )

    lr = degradation(hr)

    assert lr.shape == (4, 121, 121)
    assert torch.isfinite(lr).all()


def test_opensr_naip_four_channels():
    hr = torch.rand(4, 484, 484)

    degradation = OpenSRNAIPDegradation(
        add_noise=False,
        device="cuda",
    )

    lr = degradation(hr)

    assert lr.shape[0] == 4