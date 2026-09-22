import torch
import pytest

from fidelity.consistency import (
    reconstruct_lr_from_sr,
    l1_error_map,
    mse_error_map,
    channel_reduce,
    consistency_score,
    fidelity_check,
)


def identity_degradation(x):
    return x


def test_reconstruct_lr_from_sr():
    sr = torch.rand(3, 8, 8)

    result = reconstruct_lr_from_sr(sr, identity_degradation)

    assert torch.equal(result, sr)


def test_identical_images_have_zero_l1_error():
    lr = torch.rand(3, 8, 8)

    error = l1_error_map(lr, lr)

    assert torch.allclose(error, torch.zeros_like(error))


def test_different_images_have_nonzero_l1_error():
    lr = torch.zeros(3, 8, 8)
    reconstructed_lr = torch.ones(3, 8, 8)

    error = l1_error_map(lr, reconstructed_lr)

    assert torch.all(error > 0)


def test_identical_images_have_zero_mse():
    lr = torch.rand(3, 8, 8)

    error = mse_error_map(lr, lr)

    assert torch.allclose(error, torch.zeros_like(error))


def test_channel_reduction():
    error = torch.ones(3, 8, 8)

    spatial = channel_reduce(error, reduction="mean")

    assert spatial.shape == (8, 8)
    assert torch.allclose(spatial, torch.ones(8, 8))


def test_consistency_score_zero_for_identical_images():
    lr = torch.rand(3, 8, 8)

    score = consistency_score(lr, lr, metric="l1")

    assert torch.isclose(score, torch.tensor(0.0))


def test_fidelity_check():
    lr = torch.rand(3, 8, 8)
    sr = lr.clone()

    result = fidelity_check(
        lr,
        sr,
        identity_degradation,
        metric="l1",
    )

    assert "reconstructed_lr" in result
    assert "error_map" in result
    assert "spatial_error_map" in result
    assert "score" in result

    assert torch.isclose(result["score"], torch.tensor(0.0))