"""
Patch extraction for Safe-SR refinement training.

Why this exists: the original training loop treated each of the 8
training ROIs as ONE training example, giving 8 gradient-update
opportunities per epoch. That is far too little data for a network to
reliably learn that the risk channel should modulate the correction —
the optimizer's path of least resistance is to ignore or under-use it.

This module crops aligned patches out of HR / SR / LR_up / risk (all
in the same coordinate space, at HR/SR resolution), with random flips
and 90-degree rotations for augmentation. With patch_size=96 and
stride=32 on a 484x484 image you get on the order of 150+ patches per
ROI, i.e. well over 1000 training examples per epoch from the same 8
ROIs -- and each patch is *smaller* than a full image, so this also
reduces peak VRAM per step, which helps on a 4GB card.

patch_size and stride are kept multiples of 4 so that dividing by 4
to get the matching LR-resolution crop (for the LR-consistency loss)
is always a clean integer.
"""

import random
import torch


def extract_patch_coords(image_size, patch_size, stride):
    """
    All valid top-left coordinates along one axis, always multiples of
    `stride` (which should be a multiple of 4), plus one final
    coordinate flush with the far edge so no image content is dropped.
    """
    coords = list(range(0, image_size - patch_size + 1, stride))
    if not coords or coords[-1] != image_size - patch_size:
        coords.append(image_size - patch_size)
    return coords


def build_patch_index(roi_ids, image_size=484, patch_size=96, stride=32):
    """
    Returns a flat list of (roi_id, row, col) tuples. `row`/`col` are
    top-left corners in HR/SR/risk coordinates (0..484). Iterate over
    this list (shuffled each epoch) as your training set.
    """
    rows = extract_patch_coords(image_size, patch_size, stride)
    cols = extract_patch_coords(image_size, patch_size, stride)

    index = []
    for roi_id in roi_ids:
        for r in rows:
            for c in cols:
                index.append((roi_id, r, c))
    return index


def crop_hr(tensor, row, col, patch_size):
    """tensor: [1, C, 484, 484] -> [1, C, patch_size, patch_size]."""
    return tensor[:, :, row:row + patch_size, col:col + patch_size]


def crop_lr(tensor, row, col, patch_size):
    """
    Crop the matching region out of the true 121x121 LR tensor.
    row/col/patch_size are in HR coordinates; this divides by 4 to
    land on the corresponding LR-resolution crop. Requires row, col,
    and patch_size to all be multiples of 4 (true by construction if
    you build the patch index with a stride that's a multiple of 4).
    """
    assert row % 4 == 0 and col % 4 == 0 and patch_size % 4 == 0, (
        "row/col/patch_size must be multiples of 4 to align with the "
        "121x121 LR grid -- check your patch_size/stride."
    )
    r, c, p = row // 4, col // 4, patch_size // 4
    return tensor[:, :, r:r + p, c:c + p]


def augment(*tensors):
    """
    Apply the SAME random flip/rotation to every tensor passed in, so
    HR / SR / LR_up / risk stay spatially aligned with each other.
    Returns a list in the same order as the input.
    """
    tensors = list(tensors)

    if random.random() < 0.5:
        tensors = [t.flip(-1) for t in tensors]  # horizontal flip

    if random.random() < 0.5:
        tensors = [t.flip(-2) for t in tensors]  # vertical flip

    k = random.choice([0, 1, 2, 3])
    if k:
        tensors = [torch.rot90(t, k, dims=(-2, -1)) for t in tensors]

    return tensors