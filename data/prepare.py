import io
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import rasterio
import torch


# ============================================================
# Configuration
# ============================================================

ZIP_PATH = Path("data/sen2naip/cross-sensor/cross-sensor.zip")
OUTPUT_DIR = Path("data/processed")

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

MAX_ROIS = None  #change this to None for the full dataset.
# Set MAX_ROIS to an integer for a small debugging run.
# Example: MAX_ROIS = 20
# Keep None for the full dataset.


# ============================================================
# TIFF reader directly from ZIP
# ============================================================

def read_tiff_from_zip(zf, filename):
    data = zf.read(filename)

    with rasterio.open(io.BytesIO(data)) as src:
        image = src.read()

    return image


# ============================================================
# Normalization
# ============================================================

def normalize_lr(lr):
    """
    SEN2NAIP LR convention:
    Sentinel-2 values are scaled by 10000.
    """
    return lr.astype(np.float32) / 10000.0


def normalize_hr(hr):
    """
    SEN2NAIP HR convention:
    NAIP uint8 values are scaled by 255.
    """
    return hr.astype(np.float32) / 255.0


# ============================================================
# Main
# ============================================================

def main():

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Dataset ZIP not found: {ZIP_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)

    print(f"Opening dataset: {ZIP_PATH}")

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:

        names = zf.namelist()

        roi_ids = sorted(
            {
                Path(name).parts[1]
                for name in names
                if name.startswith("cross-sensor/")
                and name.endswith("/hr.tif")
            }
        )

        print(f"Found {len(roi_ids)} ROI pairs.")

        if MAX_ROIS is not None:
            roi_ids = roi_ids[:MAX_ROIS]
            print(f"Using first {len(roi_ids)} ROIs for debugging.")

        # ----------------------------------------------------
        # Shuffle and split
        # ----------------------------------------------------

        random.shuffle(roi_ids)

        n = len(roi_ids)

        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_ids = roi_ids[:n_train]
        val_ids = roi_ids[n_train:n_train + n_val]
        test_ids = roi_ids[n_train + n_val:]

        print()
        print("Dataset split:")
        print(f"Train: {len(train_ids)}")
        print(f"Val:   {len(val_ids)}")
        print(f"Test:  {len(test_ids)}")

        # ----------------------------------------------------
        # Save split files
        # ----------------------------------------------------

        split_dir = OUTPUT_DIR / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)

        (split_dir / "train.txt").write_text(
            "\n".join(train_ids)
        )

        (split_dir / "val.txt").write_text(
            "\n".join(val_ids)
        )

        (split_dir / "test.txt").write_text(
            "\n".join(test_ids)
        )

        # ----------------------------------------------------
        # Process every ROI
        # ----------------------------------------------------

        for index, roi_id in enumerate(roi_ids, start=1):

            hr_path = f"cross-sensor/{roi_id}/hr.tif"
            lr_path = f"cross-sensor/{roi_id}/lr.tif"
            metadata_path = f"cross-sensor/{roi_id}/metadata.json"

            hr = read_tiff_from_zip(zf, hr_path)
            lr = read_tiff_from_zip(zf, lr_path)

            # Expected:
            # LR = 4 × 121 × 121
            # HR = 4 × 484 × 484

            if lr.shape != (4, 121, 121):
                raise ValueError(
                    f"{roi_id}: unexpected LR shape {lr.shape}"
                )

            if hr.shape != (4, 484, 484):
                raise ValueError(
                    f"{roi_id}: unexpected HR shape {hr.shape}"
                )

            # Official SEN2NAIP scaling
            lr = normalize_lr(lr)
            hr = normalize_hr(hr)

            lr_tensor = torch.from_numpy(lr)
            hr_tensor = torch.from_numpy(hr)

            # Save individual processed tensors
            roi_dir = OUTPUT_DIR / roi_id
            roi_dir.mkdir(parents=True, exist_ok=True)

            torch.save(lr_tensor, roi_dir / "lr.pt")
            torch.save(hr_tensor, roi_dir / "hr.pt")

            # Preserve metadata
            metadata = json.loads(
                zf.read(metadata_path).decode("utf-8")
            )

            (roi_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2)
            )

            if index % 50 == 0 or index == 1:
                print(
                    f"[{index}/{len(roi_ids)}] "
                    f"{roi_id} | "
                    f"LR {tuple(lr_tensor.shape)} | "
                    f"HR {tuple(hr_tensor.shape)}"
                )

    print()
    print("Dataset preparation complete.")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()