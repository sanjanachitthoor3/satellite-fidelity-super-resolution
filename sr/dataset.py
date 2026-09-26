from pathlib import Path

import torch
from torch.utils.data import Dataset


class SEN2NAIPDataset(Dataset):
    """
    SEN2NAIP dataset using preprocessed LR/HR tensors.

    LR: 4 x 121 x 121
    HR: 4 x 484 x 484
    """

    def __init__(self, root="data/processed", split="train"):
        self.root = Path(root)

        split_file = self.root / "splits" / f"{split}.txt"

        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}"
            )

        self.roi_ids = [
            line.strip()
            for line in split_file.read_text().splitlines()
            if line.strip()
        ]

        if not self.roi_ids:
            raise ValueError(
                f"No ROI IDs found in {split_file}"
            )

    def __len__(self):
        return len(self.roi_ids)

    def __getitem__(self, index):

        roi_id = self.roi_ids[index]
        roi_dir = self.root / roi_id

        lr_path = roi_dir / "lr.pt"
        hr_path = roi_dir / "hr.pt"

        if not lr_path.exists():
            raise FileNotFoundError(lr_path)

        if not hr_path.exists():
            raise FileNotFoundError(hr_path)

        lr = torch.load(
            lr_path,
            map_location="cpu",
            weights_only=True,
        )

        hr = torch.load(
            hr_path,
            map_location="cpu",
            weights_only=True,
        )

        if lr.shape != (4, 121, 121):
            raise ValueError(
                f"{roi_id}: unexpected LR shape {lr.shape}"
            )

        if hr.shape != (4, 484, 484):
            raise ValueError(
                f"{roi_id}: unexpected HR shape {hr.shape}"
            )

        return {
            "lr": lr.float(),
            "hr": hr.float(),
            "roi_id": roi_id,
        }