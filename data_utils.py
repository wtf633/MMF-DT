import os
import glob
from typing import Dict, List, Tuple

import pandas as pd
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    RandAdjustContrastd,
    RandRotate90d,
    RandFlipd,
    RandZoomd,
    EnsureTyped,
    AsChannelFirstd
)
from monai.data import Dataset, DataLoader, list_data_collate

__all__ = [
    "build_file_dicts",
    "get_transforms",
    "get_dataloaders",
]


def build_file_dicts(img_dir: str,
                     csv_path: str) -> List[Dict]:
    """
    Construct a list of dictionaries required by MONAI Dataset.
    """
    img_paths = sorted(
        glob.glob(os.path.join(img_dir, "Patient_*.nii.gz"))
    )
    df = pd.read_csv(csv_path).astype({"ID": str}).sort_values("ID")

    keys = [
        "OS_status", "OS_time", "ID",
        "PFS_status", "PFS_time",
        "PVTT", "LungMet", "BoneMet",
        "Up_to_seven", "LNMet",
        "Age", "Male", "Child-Pugh",
        "HBV_infection", "stage"
    ]

    files = [
        {"input": img_paths[i],
         "OS_status": df["OS_status"].iloc[i],
         "OS_time": df["OS_time"].iloc[i],
         "ID": df["ID"].iloc[i],
         "PFS_status": df["PFS_status"].iloc[i],
         "PFS_time": df["PFS_time"].iloc[i],
         "PVTT": df["PVTT"].iloc[i],
         "LungMet": df["LungMet"].iloc[i],
         "BoneMet": df["BoneMet"].iloc[i],
         "Up_to_seven": df["Up_to_seven"].iloc[i],
         "LNMet": df["LNMet"].iloc[i],
         "Age": df["Age"].iloc[i],
         "Gender": df["Male"].iloc[i],           # renamed “Male” → “Gender”
         "Child-Pugh": df["Child-Pugh"].iloc[i],
         "HBV_infection": df["HBV_infection"].iloc[i],
         "Stage": df["stage"].iloc[i]}
        for i in range(len(img_paths))
    ]
    return files


def get_transforms(rand_p: float):
    train_tf = Compose([
        LoadImaged(keys=["input"]),
        EnsureChannelFirstd(keys=["input"], channel_dim=0),
        # Resized(keys=["input"], spatial_size=[128, 128, 128]),  # augment, flip, rotate, intensity ...
        RandAdjustContrastd(keys=["input"], prob=rand_p),
        RandRotate90d(keys=["input"], prob=rand_p, spatial_axes=(0, 2)),
        RandRotate90d(keys=["input"], prob=rand_p, spatial_axes=(0, 1)),
        RandFlipd(keys=["input"], prob=rand_p),
        RandZoomd(keys="input", prob=rand_p),
        EnsureTyped(keys=["input"]),
    ])
    val_tf = Compose([
        LoadImaged(keys=["input"]),
        AsChannelFirstd(keys=["input"], channel_dim=0),
        # Resized(keys=["input"], spatial_size=[128, 128, 128]),
        EnsureTyped(keys=["input"]),
    ])
    return train_tf, val_tf


def get_dataloaders(train_files: List[Dict],
                    val_files: List[Dict],
                    test_files: List[Dict],
                    train_tf,
                    val_tf,
                    batch_sizes: Tuple[int, int, int] = (16, 8, 8)):
    train_loader = DataLoader(
        Dataset(train_files, train_tf),
        batch_size=batch_sizes[0],
        shuffle=True,
        num_workers=8,
        collate_fn=list_data_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Dataset(val_files, val_tf),
        batch_size=batch_sizes[1],
        shuffle=False,
        num_workers=3,
        collate_fn=list_data_collate,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Dataset(test_files, val_tf),
        batch_size=batch_sizes[2],
        shuffle=False,
        num_workers=3,
        collate_fn=list_data_collate,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
