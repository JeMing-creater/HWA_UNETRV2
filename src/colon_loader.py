import os
import math
import yaml
import random
import numpy as np
import monai
import torch
from monai.transforms import SpatialPadd, RandCropByPosNegLabeld, CenterSpatialCropd
from monai.data.utils import list_data_collate
from easydict import EasyDict
from typing import Tuple, List, Dict, Set

from monai.transforms import (
    CropForegroundd,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
    ResizeWithPadOrCropd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ToTensord,
)

def load_positive_ids(txt_path: str) -> List[str]:
    with open(txt_path, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


def load_dataset_images(root: str, use_ids: List[str]) -> List[Dict]:
    """
    Make list of dicts:
      {'image': <ct_path>, 'label': <mask_path>, 'case_id': <BDMAP_...>}
    """
    data_list = []
    for cid in use_ids:
        case_dir = os.path.join(root, cid)
        img_path = os.path.join(case_dir, "ct.nii.gz")
        label_path = os.path.join(case_dir, "segmentations", "colon_lesion.nii.gz")

        if not os.path.exists(img_path):
            print(f"[Error] missing image: {img_path}")
            continue
        if not os.path.exists(label_path):
            print(f"[Error] missing label: {label_path}")
            continue

        data_list.append({"image": img_path, "label": label_path, "case_id": cid})
    return data_list


def get_colon_transforms(config: EasyDict) -> Tuple[Compose, Compose]:
    patch_size = tuple(config.colon_loader.target_size)
    a_min = config.colon_loader.intensity.a_min
    a_max = config.colon_loader.intensity.a_max

    base = [
        LoadImaged(keys=["image", "label"], image_only=False, simple_keys=True),
        EnsureChannelFirstd(keys=["image", "label"]),

        # 强度归一化
        ScaleIntensityRanged(
            keys=["image"],
            a_min=a_min, a_max=a_max,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),

        # 保证图像至少不小于 patch_size
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
    ]

    train_transform = Compose(
        base + [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=config.colon_loader.crop.pos,
                neg=config.colon_loader.crop.neg, 
                num_samples=config.colon_loader.crop.num_samples,
                image_key="image",
            ),

            RandFlipd(keys=["image", "label"], prob=config.colon_loader.augment.flip_prob, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=config.colon_loader.augment.flip_prob, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=config.colon_loader.augment.flip_prob, spatial_axis=2),

            RandScaleIntensityd(keys="image", factors=config.colon_loader.augment.scale_factor,
                                prob=config.colon_loader.augment.intensity_prob),
            RandShiftIntensityd(keys="image", offsets=config.colon_loader.augment.shift_offset,
                                prob=config.colon_loader.augment.intensity_prob),

            ToTensord(keys=["image", "label"]),
        ]
    )

    val_transform = Compose(
        base + [
        # 以label为前景裁出包含病灶的区域
        CropForegroundd(keys=["image", "label"], source_key="label", margin=32),

        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=patch_size),

        ToTensord(keys=["image", "label"]),
        ]
    )

    return train_transform, val_transform

class ColonLesionDataset(monai.data.Dataset):
    def __init__(self, data: List[Dict], transforms: Compose):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        out = self.transforms(item)

        # 训练集：RandCropByPosNegLabeld 会返回 list[dict]
        if isinstance(out, list):
            for d in out:
                d["case_id"] = item["case_id"]
            return out

        # 验证集：返回单个 dict
        out["case_id"] = item["case_id"]
        return out


# 7 1 2 
def split_list(data: List, ratios: List[float], seed: int = 42):
    random.Random(seed).shuffle(data)
    sizes = [math.ceil(len(data) * r) for r in ratios]
    total = sum(sizes)
    if total != len(data):
        sizes[-1] -= (total - len(data))

    start = 0
    parts = []
    for size in sizes:
        parts.append(data[start : start + size])
        start += size
    return parts  # [train, val, test]


def get_dataloader_colon(config: EasyDict):
    root = config.colon_loader.root
    pos_txt = config.colon_loader.pos_txt

    # 1) load ids
    pos_ids = load_positive_ids(pos_txt)
    remove_list = getattr(config.colon_loader, "leapfrag", [])
    if remove_list:
        pos_ids = [x for x in pos_ids if x not in set(remove_list)]

    # 2) build path list
    all_data = load_dataset_images(root, pos_ids)

    # 3) split
    train_data, val_data, test_data = split_list(
        all_data,
        [config.colon_loader.train_ratio, config.colon_loader.val_ratio, config.colon_loader.test_ratio],
        seed=config.colon_loader.seed,
    )

    # 4) transforms
    train_transform, val_transform = get_colon_transforms(config)

    # 5) dataset
    train_ds = ColonLesionDataset(train_data, transforms=train_transform)
    val_ds   = ColonLesionDataset(val_data, transforms=val_transform)
    test_ds  = ColonLesionDataset(test_data, transforms=val_transform)

    # 6) dataloader
    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=config.colon_loader.num_workers,
        collate_fn=list_data_collate,
    )
    val_loader = monai.data.DataLoader(
        val_ds,
        batch_size=config.trainer.batch_size,
        shuffle=False,
        num_workers=config.colon_loader.num_workers,
        collate_fn=list_data_collate,
    )
    test_loader = monai.data.DataLoader(
        test_ds,
        batch_size=config.trainer.batch_size,
        shuffle=False,
        num_workers=config.colon_loader.num_workers,
        collate_fn=list_data_collate,
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    config = EasyDict(
        yaml.load(open("/mnt/liangjm/AbdomenAtlas2/HWA_UNETRV2/config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )

    train_loader, val_loader, test_loader = get_dataloader_colon(config)

    total_sum = 0

    print("-----train-----")
    for i, batch in enumerate(train_loader):
        print(i, batch["image"].shape, batch["label"].shape, batch["case_id"][:2])
        print("label sum:", batch["label"].sum().item())
        total_sum+=1

    print("-----val-----")
    for i, batch in enumerate(val_loader):
        print(i, batch["image"].shape, batch["label"].shape, batch["case_id"][:2])
        print("label sum:", batch["label"].sum().item())
        total_sum+=1

    print("-----test-----")
    for i, batch in enumerate(test_loader):
        print(i, batch["image"].shape, batch["label"].shape, batch["case_id"][:2])
        total_sum+=1
    
    print(total_sum) # 183