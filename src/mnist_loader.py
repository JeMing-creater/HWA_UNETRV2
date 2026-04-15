from medmnist import AdrenalMNIST3D

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, RandFlipd, RandRotate90d

class MedMNIST3DWrapper(Dataset):
    """
    MedMNIST 3D 数据集包装器
    功能：
    1. 将 numpy 数据转为 Tensor
    2. 增加 Channel 维度，适配 MONAI (D,H,W) -> (C,D,H,W)
    3. 包装成字典 {"image": ..., "class_label": ..., "sample_index": ...}
    """
    def __init__(self, base_ds, transforms=None):
        self.base_ds = base_ds
        self.transforms = transforms

    def __len__(self):
        # 返回数据集的总长度
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, target = self.base_ds[idx]

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        else:
            img = torch.as_tensor(img)
        
        img = img.float()

        if img.ndim == 3:
            img = img.unsqueeze(0)  # 变成 [1, 64, 64, 64]

        label_val = int(np.array(target).reshape(-1)[0])
        class_label = torch.tensor([label_val], dtype=torch.long)

        # 记录样本索引
        sample_index = torch.tensor([idx], dtype=torch.long)

        sample = {
            "image": img,           # 形状: [1, D, H, W]
            "class_label": class_label,   # 形状: [1]
            "sample_index": sample_index  # 形状: [1]
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


def get_mnist_transforms():
    train_tf = Compose([
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
        # 只在一个平面内做 90 度旋转
        RandRotate90d(keys=["image"], prob=0.5, max_k=3, spatial_axes=(1, 2)),
    ])
    val_tf = Compose([])  # 验证/测试不做增强
    return train_tf, val_tf


def generate_medmnist_ids(dataset, split_name):
    ids = []
    for i in range(len(dataset)):
        # 格式化 ID
        s_id = f"{split_name}_{i}" 
        ids.append(s_id)
    return ids

def get_dataloader_mnist(
    root="/mnt/liangjm/AbdomenAtlas2/AdrenalMNIST3D",
    size=64,
    batch_size=4,
    num_workers=4,
    download=True,
    pin_memory=True,
):
    # medmnist 官方 split：train/val/test
    train_base = AdrenalMNIST3D(split="train", size=64, download=download, root=root)
    val_base   = AdrenalMNIST3D(split="val",   size=64, download=download, root=root)
    test_base  = AdrenalMNIST3D(split="test",  size=64, download=download, root=root)

    train_tf, val_tf = get_mnist_transforms()

    train_ds = MedMNIST3DWrapper(train_base, transforms=train_tf)
    val_ds   = MedMNIST3DWrapper(val_base,   transforms=val_tf)
    test_ds  = MedMNIST3DWrapper(test_base,  transforms=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    
    train_example = generate_medmnist_ids(train_ds, "train")
    val_example = generate_medmnist_ids(val_ds, "val")
    test_example = generate_medmnist_ids(test_ds, "test")

    example = [train_example, val_example, test_example]

    return train_loader, val_loader, test_loader, example

if __name__ == "__main__":
    tr, val, te, ex = get_dataloader_mnist(batch_size=16, num_workers=0)
    
    print("Train Loader Length:", len(tr))
    print("Example Structure Type:", type(ex))
    print("Example List Lengths (Train, Val, Test):", len(ex[0]), len(ex[1]), len(ex[2]))
    
    print("\n--- 查看前几个生成的 ID ---")
    print("Train Examples (Top 5):", ex[0][:5])
    print("Val Examples (Top 5):",   ex[1][:5])
    print("Test Examples (Top 5):",  ex[2][:5])