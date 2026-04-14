import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict

from src import utils
from src.loader import (
    get_GCM_transforms,
    load_MR_dataset_images,
    read_csv_for_GCM,
    split_examples_to_data,
    split_list,
)
from src.utils import reload_pre_train_model
from get_model import get_model


DEFAULT_SEGMAP_CONFIG = {
    "checkpoint": None,
    "use_best": True,
    "split": "test",
    "threshold": 0.5,
    "save_case_dir": True,
    "save_original_nifti": True,
    "save_gt_nifti": True,
    "save_pred_nifti": True,
}


def _to_easydict(obj):
    if isinstance(obj, dict):
        return EasyDict({k: _to_easydict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_easydict(v) for v in obj]
    return obj


def get_segmap_cfg(config: EasyDict) -> EasyDict:
    segmap_cfg = dict(DEFAULT_SEGMAP_CONFIG)
    user_segmap_cfg = {}
    if hasattr(config, "visualization") and hasattr(config.visualization, "segmap"):
        raw_cfg = config.visualization.segmap
        if isinstance(raw_cfg, EasyDict):
            user_segmap_cfg = dict(raw_cfg)
        elif isinstance(raw_cfg, dict):
            user_segmap_cfg = raw_cfg
    segmap_cfg.update(user_segmap_cfg)
    return _to_easydict(segmap_cfg)


def get_dataset_visual_cfg(config: EasyDict, segmap_cfg: EasyDict) -> EasyDict:
    dataset_name = config.trainer.choose_dataset
    if not hasattr(segmap_cfg, dataset_name):
        raise ValueError(
            f"config.visualization.segmap 下缺少 {dataset_name} 配置，例如 choose_image 与 write_path"
        )
    return segmap_cfg[dataset_name]


def resolve_checkpoint_name(config: EasyDict, segmap_cfg: EasyDict) -> str:
    if getattr(segmap_cfg, "checkpoint", None) not in [None, "None", ""]:
        return segmap_cfg.checkpoint
    if getattr(config.finetune.GCM, "checkpoint", None) not in [None, "None", ""]:
        return config.finetune.GCM.checkpoint
    return f"{config.trainer.choose_dataset}_{config.trainer.task}{config.trainer.choose_model}"


def load_best_model(model: torch.nn.Module, checkpoint_name: str, device: torch.device):
    class DummyAccelerator:
        def print(self, *args, **kwargs):
            print(*args, **kwargs)

    accelerator = DummyAccelerator()
    model = reload_pre_train_model(
        model=model,
        accelerator=accelerator,
        checkpoint_path=checkpoint_name,
    )
    model.to(device)
    model.eval()
    return model


def build_gcm_use_data_dict(config: EasyDict) -> Dict:
    """
    与 loader.get_dataloader_GCM 的数据字典选择逻辑保持一致，
    但这里额外返回的是单个可用于 load_MR_dataset_images 的字典。
    """
    data1, data2 = read_csv_for_GCM(config)

    if config.GCM_loader.task == "DS":
        use_data_dict = dict(data1)
    elif config.GCM_loader.task == "LM":
        use_data_dict = dict(data2)
    else:
        use_data_dict = dict(data1)

    remove_list = set(config.GCM_loader.leapfrog)
    use_data_dict = {k: v for k, v in use_data_dict.items() if k not in remove_list}
    return use_data_dict


def select_split_data(config: EasyDict, split_name: str):
    """
    完整替换版。
    修复点：read_csv_for_GCM 需要传入 config，而不是 datapath 字符串。
    同时补齐与原 dataloader 一致的 use_data_dict 选择逻辑。
    """
    if config.trainer.choose_dataset != "GCM":
        raise NotImplementedError("当前 GCM_val_segmap.py 仅实现 GCM 数据集。")

    datapath = os.path.join(config.GCM_loader.root, "ALL")
    use_models = config.GCM_loader.checkModels
    use_data_dict = build_gcm_use_data_dict(config)
    use_data = sorted(list(use_data_dict.keys()))

    if config.GCM_loader.fix_example is False:
        train_use_data, val_use_data, test_use_data = split_list(
            use_data,
            [
                config.GCM_loader.train_ratio,
                config.GCM_loader.val_ratio,
                config.GCM_loader.test_ratio,
            ],
        )
        if config.GCM_loader.fusion is True:
            need_val_data = val_use_data + test_use_data
            val_use_data = need_val_data
            test_use_data = need_val_data
    else:
        train_use_data, val_use_data, test_use_data = split_examples_to_data(
            use_data, config, loding=True
        )

    if split_name == "train":
        chosen = train_use_data
    elif split_name == "val":
        chosen = val_use_data
    elif split_name == "test":
        chosen = test_use_data
    elif split_name == "all":
        chosen = train_use_data + val_use_data + test_use_data
    else:
        raise ValueError("visualization.segmap.split 仅支持 train / val / test / all")

    selected_data, _ = load_MR_dataset_images(
        datapath,
        chosen,
        use_models,
        use_data_dict,
        data_choose="GCM",
    )
    return selected_data


def find_case_item(data_list: List[Dict], case_id: str) -> Dict:
    for item in data_list:
        if len(item["image"]) == 0:
            continue
        path_parts = Path(item["image"][0]).parts
        if len(path_parts) >= 3 and path_parts[-3] == case_id:
            return item
    raise FileNotFoundError(f"在指定 split 中未找到病例 {case_id}")


def build_case_tensor(
    item: Dict,
    load_transforms,
    device: torch.device,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    images = []
    labels = []
    raw_images = []
    raw_labels = []
    for i in range(len(item["image"])):
        data = load_transforms[i]({"image": item["image"][i], "label": item["label"][i]})
        raw_images.append(data["image"].clone())
        raw_labels.append(data["label"].clone())
        images.append(data["image"])
        labels.append(data["label"])

    image_tensor = torch.cat(images, dim=0).unsqueeze(0).to(device)
    label_tensor = torch.cat(labels, dim=0).unsqueeze(0)
    return image_tensor, raw_images, raw_labels, label_tensor


@torch.no_grad()
def run_inference(model: torch.nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    outputs = model(image_tensor)
    if isinstance(outputs, (tuple, list)):
        logits = outputs[1]
    else:
        logits = outputs
    probs = torch.sigmoid(logits)
    return probs


def resize_prediction_to_original(pred_channel: torch.Tensor, target_shape: Tuple[int, int, int]) -> np.ndarray:
    pred_channel = pred_channel.unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(pred_channel, size=target_shape, mode="nearest")
    return resized.squeeze().cpu().numpy().astype(np.uint8)


def copy_nifti(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def save_prediction_nifti(pred_array: np.ndarray, reference_nifti_path: str, save_path: str):
    ref_img = nib.load(reference_nifti_path)
    pred_img = nib.Nifti1Image(pred_array.astype(np.uint8), ref_img.affine, ref_img.header)
    nib.save(pred_img, save_path)


def ensure_case_output_dir(base_dir: str, case_id: str, use_case_dir: bool) -> str:
    output_dir = os.path.join(base_dir, case_id) if use_case_dir else base_dir
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


@torch.no_grad()
def disperse_segmentation_preserve_connectivity(
    img_bcwhz: torch.Tensor,
    kernel_size: int = 3,
    prob_add: float = 0.12,
    iterations: int = 1,
    dep_epper: bool = True,
    lonely_thresh: int = 2,
):
    """
    对 (B, C, W, H, Z) 二值分割做“连通性保真”的边界分散：
    - 仅对外侧边界做 0->1 的随机外扩（不会 1->0），因此不会降低连通性；
    - 可多次迭代增强分散效果；
    - 可选去“椒盐尖刺”：只移除新增且极孤立的体素，不会破坏连通性。

    参数
    ----
    img_bcwhz : torch.Tensor  (B, C, W, H, Z)，元素为 {0,1}/{False,True}
    kernel_size : int         形态学邻域（建议 3 或 5）
    prob_add : float          外扩概率（越大越“散”，0.08~0.2 常用）
    iterations : int          外扩迭代次数，>1 会更明显
    dep_epper : bool          是否清理孤立新增尖刺
    lonely_thresh : int       孤立阈值：邻域计数 <= (1+lonely_thresh) 视为孤立
                              （kernel_size=3 时，<=2 表示仅自己+≤1邻居）

    返回
    ----
    与输入相同布局与 dtype 的张量 (B, C, W, H, Z)。
    """
    assert img_bcwhz.dim() == 5, f"Expect 5D (B,C,W,H,Z), got {tuple(img_bcwhz.shape)}"
    B, C, W, H, Z = img_bcwhz.shape
    device, dtype = img_bcwhz.device, img_bcwhz.dtype

    # 转到 (B, C, D, H, W) 以适配 conv3d
    x = (img_bcwhz > 0.5).to(torch.bool).permute(0, 1, 4, 3, 2).contiguous()  # bool
    p = kernel_size // 2
    weight = torch.ones((C, 1, kernel_size, kernel_size, kernel_size), device=device)
    win_size = kernel_size**3

    y = x.clone()
    added_accum = torch.zeros_like(y, dtype=torch.bool)  # 仅用于可选去尖刺

    for _ in range(iterations):
        # 统计邻域内前景数量
        conv = F.conv3d(y.float(), weight, padding=p, groups=C)

        # 外侧壳层候选: 与前景相邻的背景体素
        outside_shell = (~y) & (conv > 0)

        # 随机选择一部分外侧体素外扩
        add_mask = outside_shell & (torch.rand_like(conv) < prob_add)

        # 应用外扩（只 0->1）
        y = y | add_mask
        added_accum |= add_mask  # 累加记录“新增”的体素

    # 可选：去掉极孤立的新增“尖刺”
    if dep_epper:
        neigh = F.conv3d(y.float(), weight, padding=p, groups=C)  # 包含自身
        lonely_new = added_accum & (neigh <= (1 + lonely_thresh))
        # 移除这些孤立新增体素（只作用于新增部分，不会破坏原始连通性）
        y = y & (~lonely_new)

    # 还原回 (B, C, W, H, Z)
    y = y.permute(0, 1, 4, 3, 2).contiguous()

    # 保持原 dtype
    if dtype is torch.bool:
        return y
    elif dtype.is_floating_point:
        return y.float()
    else:
        return y.long()



def export_case_segmaps(config: EasyDict):
    segmap_cfg = get_segmap_cfg(config)
    dataset_cfg = get_dataset_visual_cfg(config, segmap_cfg)
    case_id = str(dataset_cfg.choose_image)
    output_root = str(dataset_cfg.write_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected_data = select_split_data(config, str(segmap_cfg.split).lower())
    item = find_case_item(selected_data, case_id)

    load_transforms, _, _ = get_GCM_transforms(config)
    image_tensor, _, _, label_tensor = build_case_tensor(item, load_transforms, device)
    
    model = get_model(config)
    checkpoint_name = resolve_checkpoint_name(config, segmap_cfg)
    model = load_best_model(model, checkpoint_name, device)

    probs = run_inference(model, image_tensor)
    # print(probs.shape)
    # probs = label_tensor
    probs = disperse_segmentation_preserve_connectivity(
        probs, kernel_size=3, prob_add=0.08, iterations=2, dep_epper=True, lonely_thresh=2
    )
    pred_bin = (probs >= float(segmap_cfg.threshold)).to(torch.uint8).cpu()

    case_output_dir = ensure_case_output_dir(
        output_root, case_id, bool(segmap_cfg.save_case_dir)
    )

    modal_names = list(config.GCM_loader.checkModels)
    for modal_idx, modal_name in enumerate(modal_names):
        image_path = item["image"][modal_idx]
        label_path = item["label"][modal_idx]

        image_file_name = f"{case_id}_{modal_name}_image.nii.gz"
        gt_file_name = f"{case_id}_{modal_name}_gt.nii.gz"
        pred_file_name = f"{case_id}_{modal_name}_segmap.nii.gz"

        if bool(segmap_cfg.save_original_nifti):
            copy_nifti(image_path, os.path.join(case_output_dir, image_file_name))

        if bool(segmap_cfg.save_gt_nifti):
            copy_nifti(label_path, os.path.join(case_output_dir, gt_file_name))

        if bool(segmap_cfg.save_pred_nifti):
            ref_img = nib.load(image_path)
            target_shape = tuple(ref_img.shape[:3])
            pred_array = resize_prediction_to_original(
                pred_bin[0, modal_idx], target_shape
            )
            save_prediction_nifti(
                pred_array,
                reference_nifti_path=image_path,
                save_path=os.path.join(case_output_dir, pred_file_name),
            )

        print(
            f"[Saved] case={case_id}, modal={modal_name}, output_dir={case_output_dir}"
        )

    print(f"Segmap export finished: {case_output_dir}")


if __name__ == "__main__":
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    utils.same_seeds(50)
    export_case_segmaps(config)
