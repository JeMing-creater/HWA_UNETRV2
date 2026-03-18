import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import monai
import nibabel as nib
import numpy as np
import pandas as pd
import torch
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


DEFAULT_SINGLE_CONFIG = {
    "checkpoint": None,
    "threshold": 0.5,
    "include_train": True,
    "include_val": True,
    "include_test": True,
    "save_csv": True,
    "save_xlsx": False,
    "export_filename": "GCM_single_seg_metrics.csv",
    "verbose": True,
    "only_case": None,
}


def _to_easydict(obj):
    if isinstance(obj, dict):
        return EasyDict({k: _to_easydict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_easydict(v) for v in obj]
    return obj


def get_single_cfg(config: EasyDict) -> EasyDict:
    merged = dict(DEFAULT_SINGLE_CONFIG)
    if hasattr(config, "visualization") and hasattr(config.visualization, "single"):
        raw_cfg = config.visualization.single
        if isinstance(raw_cfg, EasyDict):
            merged.update(dict(raw_cfg))
        elif isinstance(raw_cfg, dict):
            merged.update(raw_cfg)
    return _to_easydict(merged)


def get_dataset_single_cfg(config: EasyDict, single_cfg: EasyDict) -> EasyDict:
    dataset_name = config.trainer.choose_dataset
    if hasattr(single_cfg, dataset_name):
        return single_cfg[dataset_name]
    fallback = {}
    if hasattr(config, "visualization") and hasattr(config.visualization, "visual"):
        if hasattr(config.visualization.visual, dataset_name):
            fallback = dict(config.visualization.visual[dataset_name])
    if "write_path" not in fallback:
        fallback["write_path"] = "output_single_metrics"
    if "choose_image" not in fallback:
        fallback["choose_image"] = None
    return _to_easydict(fallback)


def resolve_checkpoint_name(config: EasyDict, single_cfg: EasyDict) -> str:
    if getattr(single_cfg, "checkpoint", None) not in [None, "None", ""]:
        return str(single_cfg.checkpoint)
    if getattr(config.finetune.GCM, "checkpoint", None) not in [None, "None", ""]:
        return str(config.finetune.GCM.checkpoint)
    return f"{config.trainer.choose_dataset}_{config.trainer.task}{config.trainer.choose_model}"


def freeze_seg_unused_heads(model: torch.nn.Module):
    target = model.module if hasattr(model, "module") else model
    if hasattr(target, "Class_Decoder"):
        for p in target.Class_Decoder.parameters():
            p.requires_grad = False


def load_best_model(model: torch.nn.Module, checkpoint_name: str, device: torch.device):
    class DummyAccelerator:
        def print(self, *args, **kwargs):
            print(*args, **kwargs)

    if config.trainer.choose_model in ["HWAUNETR", "HSL_Net"]:
        freeze_seg_unused_heads(model)
    model = reload_pre_train_model(
        model=model,
        accelerator=DummyAccelerator(),
        checkpoint_path=checkpoint_name,
    )
    model.to(device)
    model.eval()
    return model


def build_use_data_dict(config: EasyDict) -> Dict:
    data1, data2 = read_csv_for_GCM(config)
    if config.GCM_loader.task == "DS":
        use_data_dict = data1
    elif config.GCM_loader.task == "LM":
        use_data_dict = data2
    else:
        use_data_dict = data1
    return use_data_dict


def select_case_ids_by_split(config: EasyDict) -> Dict[str, List[str]]:
    use_data_dict = build_use_data_dict(config)
    use_data = [item for item in use_data_dict.keys() if item not in config.GCM_loader.leapfrog]

    if config.GCM_loader.fix_example is not True:
        if config.GCM_loader.time_limit is not True:
            import random
            random.shuffle(use_data)
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

    return {
        "train": list(train_use_data),
        "val": list(val_use_data),
        "test": list(test_use_data),
        "use_data_dict": use_data_dict,
    }


def load_split_items(config: EasyDict, split_case_ids: Dict[str, List[str]]) -> List[Dict]:
    datapath = os.path.join(config.GCM_loader.root, "ALL")
    use_models = config.GCM_loader.checkModels
    use_data_dict = split_case_ids["use_data_dict"]

    results = []
    for split_name in ["train", "val", "test"]:
        selected_ids = split_case_ids[split_name]
        if len(selected_ids) == 0:
            continue
        split_items, _ = load_MR_dataset_images(
            datapath,
            selected_ids,
            use_models,
            use_data_dict,
            data_choose="GCM",
        )
        for item in split_items:
            item["split"] = split_name
            results.append(item)
    return results


def get_real_case_folder_name(item: Dict) -> str:
    if "image" not in item or len(item["image"]) == 0:
        return ""
    return Path(item["image"][0]).parts[-3]


def normalize_case_id(case_id: Optional[str]) -> Optional[str]:
    if case_id is None:
        return None
    case_id = str(case_id).strip()
    if case_id == "":
        return None
    return case_id.lstrip("0") or "0"


def case_matches(item: Dict, choose_image: Optional[str]) -> bool:
    if choose_image in [None, "", "None"]:
        return True
    real_folder = get_real_case_folder_name(item)
    requested = choose_image
    normalized_requested = normalize_case_id(requested)
    normalized_real = normalize_case_id(real_folder)
    return str(requested) == str(real_folder) or normalized_requested == normalized_real


def build_case_tensor(item: Dict, load_transforms, device: torch.device):
    images = []
    labels = []
    for i in range(len(item["image"])):
        data = load_transforms[i]({"image": item["image"][i], "label": item["label"][i]})
        images.append(data["image"])
        labels.append(data["label"])
    image_tensor = torch.cat(images, dim=0).unsqueeze(0).to(device)
    label_tensor = torch.cat(labels, dim=0).unsqueeze(0).to(device)
    return image_tensor, label_tensor


@torch.no_grad()
def run_inference(model: torch.nn.Module, image_tensor: torch.Tensor, threshold: float):
    outputs = model(image_tensor)
    logits = outputs[1] if isinstance(outputs, (tuple, list)) else outputs
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    return probs, preds


@torch.no_grad()
def compute_binary_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    pred = pred.float().unsqueeze(0).unsqueeze(0)
    target = target.float().unsqueeze(0).unsqueeze(0)

    dice_metric = monai.metrics.DiceMetric(
        include_background=True,
        reduction=monai.utils.MetricReduction.MEAN,
        get_not_nans=False,
    )
    dice_metric(y_pred=pred, y=target)
    dice_value = float(dice_metric.aggregate().detach().cpu().item())
    dice_metric.reset()

    pred_sum = float(pred.sum().item())
    target_sum = float(target.sum().item())
    if pred_sum == 0.0 and target_sum == 0.0:
        hd95_value = 0.0
    elif pred_sum == 0.0 or target_sum == 0.0:
        hd95_value = np.nan
    else:
        hd95_metric = monai.metrics.HausdorffDistanceMetric(
            percentile=95,
            include_background=True,
            reduction=monai.utils.MetricReduction.MEAN,
            get_not_nans=False,
        )
        hd95_metric(y_pred=pred, y=target)
        hd95_value = float(hd95_metric.aggregate().detach().cpu().item())
        hd95_metric.reset()

    return dice_value, hd95_value


@torch.no_grad()
def evaluate_single_case(
    item: Dict,
    model: torch.nn.Module,
    config: EasyDict,
    load_transforms,
    device: torch.device,
    threshold: float,
) -> List[Dict]:
    image_tensor, label_tensor = build_case_tensor(item, load_transforms, device)
    _, preds = run_inference(model, image_tensor, threshold)

    real_case_folder = get_real_case_folder_name(item)
    request_case_id = str(item.get("requested_case_id", real_case_folder))
    split_name = item.get("split", "unknown")
    modal_names = list(config.GCM_loader.checkModels)

    rows = []
    for modal_idx, modal_name in enumerate(modal_names):
        pred_modal = preds[0, modal_idx]
        label_modal = label_tensor[0, modal_idx]
        dice_value, hd95_value = compute_binary_metrics(pred_modal, label_modal)

        gt_voxels = int(label_modal.sum().item())
        pred_voxels = int(pred_modal.sum().item())
        image_path = item["image"][modal_idx]
        label_path = item["label"][modal_idx]

        rows.append(
            {
                "split": split_name,
                "requested_case_id": request_case_id,
                "real_case_folder": real_case_folder,
                "modal": modal_name,
                "dice": dice_value,
                "hd95": hd95_value,
                "pred_voxels": pred_voxels,
                "gt_voxels": gt_voxels,
                "image_path": image_path,
                "label_path": label_path,
            }
        )

    mean_dice = float(np.nanmean([row["dice"] for row in rows])) if rows else np.nan
    mean_hd95 = float(np.nanmean([row["hd95"] for row in rows])) if rows else np.nan
    rows.append(
        {
            "split": split_name,
            "requested_case_id": request_case_id,
            "real_case_folder": real_case_folder,
            "modal": "MEAN",
            "dice": mean_dice,
            "hd95": mean_hd95,
            "pred_voxels": int(sum(row["pred_voxels"] for row in rows)),
            "gt_voxels": int(sum(row["gt_voxels"] for row in rows)),
            "image_path": "",
            "label_path": "",
        }
    )
    return rows


def ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_metric_table(df: pd.DataFrame, output_dir: str, single_cfg: EasyDict) -> Tuple[Optional[str], Optional[str]]:
    csv_path = None
    xlsx_path = None

    export_name = str(single_cfg.export_filename)
    if not export_name.lower().endswith(".csv"):
        export_name = export_name + ".csv"
    if bool(single_cfg.save_csv):
        csv_path = os.path.join(output_dir, export_name)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    if bool(single_cfg.save_xlsx):
        xlsx_name = os.path.splitext(export_name)[0] + ".xlsx"
        xlsx_path = os.path.join(output_dir, xlsx_name)
        df.to_excel(xlsx_path, index=False)

    return csv_path, xlsx_path


def print_case_summary(df: pd.DataFrame, focus_case: Optional[str]):
    if focus_case in [None, "", "None"]:
        return

    norm_focus = normalize_case_id(focus_case)
    keep = df[
        df["real_case_folder"].astype(str).map(normalize_case_id) == norm_focus
    ]
    if len(keep) == 0:
        print(f"[Warning] 未在导出结果中找到指定病例: {focus_case}")
        return

    print("\n========== Single Case Metrics ==========")
    print(keep[["split", "requested_case_id", "real_case_folder", "modal", "dice", "hd95"]].to_string(index=False))
    print("========================================\n")



def main(config: EasyDict):
    if config.trainer.choose_dataset != "GCM":
        raise NotImplementedError("GCM_val_single.py 当前仅实现 GCM 数据集。")

    single_cfg = get_single_cfg(config)
    dataset_cfg = get_dataset_single_cfg(config, single_cfg)
    output_dir = ensure_output_dir(str(dataset_cfg.write_path))
    choose_image = getattr(dataset_cfg, "choose_image", None)
    if getattr(single_cfg, "only_case", None) not in [None, "", "None"]:
        choose_image = single_cfg.only_case

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_name = resolve_checkpoint_name(config, single_cfg)

    model = get_model(config)
    model = load_best_model(model, checkpoint_name, device)
    load_transforms, _, _ = get_GCM_transforms(config)

    split_case_ids = select_case_ids_by_split(config)
    items = load_split_items(config, split_case_ids)

    include_splits = []
    if bool(single_cfg.include_train):
        include_splits.append("train")
    if bool(single_cfg.include_val):
        include_splits.append("val")
    if bool(single_cfg.include_test):
        include_splits.append("test")

    records = []
    for item in items:
        if item.get("split") not in include_splits:
            continue
        real_case_folder = get_real_case_folder_name(item)
        item["requested_case_id"] = real_case_folder
        case_rows = evaluate_single_case(
            item=item,
            model=model,
            config=config,
            load_transforms=load_transforms,
            device=device,
            threshold=float(single_cfg.threshold),
        )
        records.extend(case_rows)
        if bool(single_cfg.verbose):
            print(f"[Done] split={item.get('split')} case={real_case_folder}")

    df = pd.DataFrame(records)
    if len(df) == 0:
        raise RuntimeError("没有生成任何评估结果，请检查数据划分、路径或配置。")

    df = df.sort_values(by=["split", "real_case_folder", "modal"]).reset_index(drop=True)
    csv_path, xlsx_path = save_metric_table(df, output_dir, single_cfg)
    print_case_summary(df, choose_image)

    print("[Finished] 单病例/全量病例分割指标计算完成")
    if csv_path is not None:
        print(f"[Saved CSV] {csv_path}")
    if xlsx_path is not None:
        print(f"[Saved XLSX] {xlsx_path}")


if __name__ == "__main__":
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    utils.same_seeds(50)
    main(config)
