import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import csv
from datetime import datetime
from typing import Dict

import monai
import torch
import yaml
import pandas as pd
import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.loader import get_dataloader_GCM as get_dataloader
from src.loader import get_GCM_transforms as get_transforms
from src.loader import read_csv_for_GCM
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import (
    Logger,
    write_example,
    resume_train_state,
    split_metrics,
    load_model_dict,
)
from src.eval import (
    calculate_f1_score,
    specificity,
    quadratic_weighted_kappa,
    top_k_accuracy,
    calculate_metrics,
    accumulate_metrics,
    compute_final_metrics,
)

from get_model import get_model


def load_model(model, accelerator, checkpoint):
    from safetensors.torch import load_file
    try:
        check_path = f"{os.getcwd()}/model_store/{checkpoint}/best/"
        weights = load_file(f"{check_path}/model.safetensors")
        accelerator.print("load model from %s" % check_path)
        # checkpoint = load_model_dict(
        #     check_path + "pytorch_model.bin",
        # )
        model.load_state_dict(weights)
        accelerator.print(f"Load checkpoint model successfully!")
    except Exception as e:
        accelerator.print(e)
        accelerator.print(f"Failed to load checkpoint model!")
    return model


def normalize_dl_scores(dl_scores):
    """
    正态化 DL-scores，并确保它们保持在 [0, 1] 区间内
    """
    # 计算所有 DL-scores 的均值和标准差
    scores = list(dl_scores.values())  # 获取所有 DL-scores 的值
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    # Z-score 标准化
    standardized_scores = {
        k: (v - mean_score) / std_score for k, v in dl_scores.items()
    }

    # 使用 sigmoid 将值映射回 [0, 1] 范围
    normalized_scores = {
        k: 1 / (1 + np.exp(-v)) for k, v in standardized_scores.items()
    }

    return normalized_scores


def write_to_xlsx(dl_score, lable_score, real_label_score, csv_name):
    def change_to_xlsx(csv_file, save_path):
        df = pd.read_csv(csv_file, dtype={0: str})

        # 清理病人编号列中的前后空格
        df.iloc[:, 0] = df.iloc[:, 0].str.strip()

        # 将数据保存为 Excel 文件
        df.to_excel(save_path, index=False, engine="openpyxl")

        os.remove(csv_file)

    csv_path = csv_name + ".csv"
    # 判断路径是否存在
    dir_path = os.path.dirname(csv_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"dir {dir_path} has been created!")
    else:
        print(f"dir {dir_path} existed!")

    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        # 写入表头
        writer.writerow(["Key", "Value", "Label", "Ground Truth"])

        # 遍历字典并写入键值对
        for key, value in dl_score.items():
            writer.writerow([str(key), value, lable_score[key], real_label_score[key]])

    xlsx_path = csv_name + ".xlsx"
    change_to_xlsx(
        os.path.join(csv_path),
        os.path.join(xlsx_path),
    )


@torch.no_grad()
def val_dl_score(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    name: str = "train",
):
    # 验证
    model.eval()
    dl_score = {}
    dl_label = {}
    real_label = {}
    for i, image_batch in enumerate(val_loader):
        fname = (
            image_batch["image"]
            .meta["filename_or_obj"][0]
            .split("/")[-1]
            .split("/")[-1]
            .split(".")[0]
        )

        if (
            config.trainer.choose_model == "HWAUNETR"
            or config.trainer.choose_model == "HSL_Net"
        ):
            logits, _ = model(image_batch["image"])
        else:
            logits = model(
                image_batch["image"]
            )  # some moedls can not accepted inference, I do not know why.

        log = ""

        dl_score[fname] = float(torch.sigmoid(logits).cpu().numpy())

        if config.GCM_loader.task_choose == 'LMM':
            labels_loss = image_batch["class_label"]
        else:
            labels_loss = image_batch["PFS_label"]

        for metric_name in metrics:
            y_pred = post_trans(logits)
            y = labels_loss
            if metric_name == "miou_metric":
                y_pred = y_pred.unsqueeze(2)
                y = y.unsqueeze(2)
            metrics[metric_name](y_pred=y_pred, y=y)

        y_pred = post_trans(logits)
        dl_label[fname] = int(y_pred.cpu().numpy())
        real_label[fname] = int(labels_loss.cpu().numpy())
        accelerator.print(f"Now is valing {fname} file", flush=True)

    # dl_score = normalize_dl_scores(dl_score)
    write_to_xlsx(
        dl_score,
        dl_label,
        real_label,
        os.path.join(config.valer.dl_score_csv_path, f"{name}_dl_score"),
    )
    metric = {}

    for metric_name in metrics:
        # for channel in range(channels):
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)

        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes

        # give every single task metric
        metrics[metric_name].reset()
        # task_num = channel + 1
        metric.update(
            {
                f"{metric_name}": float(batch_acc.mean()),
            }
        )
    accelerator.print(metric)
    return dl_score, dl_label, real_label


if __name__ == "__main__":
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    utils.same_seeds(50)
    logging_dir = (
        os.getcwd()
        + "/logs/"
        + config.finetune.GCM.checkpoint
        + str(datetime.now())
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
        .replace(".", "_")
    )
    accelerator = Accelerator(
        cpu=False, log_with=["tensorboard"], project_dir=logging_dir
    )
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    accelerator.print("load model...")
    model = get_model(config)
    if config.finetune.GCM.checkpoint != 'None':
        checkpoint_name = config.finetune.GCM.checkpoint
    else:
        checkpoint_name = config.trainer.choose_dataset + "_" + config.trainer.task + "_" + config.GCM_loader.task_choose + "_" + config.trainer.choose_model
    model = load_model(model, accelerator, checkpoint_name)

    accelerator.print("load dataset...")

    # 验证时强制batch_size=1
    config.trainer.batch_size = 1
    train_loader, val_loader, test_loader, example = get_dataloader(config)

    loss_functions = {
        "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
        "bce_loss": nn.BCEWithLogitsLoss().to(accelerator.device),
    }

    metrics = {
        "accuracy": monai.metrics.ConfusionMatrixMetric(
            include_background=False, metric_name="accuracy"
        ),
        "f1": monai.metrics.ConfusionMatrixMetric(
            include_background=False, metric_name="f1 score"
        ),
        "specificity": monai.metrics.ConfusionMatrixMetric(
            include_background=False, metric_name="specificity"
        ),
        "recall": monai.metrics.ConfusionMatrixMetric(
            include_background=False, metric_name="recall"
        ),
        "miou_metric": monai.metrics.MeanIoU(include_background=False),
    }

    post_trans = monai.transforms.Compose(
        [
            monai.transforms.Activations(sigmoid=True),
            monai.transforms.AsDiscrete(threshold=0.5),
        ]
    )

    model, train_loader, val_loader, test_loader = accelerator.prepare(
        model, train_loader, val_loader, test_loader
    )

    # start valing
    accelerator.print("Start Valing! ")
    val_dl_score(model, train_loader, metrics, post_trans, accelerator, name="train")
    val_dl_score(model, val_loader, metrics, post_trans, accelerator, name="val")
    val_dl_score(model, test_loader, metrics, post_trans, accelerator, name="test")
