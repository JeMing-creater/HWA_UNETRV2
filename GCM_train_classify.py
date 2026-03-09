from math import e
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
from datetime import datetime
from typing import Dict

import monai
import torch
import yaml
from tqdm import tqdm
import torch.nn as nn
from dataclasses import dataclass, field
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.loader import get_dataloader_GCM as get_dataloader
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger, write_example, resume_train_state, split_metrics,reload_pre_train_model,freeze_seg_decoder
from src.eval import (
    calculate_f1_score,
    specificity,
    quadratic_weighted_kappa,
    top_k_accuracy,
    calculate_metrics,
    accumulate_metrics,
    compute_final_metrics,
)

# from src.model.HWAUNETR_class import HWAUNETR
# from src.model.SwinUNETR import MultiTaskSwinUNETR
# from monai.networks.nets import SwinUNETR
# from src.model.ResNet import ResNet3DClassifier as ResNet
# from src.model.Vit import ViT3D
from get_model import get_model


def train_one_epoch(
    model: torch.nn.Module,
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch: int,
    step: int,
):
    # 训练
    model.train()
    accelerator.print(f"Training...", flush=True)
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    # for i, image_batch in enumerate(train_loader):
    for i, image_batch in loop:
        # for i, image_batch in enumerate(train_loader):
        
        if config.trainer.choose_model == "HWAUNETR" or config.trainer.choose_model == "HSL_Net":
            logits, _ = model(image_batch["image"])
        else:
            logits = model(
                image_batch["image"]
            )  # some moedls can not accepted inference, I do not know why.
        total_loss = 0
        
        
        if config.GCM_loader.task_Mu == None or config.GCM_loader.task_Mu == 1:
        
            logits_loss = logits
            if config.GCM_loader.task_choose == 'LMM':
                labels = image_batch["class_label"]
            else:
                labels = image_batch["PFS_label"]
            for name in loss_functions:
                alpth = 1
                loss = loss_functions[name](logits_loss, labels.float())
                accelerator.log({"Train/" + name: float(loss)}, step=step)
                total_loss += alpth * loss
            for metric_name in metrics[0]:
                y_pred = post_trans(logits)
                y = labels
                if metric_name == "miou_metric":
                    y_pred = y_pred.unsqueeze(2)
                    y = y.unsqueeze(2)
                metrics[0][metric_name](y_pred=y_pred, y=y)
        else:
            logits_loss = logits
            labels = image_batch["class_label"]
            PFS_labels = image_batch["PFS_label"]
            
            for name in loss_functions:
                alpth = 1
                loss1 = loss_functions[name](logits_loss[0], labels.float())
                loss2 = loss_functions[name](logits_loss[1], PFS_labels.float())
                
                accelerator.log({"Train/" + name + "_class": float(loss1)}, step=step)
                accelerator.log({"Train/" + name + "_PFS": float(loss2)}, step=step)
                
                total_loss += alpth * (loss1 + loss2)

            for metric_name in metrics[0]:
                
                logits_loss = logits
                
                y_pred_1 = post_trans(logits[0])
                y_pred_2 = post_trans(logits[1])
                y1 = labels
                y2 = PFS_labels
                if metric_name == "miou_metric":
                    y_pred_1 = y_pred_1.unsqueeze(2)
                    y_pred_2 = y_pred_2.unsqueeze(2)
                    y1 = y1.unsqueeze(2)
                    y2 = y2.unsqueeze(2)
                metrics[0][metric_name](y_pred=y_pred_1, y=y1)
                metrics[1][metric_name](y_pred=y_pred_2, y=y2)
        

        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        accelerator.log(
            {
                "Train/Total Loss": float(total_loss),
            },
            step=step,
        )
        # accelerator.print(
        #     f'Epoch [{epoch+1}/{config.trainer.num_epochs}][{i + 1}/{len(train_loader)}] Training Loss:{total_loss}',
        #     flush=True
        #     )
        # 更新信息
        loop.set_description(f"Epoch [{epoch+1}/{config.trainer.num_epochs}]")
        loop.set_postfix(loss=total_loss)
        step += 1
    scheduler.step(epoch)

    if config.GCM_loader.task_Mu == None:
        task_Mu = 1
    else:
        task_Mu = config.GCM_loader.task_Mu
    metric = {}
    for i in range(task_Mu):
        metric_s = metrics[i]
        for metric_name in metric_s:
            batch_acc = metric_s[metric_name].aggregate()[0].to(accelerator.device)

            if accelerator.num_processes > 1:
                batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes

            # give every single task metric
            metric_s[metric_name].reset()
            metric.update(
                {
                    f"Train/Task {i}/{metric_name}": float(batch_acc.mean()),
                }
            )
            
        # for metric_name in metric_s:
        #     all_data = []
        #     for key in metric_s.keys():
        #         if metric_name in key:
        #             all_data.append(metric_s[key])
        #     me_data = sum(all_data) / len(all_data)
        #     metric.update({f"Train/Task {i}/{metric_name}": float(me_data)})

    accelerator.log(metric, step=epoch)
    
    
    
    return metric, step


@torch.no_grad()
def val_one_epoch(
    model: torch.nn.Module,
    inference: monai.inferers.Inferer,
    val_loader: torch.utils.data.DataLoader,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    step: int,
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    test: bool = False,
):
    # 验证
    model.eval()
    if test:
        flag = "Test"
        accelerator.print(f"Testing...", flush=True)
    else:
        flag = "Val"
        accelerator.print(f"Valing...", flush=True)
    loop = tqdm(enumerate(val_loader), total=len(val_loader))
    # for i, image_batch in enumerate(val_loader):
    for i, image_batch in loop:
        # logits = inference(model, image_batch['image'])
        if config.trainer.choose_model == "HWAUNETR" or config.trainer.choose_model == "HSL_Net":
            logits, _ = model(image_batch["image"])
        else:
            logits = model(
                image_batch["image"]
            )  # some moedls can not accepted inference, I do not know why.
        log = ""
        total_loss = 0

        if config.GCM_loader.task_Mu == None or config.GCM_loader.task_Mu == 1:
            
        
            logits_loss = logits
            
            if config.GCM_loader.task_choose == 'LMM':
                labels_loss = image_batch["class_label"]
            else:
                labels_loss = image_batch["PFS_label"]
            

            for name in loss_functions:
                loss = loss_functions[name](logits_loss, labels_loss.float())
                accelerator.log({f"{flag}/" + name: float(loss)}, step=step)
                log += f"{name} {float(loss):1.5f} "
                total_loss += loss
            for metric_name in metrics[0]:
                y_pred = post_trans(logits)
                y = labels_loss
                if metric_name == "miou_metric":
                    y_pred = y_pred.unsqueeze(2)
                    y = y.unsqueeze(2)
                metrics[0][metric_name](y_pred=y_pred, y=y)
        else:
            logits_loss = logits
            labels_loss = image_batch["class_label"]
            PFS_labels_loss = image_batch["PFS_label"]

            for name in loss_functions:
                loss1 = loss_functions[name](logits_loss[0], labels_loss.float())
                loss2 = loss_functions[name](logits_loss[1], PFS_labels_loss.float())
                
                accelerator.log({f"{flag}/" + name + "_class": float(loss1)}, step=step)
                accelerator.log({f"{flag}/" + name + "_PFS": float(loss2)}, step=step)
                
                log += f"{name}_class {float(loss1):1.5f} "
                log += f"{name}_PFS {float(loss2):1.5f} "
                
                total_loss += (loss1 + loss2)   
            for metric_name in metrics[0]:
                
                logits_loss = logits
                
                y_pred_1 = post_trans(logits[0])
                y_pred_2 = post_trans(logits[1])
                y1 = labels_loss
                y2 = PFS_labels_loss
                if metric_name == "miou_metric":
                    y_pred_1 = y_pred_1.unsqueeze(2)
                    y_pred_2 = y_pred_2.unsqueeze(2)
                    y1 = y1.unsqueeze(2)
                    y2 = y2.unsqueeze(2)
                metrics[0][metric_name](y_pred=y_pred_1, y=y1)
                metrics[1][metric_name](y_pred=y_pred_2, y=y2)
    
        accelerator.log(
            {
                f"{flag}/Total Loss": float(total_loss),
            },
            step=step,
        )

        # accelerator.print(
        #     f'[{i + 1}/{len(val_loader)}] {flag} Validation Loading...',
        #     flush=True)
        loop.set_description(f"Epoch [{epoch+1}/{config.trainer.num_epochs}]")
        loop.set_postfix(loss=total_loss)
        step += 1
    
    
    if config.GCM_loader.task_Mu == None:
        task_Mu = 1
    else:
        task_Mu = config.GCM_loader.task_Mu
    metric = {}
    for i in range(task_Mu):
        metric_s = metrics[i]
        for metric_name in metric_s:
            # for channel in range(channels):
            batch_acc = metric_s[metric_name].aggregate()[0].to(accelerator.device)

            if accelerator.num_processes > 1:
                batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes

            # give every single task metric
            metric_s[metric_name].reset()
            # task_num = channel + 1
            metric.update(
                {
                    f"{flag}/Task {i}/{metric_name}": float(batch_acc.mean()),
                }
            )
    
    
    
    # metric = {}

    # for metric_name in metrics:
    #     # for channel in range(channels):
    #     batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)

    #     if accelerator.num_processes > 1:
    #         batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes

    #     # give every single task metric
    #     metrics[metric_name].reset()
    #     # task_num = channel + 1
    #     metric.update(
    #         {
    #             f"{flag}/{metric_name}": float(batch_acc.mean()),
    #         }
    #     )

    accelerator.log(metric, step=epoch)
    return metric, step


if __name__ == "__main__":
    config = EasyDict(
        yaml.load(open("config_class.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    utils.same_seeds(50)
    
    if config.finetune.GCM.checkpoint != 'None':
        checkpoint_name = config.finetune.GCM.checkpoint
    else:
        checkpoint_name = config.trainer.choose_dataset + "_" + config.trainer.task + "_" + config.GCM_loader.task_choose + "_" + config.trainer.choose_model
    
    logging_dir = (
        os.getcwd()
        + "/logs/"
        + checkpoint_name
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
    if config.trainer.choose_model == "HWAUNETR":
        reload_pre_train_model(model, accelerator, "GCM_SegmentationTFM_UNET_seg")
        freeze_seg_decoder(model)

    accelerator.print("load dataset...")
    train_loader, val_loader, test_loader, example = get_dataloader(config)

    # keep example log
    if accelerator.is_main_process == True:
        write_example(config, example)

    inference = monai.inferers.SlidingWindowInferer(
        roi_size=config.GCM_loader.target_size,
        overlap=0.5,
        sw_device=accelerator.device,
        device=accelerator.device,
    )

    loss_functions = {
        "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
        "bce_loss": nn.BCEWithLogitsLoss().to(accelerator.device),
    }

    metrics = []
    if config.GCM_loader.task_Mu == None:
        task_Mu = 1
    else:
        task_Mu = config.GCM_loader.task_Mu
    for i in range(task_Mu):
        metrics.append(
            {
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
        )
    
    # if metrics.__len__() == 1:
    #     metrics = metrics[0]

    post_trans = monai.transforms.Compose(
        [
            monai.transforms.Activations(sigmoid=True),
            monai.transforms.AsDiscrete(threshold=0.5),
        ]
    )

    optimizer = optim_factory.create_optimizer_v2(
        model,
        opt=config.trainer.optimizer,
        weight_decay=float(config.trainer.weight_decay),
        lr=float(config.trainer.lr),
        betas=(config.trainer.betas[0], config.trainer.betas[1]),
    )

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.trainer.warmup,
        max_epochs=config.trainer.num_epochs,
    )

    # start training
    accelerator.print("Start Training! ")
    train_step = 0
    best_eopch = -1
    val_step = 0
    best_accuracy = torch.tensor(0)
    best_metrics = {}
    best_test_accuracy = torch.tensor(0)
    best_test_metrics = {}

    starting_epoch = 0

    if config.trainer.resume:
        (
            model,
            optimizer,
            scheduler,
            starting_epoch,
            train_step,
            best_accuracy,
            best_test_accuracy,
            best_metrics,
            best_test_metrics,
        ) = utils.resume_train_state(
            model,
            "{}".format(checkpoint_name),
            optimizer,
            scheduler,
            train_loader,
            accelerator,
            seg=False,
        )
        val_step = train_step

    model, optimizer, scheduler, train_loader, val_loader, test_loader = (
        accelerator.prepare(
            model, optimizer, scheduler, train_loader, val_loader, test_loader
        )
    )

    best_accuracy = torch.Tensor([best_accuracy]).to(accelerator.device)
    best_test_accuracy = torch.Tensor([best_test_accuracy]).to(accelerator.device)

    for epoch in range(starting_epoch, config.trainer.num_epochs):
        train_metric, train_step = train_one_epoch(
            model,
            loss_functions,
            train_loader,
            optimizer,
            scheduler,
            metrics,
            post_trans,
            accelerator,
            epoch,
            train_step,
        )

        final_metrics, val_step = val_one_epoch(
            model, inference, val_loader, metrics, val_step, post_trans, accelerator
        )


        if config.GCM_loader.task_Mu == None:
            val_top = final_metrics["Val/Task 0/accuracy"]
            train_top = train_metric["Train/Task 0/accuracy"]
            val_metrics = final_metrics
        else:
            val_top = 0
            train_top = 0
            for i in range(config.GCM_loader.task_Mu):
                val_top += final_metrics["Val/Task {}/accuracy".format(i)]  
                train_top += train_metric["Train/Task {}/accuracy".format(i)]
            val_top = val_top / config.GCM_loader.task_Mu
            train_top = train_top / config.GCM_loader.task_Mu
            val_metrics = final_metrics
        
        # 保存模型
        if val_top > best_accuracy:
            accelerator.save_state(
                output_dir=f"{os.getcwd()}/model_store/{checkpoint_name}/best"
            )
            best_accuracy = val_top
            best_metrics = final_metrics
            # 记录最优test acc
            if config.GCM_loader.fusion == False:
                final_metrics, _ = val_one_epoch(
                    model,
                    inference,
                    test_loader,
                    metrics,
                    -1,
                    post_trans,
                    accelerator,
                    test=True,
                )
                if config.GCM_loader.task_Mu == None:
                    test_top = final_metrics["Test/Task 0/accuracy"]
                else:
                    test_top = 0
                    for i in range(config.GCM_loader.task_Mu):
                        test_top += final_metrics["Test/Task {}/accuracy".format(i)]  
                    test_top = test_top / config.GCM_loader.task_Mu
                best_test_accuracy = test_top
                best_test_metrics = final_metrics
            else:
                final_metrics = final_metrics

                best_test_accuracy = val_top
                best_test_metrics = final_metrics

        accelerator.print(
            f'Epoch [{epoch+1}/{config.trainer.num_epochs}] now train acc: {train_top}, now val acc: {val_top}, best acc: {best_accuracy}, best test acc: {best_test_accuracy}'
        )
        
        
        if config.GCM_loader.task_Mu == None:
            accelerator.print(
                f'Epoch [{epoch+1}/{config.trainer.num_epochs}] Train acc: {train_metric["Train/Task 0/accuracy"]}, Val acc: {val_metrics["Val/Task 0/accuracy"]}, Test acc: {final_metrics["Test/Task 0/accuracy"]}.'
            )
        else:
            for i in range(config.GCM_loader.task_Mu):
                accelerator.print(
                    f'Epoch [{epoch+1}/{config.trainer.num_epochs}] Train acc: {train_metric["Train/Task {}/accuracy".format(i)]}, Val acc: {val_metrics["Val/Task {}/accuracy".format(i)]}.'
                )

        accelerator.print("Cheakpoint...")
        accelerator.save_state(
            output_dir=f"{os.getcwd()}/model_store/{checkpoint_name}/checkpoint"
        )
        torch.save(
            {
                "epoch": epoch,
                "best_accuracy": best_accuracy,
                "best_metrics": best_metrics,
                "best_test_accuracy": best_test_accuracy,
                "best_test_metrics": best_test_metrics,
            },
            f"{os.getcwd()}/model_store/{checkpoint_name}/checkpoint/epoch.pth.tar",
        )

    accelerator.print(f"best test accuracy: {best_test_accuracy}")
    accelerator.print(f"best test metrics: {best_test_metrics}")
    accelerator.print(f"best val accuracy: {best_accuracy}")
    accelerator.print(f"best val metrics: {best_metrics}")
