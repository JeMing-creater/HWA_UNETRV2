from math import e
import os

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
from src.mnist_loader import get_dataloader_mnist as get_dataloader
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger, write_example, resume_train_state, split_metrics,reload_pre_train_model,freeze_seg_decoder

from get_model import get_model
from accelerate import Accelerator, DistributedDataParallelKwargs

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
    model.train()
    accelerator.print(f"Training Epoch {epoch}...", flush=True)
    
    # 进度条只在主进程显示
    loop = tqdm(enumerate(train_loader), total=len(train_loader), disable=not accelerator.is_local_main_process)
    
    for i, image_batch in loop:
        images = image_batch["image"]       # [B, 1, D, H, W]
        labels = image_batch["class_label"] # [B, 1]
        
        # 1. 前向传播
        if config.trainer.choose_model == "HWAUNETR" or config.trainer.choose_model == "HSL_Net":
            logits, _ = model(images)
        else:
            logits = model(images) # [B, 1]
            
        total_loss = 0
        
        # 2. 计算 Loss
        for name in loss_functions:
            # BCEWithLogitsLoss 需要 float 类型的 target
            loss = loss_functions[name](logits, labels.float())
            accelerator.log({"Train/" + name: float(loss)}, step=step)
            total_loss += loss

        # 3. 计算指标 (Metrics)
        # 将 logits 转为离散的 0/1 预测值用于计算 Acc/F1
        val_outputs_list = post_trans(logits)
        
        for metric_name in metrics:
            # ConfusionMatrixMetric 需要 list of tensor
            metrics[metric_name](y_pred=val_outputs_list, y=labels)

        # 4. 反向传播与优化
        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        
        # 5. 日志记录
        accelerator.log({"Train/Total Loss": float(total_loss)}, step=step)
        
        loop.set_description(f"Epoch [{epoch+1}/{config.trainer.num_epochs}]")
        loop.set_postfix(loss=total_loss.item())
        step += 1
        
    # 这里的 scheduler 是按 epoch 更新的
    scheduler.step(epoch)

    # 6. 聚合本 Epoch 的所有指标
    metric_results = {}
    for metric_name in metrics:
        
        batch_acc = metrics[metric_name].aggregate()
        
        if isinstance(batch_acc, list):
            batch_acc = batch_acc[0] # 取主要指标

        if not isinstance(batch_acc, torch.Tensor):
            batch_acc = torch.tensor(batch_acc)
            
        batch_acc = batch_acc.to(accelerator.device)

        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes

        metrics[metric_name].reset() # 重置指标，为下一轮做准备
        metric_results.update({f"Train/{metric_name}": float(batch_acc.mean())})

    accelerator.log(metric_results, step=epoch)
    
    return metric_results, step


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
    model.eval()
    flag = "Test" if test else "Val"
    accelerator.print(f"{flag}ing...", flush=True)
    
    loop = tqdm(enumerate(val_loader), total=len(val_loader), disable=not accelerator.is_local_main_process)
    
    for i, image_batch in loop:
        images = image_batch["image"]
        labels = image_batch["class_label"]

        if config.trainer.choose_model == "HWAUNETR" or config.trainer.choose_model == "HSL_Net":
            logits, _ = model(images)
        else:
            logits = model(
                images
            )
        
        # 处理某些模型返回 tuple 的情况
        if isinstance(logits, tuple):
            logits = logits[0]

        total_loss = 0
        log_str = ""

        # 计算 Loss
        for name in loss_functions:
            loss = loss_functions[name](logits, labels.float())
            accelerator.log({f"{flag}/" + name: float(loss)}, step=step)
            log_str += f"{name} {float(loss):.4f} "
            total_loss += loss

        # 计算指标
        val_outputs_list = post_trans(logits)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs_list, y=labels)

        accelerator.log({f"{flag}/Total Loss": float(total_loss)}, step=step)
        
        loop.set_description(f"Epoch [{epoch+1}/{config.trainer.num_epochs}]")
        loop.set_postfix(loss=total_loss.item())
        
        if not test: # Test 阶段通常不增加全局 step
            step += 1

    # 聚合指标
    metric_results = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()
        if isinstance(batch_acc, list):
            batch_acc = batch_acc[0]

        if not isinstance(batch_acc, torch.Tensor):
            batch_acc = torch.tensor(batch_acc)
        
        batch_acc = batch_acc.to(accelerator.device)
        
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes

        metrics[metric_name].reset()
        metric_results.update({f"{flag}/{metric_name}": float(batch_acc.mean())})

    accelerator.log(metric_results, step=epoch)
    return metric_results, step


if __name__ == '__main__':
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    utils.same_seeds(50)

    if config.finetune.MNIST.checkpoint !='None':
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

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)   

    accelerator = Accelerator(
        cpu=False, log_with=["tensorboard"], project_dir=logging_dir,kwargs_handlers=[ddp_kwargs]
    )
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    accelerator.print("load model...")

    model = get_model(config)

    accelerator.print("load dataset...")

    train_loader,val_loader,test_loader,example = get_dataloader(
        root=config.MNIST_loader.root,
        size=config.MNIST_loader.target_size[0],  # [64, 64, 64]
        batch_size=config.trainer.batch_size,
        num_workers=config.trainer.num_workers,
        download=True
    )

    if accelerator.is_main_process == True:
        write_example(config,example)

    inference = monai.inferers.SlidingWindowInferer(
        roi_size=config.GCM_loader.target_size,
        overlap=0.5,
        sw_device=accelerator.device,
        device=accelerator.device,
    )

    loss_functions = {
        "focal_loss": monai.losses.FocalLoss(
            to_onehot_y=False, 
            alpha=0.85,
            gamma=3.0),
    }

    # loss_functions = {
    #     "focal_loss": monai.losses.FocalLoss(
    #         to_onehot_y=False),
    #     "bce_loss": nn.BCEWithLogitsLoss().to(accelerator.device),
    # }

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
        "auc": monai.metrics.ROCAUCMetric()
    }

    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True),
        monai.transforms.AsDiscrete(threshold=0.5), # 0.5
    ])

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

    accelerator.print("Start Training!")

    # 这里的变量用于断点续训 (Resume)
    train_step = 0
    val_step = 0
    starting_epoch = 0
    best_accuracy = torch.tensor(0.0).to(accelerator.device)
    best_metrics = {}
    best_test_accuracy = torch.tensor(0.0).to(accelerator.device)
    best_test_metrics = {}

    if config.trainer.resume:
         model, optimizer, scheduler, starting_epoch, train_step, \
         best_accuracy, best_test_accuracy, best_metrics, best_test_metrics = \
         utils.resume_train_state(
            model, checkpoint_name, optimizer, scheduler, train_loader, accelerator, seg=False
         )
         val_step = train_step

    # Accelerate 准备
    model, optimizer, scheduler, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader, test_loader
    )

    for epoch in range(starting_epoch, config.trainer.num_epochs):
        # --- Train ---
        train_metric, train_step = train_one_epoch(
            model, loss_functions, train_loader, optimizer, scheduler, 
            metrics, post_trans, accelerator, epoch, train_step
        )

        # --- Val ---
        final_metrics, val_step = val_one_epoch(
            model, inference, val_loader, metrics, val_step, post_trans, accelerator
        )
        
        # 1. 获取核心指标
        val_top = final_metrics["Val/accuracy"]
        train_top = train_metric["Train/accuracy"]
        val_metrics = final_metrics
        
        # 2. 保存最佳模型 & 测试 Test 集
        if val_top > best_accuracy:
            accelerator.save_state(
                output_dir=f"{os.getcwd()}/model_store/{checkpoint_name}/best"
            )
            best_accuracy = val_top
            best_metrics = final_metrics
            
            # 只有在 Val 变好时才跑 Test
            test_loop_metrics, _ = val_one_epoch(
                model, inference, test_loader, metrics, -1, post_trans, accelerator, test=True
            )
            
            best_test_accuracy = test_loop_metrics["Test/accuracy"]
            best_test_metrics = test_loop_metrics

        # 3. 打印一行核心信息
        accelerator.print(
            f'Epoch [{epoch+1}/{config.trainer.num_epochs}] now train acc: {train_top:.4f}, now val acc: {val_top:.4f}, best acc: {best_accuracy:.4f}, best test acc: {best_test_accuracy:.4f}'
        )
        
        # 4. 打印详细信息
        accelerator.print(
            f'Epoch [{epoch+1}/{config.trainer.num_epochs}] Train acc: {train_top:.4f}, Val acc: {val_top:.4f}.'
        )

        # 5. 保存 Checkpoint
        accelerator.print("Checkpoint...")
        accelerator.save_state(
            output_dir=f"{os.getcwd()}/model_store/{checkpoint_name}/checkpoint"
        )
        if accelerator.is_main_process:
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

    accelerator.print("================ Training Finished ================")
    accelerator.print(f"best test accuracy: {best_test_accuracy}")
    accelerator.print(f"best test metrics: {best_test_metrics}")
    accelerator.print(f"best val accuracy: {best_accuracy}")
    accelerator.print(f"best val metrics: {best_metrics}")