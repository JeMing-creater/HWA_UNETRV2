import os
import sys
from datetime import datetime
from typing import Dict

import monai
import pytz
import torch
import yaml
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.loader import get_dataloader_BraTS as get_dataloader
from model.HWAUNETR_seg import HWAUNETR
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger, load_pretrain_model, resume_train_state

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致


best_acc = 0
best_class = []

total_grad_out = []
total_grad_in = []


def hook_fn_backward(module, grad_input, grad_output):
    print(module)  # 为了区分模块
    # 为了符合反向传播的顺序，我们先打印 grad_output
    print("grad_output", grad_output)
    # 再打印 grad_input
    print("grad_input", grad_input)
    # 保存到全局变量
    total_grad_in.append(grad_input)
    total_grad_out.append(grad_output)


def train(
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
    # accelerator.print(f'Start Warn Up!')
    for i, image_batch in enumerate(train_loader):
        # output
        logits = model(image_batch["image"])
        total_loss = 0
        for name in loss_functions:
            alpth = 1
            loss = loss_functions[name](logits, image_batch["label"])
            accelerator.log({"Train/" + name: float(loss)}, step=step)
            total_loss += alpth * loss
        # val_outputs = [post_trans(i) for i in logits]
        val_outputs = post_trans(logits)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch["label"])
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
        accelerator.print(
            f"Epoch [{epoch+1}/{config.trainer.num_epochs}][{i + 1}/{len(train_loader)}] Training Loss:{total_loss}",
            flush=True,
        )
        step += 1
    scheduler.step(epoch)
    metric = {}
    for metric_name in metrics:
        if metric_name == "hd95_metric":
            batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
        else:
            batch_acc = metrics[metric_name].aggregate().to(accelerator.device)
        # print(f'b : {batch_acc.device}')
        # print(f'ad : {accelerator.device}')
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update(
            {
                f"Train/mean {metric_name}": float(batch_acc.mean()),
                f"Train/TC {metric_name}": float(batch_acc[0]),
                f"Train/WT {metric_name}": float(batch_acc[1]),
                f"Train/ET {metric_name}": float(batch_acc[2]),
            }
        )
    # accelerator.print(f'Warn Up Over!')
    accelerator.log(metric, step=epoch)
    return step


@torch.no_grad()
def val_one_epoch(
    model: torch.nn.Module,
    inference: monai.inferers.Inferer,
    val_loader: torch.utils.data.DataLoader,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    step: int,
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch=0,
):
    # 验证
    model.eval()
    dice_acc = 0
    dice_class = []
    hd95_acc = 0
    hd95_class = []
    for i, image_batch in enumerate(val_loader):
        logits = inference(image_batch["image"], model)
        val_outputs = post_trans(logits)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch["label"])
        step += 1
    metric = {}
    for metric_name in metrics:
        if metric_name == "hd95_metric":
            batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
        else:
            batch_acc = metrics[metric_name].aggregate().to(accelerator.device)
        if accelerator.num_processes > 1:
            batch_acc = (
                accelerator.reduce(batch_acc.to(accelerator.device))
                / accelerator.num_processes
            )
        metrics[metric_name].reset()
        if metric_name == "dice_metric":
            metric.update(
                {
                    f"Val/mean {metric_name}": float(batch_acc.mean()),
                    f"Val/TC {metric_name}": float(batch_acc[0]),
                    f"Val/WT {metric_name}": float(batch_acc[1]),
                    f"Val/ET {metric_name}": float(batch_acc[2]),
                }
            )
            dice_acc = torch.Tensor([metric["Val/mean dice_metric"]]).to(
                accelerator.device
            )
            dice_class = batch_acc
            accelerator.log(metric, step=epoch)
        else:
            metric.update(
                {
                    f"Val/mean {metric_name}": float(batch_acc.mean()),
                    f"Val/TC {metric_name}": float(batch_acc[0]),
                    f"Val/WT {metric_name}": float(batch_acc[1]),
                    f"Val/ET {metric_name}": float(batch_acc[2]),
                }
            )
            hd95_acc = torch.Tensor([metric["Val/mean hd95_metric"]]).to(
                accelerator.device
            )
            hd95_class = batch_acc
            accelerator.log(metric, step=epoch)
    return dice_acc, dice_class, hd95_acc, hd95_class


if __name__ == "__main__":
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    utils.same_seeds(50)
    logging_dir = (
        os.getcwd()
        + "/logs/"
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
    model = HWAUNETR(
        in_chans=4,
        out_chans=3,
        fussion=[1, 2, 4, 8],
        kernel_sizes=[4, 2, 2, 2],
        depths=[1, 1, 1, 1],
        dims=[48, 96, 192, 384],
        heads=[1, 2, 4, 4],
        hidden_size=768,
        num_slices_list=[64, 32, 16, 8],
        out_indices=[0, 1, 2, 3],
    )

    image_size = config.BraTS_loader.image_size

    accelerator.print("load dataset...")
    train_loader, val_loader = get_dataloader(config)

    inference = monai.inferers.SlidingWindowInferer(
        roi_size=ensure_tuple_rep(image_size, 3),
        overlap=0.5,
        sw_device=accelerator.device,
        device=accelerator.device,
    )
    loss_functions = {
        "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
        # 'generalized_dice_loss': monai.losses.GeneralizedDiceLoss(smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False,
        #                                                           sigmoid=True),
        "dice_loss": monai.losses.DiceLoss(
            smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True
        ),
    }
    metrics = {
        "dice_metric": monai.metrics.DiceMetric(
            include_background=True,
            reduction=monai.utils.MetricReduction.MEAN_BATCH,
            get_not_nans=False,
        ),
        # 'hd95_metric': monai.metrics.HausdorffDistanceMetric(percentile=95, include_background=True,
        #                                                      reduction=monai.utils.MetricReduction.MEAN_BATCH,
        #                                                      get_not_nans=True)
    }
    post_trans = monai.transforms.Compose(
        [
            monai.transforms.Activations(sigmoid=True),
            monai.transforms.AsDiscrete(threshold=0.5),
        ]
    )

    optimizer = optim_factory.create_optimizer_v2(
        model,
        opt=config.trainer.optimizer,
        weight_decay=config.trainer.weight_decay,
        lr=config.trainer.lr,
        betas=(0.9, 0.95),
    )
    # optimizer = torch.optim.adamw(model, weight_decay=config.trainer.weight_decay,
    #                                               lr=config.trainer.lr, betas=(0.9, 0.95))
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.trainer.warmup,
        max_epochs=config.trainer.num_epochs,
    )

    # # 加载预训练模型
    model = load_pretrain_model(
        f"{os.getcwd()}/model_store/{config.finetune.BraTS.checkpoint}/best/new/pytorch_model.bin",
        model,
        accelerator,
    )

    # # 开始验证
    accelerator.print("Start Training！")
    step = 0
    starting_epoch = 0
    best_eopch = -1
    val_step = 0
    best_acc = torch.tensor(0)
    best_class = []
    best_hd95_acc = torch.tensor(0)
    best_hd95_class = []
    if config.trainer.resume:
        (
            model,
            optimizer,
            scheduler,
            start_num_epochs,
            step,
            val_step,
            best_score,
            best_metrics,
        ) = resume_train_state(
            model, config.finetune.BraTS.checkpoint, optimizer, scheduler, accelerator
        )

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )

    best_acc = best_acc.to(accelerator.device)
    best_hd95_acc = best_hd95_acc.to(accelerator.device)

    for epoch in range(starting_epoch, config.trainer.num_epochs):
        step = train(
            model,
            loss_functions,
            train_loader,
            optimizer,
            scheduler,
            metrics,
            post_trans,
            accelerator,
            epoch,
            step,
        )
        if (
            ((epoch + 1) % config.trainer.val_epochs) == 0
            or best_acc >= 0.85
            or (config.trainer.num_epochs - epoch - 1) <= 100
        ):
            dice_acc, dice_class, hd95_acc, hd95_class = val_one_epoch(
                model,
                inference,
                val_loader,
                metrics,
                val_step,
                post_trans,
                accelerator,
                epoch,
            )
            if dice_acc > best_acc:
                best_acc = dice_acc
                best_class = dice_class
                best_hd95_acc = hd95_acc
                best_hd95_class = hd95_class
                accelerator.save_state(
                    output_dir=f"{os.getcwd()}/model_store/{config.finetune.BraTS.checkpoint}/best/new/"
                )
                torch.save(
                    model.state_dict(),
                    f"{os.getcwd()}/model_store/{config.finetune.BraTS.checkpoint}/best/new/model.pth",
                )
            print(
                f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] dice acc: {dice_acc} best acc: {best_acc}"
            )
            print(
                f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] best acc: {best_acc}, best_class: {best_class}"
            )
            if best_hd95_acc != 0:
                print(
                    f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] best best_hd95_acc: {best_hd95_acc}, best_hd95_class: {hd95_class}"
                )
        else:
            print(
                f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] best acc: {best_acc}, best_class: {best_class}"
            )
            if best_hd95_acc != 0:
                print(
                    f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] best best_hd95_acc: {best_hd95_acc}, best_hd95_class: {hd95_class}"
                )

        accelerator.print("Checkout....")
        accelerator.save_state(
            output_dir=f"{os.getcwd()}/model_store/{config.finetune.BraTS.checkpoint}/checkpoint"
        )
        torch.save(
            {"epoch": epoch, "best_acc": best_acc, "best_class": best_class},
            f"{os.getcwd()}/model_store/{config.finetune.BraTS.checkpoint}/checkpoint/epoch.pth.tar",
        )
        accelerator.print("Checkout Over!")

    # accelerator.print(f"最高acc: {metric_saver.best_acc}")
    accelerator.print(f"dice acc: {best_acc}")
    accelerator.print(f"dice class : {best_class}")
    accelerator.print(f"hd95 acc: {best_hd95_acc}")
    accelerator.print(f"hd95 class : {best_hd95_class}")
    sys.exit(1)
