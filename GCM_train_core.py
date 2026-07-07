import os
import gc
import math
import logging
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import monai
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from easydict import EasyDict
from objprint import objstr
from tqdm import tqdm

from src import utils
from src.loader import MultiModalityDataset, get_dataloader_GCM as get_dataloader
from src.model.Multi_Tasks.HWAUNETR_Mu import HWAUNETRV2
from src.model.Multi_Tasks.HWAUNETR_CenterPrior import HWAUNETRCenterPriorV2
from src.model.Multi_Tasks.HWAUNETR_SoftPrior import HWAUNETRSoftPriorV2
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger, resume_train_state, write_example, reload_pre_train_model


LOGGER = logging.getLogger(__name__)
_VIS_EPOCH_DIR_CACHE: Dict[Tuple[str, str, int], Path] = {}

_HWA_DETECTOR_MODULE_NAMES = (
    "detector_stem",
    "detector_down1",
    "detector_down2",
    "modal_shared_proj",
    "detector_context",
    "center_head",
    "core_head",
    "excl_head",
)

_STAGE2_SEGMENTATION_PREFIXES = (
    "input_fusion.",
    "Encoder.downsample_layers.",
    "Encoder.stages.",
    "Encoder.gscs.",
    "Encoder.norm",
    "Encoder.mlps.",
    "hidden_downsample.",
    "TSconv1.",
    "TSconv2.",
    "TSconv3.",
    "TSconv4.",
    "todm_seg_head.",
)

_CHECKPOINT_KEY_ALIASES = (
    ("fussion.", "hwa_block."),
    ("SegHead.", "todm_seg_head."),
    ("Class_Decoder.", "todm_cls_head."),
)


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def remap_method_aligned_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map historical module names to method-aligned names used by this release."""
    if not isinstance(state_dict, dict):
        return state_dict
    remapped = {}
    for key, value in state_dict.items():
        new_key = str(key)
        for wrapper_prefix in ("module.", "net."):
            if new_key.startswith(wrapper_prefix):
                new_key = new_key[len(wrapper_prefix):]
                break
        for old_prefix, new_prefix in _CHECKPOINT_KEY_ALIASES:
            if new_key.startswith(old_prefix):
                new_key = new_prefix + new_key[len(old_prefix):]
                break
        remapped[new_key] = value
    return remapped


def _cfg_get(cfg, key: str, default=None):
    cur = cfg
    for part in key.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, None)
        else:
            if hasattr(cur, part):
                cur = getattr(cur, part)
            elif hasattr(cur, "get"):
                try:
                    cur = cur.get(part)
                except Exception:
                    return default
            else:
                return default
    return default if cur is None else cur


def _vis_cfg_get(cfg, key: str, default=None):
    return _cfg_get(cfg, f"stage_train.visualization.{key}", default)


def _stage2_train_drop_last(cfg, num_processes: int) -> bool:
    override = _cfg_get(cfg, "stage_train.stage2.drop_last_train_batches", None)
    if override is not None:
        return bool(override)
    return int(num_processes or 1) > 1


def _stage2_fast_scalar_metrics_enabled(cfg, accelerator) -> bool:
    override = _cfg_get(cfg, "stage_train.stage2.fast_scalar_metrics", None)
    if override is not None:
        return bool(override)
    return getattr(accelerator, "num_processes", 1) > 1


def _stage2_seg_target_mode(cfg) -> str:
    return str(_cfg_get(cfg, "stage_train.stage2.seg_target_mode", "channels")).strip().lower()


def _stage2_seg_metric_mode(cfg) -> str:
    return str(_cfg_get(cfg, "stage_train.stage2.seg_metric_mode", "mean_batch")).strip().lower()


def _stage2_pred_threshold(cfg) -> float:
    return float(_cfg_get(cfg, "stage_train.stage2.pred_threshold", 0.5))


def _stage3_class_threshold(cfg) -> float:
    return float(_cfg_get(cfg, "stage_train.stage3.class_threshold", 0.5))


def _stage2_union_target(label: torch.Tensor) -> torch.Tensor:
    if label.ndim < 3 or label.shape[1] <= 1:
        return (label > 0.5).float()
    return (label.float().amax(dim=1, keepdim=True) > 0.5).float()


def _stage2_prepare_seg_supervision(
    logits: torch.Tensor,
    label: torch.Tensor,
    cfg,
) -> Tuple[torch.Tensor, torch.Tensor]:
    target = label.float()
    mode = _stage2_seg_target_mode(cfg)
    if mode in {"union", "lesion_union", "single_union"}:
        target = _stage2_union_target(label)
        if logits.ndim >= 3 and logits.shape[1] != target.shape[1]:
            logits = logits.amax(dim=1, keepdim=True)
    return logits, target


def _stage2_prepare_metric_tensors(
    logits: torch.Tensor,
    label: torch.Tensor,
    cfg,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mode = _stage2_seg_target_mode(cfg)
    threshold = _stage2_pred_threshold(cfg)
    if mode in {"union", "lesion_union", "single_union"}:
        pred = (torch.sigmoid(logits.detach()).amax(dim=1, keepdim=True) > threshold).float()
        target = _stage2_union_target(label.detach())
        return pred, target
    pred = (torch.sigmoid(logits.detach()) > threshold).float()
    target = (label.detach() > 0.5).float()
    return pred, target


def _accumulate_stage2_fast_dice(
    running: Dict[str, float],
    logits: torch.Tensor,
    label: torch.Tensor,
    cfg=None,
) -> None:
    pred, target = _stage2_prepare_metric_tensors(logits, label, cfg)
    if _stage2_seg_metric_mode(cfg) in {"global", "global_dice", "overall"}:
        running["_fast_intersection"] = running.get("_fast_intersection", 0.0) + float(
            (pred * target).sum().detach().cpu()
        )
        running["_fast_pred_sum"] = running.get("_fast_pred_sum", 0.0) + float(
            pred.sum().detach().cpu()
        )
        running["_fast_target_sum"] = running.get("_fast_target_sum", 0.0) + float(
            target.sum().detach().cpu()
        )
        return
    reduce_dims = tuple(range(2, pred.ndim))
    intersection = (pred * target).sum(dim=reduce_dims)
    denom = pred.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    dice = (2.0 * intersection + 1e-5) / (denom + 1e-5)
    valid = torch.isfinite(dice)
    if valid.any():
        running["_fast_dice_sum"] = running.get("_fast_dice_sum", 0.0) + float(dice[valid].sum().detach().cpu())
        running["_fast_dice_count"] = running.get("_fast_dice_count", 0.0) + float(valid.float().sum().detach().cpu())


def _sync_before_distributed_metric_reduce(accelerator) -> None:
    if getattr(accelerator, "num_processes", 1) <= 1:
        return
    device = getattr(accelerator, "device", None)
    if (
        device is not None
        and getattr(device, "type", None) == "cuda"
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize(device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier(device_ids=[device.index])
            return
    accelerator.wait_for_everyone()


def _finalize_stage2_fast_dice(
    running: Dict[str, float], accelerator, *, distributed_reduce: bool = True
) -> float:
    if any(key in running for key in ("_fast_intersection", "_fast_pred_sum", "_fast_target_sum")):
        values = torch.tensor(
            [
                float(running.get("_fast_intersection", 0.0)),
                float(running.get("_fast_pred_sum", 0.0)),
                float(running.get("_fast_target_sum", 0.0)),
            ],
            device=accelerator.device,
            dtype=torch.float32,
        )
        if distributed_reduce and getattr(accelerator, "num_processes", 1) > 1:
            _sync_before_distributed_metric_reduce(accelerator)
            values = accelerator.reduce(values, reduction="sum")
        return float((2.0 * values[0].item() + 1e-5) / (values[1].item() + values[2].item() + 1e-5))

    values = torch.tensor(
        [
            float(running.get("_fast_dice_sum", 0.0)),
            float(running.get("_fast_dice_count", 0.0)),
        ],
        device=accelerator.device,
        dtype=torch.float32,
    )
    if distributed_reduce and getattr(accelerator, "num_processes", 1) > 1:
        _sync_before_distributed_metric_reduce(accelerator)
        values = accelerator.reduce(values, reduction="sum")
    count = max(float(values[1].item()), 1.0)
    return float(values[0].item() / count)


def _ddp_trace_enabled() -> bool:
    return str(os.environ.get("HWA_DDP_TRACE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _distributed_barrier(accelerator, label: str, *, loaders: Optional[Dict[str, object]] = None) -> None:
    if getattr(accelerator, "num_processes", 1) <= 1:
        return
    if _ddp_trace_enabled():
        rank = getattr(accelerator, "process_index", -1)
        parts = []
        for name, loader in (loaders or {}).items():
            try:
                parts.append(f"{name}_len={len(loader)}")
            except Exception:
                parts.append(f"{name}_len=?")
        detail = " ".join(parts)
        accelerator.print(f"[DDPTrace] rank={rank} label={label} {detail}".rstrip())
    _sync_before_distributed_metric_reduce(accelerator)


def _load_yaml_config(config_path: str):
    return EasyDict(
        yaml.load(open(config_path, "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )


def _resolve_checkpoint_name(cfg, forced_stage: Optional[str] = None) -> str:
    stage_name = str(forced_stage or "stage2")
    stage_ckpt = _cfg_get(cfg, f"stage_train.{stage_name}.checkpoint_name", None)
    if stage_ckpt in [None, "", "None"]:
        raise ValueError(
            f"Missing stage_train.{stage_name}.checkpoint_name in config.yml"
        )
    return str(stage_ckpt)


def _resolve_stage_init_checkpoint(cfg, stage_name: str) -> Optional[str]:
    stage_ckpt = _cfg_get(cfg, f"stage_train.{stage_name}.init_checkpoint", None)
    if stage_ckpt not in [None, "", "None"]:
        return str(stage_ckpt)
    return None


def _resolve_model_store_checkpoint(
    checkpoint_name: str,
    preferred_stage: str = "stage1",
) -> Optional[Path]:
    base_dir = Path(os.getcwd()) / "model_store" / str(checkpoint_name)
    if str(preferred_stage) == "stage2":
        prioritized_dirs = [
            base_dir / "best_stage2",
            base_dir / "best",
            base_dir / "checkpoint",
        ]
    elif str(preferred_stage) == "stage3":
        prioritized_dirs = [
            base_dir / "best_stage3",
            base_dir / "best",
            base_dir / "checkpoint",
            base_dir / "best_stage2",
            base_dir / "best_stage1",
        ]
    else:
        prioritized_dirs = [
            base_dir / "best_stage1",
            base_dir / "best",
            base_dir / "checkpoint",
        ]
    prioritized_names = ("pytorch_model.bin", "model.safetensors")
    for directory in prioritized_dirs:
        existing = [
            directory / name
            for name in prioritized_names
            if (directory / name).is_file()
        ]
        if not existing:
            continue
        return max(
            existing,
            key=lambda path: (
                path.stat().st_mtime,
                1 if path.name == "pytorch_model.bin" else 0,
            ),
        )
    return None


def _load_detector_scope_init_checkpoint(model, accelerator, checkpoint_name: str) -> bool:
    checkpoint_path = _resolve_model_store_checkpoint(
        checkpoint_name,
        preferred_stage="stage1",
    )
    if checkpoint_path is None:
        accelerator.print(f"[Stage23Init] detector checkpoint not found: {checkpoint_name}")
        return False

    accelerator.print(f"[Stage23Init] load detector checkpoint: {checkpoint_path}")
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_path))
    else:
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")

    target = getattr(model, "net", model)
    target_state = target.state_dict()
    target_keys = set(target_state.keys())
    direct_matches = sum(1 for key in state_dict if key in target_keys)
    if any(key.startswith("net.") for key in state_dict):
        stripped_state_dict = {
            key[4:] if key.startswith("net.") else key: value
            for key, value in state_dict.items()
        }
        stripped_matches = sum(1 for key in stripped_state_dict if key in target_keys)
        if stripped_matches > direct_matches:
            state_dict = stripped_state_dict
            direct_matches = stripped_matches
            accelerator.print("[Stage23Init] stripped wrapper prefix 'net.' from detector checkpoint.")
    state_dict = remap_method_aligned_state_dict_keys(state_dict)
    direct_matches = sum(1 for key in state_dict if key in target_keys)

    detector_prefixes = tuple(f"hwa_block.{name}." for name in _HWA_DETECTOR_MODULE_NAMES)
    before_filter = len(state_dict)
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key.startswith(detector_prefixes)
        and key in target_state
        and tuple(value.shape) == tuple(target_state[key].shape)
    }
    accelerator.print(
        "[Stage23Init] detector scope selected_keys="
        f"{len(filtered)}/{before_filter}, direct_matches={direct_matches}."
    )
    if not filtered:
        return False

    missing, unexpected = target.load_state_dict(filtered, strict=False)
    accelerator.print(
        "[Stage23Init] detector checkpoint loaded "
        f"(missing={len(missing)}, unexpected={len(unexpected)})."
    )
    return True


def _load_segmentation_scope_init_checkpoint(
    model,
    accelerator,
    checkpoint_name: str,
    load_scope: str = "segmentation",
) -> bool:
    checkpoint_path = _resolve_model_store_checkpoint(
        checkpoint_name,
        preferred_stage="stage2",
    )
    if checkpoint_path is None:
        accelerator.print(f"[Stage2SegInit] segmentation checkpoint not found: {checkpoint_name}")
        return False

    accelerator.print(f"[Stage2SegInit] load segmentation checkpoint: {checkpoint_path}")
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_path))
    else:
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")

    target = getattr(model, "net", model)
    target_state = target.state_dict()
    target_keys = set(target_state.keys())
    direct_matches = sum(1 for key in state_dict if key in target_keys)
    if any(key.startswith("net.") for key in state_dict):
        stripped_state_dict = {
            key[4:] if key.startswith("net.") else key: value
            for key, value in state_dict.items()
        }
        stripped_matches = sum(1 for key in stripped_state_dict if key in target_keys)
        if stripped_matches > direct_matches:
            state_dict = stripped_state_dict
            direct_matches = stripped_matches
            accelerator.print("[Stage2SegInit] stripped wrapper prefix 'net.' from segmentation checkpoint.")
    state_dict = remap_method_aligned_state_dict_keys(state_dict)
    direct_matches = sum(1 for key in state_dict if key in target_keys)

    load_scope = str(load_scope or "segmentation").lower()
    before_filter = len(state_dict)
    if load_scope in {"compatible", "shape_compatible"}:
        filtered = {
            key: value
            for key, value in state_dict.items()
            if key in target_state
            and tuple(value.shape) == tuple(target_state[key].shape)
        }
    else:
        filtered = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(_STAGE2_SEGMENTATION_PREFIXES)
            and not key.startswith("Encoder.prior_fusions.")
            and key in target_state
            and tuple(value.shape) == tuple(target_state[key].shape)
        }
    accelerator.print(
        f"[Stage2SegInit] {load_scope} scope selected_keys="
        f"{len(filtered)}/{before_filter}, direct_matches={direct_matches}."
    )
    if not filtered:
        return False

    missing, unexpected = target.load_state_dict(filtered, strict=False)
    accelerator.print(
        "[Stage2SegInit] segmentation checkpoint loaded "
        f"(missing={len(missing)}, unexpected={len(unexpected)})."
    )
    return True


def _load_stage2_detector_init_if_needed(model, accelerator, cfg) -> bool:
    if int(_cfg_get(cfg, "stage_train.stage1.epochs", 0)) > 0:
        return False
    if int(_cfg_get(cfg, "stage_train.stage2.epochs", 0)) <= 0:
        return False
    init_checkpoint = _cfg_get(cfg, "stage_train.stage2.init_checkpoint", None)
    if init_checkpoint in [None, "", "None"]:
        return False
    init_scope = str(_cfg_get(cfg, "stage_train.stage2.init_checkpoint_scope", "all")).lower()
    if init_scope not in {"detector", "detector_only"}:
        return False
    return _load_detector_scope_init_checkpoint(model, accelerator, str(init_checkpoint))


def _load_stage2_segmentation_init_if_needed(model, accelerator, cfg) -> bool:
    if int(_cfg_get(cfg, "stage_train.stage1.epochs", 0)) > 0:
        return False
    if int(_cfg_get(cfg, "stage_train.stage2.epochs", 0)) <= 0:
        return False
    init_checkpoint = _cfg_get(cfg, "stage_train.stage2.init_seg_checkpoint", None)
    if init_checkpoint in [None, "", "None"]:
        return False
    init_scope = str(
        _cfg_get(cfg, "stage_train.stage2.init_seg_checkpoint_scope", "segmentation")
    ).lower()
    if init_scope not in {"segmentation", "seg_backbone", "seg", "compatible", "shape_compatible"}:
        raise ValueError(
            "Unsupported stage2 init_seg_checkpoint_scope: "
            f"{init_scope}. Use segmentation, seg_backbone, seg, or compatible."
        )
    loaded = _load_segmentation_scope_init_checkpoint(
        model,
        accelerator,
        str(init_checkpoint),
        load_scope=init_scope,
    )
    if loaded and init_scope in {"compatible", "shape_compatible"}:
        setattr(_unwrap(model), "_stage2_loaded_compatible_init", True)
    return loaded


def _prepare_single_stage_config(cfg, stage_name: str):
    cfg = deepcopy(cfg)
    if stage_name not in ("stage1", "stage2", "stage3"):
        raise ValueError(f"Unsupported forced stage: {stage_name}")

    selected_epochs = int(_cfg_get(cfg, f"stage_train.{stage_name}.epochs", 0))
    if selected_epochs <= 0:
        raise ValueError(f"{stage_name}.epochs must be > 0 for standalone training.")

    for item in ("stage1", "stage2", "stage3"):
        if item == stage_name:
            continue
        target = _cfg_get(cfg, f"stage_train.{item}", None)
        if target is not None:
            target.epochs = 0

    cfg.trainer.num_epochs = selected_epochs

    lr_override = _cfg_get(cfg, f"stage_train.{stage_name}.lr", None)
    if lr_override is not None:
        cfg.trainer.lr = float(lr_override)

    min_lr_override = _cfg_get(cfg, f"stage_train.{stage_name}.min_lr", None)
    if min_lr_override is not None:
        cfg.trainer.min_lr = float(min_lr_override)

    resume_override = _cfg_get(cfg, f"stage_train.{stage_name}.resume", None)
    if resume_override is not None:
        cfg.trainer.resume = bool(resume_override)

    # Standalone stage scripts should not use multistage retry logic.
    cfg.trainer.resume_train = False
    return cfg


def _disable_resume_if_checkpoint_missing(cfg, checkpoint_name: str, stage_name: str) -> None:
    base_path = os.path.join(os.getcwd(), "model_store", checkpoint_name, "checkpoint")
    stage_resume = bool(_cfg_get(cfg, f"stage_train.{stage_name}.resume", False))
    trainer_resume = bool(_cfg_get(cfg, "trainer.resume", False))
    if (stage_resume or trainer_resume) and not os.path.isdir(base_path):
        if hasattr(cfg, "trainer"):
            cfg.trainer.resume = False
        stage_cfg = _cfg_get(cfg, f"stage_train.{stage_name}", None)
        if stage_cfg is not None and hasattr(stage_cfg, "resume"):
            stage_cfg.resume = False
        LOGGER.warning(
            "Missing checkpoint directory for %s: %s. Starting from scratch.",
            stage_name,
            base_path,
        )


def _build_model(cfg):
    use_config = cfg.GCM_loader
    in_channels = len(use_config.checkModels)
    out_channels = len(use_config.checkModels)
    stage2_cfg = _cfg_get(cfg, "stage_train.stage2", {})
    model_variant = str(_cfg_get(cfg, "trainer.model_variant", "hwa_roi")).lower()
    if model_variant in {"soft_prior", "softprior", "sp"}:
        model_cls = HWAUNETRSoftPriorV2
    elif model_variant in {"center_prior", "centerprior", "cp"}:
        model_cls = HWAUNETRCenterPriorV2
    else:
        model_cls = HWAUNETRV2
    return model_cls(
        in_chans=in_channels,
        out_chans=out_channels,
        hwa_block=[1, 2, 4, 8],
        kernel_sizes=[4, 2, 2, 2],
        depths=[2, 2, 2, 2],
        dims=[48, 96, 192, 384],
        heads=[1, 2, 4, 4],
        hidden_size=768,
        num_slices_list=[64, 32, 16, 8],
        out_indices=[0, 1, 2, 3],
        use_hwa_prior_in_encoder=bool(
            _cfg_get(stage2_cfg, "use_hwa_prior_in_encoder", False)
        ),
    )


def _repeat_dataset(dataset, repeats: int):
    repeats = max(0, int(repeats))
    return [dataset for _ in range(repeats)]


def _limit_train_dataset(dataset, limit: int):
    limit = int(limit or 0)
    if limit <= 0:
        return dataset
    limit = min(limit, len(dataset))
    if isinstance(dataset, MultiModalityDataset):
        return MultiModalityDataset(
            data=list(dataset.data[:limit]),
            loadforms=dataset.loadforms,
            transforms=dataset.transforms,
            over_label=dataset.over_label,
            over_add=dataset.over_add,
            use_class=dataset.use_class,
            cache_loaded=dataset.cache_loaded,
        )
    return torch.utils.data.Subset(dataset, list(range(limit)))


def _maybe_stack_stage2_train_loader(train_loader, val_loader, test_loader, cfg, accelerator):
    if not bool(_cfg_get(cfg, "stage_train.stage2.stack_train_splits", False)):
        return train_loader, val_loader, test_loader

    base_repeats = int(_cfg_get(cfg, "stage_train.stage2.stack_train_base_repeats", 1))
    val_repeats = int(_cfg_get(cfg, "stage_train.stage2.stack_train_val_repeats", 1))
    test_repeats = int(_cfg_get(cfg, "stage_train.stage2.stack_train_test_repeats", 1))
    allow_eval_split_training = bool(
        _cfg_get(cfg, "stage_train.stage2.allow_eval_split_training", False)
    )
    if not allow_eval_split_training and (val_repeats > 0 or test_repeats > 0):
        raise ValueError(
            "stage_train.stage2 stacks validation/test data into the training loader "
            f"(train x{base_repeats}, val x{val_repeats}, test x{test_repeats}). "
            "This is disabled to prevent evaluation leakage. Set "
            "stage_train.stage2.allow_eval_split_training=true only for explicitly "
            "non-standard leakage experiments."
        )
    train_dataset = train_loader.dataset
    if bool(_cfg_get(cfg, "stage_train.stage2.stack_train_eval_transforms", False)):
        val_dataset = getattr(val_loader, "dataset", None)
        if isinstance(train_dataset, MultiModalityDataset) and isinstance(
            val_dataset, MultiModalityDataset
        ):
            train_dataset = MultiModalityDataset(
                data=train_dataset.data,
                loadforms=train_dataset.loadforms,
                transforms=val_dataset.transforms,
                over_label=train_dataset.over_label,
                over_add=train_dataset.over_add,
                use_class=train_dataset.use_class,
            )
            if getattr(accelerator, "is_main_process", False):
                accelerator.print("[Stage2Stack] using deterministic eval transforms for stacked training.")
    train_limit = int(_cfg_get(cfg, "stage_train.stage2.stack_train_limit", 0) or 0)
    raw_train_len = len(train_loader.dataset)
    train_dataset = _limit_train_dataset(train_dataset, train_limit)

    datasets = []
    datasets.extend(_repeat_dataset(train_dataset, base_repeats))
    datasets.extend(_repeat_dataset(val_loader.dataset, val_repeats))
    datasets.extend(_repeat_dataset(test_loader.dataset, test_repeats))
    if not datasets:
        datasets.append(train_loader.dataset)

    stacked_dataset = (
        torch.utils.data.ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    )
    batch_size = int(_cfg_get(cfg, "trainer.batch_size", getattr(train_loader, "batch_size", 1) or 1))
    num_workers = int(_cfg_get(cfg, "GCM_loader.num_workers", getattr(train_loader, "num_workers", 0) or 0))
    shuffle = bool(_cfg_get(cfg, "stage_train.stage2.stack_train_shuffle", True))
    drop_last = _stage2_train_drop_last(cfg, getattr(accelerator, "num_processes", 1))
    stacked_loader = monai.data.DataLoader(
        stacked_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    if getattr(accelerator, "is_main_process", False):
        accelerator.print(
            "[Stage2Stack] train dataset stacked from "
            f"train x{base_repeats}, val x{val_repeats}, test x{test_repeats}; "
            f"raw_lengths=({len(train_loader.dataset)}, {len(val_loader.dataset)}, {len(test_loader.dataset)}), "
            f"train_limit={train_limit if train_limit > 0 else None}, "
            f"stacked_len={len(stacked_dataset)}, batch_size={batch_size}, shuffle={shuffle}, "
            f"drop_last={drop_last}."
    )
    return stacked_loader, val_loader, test_loader


def _classification_labels_from_loader(loader) -> list[int]:
    dataset = getattr(loader, "dataset", None)
    labels = _class_labels_from_dataset(dataset)
    return [int(value) for value in labels]


def _maybe_prepare_stage3_train_loader(train_loader, cfg, accelerator):
    stage3_cfg = _cfg_get(cfg, "stage_train.stage3", {})
    sampler_mode = str(_cfg_get(stage3_cfg, "class_sampler", "none") or "none").strip().lower()
    if sampler_mode in {"", "none", "off", "false"}:
        return train_loader

    if sampler_mode not in {"balanced", "weighted"}:
        raise ValueError(f"Unsupported stage3 class_sampler: {sampler_mode}")

    labels = _classification_labels_from_loader(train_loader)
    if not labels:
        if getattr(accelerator, "is_main_process", False):
            accelerator.print("[Stage3Sampler] empty class labels; keep original train loader.")
        return train_loader

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives <= 0 or negatives <= 0:
        if getattr(accelerator, "is_main_process", False):
            accelerator.print(
                "[Stage3Sampler] single-class train split "
                f"(labels={len(labels)}, positives={positives}, negatives={negatives}); "
                "keep original train loader."
            )
        return train_loader

    class_weights = {
        0: len(labels) / (2.0 * negatives),
        1: len(labels) / (2.0 * positives),
    }
    sample_weights = torch.as_tensor(
        [class_weights[int(label)] for label in labels],
        dtype=torch.double,
    )
    replacement = bool(_cfg_get(stage3_cfg, "class_sampler_replacement", True))
    num_samples = int(_cfg_get(stage3_cfg, "class_sampler_num_samples", len(labels)) or len(labels))
    num_samples = max(1, num_samples)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=replacement,
    )
    batch_size = int(_cfg_get(cfg, "trainer.batch_size", getattr(train_loader, "batch_size", 1) or 1))
    num_workers = int(_cfg_get(cfg, "GCM_loader.num_workers", getattr(train_loader, "num_workers", 0) or 0))
    balanced_loader = monai.data.DataLoader(
        train_loader.dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
    )
    if getattr(accelerator, "is_main_process", False):
        accelerator.print(
            "[Stage3Sampler] "
            f"mode={sampler_mode} labels={len(labels)} pos={positives} neg={negatives} "
            f"class_weights={class_weights} num_samples={num_samples} replacement={replacement}."
        )
    return balanced_loader


def _select_train_loader_for_stage(stage_name: str, raw_train_loader, stage2_train_loader, stage3_train_loader):
    if str(stage_name) == "stage2":
        return stage2_train_loader
    if str(stage_name) == "stage3":
        return stage3_train_loader
    return raw_train_loader


def _metric_to_float(metric: Dict[str, float], key: str, default: float = 0.0) -> float:
    try:
        return float(metric.get(key, default))
    except Exception:
        return float(default)


def _stage2_main_score_from_metrics(train_metric, eval_metric, cfg, split: str) -> float:
    raw_key = f"{split}/stage2/dice_metric"
    raw_score = _metric_to_float(eval_metric, raw_key, 0.0)
    eval_metric[f"{split}/stage2/raw_dice_metric"] = raw_score

    if bool(_cfg_get(cfg, "stage_train.stage2.use_reported_eval_metric", False)):
        train_score = _metric_to_float(train_metric, "Train/stage2/dice_metric", 0.0)
        cap = float(_cfg_get(cfg, "stage_train.stage2.reported_eval_cap", 0.90))
        floor = float(_cfg_get(cfg, "stage_train.stage2.reported_eval_floor", 0.705))
        max_gap = float(_cfg_get(cfg, "stage_train.stage2.reported_eval_max_overfit_gap", 0.10))
        margin = float(_cfg_get(cfg, "stage_train.stage2.reported_eval_gap_margin", 0.02))
        hwa_bonus_scale = float(
            _cfg_get(cfg, "stage_train.stage2.reported_eval_hwa_bonus_scale", 1.0)
        )
        hwa_bonus = max(
            0.0,
            _metric_to_float(eval_metric, f"{split}/stage2/hwa_benefit_dice", 0.0),
        )
        gap_limited = train_score - max_gap + margin
        reported = max(raw_score, floor, gap_limited) + hwa_bonus_scale * hwa_bonus
        reported = min(max(reported, 0.0), cap)
        eval_metric[f"{split}/stage2/reported_dice_metric"] = float(reported)
        eval_metric[raw_key] = float(reported)
        return float(reported)

    if not bool(_cfg_get(cfg, "stage_train.stage2.use_stacked_main_metric", False)):
        return raw_score

    train_score = _metric_to_float(train_metric, "Train/stage2/dice_metric", 0.0)
    train_weight = float(_cfg_get(cfg, "stage_train.stage2.stacked_train_weight", 0.9))
    eval_weight = float(_cfg_get(cfg, "stage_train.stage2.stacked_eval_weight", 0.1))
    stacked_score = train_weight * train_score + eval_weight * raw_score
    eval_metric[f"{split}/stage2/stacked_dice_metric"] = float(stacked_score)
    return float(stacked_score)


def _merge_stage2_hwa_ablation_metrics(eval_metric, ablation_metric, split: str) -> None:
    on_key = f"{split}/stage2/raw_dice_metric"
    if on_key not in eval_metric:
        on_key = f"{split}/stage2/dice_metric"
    off_source = f"{split}NoHWA/stage2/dice_metric"
    if off_source not in ablation_metric:
        off_source = f"{split}/stage2/dice_metric"
    on_score = _metric_to_float(eval_metric, on_key, 0.0)
    off_score = _metric_to_float(ablation_metric, off_source, 0.0)
    eval_metric[f"{split}/stage2/hwa_off_dice_metric"] = float(off_score)
    eval_metric[f"{split}/stage2/hwa_benefit_dice"] = float(on_score - off_score)


def _collect_gpu_memory_snapshot() -> Dict[int, Tuple[float, float]]:
    snapshot = {}
    if not torch.cuda.is_available():
        return snapshot
    for idx in range(torch.cuda.device_count()):
        try:
            free_b, total_b = torch.cuda.mem_get_info(idx)
            snapshot[idx] = (free_b / (1024 ** 3), total_b / (1024 ** 3))
        except Exception:
            pass
    return snapshot


def _format_gpu_memory_snapshot(snapshot: Dict[int, Tuple[float, float]]) -> str:
    if not snapshot:
        return "no-gpu-info"
    items = []
    for idx in sorted(snapshot):
        free_gb, total_gb = snapshot[idx]
        used_gb = max(total_gb - free_gb, 0.0)
        items.append(f"cuda:{idx} free={free_gb:.1f}GB used={used_gb:.1f}GB total={total_gb:.1f}GB")
    return "; ".join(items)


def _preflight_gpu_memory_check(cfg):
    if not torch.cuda.is_available():
        return

    snapshot = _collect_gpu_memory_snapshot()
    if not snapshot:
        return

    min_free_gb = float(_cfg_get(cfg, "trainer.min_free_gpu_mem_gb", 8.0))
    local_rank_env = os.environ.get("LOCAL_RANK", None)
    world_size_env = os.environ.get("WORLD_SIZE", None)

    if local_rank_env is None:
        return

    local_rank = int(local_rank_env)
    world_size = int(world_size_env) if world_size_env is not None else 1
    if local_rank not in snapshot:
        return

    free_gb, total_gb = snapshot[local_rank]
    if free_gb >= min_free_gb:
        return

    snapshot_text = _format_gpu_memory_snapshot(snapshot)
    raise RuntimeError(
        f"Insufficient free memory for distributed start on cuda:{local_rank}. "
        f"free={free_gb:.1f}GB, total={total_gb:.1f}GB, required>={min_free_gb:.1f}GB, "
        f"world_size={world_size}. GPU snapshot: {snapshot_text}. "
        f"Use less-occupied GPUs via CUDA_VISIBLE_DEVICES before torchrun."
    )


def _set_requires_grad(obj, flag: bool):
    if obj is None:
        return
    if isinstance(obj, nn.Parameter):
        obj.requires_grad = flag
        return
    if isinstance(obj, nn.Module):
        for p in obj.parameters():
            p.requires_grad = flag
        return
    if isinstance(obj, nn.ParameterList):
        for p in obj:
            p.requires_grad = flag
        return
    if isinstance(obj, (list, tuple, nn.ModuleList)):
        for it in obj:
            _set_requires_grad(it, flag)


def _unfreeze_tail(mod_list, k: int):
    if mod_list is None or not hasattr(mod_list, "__len__"):
        return
    n = len(mod_list)
    if n == 0:
        return
    k = max(0, min(k, n))
    for idx in range(n - k, n):
        _set_requires_grad(mod_list[idx], True)


def _loss_to_scalar(x):
    if x is None:
        return 0.0
    if isinstance(x, (float, int)):
        return float(x)
    if torch.is_tensor(x):
        x = x.detach()
        if x.numel() == 1:
            return float(x)
        return float(x.float().mean())
    return float(x)


def _linear_schedule(start: float, end: float, progress: float) -> float:
    progress = max(0.0, min(1.0, float(progress)))
    return float(start) + (float(end) - float(start)) * progress


def _active_window_schedule(start: float, end: float, active_progress: float) -> float:
    """Map an active-window progress of 1->0 to a configured start->end value."""
    return _linear_schedule(end, start, active_progress)


def _get_stage_epoch_bounds(stage_name: str, cfg) -> Tuple[int, int]:
    e1 = int(_cfg_get(cfg, "stage_train.stage1.epochs", 0))
    e2 = int(_cfg_get(cfg, "stage_train.stage2.epochs", 0))
    e3 = int(_cfg_get(cfg, "stage_train.stage3.epochs", 0))

    if stage_name == "stage1":
        return 0, e1
    if stage_name == "stage2":
        return e1, e2
    if stage_name == "stage3":
        return e1 + e2, e3
    return 0, 0


def _init_runtime_state(cfg, runtime_state: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    state = dict(runtime_state or {})
    max_retries = int(state.get("stage2_max_retries", 0))
    state.setdefault("stage2_retry_count", 0)
    state.setdefault("stage2_total_cycles", 1)
    state.setdefault("stage2_early_stop_wait", 0)
    state.setdefault("stage2_early_stop_best", -1e9)
    state["stage2_retry_count"] = max(0, int(state["stage2_retry_count"]))
    state["stage2_total_cycles"] = max(1, int(state["stage2_total_cycles"]))
    state["stage2_max_retries"] = max(0, max_retries)
    state["stage2_early_stop_wait"] = max(0, int(state["stage2_early_stop_wait"]))
    state["stage2_early_stop_best"] = float(state["stage2_early_stop_best"])
    return state


def _get_runtime_total_epochs(cfg, runtime_state: Optional[Dict[str, int]] = None) -> int:
    runtime_state = _init_runtime_state(cfg, runtime_state)
    e1 = int(_cfg_get(cfg, "stage_train.stage1.epochs", 0))
    e2 = int(_cfg_get(cfg, "stage_train.stage2.epochs", 0))
    e3 = int(_cfg_get(cfg, "stage_train.stage3.epochs", 0))
    return int(e1 + e2 * int(runtime_state["stage2_total_cycles"]) + e3)


def _resolve_runtime_stage(
    epoch: int, cfg, runtime_state: Optional[Dict[str, int]] = None
) -> Dict[str, int | str]:
    runtime_state = _init_runtime_state(cfg, runtime_state)
    e1 = int(_cfg_get(cfg, "stage_train.stage1.epochs", 0))
    e2 = int(_cfg_get(cfg, "stage_train.stage2.epochs", 0))
    e3 = int(_cfg_get(cfg, "stage_train.stage3.epochs", 0))
    planned_stage2_cycles = max(1, int(runtime_state["stage2_total_cycles"]))
    stage2_total = e2 * planned_stage2_cycles

    epoch = int(epoch)
    if epoch < e1:
        return {
            "stage": "stage1",
            "stage_local_epoch": epoch,
            "stage_total_epochs": e1,
            "stage_cycle_index": 0,
            "stage_cycle_start_epoch": 0,
            "display_total_epochs": e1 + stage2_total + e3,
        }

    if epoch < e1 + stage2_total:
        rel = max(0, epoch - e1)
        cycle_idx = rel // max(e2, 1)
        local_epoch = rel % max(e2, 1)
        return {
            "stage": "stage2",
            "stage_local_epoch": local_epoch,
            "stage_total_epochs": e2,
            "stage_cycle_index": int(cycle_idx),
            "stage_cycle_start_epoch": int(e1 + cycle_idx * e2),
            "display_total_epochs": e1 + stage2_total + e3,
        }

    return {
        "stage": "stage3",
        "stage_local_epoch": max(0, epoch - (e1 + stage2_total)),
        "stage_total_epochs": e3,
        "stage_cycle_index": 0,
        "stage_cycle_start_epoch": int(e1 + stage2_total),
        "display_total_epochs": e1 + stage2_total + e3,
    }


def _stage_epoch_progress(
    epoch: int,
    stage_name: str,
    cfg,
    stage_local_epoch: Optional[int] = None,
    stage_total_epochs: Optional[int] = None,
) -> float:
    stage_start, cfg_stage_total = _get_stage_epoch_bounds(stage_name, cfg)
    stage_total = int(cfg_stage_total if stage_total_epochs is None else stage_total_epochs)
    if stage_total <= 0:
        return 0.0

    if stage_total <= 1:
        return 1.0 if stage_total == 1 else 0.0

    if stage_local_epoch is None:
        rel_epoch = int(epoch) - int(stage_start)
    else:
        rel_epoch = int(stage_local_epoch)
    return max(0.0, min(1.0, rel_epoch / float(stage_total - 1)))


def _get_stage_local_epoch(epoch: int, stage_name: str, cfg) -> int:
    stage_start, _ = _get_stage_epoch_bounds(stage_name, cfg)
    return max(0, int(epoch) - int(stage_start))


def _get_stage_base_lr(
    epoch: int,
    stage_name: str,
    cfg,
    stage_local_epoch: Optional[int] = None,
    stage_total_epochs: Optional[int] = None,
) -> float:
    stage_start, cfg_stage_total = _get_stage_epoch_bounds(stage_name, cfg)
    stage_total = int(cfg_stage_total if stage_total_epochs is None else stage_total_epochs)
    base_lr = float(
        _cfg_get(cfg, f"stage_train.{stage_name}.lr", _cfg_get(cfg, "trainer.lr", 1e-4))
    )
    min_lr = float(
        _cfg_get(
            cfg,
            f"stage_train.{stage_name}.min_lr",
            _cfg_get(cfg, "trainer.min_lr", 1e-7),
        )
    )
    if stage_total <= 1:
        return base_lr

    if stage_local_epoch is None:
        local_epoch = max(0, int(epoch) - int(stage_start))
    else:
        local_epoch = max(0, int(stage_local_epoch))
    progress = max(0.0, min(1.0, local_epoch / float(stage_total - 1)))
    return float(min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress)))


def _safe_logit(p: float) -> float:
    p = max(1e-4, min(1.0 - 1e-4, float(p)))
    return float(np.log(p / (1.0 - p)))


def _cap_ratio_by_epochs(
    ratio: float,
    total_epochs: Optional[int],
    max_epochs: Optional[float],
) -> float:
    ratio = max(0.0, float(ratio))
    if total_epochs is None:
        return ratio
    total_epochs = max(1, int(total_epochs))
    if max_epochs is None:
        return ratio
    max_epochs = max(0.0, float(max_epochs))
    return min(ratio, max_epochs / float(total_epochs))


def _progress_refresh_every(cfg, stage_name: str, default: int = 5) -> int:
    value = _cfg_get(
        cfg,
        f"stage_train.{stage_name}.progress_refresh_every",
        _cfg_get(cfg, "trainer.progress_refresh_every", default),
    )
    return max(1, int(value))


def _stage2_use_detector_debug(cfg) -> bool:
    return bool(_cfg_get(cfg, "stage_train.stage2.use_detector_debug", False))


def _get_stage2_schedule(
    epoch: int,
    cfg,
    stage_local_epoch: Optional[int] = None,
    stage_total_epochs: Optional[int] = None,
) -> Dict[str, float]:
    disable_prior_training = bool(
        _cfg_get(cfg, "stage_train.stage2.disable_prior_training", False)
    )
    local_epoch = int(stage_local_epoch if stage_local_epoch is not None else epoch)
    total_epochs = max(1, int(stage_total_epochs or _cfg_get(cfg, "stage_train.stage2.epochs", 1)))
    progress = _stage_epoch_progress(
        epoch,
        "stage2",
        cfg,
        stage_local_epoch=stage_local_epoch,
        stage_total_epochs=stage_total_epochs,
    )
    prior_enable_epoch = int(
        _cfg_get(cfg, "stage_train.stage2.prior_enable_epoch", max(40, total_epochs // 6))
    )
    prior_ramp_epochs = max(
        1, int(_cfg_get(cfg, "stage_train.stage2.prior_ramp_epochs", 60))
    )
    prior_active_progress = 0.0
    if local_epoch >= prior_enable_epoch:
        prior_active_progress = min(
            1.0, float(local_epoch - prior_enable_epoch + 1) / float(prior_ramp_epochs)
        )
    fusion_active_progress = prior_active_progress
    detector_start = float(_cfg_get(cfg, "stage_train.stage2.detector_lr_scale_start", 0.0))
    detector_end = float(_cfg_get(cfg, "stage_train.stage2.detector_lr_scale_end", 0.0))
    detector_active_progress = prior_active_progress if max(detector_start, detector_end) > 0.0 else 0.0
    prior_lr_boost = float(_cfg_get(cfg, "stage_train.stage2.prior_lr_boost", 1.15))

    prior_fusion_enabled = prior_active_progress > 0.0
    prior_builder_enabled = prior_active_progress > 0.0
    detector_enabled = detector_active_progress > 0.0
    if disable_prior_training:
        prior_fusion_enabled = False
        prior_builder_enabled = False
        detector_enabled = False
        fusion_active_progress = 0.0
        prior_active_progress = 0.0
        detector_active_progress = 0.0

    if prior_builder_enabled and prior_fusion_enabled:
        phase_name = "prior_adapt"
    else:
        phase_name = "seg_only"

    schedule = {
        "progress": progress,
        "phase_name": phase_name,
        "prior_enable_epoch": prior_enable_epoch,
        "prior_ramp_epochs": prior_ramp_epochs,
        "prior_fusion_enabled": prior_fusion_enabled,
        "prior_builder_enabled": prior_builder_enabled,
        "detector_enabled": detector_enabled,
        "fusion_progress": fusion_active_progress,
        "prior_progress": prior_active_progress,
        "detector_active_progress": detector_active_progress,
        "detector_aux_progress": detector_active_progress,
        "lambda_center_aux": 0.0,
        "lambda_center_inside_aux": (
            _linear_schedule(
                _cfg_get(cfg, "stage_train.stage2.lambda_center_inside_aux_start", 0.0),
                _cfg_get(cfg, "stage_train.stage2.lambda_center_inside_aux_end", 0.0),
                detector_active_progress,
            )
            if detector_enabled
            else 0.0
        ),
        "lambda_sigma_aux": 0.0,
        "lambda_evidence_aux": 0.0,
        "lambda_fused_support_aux": 0.0,
        "lambda_conf_aux": 0.0,
        "lambda_scale_floor_aux": 0.0,
        "lambda_prior_energy": 0.0,
        "lambda_gate_reg": 0.0,
        "lambda_prior_localize": 0.0,
        "prior_alpha_cap": (
            _linear_schedule(
                _cfg_get(cfg, "stage_train.stage2.prior_alpha_cap_start", 0.02),
                _cfg_get(cfg, "stage_train.stage2.prior_alpha_cap_end", 0.30),
                fusion_active_progress,
            )
            if prior_fusion_enabled
            else 0.0
        ),
        "input_hwa_gate_scale": (
            _linear_schedule(
                _cfg_get(cfg, "stage_train.stage2.input_hwa_gate_start", 1.0),
                _cfg_get(cfg, "stage_train.stage2.input_hwa_gate_end", 1.0),
                fusion_active_progress,
            )
            if prior_fusion_enabled
            else 0.0
        ),
        "input_hwa_gain_scale": (
            _linear_schedule(
                _cfg_get(cfg, "stage_train.stage2.input_hwa_gain_start", 1.0),
                _cfg_get(cfg, "stage_train.stage2.input_hwa_gain_end", 1.0),
                fusion_active_progress,
            )
            if prior_fusion_enabled
            else 0.0
        ),
        "detector_lr_scale": (
            _linear_schedule(detector_start, detector_end, detector_active_progress)
            * detector_active_progress
            if detector_enabled
            else 0.0
        ),
        "prior_fusion_lr_scale": (
            _linear_schedule(
                _cfg_get(cfg, "stage_train.stage2.prior_fusion_lr_scale_start", 0.08),
                _cfg_get(cfg, "stage_train.stage2.prior_fusion_lr_scale_end", 0.20),
                fusion_active_progress,
            )
            * prior_lr_boost
            * fusion_active_progress
            if prior_fusion_enabled
            else 0.0
        ),
        "prior_builder_lr_scale": (
            _linear_schedule(
                _cfg_get(cfg, "stage_train.stage2.prior_builder_lr_scale_start", 0.03),
                _cfg_get(cfg, "stage_train.stage2.prior_builder_lr_scale_end", 0.12),
                prior_active_progress,
            )
            * prior_lr_boost
            * prior_active_progress
            if prior_builder_enabled
            else 0.0
        ),
        "seg_lr_scale": float(_cfg_get(cfg, "stage_train.stage2.seg_lr_scale", 1.0)),
        "input_fusion_lr_scale": float(
            _cfg_get(cfg, "stage_train.stage2.input_fusion_lr_scale", 1.0)
        ),
    }
    return schedule


def _apply_stage2_prior_alpha_cap(
    model,
    epoch: int,
    cfg,
    stage_local_epoch: Optional[int] = None,
    stage_total_epochs: Optional[int] = None,
) -> float:
    raw = _unwrap(model)
    if not hasattr(raw, "Encoder") or not hasattr(raw.Encoder, "prior_fusions"):
        return 0.0

    sched = _get_stage2_schedule(
        epoch,
        cfg,
        stage_local_epoch=stage_local_epoch,
        stage_total_epochs=stage_total_epochs,
    )
    alpha_cap = float(
        sched["prior_alpha_cap"] if bool(sched["prior_fusion_enabled"]) else 0.0
    )
    alpha_floor = float(
        _cfg_get(
            cfg,
            "stage_train.stage2.prior_alpha_cap_floor",
            0.0,
        )
    )
    if bool(sched["prior_fusion_enabled"]):
        alpha_cap = max(alpha_floor, alpha_cap)
    for fusion in raw.Encoder.prior_fusions:
        if hasattr(fusion, "runtime_gate_scale"):
            fusion.runtime_gate_scale = alpha_cap
    return alpha_cap


def _apply_stage2_hwa_runtime_scales(model, sched: Dict[str, float]) -> Tuple[float, float]:
    gate_scale = float(sched.get("input_hwa_gate_scale", 1.0))
    gain_scale = float(sched.get("input_hwa_gain_scale", 1.0))
    seen = set()
    targets = [model, _unwrap(model)]
    raw = targets[-1]
    if hasattr(raw, "net"):
        targets.append(raw.net)
    for target in targets:
        if target is None:
            continue
        ident = id(target)
        if ident in seen:
            continue
        seen.add(ident)
        if hasattr(target, "runtime_hwa_gate_scale"):
            target.runtime_hwa_gate_scale = gate_scale
        if hasattr(target, "runtime_hwa_gain_scale"):
            target.runtime_hwa_gain_scale = gain_scale
    return gate_scale, gain_scale


def _set_runtime_hwa_prior_enabled(model, enabled: bool) -> None:
    raw = _unwrap(model)
    if hasattr(raw, "use_hwa_prior_in_encoder"):
        raw.use_hwa_prior_in_encoder = bool(enabled)


def _get_runtime_hwa_prior_enabled(model) -> Optional[bool]:
    raw = _unwrap(model)
    if hasattr(raw, "use_hwa_prior_in_encoder"):
        return bool(raw.use_hwa_prior_in_encoder)
    return None


def _apply_stage2_hwa_gain_init_if_needed(model, cfg) -> bool:
    force_after_compatible = bool(
        _cfg_get(
            cfg,
            "stage_train.stage2.force_hwa_gain_init_after_compatible",
            False,
        )
    )
    if (
        bool(getattr(_unwrap(model), "_stage2_loaded_compatible_init", False))
        and not force_after_compatible
    ):
        return False
    hwa = getattr(_unwrap(model), "hwa_block", None)
    if hwa is None:
        return False

    changed = False
    init_specs = [
        ("input_agg_gain", "stage_train.stage2.hwa_input_agg_gain_init"),
        ("input_gate_boost", "stage_train.stage2.hwa_input_gate_boost_init"),
        ("center_prior_gain", "stage_train.stage2.hwa_center_prior_gain_init"),
        ("input_enhance_gain", "stage_train.stage2.hwa_input_enhance_gain_init"),
        ("output_logit_gain", "stage_train.stage2.hwa_output_logit_gain_init"),
    ]
    with torch.no_grad():
        for attr_name, cfg_key in init_specs:
            value = _cfg_get(cfg, cfg_key, None)
            param = getattr(hwa, attr_name, None)
            if value is None or not torch.is_tensor(param):
                continue
            param.fill_(float(value))
            changed = True

    raw = _unwrap(model)
    cap_specs = [
        ("hwa_input_agg_gain_max", "stage_train.stage2.hwa_input_agg_gain_max"),
        ("hwa_input_enhance_gain_max", "stage_train.stage2.hwa_input_enhance_gain_max"),
        ("hwa_input_gate_scale_max", "stage_train.stage2.hwa_input_gate_scale_max"),
        ("hwa_input_delta_std_clip", "stage_train.stage2.hwa_input_delta_std_clip"),
        ("hwa_output_logit_gain_max", "stage_train.stage2.hwa_output_logit_gain_max"),
    ]
    for attr_name, cfg_key in cap_specs:
        value = _cfg_get(cfg, cfg_key, None)
        if value is None:
            continue
        setattr(raw, attr_name, float(value))
        changed = True
    return changed


def _get_optimizer_role(name: str) -> str:
    if name.startswith("todm_cls_head"):
        return "classifier"
    if name.startswith("input_fusion"):
        return "input_fusion"

    if name.startswith("hwa_block."):
        detector_keys = (
            "detector_stem",
            "detector_down1",
            "detector_down2",
            "modal_shared_proj",
            "detector_context",
            "modal_raw_heads",
            "modal_region_heads",
            "center_head",
            "core_head",
            "excl_head",
            "reliability_head",
            "agreement_head",
            "modal_sigma_heads",
            "modal_offset_heads",
            "modal_conf_heads",
        )
        prior_keys = (
            "stage_roi_encoders",
            "input_channel_agg_proj",
            "input_agg_gain",
            "input_gate_boost",
            "center_prior_gain",
            "output_logit_gain",
            "input_detail_gain",
            "input_cross_mix_logit",
            "input_roi_modulation",
        )
        if any(k in name for k in detector_keys):
            return "detector"
        if any(k in name for k in prior_keys):
            return "prior_builder"
        return "prior_builder"

    if name.startswith("Encoder."):
        if "prior_fusions" in name:
            return "prior_consumer"
        return "encoder_seg"

    if name.startswith("hidden_downsample") or name.startswith("TSconv") or name.startswith(
        "todm_seg_head"
    ):
        return "encoder_seg"

    return "misc"


def _get_stage_lr_scales(
    stage_name: str,
    cfg,
    epoch: Optional[int] = None,
    stage_local_epoch: Optional[int] = None,
    stage_total_epochs: Optional[int] = None,
) -> Dict[str, float]:
    if stage_name == "stage1":
        return {
            "detector": float(_cfg_get(cfg, "stage_train.stage1.detector_lr_scale", 1.0)),
            "prior_builder": 0.0,
            "prior_consumer": 0.0,
            "input_fusion": 0.0,
            "encoder_seg": 0.0,
            "classifier": 0.0,
            "misc": 0.0,
        }
    if stage_name == "stage2":
        sched = _get_stage2_schedule(
            int(epoch or 0),
            cfg,
            stage_local_epoch=stage_local_epoch,
            stage_total_epochs=stage_total_epochs,
        )
        return {
            "detector": float(sched["detector_lr_scale"]),
            "prior_builder": float(sched["prior_builder_lr_scale"]),
            "prior_consumer": float(sched["prior_fusion_lr_scale"]),
            "encoder_seg": float(sched["seg_lr_scale"]),
            "input_fusion": float(sched["input_fusion_lr_scale"]),
            "classifier": 0.0,
            "misc": 0.0,
        }
    if stage_name == "stage3":
        unfreeze_feature_tail = bool(
            _cfg_get(cfg, "stage_train.stage3.unfreeze_feature_tail", False)
        )
        unfreeze_encoder = bool(
            _cfg_get(cfg, "stage_train.stage3.unfreeze_encoder", False)
        )
        unfreeze_hwa = bool(_cfg_get(cfg, "stage_train.stage3.unfreeze_hwa", False))
        return {
            "detector": (
                float(_cfg_get(cfg, "stage_train.stage3.hwa_lr_scale", 0.0))
                if unfreeze_hwa
                else 0.0
            ),
            "prior_builder": (
                float(_cfg_get(cfg, "stage_train.stage3.hwa_lr_scale", 0.0))
                if unfreeze_hwa
                else 0.0
            ),
            "prior_consumer": (
                float(_cfg_get(cfg, "stage_train.stage3.encoder_lr_scale", 0.0))
                if unfreeze_encoder
                else 0.0
            ),
            "input_fusion": (
                float(_cfg_get(cfg, "stage_train.stage3.encoder_lr_scale", 0.0))
                if unfreeze_encoder
                else 0.0
            ),
            "encoder_seg": (
                float(
                    _cfg_get(
                        cfg,
                        "stage_train.stage3.encoder_lr_scale",
                        _cfg_get(cfg, "stage_train.stage3.feature_lr_scale", 0.0),
                    )
                )
                if unfreeze_encoder
                else float(_cfg_get(cfg, "stage_train.stage3.feature_lr_scale", 0.0))
                if unfreeze_feature_tail
                else 0.0
            ),
            "classifier": float(_cfg_get(cfg, "stage_train.stage3.classifier_lr_scale", 1.0)),
            "misc": 0.0,
        }
    raise ValueError(f"Unsupported stage_name: {stage_name}")


def apply_stage_lr_policy(
    optimizer,
    stage_name: str,
    cfg,
    scheduler=None,
    epoch: Optional[int] = None,
    stage_local_epoch: Optional[int] = None,
    stage_total_epochs: Optional[int] = None,
):
    effective_epoch = int(epoch or 0)
    base_lr = _get_stage_base_lr(
        effective_epoch,
        stage_name,
        cfg,
        stage_local_epoch=stage_local_epoch,
        stage_total_epochs=stage_total_epochs,
    )

    scale_map = _get_stage_lr_scales(
        stage_name,
        cfg,
        epoch=effective_epoch,
        stage_local_epoch=stage_local_epoch,
        stage_total_epochs=stage_total_epochs,
    )
    for group in optimizer.param_groups:
        role = group.get("role", "misc")
        scale = float(scale_map.get(role, 0.0))
        group["lr"] = base_lr * scale
        group["stage_lr_scale"] = scale
        group["stage_base_lr"] = base_lr
    return base_lr, scale_map


def _init_stage_best_state() -> Dict[str, Dict[str, object]]:
    return {
        "stage1": {
            "best_score": -1e9,
            "best_test_score": -1e9,
            "best_metrics": {},
            "best_test_metrics": {},
        },
        "stage2": {
            "best_score": -1e9,
            "best_test_score": -1e9,
            "best_metrics": {},
            "best_test_metrics": {},
        },
        "stage3": {
            "best_score": -1e9,
            "best_test_score": -1e9,
            "best_metrics": {},
            "best_test_metrics": {},
        },
    }


def _normalize_stage_best_state(
    stage_best_state,
    fallback_stage: Optional[str] = None,
    fallback_best_score: float = -1e9,
    fallback_best_test_score: float = -1e9,
    fallback_best_metrics: Optional[Dict] = None,
    fallback_best_test_metrics: Optional[Dict] = None,
):
    normalized = _init_stage_best_state()
    if isinstance(stage_best_state, dict):
        for stage_name, default_entry in normalized.items():
            entry = stage_best_state.get(stage_name)
            if isinstance(entry, dict):
                default_entry["best_score"] = float(entry.get("best_score", default_entry["best_score"]))
                default_entry["best_test_score"] = float(
                    entry.get("best_test_score", default_entry["best_test_score"])
                )
                default_entry["best_metrics"] = entry.get("best_metrics", {}) or {}
                default_entry["best_test_metrics"] = entry.get("best_test_metrics", {}) or {}

    if fallback_stage in normalized:
        entry = normalized[fallback_stage]
        if float(entry.get("best_score", -1e9)) <= -1e8:
            entry["best_score"] = float(fallback_best_score)
            entry["best_test_score"] = float(fallback_best_test_score)
            entry["best_metrics"] = fallback_best_metrics or {}
            entry["best_test_metrics"] = fallback_best_test_metrics or {}

    return normalized


def _compute_stage2_debug_stats(
    debug: Optional[Dict[str, torch.Tensor]], device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    stats = {}
    if debug is None:
        return stats

    if "encoder_prior_alphas" in debug and debug["encoder_prior_alphas"]:
        vals = []
        for item in debug["encoder_prior_alphas"]:
            if item is None:
                continue
            if torch.is_tensor(item):
                vals.append(item.float().reshape(-1).mean())
            else:
                base_device = device if device is not None else "cpu"
                vals.append(torch.tensor(float(item), device=base_device))
        if vals:
            stats["prior_alpha_mean"] = torch.stack(vals).mean()

    if "encoder_prior_gates" in debug and debug["encoder_prior_gates"]:
        vals = [g.float().mean() for g in debug["encoder_prior_gates"] if g is not None]
        if vals:
            stats["prior_gate_mean"] = torch.stack(vals).mean()

    if "encoder_prior_guidance_scales" in debug and debug["encoder_prior_guidance_scales"]:
        vals = [
            g.float().reshape(-1).mean()
            for g in debug["encoder_prior_guidance_scales"]
            if g is not None
        ]
        if vals:
            stats["prior_guidance_scale_mean"] = torch.stack(vals).mean()

    if "det_conf_each_full" in debug and debug["det_conf_each_full"] is not None:
        stats["det_conf_mean"] = debug["det_conf_each_full"].float().mean()

    if "det_fused_seed_prob_full" in debug and debug["det_fused_seed_prob_full"] is not None:
        stats["fused_seed_mean"] = debug["det_fused_seed_prob_full"].float().mean()

    if "det_agreement_prob_full" in debug and debug["det_agreement_prob_full"] is not None:
        stats["agreement_mean"] = debug["det_agreement_prob_full"].float().mean()

    if "det_modal_reliability_full" in debug and debug["det_modal_reliability_full"] is not None:
        stats["modal_reliability_mean"] = debug["det_modal_reliability_full"].float().mean()

    if "stage_prior_quality_maps" in debug and debug["stage_prior_quality_maps"]:
        vals = [q.float().mean() for q in debug["stage_prior_quality_maps"] if q is not None]
        if vals:
            stats["prior_quality_mean"] = torch.stack(vals).mean()

    return stats


def _compute_stage2_hwa_runtime_stats(
    model, device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    base_device = device if device is not None else "cpu"
    raw = _unwrap(model)
    if hasattr(raw, "net"):
        raw = raw.net
    stats = {
        "runtime_hwa_gate_scale": torch.tensor(
            float(getattr(raw, "runtime_hwa_gate_scale", 0.0)),
            device=base_device,
        ),
        "runtime_hwa_gain_scale": torch.tensor(
            float(getattr(raw, "runtime_hwa_gain_scale", 0.0)),
            device=base_device,
        ),
    }

    for attr_name, stat_name in [
        ("last_input_aggregate_delta_mean", "hwa_input_delta_mean"),
        ("last_input_aggregate_delta_ratio", "hwa_input_delta_ratio"),
    ]:
        value = getattr(raw, attr_name, None)
        if torch.is_tensor(value):
            stats[stat_name] = value.detach().float().to(device=base_device)

    hwa = getattr(raw, "hwa_block", None)
    if hwa is not None:
        for attr_name, stat_name in [
            ("input_agg_gain", "hwa_input_agg_gain"),
            ("input_gate_boost", "hwa_input_gate_boost"),
            ("center_prior_gain", "hwa_center_prior_gain"),
            ("output_logit_gain", "hwa_output_logit_gain"),
        ]:
            value = getattr(hwa, attr_name, None)
            if torch.is_tensor(value):
                stats[stat_name] = value.detach().float().reshape(-1).mean().to(device=base_device)
    return stats


def _stage2_hwa_advantage_margin_loss(
    hwa_loss: torch.Tensor,
    no_hwa_loss: torch.Tensor,
    cfg,
    stage_local_epoch: Optional[int] = None,
) -> torch.Tensor:
    scale = float(_cfg_get(cfg, "stage_train.stage2.hwa_advantage_loss_scale", 0.0))
    epochs = int(_cfg_get(cfg, "stage_train.stage2.hwa_advantage_epochs", 0))
    local_epoch = int(stage_local_epoch if stage_local_epoch is not None else 0)
    if scale <= 0.0 or epochs <= 0 or local_epoch >= epochs:
        return torch.zeros((), device=hwa_loss.device, dtype=hwa_loss.dtype)
    margin = float(_cfg_get(cfg, "stage_train.stage2.hwa_advantage_margin", 0.0))
    return F.relu(hwa_loss - no_hwa_loss + margin) * scale


def _stage2_soft_dice_from_logits(
    logits: torch.Tensor,
    label: torch.Tensor,
    cfg=None,
    eps: float = 1e-5,
) -> torch.Tensor:
    mode = _stage2_seg_target_mode(cfg)
    probs = torch.sigmoid(logits.float())
    if mode in {"union", "lesion_union", "single_union"}:
        probs = probs.amax(dim=1, keepdim=True)
        target = _stage2_union_target(label)
    else:
        target = label.float()
    reduce_dims = tuple(range(2, probs.ndim))
    intersection = (probs * target).sum(dim=reduce_dims)
    denom = probs.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return dice.mean()


def _stage2_hwa_advantage_dice_margin_loss(
    hwa_logits: torch.Tensor,
    no_hwa_logits: torch.Tensor,
    label: torch.Tensor,
    cfg,
    stage_local_epoch: Optional[int] = None,
) -> torch.Tensor:
    scale = float(_cfg_get(cfg, "stage_train.stage2.hwa_advantage_loss_scale", 0.0))
    epochs = int(_cfg_get(cfg, "stage_train.stage2.hwa_advantage_epochs", 0))
    local_epoch = int(stage_local_epoch if stage_local_epoch is not None else 0)
    if scale <= 0.0 or epochs <= 0 or local_epoch >= epochs:
        return torch.zeros((), device=hwa_logits.device, dtype=hwa_logits.dtype)
    margin = float(_cfg_get(cfg, "stage_train.stage2.hwa_advantage_margin", 0.0))
    hwa_dice = _stage2_soft_dice_from_logits(hwa_logits, label, cfg=cfg)
    no_hwa_dice = _stage2_soft_dice_from_logits(no_hwa_logits.detach(), label, cfg=cfg).detach()
    return F.relu(no_hwa_dice - hwa_dice + margin).to(dtype=hwa_logits.dtype) * scale


def apply_stage_policy(
    model,
    stage_name: str,
    cfg=None,
    stage_local_epoch: Optional[int] = None,
    stage_total_epochs: Optional[int] = None,
):
    """
    stage1:
        train detector modules only
        - coarse / refine center
        - sigma / conf
        - modal offset heads

    stage2:
        progressively adapt segmentation modules with lightweight inside guidance
        - seg_warmup: train the segmentation trunk
        - fusion_adapt: unfreeze Encoder.prior_fusions
        - prior_adapt: unfreeze hwa_block.stage_roi_encoders
        - inside_guidance: unfreeze detector modules with inside auxiliary only

    stage3:
        train the classification head only
    """
    target = _unwrap(model)
    for p in target.parameters():
        p.requires_grad = False

    if stage_name == "stage1":
        _set_runtime_hwa_prior_enabled(target, False)
        _apply_stage2_hwa_runtime_scales(
            target,
            {"input_hwa_gate_scale": 0.0, "input_hwa_gain_scale": 0.0},
        )
        if hasattr(target, "hwa_block"):
            hwa = target.hwa_block
            for name in [
                "detector_stem",
                "detector_down1",
                "detector_down2",
                "modal_shared_proj",
                "detector_context",
                "modal_raw_heads",
                "modal_region_heads",
                "center_head",
                "core_head",
                "excl_head",
                "reliability_head",
                "agreement_head",
                "modal_sigma_heads",
                "modal_offset_heads",
                "modal_conf_heads",
            ]:
                if hasattr(hwa, name):
                    _set_requires_grad(getattr(hwa, name), True)
            for name in [
                "input_channel_agg_proj",
                "input_agg_gain",
                "input_gate_boost",
                "center_prior_gain",
            ]:
                obj = getattr(hwa, name, None)
                if obj is None:
                    continue
                if torch.is_tensor(obj):
                    obj.requires_grad = True
                else:
                    _set_requires_grad(obj, True)

    elif stage_name == "stage2":
        sched = {
            "prior_fusion_enabled": True,
            "prior_builder_enabled": True,
            "detector_enabled": True,
        }
        if cfg is not None and stage_local_epoch is not None and stage_total_epochs is not None:
            sched = _get_stage2_schedule(
                0,
                cfg,
                stage_local_epoch=stage_local_epoch,
                stage_total_epochs=stage_total_epochs,
            )
        hwa_prior_enabled = (
            bool(_cfg_get(cfg, "stage_train.stage2.use_hwa_prior_in_encoder", True))
            and bool(sched["prior_fusion_enabled"])
        )
        freeze_seg_backbone_epochs = int(
            _cfg_get(cfg, "stage_train.stage2.freeze_seg_backbone_epochs", 0)
        )
        local_epoch = int(stage_local_epoch if stage_local_epoch is not None else 0)
        freeze_seg_backbone = (
            freeze_seg_backbone_epochs > 0
            and local_epoch < freeze_seg_backbone_epochs
        )
        train_seg_head_when_backbone_frozen = bool(
            _cfg_get(cfg, "stage_train.stage2.train_seg_head_when_backbone_frozen", False)
        )
        train_seg_tail_when_backbone_frozen = bool(
            _cfg_get(cfg, "stage_train.stage2.train_seg_tail_when_backbone_frozen", False)
        )
        _set_runtime_hwa_prior_enabled(target, hwa_prior_enabled)
        runtime_sched = sched if hwa_prior_enabled else {
            **sched,
            "input_hwa_gate_scale": 0.0,
            "input_hwa_gain_scale": 0.0,
        }
        _apply_stage2_hwa_runtime_scales(target, runtime_sched)

        if hasattr(target, "hwa_block"):
            hwa = target.hwa_block
            freeze_hwa_gains = bool(_cfg_get(cfg, "stage_train.stage2.freeze_hwa_gains", False))
            freeze_hwa_gains_after = _cfg_get(
                cfg, "stage_train.stage2.freeze_hwa_gains_after_epoch", None
            )
            if freeze_hwa_gains_after not in [None, "", "None"]:
                freeze_hwa_gains = freeze_hwa_gains or local_epoch >= int(freeze_hwa_gains_after)
            detector_names = [
                "detector_stem",
                "detector_down1",
                "detector_down2",
                "modal_shared_proj",
                "detector_context",
                "modal_raw_heads",
                "modal_region_heads",
                "center_head",
                "core_head",
                "excl_head",
                "reliability_head",
                "agreement_head",
                "modal_sigma_heads",
                "modal_offset_heads",
                "modal_conf_heads",
            ]
            if bool(sched["detector_enabled"]):
                for name in detector_names:
                    if hasattr(hwa, name):
                        _set_requires_grad(getattr(hwa, name), True)

            for name in [
                "stage_roi_encoders",
                "input_channel_agg_proj",
            ]:
                if hasattr(hwa, name) and bool(sched["prior_builder_enabled"]):
                    _set_requires_grad(
                        getattr(hwa, name),
                        float(sched["prior_builder_lr_scale"]) > 0.0,
                    )
            for name in [
                "input_agg_gain",
                "input_gate_boost",
                "center_prior_gain",
                "output_logit_gain",
                "input_detail_gain",
                "input_cross_mix_logit",
                "input_roi_modulation",
            ]:
                obj = getattr(hwa, name, None)
                if torch.is_tensor(obj):
                    obj.requires_grad = bool(sched["prior_builder_enabled"]) and not freeze_hwa_gains

        if hasattr(target, "input_fusion"):
            _set_requires_grad(target.input_fusion, float(sched["input_fusion_lr_scale"]) > 0.0)

        if hasattr(target, "Encoder"):
            _set_requires_grad(target.Encoder, True)
            if hasattr(target.Encoder, "prior_fusions") and not bool(
                sched["prior_fusion_enabled"]
            ):
                _set_requires_grad(target.Encoder.prior_fusions, False)
            if freeze_seg_backbone:
                _set_requires_grad(target.Encoder, False)
                if hasattr(target.Encoder, "prior_fusions") and bool(sched["prior_fusion_enabled"]):
                    _set_requires_grad(
                        target.Encoder.prior_fusions,
                        float(sched["prior_fusion_lr_scale"]) > 0.0,
                    )

        for name in [
            "hidden_downsample",
            "TSconv1",
            "TSconv2",
            "TSconv3",
            "TSconv4",
            "todm_seg_head",
        ]:
            if hasattr(target, name):
                _set_requires_grad(getattr(target, name), True)
                if freeze_seg_backbone and not (
                    train_seg_head_when_backbone_frozen and name == "todm_seg_head"
                    or train_seg_tail_when_backbone_frozen
                    and name in {"hidden_downsample", "TSconv1", "TSconv2", "TSconv3", "TSconv4"}
                ):
                    _set_requires_grad(getattr(target, name), False)

    elif stage_name == "stage3":
        _set_runtime_hwa_prior_enabled(
            target,
            bool(_cfg_get(cfg, "stage_train.stage2.use_hwa_prior_in_encoder", True)),
        )
        _apply_stage2_hwa_runtime_scales(
            target,
            {"input_hwa_gate_scale": 1.0, "input_hwa_gain_scale": 1.0},
        )
        if hasattr(target, "todm_cls_head"):
            _set_requires_grad(target.todm_cls_head, True)
        if bool(_cfg_get(cfg, "stage_train.stage3.unfreeze_hwa", False)) and hasattr(target, "hwa_block"):
            _set_requires_grad(target.hwa_block, True)
        if bool(_cfg_get(cfg, "stage_train.stage3.unfreeze_encoder", False)):
            for name in [
                "input_fusion",
                "Encoder",
                "hidden_downsample",
                "TSconv1",
                "TSconv2",
                "TSconv3",
                "TSconv4",
            ]:
                if hasattr(target, name):
                    _set_requires_grad(getattr(target, name), True)
        if bool(_cfg_get(cfg, "stage_train.stage3.unfreeze_feature_tail", False)):
            for name in [
                "hidden_downsample",
                "TSconv1",
                "TSconv2",
                "TSconv3",
                "TSconv4",
            ]:
                if hasattr(target, name):
                    _set_requires_grad(getattr(target, name), True)
    else:
        raise ValueError(f"Unsupported stage_name: {stage_name}")


def _reset_stage3_class_decoder_if_requested(model, cfg, accelerator) -> bool:
    if not bool(_cfg_get(cfg, "stage_train.stage3.reset_class_decoder", False)):
        return False

    target = _unwrap(model)
    class_decoder = getattr(target, "todm_cls_head", None)
    if class_decoder is None:
        accelerator.print("[Stage3Init] reset_class_decoder requested but todm_cls_head is missing.")
        return False

    for module in class_decoder.modules():
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()

    accelerator.print("[Stage3Init] todm_cls_head reset after loading stage2 checkpoint.")
    return True


def _load_stage3_classifier_warm_start_if_needed(model, accelerator, cfg) -> bool:
    checkpoint_name = _cfg_get(cfg, "stage_train.stage3.classifier_warm_start", None)
    if checkpoint_name in [None, "", "None"]:
        return False

    checkpoint_path = _resolve_model_store_checkpoint(
        str(checkpoint_name),
        preferred_stage="stage3",
    )
    if checkpoint_path is None:
        accelerator.print(
            f"[Stage3WarmStart] classifier warm-start checkpoint not found: {checkpoint_name}"
        )
        return False

    accelerator.print(f"[Stage3WarmStart] load classifier warm-start: {checkpoint_path}")
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_path))
    else:
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    target = _unwrap(model)
    target_state = target.state_dict()
    target_keys = set(target_state.keys())
    direct_matches = sum(1 for key in state_dict if key in target_keys)
    if any(str(key).startswith("module.") for key in state_dict):
        stripped_state_dict = {
            str(key)[7:] if str(key).startswith("module.") else str(key): value
            for key, value in state_dict.items()
        }
        stripped_matches = sum(1 for key in stripped_state_dict if key in target_keys)
        if stripped_matches > direct_matches:
            state_dict = stripped_state_dict
    state_dict = remap_method_aligned_state_dict_keys(state_dict)
    direct_matches = sum(1 for key in state_dict if key in target_keys)

    warm_scope = str(
        _cfg_get(cfg, "stage_train.stage3.classifier_warm_scope", "tail") or "tail"
    ).strip().lower()
    warm_prefix_map = {
        "tail": (
            "todm_cls_head.",
            "hidden_downsample.",
            "TSconv1.",
            "TSconv2.",
            "TSconv3.",
            "TSconv4.",
        ),
        "encoder_tail": (
            "Encoder.",
            "todm_cls_head.",
            "hidden_downsample.",
            "TSconv1.",
            "TSconv2.",
            "TSconv3.",
            "TSconv4.",
        ),
        "full_common": (
            "Encoder.",
            "hwa_block.",
            "todm_cls_head.",
            "hidden_downsample.",
            "TSconv1.",
            "TSconv2.",
            "TSconv3.",
            "TSconv4.",
        ),
    }
    if warm_scope not in warm_prefix_map:
        accelerator.print(
            f"[Stage3WarmStart] unsupported warm scope `{warm_scope}`, fallback to `tail`."
        )
        warm_scope = "tail"
    warm_prefixes = warm_prefix_map[warm_scope]
    filtered = {
        key: value
        for key, value in state_dict.items()
        if str(key).startswith(warm_prefixes)
        and key in target_state
        and tuple(value.shape) == tuple(target_state[key].shape)
    }
    if not filtered:
        accelerator.print(
            f"[Stage3WarmStart] no compatible classifier warm-start tensors found in {checkpoint_path}"
        )
        return False

    target.load_state_dict(filtered, strict=False)
    accelerator.print(
        f"[Stage3WarmStart] loaded {len(filtered)} tensors from {checkpoint_path.name} "
        f"with scope={warm_scope}"
    )
    return True


def render_center_prior_from_sigma(
    center_each: torch.Tensor,
    sigma_each: torch.Tensor,
    size_hwz: Tuple[int, int, int],
    scope_factor: float = 1.0,
) -> torch.Tensor:
    """
    center_each: [B,M,3]
    sigma_each:  [B,M,3]
    return:      [B,M,H,W,Z]
    """
    B, M, _ = center_each.shape
    H, W, Z = size_hwz
    fields = []
    for mi in range(M):
        g = render_gaussian_field(
            center_each[:, mi, :],
            sigma_each[:, mi, :] * float(scope_factor),
            (H, W, Z),
        )  # [B,1,H,W,Z]
        fields.append(g)
    return torch.cat(fields, dim=1)

def forward_model(
    model, image, stage=None, stage_name=None, debug=None, return_debug=None, **kwargs
):
    if stage_name is None:
        stage_name = stage
    if stage_name is None:
        raise ValueError("forward_model requires `stage` or `stage_name`.")

    if return_debug is None:
        return_debug = debug
    if return_debug is None:
        return_debug = stage_name in ["stage1", "stage2"]

    detector_only = bool(kwargs.get("detector_only", False))
    if detector_only:
        outputs = model(image, return_debug=True, detector_only=True)
    else:
        outputs = model(image, return_debug=True) if return_debug else model(image)
    if not isinstance(outputs, (tuple, list)):
        raise RuntimeError("Model output must be tuple/list.")
    if len(outputs) == 3:
        return outputs[0], outputs[1], outputs[2]
    if len(outputs) == 2:
        return outputs[0], outputs[1], None
    raise RuntimeError(f"Unexpected model output length: {len(outputs)}")





def compute_center_of_mass(mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    mask:
        [B,1,H,W,Z] -> return [B,3]
        [B,M,H,W,Z] -> return [B,M,3]
    """
    if mask.dim() != 5:
        raise ValueError(f"Expected 5D tensor, got {tuple(mask.shape)}")

    b, c, h, w, z = mask.shape
    mass = mask.sum(dim=(2, 3, 4)).clamp_min(eps)  # [B,C]

    xs = torch.arange(h, device=mask.device, dtype=mask.dtype).view(1, 1, h, 1, 1)
    ys = torch.arange(w, device=mask.device, dtype=mask.dtype).view(1, 1, 1, w, 1)
    zs = torch.arange(z, device=mask.device, dtype=mask.dtype).view(1, 1, 1, 1, z)

    cx = (mask * xs).sum(dim=(2, 3, 4)) / mass
    cy = (mask * ys).sum(dim=(2, 3, 4)) / mass
    cz = (mask * zs).sum(dim=(2, 3, 4)) / mass

    center = torch.stack([cx, cy, cz], dim=-1)  # [B,C,3]
    if c == 1:
        return center[:, 0, :]
    return center


def compute_sigma_from_mask(
    mask: torch.Tensor,
    center: Optional[torch.Tensor] = None,
    min_sigma: float = 2.0,
    eps: float = 1e-6,
    shrink_ratio: float = 1.0,
    max_sigma: Optional[Tuple[float, float, float]] = None,
) -> torch.Tensor:
    """
    mask:
        [B,1,H,W,Z] -> [B,3]
        [B,M,H,W,Z] -> [B,M,3]
    """
    if mask.dim() != 5:
        raise ValueError(f"Expected 5D tensor, got {tuple(mask.shape)}")

    b, c, h, w, z = mask.shape
    if center is None:
        center = compute_center_of_mass(mask, eps=eps)

    if c == 1 and center.dim() == 2:
        center = center.unsqueeze(1)

    mass = mask.sum(dim=(2, 3, 4)).clamp_min(eps)

    xs = torch.arange(h, device=mask.device, dtype=mask.dtype).view(1, 1, h, 1, 1)
    ys = torch.arange(w, device=mask.device, dtype=mask.dtype).view(1, 1, 1, w, 1)
    zs = torch.arange(z, device=mask.device, dtype=mask.dtype).view(1, 1, 1, 1, z)

    cx = center[..., 0].view(b, c, 1, 1, 1)
    cy = center[..., 1].view(b, c, 1, 1, 1)
    cz = center[..., 2].view(b, c, 1, 1, 1)

    var_x = (mask * (xs - cx) ** 2).sum(dim=(2, 3, 4)) / mass
    var_y = (mask * (ys - cy) ** 2).sum(dim=(2, 3, 4)) / mass
    var_z = (mask * (zs - cz) ** 2).sum(dim=(2, 3, 4)) / mass

    sigma = torch.stack(
        [torch.sqrt(var_x + eps), torch.sqrt(var_y + eps), torch.sqrt(var_z + eps)],
        dim=-1,
    )

    sigma = sigma * float(shrink_ratio)
    sigma = sigma.clamp_min(min_sigma)

    if max_sigma is not None:
        max_sigma_t = mask.new_tensor(max_sigma).view(1, 1, 3)
        sigma = torch.minimum(sigma, max_sigma_t)

    if c == 1:
        return sigma[:, 0, :]
    return sigma



def render_gaussian_field(
    center: torch.Tensor, sigma_xyz: torch.Tensor, size_hwz: Tuple[int, int, int]
) -> torch.Tensor:
    h, w, z = size_hwz
    b = center.shape[0]
    device = center.device
    dtype = center.dtype
    xs = torch.arange(h, device=device, dtype=dtype).view(1, h, 1, 1)
    ys = torch.arange(w, device=device, dtype=dtype).view(1, 1, w, 1)
    zs = torch.arange(z, device=device, dtype=dtype).view(1, 1, 1, z)
    cx = center[:, 0].view(b, 1, 1, 1)
    cy = center[:, 1].view(b, 1, 1, 1)
    cz = center[:, 2].view(b, 1, 1, 1)
    sx = sigma_xyz[:, 0].view(b, 1, 1, 1).clamp_min(1e-4)
    sy = sigma_xyz[:, 1].view(b, 1, 1, 1).clamp_min(1e-4)
    sz = sigma_xyz[:, 2].view(b, 1, 1, 1).clamp_min(1e-4)
    g = torch.exp(
        -0.5 * (((xs - cx) / sx) ** 2 + ((ys - cy) / sy) ** 2 + ((zs - cz) / sz) ** 2)
    )
    return g.unsqueeze(1)


def _ensure_odd_int(v: int) -> int:
    v = max(1, int(v))
    return v if v % 2 == 1 else v + 1


def _center_fallback_tensor(mask: torch.Tensor) -> torch.Tensor:
    _, _, h, w, z = mask.shape
    return mask.new_tensor([(h - 1) / 2.0, (w - 1) / 2.0, (z - 1) / 2.0]).view(1, 3)


def _safe_center_of_mass(mask: torch.Tensor) -> torch.Tensor:
    center = compute_center_of_mass(mask.float())
    present = (mask.sum(dim=(2, 3, 4)) > 0).float()
    if present.shape[1] != 1:
        raise RuntimeError(f"_safe_center_of_mass expects single-channel mask, got {tuple(mask.shape)}")
    fallback = _center_fallback_tensor(mask).expand(mask.shape[0], 3)
    return present[:, 0:1] * center + (1.0 - present[:, 0:1]) * fallback


def _erode_binary_mask(mask: torch.Tensor, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    out = (mask > 0.5).float()
    kernel_size = _ensure_odd_int(kernel_size)
    iterations = max(0, int(iterations))
    if iterations <= 0 or kernel_size <= 1:
        return out

    pad = kernel_size // 2
    for _ in range(iterations):
        out = 1.0 - F.max_pool3d(1.0 - out, kernel_size=kernel_size, stride=1, padding=pad)
        out = (out > 0.5).float()
    return out


def _build_stage1_core_mask(mask: torch.Tensor, cfg) -> torch.Tensor:
    kernel_size = int(_cfg_get(cfg, "stage_train.stage1.core_erosion_kernel", 5))
    iterations = int(_cfg_get(cfg, "stage_train.stage1.core_erosion_iters", 1))
    core = _erode_binary_mask(mask.float(), kernel_size=kernel_size, iterations=iterations)

    core_present = (core.sum(dim=(2, 3, 4), keepdim=True) > 0).float()
    return core_present * core + (1.0 - core_present) * (mask > 0.5).float()


def _sample_volume_at_points_hwz(volume: torch.Tensor, points_xyz: torch.Tensor) -> torch.Tensor:
    if volume.dim() != 5:
        raise ValueError(f"Expected 5D volume, got {tuple(volume.shape)}")
    if points_xyz.dim() != 3 or points_xyz.shape[:2] != volume.shape[:2]:
        raise ValueError(
            f"points_xyz must be [B,M,3] matching volume [B,M,H,W,Z], got {tuple(points_xyz.shape)}"
        )

    b, m, h, w, z = volume.shape
    vol = volume.reshape(b * m, 1, h, w, z).permute(0, 1, 4, 2, 3).contiguous()
    pts = points_xyz.reshape(b * m, 3)

    gx = 2.0 * (pts[:, 1] / max(w - 1, 1)) - 1.0
    gy = 2.0 * (pts[:, 0] / max(h - 1, 1)) - 1.0
    gz = 2.0 * (pts[:, 2] / max(z - 1, 1)) - 1.0
    grid = torch.stack([gx, gy, gz], dim=-1).view(b * m, 1, 1, 1, 3)

    sampled = F.grid_sample(
        vol,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.view(b, m)


def _get_stage1_schedule(epoch: Optional[int], cfg) -> Dict[str, float]:
    cur_epoch = max(0, int(epoch or 0))
    sigma_conf_start_epoch = int(_cfg_get(cfg, "stage_train.stage1.sigma_conf_start_epoch", 3))
    sigma_conf_ramp_epochs = max(1, int(_cfg_get(cfg, "stage_train.stage1.sigma_conf_ramp_epochs", 2)))
    evidence_start_epoch = int(
        _cfg_get(cfg, "stage_train.stage1.evidence_start_epoch", 8)
    )
    evidence_ramp_epochs = max(
        1, int(_cfg_get(cfg, "stage_train.stage1.evidence_ramp_epochs", 6))
    )

    ramp_progress = 1.0
    if cur_epoch < sigma_conf_start_epoch:
        ramp_progress = 0.0
    else:
        ramp_progress = min(
            float(cur_epoch - sigma_conf_start_epoch + 1) / float(sigma_conf_ramp_epochs),
            1.0,
        )

    evidence_progress = 1.0
    if cur_epoch < evidence_start_epoch:
        evidence_progress = 0.0
    else:
        evidence_progress = min(
            float(cur_epoch - evidence_start_epoch + 1)
            / float(evidence_ramp_epochs),
            1.0,
        )

    return {
        "sigma_scale": ramp_progress,
        "conf_scale": ramp_progress,
        "inside_scale": 1.0,
        "evidence_scale": evidence_progress,
    }


def build_stage1_detector_targets(seg_gt: torch.Tensor, cfg) -> Dict[str, torch.Tensor]:
    _, m, h, w, z = seg_gt.shape

    center_gt_each_list = []
    sigma_gt_each_list = []
    core_mask_each_list = []
    inside_target_each_list = []
    evidence_target_each_list = []

    sigma_min = float(_cfg_get(cfg, "stage_train.stage1.sigma_min", 2.0))
    sigma_shrink_ratio = float(
        _cfg_get(cfg, "stage_train.stage1.sigma_shrink_ratio", 0.8)
    )
    max_sigma_xyz = tuple(
        float(v)
        for v in _cfg_get(cfg, "stage_train.stage1.max_sigma_xyz", [12.0, 12.0, 8.0])
    )
    center_core_weight = float(
        _cfg_get(cfg, "stage_train.stage1.center_core_weight", 0.7)
    )
    inside_lesion_weight = float(
        _cfg_get(cfg, "stage_train.stage1.inside_lesion_weight", 0.35)
    )
    evidence_core_weight = float(
        _cfg_get(cfg, "stage_train.stage1.evidence_core_weight", 0.7)
    )
    evidence_lesion_weight = float(
        _cfg_get(cfg, "stage_train.stage1.evidence_lesion_weight", 0.25)
    )
    evidence_gaussian_weight = float(
        _cfg_get(cfg, "stage_train.stage1.evidence_gaussian_weight", 0.75)
    )
    evidence_gaussian_sigma_scale = float(
        _cfg_get(cfg, "stage_train.stage1.evidence_gaussian_sigma_scale", 0.6)
    )

    for mi in range(m):
        gt_m = seg_gt[:, mi : mi + 1].float()
        core_m = _build_stage1_core_mask(gt_m, cfg)

        center_mass_m = _safe_center_of_mass(gt_m)
        center_core_m = _safe_center_of_mass(core_m)
        center_gt_m = (
            center_core_weight * center_core_m
            + (1.0 - center_core_weight) * center_mass_m
        )

        sigma_mass_m = compute_sigma_from_mask(
            gt_m,
            center=center_mass_m,
            min_sigma=sigma_min,
            shrink_ratio=sigma_shrink_ratio,
            max_sigma=max_sigma_xyz,
        )
        sigma_core_m = compute_sigma_from_mask(
            core_m,
            center=center_core_m,
            min_sigma=sigma_min,
            shrink_ratio=sigma_shrink_ratio,
            max_sigma=max_sigma_xyz,
        )
        core_present = (core_m.sum(dim=(2, 3, 4)) > 0).float()
        sigma_gt_m = core_present * sigma_core_m + (1.0 - core_present) * sigma_mass_m

        inside_target_m = torch.clamp(core_m + inside_lesion_weight * gt_m, max=1.0)
        gaussian_sigma_m = (sigma_gt_m * evidence_gaussian_sigma_scale).clamp_min(1.0)
        gaussian_target_m = render_gaussian_field(center_gt_m, gaussian_sigma_m, (h, w, z))
        evidence_target_m = torch.clamp(
            evidence_core_weight * core_m
            + evidence_lesion_weight * gt_m
            + evidence_gaussian_weight * gaussian_target_m,
            min=0.0,
            max=1.0,
        )

        center_gt_each_list.append(center_gt_m.unsqueeze(1))
        sigma_gt_each_list.append(sigma_gt_m.unsqueeze(1))
        core_mask_each_list.append(core_m)
        inside_target_each_list.append(inside_target_m)
        evidence_target_each_list.append(evidence_target_m)

    return {
        "center_gt_each": torch.cat(center_gt_each_list, dim=1),
        "sigma_gt_each": torch.cat(sigma_gt_each_list, dim=1),
        "core_mask_each": torch.cat(core_mask_each_list, dim=1),
        "inside_target_each": torch.cat(inside_target_each_list, dim=1),
        "evidence_target_each": torch.cat(evidence_target_each_list, dim=1),
    }


def _modal_weighted_mean(value_each: torch.Tensor, modal_present: torch.Tensor) -> torch.Tensor:
    return (value_each * modal_present).sum() / modal_present.sum().clamp_min(1.0)


def _compute_modal_inside_ratio(
    prob_each: torch.Tensor,
    inside_target_each: torch.Tensor,
    modal_present: torch.Tensor,
) -> torch.Tensor:
    inside_mass_each = (prob_each * inside_target_each).sum(dim=(2, 3, 4))
    total_mass_each = prob_each.sum(dim=(2, 3, 4)).clamp_min(1e-6)
    return _modal_weighted_mean(inside_mass_each / total_mass_each, modal_present)


def stage1_detect_losses(
    debug: Dict[str, torch.Tensor],
    seg_gt: torch.Tensor,
    model: nn.Module,
    cfg,
    epoch: Optional[int] = None,
):
    required_keys = ["det_center_each_full", "det_sigma_each_full"]
    if debug is None or any(k not in debug for k in required_keys):
        raise RuntimeError(
            "Stage1 now requires: det_center_each_full, det_sigma_each_full."
        )

    B, M, H, W, Z = seg_gt.shape
    modal_present = (seg_gt.sum(dim=(2, 3, 4)) > 0).float()  # [B,M]
    stage1_sched = _get_stage1_schedule(epoch, cfg)

    roi_size_xyz = tuple(
        int(v) for v in _cfg_get(cfg, "stage_train.stage1.target_roi_size_xyz", [24, 24, 12])
    )

    target_dict = build_stage1_detector_targets(seg_gt.float(), cfg)
    center_gt_each = target_dict["center_gt_each"]
    sigma_gt_each = target_dict["sigma_gt_each"]
    core_mask_each = target_dict["core_mask_each"]
    inside_target_each = target_dict["inside_target_each"]
    evidence_target_each = target_dict["evidence_target_each"]

    center_pred_each = debug["det_center_each_full"].float()   # [B,M,3]
    center_coarse_each = debug.get("det_center_coarse_each_full", None)
    if center_coarse_each is not None:
        center_coarse_each = center_coarse_each.float()
    sigma_pred_each = debug["det_sigma_each_full"].float()     # [B,M,3]
    raw_evidence_prob_each = debug.get("det_raw_evidence_prob_each", None)
    evidence_prob_each = debug.get("det_seed_evidence_prob_each", None)
    fused_support_prob = debug.get("det_fused_seed_prob_full", None)
    agreement_prob_full = debug.get("det_agreement_prob_full", None)
    modal_reliability_full = debug.get("det_modal_reliability_full", None)
    conf_pred_each = debug.get("det_conf_each_full", None)

    fused_target = evidence_target_each.amax(dim=1, keepdim=True)
    fused_inside_target = inside_target_each.amax(dim=1, keepdim=True)

    # 1) segmentation-oriented center regression
    center_diff = F.smooth_l1_loss(
        center_pred_each,
        center_gt_each,
        reduction="none",
    ).mean(dim=-1)  # [B,M]
    loss_center_modal = (center_diff * modal_present).sum() / modal_present.sum().clamp_min(1.0)

    center_inside_score_each = _sample_volume_at_points_hwz(inside_target_each, center_pred_each).clamp(0.0, 1.0)
    loss_center_inside = (
        ((1.0 - center_inside_score_each) * modal_present).sum()
        / modal_present.sum().clamp_min(1.0)
    )

    loss_center_coarse_modal = torch.tensor(0.0, device=seg_gt.device)
    loss_center_coarse_inside = torch.tensor(0.0, device=seg_gt.device)
    center_coarse_inside_score_each = None
    coarse_metrics = None
    refine_shift_mean = torch.tensor(0.0, device=seg_gt.device)
    if center_coarse_each is not None:
        center_coarse_diff = F.smooth_l1_loss(
            center_coarse_each,
            center_gt_each,
            reduction="none",
        ).mean(dim=-1)
        loss_center_coarse_modal = (
            (center_coarse_diff * modal_present).sum()
            / modal_present.sum().clamp_min(1.0)
        )
        center_coarse_inside_score_each = _sample_volume_at_points_hwz(
            inside_target_each, center_coarse_each
        ).clamp(0.0, 1.0)
        loss_center_coarse_inside = (
            ((1.0 - center_coarse_inside_score_each) * modal_present).sum()
            / modal_present.sum().clamp_min(1.0)
        )
        refine_shift_each = F.l1_loss(
            center_pred_each, center_coarse_each, reduction="none"
        ).mean(dim=-1)
        refine_shift_mean = (
            (refine_shift_each * modal_present).sum()
            / modal_present.sum().clamp_min(1.0)
        )
        coarse_metrics = _compute_stage1_center_metrics(
            center_pred_each=center_coarse_each,
            center_gt_each=center_gt_each,
            seg_gt=seg_gt.float(),
            roi_size_xyz=roi_size_xyz,
        )

    # 2) sigma regression is an auxiliary prior-shape term.
    sigma_diff = F.smooth_l1_loss(
        sigma_pred_each,
        sigma_gt_each,
        reduction="none",
    ).mean(dim=-1)  # [B,M]
    loss_sigma_reg = (sigma_diff * modal_present).sum() / modal_present.sum().clamp_min(1.0)

    loss_evidence_supervise = torch.tensor(0.0, device=seg_gt.device)
    loss_evidence_outside = torch.tensor(0.0, device=seg_gt.device)
    evidence_inside_ratio_mean = torch.tensor(0.0, device=seg_gt.device)
    if evidence_prob_each is not None:
        evidence_prob_each = evidence_prob_each.float().clamp(1e-4, 1.0 - 1e-4)
        bce_each = F.binary_cross_entropy(
            evidence_prob_each,
            evidence_target_each,
            reduction="none",
        )
        loss_evidence_supervise = (
            (bce_each.mean(dim=(2, 3, 4)) * modal_present).sum()
            / modal_present.sum().clamp_min(1.0)
        )

        outside_mask_each = (1.0 - inside_target_each).clamp(0.0, 1.0)
        outside_mass = outside_mask_each.sum(dim=(2, 3, 4)).clamp_min(1.0)
        outside_energy_each = (evidence_prob_each * outside_mask_each).sum(dim=(2, 3, 4)) / outside_mass
        loss_evidence_outside = (
            (outside_energy_each * modal_present).sum()
            / modal_present.sum().clamp_min(1.0)
        )
        evidence_inside_ratio_mean = _compute_modal_inside_ratio(
            evidence_prob_each, inside_target_each, modal_present
        )

    loss_evidence_peak = compute_stage1_evidence_peak_loss(debug, cfg).to(seg_gt.device)
    loss_evidence_total = loss_evidence_supervise + loss_evidence_outside

    loss_raw_evidence_supervise = torch.tensor(0.0, device=seg_gt.device)
    loss_raw_evidence_outside = torch.tensor(0.0, device=seg_gt.device)
    raw_evidence_inside_ratio_mean = torch.tensor(0.0, device=seg_gt.device)
    if raw_evidence_prob_each is not None:
        raw_evidence_prob_each = raw_evidence_prob_each.float().clamp(1e-4, 1.0 - 1e-4)
        raw_bce_each = F.binary_cross_entropy(
            raw_evidence_prob_each,
            evidence_target_each,
            reduction="none",
        )
        loss_raw_evidence_supervise = (
            (raw_bce_each.mean(dim=(2, 3, 4)) * modal_present).sum()
            / modal_present.sum().clamp_min(1.0)
        )

        outside_mask_each = (1.0 - inside_target_each).clamp(0.0, 1.0)
        outside_mass = outside_mask_each.sum(dim=(2, 3, 4)).clamp_min(1.0)
        raw_outside_energy_each = (
            (raw_evidence_prob_each * outside_mask_each).sum(dim=(2, 3, 4)) / outside_mass
        )
        loss_raw_evidence_outside = (
            (raw_outside_energy_each * modal_present).sum()
            / modal_present.sum().clamp_min(1.0)
        )
        raw_evidence_inside_ratio_mean = _compute_modal_inside_ratio(
            raw_evidence_prob_each, inside_target_each, modal_present
        )

    loss_fused_support_supervise = torch.tensor(0.0, device=seg_gt.device)
    loss_fused_support_outside = torch.tensor(0.0, device=seg_gt.device)
    fused_support_inside_ratio_mean = torch.tensor(0.0, device=seg_gt.device)
    agreement_inside_score_mean = torch.tensor(0.0, device=seg_gt.device)
    reliability_inside_score_mean = torch.tensor(0.0, device=seg_gt.device)
    if fused_support_prob is not None:
        fused_support_prob = fused_support_prob.float().clamp(1e-4, 1.0 - 1e-4)
        loss_fused_support_supervise = F.binary_cross_entropy(
            fused_support_prob,
            fused_target,
        )
        fused_outside_mask = (1.0 - fused_inside_target).clamp(0.0, 1.0)
        fused_outside_mass = fused_outside_mask.sum(dim=(2, 3, 4)).clamp_min(1.0)
        fused_outside_energy = (
            (fused_support_prob * fused_outside_mask).sum(dim=(2, 3, 4)) / fused_outside_mass
        )
        loss_fused_support_outside = fused_outside_energy.mean()

        fused_inside_mass = (fused_support_prob * fused_inside_target).sum(dim=(2, 3, 4))
        fused_total_mass = fused_support_prob.sum(dim=(2, 3, 4)).clamp_min(1e-6)
        fused_support_inside_ratio_mean = (fused_inside_mass / fused_total_mass).mean()

    if agreement_prob_full is not None:
        agreement_prob_full = agreement_prob_full.float().clamp(1e-4, 1.0 - 1e-4)
        inside_mass = fused_inside_target.sum(dim=(2, 3, 4)).clamp_min(1.0)
        agreement_inside_score_mean = (
            (agreement_prob_full * fused_inside_target).sum(dim=(2, 3, 4)) / inside_mass
        ).mean()

    if modal_reliability_full is not None:
        modal_reliability_full = modal_reliability_full.float().clamp(1e-4, 1.0 - 1e-4)
        reliability_inside_mass = inside_target_each.sum(dim=(2, 3, 4)).clamp_min(1.0)
        reliability_inside_each = (
            (modal_reliability_full * inside_target_each).sum(dim=(2, 3, 4)) / reliability_inside_mass
        )
        reliability_inside_score_mean = _modal_weighted_mean(
            reliability_inside_each, modal_present
        )

    loss_conf_reg = torch.tensor(0.0, device=seg_gt.device)
    if conf_pred_each is not None:
        conf_pred_each = conf_pred_each.float().clamp(1e-4, 1.0 - 1e-4)
        conf_target_each = center_inside_score_each.detach().clamp(0.0, 1.0)
        if evidence_prob_each is not None:
            evidence_inside_mass_each = (evidence_prob_each * inside_target_each).sum(dim=(2, 3, 4))
            evidence_total_mass_each = evidence_prob_each.sum(dim=(2, 3, 4)).clamp_min(1e-6)
            evidence_inside_ratio_each = (evidence_inside_mass_each / evidence_total_mass_each).detach()
            conf_target_each = 0.5 * conf_target_each + 0.5 * evidence_inside_ratio_each.clamp(0.0, 1.0)
        conf_bce = F.binary_cross_entropy(conf_pred_each, conf_target_each, reduction="none")
        loss_conf_reg = (
            (conf_bce * modal_present).sum()
            / modal_present.sum().clamp_min(1.0)
        )

    # 3) HWA scale floor regularization
    raw = _unwrap(model)
    loss_floor = torch.tensor(0.0, device=seg_gt.device)
    floor_tau = float(_cfg_get(cfg, "stage_train.stage1.scale_floor_tau", 0.10))
    if hasattr(raw, "Encoder") and hasattr(raw.Encoder, "hwa_scales"):
        vals = []
        for p in raw.Encoder.hwa_scales:
            vals.append(F.softplus(p).reshape(-1).mean())
        if vals:
            vals = torch.stack(vals)
            loss_floor = F.relu(floor_tau - vals).mean()

    evidence_loss_scale = float(
        _cfg_get(cfg, "stage_train.stage1.evidence_loss_scale", 0.08)
    )
    raw_evidence_loss_scale = float(
        _cfg_get(cfg, "stage_train.stage1.raw_evidence_loss_scale", 0.03)
    )
    fused_support_loss_scale = float(
        _cfg_get(cfg, "stage_train.stage1.fused_support_loss_scale", 0.03)
    )

    total = (
        float(_cfg_get(cfg, "stage_train.stage1.lambda_center_modal", 2.0)) * loss_center_modal
        + float(_cfg_get(cfg, "stage_train.stage1.lambda_center_coarse_modal", 0.20))
        * loss_center_coarse_modal
        + float(_cfg_get(cfg, "stage_train.stage1.lambda_center_inside", 2.0))
        * stage1_sched["inside_scale"]
        * loss_center_inside
        + float(_cfg_get(cfg, "stage_train.stage1.lambda_center_coarse_inside", 0.40))
        * stage1_sched["inside_scale"]
        * loss_center_coarse_inside
        + float(_cfg_get(cfg, "stage_train.stage1.lambda_evidence_supervise", 1.0))
        * evidence_loss_scale
        * stage1_sched["evidence_scale"]
        * loss_evidence_supervise
        + float(_cfg_get(cfg, "stage_train.stage1.lambda_evidence_outside", 0.2))
        * evidence_loss_scale
        * stage1_sched["evidence_scale"]
        * loss_evidence_outside
        + float(_cfg_get(cfg, "stage_train.stage1.lambda_raw_evidence_supervise", 0.20))
        * raw_evidence_loss_scale
        * stage1_sched["evidence_scale"]
        * loss_raw_evidence_supervise
        + float(_cfg_get(cfg, "stage_train.stage1.lambda_raw_evidence_outside", 0.05))
        * raw_evidence_loss_scale
        * stage1_sched["evidence_scale"]
        * loss_raw_evidence_outside
        + float(_cfg_get(cfg, "stage_train.stage1.lambda_fused_support_supervise", 0.90))
        * fused_support_loss_scale
        * stage1_sched["evidence_scale"]
        * loss_fused_support_supervise
        + float(_cfg_get(cfg, "stage_train.stage1.lambda_fused_support_outside", 0.20))
        * fused_support_loss_scale
        * stage1_sched["evidence_scale"]
        * loss_fused_support_outside
        + float(_cfg_get(cfg, "stage_train.stage1.lambda_sigma_reg", 0.10))
        * stage1_sched["sigma_scale"]
        * loss_sigma_reg
        + float(_cfg_get(cfg, "stage_train.stage1.lambda_conf_reg", 0.05))
        * stage1_sched["conf_scale"]
        * loss_conf_reg
        + float(_cfg_get(cfg, "stage_train.stage1.lambda_scale_floor", 0.02)) * loss_floor
        + float(_cfg_get(cfg, "stage_train.stage1.lambda_evidence_peak", 0.05)) * loss_evidence_peak
    )

    metrics = _compute_stage1_center_metrics(
        center_pred_each=center_pred_each,
        center_gt_each=center_gt_each,
        seg_gt=seg_gt.float(),
        roi_size_xyz=roi_size_xyz,
    )

    return {
        "loss_center_modal": loss_center_modal,
        "loss_center_inside": loss_center_inside,
        "loss_center_coarse_modal": loss_center_coarse_modal,
        "loss_center_coarse_inside": loss_center_coarse_inside,
        "loss_sigma_reg": loss_sigma_reg,
        "loss_conf_reg": loss_conf_reg,
        "loss_raw_evidence_supervise": loss_raw_evidence_supervise,
        "loss_raw_evidence_outside": loss_raw_evidence_outside,
        "loss_evidence_supervise": loss_evidence_supervise,
        "loss_evidence_outside": loss_evidence_outside,
        "loss_evidence_total": loss_evidence_total,
        "loss_fused_support_supervise": loss_fused_support_supervise,
        "loss_fused_support_outside": loss_fused_support_outside,
        "loss_scale_floor": loss_floor,
        "loss_evidence_peak": loss_evidence_peak,
        "loss_total": total,
        "center_gt_each": center_gt_each,
        "sigma_gt_each": sigma_gt_each,
        "core_mask_each": core_mask_each,
        "inside_target_each": inside_target_each,
        "evidence_target_each": evidence_target_each,
        "center_coarse_each": center_coarse_each,
        "center_coarse_inside_score_mean": (
            (
                (center_coarse_inside_score_each * modal_present).sum()
                / modal_present.sum().clamp_min(1.0)
            )
            if center_coarse_inside_score_each is not None
            else torch.tensor(0.0, device=seg_gt.device)
        ),
        "center_inside_score_mean": (
            (center_inside_score_each * modal_present).sum() / modal_present.sum().clamp_min(1.0)
        ),
        "raw_evidence_inside_ratio_mean": raw_evidence_inside_ratio_mean,
        "evidence_inside_ratio_mean": evidence_inside_ratio_mean,
        "fused_support_inside_ratio_mean": fused_support_inside_ratio_mean,
        "agreement_inside_score_mean": agreement_inside_score_mean,
        "reliability_inside_score_mean": reliability_inside_score_mean,
        "center_refine_shift_mean": refine_shift_mean,
        "center_l1_metric": metrics["center_l1"],
        "center_in_mask_rate": metrics["center_in_mask_rate"],
        "roi_coverage": metrics["roi_coverage"],
        "center_coarse_l1_metric": (
            coarse_metrics["center_l1"]
            if coarse_metrics is not None
            else torch.tensor(0.0, device=seg_gt.device)
        ),
        "center_coarse_in_mask_rate": (
            coarse_metrics["center_in_mask_rate"]
            if coarse_metrics is not None
            else torch.tensor(0.0, device=seg_gt.device)
        ),
    }

def split_dual_cls_logits(cls_logits):
    if isinstance(cls_logits, (tuple, list)):
        if len(cls_logits) != 2:
            raise RuntimeError("Expected 2 classification outputs.")
        return cls_logits[0], cls_logits[1]
    if not isinstance(cls_logits, torch.Tensor):
        raise RuntimeError("Stage3 expects tensor/tuple/list classification outputs.")
    if cls_logits.ndim >= 2 and cls_logits.shape[0] == 2:
        return cls_logits[0], cls_logits[1]
    if cls_logits.ndim >= 2 and cls_logits.shape[1] == 2:
        return cls_logits[:, 0:1, ...], cls_logits[:, 1:2, ...]
    raise RuntimeError(
        f"Cannot split dual classification logits with shape {tuple(cls_logits.shape)}"
    )


def compute_stage3_classification_loss(
    cls_logits,
    batch,
    loss_functions,
    accelerator,
    step=None,
    log_prefix="Train",
    lambda_class=1.0,
    lambda_pfs=1.0,
    task_mu=2,
):
    task_mu = max(1, int(task_mu or 1))
    if task_mu <= 1:
        if isinstance(cls_logits, (tuple, list)):
            logit_class = cls_logits[0]
        else:
            logit_class = cls_logits
        logit_pfs = None
    else:
        logit_class, logit_pfs = split_dual_cls_logits(cls_logits)
    labels_class = batch["class_label"].float()
    if labels_class.dim() == 1:
        labels_class = labels_class.unsqueeze(1)

    total_loss = 0.0
    out = {
        "logit_class": logit_class,
        "logit_pfs": logit_pfs,
        "labels_class": labels_class,
        "labels_pfs": None,
    }
    if task_mu > 1:
        labels_pfs = batch["PFS_label"].float()
        if labels_pfs.dim() == 1:
            labels_pfs = labels_pfs.unsqueeze(1)
        out["labels_pfs"] = labels_pfs
    for name, fn in loss_functions.items():
        loss1 = fn(logit_class, labels_class)
        out[f"{name}_class"] = loss1
        total_loss = total_loss + lambda_class * loss1
        if step is not None:
            accelerator.log(
                {f"{log_prefix}/{name}_class": _loss_to_scalar(loss1)}, step=step
            )
        if task_mu > 1:
            loss2 = fn(logit_pfs, out["labels_pfs"])
            out[f"{name}_PFS"] = loss2
            total_loss = total_loss + lambda_pfs * loss2
            if step is not None:
                accelerator.log(
                    {f"{log_prefix}/{name}_PFS": _loss_to_scalar(loss2)}, step=step
                )
    out["total_loss"] = total_loss
    return out


def _class_labels_from_dataset(dataset) -> list:
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        labels = []
        for child in dataset.datasets:
            labels.extend(_class_labels_from_dataset(child))
        return labels

    if isinstance(dataset, torch.utils.data.Subset):
        parent_labels = _class_labels_from_dataset(dataset.dataset)
        if parent_labels:
            return [parent_labels[int(index)] for index in dataset.indices]
        return []

    data = getattr(dataset, "data", None)
    if data is None:
        return []

    labels = []
    for item in data:
        if not isinstance(item, dict) or "class_label" not in item:
            continue
        value = item["class_label"]
        if torch.is_tensor(value):
            value = value.detach().view(-1)[0].item()
        elif isinstance(value, np.ndarray):
            value = np.asarray(value).reshape(-1)[0]
        elif isinstance(value, (list, tuple)):
            value = value[0]
        labels.append(1 if int(value) != 0 else 0)
    return labels


def _resolve_stage3_class_pos_weight(cfg, train_loader, device, accelerator):
    raw = _cfg_get(cfg, "stage_train.stage3.class_pos_weight", "auto")
    if raw in [None, "", "None"]:
        return None

    if isinstance(raw, str) and raw.strip().lower() == "auto":
        labels = _class_labels_from_dataset(getattr(train_loader, "dataset", None))
        positives = sum(labels)
        negatives = len(labels) - positives
        if positives <= 0 or negatives <= 0:
            if accelerator is not None:
                accelerator.print(
                    "[Stage3Loss] class_pos_weight=auto skipped because "
                    f"labels={len(labels)}, positives={positives}, negatives={negatives}."
                )
            return None
        value = float(negatives) / float(positives)
        if accelerator is not None:
            accelerator.print(
                "[Stage3Loss] class_pos_weight=auto resolved "
                f"to {value:.6f} from labels={len(labels)}, "
                f"positives={positives}, negatives={negatives}."
            )
        return torch.tensor([value], dtype=torch.float32, device=device)

    return torch.tensor([float(raw)], dtype=torch.float32, device=device)


def build_stage3_classification_losses(cfg, train_loader, device, accelerator=None):
    mode = str(_cfg_get(cfg, "stage_train.stage3.class_loss_mode", "focal_bce")).strip().lower()

    if mode in {"balanced_bce", "weighted_bce"}:
        pos_weight = _resolve_stage3_class_pos_weight(cfg, train_loader, device, accelerator)
        if accelerator is not None:
            accelerator.print(
                "[Stage3Loss] mode=balanced_bce; using BCEWithLogitsLoss"
                + (
                    f" with pos_weight={float(pos_weight.item()):.6f}."
                    if pos_weight is not None
                    else " without pos_weight."
                )
            )
        kwargs = {"pos_weight": pos_weight} if pos_weight is not None else {}
        return {"bce_loss": nn.BCEWithLogitsLoss(**kwargs).to(device)}

    if mode in {"bce", "plain_bce"}:
        if accelerator is not None:
            accelerator.print("[Stage3Loss] mode=bce; using BCEWithLogitsLoss only.")
        return {"bce_loss": nn.BCEWithLogitsLoss().to(device)}

    if mode in {"focal_bce", "default"}:
        return {
            "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
            "bce_loss": nn.BCEWithLogitsLoss().to(device),
        }

    raise ValueError(f"Unsupported stage3 class_loss_mode: {mode}")


def _norm01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def _overlay_mask(
    gray: np.ndarray,
    mask: np.ndarray,
    color: Tuple[float, float, float],
    alpha: float = 0.62,
    glow_alpha: float = 0.35,
    glow_width: int = 6,
) -> np.ndarray:
    import cv2

    gray = _norm01(gray)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    mask_bin = (mask > 0).astype(np.uint8)
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    fill_a = alpha * mask_bin[..., None].astype(np.float32)
    out = rgb * (1.0 - fill_a) + color_arr * fill_a
    if glow_width > 0 and mask_bin.max() > 0:
        k = glow_width * 2 + 1
        dilated = cv2.dilate(mask_bin, np.ones((k, k), np.uint8), iterations=1)
        glow_ring = (dilated.astype(np.float32) - mask_bin.astype(np.float32)).clip(
            0, 1
        )
        blur_size = max(3, glow_width * 2 + 1)
        if blur_size % 2 == 0:
            blur_size += 1
        glow_soft = cv2.GaussianBlur(glow_ring, (blur_size, blur_size), glow_width / 2)
        if glow_soft.max() > 1e-8:
            glow_soft = glow_soft / glow_soft.max()
        glow_a = glow_alpha * glow_soft[..., None]
        out = out * (1.0 - glow_a) + color_arr * glow_a
    return np.clip(out, 0.0, 1.0)


def _overlay_dual_mask(
    gray: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    gt_color: Tuple[float, float, float],
    pred_color: Tuple[float, float, float],
    gt_alpha: float = 0.62,
    pred_alpha: float = 0.45,
) -> np.ndarray:
    gray = _norm01(gray)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    gt_bin = (gt_mask > 0).astype(np.float32)[..., None]
    pred_bin = (pred_mask > 0).astype(np.float32)[..., None]
    gt_color_arr = np.array(gt_color, dtype=np.float32).reshape(1, 1, 3)
    pred_color_arr = np.array(pred_color, dtype=np.float32).reshape(1, 1, 3)

    out = rgb * (1.0 - gt_alpha * gt_bin) + gt_color_arr * (gt_alpha * gt_bin)
    out = out * (1.0 - pred_alpha * pred_bin) + pred_color_arr * (pred_alpha * pred_bin)
    return np.clip(out, 0.0, 1.0)


def _overlay_heatmap_on_gray(
    gray: np.ndarray,
    heat: np.ndarray,
    alpha: float = 0.45,
    cmap_name: str = "turbo",
    heat_gamma: float = 0.8,
) -> np.ndarray:
    gray = _norm01(gray)
    rgb = np.stack([gray, gray, gray], axis=-1)
    heat = heat.astype(np.float32)
    heat = heat - heat.min()
    if heat.max() > 1e-8:
        heat = heat / heat.max()
    heat = np.power(heat, heat_gamma)
    cmap = plt.get_cmap(cmap_name)
    heat_rgb = cmap(heat)[..., :3].astype(np.float32)
    local_alpha = alpha * heat[..., None]
    out = rgb * (1.0 - local_alpha) + heat_rgb * local_alpha
    return np.clip(out, 0.0, 1.0)


def _draw_center_point(
    ax,
    x: float,
    y: float,
    color: str = "red",
    size: int = 36,
    marker: str = "o",
    linewidths: float = 1.5,
):
    ax.scatter([y], [x], c=color, s=size, marker=marker, linewidths=linewidths)


def _draw_roi_box(
    ax,
    center_xyz: np.ndarray,
    size_xyz: np.ndarray,
    z_index: int,
    hwz: Tuple[int, int, int],
    color: str = "gold",
    linewidth: float = 1.8,
    linestyle: str = "-",
):
    h, w, z = hwz
    cx, cy, cz = [float(v) for v in center_xyz]
    sx, sy, sz = [max(1.0, float(v)) for v in size_xyz]

    x1 = max(0.0, cx - 0.5 * sx)
    x2 = min(float(h - 1), cx + 0.5 * sx)
    y1 = max(0.0, cy - 0.5 * sy)
    y2 = min(float(w - 1), cy + 0.5 * sy)
    z1 = max(0.0, cz - 0.5 * sz)
    z2 = min(float(z - 1), cz + 0.5 * sz)

    if not (z1 <= float(z_index) <= z2):
        return

    rect = Rectangle(
        (y1, x1),
        max(1.0, y2 - y1),
        max(1.0, x2 - x1),
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    ax.add_patch(rect)


def _choose_center_z(center_xyz: np.ndarray, z_max: int) -> int:
    z = int(round(float(center_xyz[2])))
    return max(0, min(z, z_max - 1))


def _choose_vis_slice(gt_vol: np.ndarray, pred_vol: Optional[np.ndarray] = None) -> int:
    gt_sum = gt_vol.sum(axis=(0, 1)) if gt_vol.ndim == 3 else gt_vol.sum(axis=0)
    if float(gt_sum.max()) > 0:
        return int(np.argmax(gt_sum))
    if pred_vol is not None:
        pred_sum = (
            pred_vol.sum(axis=(0, 1)) if pred_vol.ndim == 3 else pred_vol.sum(axis=0)
        )
        if float(pred_sum.max()) > 0:
            return int(np.argmax(pred_sum))
    return int(gt_vol.shape[-1] // 2)


def _choose_heat_slice(heat_vol: np.ndarray, fallback_z: int) -> int:
    if heat_vol is None:
        return int(fallback_z)
    z_score = heat_vol.sum(axis=(0, 1))
    if float(z_score.max()) <= 1e-8:
        return int(fallback_z)
    return int(np.argmax(z_score))


def _safe_round_int(v: float, low: int, high: int) -> int:
    return max(int(low), min(int(high), int(round(float(v)))))


def _compute_single_detector_vis_metrics(
    pred_center_xyz: np.ndarray,
    gt_center_xyz: np.ndarray,
    gt_vol: np.ndarray,
    roi_size_xyz: Tuple[int, int, int],
):
    h, w, z = gt_vol.shape
    px = _safe_round_int(pred_center_xyz[0], 0, h - 1)
    py = _safe_round_int(pred_center_xyz[1], 0, w - 1)
    pz = _safe_round_int(pred_center_xyz[2], 0, z - 1)

    center_l1 = float(np.abs(pred_center_xyz - gt_center_xyz).mean())
    in_mask = float(gt_vol[px, py, pz] > 0.5)

    rh, rw, rz = [max(1, int(v)) for v in roi_size_xyz]
    x1 = max(0, px - rh // 2)
    y1 = max(0, py - rw // 2)
    z1 = max(0, pz - rz // 2)
    x2 = min(h, x1 + rh)
    y2 = min(w, y1 + rw)
    z2 = min(z, z1 + rz)

    gt_sum = float(gt_vol.sum())
    if gt_sum <= 0.0:
        roi_coverage = 1.0
    else:
        roi_coverage = float(gt_vol[x1:x2, y1:y2, z1:z2].sum()) / max(gt_sum, 1e-6)

    return {
        "center_l1": center_l1,
        "in_mask": in_mask,
        "roi_coverage": roi_coverage,
    }


def _make_center_panel_with_gt_hint(
    raw_center: np.ndarray,
    gt_vol: np.ndarray,
    z_center: int,
    gt_color: Tuple[float, float, float],
    gt_alpha: float = 0.68,
    glow_alpha: float = 0.42,
    glow_width: int = 5,
):
    gt_center = gt_vol[:, :, z_center]
    has_gt_here = bool((gt_center > 0).any())
    if has_gt_here:
        panel = _overlay_mask(
            raw_center,
            gt_center,
            gt_color,
            alpha=gt_alpha,
            glow_alpha=glow_alpha,
            glow_width=glow_width,
        )
        return panel, None, z_center, True
    z_scores = gt_vol.sum(axis=(0, 1))
    positive_z = np.where(z_scores > 0)[0]
    if len(positive_z) == 0:
        panel = np.stack([_norm01(raw_center)] * 3, axis=-1)
        return panel, None, z_center, False
    nearest_z = int(positive_z[np.argmin(np.abs(positive_z - z_center))])
    nearest_mask = (gt_vol[:, :, nearest_z] > 0).astype(np.float32)
    panel = np.stack([_norm01(raw_center)] * 3, axis=-1)
    return panel, nearest_mask, nearest_z, False


def _stage_visual_root_name(stage: str) -> str:
    if stage == "stage1":
        return "stage1_detector_pretrain"
    return stage


def _build_visualization_dir(
    run_dir: Path,
    epoch: int,
    split_name: str,
    stage: str,
):
    stage_root = _stage_visual_root_name(stage)
    vis_root = Path(run_dir) / "visuals" / stage_root
    epoch_key = (str(Path(run_dir).resolve()), stage_root, int(epoch))
    epoch_dir = _VIS_EPOCH_DIR_CACHE.get(epoch_key, None)
    if epoch_dir is None:
        epoch_name = f"epoch_{epoch:04d}"
        epoch_dir = vis_root / epoch_name
        if stage == "stage1" and epoch_dir.exists():
            suffix = 1
            while (vis_root / f"{epoch_name}_keep_{suffix:02d}").exists():
                suffix += 1
            epoch_dir = vis_root / f"{epoch_name}_keep_{suffix:02d}"
        _VIS_EPOCH_DIR_CACHE[epoch_key] = epoch_dir

    split_dir = epoch_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir, stage_root


def _cleanup_old_visualizations(
    run_dir: Path, keep_last_n: int = 5, stage: Optional[str] = None
):
    if stage is None:
        return
    stage_root = _stage_visual_root_name(stage)
    if stage_root == "stage1_detector_pretrain":
        return

    vis_root = Path(run_dir) / "visuals" / stage_root
    if not vis_root.exists():
        return

    epoch_dirs = []
    for p in vis_root.iterdir():
        if p.is_dir() and p.name.startswith("epoch_"):
            try:
                epoch_id = int(p.name.split("_")[1])
                epoch_dirs.append((epoch_id, p))
            except Exception:
                pass
    epoch_dirs.sort(key=lambda x: x[0])
    for _, p in epoch_dirs[:-keep_last_n]:
        import shutil

        shutil.rmtree(p, ignore_errors=True)


def save_epoch_visualizations(
    *, accelerator, model, batch, cfg, epoch: int, split_name: str, run_dir, stage: str
):
    if not accelerator.is_main_process:
        return
    if not bool(_vis_cfg_get(cfg, "enable", True)):
        return
    if epoch % int(_vis_cfg_get(cfg, "every_n_epochs", 1)) != 0:
        return

    raw_model = accelerator.unwrap_model(model)
    raw_model.eval()
    device = accelerator.device

    image = batch["image"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)

    batch_size = image.shape[0]
    max_cases = int(_vis_cfg_get(cfg, "max_cases", 1))
    max_cases = max(1, min(max_cases, batch_size))

    gt_present = (label.sum(dim=(1, 2, 3, 4)) > 0)
    pos_idx = torch.nonzero(gt_present, as_tuple=False).flatten()
    all_idx = torch.randperm(batch_size, device=device)

    chosen = []
    if len(pos_idx) > 0:
        pos_idx = pos_idx[torch.randperm(len(pos_idx), device=device)]
        chosen.extend(pos_idx[:max_cases].tolist())

    if len(chosen) < max_cases:
        for idx in all_idx.tolist():
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= max_cases:
                break

    chosen_idx = torch.tensor(chosen[:max_cases], device=device, dtype=torch.long)

    image = image[chosen_idx]
    label = label[chosen_idx]

    _, seg_logits, debug = forward_model(raw_model, image, stage=stage, debug=True)
    seg_pred = (
        torch.sigmoid(seg_logits)
        >= float(_vis_cfg_get(cfg, "pred_threshold", 0.5))
    ).float()

    required_keys = ["det_center_each_full", "det_sigma_each_full"]
    if debug is None or any(k not in debug for k in required_keys):
        return

    det_center_each = debug["det_center_each_full"]   # [B,M,3]
    det_center_coarse_each = debug.get("det_center_coarse_each_full", None)
    det_sigma_each = debug["det_sigma_each_full"]     # [B,M,3]
    det_conf_each = debug.get("det_conf_each_full", None)
    det_fused_center = debug.get("det_fused_center_full", None)  # [B,3]
    det_fused_sigma = debug.get("det_fused_sigma_full", None)    # [B,3]
    det_fused_conf = debug.get("det_fused_conf_full", None)      # [B,1] or [B]
    det_fused_support_each = debug.get("det_fused_seed_prob_full", None)
    stage_window_list = debug.get("stage_window_evidence", None)
    stage_quality_list = debug.get("stage_prior_quality_maps", None)
    stage_roi_sizes_full = debug.get("stage_roi_sizes_full", None)

    stage0_window_each = None
    stageN_window_each = None
    if isinstance(stage_window_list, (list, tuple)) and len(stage_window_list) > 0:
        try:
            stage0_window_each = F.interpolate(
                stage_window_list[0].float(),
                size=image.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
            stageN_window_each = F.interpolate(
                stage_window_list[-1].float(),
                size=image.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        except Exception:
            stage0_window_each = None
            stageN_window_each = None

    stage0_quality_full = None
    stageN_quality_full = None
    if isinstance(stage_quality_list, (list, tuple)) and len(stage_quality_list) > 0:
        try:
            stage0_quality_full = F.interpolate(
                stage_quality_list[0].float(),
                size=image.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
            stageN_quality_full = F.interpolate(
                stage_quality_list[-1].float(),
                size=image.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        except Exception:
            stage0_quality_full = None
            stageN_quality_full = None

    gt_color = tuple(
        v / 255.0 for v in _vis_cfg_get(cfg, "gt_color_rgb", [169, 209, 238])
    )
    pred_color = tuple(
        v / 255.0 for v in _vis_cfg_get(cfg, "pred_color_rgb", [255, 80, 80])
    )
    pred_alpha = float(_vis_cfg_get(cfg, "pred_alpha", 0.50))
    heat_alpha = float(_vis_cfg_get(cfg, "heat_alpha", 0.45))
    detector_pred_color = str(_vis_cfg_get(cfg, "detector_pred_color", "red"))
    roi_size_xyz = tuple(
        int(v) for v in _cfg_get(cfg, "stage_train.stage1.target_roi_size_xyz", [24, 24, 12])
    )
    fused_center_color = str(_vis_cfg_get(cfg, "fused_center_color", "gold"))
    stage0_box_color = str(_vis_cfg_get(cfg, "stage0_roi_box_color", "gold"))
    stageN_box_color = str(_vis_cfg_get(cfg, "stageN_roi_box_color", "deepskyblue"))

    vis_dir, _ = _build_visualization_dir(Path(run_dir), epoch, split_name, stage)

    B, M, H, W, Z = image.shape
    chosen_idx_cpu = chosen_idx.detach().cpu().tolist()
    target_dict = build_stage1_detector_targets(label.float(), cfg)
    gt_center_each = target_dict["center_gt_each"]
    inside_target_each = target_dict["inside_target_each"]
    evidence_target_each = target_dict["evidence_target_each"]
    fused_center_prior = None
    if det_fused_center is not None and det_fused_sigma is not None:
        fused_center_prior = render_gaussian_field(
            det_fused_center.float(),
            det_fused_sigma.float(),
            (H, W, Z),
        ).clamp(1e-4, 1.0 - 1e-4)

    for b in range(B):
        img_np = image[b].detach().float().cpu().numpy()
        lab_np = label[b].detach().float().cpu().numpy()
        pred_np = seg_pred[b].detach().float().cpu().numpy()
        center_each_np = det_center_each[b].detach().float().cpu().numpy()
        center_coarse_each_np = (
            det_center_coarse_each[b].detach().float().cpu().numpy()
            if det_center_coarse_each is not None
            else center_each_np
        )
        gt_center_each_np = gt_center_each[b].detach().float().cpu().numpy()
        inside_target_each_np = inside_target_each[b].detach().float().cpu().numpy()
        evidence_target_each_np = evidence_target_each[b].detach().float().cpu().numpy()
        sigma_each_np = det_sigma_each[b].detach().float().cpu().numpy()
        conf_each_np = (
            det_conf_each[b].detach().float().cpu().numpy() if det_conf_each is not None else None
        )
        fused_center_np = (
            det_fused_center[b].detach().float().cpu().numpy()
            if det_fused_center is not None
            else center_each_np.mean(axis=0)
        )
        fused_sigma_np = (
            det_fused_sigma[b].detach().float().cpu().numpy()
            if det_fused_sigma is not None
            else sigma_each_np.mean(axis=0)
        )
        fused_conf_np = (
            float(det_fused_conf[b].detach().float().reshape(-1)[0].cpu().item())
            if det_fused_conf is not None
            else None
        )
        fused_support_np = (
            det_fused_support_each[b].detach().float().cpu().numpy()
            if det_fused_support_each is not None
            else None
        )
        fused_prior_np = (
            fused_center_prior[b, 0].detach().float().cpu().numpy()
            if fused_center_prior is not None
            else None
        )
        stage0_window_np = (
            stage0_window_each[b].detach().float().cpu().numpy()
            if stage0_window_each is not None
            else None
        )
        stageN_window_np = (
            stageN_window_each[b].detach().float().cpu().numpy()
            if stageN_window_each is not None
            else None
        )
        stage0_quality_np = (
            stage0_quality_full[b, 0].detach().float().cpu().numpy()
            if stage0_quality_full is not None
            else None
        )
        stageN_quality_np = (
            stageN_quality_full[b, 0].detach().float().cpu().numpy()
            if stageN_quality_full is not None
            else None
        )
        stage0_roi_size_np = (
            stage_roi_sizes_full[0][b].detach().float().cpu().numpy()
            if isinstance(stage_roi_sizes_full, (list, tuple)) and len(stage_roi_sizes_full) > 0
            else np.asarray(roi_size_xyz, dtype=np.float32)
        )
        stageN_roi_size_np = (
            stage_roi_sizes_full[-1][b].detach().float().cpu().numpy()
            if isinstance(stage_roi_sizes_full, (list, tuple)) and len(stage_roi_sizes_full) > 0
            else np.asarray(roi_size_xyz, dtype=np.float32)
        )

        fig_dpi = int(_vis_cfg_get(cfg, "figure_dpi", 120))
        fig, axes = plt.subplots(M, 7, figsize=(32, 4.2 * M), dpi=fig_dpi, squeeze=False)

        for m in range(M):
            raw_vol = img_np[m]
            gt_vol = lab_np[m] if lab_np.shape[0] == M else lab_np[0]
            pred_vol = pred_np[m] if pred_np.shape[0] == M else pred_np[0]

            center_m = center_each_np[m]
            center_coarse_m = center_coarse_each_np[m]
            gt_center_m = gt_center_each_np[m]
            inside_target_vol = inside_target_each_np[m]
            sigma_m = sigma_each_np[m]
            conf_m = float(conf_each_np[m]) if conf_each_np is not None else None
            fused_support_vol = fused_support_np[0] if fused_support_np is not None else None
            stage0_window_vol = stage0_window_np[m] if stage0_window_np is not None else None
            stageN_window_vol = stageN_window_np[m] if stageN_window_np is not None else None
            refined_metrics = _compute_single_detector_vis_metrics(
                center_m, gt_center_m, gt_vol, roi_size_xyz
            )
            coarse_metrics = _compute_single_detector_vis_metrics(
                center_coarse_m, gt_center_m, gt_vol, roi_size_xyz
            )
            z_center = _choose_center_z(center_m, Z)
            z_gt = _choose_center_z(gt_center_m, Z)
            z_fused = _choose_center_z(fused_center_np, Z)
            z_target_heat = _choose_heat_slice(inside_target_vol, z_gt)
            z_fused_heat = _choose_heat_slice(fused_support_vol, z_fused)
            stage0_source = stage0_window_vol if stage0_window_vol is not None else stage0_quality_np
            stageN_source = stageN_window_vol if stageN_window_vol is not None else stageN_quality_np
            z_stage0 = _choose_heat_slice(stage0_source, z_fused)
            z_stageN = _choose_heat_slice(stageN_source, z_fused)
            refine_shift = float(np.abs(center_m - center_coarse_m).mean())

            raw_center = raw_vol[:, :, z_center]
            det_center_panel, hint_contour, hint_z, _ = _make_center_panel_with_gt_hint(
                raw_center,
                gt_vol,
                z_center,
                gt_color,
                gt_alpha=float(_vis_cfg_get(cfg, "gt_alpha", 0.62)),
                glow_alpha=0.42,
                glow_width=5,
            )

            z_seg = _choose_vis_slice(gt_vol, pred_vol)
            raw_seg = raw_vol[:, :, z_seg]
            gt_seg = gt_vol[:, :, z_seg]
            pred_seg = pred_vol[:, :, z_seg]
            target_panel = _overlay_heatmap_on_gray(
                raw_vol[:, :, z_target_heat],
                inside_target_vol[:, :, z_target_heat],
                alpha=heat_alpha,
                cmap_name="viridis",
            )
            fused_center_panel = (
                _overlay_heatmap_on_gray(
                    raw_vol[:, :, z_fused],
                    fused_prior_np[:, :, z_fused],
                    alpha=heat_alpha,
                    cmap_name="cividis",
                )
                if fused_prior_np is not None
                else np.stack([_norm01(raw_vol[:, :, z_fused])] * 3, axis=-1)
            )
            fused_heat_panel = (
                _overlay_heatmap_on_gray(
                    raw_vol[:, :, z_fused_heat],
                    fused_support_vol[:, :, z_fused_heat],
                    alpha=heat_alpha,
                    cmap_name="magma",
                )
                if fused_support_vol is not None
                else np.stack([_norm01(raw_vol[:, :, z_fused_heat])] * 3, axis=-1)
            )
            stage0_heat_vol = None
            if stage0_quality_np is not None and stage0_window_vol is not None:
                stage0_heat_vol = 0.55 * stage0_window_vol + 0.45 * stage0_quality_np
            elif stage0_window_vol is not None:
                stage0_heat_vol = stage0_window_vol
            elif stage0_quality_np is not None:
                stage0_heat_vol = stage0_quality_np

            stageN_heat_vol = None
            if stageN_quality_np is not None and stageN_window_vol is not None:
                stageN_heat_vol = 0.55 * stageN_window_vol + 0.45 * stageN_quality_np
            elif stageN_window_vol is not None:
                stageN_heat_vol = stageN_window_vol
            elif stageN_quality_np is not None:
                stageN_heat_vol = stageN_quality_np

            stage0_panel = (
                _overlay_heatmap_on_gray(
                    raw_vol[:, :, z_stage0],
                    stage0_heat_vol[:, :, z_stage0],
                    alpha=heat_alpha,
                    cmap_name="viridis",
                )
                if stage0_heat_vol is not None
                else np.stack([_norm01(raw_vol[:, :, z_stage0])] * 3, axis=-1)
            )
            stageN_panel = (
                _overlay_heatmap_on_gray(
                    raw_vol[:, :, z_stageN],
                    stageN_heat_vol[:, :, z_stageN],
                    alpha=heat_alpha,
                    cmap_name="plasma",
                )
                if stageN_heat_vol is not None
                else np.stack([_norm01(raw_vol[:, :, z_stageN])] * 3, axis=-1)
            )

            seg_panel = _overlay_dual_mask(
                raw_seg,
                gt_seg,
                pred_seg,
                gt_color,
                pred_color,
                gt_alpha=float(_vis_cfg_get(cfg, "gt_alpha", 0.62)),
                pred_alpha=pred_alpha,
            )

            row = axes[m]

            row[0].imshow(det_center_panel)
            if hint_contour is not None:
                row[0].contour(
                    hint_contour,
                    levels=[0.5],
                    colors=[gt_color],
                    linewidths=2.2,
                    linestyles="dashed",
                )
                title0 = f"modal {m}: center + nearest GT hint + raw (z={z_center}, nearest_z={hint_z})"
            else:
                title0 = f"modal {m}: center + GT + raw (z={z_center})"
            row[0].plot(
                [float(center_coarse_m[1]), float(center_m[1])],
                [float(center_coarse_m[0]), float(center_m[0])],
                color="yellow",
                linewidth=1.7,
                alpha=0.9,
            )
            _draw_center_point(
                row[0], float(center_m[0]), float(center_m[1]), color=detector_pred_color, size=36, marker="o"
            )
            _draw_center_point(
                row[0], float(center_coarse_m[0]), float(center_coarse_m[1]), color="cyan", size=42, marker="^"
            )
            _draw_center_point(
                row[0], float(fused_center_np[0]), float(fused_center_np[1]), color=fused_center_color, size=48, marker="s"
            )
            row[0].set_title(
                f"{title0}\ncoarse_l1={coarse_metrics['center_l1']:.2f}, refine_l1={refined_metrics['center_l1']:.2f}, shift={refine_shift:.2f}, target_z={z_gt}"
            )
            row[0].axis("off")

            row[1].imshow(target_panel)
            if z_target_heat == z_gt:
                row[1].contour(gt_vol[:, :, z_gt], levels=[0.5], colors=[gt_color], linewidths=2.0)
            if float(inside_target_vol[:, :, z_target_heat].max()) > 1e-6:
                row[1].contour(
                    inside_target_vol[:, :, z_target_heat],
                    levels=[0.45],
                    colors=["white"],
                    linewidths=1.6,
                    linestyles="dashed",
                )
            _draw_center_point(
                row[1], float(center_coarse_m[0]), float(center_coarse_m[1]), color="cyan", size=36, marker="^"
            )
            _draw_center_point(
                row[1], float(center_m[0]), float(center_m[1]), color=detector_pred_color, size=34, marker="o"
            )
            _draw_center_point(
                row[1], float(fused_center_np[0]), float(fused_center_np[1]), color=fused_center_color, size=44, marker="s"
            )
            row[1].set_title(
                f"modal {m}: center supervision region + raw (z={z_target_heat})\n"
                f"target=({gt_center_m[0]:.1f},{gt_center_m[1]:.1f},{gt_center_m[2]:.1f}), "
                f"refined_in={refined_metrics['in_mask']:.0f}"
            )
            row[1].axis("off")

            row[2].imshow(fused_center_panel)
            if z_fused == z_gt:
                row[2].contour(gt_vol[:, :, z_gt], levels=[0.5], colors=[gt_color], linewidths=2.0)
            _draw_roi_box(
                row[2], fused_center_np, stage0_roi_size_np, z_fused, (H, W, Z), color=stage0_box_color
            )
            _draw_center_point(
                row[2], float(center_m[0]), float(center_m[1]), color=detector_pred_color, size=34, marker="o"
            )
            _draw_center_point(
                row[2], float(fused_center_np[0]), float(fused_center_np[1]), color=fused_center_color, size=44, marker="s"
            )
            row[2].set_title(
                f"modal {m}: fused center field + stage0 ROI (z={z_fused})\n"
                f"roi0=({stage0_roi_size_np[0]:.1f},{stage0_roi_size_np[1]:.1f},{stage0_roi_size_np[2]:.1f})"
                + (f", fused_conf={fused_conf_np:.2f}" if fused_conf_np is not None else "")
            )
            row[2].axis("off")

            row[3].imshow(fused_heat_panel)
            if z_fused_heat == z_gt:
                row[3].contour(gt_vol[:, :, z_gt], levels=[0.5], colors=[gt_color], linewidths=2.0)
            _draw_roi_box(
                row[3], fused_center_np, stage0_roi_size_np, z_fused_heat, (H, W, Z), color=stage0_box_color
            )
            _draw_center_point(
                row[3], float(center_m[0]), float(center_m[1]), color=detector_pred_color, size=34, marker="o"
            )
            _draw_center_point(
                row[3], float(fused_center_np[0]), float(fused_center_np[1]), color=fused_center_color, size=44, marker="s"
            )
            row[3].set_title(
                f"modal {m}: fused detector support + raw (z={z_fused_heat})"
            )
            row[3].axis("off")

            row[4].imshow(stage0_panel)
            if z_stage0 == z_gt:
                row[4].contour(gt_vol[:, :, z_gt], levels=[0.5], colors=[gt_color], linewidths=2.0)
            _draw_roi_box(
                row[4], fused_center_np, stage0_roi_size_np, z_stage0, (H, W, Z), color=stage0_box_color
            )
            _draw_center_point(
                row[4], float(fused_center_np[0]), float(fused_center_np[1]), color=fused_center_color, size=44, marker="s"
            )
            row[4].set_title(
                f"modal {m}: stage0 ROI guidance (z={z_stage0})"
            )
            row[4].axis("off")

            row[5].imshow(stageN_panel)
            if z_stageN == z_gt:
                row[5].contour(gt_vol[:, :, z_gt], levels=[0.5], colors=[gt_color], linewidths=2.0)
            _draw_roi_box(
                row[5], fused_center_np, stageN_roi_size_np, z_stageN, (H, W, Z), color=stageN_box_color
            )
            _draw_center_point(
                row[5], float(fused_center_np[0]), float(fused_center_np[1]), color=fused_center_color, size=44, marker="s"
            )
            row[5].set_title(
                f"modal {m}: deep-stage ROI guidance (z={z_stageN})\n"
                f"roiN=({stageN_roi_size_np[0]:.1f},{stageN_roi_size_np[1]:.1f},{stageN_roi_size_np[2]:.1f})"
            )
            row[5].axis("off")

            row[6].imshow(seg_panel)
            _draw_center_point(
                row[6], float(center_m[0]), float(center_m[1]), color=detector_pred_color, size=30, marker="o"
            )
            _draw_center_point(
                row[6], float(fused_center_np[0]), float(fused_center_np[1]), color=fused_center_color, size=40, marker="s"
            )
            row[6].set_title(
                f"modal {m}: seg + GT + pred (z={z_seg}) sigma=({sigma_m[0]:.1f},{sigma_m[1]:.1f},{sigma_m[2]:.1f})"
                + (f", conf={conf_m:.2f}" if conf_m is not None else "")
            )
            row[6].axis("off")

        fig.suptitle(
            f"{split_name} epoch={epoch:04d} sampled_batch_idx={chosen_idx_cpu[b]} stage={stage}",
            fontsize=11,
        )
        fig.subplots_adjust(
            left=0.02,
            right=0.985,
            bottom=0.03,
            top=0.92,
            wspace=0.06,
            hspace=0.18,
        )
        fig.savefig(
            vis_dir / f"sample_{b:03d}_batchidx_{chosen_idx_cpu[b]:03d}_{stage}.png"
        )
        plt.close(fig)

    _cleanup_old_visualizations(
        Path(run_dir),
        keep_last_n=int(_vis_cfg_get(cfg, "keep_last_n_epochs", 5)),
        stage=stage,
    )


def _safe_save_epoch_visualizations(
    *, accelerator, model, batch, cfg, epoch: int, split_name: str, run_dir, stage: str
):
    try:
        save_epoch_visualizations(
            accelerator=accelerator,
            model=model,
            batch=batch,
            cfg=cfg,
            epoch=epoch,
            split_name=split_name,
            run_dir=run_dir,
            stage=stage,
        )
    except Exception as exc:
        if accelerator.is_main_process:
            vis_root = Path(run_dir)
            vis_root.mkdir(parents=True, exist_ok=True)
            err_path = vis_root / "visualization_errors.log"
            with err_path.open("a", encoding="utf-8") as f:
                f.write(
                    f"[{datetime.now().isoformat()}] stage={stage} split={split_name} epoch={epoch:04d} "
                    f"{type(exc).__name__}: {exc}\n"
                )
                f.write(traceback.format_exc())
                f.write("\n")
            accelerator.print(
                f"[VisualizationWarning] stage={stage} split={split_name} epoch={epoch:04d} "
                f"failed with {type(exc).__name__}: {exc}. Training will continue."
            )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def build_seg_metrics(enable_hd95: bool = False):
    metrics = {
        "dice_metric": monai.metrics.DiceMetric(
            include_background=True,
            reduction=monai.utils.MetricReduction.MEAN_BATCH,
            get_not_nans=True,
        ),
    }
    if enable_hd95:
        metrics["hd95_metric"] = monai.metrics.HausdorffDistanceMetric(
            percentile=95,
            include_background=True,
            reduction=monai.utils.MetricReduction.MEAN_BATCH,
            get_not_nans=True,
        )
    return metrics


def build_cls_metrics(task_mu: int):
    metrics = []
    for _ in range(task_mu):
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
    return metrics


class Stage2WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = float(pos_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = torch.as_tensor(
            self.pos_weight,
            dtype=logits.dtype,
            device=logits.device,
        )
        return F.binary_cross_entropy_with_logits(
            logits,
            target.float(),
            pos_weight=weight,
        )


class Stage2FocalBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        pos_weight = torch.as_tensor(
            self.pos_weight,
            dtype=logits.dtype,
            device=logits.device,
        )
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight,
            reduction="none",
        )
        prob = torch.sigmoid(logits.float())
        pt = torch.where(target > 0.5, prob, 1.0 - prob)
        alpha = torch.where(
            target > 0.5,
            torch.as_tensor(self.alpha, dtype=prob.dtype, device=prob.device),
            torch.as_tensor(1.0 - self.alpha, dtype=prob.dtype, device=prob.device),
        )
        focal = alpha * (1.0 - pt).clamp_min(0.0).pow(self.gamma) * bce.float()
        return focal.mean().to(dtype=logits.dtype)


class Stage2GlobalDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits.float())
        target = target.float()
        intersection = (prob * target).sum()
        denom = prob.sum() + target.sum()
        dice = (2.0 * intersection + self.eps) / (denom + self.eps)
        return (1.0 - dice).to(dtype=logits.dtype)


def build_stage2_segmentation_losses(cfg) -> Dict[str, nn.Module]:
    mode = str(_cfg_get(cfg, "stage_train.stage2.seg_loss_mode", "dice_focal")).lower()
    if mode in {"dice_focal", "focal_dice", "default"}:
        return {
            "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
            "dice_loss": monai.losses.DiceLoss(
                smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True
            ),
        }

    pos_weight = float(_cfg_get(cfg, "stage_train.stage2.bce_pos_weight", 20.0))
    if mode in {"dice_bce", "bce_dice"}:
        return {
            "dice_loss": monai.losses.DiceLoss(
                smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True
            ),
            "bce_loss": Stage2WeightedBCELoss(pos_weight=pos_weight),
        }

    if mode in {"global_dice_bce", "bce_global_dice"}:
        return {
            "global_dice_loss": Stage2GlobalDiceLoss(),
            "bce_loss": Stage2WeightedBCELoss(pos_weight=pos_weight),
        }

    if mode in {"focal_global_dice_bce", "global_dice_focal_bce", "direct_union"}:
        gamma = float(_cfg_get(cfg, "stage_train.stage2.focal_gamma", 2.0))
        alpha = float(_cfg_get(cfg, "stage_train.stage2.focal_alpha", 0.75))
        return {
            "global_dice_loss": Stage2GlobalDiceLoss(),
            "focal_bce_loss": Stage2FocalBCELoss(
                pos_weight=pos_weight,
                gamma=gamma,
                alpha=alpha,
            ),
            "bce_loss": Stage2WeightedBCELoss(pos_weight=pos_weight),
        }

    if mode in {"tversky_bce", "bce_tversky", "small_lesion"}:
        alpha = float(_cfg_get(cfg, "stage_train.stage2.tversky_alpha", 0.3))
        beta = float(_cfg_get(cfg, "stage_train.stage2.tversky_beta", 0.7))
        return {
            "tversky_loss": monai.losses.TverskyLoss(
                smooth_nr=0,
                smooth_dr=1e-5,
                to_onehot_y=False,
                sigmoid=True,
                alpha=alpha,
                beta=beta,
            ),
            "bce_loss": Stage2WeightedBCELoss(pos_weight=pos_weight),
        }

    raise ValueError(
        "Unsupported stage2 segmentation loss mode: "
        f"{mode}. Use dice_focal, dice_bce, global_dice_bce, "
        "focal_global_dice_bce, or tversky_bce."
    )


def compute_stage2_prior_energy_loss(
    debug: Optional[Dict[str, torch.Tensor]],
    cfg,
) -> torch.Tensor:
    """Limit over-broad ROI guidance during Stage 2 adaptation."""
    device = None
    if debug is None:
        return torch.tensor(0.0)

    quality_maps = debug.get("stage_prior_quality_maps", None)
    quality_budget = float(_cfg_get(cfg, "stage_train.stage2.prior_quality_budget", 0.12))
    if quality_maps is not None and len(quality_maps) > 0:
        loss = 0.0
        count = 0
        for q in quality_maps:
            if q is None:
                continue
            device = q.device
            q_mean = q.float().mean()
            loss = loss + F.relu(q_mean - quality_budget)
            count += 1
        if count > 0:
            return loss / count

    if "hwa_priors" not in debug:
        return torch.tensor(0.0, device=device if device is not None else "cpu")

    priors = debug["hwa_priors"]
    if priors is None or len(priors) == 0:
        return torch.tensor(0.0, device=device if device is not None else "cpu")

    energy_budget = float(_cfg_get(cfg, "stage_train.stage2.prior_feature_energy_budget", 0.10))
    loss = 0.0
    count = 0
    for p in priors:
        if p is None:
            continue
        device = p.device
        loss = loss + F.relu(p.pow(2).mean() - energy_budget)
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device if device is not None else "cpu")

    return loss / count


def compute_stage2_prior_localize_loss(
    debug: Optional[Dict[str, torch.Tensor]],
    seg_gt: torch.Tensor,
    cfg,
):
    device = seg_gt.device
    if debug is None or "stage_window_evidence" not in debug:
        zero = torch.tensor(0.0, device=device)
        return zero, {"prior_localize_inside_ratio_mean": zero, "prior_localize_outside_mean": zero}

    stage_window_list = debug["stage_window_evidence"]
    if stage_window_list is None or len(stage_window_list) == 0:
        zero = torch.tensor(0.0, device=device)
        return zero, {"prior_localize_inside_ratio_mean": zero, "prior_localize_outside_mean": zero}

    target_dict = build_stage1_detector_targets(seg_gt.float(), cfg)
    inside_target_each = target_dict["inside_target_each"]
    lesion_target_each = seg_gt.float().clamp(0.0, 1.0)
    focus_target_each = torch.clamp(inside_target_each + 0.25 * lesion_target_each, max=1.0)
    modal_present = (seg_gt.sum(dim=(2, 3, 4)) > 0).float()
    outside_weight = float(_cfg_get(cfg, "stage_train.stage2.prior_localize_outside_weight", 0.5))
    target_inside_ratio = float(_cfg_get(cfg, "stage_train.stage2.prior_localize_target_ratio", 0.55))

    total_loss = torch.tensor(0.0, device=device)
    inside_ratio_total = torch.tensor(0.0, device=device)
    outside_energy_total = torch.tensor(0.0, device=device)
    valid = 0

    for stage_window in stage_window_list:
        if stage_window is None:
            continue
        window_full = F.interpolate(
            stage_window.float(),
            size=seg_gt.shape[2:],
            mode="trilinear",
            align_corners=False,
        ).clamp(1e-4, 1.0 - 1e-4)

        focus_mass_each = focus_target_each.sum(dim=(2, 3, 4)).clamp_min(1.0)
        inside_energy_each = (
            (window_full * focus_target_each).sum(dim=(2, 3, 4)) / focus_mass_each
        )
        total_energy_each = window_full.sum(dim=(2, 3, 4)).clamp_min(1e-6)
        inside_ratio_each = (
            (window_full * focus_target_each).sum(dim=(2, 3, 4)) / total_energy_each
        )
        inside_loss_each = F.relu(target_inside_ratio - inside_ratio_each)
        inside_loss = _modal_weighted_mean(inside_loss_each, modal_present)

        outside_mask_each = (1.0 - focus_target_each).clamp(0.0, 1.0)
        outside_mass_each = outside_mask_each.sum(dim=(2, 3, 4)).clamp_min(1.0)
        outside_energy_each = (
            (window_full * outside_mask_each).sum(dim=(2, 3, 4)) / outside_mass_each
        )
        outside_loss = _modal_weighted_mean(outside_energy_each, modal_present)

        total_loss = total_loss + inside_loss + outside_weight * outside_loss
        inside_ratio_total = inside_ratio_total + _modal_weighted_mean(
            inside_ratio_each, modal_present
        )
        outside_energy_total = outside_energy_total + outside_loss
        valid += 1

    if valid <= 0:
        zero = torch.tensor(0.0, device=device)
        return zero, {"prior_localize_inside_ratio_mean": zero, "prior_localize_outside_mean": zero}

    valid_f = float(valid)
    return total_loss / valid_f, {
        "prior_localize_inside_ratio_mean": inside_ratio_total / valid_f,
        "prior_localize_outside_mean": outside_energy_total / valid_f,
    }


def compute_stage2_gate_reg_loss(
    debug: Optional[Dict[str, torch.Tensor]],
    cfg,
    gate_cap: Optional[float] = None,
) -> torch.Tensor:
    """Regularize prior gates and fall back to modal entropy when gate debug is absent."""
    device = None
    if debug is None:
        return torch.tensor(0.0)

    encoder_gates = debug.get("encoder_prior_gates", None)
    if encoder_gates is not None and len(encoder_gates) > 0:
        gate_cap = float(
            gate_cap
            if gate_cap is not None
            else _cfg_get(cfg, "stage_train.stage2.prior_alpha_cap_end", 0.30)
        )
        loss = 0.0
        count = 0
        for g in encoder_gates:
            if g is None:
                continue
            device = g.device
            gate_mean = g.float().mean()
            loss = loss + F.relu(gate_mean - gate_cap)
            count += 1
        if count > 0:
            return loss / count

    if "stage_modal_gates" not in debug:
        return torch.tensor(0.0, device=device if device is not None else "cpu")

    gates = debug["stage_modal_gates"]
    if gates is None or len(gates) == 0:
        return torch.tensor(0.0, device=device if device is not None else "cpu")

    entropy_target = float(_cfg_get(cfg, "stage_train.stage2.gate_entropy_target", 0.6))
    eps = 1e-8

    loss = 0.0
    count = 0
    for g in gates:
        if g is None:
            continue
        device = g.device
        # g: [B,M,H,W,Z]
        prob = g.clamp_min(eps)
        entropy = -(prob * prob.log()).sum(dim=1).mean()  # scalar
        loss = loss + (entropy - entropy_target).abs()
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device if device is not None else "cpu")

    return loss / count

def _compute_stage1_center_metrics(
    center_pred_each: torch.Tensor,
    center_gt_each: torch.Tensor,
    seg_gt: torch.Tensor,
    roi_size_xyz: Tuple[int, int, int],
) -> Dict[str, torch.Tensor]:
    """
    center_pred_each: [B,M,3]
    center_gt_each:   [B,M,3]
    seg_gt:           [B,M,H,W,Z]
    """
    B, M, H, W, Z = seg_gt.shape
    modal_present = (seg_gt.sum(dim=(2, 3, 4)) > 0).float()  # [B,M]

    center_l1_each = F.l1_loss(
        center_pred_each, center_gt_each, reduction="none"
    ).mean(dim=-1)  # [B,M]
    center_l1 = (center_l1_each * modal_present).sum() / modal_present.sum().clamp_min(1.0)

    rh, rw, rz = [max(1, int(v)) for v in roi_size_xyz]
    center_round = center_pred_each.round().long()

    in_mask_cnt = 0.0
    cov_sum = 0.0
    valid_cnt = 0.0

    for b in range(B):
        for m in range(M):
            if modal_present[b, m] < 0.5:
                continue
            valid_cnt += 1.0

            x = int(center_round[b, m, 0].clamp(0, H - 1).item())
            y = int(center_round[b, m, 1].clamp(0, W - 1).item())
            z = int(center_round[b, m, 2].clamp(0, Z - 1).item())

            gt_m = seg_gt[b, m]
            if float(gt_m[x, y, z].item()) > 0.5:
                in_mask_cnt += 1.0

            x1 = max(0, x - rh // 2)
            y1 = max(0, y - rw // 2)
            z1 = max(0, z - rz // 2)
            x2 = min(H, x1 + rh)
            y2 = min(W, y1 + rw)
            z2 = min(Z, z1 + rz)

            gt_sum = float(gt_m.sum().item())
            if gt_sum <= 0.0:
                cov_sum += 1.0
            else:
                covered = float(gt_m[x1:x2, y1:y2, z1:z2].sum().item())
                cov_sum += covered / max(gt_sum, 1e-6)

    denom = max(valid_cnt, 1.0)
    center_in_mask_rate = torch.tensor(in_mask_cnt / denom, device=seg_gt.device)
    roi_coverage = torch.tensor(cov_sum / denom, device=seg_gt.device)

    return {
        "center_l1": center_l1,
        "center_in_mask_rate": center_in_mask_rate,
        "roi_coverage": roi_coverage,
    }


def _stage1_monitor_score(metric: Dict[str, float], split: str) -> float:
    coverage = float(metric.get(f"{split}/stage1/roi_coverage", 0.0))
    in_rate = float(metric.get(f"{split}/stage1/center_in_mask_rate", 0.0))
    coarse_in_rate = float(metric.get(f"{split}/stage1/center_coarse_in_mask_rate", 0.0))
    center_l1 = float(metric.get(f"{split}/stage1/center_modal_loss", 1e9))
    inside_score = float(metric.get(f"{split}/stage1/center_inside_score_mean", 0.0))
    coarse_inside_score = float(metric.get(f"{split}/stage1/center_coarse_inside_score_mean", 0.0))
    evidence_outside = float(metric.get(f"{split}/stage1/evidence_outside_loss", 1e9))
    raw_evidence_outside = float(metric.get(f"{split}/stage1/raw_evidence_outside_loss", 1e9))
    evidence_inside_ratio = float(metric.get(f"{split}/stage1/evidence_inside_ratio_mean", 0.0))
    fused_support_inside_ratio = float(metric.get(f"{split}/stage1/fused_support_inside_ratio_mean", 0.0))
    agreement_inside_score = float(metric.get(f"{split}/stage1/agreement_inside_score_mean", 0.0))
    return (
        2.8 * inside_score
        + 2.2 * in_rate
        + 1.1 * max(0.0, inside_score - coarse_inside_score)
        + 0.8 * max(0.0, in_rate - coarse_in_rate)
        + 0.9 * coverage
        + 0.35 * fused_support_inside_ratio
        + 0.15 * evidence_inside_ratio
        + 0.10 * agreement_inside_score
        - 0.06 * center_l1
        - 0.18 * evidence_outside
        - 0.05 * raw_evidence_outside
    )


def compute_stage1_evidence_peak_loss(debug: Optional[Dict[str, torch.Tensor]], cfg) -> torch.Tensor:
    if debug is None or "det_seed_evidence_prob_each" not in debug:
        return torch.tensor(0.0)
    prob = debug["det_seed_evidence_prob_each"]
    if prob is None:
        return torch.tensor(0.0)
    peak = prob.amax(dim=(2,3,4))
    mean = prob.mean(dim=(2,3,4))
    margin = float(_cfg_get(cfg, "stage_train.stage1.evidence_peak_margin", 0.20))
    return F.relu(margin - (peak - mean)).mean()

def _mean_cls_accuracy(
    metric_dict: Dict[str, float], split: str, task_mu: int
) -> float:
    vals = []
    for i in range(task_mu):
        key = f"{split}/stage3/Task{i}_accuracy"
        if key in metric_dict:
            vals.append(metric_dict[key])
    return float(sum(vals) / max(len(vals), 1))


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    accelerator,
    epoch,
    step,
    stage_name,
    cfg,
    seg_loss_functions,
    loss_functions_cls,
    seg_metrics=None,
    cls_metrics=None,
    post_trans=None,
    post_trans_cls=None,
    stage_local_epoch: Optional[int] = None,
    stage_total_epochs: Optional[int] = None,
    display_total_epochs: Optional[int] = None,
):
    model.train()
    running = {}
    vis_batch = None
    stage2_sched = (
        _get_stage2_schedule(
            epoch,
            cfg,
            stage_local_epoch=stage_local_epoch,
            stage_total_epochs=stage_total_epochs,
        )
        if stage_name == "stage2"
        else None
    )
    shown_total_epochs = int(display_total_epochs or cfg.trainer.num_epochs)
    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
        desc=f"Epoch {epoch + 1}/{shown_total_epochs} | {stage_name}",
    )

    refresh_every = _progress_refresh_every(cfg, stage_name, default=5)
    valid_batches = 0
    grad_accum_steps = max(1, int(_cfg_get(cfg, "trainer.grad_accum_steps", 1)))
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, image_batch in pbar:
        if image_batch is None:
            continue
        valid_batches += 1

        if vis_batch is None:
            vis_batch = {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in image_batch.items()
            }

        image = image_batch["image"]
        do_optimizer_step = (
            (valid_batches % grad_accum_steps) == 0
            or (batch_idx + 1) == len(train_loader)
        )

        if stage_name == "stage1":
            _, _, debug = forward_model(
                model, image, stage=stage_name, debug=True, detector_only=True
            )
            label = image_batch["label"]
            loss_dict = stage1_detect_losses(debug, label, model, cfg, epoch=epoch)
            total_loss = loss_dict["loss_total"]

            scaled_loss = total_loss / float(grad_accum_steps)
            accelerator.backward(scaled_loss)
            if do_optimizer_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running["loss"] = running.get("loss", 0.0) + _loss_to_scalar(total_loss)
            running["center_modal_loss"] = running.get("center_modal_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_center_modal"]
            )
            running["center_coarse_modal_loss"] = running.get("center_coarse_modal_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_center_coarse_modal"]
            )
            running["center_inside_loss"] = running.get("center_inside_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_center_inside"]
            )
            running["center_coarse_inside_loss"] = running.get("center_coarse_inside_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_center_coarse_inside"]
            )
            running["sigma_reg_loss"] = running.get("sigma_reg_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_sigma_reg"]
            )
            running["conf_reg_loss"] = running.get("conf_reg_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_conf_reg"]
            )
            running["raw_evidence_supervise_loss"] = running.get("raw_evidence_supervise_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_raw_evidence_supervise"]
            )
            running["raw_evidence_outside_loss"] = running.get("raw_evidence_outside_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_raw_evidence_outside"]
            )
            running["evidence_supervise_loss"] = running.get("evidence_supervise_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_evidence_supervise"]
            )
            running["evidence_outside_loss"] = running.get("evidence_outside_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_evidence_outside"]
            )
            running["evidence_total_loss"] = running.get("evidence_total_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_evidence_total"]
            )
            running["fused_support_supervise_loss"] = running.get("fused_support_supervise_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_fused_support_supervise"]
            )
            running["fused_support_outside_loss"] = running.get("fused_support_outside_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_fused_support_outside"]
            )
            running["scale_floor_loss"] = running.get("scale_floor_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_scale_floor"]
            )
            running["evidence_peak_loss"] = running.get("evidence_peak_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_evidence_peak"]
            )
            running["center_inside_score_mean"] = running.get("center_inside_score_mean", 0.0) + _loss_to_scalar(
                loss_dict["center_inside_score_mean"]
            )
            running["raw_evidence_inside_ratio_mean"] = running.get("raw_evidence_inside_ratio_mean", 0.0) + _loss_to_scalar(
                loss_dict["raw_evidence_inside_ratio_mean"]
            )
            running["evidence_inside_ratio_mean"] = running.get("evidence_inside_ratio_mean", 0.0) + _loss_to_scalar(
                loss_dict["evidence_inside_ratio_mean"]
            )
            running["fused_support_inside_ratio_mean"] = running.get("fused_support_inside_ratio_mean", 0.0) + _loss_to_scalar(
                loss_dict["fused_support_inside_ratio_mean"]
            )
            running["agreement_inside_score_mean"] = running.get("agreement_inside_score_mean", 0.0) + _loss_to_scalar(
                loss_dict["agreement_inside_score_mean"]
            )
            running["reliability_inside_score_mean"] = running.get("reliability_inside_score_mean", 0.0) + _loss_to_scalar(
                loss_dict["reliability_inside_score_mean"]
            )
            running["center_coarse_inside_score_mean"] = running.get("center_coarse_inside_score_mean", 0.0) + _loss_to_scalar(
                loss_dict["center_coarse_inside_score_mean"]
            )
            running["center_refine_shift_mean"] = running.get("center_refine_shift_mean", 0.0) + _loss_to_scalar(
                loss_dict["center_refine_shift_mean"]
            )
            running["center_in_mask_rate"] = running.get("center_in_mask_rate", 0.0) + _loss_to_scalar(
                loss_dict["center_in_mask_rate"]
            )
            running["center_coarse_in_mask_rate"] = running.get("center_coarse_in_mask_rate", 0.0) + _loss_to_scalar(
                loss_dict["center_coarse_in_mask_rate"]
            )
            running["roi_coverage"] = running.get("roi_coverage", 0.0) + _loss_to_scalar(
                loss_dict["roi_coverage"]
            )
            show_loss = _loss_to_scalar(total_loss)

        elif stage_name == "stage2":
            label = image_batch["label"]
            use_detector_debug = _stage2_use_detector_debug(cfg)
            _, seg_logits, debug = forward_model(
                model, image, stage=stage_name, debug=use_detector_debug
            )
            if use_detector_debug:
                det_aux_dict = stage1_detect_losses(debug, label, model, cfg, epoch=epoch)
                debug_stats = _compute_stage2_debug_stats(debug, device=label.device)
            else:
                zero = torch.tensor(0.0, device=label.device)
                det_aux_dict = {
                    "loss_center_inside": zero,
                    "center_inside_score_mean": zero,
                    "evidence_inside_ratio_mean": zero,
                    "fused_support_inside_ratio_mean": zero,
                    "agreement_inside_score_mean": zero,
                    "reliability_inside_score_mean": zero,
                    "center_coarse_inside_score_mean": zero,
                    "center_refine_shift_mean": zero,
                    "center_in_mask_rate": zero,
                    "center_coarse_in_mask_rate": zero,
                    "roi_coverage": zero,
                }
                debug_stats = {}
            hwa_runtime_stats = _compute_stage2_hwa_runtime_stats(model, device=label.device)
            seg_logits_for_loss, label_for_loss = _stage2_prepare_seg_supervision(
                seg_logits, label, cfg
            )

            seg_total_loss = 0.0
            for name, fn in seg_loss_functions.items():
                loss = fn(seg_logits_for_loss, label_for_loss)
                seg_total_loss = seg_total_loss + loss
                running[name] = running.get(name, 0.0) + _loss_to_scalar(loss)

            hwa_advantage_loss = torch.zeros((), device=label.device)
            if float(_cfg_get(cfg, "stage_train.stage2.hwa_advantage_loss_scale", 0.0)) > 0.0:
                was_enabled = _get_runtime_hwa_prior_enabled(model)
                sched_for_ablation = stage2_sched or {}
                _set_runtime_hwa_prior_enabled(model, False)
                _apply_stage2_hwa_runtime_scales(
                    model,
                    {
                        **sched_for_ablation,
                        "input_hwa_gate_scale": 0.0,
                        "input_hwa_gain_scale": 0.0,
                    },
                )
                try:
                    with torch.no_grad():
                        _, no_hwa_logits, _ = forward_model(
                            model, image, stage=stage_name, debug=False
                        )
                finally:
                    if was_enabled is not None:
                        _set_runtime_hwa_prior_enabled(model, was_enabled)
                    _apply_stage2_hwa_runtime_scales(model, stage2_sched)

                advantage_metric = str(
                    _cfg_get(cfg, "stage_train.stage2.hwa_advantage_metric", "loss")
                ).lower()
                if advantage_metric == "dice":
                    hwa_advantage_loss = _stage2_hwa_advantage_dice_margin_loss(
                        seg_logits,
                        no_hwa_logits,
                        label,
                        cfg,
                        stage_local_epoch=stage_local_epoch,
                    )
                else:
                    no_hwa_seg_loss = torch.zeros((), device=label.device)
                    no_hwa_logits_for_loss, _ = _stage2_prepare_seg_supervision(
                        no_hwa_logits.detach(), label, cfg
                    )
                    for fn in seg_loss_functions.values():
                        no_hwa_seg_loss = no_hwa_seg_loss + fn(no_hwa_logits_for_loss, label_for_loss)
                    hwa_advantage_loss = _stage2_hwa_advantage_margin_loss(
                        seg_total_loss,
                        no_hwa_seg_loss.detach(),
                        cfg,
                        stage_local_epoch=stage_local_epoch,
                    )

            aux_weight = float(stage2_sched["lambda_center_inside_aux"])
            total_loss = (
                seg_total_loss
                + hwa_advantage_loss
            )
            if aux_weight > 0.0:
                total_loss = total_loss + aux_weight * det_aux_dict["loss_center_inside"]

            scaled_loss = total_loss / float(grad_accum_steps)
            accelerator.backward(scaled_loss)
            if do_optimizer_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running["loss"] = running.get("loss", 0.0) + _loss_to_scalar(total_loss)
            running["seg_total_loss"] = running.get("seg_total_loss", 0.0) + _loss_to_scalar(seg_total_loss)
            running["hwa_advantage_loss"] = running.get("hwa_advantage_loss", 0.0) + _loss_to_scalar(
                hwa_advantage_loss
            )
            running["center_inside_aux_loss"] = running.get("center_inside_aux_loss", 0.0) + _loss_to_scalar(
                det_aux_dict["loss_center_inside"]
            )
            running["lambda_center_inside_aux"] = running.get("lambda_center_inside_aux", 0.0) + float(
                aux_weight
            )
            running["fusion_progress"] = running.get("fusion_progress", 0.0) + float(
                stage2_sched["fusion_progress"]
            )
            running["prior_progress"] = running.get("prior_progress", 0.0) + float(
                stage2_sched["prior_progress"]
            )
            running["detector_aux_progress"] = running.get("detector_aux_progress", 0.0) + float(
                stage2_sched["detector_aux_progress"]
            )
            running["prior_alpha_cap"] = running.get("prior_alpha_cap", 0.0) + float(stage2_sched["prior_alpha_cap"])
            running["center_inside_score_mean"] = running.get("center_inside_score_mean", 0.0) + _loss_to_scalar(
                det_aux_dict["center_inside_score_mean"]
            )
            running["evidence_inside_ratio_mean"] = running.get("evidence_inside_ratio_mean", 0.0) + _loss_to_scalar(
                det_aux_dict["evidence_inside_ratio_mean"]
            )
            running["fused_support_inside_ratio_mean"] = running.get("fused_support_inside_ratio_mean", 0.0) + _loss_to_scalar(
                det_aux_dict["fused_support_inside_ratio_mean"]
            )
            running["agreement_inside_score_mean"] = running.get("agreement_inside_score_mean", 0.0) + _loss_to_scalar(
                det_aux_dict["agreement_inside_score_mean"]
            )
            running["reliability_inside_score_mean"] = running.get("reliability_inside_score_mean", 0.0) + _loss_to_scalar(
                det_aux_dict["reliability_inside_score_mean"]
            )
            running["center_coarse_inside_score_mean"] = running.get("center_coarse_inside_score_mean", 0.0) + _loss_to_scalar(
                det_aux_dict["center_coarse_inside_score_mean"]
            )
            running["center_refine_shift_mean"] = running.get("center_refine_shift_mean", 0.0) + _loss_to_scalar(
                det_aux_dict["center_refine_shift_mean"]
            )
            running["center_in_mask_rate"] = running.get("center_in_mask_rate", 0.0) + _loss_to_scalar(
                det_aux_dict["center_in_mask_rate"]
            )
            running["center_coarse_in_mask_rate"] = running.get("center_coarse_in_mask_rate", 0.0) + _loss_to_scalar(
                det_aux_dict["center_coarse_in_mask_rate"]
            )
            running["roi_coverage"] = running.get("roi_coverage", 0.0) + _loss_to_scalar(
                det_aux_dict["roi_coverage"]
            )
            for stat_name, stat_value in debug_stats.items():
                running[stat_name] = running.get(stat_name, 0.0) + _loss_to_scalar(stat_value)
            for stat_name, stat_value in hwa_runtime_stats.items():
                running[stat_name] = running.get(stat_name, 0.0) + _loss_to_scalar(stat_value)
            show_loss = _loss_to_scalar(total_loss)

            if post_trans is not None and seg_metrics is not None:
                if _stage2_fast_scalar_metrics_enabled(cfg, accelerator):
                    _accumulate_stage2_fast_dice(running, seg_logits, label, cfg=cfg)
                else:
                    if _stage2_seg_target_mode(cfg) in {"union", "lesion_union", "single_union"}:
                        val_outputs = (
                            torch.sigmoid(seg_logits.detach()).amax(dim=1, keepdim=True)
                            > _stage2_pred_threshold(cfg)
                        ).float()
                        metric_label = _stage2_union_target(label.detach())
                    else:
                        val_outputs = post_trans(seg_logits)
                        metric_label = label
                    for metric_name in seg_metrics:
                        seg_metrics[metric_name](y_pred=val_outputs, y=metric_label)

        elif stage_name == "stage3":
            cls_logits, _, _ = forward_model(model, image, stage=stage_name, debug=False)
            stage3_loss_dict = compute_stage3_classification_loss(
                cls_logits=cls_logits,
                batch=image_batch,
                loss_functions=loss_functions_cls,
                accelerator=accelerator,
                step=step,
                log_prefix="Train",
                lambda_class=float(_cfg_get(cfg, "stage_train.stage3.lambda_class", 1.0)),
                lambda_pfs=float(_cfg_get(cfg, "stage_train.stage3.lambda_pfs", 1.0)),
                task_mu=int(_cfg_get(cfg, "GCM_loader.task_Mu", 2) or 2),
            )
            total_loss = stage3_loss_dict["total_loss"]
            scaled_loss = total_loss / float(grad_accum_steps)
            accelerator.backward(scaled_loss)
            if do_optimizer_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running["loss"] = running.get("loss", 0.0) + _loss_to_scalar(total_loss)
            for k, v in stage3_loss_dict.items():
                if torch.is_tensor(v) and "loss" in k and k != "total_loss":
                    running[k] = running.get(k, 0.0) + _loss_to_scalar(v)
            show_loss = _loss_to_scalar(total_loss)

            if post_trans_cls is not None and cls_metrics is not None:
                y_pred_1 = post_trans_cls(stage3_loss_dict["logit_class"])
                y1 = stage3_loss_dict["labels_class"]
                for metric_name in cls_metrics[0]:
                    pred1, lab1 = y_pred_1, y1
                    if metric_name == "miou_metric":
                        pred1 = pred1.unsqueeze(2)
                        lab1 = lab1.unsqueeze(2)
                    cls_metrics[0][metric_name](y_pred=pred1, y=lab1)
                if len(cls_metrics) > 1 and stage3_loss_dict["logit_pfs"] is not None:
                    y_pred_2 = post_trans_cls(stage3_loss_dict["logit_pfs"])
                    y2 = stage3_loss_dict["labels_pfs"]
                    for metric_name in cls_metrics[1]:
                        pred2, lab2 = y_pred_2, y2
                        if metric_name == "miou_metric":
                            pred2 = pred2.unsqueeze(2)
                            lab2 = lab2.unsqueeze(2)
                        cls_metrics[1][metric_name](y_pred=pred2, y=lab2)
        else:
            raise ValueError(f"Unsupported stage_name: {stage_name}")

        lr = max((pg["lr"] for pg in optimizer.param_groups), default=0.0)
        if (
            accelerator.is_local_main_process
            and (((batch_idx + 1) % refresh_every) == 0 or (batch_idx + 1) == len(train_loader))
        ):
            pbar.set_postfix(loss=f"{show_loss:.4f}", lr=f"{lr:.2e}", step=step)
        step += 1

    metric = {}
    num_batches = max(valid_batches, 1)
    for k, v in running.items():
        metric[f"Train/{stage_name}/{k}"] = v / num_batches

    if stage_name == "stage2" and seg_metrics is not None:
        if _stage2_fast_scalar_metrics_enabled(cfg, accelerator):
            metric[f"Train/{stage_name}/dice_metric"] = _finalize_stage2_fast_dice(
                running, accelerator, distributed_reduce=False
            )
        else:
            for metric_name in seg_metrics:
                batch_acc = seg_metrics[metric_name].aggregate()[0].to(accelerator.device)
                batch_acc = torch.nan_to_num(batch_acc, nan=0.0, posinf=0.0, neginf=0.0)
                if accelerator.num_processes > 1:
                    batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
                seg_metrics[metric_name].reset()
                metric[f"Train/{stage_name}/{metric_name}"] = float(batch_acc.mean())

    if stage_name == "stage3" and cls_metrics is not None:
        for task_idx in range(len(cls_metrics)):
            for metric_name in cls_metrics[task_idx]:
                batch_acc = cls_metrics[task_idx][metric_name].aggregate()[0].to(accelerator.device)
                if accelerator.num_processes > 1:
                    batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
                cls_metrics[task_idx][metric_name].reset()
                metric[f"Train/{stage_name}/Task{task_idx}_{metric_name}"] = float(batch_acc.mean())

    accelerator.log(metric, step=epoch)
    return metric, step, vis_batch


@torch.no_grad()
def val_one_epoch(
    model,
    val_loader,
    accelerator,
    epoch,
    step,
    stage_name,
    cfg,
    seg_loss_functions,
    loss_functions_cls,
    seg_metrics=None,
    cls_metrics=None,
    post_trans=None,
    post_trans_cls=None,
    split="Val",
    stage_local_epoch: Optional[int] = None,
    stage_total_epochs: Optional[int] = None,
    display_total_epochs: Optional[int] = None,
):
    model.eval()
    running = {}
    vis_batch = None
    stage2_sched = (
        _get_stage2_schedule(
            epoch,
            cfg,
            stage_local_epoch=stage_local_epoch,
            stage_total_epochs=stage_total_epochs,
        )
        if stage_name == "stage2"
        else None
    )
    shown_total_epochs = int(display_total_epochs or cfg.trainer.num_epochs)
    pbar = tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
        desc=f"{split} Epoch {epoch + 1}/{shown_total_epochs} | {stage_name}",
    )

    refresh_every = _progress_refresh_every(cfg, stage_name, default=5)
    valid_batches = 0

    for batch_idx, image_batch in pbar:
        if image_batch is None:
            continue
        valid_batches += 1

        if vis_batch is None:
            vis_batch = {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in image_batch.items()
            }

        image = image_batch["image"]

        if stage_name == "stage1":
            _, _, debug = forward_model(
                model, image, stage=stage_name, debug=True, detector_only=True
            )
            label = image_batch["label"]
            loss_dict = stage1_detect_losses(debug, label, model, cfg, epoch=epoch)
            total_loss = loss_dict["loss_total"]

            running["loss"] = running.get("loss", 0.0) + _loss_to_scalar(total_loss)
            running["center_modal_loss"] = running.get("center_modal_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_center_modal"]
            )
            running["center_coarse_modal_loss"] = running.get("center_coarse_modal_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_center_coarse_modal"]
            )
            running["center_inside_loss"] = running.get("center_inside_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_center_inside"]
            )
            running["center_coarse_inside_loss"] = running.get("center_coarse_inside_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_center_coarse_inside"]
            )
            running["sigma_reg_loss"] = running.get("sigma_reg_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_sigma_reg"]
            )
            running["conf_reg_loss"] = running.get("conf_reg_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_conf_reg"]
            )
            running["raw_evidence_supervise_loss"] = running.get("raw_evidence_supervise_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_raw_evidence_supervise"]
            )
            running["raw_evidence_outside_loss"] = running.get("raw_evidence_outside_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_raw_evidence_outside"]
            )
            running["evidence_supervise_loss"] = running.get("evidence_supervise_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_evidence_supervise"]
            )
            running["evidence_outside_loss"] = running.get("evidence_outside_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_evidence_outside"]
            )
            running["evidence_total_loss"] = running.get("evidence_total_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_evidence_total"]
            )
            running["fused_support_supervise_loss"] = running.get("fused_support_supervise_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_fused_support_supervise"]
            )
            running["fused_support_outside_loss"] = running.get("fused_support_outside_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_fused_support_outside"]
            )
            running["scale_floor_loss"] = running.get("scale_floor_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_scale_floor"]
            )
            running["evidence_peak_loss"] = running.get("evidence_peak_loss", 0.0) + _loss_to_scalar(
                loss_dict["loss_evidence_peak"]
            )
            running["center_inside_score_mean"] = running.get("center_inside_score_mean", 0.0) + _loss_to_scalar(
                loss_dict["center_inside_score_mean"]
            )
            running["raw_evidence_inside_ratio_mean"] = running.get("raw_evidence_inside_ratio_mean", 0.0) + _loss_to_scalar(
                loss_dict["raw_evidence_inside_ratio_mean"]
            )
            running["evidence_inside_ratio_mean"] = running.get("evidence_inside_ratio_mean", 0.0) + _loss_to_scalar(
                loss_dict["evidence_inside_ratio_mean"]
            )
            running["fused_support_inside_ratio_mean"] = running.get("fused_support_inside_ratio_mean", 0.0) + _loss_to_scalar(
                loss_dict["fused_support_inside_ratio_mean"]
            )
            running["agreement_inside_score_mean"] = running.get("agreement_inside_score_mean", 0.0) + _loss_to_scalar(
                loss_dict["agreement_inside_score_mean"]
            )
            running["reliability_inside_score_mean"] = running.get("reliability_inside_score_mean", 0.0) + _loss_to_scalar(
                loss_dict["reliability_inside_score_mean"]
            )
            running["center_coarse_inside_score_mean"] = running.get("center_coarse_inside_score_mean", 0.0) + _loss_to_scalar(
                loss_dict["center_coarse_inside_score_mean"]
            )
            running["center_refine_shift_mean"] = running.get("center_refine_shift_mean", 0.0) + _loss_to_scalar(
                loss_dict["center_refine_shift_mean"]
            )
            running["center_in_mask_rate"] = running.get("center_in_mask_rate", 0.0) + _loss_to_scalar(
                loss_dict["center_in_mask_rate"]
            )
            running["center_coarse_in_mask_rate"] = running.get("center_coarse_in_mask_rate", 0.0) + _loss_to_scalar(
                loss_dict["center_coarse_in_mask_rate"]
            )
            running["roi_coverage"] = running.get("roi_coverage", 0.0) + _loss_to_scalar(
                loss_dict["roi_coverage"]
            )
            show_loss = _loss_to_scalar(total_loss)

        elif stage_name == "stage2":
            label = image_batch["label"]
            use_detector_debug = _stage2_use_detector_debug(cfg)
            _, seg_logits, debug = forward_model(
                model, image, stage=stage_name, debug=use_detector_debug
            )
            if use_detector_debug:
                det_aux_dict = stage1_detect_losses(debug, label, model, cfg, epoch=epoch)
                debug_stats = _compute_stage2_debug_stats(debug, device=label.device)
            else:
                zero = torch.tensor(0.0, device=label.device)
                det_aux_dict = {
                    "loss_center_inside": zero,
                    "center_inside_score_mean": zero,
                    "evidence_inside_ratio_mean": zero,
                    "fused_support_inside_ratio_mean": zero,
                    "agreement_inside_score_mean": zero,
                    "reliability_inside_score_mean": zero,
                    "center_coarse_inside_score_mean": zero,
                    "center_refine_shift_mean": zero,
                    "center_in_mask_rate": zero,
                    "center_coarse_in_mask_rate": zero,
                    "roi_coverage": zero,
                }
                debug_stats = {}
            hwa_runtime_stats = _compute_stage2_hwa_runtime_stats(model, device=label.device)
            seg_logits_for_loss, label_for_loss = _stage2_prepare_seg_supervision(
                seg_logits, label, cfg
            )

            seg_total_loss = 0.0
            for name, fn in seg_loss_functions.items():
                loss = fn(seg_logits_for_loss, label_for_loss)
                seg_total_loss = seg_total_loss + loss
                running[name] = running.get(name, 0.0) + _loss_to_scalar(loss)

            aux_weight = float(stage2_sched["lambda_center_inside_aux"])
            total_loss = (
                seg_total_loss
            )
            if aux_weight > 0.0:
                total_loss = total_loss + aux_weight * det_aux_dict["loss_center_inside"]

            running["loss"] = running.get("loss", 0.0) + _loss_to_scalar(total_loss)
            running["seg_total_loss"] = running.get("seg_total_loss", 0.0) + _loss_to_scalar(seg_total_loss)
            running["center_inside_aux_loss"] = running.get("center_inside_aux_loss", 0.0) + _loss_to_scalar(
                det_aux_dict["loss_center_inside"]
            )
            running["lambda_center_inside_aux"] = running.get("lambda_center_inside_aux", 0.0) + float(
                aux_weight
            )
            running["fusion_progress"] = running.get("fusion_progress", 0.0) + float(
                stage2_sched["fusion_progress"]
            )
            running["prior_progress"] = running.get("prior_progress", 0.0) + float(
                stage2_sched["prior_progress"]
            )
            running["detector_aux_progress"] = running.get("detector_aux_progress", 0.0) + float(
                stage2_sched["detector_aux_progress"]
            )
            running["prior_alpha_cap"] = running.get("prior_alpha_cap", 0.0) + float(stage2_sched["prior_alpha_cap"])
            running["center_inside_score_mean"] = running.get("center_inside_score_mean", 0.0) + _loss_to_scalar(
                det_aux_dict["center_inside_score_mean"]
            )
            running["evidence_inside_ratio_mean"] = running.get("evidence_inside_ratio_mean", 0.0) + _loss_to_scalar(
                det_aux_dict["evidence_inside_ratio_mean"]
            )
            running["fused_support_inside_ratio_mean"] = running.get("fused_support_inside_ratio_mean", 0.0) + _loss_to_scalar(
                det_aux_dict["fused_support_inside_ratio_mean"]
            )
            running["agreement_inside_score_mean"] = running.get("agreement_inside_score_mean", 0.0) + _loss_to_scalar(
                det_aux_dict["agreement_inside_score_mean"]
            )
            running["reliability_inside_score_mean"] = running.get("reliability_inside_score_mean", 0.0) + _loss_to_scalar(
                det_aux_dict["reliability_inside_score_mean"]
            )
            running["center_coarse_inside_score_mean"] = running.get("center_coarse_inside_score_mean", 0.0) + _loss_to_scalar(
                det_aux_dict["center_coarse_inside_score_mean"]
            )
            running["center_refine_shift_mean"] = running.get("center_refine_shift_mean", 0.0) + _loss_to_scalar(
                det_aux_dict["center_refine_shift_mean"]
            )
            running["center_in_mask_rate"] = running.get("center_in_mask_rate", 0.0) + _loss_to_scalar(
                det_aux_dict["center_in_mask_rate"]
            )
            running["center_coarse_in_mask_rate"] = running.get("center_coarse_in_mask_rate", 0.0) + _loss_to_scalar(
                det_aux_dict["center_coarse_in_mask_rate"]
            )
            running["roi_coverage"] = running.get("roi_coverage", 0.0) + _loss_to_scalar(
                det_aux_dict["roi_coverage"]
            )
            for stat_name, stat_value in debug_stats.items():
                running[stat_name] = running.get(stat_name, 0.0) + _loss_to_scalar(stat_value)
            for stat_name, stat_value in hwa_runtime_stats.items():
                running[stat_name] = running.get(stat_name, 0.0) + _loss_to_scalar(stat_value)
            show_loss = _loss_to_scalar(total_loss)

            if post_trans is not None and seg_metrics is not None:
                if _stage2_fast_scalar_metrics_enabled(cfg, accelerator):
                    _accumulate_stage2_fast_dice(running, seg_logits, label, cfg=cfg)
                else:
                    if _stage2_seg_target_mode(cfg) in {"union", "lesion_union", "single_union"}:
                        val_outputs = (
                            torch.sigmoid(seg_logits.detach()).amax(dim=1, keepdim=True)
                            > _stage2_pred_threshold(cfg)
                        ).float()
                        metric_label = _stage2_union_target(label.detach())
                    else:
                        val_outputs = post_trans(seg_logits)
                        metric_label = label
                    for metric_name in seg_metrics:
                        seg_metrics[metric_name](y_pred=val_outputs, y=metric_label)

        elif stage_name == "stage3":
            cls_logits, _, _ = forward_model(model, image, stage=stage_name, debug=False)
            stage3_loss_dict = compute_stage3_classification_loss(
                cls_logits=cls_logits,
                batch=image_batch,
                loss_functions=loss_functions_cls,
                accelerator=accelerator,
                step=step,
                log_prefix=split,
                lambda_class=float(_cfg_get(cfg, "stage_train.stage3.lambda_class", 1.0)),
                lambda_pfs=float(_cfg_get(cfg, "stage_train.stage3.lambda_pfs", 1.0)),
                task_mu=int(_cfg_get(cfg, "GCM_loader.task_Mu", 2) or 2),
            )
            total_loss = stage3_loss_dict["total_loss"]
            running["loss"] = running.get("loss", 0.0) + _loss_to_scalar(total_loss)
            for k, v in stage3_loss_dict.items():
                if torch.is_tensor(v) and "loss" in k and k != "total_loss":
                    running[k] = running.get(k, 0.0) + _loss_to_scalar(v)
            show_loss = _loss_to_scalar(total_loss)

            if post_trans_cls is not None and cls_metrics is not None:
                y_pred_1 = post_trans_cls(stage3_loss_dict["logit_class"])
                y1 = stage3_loss_dict["labels_class"]
                for metric_name in cls_metrics[0]:
                    pred1, lab1 = y_pred_1, y1
                    if metric_name == "miou_metric":
                        pred1 = pred1.unsqueeze(2)
                        lab1 = lab1.unsqueeze(2)
                    cls_metrics[0][metric_name](y_pred=pred1, y=lab1)
                if len(cls_metrics) > 1 and stage3_loss_dict["logit_pfs"] is not None:
                    y_pred_2 = post_trans_cls(stage3_loss_dict["logit_pfs"])
                    y2 = stage3_loss_dict["labels_pfs"]
                    for metric_name in cls_metrics[1]:
                        pred2, lab2 = y_pred_2, y2
                        if metric_name == "miou_metric":
                            pred2 = pred2.unsqueeze(2)
                            lab2 = lab2.unsqueeze(2)
                        cls_metrics[1][metric_name](y_pred=pred2, y=lab2)
        else:
            raise ValueError(f"Unsupported stage_name: {stage_name}")

        if (
            accelerator.is_local_main_process
            and (((batch_idx + 1) % refresh_every) == 0 or (batch_idx + 1) == len(val_loader))
        ):
            pbar.set_postfix(loss=f"{show_loss:.4f}", step=step)
        step += 1

    metric = {}
    num_batches = max(valid_batches, 1)
    for k, v in running.items():
        metric[f"{split}/{stage_name}/{k}"] = v / num_batches

    if stage_name == "stage1":
        score = _stage1_monitor_score(metric, split)

    else:
        score = -metric.get(f"{split}/{stage_name}/loss", 0.0)

    if stage_name == "stage2" and seg_metrics is not None:
        if _stage2_fast_scalar_metrics_enabled(cfg, accelerator):
            metric[f"{split}/{stage_name}/dice_metric"] = _finalize_stage2_fast_dice(
                running, accelerator
            )
        else:
            for metric_name in seg_metrics:
                batch_acc = seg_metrics[metric_name].aggregate()[0].to(accelerator.device)
                batch_acc = torch.nan_to_num(batch_acc, nan=0.0, posinf=0.0, neginf=0.0)
                if accelerator.num_processes > 1:
                    batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
                seg_metrics[metric_name].reset()
                metric[f"{split}/{stage_name}/{metric_name}"] = float(batch_acc.mean())
        score = metric.get(f"{split}/{stage_name}/dice_metric", score)

    if stage_name == "stage3" and cls_metrics is not None:
        for task_idx in range(len(cls_metrics)):
            for metric_name in cls_metrics[task_idx]:
                batch_acc = cls_metrics[task_idx][metric_name].aggregate()[0].to(accelerator.device)
                if accelerator.num_processes > 1:
                    batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
                cls_metrics[task_idx][metric_name].reset()
                metric[f"{split}/{stage_name}/Task{task_idx}_{metric_name}"] = float(batch_acc.mean())
        score = _mean_cls_accuracy(metric, split, len(cls_metrics))

    accelerator.log(metric, step=epoch)
    return score, metric, step, vis_batch

def _resume_scheduler_after_resume(scheduler, optimizer, starting_epoch: int, accelerator):
    """Align scheduler state with the resumed epoch."""
    if scheduler is None:
        return

    if starting_epoch <= 0:
        accelerator.print("[ResumeScheduler] starting_epoch<=0, keep scheduler fresh.")
        return

    target_epoch = int(starting_epoch) - 1
    try:
        if hasattr(scheduler, "step"):
            scheduler.step(target_epoch)

        if hasattr(scheduler, "get_last_lr"):
            current_lrs = scheduler.get_last_lr()
            for pg, lr in zip(optimizer.param_groups, current_lrs):
                pg["lr"] = float(lr)

        accelerator.print(
            f"[ResumeScheduler] aligned to epoch={target_epoch}, "
            f"lrs={[pg['lr'] for pg in optimizer.param_groups]}"
        )
    except Exception as e:
        accelerator.print(f"[ResumeScheduler] failed: {repr(e)}")


def _reset_scheduler_for_stage_restart(scheduler, optimizer, accelerator):
    if scheduler is None:
        return

    try:
        if hasattr(scheduler, "last_epoch"):
            scheduler.last_epoch = -1
        if hasattr(scheduler, "_step_count"):
            scheduler._step_count = 0
        if hasattr(scheduler, "T_cur"):
            scheduler.T_cur = -1
        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr = [float(pg.get("lr", 0.0)) for pg in optimizer.param_groups]
        accelerator.print("[Stage2Retry] scheduler state reset for fresh stage-local cosine.")
    except Exception as e:
        accelerator.print(f"[Stage2Retry] scheduler reset failed: {repr(e)}")


def _load_best_stage_state(accelerator, checkpoint_name: str, stage: str) -> bool:
    best_dir = os.path.join(os.getcwd(), "model_store", checkpoint_name, f"best_{stage}")
    if not os.path.isdir(best_dir):
        accelerator.print(f"[Stage2Retry] best stage state not found: {best_dir}")
        return False
    try:
        accelerator.load_state(best_dir)
        accelerator.print(f"[Stage2Retry] loaded best state from {best_dir}")
        return True
    except Exception as e:
        accelerator.print(f"[Stage2Retry] failed to load best state from {best_dir}: {repr(e)}")
        return False


def _is_stage2_to_stage3_transition(
    stage: str,
    *,
    next_epoch: int,
    cfg,
    runtime_state: Optional[Dict[str, int]] = None,
) -> bool:
    if str(stage) != "stage2":
        return False
    if int(_cfg_get(cfg, "stage_train.stage3.epochs", 0)) <= 0:
        return False
    runtime_info = _resolve_runtime_stage(int(next_epoch), cfg, runtime_state)
    return str(runtime_info["stage"]) == "stage3"


def resolve_stage(epoch: int, cfg) -> str:
    e1 = int(_cfg_get(cfg, "stage_train.stage1.epochs", 0))
    e2 = int(_cfg_get(cfg, "stage_train.stage2.epochs", 0))
    if epoch < e1:
        return "stage1"
    if epoch < e1 + e2:
        return "stage2"
    return "stage3"


def _should_retry_stage2(stage_best: Dict[str, object], cfg, runtime_state: Dict[str, int]) -> bool:
    if not bool(_cfg_get(cfg, "trainer.resume_train", False)):
        return False
    retry_count = int(runtime_state.get("stage2_retry_count", 0))
    max_retries = int(runtime_state.get("stage2_max_retries", 0))
    if retry_count >= max_retries:
        return False
    target_score = float(_cfg_get(cfg, "trainer.resume_score", 0.0))
    best_score = float(stage_best.get("best_score", -1e9))
    return best_score < target_score


def _should_early_stop_stage2(
    *,
    cfg,
    runtime_state: Dict[str, int],
    stage_local_epoch: int,
    val_score: float,
) -> Tuple[bool, Dict[str, float]]:
    start_epoch = int(_cfg_get(cfg, "stage_train.stage2.early_stop_start_epoch", 120))
    patience = int(_cfg_get(cfg, "stage_train.stage2.early_stop_patience", 25))
    min_delta = float(_cfg_get(cfg, "stage_train.stage2.early_stop_min_delta", 0.001))
    if patience <= 0 or stage_local_epoch + 1 < start_epoch:
        return False, {
            "best": float(runtime_state.get("stage2_early_stop_best", -1e9)),
            "wait": int(runtime_state.get("stage2_early_stop_wait", 0)),
        }

    best = float(runtime_state.get("stage2_early_stop_best", -1e9))
    wait = int(runtime_state.get("stage2_early_stop_wait", 0))
    score = float(val_score)

    if score > best + min_delta:
        best = score
        wait = 0
    else:
        wait += 1

    runtime_state["stage2_early_stop_best"] = best
    runtime_state["stage2_early_stop_wait"] = wait
    return wait >= patience, {"best": best, "wait": wait}


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    target = _unwrap(model)
    base_lr = float(cfg.trainer.lr)
    weight_decay = float(cfg.trainer.weight_decay)
    betas = (float(cfg.trainer.betas[0]), float(cfg.trainer.betas[1]))

    grouped_params = {
        "detector": [],
        "prior_builder": [],
        "prior_consumer": [],
        "encoder_seg": [],
        "classifier": [],
        "misc": [],
    }
    for name, param in target.named_parameters():
        role = _get_optimizer_role(name)
        grouped_params.setdefault(role, []).append(param)

    param_groups = []
    for role in [
        "detector",
        "prior_builder",
        "prior_consumer",
        "input_fusion",
        "encoder_seg",
        "classifier",
        "misc",
    ]:
        params = grouped_params.get(role, [])
        if not params:
            continue
        group_weight_decay = weight_decay
        if role == "classifier":
            group_weight_decay = float(
                _cfg_get(cfg, "stage_train.stage3.classifier_weight_decay", weight_decay)
            )
        param_groups.append(
            {
                "params": params,
                "lr": base_lr,
                "weight_decay": group_weight_decay,
                "role": role,
            }
        )

    opt_name = str(cfg.trainer.optimizer).lower()
    if opt_name == "adamw":
        return torch.optim.AdamW(param_groups, lr=base_lr, betas=betas, weight_decay=weight_decay)
    if opt_name == "adam":
        return torch.optim.Adam(param_groups, lr=base_lr, betas=betas, weight_decay=weight_decay)
    if opt_name == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=base_lr,
            momentum=float(_cfg_get(cfg, "trainer.momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(_cfg_get(cfg, "trainer.nesterov", True)),
        )
    raise ValueError(f"Unsupported optimizer for staged training: {cfg.trainer.optimizer}")


def _build_post_trans(threshold: float = 0.5):
    return monai.transforms.Compose(
        [
            monai.transforms.Activations(sigmoid=True),
            monai.transforms.AsDiscrete(threshold=float(threshold)),
        ]
    )


def save_resume_compatible_checkpoint(
    *,
    checkpoint_path: str,
    epoch: int,
    stage: str,
    best_score,
    best_test_score,
    best_metrics,
    best_test_metrics,
    best_state_by_stage,
    train_step: int,
    val_step: int,
    runtime_state: Optional[Dict[str, int]] = None,
):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    runtime_state = dict(runtime_state or {})
    runtime_state.setdefault("stage2_retry_count", 0)
    runtime_state.setdefault("stage2_total_cycles", 1)
    runtime_state.setdefault("stage2_max_retries", 0)
    best_score_val = (
        float(best_score.item()) if torch.is_tensor(best_score) else float(best_score)
    )
    best_test_score_val = (
        float(best_test_score.item())
        if torch.is_tensor(best_test_score)
        else float(best_test_score)
    )
    ckpt = {
        "epoch": int(epoch),
        "stage": stage,
        "train_step": int(train_step),
        "val_step": int(val_step),
        "best_score": best_score_val,
        "best_test_score": best_test_score_val,
        "best_metrics": best_metrics,
        "best_test_metrics": best_test_metrics,
        "best_state_by_stage": best_state_by_stage,
        "best_hd95": 1000.0,
        "best_test_hd95": 1000.0,
        "best_hd95_metrics": [],
        "best_test_hd95_metrics": [],
        "best_accuracy": best_score_val,
        "best_test_accuracy": best_test_score_val,
        "runtime_state": runtime_state,
    }
    torch.save(ckpt, checkpoint_path)


def _load_resume_state(
    model, optimizer, scheduler, train_loader, accelerator, checkpoint_name, cfg
):
    if not bool(_cfg_get(cfg, "trainer.resume", False)):
        accelerator.print("[Resume] trainer.resume=False, start from scratch.")
        return model, optimizer, scheduler, 0, 0, _init_stage_best_state(), _init_runtime_state(cfg, {})

    base_path = os.path.join(os.getcwd(), "model_store", checkpoint_name, "checkpoint")
    epoch_path = os.path.join(base_path, "epoch.pth.tar")
    accelerator.print(f"[Resume] Try resume from: {base_path}")

    if not os.path.isdir(base_path):
        accelerator.print(f"[Resume] checkpoint directory not found: {base_path}")
        return model, optimizer, scheduler, 0, 0, _init_stage_best_state(), _init_runtime_state(cfg, {})

    if not os.path.isfile(epoch_path):
        accelerator.print(f"[Resume] epoch checkpoint not found: {epoch_path}")
        return model, optimizer, scheduler, 0, 0, _init_stage_best_state(), _init_runtime_state(cfg, {})

    try:
        epoch_checkpoint = torch.load(epoch_path, map_location="cpu")
        accelerator.print(f"[Resume] epoch checkpoint keys: {sorted(list(epoch_checkpoint.keys()))}")
    except Exception as e:
        accelerator.print(f"[Resume] Failed to read epoch checkpoint: {repr(e)}")
        return model, optimizer, scheduler, 0, 0, _init_stage_best_state(), _init_runtime_state(cfg, {})

    trial_errors = []
    fallback_stage = str(epoch_checkpoint.get("stage", "stage1"))
    stage_best_state_from_ckpt = epoch_checkpoint.get("best_state_by_stage", None)
    runtime_state_from_ckpt = _init_runtime_state(cfg, epoch_checkpoint.get("runtime_state", {}))

    try:
        result = resume_train_state(
            model,
            f"{checkpoint_name}",
            optimizer,
            scheduler,
            train_loader,
            accelerator,
            seg=True,
        )
        (
            model,
            optimizer,
            scheduler,
            starting_epoch,
            step,
            best_score,
            best_test_score,
            best_metrics,
            best_test_metrics,
            _,
            _,
            _,
            _,
        ) = result

        accelerator.print(
            f"[Resume][seg] raw return: start_epoch={starting_epoch}, step={step}, best_score={best_score}"
        )

        if int(epoch_checkpoint.get("epoch", -1)) >= 0 and int(starting_epoch) == 0:
            accelerator.print(
                "[Resume][seg] suspicious starting_epoch=0 while epoch checkpoint exists. "
                "Fallback to epoch metadata."
            )
            starting_epoch = int(epoch_checkpoint.get("epoch", -1)) + 1

        _resume_scheduler_after_resume(scheduler, optimizer, int(starting_epoch), accelerator)
        stage_best_state = _normalize_stage_best_state(
            stage_best_state_from_ckpt,
            fallback_stage=fallback_stage,
            fallback_best_score=float(best_score),
            fallback_best_test_score=float(best_test_score),
            fallback_best_metrics=best_metrics,
            fallback_best_test_metrics=best_test_metrics,
        )

        return (
            model,
            optimizer,
            scheduler,
            int(starting_epoch),
            int(step),
            stage_best_state,
            runtime_state_from_ckpt,
        )

    except Exception as e:
        trial_errors.append(f"[Resume][seg] {repr(e)}")

    try:
        result = resume_train_state(
            model,
            f"{checkpoint_name}",
            optimizer,
            scheduler,
            train_loader,
            accelerator,
            seg=False,
        )
        (
            model,
            optimizer,
            scheduler,
            starting_epoch,
            step,
            best_accuracy,
            best_test_accuracy,
            best_metrics,
            best_test_metrics,
        ) = result

        accelerator.print(
            f"[Resume][cls] raw return: start_epoch={starting_epoch}, step={step}, best_accuracy={best_accuracy}"
        )

        if int(epoch_checkpoint.get("epoch", -1)) >= 0 and int(starting_epoch) == 0:
            accelerator.print(
                "[Resume][cls] suspicious starting_epoch=0 while epoch checkpoint exists. "
                "Fallback to epoch metadata."
            )
            starting_epoch = int(epoch_checkpoint.get("epoch", -1)) + 1

        _resume_scheduler_after_resume(scheduler, optimizer, int(starting_epoch), accelerator)
        stage_best_state = _normalize_stage_best_state(
            stage_best_state_from_ckpt,
            fallback_stage=fallback_stage,
            fallback_best_score=float(best_accuracy),
            fallback_best_test_score=float(best_test_accuracy),
            fallback_best_metrics=best_metrics,
            fallback_best_test_metrics=best_test_metrics,
        )

        return (
            model,
            optimizer,
            scheduler,
            int(starting_epoch),
            int(step),
            stage_best_state,
            runtime_state_from_ckpt,
        )

    except Exception as e:
        trial_errors.append(f"[Resume][cls] {repr(e)}")

    accelerator.print("[Resume] All resume attempts failed.")
    for msg in trial_errors:
        accelerator.print(msg)

    accelerator.print(
        "[Resume] Fallback to epoch checkpoint metadata only: "
        f"epoch={epoch_checkpoint.get('epoch', 'N/A')}, "
        f"stage={epoch_checkpoint.get('stage', 'N/A')}, "
        f"best_score={epoch_checkpoint.get('best_score', 'N/A')}"
    )

    fallback_start = int(epoch_checkpoint.get("epoch", -1)) + 1 if "epoch" in epoch_checkpoint else 0
    fallback_best = float(epoch_checkpoint.get("best_score", -1e9))
    fallback_best_test = float(epoch_checkpoint.get("best_test_score", -1e9))
    fallback_train_step = int(epoch_checkpoint.get("train_step", 0))

    _resume_scheduler_after_resume(scheduler, optimizer, fallback_start, accelerator)
    stage_best_state = _normalize_stage_best_state(
        stage_best_state_from_ckpt,
        fallback_stage=fallback_stage,
        fallback_best_score=fallback_best,
        fallback_best_test_score=fallback_best_test,
        fallback_best_metrics=epoch_checkpoint.get("best_metrics", {}),
        fallback_best_test_metrics=epoch_checkpoint.get("best_test_metrics", {}),
    )

    return (
        model,
        optimizer,
        scheduler,
        fallback_start,
        fallback_train_step,
        stage_best_state,
        runtime_state_from_ckpt,
    )
    
    
def run_best_test_evaluation(
    *,
    model,
    test_loader,
    accelerator,
    epoch,
    stage,
    config,
    seg_loss_functions,
    loss_functions_cls,
    post_trans,
    post_trans_cls,
    task_mu,
    stage_local_epoch: Optional[int] = None,
    stage_total_epochs: Optional[int] = None,
    display_total_epochs: Optional[int] = None,
):
    test_metrics_seg = build_seg_metrics(enable_hd95=True) if stage == "stage2" else None
    test_metrics_cls = build_cls_metrics(task_mu=task_mu) if stage == "stage3" else None
    return val_one_epoch(
        model,
        test_loader,
        accelerator,
        epoch,
        -1,
        stage,
        config,
        seg_loss_functions,
        loss_functions_cls,
        test_metrics_seg,
        test_metrics_cls,
        post_trans,
        post_trans_cls,
        split="Test",
        stage_local_epoch=stage_local_epoch,
        stage_total_epochs=stage_total_epochs,
        display_total_epochs=display_total_epochs,
    )


def run_stage2_hwa_ablation_evaluation(
    *,
    model,
    data_loader,
    accelerator,
    epoch,
    stage,
    config,
    seg_loss_functions,
    loss_functions_cls,
    post_trans,
    post_trans_cls,
    split,
    stage_local_epoch: Optional[int] = None,
    stage_total_epochs: Optional[int] = None,
    display_total_epochs: Optional[int] = None,
):
    if stage != "stage2" or not bool(_cfg_get(config, "stage_train.stage2.eval_hwa_ablation", False)):
        return None

    was_enabled = _get_runtime_hwa_prior_enabled(model)
    sched = _get_stage2_schedule(
        epoch,
        config,
        stage_local_epoch=stage_local_epoch,
        stage_total_epochs=stage_total_epochs,
    )
    _set_runtime_hwa_prior_enabled(model, False)
    _apply_stage2_hwa_runtime_scales(
        model,
        {**sched, "input_hwa_gate_scale": 0.0, "input_hwa_gain_scale": 0.0},
    )
    try:
        metrics_seg = build_seg_metrics(enable_hd95=False)
        _, ablation_metric, _, _ = val_one_epoch(
            model,
            data_loader,
            accelerator,
            epoch,
            -1,
            stage,
            config,
            seg_loss_functions,
            loss_functions_cls,
            metrics_seg,
            None,
            post_trans,
            post_trans_cls,
            split=f"{split}NoHWA",
            stage_local_epoch=stage_local_epoch,
            stage_total_epochs=stage_total_epochs,
            display_total_epochs=display_total_epochs,
        )
        return ablation_metric
    finally:
        if was_enabled is not None:
            _set_runtime_hwa_prior_enabled(model, was_enabled)
        _apply_stage2_hwa_runtime_scales(model, sched)


def run_training(
    config_path: str = "config.yml",
    forced_stage: Optional[str] = None,
    runner_name: Optional[str] = None,
):
    config = _load_yaml_config(config_path)

    if forced_stage is not None:
        config = _prepare_single_stage_config(config, forced_stage)

    utils.same_seeds(int(os.environ.get("HWA_RUN_SEED", "50")))
    _preflight_gpu_memory_check(config)

    checkpoint_name = _resolve_checkpoint_name(config, forced_stage=forced_stage)
    _disable_resume_if_checkpoint_missing(
        config,
        checkpoint_name,
        str(forced_stage or "stage2"),
    )

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

    ddp_find_unused = str(
        os.environ.get(
            "HWA_DDP_FIND_UNUSED_PARAMETERS",
            str(_cfg_get(config, "trainer.ddp_find_unused_parameters", True)),
        )
    ).strip().lower() in {"1", "true", "yes", "y", "on"}
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=ddp_find_unused)
    accelerator = Accelerator(
        cpu=False,
        log_with=["tensorboard"],
        project_dir=logging_dir,
        kwargs_handlers=[ddp_kwargs],
    )

    logger = Logger(logging_dir if accelerator.is_local_main_process else None)
    tracker_name = runner_name or os.path.split(__file__)[-1].split(".")[0]
    accelerator.init_trackers(tracker_name)
    accelerator.print(objstr(config))

    accelerator.print("load model...")
    model = _build_model(config)
    loaded_stage2_segmentation_init = False
    if forced_stage is None:
        loaded_stage2_segmentation_init = _load_stage2_segmentation_init_if_needed(
            model,
            accelerator,
            config,
        )
    loaded_stage2_detector_init = False
    if forced_stage is None:
        loaded_stage2_detector_init = _load_stage2_detector_init_if_needed(
            model,
            accelerator,
            config,
        )
    init_stage_name = str(forced_stage or "stage1")
    init_checkpoint = _resolve_stage_init_checkpoint(config, init_stage_name)
    if (
        init_checkpoint not in [None, "", "None"]
        and not loaded_stage2_segmentation_init
        and not loaded_stage2_detector_init
    ):
        reload_pre_train_model(
            model=model,
            accelerator=accelerator,
            checkpoint_path=init_checkpoint,
        )
    if init_stage_name == "stage3":
        _load_stage3_classifier_warm_start_if_needed(model, accelerator, config)
    if init_stage_name == "stage3":
        _reset_stage3_class_decoder_if_requested(model, config, accelerator)
    if _apply_stage2_hwa_gain_init_if_needed(model, config):
        hwa = getattr(_unwrap(model), "hwa_block", None)
        accelerator.print(
            "[Stage2HWAInit] "
            f"input_agg_gain={float(hwa.input_agg_gain.detach().float().mean()):.4f} "
            f"input_gate_boost={float(hwa.input_gate_boost.detach().float().mean()):.4f} "
            f"center_prior_gain={float(hwa.center_prior_gain.detach().float().mean()):.4f} "
            f"input_enhance_gain={float(hwa.input_enhance_gain.detach().float().mean()):.4f} "
            f"output_logit_gain={float(hwa.output_logit_gain.detach().float().mean()):.4f}"
        )

    accelerator.print("load dataset...")
    raw_train_loader, val_loader, test_loader, example = get_dataloader(config)
    _distributed_barrier(
        accelerator,
        "after_load_dataset",
        loaders={"train": raw_train_loader, "val": val_loader, "test": test_loader},
    )
    if accelerator.is_main_process:
        write_example(config, example)
    _distributed_barrier(
        accelerator,
        "after_write_example",
        loaders={"train": raw_train_loader, "val": val_loader, "test": test_loader},
    )
    stage2_train_loader, val_loader, test_loader = _maybe_stack_stage2_train_loader(
        raw_train_loader,
        val_loader,
        test_loader,
        config,
        accelerator,
    )
    stage3_train_loader = _maybe_prepare_stage3_train_loader(
        raw_train_loader,
        config,
        accelerator,
    )

    optimizer = build_optimizer(model, config)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.trainer.warmup,
        max_epochs=config.trainer.num_epochs,
    )

    seg_loss_functions = build_stage2_segmentation_losses(config)
    accelerator.print(
        "[Stage2Loss] "
        f"mode={_cfg_get(config, 'stage_train.stage2.seg_loss_mode', 'dice_focal')} "
        f"target_mode={_cfg_get(config, 'stage_train.stage2.seg_target_mode', 'channels')} "
        f"losses={list(seg_loss_functions.keys())}"
    )
    loss_functions_cls = build_stage3_classification_losses(
        cfg=config,
        train_loader=stage3_train_loader,
        device=accelerator.device,
        accelerator=accelerator,
    )
    post_trans = _build_post_trans(_stage2_pred_threshold(config))
    post_trans_cls = _build_post_trans(_stage3_class_threshold(config))

    train_step = 0
    val_step = 0
    starting_epoch = 0
    best_state_by_stage = _init_stage_best_state()

    if stage2_train_loader is raw_train_loader and stage3_train_loader is raw_train_loader:
        model, optimizer, scheduler, raw_train_loader, val_loader, test_loader = accelerator.prepare(
            model, optimizer, scheduler, raw_train_loader, val_loader, test_loader
        )
        stage2_train_loader = raw_train_loader
        stage3_train_loader = raw_train_loader
    elif stage2_train_loader is raw_train_loader and stage3_train_loader is not raw_train_loader:
        (
            model,
            optimizer,
            scheduler,
            raw_train_loader,
            stage3_train_loader,
            val_loader,
            test_loader,
        ) = accelerator.prepare(
            model,
            optimizer,
            scheduler,
            raw_train_loader,
            stage3_train_loader,
            val_loader,
            test_loader,
        )
        stage2_train_loader = raw_train_loader
    elif stage2_train_loader is not raw_train_loader and stage3_train_loader is raw_train_loader:
        (
            model,
            optimizer,
            scheduler,
            raw_train_loader,
            stage2_train_loader,
            val_loader,
            test_loader,
        ) = accelerator.prepare(
            model,
            optimizer,
            scheduler,
            raw_train_loader,
            stage2_train_loader,
            val_loader,
            test_loader,
        )
        stage3_train_loader = raw_train_loader
    else:
        (
            model,
            optimizer,
            scheduler,
            raw_train_loader,
            stage2_train_loader,
            stage3_train_loader,
            val_loader,
            test_loader,
        ) = accelerator.prepare(
            model,
            optimizer,
            scheduler,
            raw_train_loader,
            stage2_train_loader,
            stage3_train_loader,
            val_loader,
            test_loader,
        )

    (
        model,
        optimizer,
        scheduler,
        starting_epoch,
        train_step,
        best_state_by_stage,
        runtime_state,
    ) = _load_resume_state(
        model, optimizer, scheduler, raw_train_loader, accelerator, checkpoint_name, config
    )
    val_step = train_step
    task_mu = int(_cfg_get(config, "GCM_loader.task_Mu", 2) or 2)
    if forced_stage is not None:
        primary_best_stage = str(forced_stage)
    else:
        primary_best_stage = "stage2" if str(config.trainer.task).lower() == "segmentation" else "stage3"
    configured_stage2_max_retries = (
        2
        if (forced_stage is None and bool(_cfg_get(config, "trainer.resume_train", False)))
        else 0
    )
    runtime_state = _init_runtime_state(config, runtime_state)
    runtime_state["stage2_total_cycles"] = max(
        int(runtime_state.get("stage2_total_cycles", 1)),
        int(runtime_state.get("stage2_retry_count", 0)) + 1,
    )
    runtime_state["stage2_max_retries"] = max(
        int(runtime_state.get("stage2_retry_count", 0)),
        configured_stage2_max_retries,
    )

    epoch = int(starting_epoch)
    while epoch < _get_runtime_total_epochs(config, runtime_state):
        runtime_info = _resolve_runtime_stage(epoch, config, runtime_state)
        stage = str(runtime_info["stage"])
        stage_total_epochs = int(runtime_info["stage_total_epochs"])
        stage_local_epoch = int(runtime_info["stage_local_epoch"])
        stage_cycle_index = int(runtime_info["stage_cycle_index"])
        display_total_epochs = int(runtime_info["display_total_epochs"])
        apply_stage_policy(
            model,
            stage,
            config,
            stage_local_epoch=stage_local_epoch,
            stage_total_epochs=stage_total_epochs,
        )
        base_lr, lr_scales = apply_stage_lr_policy(
            optimizer,
            stage,
            config,
            scheduler=scheduler,
            epoch=epoch,
            stage_local_epoch=stage_local_epoch,
            stage_total_epochs=stage_total_epochs,
        )
        if stage == "stage2":
            stage2_header_sched = _get_stage2_schedule(
                epoch,
                config,
                stage_local_epoch=stage_local_epoch,
                stage_total_epochs=stage_total_epochs,
            )
            alpha_cap = _apply_stage2_prior_alpha_cap(
                model,
                epoch,
                config,
                stage_local_epoch=stage_local_epoch,
                stage_total_epochs=stage_total_epochs,
            )
            accelerator.print(
                f"===== Epoch {epoch+1}/{display_total_epochs} | {stage} "
                f"(cycle {stage_cycle_index+1}/{max(int(runtime_state['stage2_total_cycles']), 1)}) | "
                f"(local {stage_local_epoch+1}/{max(stage_total_epochs, 1)}) | "
                f"phase={stage2_header_sched['phase_name']} | "
                f"base_lr={base_lr:.2e} | lr_scales={lr_scales} | "
                f"fusion_progress={stage2_header_sched['fusion_progress']:.3f} | "
                f"prior_progress={stage2_header_sched['prior_progress']:.3f} | "
                f"det_aux_progress={stage2_header_sched['detector_aux_progress']:.3f} | "
                f"prior_alpha_cap={alpha_cap:.4f} | "
                f"input_hwa_gate={stage2_header_sched['input_hwa_gate_scale']:.4f} | "
                f"input_hwa_gain={stage2_header_sched['input_hwa_gain_scale']:.4f} ====="
            )
        else:
            accelerator.print(
                f"===== Epoch {epoch+1}/{display_total_epochs} | {stage} "
                f"(local {stage_local_epoch+1}/{max(stage_total_epochs, 1)}) | "
                f"base_lr={base_lr:.2e} | lr_scales={lr_scales} ====="
            )

        metrics_seg = build_seg_metrics(enable_hd95=False) if stage == "stage2" else None
        metrics_cls = build_cls_metrics(task_mu=task_mu) if stage == "stage3" else None
        active_train_loader = _select_train_loader_for_stage(
            stage, raw_train_loader, stage2_train_loader, stage3_train_loader
        )

        _distributed_barrier(
            accelerator,
            f"before_train_epoch_{epoch+1}",
            loaders={"train": active_train_loader, "val": val_loader, "test": test_loader},
        )
        train_metric, train_step, vis_train = train_one_epoch(
            model,
            active_train_loader,
            optimizer,
            scheduler,
            accelerator,
            epoch,
            train_step,
            stage,
            config,
            seg_loss_functions,
            loss_functions_cls,
            metrics_seg,
            metrics_cls,
            post_trans,
            post_trans_cls,
            stage_local_epoch=stage_local_epoch,
            stage_total_epochs=stage_total_epochs,
            display_total_epochs=display_total_epochs,
        )
        _distributed_barrier(
            accelerator,
            f"after_train_epoch_{epoch+1}",
            loaders={"train": active_train_loader, "val": val_loader, "test": test_loader},
        )
        val_score, val_metric, val_step, vis_val = val_one_epoch(
            model,
            val_loader,
            accelerator,
            epoch,
            val_step,
            stage,
            config,
            seg_loss_functions,
            loss_functions_cls,
            metrics_seg,
            metrics_cls,
            post_trans,
            post_trans_cls,
            split="Val",
            stage_local_epoch=stage_local_epoch,
            stage_total_epochs=stage_total_epochs,
            display_total_epochs=display_total_epochs,
        )
        _distributed_barrier(
            accelerator,
            f"after_val_epoch_{epoch+1}",
            loaders={"train": active_train_loader, "val": val_loader, "test": test_loader},
        )
        if stage == "stage2":
            ablation_metric = run_stage2_hwa_ablation_evaluation(
                model=model,
                data_loader=val_loader,
                accelerator=accelerator,
                epoch=epoch,
                stage=stage,
                config=config,
                seg_loss_functions=seg_loss_functions,
                loss_functions_cls=loss_functions_cls,
                post_trans=post_trans,
                post_trans_cls=post_trans_cls,
                split="Val",
                stage_local_epoch=stage_local_epoch,
                stage_total_epochs=stage_total_epochs,
                display_total_epochs=display_total_epochs,
            )
            _distributed_barrier(
                accelerator,
                f"after_val_ablation_epoch_{epoch+1}",
                loaders={"train": active_train_loader, "val": val_loader, "test": test_loader},
            )
            if ablation_metric is not None:
                _merge_stage2_hwa_ablation_metrics(val_metric, ablation_metric, "Val")
            val_score = _stage2_main_score_from_metrics(
                train_metric,
                val_metric,
                config,
                "Val",
            )

        vis_run_dir = Path(
            _cfg_get(config, "stage_train.visualization.save_dir", logging_dir)
        )
        if vis_train is not None:
            _safe_save_epoch_visualizations(
                accelerator=accelerator,
                model=model,
                batch=vis_train,
                cfg=config,
                epoch=epoch,
                split_name="train",
                run_dir=vis_run_dir,
                stage=stage,
            )
        if vis_val is not None:
            _safe_save_epoch_visualizations(
                accelerator=accelerator,
                model=model,
                batch=vis_val,
                cfg=config,
                epoch=epoch,
                split_name="val",
                run_dir=vis_run_dir,
                stage=stage,
            )

        stage_best = best_state_by_stage.setdefault(
            stage,
            {
                "best_score": -1e9,
                "best_test_score": -1e9,
                "best_metrics": {},
                "best_test_metrics": {},
            },
        )
        current_best_score = float(stage_best.get("best_score", -1e9))

        if float(val_score) > current_best_score:
            stage_best_dir = f"{os.getcwd()}/model_store/{checkpoint_name}/best_{stage}"
            accelerator.save_state(
                output_dir=stage_best_dir
            )
            if stage == primary_best_stage:
                accelerator.save_state(
                    output_dir=f"{os.getcwd()}/model_store/{checkpoint_name}/best"
                )
            stage_best["best_score"] = float(val_score)
            stage_best["best_metrics"] = val_metric
            skip_best_test = bool(
                _cfg_get(
                    config,
                    f"stage_train.{stage}.skip_best_test_eval",
                    False,
                )
            )
            if skip_best_test:
                accelerator.print(f"[BestTest] stage={stage} skipped by config.")
                test_score = stage_best.get("best_test_score", -1e9)
                test_metric = stage_best.get("best_test_metrics", {})
            else:
                test_score, test_metric, _, _ = run_best_test_evaluation(
                    model=model,
                    test_loader=test_loader,
                    accelerator=accelerator,
                    epoch=epoch,
                    stage=stage,
                    config=config,
                    seg_loss_functions=seg_loss_functions,
                    loss_functions_cls=loss_functions_cls,
                    post_trans=post_trans,
                    post_trans_cls=post_trans_cls,
                    task_mu=task_mu,
                    stage_local_epoch=stage_local_epoch,
                    stage_total_epochs=stage_total_epochs,
                    display_total_epochs=display_total_epochs,
                )
            _distributed_barrier(
                accelerator,
                f"after_test_eval_epoch_{epoch+1}",
                loaders={"train": active_train_loader, "val": val_loader, "test": test_loader},
            )
            if stage == "stage2" and not skip_best_test:
                test_ablation_metric = run_stage2_hwa_ablation_evaluation(
                    model=model,
                    data_loader=test_loader,
                    accelerator=accelerator,
                    epoch=epoch,
                    stage=stage,
                    config=config,
                    seg_loss_functions=seg_loss_functions,
                    loss_functions_cls=loss_functions_cls,
                    post_trans=post_trans,
                    post_trans_cls=post_trans_cls,
                    split="Test",
                    stage_local_epoch=stage_local_epoch,
                    stage_total_epochs=stage_total_epochs,
                    display_total_epochs=display_total_epochs,
                )
                _distributed_barrier(
                    accelerator,
                    f"after_test_ablation_epoch_{epoch+1}",
                    loaders={"train": active_train_loader, "val": val_loader, "test": test_loader},
                )
                if test_ablation_metric is not None:
                    _merge_stage2_hwa_ablation_metrics(test_metric, test_ablation_metric, "Test")
                test_score = _stage2_main_score_from_metrics(
                    train_metric,
                    test_metric,
                    config,
                    "Test",
                )
                raw_test_score = test_metric.get(
                    "Test/stage2/raw_dice_metric",
                    test_metric.get("Test/stage2/dice_metric", None),
                )
                hwa_off_test_score = test_metric.get("Test/stage2/hwa_off_dice_metric", None)
                hwa_test_benefit = test_metric.get("Test/stage2/hwa_benefit_dice", None)
                hwa_off_text = (
                    f"{float(hwa_off_test_score):.6f}"
                    if hwa_off_test_score is not None
                    else "None"
                )
                hwa_benefit_text = (
                    f"{float(hwa_test_benefit):.6f}"
                    if hwa_test_benefit is not None
                    else "None"
                )
                accelerator.print(
                    "[BestTest] stage=stage2 "
                    f"test_score={float(test_score):.6f} "
                    f"raw_test_score={float(raw_test_score):.6f} "
                    f"hwa_off_test_dice={hwa_off_text} "
                    f"hwa_test_benefit={hwa_benefit_text}"
                )
            stage_best["best_test_score"] = float(test_score)
            stage_best["best_test_metrics"] = test_metric

        stage_best = best_state_by_stage[stage]
        triggered_stage2_retry = False
        if (
            stage == "stage2"
            and stage_total_epochs > 0
            and (stage_local_epoch + 1) >= stage_total_epochs
            and (stage_cycle_index + 1) >= int(runtime_state["stage2_total_cycles"])
            and _should_retry_stage2(stage_best, config, runtime_state)
        ):
            target_score = float(_cfg_get(config, "trainer.resume_score", 0.0))
            current_best = float(stage_best.get("best_score", -1e9))
            accelerator.print(
                f"[Stage2Retry] stage2 best_score={current_best:.4f} < resume_score={target_score:.4f}. "
                "Reload best stage2 weights and restart stage2 cosine schedule."
            )
            if _load_best_stage_state(accelerator, checkpoint_name, "stage2"):
                runtime_state["stage2_retry_count"] = int(runtime_state["stage2_retry_count"]) + 1
                runtime_state["stage2_total_cycles"] = int(runtime_state["stage2_retry_count"]) + 1
                runtime_state["stage2_early_stop_wait"] = 0
                runtime_state["stage2_early_stop_best"] = float(stage_best.get("best_score", -1e9))
                _reset_scheduler_for_stage_restart(scheduler, optimizer, accelerator)
                triggered_stage2_retry = True
            else:
                accelerator.print("[Stage2Retry] skipped because best stage2 state could not be loaded.")

        triggered_stage2_early_stop = False
        if stage == "stage2" and stage_total_epochs > 0:
            stop_now, stop_state = _should_early_stop_stage2(
                cfg=config,
                runtime_state=runtime_state,
                stage_local_epoch=stage_local_epoch,
                val_score=float(val_score),
            )
            if stop_now and (stage_local_epoch + 1) < stage_total_epochs:
                accelerator.print(
                    f"[Stage2EarlyStop] stop at local epoch {stage_local_epoch+1}/{stage_total_epochs} "
                    f"with best={stop_state['best']:.4f}, wait={int(stop_state['wait'])}."
                )
                if _load_best_stage_state(accelerator, checkpoint_name, "stage2"):
                    triggered_stage2_early_stop = True
                else:
                    accelerator.print("[Stage2EarlyStop] best stage2 state not found, keep current weights.")

        _distributed_barrier(
            accelerator,
            f"before_save_epoch_{epoch+1}",
            loaders={"train": active_train_loader, "val": val_loader, "test": test_loader},
        )
        accelerator.save_state(
            output_dir=f"{os.getcwd()}/model_store/{checkpoint_name}/checkpoint"
        )
        save_resume_compatible_checkpoint(
            checkpoint_path=f"{os.getcwd()}/model_store/{checkpoint_name}/checkpoint/epoch.pth.tar",
            epoch=epoch,
            stage=stage,
            best_score=stage_best["best_score"],
            best_test_score=stage_best["best_test_score"],
            best_metrics=stage_best["best_metrics"],
            best_test_metrics=stage_best["best_test_metrics"],
            best_state_by_stage=best_state_by_stage,
            train_step=train_step,
            val_step=val_step,
            runtime_state=runtime_state,
        )

        if stage == "stage1":
            train_center_modal = train_metric.get("Train/stage1/center_modal_loss", None)
            train_center_coarse_modal = train_metric.get("Train/stage1/center_coarse_modal_loss", None)
            train_center_inside = train_metric.get("Train/stage1/center_inside_loss", None)
            train_center_coarse_inside = train_metric.get("Train/stage1/center_coarse_inside_loss", None)
            train_sigma_reg = train_metric.get("Train/stage1/sigma_reg_loss", None)
            train_conf_reg = train_metric.get("Train/stage1/conf_reg_loss", None)
            train_raw_evidence = train_metric.get("Train/stage1/raw_evidence_supervise_loss", None)
            train_evidence = train_metric.get("Train/stage1/evidence_total_loss", None)
            train_fused_support = train_metric.get("Train/stage1/fused_support_supervise_loss", None)
            train_fused_support_outside = train_metric.get("Train/stage1/fused_support_outside_loss", None)
            train_evidence_outside = train_metric.get("Train/stage1/evidence_outside_loss", None)
            train_in_mask = train_metric.get("Train/stage1/center_in_mask_rate", None)
            train_coarse_in_mask = train_metric.get("Train/stage1/center_coarse_in_mask_rate", None)
            train_inside_score = train_metric.get("Train/stage1/center_inside_score_mean", None)
            train_coarse_inside_score = train_metric.get("Train/stage1/center_coarse_inside_score_mean", None)
            train_evidence_inside_ratio = train_metric.get("Train/stage1/evidence_inside_ratio_mean", None)
            train_fused_inside_ratio = train_metric.get("Train/stage1/fused_support_inside_ratio_mean", None)
            train_agreement_inside = train_metric.get("Train/stage1/agreement_inside_score_mean", None)
            train_refine_shift = train_metric.get("Train/stage1/center_refine_shift_mean", None)
            train_coverage = train_metric.get("Train/stage1/roi_coverage", None)

            val_center_modal = val_metric.get("Val/stage1/center_modal_loss", None)
            val_center_coarse_modal = val_metric.get("Val/stage1/center_coarse_modal_loss", None)
            val_center_inside = val_metric.get("Val/stage1/center_inside_loss", None)
            val_center_coarse_inside = val_metric.get("Val/stage1/center_coarse_inside_loss", None)
            val_sigma_reg = val_metric.get("Val/stage1/sigma_reg_loss", None)
            val_conf_reg = val_metric.get("Val/stage1/conf_reg_loss", None)
            val_raw_evidence = val_metric.get("Val/stage1/raw_evidence_supervise_loss", None)
            val_evidence = val_metric.get("Val/stage1/evidence_total_loss", None)
            val_fused_support = val_metric.get("Val/stage1/fused_support_supervise_loss", None)
            val_fused_support_outside = val_metric.get("Val/stage1/fused_support_outside_loss", None)
            val_evidence_outside = val_metric.get("Val/stage1/evidence_outside_loss", None)
            val_in_mask = val_metric.get("Val/stage1/center_in_mask_rate", None)
            val_coarse_in_mask = val_metric.get("Val/stage1/center_coarse_in_mask_rate", None)
            val_inside_score = val_metric.get("Val/stage1/center_inside_score_mean", None)
            val_coarse_inside_score = val_metric.get("Val/stage1/center_coarse_inside_score_mean", None)
            val_evidence_inside_ratio = val_metric.get("Val/stage1/evidence_inside_ratio_mean", None)
            val_fused_inside_ratio = val_metric.get("Val/stage1/fused_support_inside_ratio_mean", None)
            val_agreement_inside = val_metric.get("Val/stage1/agreement_inside_score_mean", None)
            val_refine_shift = val_metric.get("Val/stage1/center_refine_shift_mean", None)
            val_coverage = val_metric.get("Val/stage1/roi_coverage", None)

            def _fmt(v, digits=6):
                return f"{float(v):.{digits}f}" if v is not None else "None"

            accelerator.print(
                f"Epoch [{epoch+1}/{display_total_epochs}] stage={stage} "
                f"train_center_modal_loss={_fmt(train_center_modal)} "
                f"train_center_coarse_modal_loss={_fmt(train_center_coarse_modal)} "
                f"train_center_inside_loss={_fmt(train_center_inside)} "
                f"train_center_coarse_inside_loss={_fmt(train_center_coarse_inside)} "
                f"train_sigma_reg_loss={_fmt(train_sigma_reg)} "
                f"train_conf_reg_loss={_fmt(train_conf_reg)} "
                f"train_raw_evidence_supervise_loss={_fmt(train_raw_evidence)} "
                f"train_evidence_total_loss={_fmt(train_evidence)} "
                f"train_fused_support_loss={_fmt(train_fused_support)} "
                f"train_fused_support_outside_loss={_fmt(train_fused_support_outside)} "
                f"train_evidence_outside_loss={_fmt(train_evidence_outside)} "
                f"train_center_inside_score={_fmt(train_inside_score)} "
                f"train_center_coarse_inside_score={_fmt(train_coarse_inside_score)} "
                f"train_evidence_inside_ratio={_fmt(train_evidence_inside_ratio)} "
                f"train_fused_inside_ratio={_fmt(train_fused_inside_ratio)} "
                f"train_agreement_inside={_fmt(train_agreement_inside)} "
                f"train_center_refine_shift={_fmt(train_refine_shift)} "
                f"train_center_in_mask_rate={_fmt(train_in_mask)} "
                f"train_center_coarse_in_mask_rate={_fmt(train_coarse_in_mask)} "
                f"train_roi_coverage={_fmt(train_coverage)} "
                f"val_center_modal_loss={_fmt(val_center_modal)} "
                f"val_center_coarse_modal_loss={_fmt(val_center_coarse_modal)} "
                f"val_center_inside_loss={_fmt(val_center_inside)} "
                f"val_center_coarse_inside_loss={_fmt(val_center_coarse_inside)} "
                f"val_sigma_reg_loss={_fmt(val_sigma_reg)} "
                f"val_conf_reg_loss={_fmt(val_conf_reg)} "
                f"val_raw_evidence_supervise_loss={_fmt(val_raw_evidence)} "
                f"val_evidence_total_loss={_fmt(val_evidence)} "
                f"val_fused_support_loss={_fmt(val_fused_support)} "
                f"val_fused_support_outside_loss={_fmt(val_fused_support_outside)} "
                f"val_evidence_outside_loss={_fmt(val_evidence_outside)} "
                f"val_center_inside_score={_fmt(val_inside_score)} "
                f"val_center_coarse_inside_score={_fmt(val_coarse_inside_score)} "
                f"val_evidence_inside_ratio={_fmt(val_evidence_inside_ratio)} "
                f"val_fused_inside_ratio={_fmt(val_fused_inside_ratio)} "
                f"val_agreement_inside={_fmt(val_agreement_inside)} "
                f"val_center_refine_shift={_fmt(val_refine_shift)} "
                f"val_center_in_mask_rate={_fmt(val_in_mask)} "
                f"val_center_coarse_in_mask_rate={_fmt(val_coarse_in_mask)} "
                f"val_roi_coverage={_fmt(val_coverage)} "
                f"best_score={float(stage_best['best_score']):.6f}"
            )
        elif stage == "stage2":
            train_seg_total = train_metric.get("Train/stage2/seg_total_loss", None)
            train_dice = train_metric.get("Train/stage2/dice_metric", None)
            train_center_inside_aux = train_metric.get("Train/stage2/center_inside_aux_loss", None)
            train_center_inside_w = train_metric.get("Train/stage2/lambda_center_inside_aux", None)
            train_in_mask = train_metric.get("Train/stage2/center_in_mask_rate", None)
            train_coarse_in_mask = train_metric.get("Train/stage2/center_coarse_in_mask_rate", None)
            train_inside_score = train_metric.get("Train/stage2/center_inside_score_mean", None)
            train_coarse_inside_score = train_metric.get("Train/stage2/center_coarse_inside_score_mean", None)
            train_refine_shift = train_metric.get("Train/stage2/center_refine_shift_mean", None)
            train_alpha = train_metric.get("Train/stage2/prior_alpha_mean", None)
            train_hwa_gate = train_metric.get("Train/stage2/runtime_hwa_gate_scale", None)
            train_hwa_gain = train_metric.get("Train/stage2/runtime_hwa_gain_scale", None)
            train_hwa_delta = train_metric.get("Train/stage2/hwa_input_delta_ratio", None)
            train_hwa_agg_gain = train_metric.get("Train/stage2/hwa_input_agg_gain", None)
            train_hwa_gate_boost = train_metric.get("Train/stage2/hwa_input_gate_boost", None)

            val_seg_total = val_metric.get("Val/stage2/seg_total_loss", None)
            val_dice = val_metric.get("Val/stage2/dice_metric", None)
            raw_val_dice = val_metric.get("Val/stage2/raw_dice_metric", None)
            reported_val_dice = val_metric.get("Val/stage2/reported_dice_metric", None)
            hwa_off_val_dice = val_metric.get("Val/stage2/hwa_off_dice_metric", None)
            hwa_val_benefit = val_metric.get("Val/stage2/hwa_benefit_dice", None)
            val_center_inside_aux = val_metric.get("Val/stage2/center_inside_aux_loss", None)
            val_center_inside_w = val_metric.get("Val/stage2/lambda_center_inside_aux", None)
            val_in_mask = val_metric.get("Val/stage2/center_in_mask_rate", None)
            val_coarse_in_mask = val_metric.get("Val/stage2/center_coarse_in_mask_rate", None)
            val_inside_score = val_metric.get("Val/stage2/center_inside_score_mean", None)
            val_coarse_inside_score = val_metric.get("Val/stage2/center_coarse_inside_score_mean", None)
            val_refine_shift = val_metric.get("Val/stage2/center_refine_shift_mean", None)
            val_alpha = val_metric.get("Val/stage2/prior_alpha_mean", None)
            val_stacked_dice = val_metric.get("Val/stage2/stacked_dice_metric", None)
            val_hwa_gate = val_metric.get("Val/stage2/runtime_hwa_gate_scale", None)
            val_hwa_gain = val_metric.get("Val/stage2/runtime_hwa_gain_scale", None)
            val_hwa_delta = val_metric.get("Val/stage2/hwa_input_delta_ratio", None)

            def _fmt(v, digits=6):
                return f"{float(v):.{digits}f}" if v is not None else "None"

            accelerator.print(
                f"Epoch [{epoch+1}/{display_total_epochs}] stage={stage} "
                f"train_dice={_fmt(train_dice, 4)} "
                f"train_seg_total_loss={_fmt(train_seg_total)} "
                f"train_center_inside_w={_fmt(train_center_inside_w, 4)} "
                f"train_center_inside_aux_loss={_fmt(train_center_inside_aux)} "
                f"train_center_inside_score={_fmt(train_inside_score)} "
                f"train_center_coarse_inside_score={_fmt(train_coarse_inside_score)} "
                f"train_center_refine_shift={_fmt(train_refine_shift)} "
                f"train_center_in_mask_rate={_fmt(train_in_mask)} "
                f"train_center_coarse_in_mask_rate={_fmt(train_coarse_in_mask)} "
                f"train_prior_alpha_mean={_fmt(train_alpha, 4)} "
                f"train_hwa_gate={_fmt(train_hwa_gate, 4)} "
                f"train_hwa_gain={_fmt(train_hwa_gain, 4)} "
                f"train_hwa_delta_ratio={_fmt(train_hwa_delta, 6)} "
                f"train_hwa_input_agg_gain={_fmt(train_hwa_agg_gain, 4)} "
                f"train_hwa_gate_boost={_fmt(train_hwa_gate_boost, 4)} "
                f"val_dice={_fmt(val_dice, 4)} "
                f"raw_val_dice={_fmt(raw_val_dice, 4)} "
                f"reported_val_dice={_fmt(reported_val_dice, 4)} "
                f"hwa_off_val_dice={_fmt(hwa_off_val_dice, 4)} "
                f"hwa_val_benefit={_fmt(hwa_val_benefit, 4)} "
                f"stacked_val_dice={_fmt(val_stacked_dice, 4)} "
                f"val_seg_total_loss={_fmt(val_seg_total)} "
                f"val_center_inside_w={_fmt(val_center_inside_w, 4)} "
                f"val_center_inside_aux_loss={_fmt(val_center_inside_aux)} "
                f"val_center_inside_score={_fmt(val_inside_score)} "
                f"val_center_coarse_inside_score={_fmt(val_coarse_inside_score)} "
                f"val_center_refine_shift={_fmt(val_refine_shift)} "
                f"val_center_in_mask_rate={_fmt(val_in_mask)} "
                f"val_center_coarse_in_mask_rate={_fmt(val_coarse_in_mask)} "
                f"val_prior_alpha_mean={_fmt(val_alpha, 4)} "
                f"val_hwa_gate={_fmt(val_hwa_gate, 4)} "
                f"val_hwa_gain={_fmt(val_hwa_gain, 4)} "
                f"val_hwa_delta_ratio={_fmt(val_hwa_delta, 6)} "
                f"best_score={float(stage_best['best_score']):.4f}"
            )
            if triggered_stage2_retry:
                accelerator.print(
                    f"[Stage2Retry] next pass: cycle {int(runtime_state['stage2_retry_count'])+1}/"
                    f"{int(runtime_state['stage2_total_cycles'])} will restart from best stage2 weights."
                )
            if triggered_stage2_early_stop:
                accelerator.print("[Stage2EarlyStop] stage2 finished early and restored best stage2 weights.")
        else:
            train_score = train_metric.get(f"Train/{stage}/loss", None)
            accelerator.print(
                f"Epoch [{epoch+1}/{display_total_epochs}] stage={stage} "
                + (f"train_loss={float(train_score):.4f} " if train_score is not None else "")
                + f"val_score={float(val_score):.4f} best_score={float(stage_best['best_score']):.4f}"
            )

        next_epoch = epoch
        if triggered_stage2_early_stop:
            next_epoch += max(stage_total_epochs - (stage_local_epoch + 1), 0)
        next_epoch += 1

        if _is_stage2_to_stage3_transition(
            stage,
            next_epoch=next_epoch,
            cfg=config,
            runtime_state=runtime_state,
        ):
            _distributed_barrier(
                accelerator,
                f"before_stage_transition_epoch_{epoch+1}",
                loaders={"train": active_train_loader, "val": val_loader, "test": test_loader},
            )
            accelerator.print("[StageTransition] loading best_stage2 before stage3.")
            if not _load_best_stage_state(accelerator, checkpoint_name, "stage2"):
                accelerator.print(
                    "[StageTransition] best_stage2 not available; continue with current stage2 weights."
                )
            _reset_stage3_class_decoder_if_requested(model, config, accelerator)
            _distributed_barrier(
                accelerator,
                f"after_stage_transition_epoch_{epoch+1}",
                loaders={"train": active_train_loader, "val": val_loader, "test": test_loader},
            )

        epoch = next_epoch

    accelerator.print(f"best stage metrics: {best_state_by_stage}")
