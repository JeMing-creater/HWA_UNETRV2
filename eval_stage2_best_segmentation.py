import json
import os
import re
from datetime import datetime
from pathlib import Path

import monai
import torch
from accelerate import Accelerator

from GCM_train_core import (
    _apply_stage2_prior_alpha_cap,
    _apply_stage2_hwa_gain_init_if_needed,
    _build_model,
    _build_post_trans,
    _cfg_get,
    _load_yaml_config,
    apply_stage_policy,
    build_seg_metrics,
    remap_method_aligned_state_dict_keys,
    val_one_epoch,
)
from src import utils
from src.loader import MultiModalityDataset, get_dataloader_GCM as get_dataloader


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in [None, ""]:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_splits(name: str, default):
    raw = os.environ.get(name)
    if raw in [None, ""]:
        return list(default)
    valid = {"train", "val", "test"}
    splits = []
    for item in str(raw).replace(";", ",").split(","):
        split = item.strip()
        if not split:
            continue
        normalized = split.lower()
        if normalized not in valid:
            raise ValueError(f"Unsupported split in {name}: {split}")
        splits.append(normalized.title())
    return splits or list(default)


def _to_float_dict(metric):
    out = {}
    for key, value in metric.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = value
    return out


def _env_int_optional(name: str):
    raw = os.environ.get(name)
    if raw in [None, "", "None", "none"]:
        return None
    return int(raw)


def _env_str(name: str):
    raw = os.environ.get(name)
    if raw in [None, "", "None", "none"]:
        return None
    return str(raw)


def _allow_eval_split_override(prefix: str) -> bool:
    return _env_bool(f"{prefix}_ALLOW_SPLIT_OVERRIDE", False)


def _maybe_apply_eval_hwa_gain_init(model, cfg) -> bool:
    if not _env_bool("HWA_STAGE2_EVAL_APPLY_HWA_GAIN_INIT", False):
        return False
    return bool(_apply_stage2_hwa_gain_init_if_needed(model, cfg))


def _apply_eval_split_path_overrides(cfg, prefix: str = "HWA_STAGE2_EVAL") -> None:
    current = getattr(cfg, "GCM_loader", None)
    if current is None:
        return
    if not _allow_eval_split_override(prefix):
        return
    mappings = {
        "train_examples_path": f"{prefix}_TRAIN_EXAMPLES_PATH",
        "val_examples_path": f"{prefix}_VAL_EXAMPLES_PATH",
        "test_examples_path": f"{prefix}_TEST_EXAMPLES_PATH",
    }
    for attr_name, env_name in mappings.items():
        value = _env_str(env_name)
        if value is not None:
            setattr(current, attr_name, value)


def _eval_pred_threshold(cfg) -> float:
    raw = os.environ.get("HWA_STAGE2_EVAL_THRESHOLD")
    if raw not in [None, "", "None", "none"]:
        return float(raw)
    return float(_cfg_get(cfg, "stage_train.stage2.pred_threshold", 0.5))


def _infer_best_stage2_epoch_from_log(log_path: Path):
    if not log_path or not Path(log_path).is_file():
        return None
    best = None
    pattern = re.compile(
        r"Epoch \[(?P<epoch>\d+)/(?:\d+)\] stage=stage2\b.*?"
        r"raw_val_dice=(?P<raw_val>[0-9.]+|None)"
    )
    for line in Path(log_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        raw_val = match.group("raw_val")
        if raw_val in ["None", None]:
            continue
        epoch = int(match.group("epoch"))
        item = {
            "epoch": epoch,
            "stage_local_epoch": max(0, epoch - 1),
            "raw_val_dice": float(raw_val),
        }
        if best is None or item["raw_val_dice"] > best["raw_val_dice"]:
            best = item
    return best


def _find_latest_run_log(run_name: str):
    candidates = sorted(
        Path("logs").glob(f"{run_name}*/log.txt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _stable_train_eval_loader(train_loader, val_loader, cfg):
    train_dataset = getattr(train_loader, "dataset", None)
    val_dataset = getattr(val_loader, "dataset", None)
    if not isinstance(train_dataset, MultiModalityDataset):
        return train_loader
    if not isinstance(val_dataset, MultiModalityDataset):
        return train_loader

    stable_dataset = MultiModalityDataset(
        data=train_dataset.data,
        loadforms=train_dataset.loadforms,
        transforms=val_dataset.transforms,
        over_label=train_dataset.over_label,
        over_add=train_dataset.over_add,
        use_class=train_dataset.use_class,
    )
    return monai.data.DataLoader(
        stable_dataset,
        num_workers=int(cfg.GCM_loader.num_workers),
        batch_size=int(cfg.trainer.batch_size),
        shuffle=False,
    )


def main():
    config_path = os.environ.get(
        "HWA_STAGE2_EVAL_CONFIG", "configs/stage2_segmentation.yaml"
    )
    run_name = os.environ.get(
        "HWA_STAGE2_EVAL_RUN", "HWA_stage2_segmentation"
    )
    checkpoint_path = Path(
        os.environ.get(
            "HWA_STAGE2_EVAL_CKPT",
            f"model_store/{run_name}/best_stage2/pytorch_model.bin",
        )
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(
        os.environ.get(
            "HWA_STAGE2_EVAL_OUT",
            f"evaluation_records/{run_name}_best_stage2_seg_{timestamp}",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = _load_yaml_config(config_path)
    cfg.trainer.model_variant = "soft_prior"
    cfg.stage_train.visualization.enable = False
    cfg.stage_train.stage2.use_hwa_prior_in_encoder = True
    cfg.stage_train.stage2.disable_prior_training = False
    cfg.stage_train.stage2.use_detector_debug = True
    cfg.GCM_loader.fix_example = _env_bool(
        "HWA_STAGE2_EVAL_FIX_EXAMPLE",
        bool(_cfg_get(cfg, "GCM_loader.fix_example", True)),
    )
    _apply_eval_split_path_overrides(cfg, "HWA_STAGE2_EVAL")

    utils.same_seeds(50)
    accelerator = Accelerator(cpu=False)
    accelerator.log = lambda *args, **kwargs: None

    accelerator.print(f"[Stage2Eval] config={config_path}")
    accelerator.print(f"[Stage2Eval] checkpoint={checkpoint_path}")
    accelerator.print(f"[Stage2Eval] output={out_dir}")
    enable_hd95 = _env_bool("HWA_STAGE2_EVAL_HD95", True)
    accelerator.print(f"[Stage2Eval] enable_hd95={enable_hd95}")

    model = _build_model(cfg)
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(checkpoint_path))
    else:
        state = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    target_keys = set(model.state_dict().keys())
    for prefix in ("module.", "net."):
        if not any(str(key).startswith(prefix) for key in state.keys()):
            continue
        stripped_state = {
            str(key)[len(prefix):] if str(key).startswith(prefix) else str(key): value
            for key, value in state.items()
        }
        direct_matches = sum(1 for key in state if key in target_keys)
        stripped_matches = sum(1 for key in stripped_state if key in target_keys)
        if stripped_matches > direct_matches:
            state = stripped_state
    state = remap_method_aligned_state_dict_keys(state)
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    reapplied_hwa_init = _maybe_apply_eval_hwa_gain_init(model, cfg)

    raw_train_loader, val_loader, test_loader, _ = get_dataloader(cfg)
    raw_train_loader = _stable_train_eval_loader(raw_train_loader, val_loader, cfg)
    split_lengths = {
        "Train": len(raw_train_loader.dataset),
        "Val": len(val_loader.dataset),
        "Test": len(test_loader.dataset),
    }
    accelerator.print(f"[Stage2Eval] split_lengths={split_lengths}")

    model, raw_train_loader, val_loader, test_loader = accelerator.prepare(
        model, raw_train_loader, val_loader, test_loader
    )

    stage1_epochs = int(_cfg_get(cfg, "stage_train.stage1.epochs", 60))
    stage2_epochs = int(_cfg_get(cfg, "stage_train.stage2.epochs", 300))
    stage_local_epoch = _env_int_optional("HWA_STAGE2_EVAL_STAGE_LOCAL_EPOCH")
    best_epoch_info = None
    if stage_local_epoch is None:
        explicit_log = os.environ.get("HWA_STAGE2_EVAL_LOG", "")
        log_path = Path(explicit_log) if explicit_log else _find_latest_run_log(run_name)
        best_epoch_info = _infer_best_stage2_epoch_from_log(log_path) if log_path else None
        if best_epoch_info is not None:
            stage_local_epoch = int(best_epoch_info["stage_local_epoch"])
    if stage_local_epoch is None:
        stage_local_epoch = max(0, stage2_epochs - 1)
    epoch = stage1_epochs + stage_local_epoch
    display_total_epochs = int(
        _cfg_get(cfg, "trainer.num_epochs", stage1_epochs + stage2_epochs)
    )
    accelerator.print(f"[Stage2Eval] stage_local_epoch={stage_local_epoch}")
    if best_epoch_info is not None:
        accelerator.print(f"[Stage2Eval] inferred_best_epoch={best_epoch_info}")

    apply_stage_policy(
        model,
        "stage2",
        cfg=cfg,
        stage_local_epoch=stage_local_epoch,
        stage_total_epochs=stage2_epochs,
    )
    prior_alpha_cap = _apply_stage2_prior_alpha_cap(
        model,
        epoch,
        cfg,
        stage_local_epoch=stage_local_epoch,
        stage_total_epochs=stage2_epochs,
    )
    accelerator.print(f"[Stage2Eval] prior_alpha_cap={prior_alpha_cap:.6f}")

    seg_loss_functions = {
        "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
        "dice_loss": monai.losses.DiceLoss(
            smooth_nr=0,
            smooth_dr=1e-5,
            to_onehot_y=False,
            sigmoid=True,
        ),
    }
    pred_threshold = _eval_pred_threshold(cfg)
    post_trans = _build_post_trans(pred_threshold)

    results = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_mtime": datetime.fromtimestamp(
            checkpoint_path.stat().st_mtime
        ).isoformat(timespec="seconds"),
        "output_dir": str(out_dir),
        "split_lengths": split_lengths,
        "stage": "stage2",
        "stage_local_epoch": stage_local_epoch,
        "stage_total_epochs": stage2_epochs,
        "epoch": epoch,
        "pred_threshold": float(pred_threshold),
        "inferred_best_epoch": best_epoch_info,
        "prior_alpha_cap": float(prior_alpha_cap),
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
        "reapplied_hwa_gain_init": bool(reapplied_hwa_init),
    }

    loaders_by_split = {
        "Train": raw_train_loader,
        "Val": val_loader,
        "Test": test_loader,
    }
    eval_splits = _env_splits("HWA_STAGE2_EVAL_SPLITS", ("Train", "Val", "Test"))
    results["eval_splits"] = eval_splits
    accelerator.print(f"[Stage2Eval] eval_splits={eval_splits}")

    for split in eval_splits:
        loader = loaders_by_split[split]
        metrics = build_seg_metrics(enable_hd95=enable_hd95)
        score, metric, _, _ = val_one_epoch(
            model,
            loader,
            accelerator,
            epoch,
            0,
            "stage2",
            cfg,
            seg_loss_functions,
            {},
            seg_metrics=metrics,
            cls_metrics=None,
            post_trans=post_trans,
            post_trans_cls=None,
            split=split,
            stage_local_epoch=stage_local_epoch,
            stage_total_epochs=stage2_epochs,
            display_total_epochs=display_total_epochs,
        )
        metric = _to_float_dict(metric)
        results[split] = {
            "score": float(score),
            "metrics": metric,
            "dice": metric.get(f"{split}/stage2/dice_metric"),
            "hd95": metric.get(f"{split}/stage2/hd95_metric"),
            "loss": metric.get(f"{split}/stage2/loss"),
            "seg_total_loss": metric.get(f"{split}/stage2/seg_total_loss"),
            "prior_alpha_mean": metric.get(f"{split}/stage2/prior_alpha_mean"),
            "center_in_mask_rate": metric.get(f"{split}/stage2/center_in_mask_rate"),
            "center_inside_score_mean": metric.get(
                f"{split}/stage2/center_inside_score_mean"
            ),
        }
        accelerator.print(
            "[Stage2Eval] "
            f"{split}: dice={results[split]['dice']} "
            f"hd95={results[split]['hd95']} "
            f"loss={results[split]['loss']} "
            f"prior_alpha_mean={results[split]['prior_alpha_mean']}"
        )

    if accelerator.is_main_process:
        metrics_path = out_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        accelerator.print(f"[Stage2Eval] saved={metrics_path}")


if __name__ == "__main__":
    main()
