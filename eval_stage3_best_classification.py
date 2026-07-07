import json
import os
from datetime import datetime
from pathlib import Path

import monai
import torch
from accelerate import Accelerator
from easydict import EasyDict
import yaml

from GCM_train_core import (
    _build_model,
    _build_post_trans,
    _cfg_get,
    apply_stage_policy,
    build_cls_metrics,
    build_stage3_classification_losses,
    remap_method_aligned_state_dict_keys,
    val_one_epoch,
)
from src import utils
from src.loader import MultiModalityDataset, get_dataloader_GCM as get_dataloader


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in [None, ""]:
        return int(default)
    return int(raw)


def _env_str(name: str):
    raw = os.environ.get(name)
    if raw in [None, "", "None", "none"]:
        return None
    return str(raw)


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


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in [None, ""]:
        return float(default)
    return float(raw)


def _apply_eval_split_path_overrides(cfg, prefix: str = "HWA_STAGE3_EVAL") -> None:
    current = getattr(cfg, "GCM_loader", None)
    if current is None:
        return
    if not _env_bool(f"{prefix}_ALLOW_SPLIT_OVERRIDE", False):
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


def _to_float_dict(metric):
    out = {}
    for key, value in metric.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = value
    return out


def _load_yaml_config(config_path: str):
    return EasyDict(
        yaml.load(open(config_path, "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )


def _resolve_stage3_checkpoint(root: Path, run_name: str, checkpoint_path: str | None) -> Path:
    if checkpoint_path not in [None, ""]:
        path = Path(str(checkpoint_path))
        return path if path.is_absolute() else root / path
    base = root / "model_store" / run_name
    for subdir in ("best_stage3", "best", "checkpoint"):
        path = base / subdir / "pytorch_model.bin"
        if path.is_file():
            return path
    return base / "best_stage3" / "pytorch_model.bin"


def _strip_module_prefix_if_needed(state: dict) -> dict:
    if not any(str(key).startswith("module.") for key in state.keys()):
        return state
    return {
        str(key)[7:] if str(key).startswith("module.") else str(key): value
        for key, value in state.items()
    }


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


def main() -> None:
    root = Path(__file__).resolve().parent
    config_path = os.environ.get("HWA_STAGE3_EVAL_CONFIG", "configs/stage3_classification.yaml")
    run_name = os.environ.get("HWA_STAGE3_EVAL_RUN", os.environ.get("HWA_STAGE3_RUN_NAME", "HWA_stage3_classification"))
    checkpoint = _resolve_stage3_checkpoint(
        root,
        run_name,
        os.environ.get("HWA_STAGE3_EVAL_CKPT"),
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(
        os.environ.get(
            "HWA_STAGE3_EVAL_OUT",
            str(root / "evaluation_records" / f"{run_name}_best_stage3_cls_{timestamp}"),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cfg = _load_yaml_config(str(root / config_path if not Path(config_path).is_absolute() else config_path))
    cfg.trainer.model_variant = "soft_prior"
    cfg.trainer.task = "classification"
    cfg.trainer.batch_size = _env_int(
        "HWA_STAGE3_EVAL_BATCH_SIZE",
        int(_cfg_get(cfg, "trainer.batch_size", 1) or 1),
    )
    if hasattr(cfg.stage_train, "visualization"):
        cfg.stage_train.visualization.enable = False
    cfg.stage_train.stage2.use_hwa_prior_in_encoder = True
    cfg.stage_train.stage3.class_threshold = _env_float(
        "HWA_STAGE3_EVAL_CLASS_THRESHOLD",
        float(_cfg_get(cfg, "stage_train.stage3.class_threshold", 0.5)),
    )
    cfg.GCM_loader.task_Mu = 1
    _apply_eval_split_path_overrides(cfg, "HWA_STAGE3_EVAL")

    utils.same_seeds(_env_int("HWA_STAGE3_EVAL_SEED", 50))
    accelerator = Accelerator(cpu=False)
    accelerator.log = lambda *args, **kwargs: None
    accelerator.print(f"[Stage3Eval] config={config_path}")
    accelerator.print(f"[Stage3Eval] checkpoint={checkpoint}")
    accelerator.print(f"[Stage3Eval] output={out_dir}")

    model = _build_model(cfg)
    state = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = _strip_module_prefix_if_needed(state)
    state = remap_method_aligned_state_dict_keys(state)
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)

    train_loader, val_loader, test_loader, _ = get_dataloader(cfg)
    train_loader = _stable_train_eval_loader(train_loader, val_loader, cfg)
    split_lengths = {
        "Train": len(train_loader.dataset),
        "Val": len(val_loader.dataset),
        "Test": len(test_loader.dataset),
    }
    accelerator.print(f"[Stage3Eval] split_lengths={split_lengths}")

    model, train_loader, val_loader, test_loader = accelerator.prepare(
        model, train_loader, val_loader, test_loader
    )

    stage3_epochs = int(_cfg_get(cfg, "stage_train.stage3.epochs", _cfg_get(cfg, "trainer.num_epochs", 1)))
    stage_local_epoch = max(0, stage3_epochs - 1)
    epoch = stage_local_epoch
    display_total_epochs = int(_cfg_get(cfg, "trainer.num_epochs", stage3_epochs))
    apply_stage_policy(
        model,
        "stage3",
        cfg=cfg,
        stage_local_epoch=stage_local_epoch,
        stage_total_epochs=stage3_epochs,
    )

    loss_functions_cls = build_stage3_classification_losses(
        cfg=cfg,
        train_loader=train_loader,
        device=accelerator.device,
        accelerator=accelerator,
    )
    post_trans_cls = _build_post_trans(
        float(_cfg_get(cfg, "stage_train.stage3.class_threshold", 0.5))
    )
    task_mu = int(_cfg_get(cfg, "GCM_loader.task_Mu", 1) or 1)

    results = {
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_mtime": datetime.fromtimestamp(
            checkpoint.stat().st_mtime
        ).isoformat(timespec="seconds"),
        "output_dir": str(out_dir),
        "split_lengths": split_lengths,
        "stage": "stage3",
        "stage_local_epoch": stage_local_epoch,
        "stage_total_epochs": stage3_epochs,
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
    }

    loaders_by_split = {
        "Train": train_loader,
        "Val": val_loader,
        "Test": test_loader,
    }
    eval_splits = _env_splits("HWA_STAGE3_EVAL_SPLITS", ("Train", "Val", "Test"))
    results["eval_splits"] = eval_splits
    accelerator.print(f"[Stage3Eval] eval_splits={eval_splits}")

    for split in eval_splits:
        loader = loaders_by_split[split]
        cls_metrics = build_cls_metrics(task_mu=task_mu)
        score, metric, _, _ = val_one_epoch(
            model,
            loader,
            accelerator,
            epoch,
            0,
            "stage3",
            cfg,
            {},
            loss_functions_cls,
            seg_metrics=None,
            cls_metrics=cls_metrics,
            post_trans=None,
            post_trans_cls=post_trans_cls,
            split=split,
            stage_local_epoch=stage_local_epoch,
            stage_total_epochs=stage3_epochs,
            display_total_epochs=display_total_epochs,
        )
        metric = _to_float_dict(metric)
        accuracy = metric.get(f"{split}/stage3/Task0_accuracy")
        results[split] = {
            "score": float(score),
            "accuracy": accuracy,
            "f1": metric.get(f"{split}/stage3/Task0_f1"),
            "specificity": metric.get(f"{split}/stage3/Task0_specificity"),
            "recall": metric.get(f"{split}/stage3/Task0_recall"),
            "loss": metric.get(f"{split}/stage3/loss"),
            "metrics": metric,
        }
        accelerator.print(
            "[Stage3Eval] "
            f"{split}: accuracy={results[split]['accuracy']} "
            f"f1={results[split]['f1']} "
            f"specificity={results[split]['specificity']} "
            f"recall={results[split]['recall']} "
            f"loss={results[split]['loss']}"
        )

    if accelerator.is_main_process:
        metrics_path = out_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        accelerator.print(f"[Stage3Eval] saved={metrics_path}")


if __name__ == "__main__":
    main()
