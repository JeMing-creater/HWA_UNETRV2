import os
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator


def remap_method_aligned_state_dict_keys(state_dict):
    """Map historical checkpoint module names to the release names."""
    if not isinstance(state_dict, dict):
        return state_dict

    aliases = (
        ("fussion.", "hwa_block."),
        ("SegHead.", "todm_seg_head."),
        ("Class_Decoder.", "todm_cls_head."),
    )
    remapped = {}
    for key, value in state_dict.items():
        new_key = str(key)
        for wrapper_prefix in ("module.", "net."):
            if new_key.startswith(wrapper_prefix):
                new_key = new_key[len(wrapper_prefix):]
                break
        for old_prefix, new_prefix in aliases:
            if new_key.startswith(old_prefix):
                new_key = new_prefix + new_key[len(old_prefix):]
                break
        remapped[new_key] = value
    return remapped


class Logger:
    def __init__(self, logdir: str):
        self.console = sys.stdout
        self._closed = False
        if logdir is not None:
            os.makedirs(logdir, exist_ok=True)
            self.log_file = open(os.path.join(logdir, "log.txt"), "w", encoding="utf-8")
        else:
            self.log_file = None
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.log_file is not None:
            self.log_file.write(msg)

    def flush(self):
        self.console.flush()
        if self.log_file is not None:
            self.log_file.flush()
            os.fsync(self.log_file.fileno())

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.log_file is not None and not self.log_file.closed:
            try:
                self.log_file.flush()
            except Exception:
                pass
            self.log_file.close()


def load_model_dict(download_path, save_path=None, check_hash=True) -> OrderedDict:
    if str(download_path).startswith("http"):
        return torch.hub.load_state_dict_from_url(
            download_path,
            model_dir=save_path,
            check_hash=check_hash,
            map_location=torch.device("cpu"),
        )
    return torch.load(download_path, map_location=torch.device("cpu"))


def _ensure_accelerate_model_file(base_path: str, accelerator: Accelerator) -> None:
    bin_path = os.path.join(base_path, "pytorch_model.bin")
    if os.path.isfile(bin_path):
        return

    safetensors_path = os.path.join(base_path, "model.safetensors")
    if not os.path.isfile(safetensors_path):
        return

    try:
        from safetensors.torch import load_file

        torch.save(load_file(safetensors_path), bin_path)
    except Exception as exc:
        accelerator.print(
            f"Failed to materialize accelerate-compatible checkpoint from {safetensors_path}: {exc}"
        )
    finally:
        if hasattr(accelerator, "wait_for_everyone"):
            accelerator.wait_for_everyone()


def resume_train_state(
    model,
    path: str,
    optimizer,
    scheduler,
    train_loader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
    seg: bool = True,
):
    base_path = os.path.join(os.getcwd(), "model_store", path, "checkpoint")
    try:
        epoch_checkpoint = torch.load(
            os.path.join(base_path, "epoch.pth.tar"),
            map_location="cpu",
        )
        starting_epoch = epoch_checkpoint["epoch"] + 1
        step = starting_epoch * len(train_loader)
        if accelerator.is_main_process:
            _ensure_accelerate_model_file(base_path, accelerator)
        if hasattr(accelerator, "wait_for_everyone"):
            accelerator.wait_for_everyone()
        accelerator.load_state(base_path)

        if not seg:
            best_accuracy = epoch_checkpoint["best_accuracy"]
            best_test_accuracy = epoch_checkpoint["best_test_accuracy"]
            best_metrics = epoch_checkpoint["best_metrics"]
            best_test_metrics = epoch_checkpoint["best_test_metrics"]
            accelerator.print(
                f"Loading training state successfully. Start training from {starting_epoch}, Best Acc: {best_accuracy}"
            )
            return (
                model,
                optimizer,
                scheduler,
                starting_epoch,
                step,
                best_accuracy,
                best_test_accuracy,
                best_metrics,
                best_test_metrics,
            )

        best_score = epoch_checkpoint["best_score"]
        best_test_score = epoch_checkpoint["best_test_score"]
        best_metrics = epoch_checkpoint["best_metrics"]
        best_test_metrics = epoch_checkpoint["best_test_metrics"]
        best_hd95 = epoch_checkpoint["best_hd95"]
        best_test_hd95 = epoch_checkpoint["best_test_hd95"]
        best_hd95_metrics = epoch_checkpoint["best_hd95_metrics"]
        best_test_hd95_metrics = epoch_checkpoint["best_test_hd95_metrics"]
        accelerator.print(
            f"Loading training state successfully. Start training from {starting_epoch}, Best Acc: {best_score}"
        )
        return (
            model,
            optimizer,
            scheduler,
            starting_epoch,
            step,
            best_score,
            best_test_score,
            best_metrics,
            best_test_metrics,
            best_hd95,
            best_test_hd95,
            best_hd95_metrics,
            best_test_hd95_metrics,
        )
    except Exception as exc:
        accelerator.print(exc)
        accelerator.print("Failed to load training state.")
        if not seg:
            return (
                model,
                optimizer,
                scheduler,
                0,
                0,
                torch.tensor(0),
                torch.tensor(0),
                {},
                {},
            )
        return (
            model,
            optimizer,
            scheduler,
            0,
            0,
            torch.tensor(0),
            torch.tensor(0),
            [],
            [],
            torch.tensor(1000),
            torch.tensor(1000),
            [],
            [],
        )


def same_seeds(seed):
    torch.multiprocessing.set_sharing_strategy("file_system")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True


def write_example(config, example):
    def _cfg_value(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    data_root = config.GCM_loader.root
    paths = (
        _cfg_value(config.GCM_loader, "train_examples_path", os.path.join(data_root, "train_examples.txt")),
        _cfg_value(config.GCM_loader, "val_examples_path", os.path.join(data_root, "val_examples.txt")),
        _cfg_value(config.GCM_loader, "test_examples_path", os.path.join(data_root, "test_examples.txt")),
    )

    for split_path, split_values in zip(paths, example[:3]):
        parent = os.path.dirname(split_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(split_path, "w", encoding="utf-8") as file:
            for item in split_values:
                file.write(str(item) + "\n")

    if len(example) == 6:
        lack_paths = (
            os.path.join(data_root, "train_lack_example.txt"),
            os.path.join(data_root, "val_lack_example.txt"),
            os.path.join(data_root, "test_lack_example.txt"),
        )
        for split_path, split_values in zip(lack_paths, example[3:]):
            with open(split_path, "w", encoding="utf-8") as file:
                for item in split_values:
                    file.write(str(item) + "\n")


def reload_pre_train_model(
    model, accelerator, checkpoint_path="HWA_stage3_classification"
):
    base_path = Path(os.getcwd()) / "model_store" / str(checkpoint_path)
    check_dirs = [
        base_path / "best",
        base_path / "best_stage3",
        base_path / "best_stage2",
        base_path / "best_stage1",
        base_path / "checkpoint",
    ]
    try:
        selected_dir = None
        selected_path = None
        for check_dir in check_dirs:
            bin_path = check_dir / "pytorch_model.bin"
            safetensors_path = check_dir / "model.safetensors"
            if bin_path.is_file():
                selected_dir = check_dir
                selected_path = bin_path
                break
            if safetensors_path.is_file():
                selected_dir = check_dir
                selected_path = safetensors_path
                break

        if selected_path is None:
            raise FileNotFoundError(f"no model file under {base_path}")

        accelerator.print("load pretrain model from %s" % (str(selected_dir) + "/"))
        if selected_path.name == "pytorch_model.bin":
            checkpoint = load_model_dict(str(selected_path))
        else:
            from safetensors.torch import load_file

            checkpoint = load_file(str(selected_path))
        checkpoint = remap_method_aligned_state_dict_keys(checkpoint)
        model.load_state_dict(checkpoint, strict=False)
        accelerator.print("Load checkpoint model successfully.")
    except Exception as exc:
        accelerator.print(f"Load checkpoint model failed: {repr(exc)}")
    return model
