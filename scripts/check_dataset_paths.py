import argparse
import sys
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Check dataset and split paths in a YAML config.")
    parser.add_argument(
        "--config",
        default="configs/stage2_segmentation.yaml",
        help="Path to the training config to inspect.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when any configured path is missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    loader_cfg = cfg.get("GCM_loader", {})
    paths = {
        "dataset root": Path(loader_cfg.get("root", "")),
        "train split": Path(loader_cfg.get("train_examples_path", "")),
        "validation split": Path(loader_cfg.get("val_examples_path", "")),
        "test split": Path(loader_cfg.get("test_examples_path", "")),
    }

    missing = []
    for name, path in paths.items():
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        sys.stdout.write(f"{name}: {path} [{status}]\n")
        if not exists:
            missing.append(name)

    if args.strict and missing:
        sys.stderr.write(f"Missing configured paths: {', '.join(missing)}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
