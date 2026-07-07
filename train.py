import argparse
from pathlib import Path

from GCM_train_core import run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Train HWA-UNETRv2 by stage.")
    parser.add_argument(
        "--stage",
        choices=("stage1", "stage2", "stage3"),
        required=True,
        help="Training stage to run.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config. Defaults to configs/<stage>.yaml.",
    )
    return parser.parse_args()


def default_config(stage: str) -> Path:
    mapping = {
        "stage1": "configs/stage1_detector.yaml",
        "stage2": "configs/stage2_segmentation.yaml",
        "stage3": "configs/stage3_classification.yaml",
    }
    return Path(mapping[stage])


def main():
    args = parse_args()
    config_path = Path(args.config) if args.config else default_config(args.stage)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    forced_stage = args.stage
    if args.stage == "stage2":
        forced_stage = None

    run_training(
        config_path=str(config_path),
        forced_stage=forced_stage,
        runner_name=f"HWA_UNETRv2_{args.stage}",
    )


if __name__ == "__main__":
    main()
