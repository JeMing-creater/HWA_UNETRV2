from GCM_train_core import run_training


if __name__ == "__main__":
    run_training(
        config_path="configs/stage2_segmentation.yaml",
        forced_stage=None,
        runner_name="GCM_train_stage2",
    )
