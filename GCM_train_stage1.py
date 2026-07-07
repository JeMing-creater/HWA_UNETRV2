from GCM_train_core import run_training


if __name__ == "__main__":
    run_training(
        config_path="configs/stage1_detector.yaml",
        forced_stage="stage1",
        runner_name="GCM_train_stage1",
    )
