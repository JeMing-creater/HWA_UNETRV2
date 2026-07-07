from GCM_train_core import run_training


if __name__ == "__main__":
    run_training(
        config_path="configs/stage3_classification.yaml",
        forced_stage="stage3",
        runner_name="GCM_train_stage3",
    )
