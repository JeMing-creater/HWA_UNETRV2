#!/usr/bin/env bash
set -euo pipefail

export HWA_STAGE2_EVAL_CONFIG="${HWA_STAGE2_EVAL_CONFIG:-configs/stage2_segmentation.yaml}"
export HWA_STAGE2_EVAL_RUN="${HWA_STAGE2_EVAL_RUN:-HWA_stage2_segmentation}"
export HWA_STAGE2_EVAL_CKPT="${HWA_STAGE2_EVAL_CKPT:-model_store/${HWA_STAGE2_EVAL_RUN}/best_stage2/pytorch_model.bin}"
export HWA_STAGE2_EVAL_OUT="${HWA_STAGE2_EVAL_OUT:-evaluation_records/${HWA_STAGE2_EVAL_RUN}_stage2_eval}"
export HWA_STAGE2_EVAL_SPLITS="${HWA_STAGE2_EVAL_SPLITS:-Val,Test}"
export HWA_STAGE2_EVAL_HD95="${HWA_STAGE2_EVAL_HD95:-false}"

python eval_stage2_best_segmentation.py
