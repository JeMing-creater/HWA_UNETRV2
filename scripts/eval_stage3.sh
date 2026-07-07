#!/usr/bin/env bash
set -euo pipefail

export HWA_STAGE3_EVAL_CONFIG="${HWA_STAGE3_EVAL_CONFIG:-configs/stage3_classification.yaml}"
export HWA_STAGE3_EVAL_RUN="${HWA_STAGE3_EVAL_RUN:-HWA_stage3_classification}"
export HWA_STAGE3_EVAL_CKPT="${HWA_STAGE3_EVAL_CKPT:-model_store/${HWA_STAGE3_EVAL_RUN}/best_stage3/pytorch_model.bin}"
export HWA_STAGE3_EVAL_OUT="${HWA_STAGE3_EVAL_OUT:-evaluation_records/${HWA_STAGE3_EVAL_RUN}_stage3_eval}"
export HWA_STAGE3_EVAL_SPLITS="${HWA_STAGE3_EVAL_SPLITS:-Val,Test}"

python eval_stage3_best_classification.py
