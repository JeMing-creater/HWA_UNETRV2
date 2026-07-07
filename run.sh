#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-stage2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

case "${STAGE}" in
  stage1)
    NPROC_PER_NODE="${NPROC_PER_NODE}" bash scripts/train_stage1.sh configs/stage1_detector.yaml
    ;;
  stage2)
    NPROC_PER_NODE="${NPROC_PER_NODE}" bash scripts/train_stage2.sh configs/stage2_segmentation.yaml
    ;;
  stage3)
    NPROC_PER_NODE="${NPROC_PER_NODE}" bash scripts/train_stage3.sh configs/stage3_classification.yaml
    ;;
  *)
    echo "Usage: bash run.sh [stage1|stage2|stage3]" >&2
    exit 2
    ;;
esac
