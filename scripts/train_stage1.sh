#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/stage1_detector.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"

torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" \
  train.py --stage stage1 --config "${CONFIG}"
