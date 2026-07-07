#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/stage3_classification.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29503}"

torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" \
  train.py --stage stage3 --config "${CONFIG}"
