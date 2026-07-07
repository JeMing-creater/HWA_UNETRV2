# Open Source Notes

This directory contains the public HWA-UNETRv2 project package. It includes
the model code, stage-wise training entry points, evaluation utilities,
sample split manifests, and sanitized configuration templates.

Excluded artifacts:

- private dataset volumes and labels
- historical experiment logs
- visualization caches
- trained checkpoints
- local absolute paths

The shared training logic is implemented in `GCM_train_core.py`, while the
three stages are exposed as separate entry points:

1. Stage 1 trains the center anchoring detector.
2. Stage 2 loads the detector and trains HWA-guided segmentation.
3. Stage 3 loads the segmentation checkpoint and trains patient-level prediction.
