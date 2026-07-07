# Training Guide

This guide describes how to train HWA-UNETRv2 on multimodal gastric MRI data.
The public codebase exposes three stages: center anchoring detector training,
HWA-guided lesion segmentation, and patient-level prognostic prediction.

## 1. Environment

Create and activate a Python environment:

```bash
conda create -n hwav2 python=3.11 -y
conda activate hwav2
bash scripts/install_patched_mamba.sh
```

The installation script first installs the common Python dependencies, then
builds `requirements/Mamba/causal-conv1d` and `requirements/Mamba/mamba` from
the local sources. Before building Mamba, it copies
`requirements/mamba_simple.py` into the local Mamba package so that TDR-Mamba
can call `Mamba(..., bimamba_type="v3", nslices=...)` and receive four outputs
`(out, fwd, bwd, slc)`. Do not replace this step with plain
`pip install mamba-ssm causal-conv1d`. If the local build fails, confirm that
the active environment has a compatible CUDA build toolchain. The default
script installs the CUDA 12.4 PyTorch wheel and a Mamba-compatible Transformers
version before building the patched Mamba packages.

After installation, you can re-run the interface check:

```bash
python scripts/check_tdr_mamba.py
```

## 2. Data Preparation

Set the dataset root in `config.yml` or in the stage-specific config files under
`configs/`:

```yaml
GCM_loader:
  root: /path/to/GCM_dataset
```

The expected data layout is:

```text
/path/to/GCM_dataset/
  Classification.xlsx
  ALL/
    case_id_1/
      ADC/
        case_id_1.nii.gz
        case_id_1seg.nii.gz
      T2_FS/
        case_id_1.nii.gz
        case_id_1seg.nii.gz
      V/
        case_id_1.nii.gz
        case_id_1seg.nii.gz
```

The default modality list is:

```yaml
checkModels:
  - ADC
  - T2_FS
  - V
```

Prepare split files with one case identifier per line:

```text
splits/train_examples.txt
splits/val_examples.txt
splits/test_examples.txt
```

The default configs already point to these relative split files.

## 3. Config Files

Use the stage-specific configs for normal training:

```text
configs/stage1_detector.yaml        Stage 1 detector pre-training
configs/stage2_segmentation.yaml    Stage 2 HWA-guided segmentation
configs/stage3_classification.yaml  Stage 3 prognostic prediction
```

Important fields:

```yaml
trainer:
  batch_size: 6
  num_epochs: 160
  lr: 1.2e-05
  grad_accum_steps: 3

GCM_loader:
  root: /path/to/GCM_dataset
  train_examples_path: splits/train_examples.txt
  val_examples_path: splits/val_examples.txt
  test_examples_path: splits/test_examples.txt
  target_size: [128, 128, 64]
```

Checkpoint names are defined in `stage_train`:

```yaml
stage_train:
  stage1:
    checkpoint_name: HWA_stage1_detector
  stage2:
    checkpoint_name: HWA_stage2_segmentation
    init_checkpoint: HWA_stage1_detector
  stage3:
    checkpoint_name: HWA_stage3_classification
    init_checkpoint: HWA_stage2_segmentation
```

## 4. Training

Train the three stages in order.
`GCM_train_core.py` contains the shared training implementation, and
`GCM_train_stage1.py`, `GCM_train_stage2.py`, and `GCM_train_stage3.py` are
stage-specific launch wrappers.

Stage 1 trains the center anchoring detector:

```bash
bash scripts/train_stage1.sh configs/stage1_detector.yaml
```

Stage 2 loads the Stage 1 detector checkpoint and trains HWA-guided lesion
segmentation:

```bash
bash scripts/train_stage2.sh configs/stage2_segmentation.yaml
```

Stage 3 loads the Stage 2 segmentation checkpoint and trains the patient-level
prediction branch:

```bash
bash scripts/train_stage3.sh configs/stage3_classification.yaml
```

You can also use the unified wrapper:

```bash
bash run.sh stage1
bash run.sh stage2
bash run.sh stage3
```

## 5. Multi-GPU Training

Set `NPROC_PER_NODE` before launching a stage:

```bash
NPROC_PER_NODE=2 bash scripts/train_stage2.sh configs/stage2_segmentation.yaml
```

If the default ports are occupied, set `MASTER_PORT`:

```bash
MASTER_PORT=29602 NPROC_PER_NODE=2 bash scripts/train_stage2.sh configs/stage2_segmentation.yaml
```

## 6. Outputs

Training outputs are written under:

```text
model_store/<checkpoint_name>/
logs/<checkpoint_name><timestamp>/
```

The main checkpoint locations are:

```text
model_store/HWA_stage1_detector/best_stage1/
model_store/HWA_stage2_segmentation/best_stage2/
model_store/HWA_stage3_classification/best_stage3/
```

These output directories are ignored by Git and should not be committed.

## 7. Evaluation

Evaluate the best Stage 2 segmentation checkpoint:

```bash
bash scripts/eval_stage2.sh
```

Evaluate the best Stage 3 classification checkpoint:

```bash
bash scripts/eval_stage3.sh
```

Override checkpoint paths when needed:

```bash
HWA_STAGE2_EVAL_CKPT=/path/to/pytorch_model.bin bash scripts/eval_stage2.sh
HWA_STAGE3_EVAL_CKPT=/path/to/pytorch_model.bin bash scripts/eval_stage3.sh
```

Evaluation records are saved under `evaluation_records/`.

## 8. Quick Checks

Before training, verify the split files and dataset root:

```bash
python scripts/check_dataset_paths.py --config configs/stage2_segmentation.yaml
```

If training cannot find a checkpoint, confirm that the previous stage finished
and that the corresponding `checkpoint_name` matches the next stage
`init_checkpoint`.
