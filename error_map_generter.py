import os
import json
import argparse
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


def load_nii(path: str) -> np.ndarray:
    nii = nib.load(path)
    return nii.get_fdata()


def normalize_to_uint8(image_2d: np.ndarray, pmin: float = 1.0, pmax: float = 99.0) -> np.ndarray:
    image_2d = np.asarray(image_2d, dtype=np.float32)
    lo = np.percentile(image_2d, pmin)
    hi = np.percentile(image_2d, pmax)

    if hi <= lo:
        lo = float(np.min(image_2d))
        hi = float(np.max(image_2d))

    if hi <= lo:
        return np.zeros_like(image_2d, dtype=np.uint8)

    image_2d = np.clip(image_2d, lo, hi)
    image_2d = (image_2d - lo) / (hi - lo + 1e-8)
    return (image_2d * 255.0).astype(np.uint8)


def binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (mask > threshold).astype(np.uint8)


def get_slice_from_itk_cursor(axis: int, cursor_xyz: List[int]) -> int:
    x, y, z = cursor_xyz
    if axis == 0:
        return int(x)
    if axis == 1:
        return int(y)
    if axis == 2:
        return int(z)
    raise ValueError("axis must be 0 / 1 / 2")


def get_slice(volume: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Only 3D volume is supported, got shape={volume.shape}")

    if axis == 0:
        return volume[idx, :, :]
    if axis == 1:
        return volume[:, idx, :]
    if axis == 2:
        return volume[:, :, idx]
    raise ValueError("axis must be 0 / 1 / 2")


def rotate_if_needed(arr_2d: np.ndarray, rotate_k: int) -> np.ndarray:
    return np.rot90(arr_2d, k=rotate_k)


def build_error_map_union(gt_2d: np.ndarray, pred_2d: np.ndarray) -> np.ndarray:
    gt = binarize(gt_2d)
    pred = binarize(pred_2d)

    union = (gt > 0) | (pred > 0)
    inter = (gt > 0) & (pred > 0)
    mismatch = union & (~inter)

    err = np.zeros_like(gt, dtype=np.uint8)
    err[inter] = 1
    err[mismatch] = 2
    return err


def gray_to_rgb(image_uint8: np.ndarray) -> np.ndarray:
    return np.stack([image_uint8, image_uint8, image_uint8], axis=-1)


def apply_binary_highlight_overlay(
    image_2d: np.ndarray,
    mask_2d: np.ndarray,
    highlight_color: str,
    alpha: float,
) -> np.ndarray:
    image_uint8 = normalize_to_uint8(image_2d)
    base = gray_to_rgb(image_uint8).astype(np.float32) / 255.0
    mask = binarize(mask_2d).astype(bool)
    color = np.array(to_rgb(highlight_color), dtype=np.float32)

    out = base.copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def render_binary_mask_png(mask_2d: np.ndarray, color: str) -> np.ndarray:
    mask = binarize(mask_2d).astype(np.uint8)
    color_rgb = (np.array(to_rgb(color)) * 255.0).astype(np.uint8)
    canvas = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    canvas[mask > 0] = color_rgb
    return canvas


def apply_error_overlay(
    image_2d: np.ndarray,
    error_map_2d: np.ndarray,
    correct_color: str,
    wrong_color: str,
    alpha: float,
) -> np.ndarray:
    image_uint8 = normalize_to_uint8(image_2d)
    base = gray_to_rgb(image_uint8).astype(np.float32) / 255.0
    out = base.copy()

    correct_mask = error_map_2d == 1
    wrong_mask = error_map_2d == 2

    correct_rgb = np.array(to_rgb(correct_color), dtype=np.float32)
    wrong_rgb = np.array(to_rgb(wrong_color), dtype=np.float32)

    out[correct_mask] = (1.0 - alpha) * out[correct_mask] + alpha * correct_rgb
    out[wrong_mask] = (1.0 - alpha) * out[wrong_mask] + alpha * wrong_rgb
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def save_png(rgb_array: np.ndarray, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.imsave(save_path, rgb_array)
    print(f"[Saved] {save_path}")


def resolve_triplet_paths(model_dir: str, case_id: str, modality: str) -> Tuple[str, str, str]:
    base = os.path.join(model_dir, f"{case_id}_{modality}")
    image_path = base + "_image.nii.gz"
    gt_path = base + "_gt.nii.gz"
    pred_path = base + "_segmap.nii.gz"

    missing = [p for p in [image_path, gt_path, pred_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    return image_path, gt_path, pred_path


def process_single_modality(
    model_dir: str,
    model_name: str,
    case_id: str,
    modality: str,
    cursor_xyz: List[int],
    axis: int,
    save_root_dir: str,
    gt_color: str,
    correct_color: str,
    wrong_color: str,
    gt_alpha: float,
    error_alpha: float,
    rotate_k: int,
) -> None:
    image_path, gt_path, pred_path = resolve_triplet_paths(model_dir, case_id, modality)

    image_3d = load_nii(image_path)
    gt_3d = load_nii(gt_path)
    pred_3d = load_nii(pred_path)

    if image_3d.shape != gt_3d.shape or gt_3d.shape != pred_3d.shape:
        raise ValueError(
            f"Shape mismatch for {model_name}/{modality}: image={image_3d.shape}, gt={gt_3d.shape}, pred={pred_3d.shape}"
        )

    slice_idx = get_slice_from_itk_cursor(axis, cursor_xyz)
    axis_len = image_3d.shape[axis]
    if not (0 <= slice_idx < axis_len):
        raise IndexError(
            f"slice_idx={slice_idx} is out of range for model={model_name}, modality={modality}, axis={axis}, axis_len={axis_len}"
        )

    image_2d = rotate_if_needed(get_slice(image_3d, axis, slice_idx), rotate_k)
    gt_2d = rotate_if_needed(get_slice(gt_3d, axis, slice_idx), rotate_k)
    pred_2d = rotate_if_needed(get_slice(pred_3d, axis, slice_idx), rotate_k)

    error_map_2d = build_error_map_union(gt_2d, pred_2d)

    gt_overlay_png = apply_binary_highlight_overlay(
        image_2d=image_2d,
        mask_2d=gt_2d,
        highlight_color=gt_color,
        alpha=gt_alpha,
    )
    gt_mask_png = render_binary_mask_png(gt_2d, color=gt_color)
    error_overlay_png = apply_error_overlay(
        image_2d=image_2d,
        error_map_2d=error_map_2d,
        correct_color=correct_color,
        wrong_color=wrong_color,
        alpha=error_alpha,
    )

    out_dir = os.path.join(save_root_dir, model_name, modality)
    stem = f"{case_id}_{modality}_axis{axis}_slice{slice_idx}"

    save_png(gt_overlay_png, os.path.join(out_dir, stem + "_gt_overlay.png"))
    save_png(gt_mask_png, os.path.join(out_dir, stem + "_gt_mask.png"))
    save_png(error_overlay_png, os.path.join(out_dir, stem + "_error_map.png"))


def process_all_models(
    models_root_dir: str,
    case_id: str,
    cursor_dict: Dict[str, List[int]],
    axis: int,
    save_root_dir: str,
    gt_color: str,
    correct_color: str,
    wrong_color: str,
    gt_alpha: float,
    error_alpha: float,
    rotate_k: int,
) -> None:
    model_names = sorted(
        [name for name in os.listdir(models_root_dir) if os.path.isdir(os.path.join(models_root_dir, name))]
    )

    if not model_names:
        raise FileNotFoundError(f"No model folders found in: {models_root_dir}")

    print("\n[Processing all models and modalities]")
    for model_name in model_names:
        model_dir = os.path.join(models_root_dir, model_name)
        print(f"\n=== Model: {model_name} ===")
        for modality, cursor_xyz in cursor_dict.items():
            try:
                print(f"--> modality={modality}, cursor={cursor_xyz}")
                process_single_modality(
                    model_dir=model_dir,
                    model_name=model_name,
                    case_id=case_id,
                    modality=modality,
                    cursor_xyz=cursor_xyz,
                    axis=axis,
                    save_root_dir=save_root_dir,
                    gt_color=gt_color,
                    correct_color=correct_color,
                    wrong_color=wrong_color,
                    gt_alpha=gt_alpha,
                    error_alpha=error_alpha,
                    rotate_k=rotate_k,
                )
            except Exception as e:
                print(f"[Skipped] model={model_name}, modality={modality}, reason={e}")

    print("\n[All Done]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate GT overlays and error maps for all models and modalities.")
    parser.add_argument("--models_root_dir", type=str, required=True, help="Root directory containing model folders.")
    parser.add_argument("--case_id", type=str, required=True, help="Case ID, e.g. 0002367533")
    parser.add_argument(
        "--cursor_dict",
        type=str,
        default='{"ADC":[157,68,16],"T2_FS":[183,110,16],"V":[161,78,53]}',
        help='JSON dict: {"ADC":[157,68,16],"T2_FS":[183,110,16],"V":[161,78,53]}'
    )
    parser.add_argument("--axis", type=int, default=2, choices=[0, 1, 2], help="Slice axis")
    parser.add_argument("--save_root_dir", type=str, required=True, help="Directory to save all PNG outputs")
    parser.add_argument("--gt_color", type=str, default="yellow", help="GT highlight color")
    parser.add_argument("--correct_color", type=str, default="green", help="Color for GT∩Pred")
    parser.add_argument("--wrong_color", type=str, default="red", help="Color for union mismatch region")
    parser.add_argument("--gt_alpha", type=float, default=0.55, help="Alpha for GT overlay")
    parser.add_argument("--error_alpha", type=float, default=0.65, help="Alpha for error overlay")
    parser.add_argument("--rotate_k", type=int, default=1, help="Rotate 90 degrees k times before saving")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cursor_dict = json.loads(args.cursor_dict)

    process_all_models(
        models_root_dir=args.models_root_dir,
        case_id=args.case_id,
        cursor_dict=cursor_dict,
        axis=args.axis,
        save_root_dir=args.save_root_dir,
        gt_color=args.gt_color,
        correct_color=args.correct_color,
        wrong_color=args.wrong_color,
        gt_alpha=args.gt_alpha,
        error_alpha=args.error_alpha,
        rotate_k=args.rotate_k,
    )


if __name__ == "__main__":
    main()
    # python error_map_generter.py \
    # --models_root_dir /workspace/HWA/segmap/GCM/ \
    # --case_id "0002682076" \
    # --cursor_dict '{"ADC":[157,68,16],"T2_FS":[183,110,16],"V":[161,78,53]}' \
    # --axis 2 \
    # --save_root_dir /workspace/HWA/vis_all_modalities/ \
    # --gt_color red \
    # --correct_color green \
    # --wrong_color red