# 新版本 v3：支持 cursor_dict，一次性生成所有模态 error map

import os
import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json


def load_nii(path):
    nii = nib.load(path)
    return nii.get_fdata()


def get_slice_from_itk_cursor(axis, cursor_xyz):
    x, y, z = cursor_xyz
    if axis == 0:
        return int(x)
    elif axis == 1:
        return int(y)
    elif axis == 2:
        return int(z)
    else:
        raise ValueError("axis must be 0/1/2")


def get_slice(volume, axis, idx):
    if axis == 0:
        return volume[idx, :, :]
    elif axis == 1:
        return volume[:, idx, :]
    else:
        return volume[:, :, idx]


def binarize(x):
    return (x > 0).astype(np.uint8)


def build_error_map_union(gt, pred):
    gt = binarize(gt)
    pred = binarize(pred)

    union = (gt > 0) | (pred > 0)
    inter = (gt > 0) & (pred > 0)

    err = np.zeros_like(gt)
    err[inter] = 1
    err[union & (~inter)] = 2
    return err


def plot_and_save(image, gt, pred, err, save_path, correct_color, wrong_color):
    cmap = ListedColormap([
        (0,0,0,0),
        correct_color,
        wrong_color
    ])

    plt.figure(figsize=(16,4))

    plt.subplot(1,4,1)
    plt.imshow(image, cmap='gray')
    plt.title("Image")
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.imshow(image, cmap='gray')
    plt.imshow(gt, alpha=0.4)
    plt.title("GT")
    plt.axis('off')

    plt.subplot(1,4,3)
    plt.imshow(image, cmap='gray')
    plt.imshow(pred, alpha=0.4)
    plt.title("Pred")
    plt.axis('off')

    plt.subplot(1,4,4)
    plt.imshow(image, cmap='gray')
    plt.imshow(err, cmap=cmap, alpha=0.5, vmin=0, vmax=2)
    plt.title("Error Map")
    plt.axis('off')

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"[Saved] {save_path}")

    plt.close()


def process_single_modality(model_dir, case_id, modality, cursor_xyz, axis, save_dir, correct_color, wrong_color):
    base = os.path.join(model_dir, f"{case_id}_{modality}")

    img = load_nii(base + "_image.nii.gz")
    gt = load_nii(base + "_gt.nii.gz")
    pred = load_nii(base + "_segmap.nii.gz")

    slice_idx = get_slice_from_itk_cursor(axis, cursor_xyz)

    img2d = get_slice(img, axis, slice_idx)
    gt2d = get_slice(gt, axis, slice_idx)
    pred2d = get_slice(pred, axis, slice_idx)

    err2d = build_error_map_union(gt2d, pred2d)

    save_path = os.path.join(
        save_dir,
        f"{case_id}_{modality}_axis{axis}_slice{slice_idx}_errormap.png"
    )

    plot_and_save(img2d, gt2d, pred2d, err2d,
                  save_path, correct_color, wrong_color)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--case_id', type=str, required=True)

    # 🔥 核心：cursor_dict（JSON字符串）
    parser.add_argument('--cursor_dict', type=str, required=True,
                        help='JSON格式，例如 {"ADC":[157,68,16],"T2_FS":[183,110,16],"V":[161,78,53]}')

    parser.add_argument('--axis', type=int, default=2)
    parser.add_argument('--correct_color', type=str, default='lime')
    parser.add_argument('--wrong_color', type=str, default='red')
    parser.add_argument('--save_dir', type=str, required=True)

    args = parser.parse_args()

    cursor_dict = json.loads(args.cursor_dict)

    print("\n[Processing all modalities]")

    for modality, cursor_xyz in cursor_dict.items():
        print(f"--> {modality} | cursor={cursor_xyz}")

        process_single_modality(
            model_dir=args.model_dir,
            case_id=args.case_id,
            modality=modality,
            cursor_xyz=cursor_xyz,
            axis=args.axis,
            save_dir=args.save_dir,
            correct_color=args.correct_color,
            wrong_color=args.wrong_color
        )

    print("\n[All Done]")


if __name__ == '__main__':
    main()
    # python error_map_generter.py \
    # --model_dir /workspace/HWA/segmap/GCM/HWA_UNETR_v2 \
    # --case_id 0002367533 \
    # --cursor_dict '{"ADC":[157,68,16],"T2_FS":[183,110,16],"V":[161,78,53]}' \
    # --axis 2 \
    # --correct_color green \
    # --wrong_color red \
    # --save_dir ./vis_all_modalities