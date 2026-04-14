import os
import yaml
import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt

from easydict import EasyDict
from accelerate import Accelerator
from timm.optim import optim_factory

from get_model import get_model
from src.optimizer import LinearWarmupCosineAnnealingLR
import copy

def build_checkpoint_dir(model_store_root, model_name):
    return os.path.join(
        model_store_root,
        f"Colon_Segmentation{model_name}",
        "best",
    )


def run_one_model(
    model_name,
    base_config,
    target_cases,
    model_store_root,
    save_root,
    case_cache,
):
    print("\n" + "#" * 110)
    print(f"[Model] start: {model_name}")

    config = copy.deepcopy(base_config)
    config.trainer.choose_model = model_name

    checkpoint_dir = build_checkpoint_dir(model_store_root, model_name)
    model_save_root = os.path.join(save_root, model_name)

    if not os.path.exists(checkpoint_dir):
        print(f"[Warn] checkpoint not found, skip: {checkpoint_dir}")
        return

    if model_name != "HWAUNETR" or model_name != "UNETR_PP":
        accelerator, model = restore_with_accelerate(config, checkpoint_dir)
        device = accelerator.device

    for case_id, z_raw in target_cases:
        print("\n" + "=" * 90)
        print(f"[Model] {model_name}")
        print(f"[Case] {case_id}")
        print(f"[Case] target original z = {z_raw}")

        if case_id not in case_cache:
            image_raw, label_raw, ct_path, gt_path = load_case_nifti(case_id)
            case_cache[case_id] = {
                "image_raw": image_raw,
                "label_raw": label_raw,
                "ct_path": ct_path,
                "gt_path": gt_path,
            }

        item = case_cache[case_id]
        image_raw = item["image_raw"]
        label_raw = item["label_raw"]

        print(f"[Path] CT = {item['ct_path']}")
        print(f"[Path] GT = {item['gt_path']}")
        print(f"[Shape] raw image = {image_raw.shape}, raw label = {label_raw.shape}")

        image_full_scaled, gt_full_bin, image_patch, label_patch, meta = preprocess_case_trainlike_fixed_z(
            image_raw=image_raw,
            label_raw=label_raw,
            z_raw=z_raw,
            config=config,
        )

        print(f"[Meta] patch_start = {meta['patch_start'].tolist()}")
        print(f"[Meta] patch_end   = {meta['patch_end'].tolist()}")
        print(f"[Meta] z_pad       = {meta['z_pad']}")
        print(f"[Meta] z_in_patch  = {meta['z_in_patch']}")

        
        

            
        logits_patch, prob_patch, pred_patch = infer_one_case(
            model=model,
            device=device,
            image_patch=image_patch,
            config=config,
        )

        gt_patch_sum = label_patch.sum()
        pred_patch_sum = pred_patch.sum()

        print(
            f"[Debug] patch label sum = {gt_patch_sum:.6f} | "
            f"patch pred sum@0.5 = {pred_patch_sum:.6f}"
        )
        print(
            f"[Debug] patch prob min/max/mean = "
            f"{prob_patch.min():.6f} / {prob_patch.max():.6f} / {prob_patch.mean():.6f}"
        )

        pred_full = restore_patch_to_original_space(pred_patch, meta)

        gt_sum_z = gt_full_bin[:, :, z_raw].sum()
        pred_sum_z = pred_full[:, :, z_raw].sum()

        print(
            f"[Debug] original z={z_raw} | "
            f"gt_sum={gt_sum_z:.6f} | pred_sum@0.5={pred_sum_z:.6f}"
        )

        case_save_dir = os.path.join(model_save_root, case_id)
        out_path = save_visualization(
            case_id=case_id,
            z_raw=z_raw,
            image_full=image_full_scaled,
            gt_full=gt_full_bin,
            pred_full=pred_full,
            save_dir=case_save_dir,
        )
        print(f"[Save] {out_path}")


    
    del model
    del accelerator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Model] done: {model_name}")

def maybe_extract_logits(model, x, choose_model_name: str):
    with torch.no_grad():
        if choose_model_name in ["HWAUNETR", "HSL_Net"]:
            _, logits = model(x)
        else:
            logits = model(x)
    return logits


def build_model_optimizer_scheduler(config):
    model = get_model(config)

    optimizer = optim_factory.create_optimizer_v2(
        model,
        opt=config.trainer.optimizer,
        weight_decay=float(config.trainer.weight_decay),
        lr=float(config.trainer.lr),
        betas=(config.trainer.betas[0], config.trainer.betas[1]),
    )

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.trainer.warmup,
        max_epochs=config.trainer.num_epochs,
    )
    return model, optimizer, scheduler


def restore_with_accelerate(config, checkpoint_dir):
    accelerator = Accelerator(cpu=False)

    model, optimizer, scheduler = build_model_optimizer_scheduler(config)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    accelerator.print(f"[Restore] loading state from: {checkpoint_dir}")
    # accelerator.load_state(checkpoint_dir)
    accelerator.print("[Restore] state restored successfully.")

    model.eval()
    return accelerator, model


def scale_intensity_range(image, a_min, a_max):
    image = image.astype(np.float32)
    image = np.clip(image, a_min, a_max)
    image = (image - a_min) / (a_max - a_min + 1e-8)
    return image.astype(np.float32)


def pad_to_min_shape(arr, target_shape):
    pad_width = []
    for s, t in zip(arr.shape, target_shape):
        total = max(t - s, 0)
        before = total // 2
        after = total - before
        pad_width.append((before, after))
    out = np.pad(arr, pad_width, mode="constant", constant_values=0)
    return out, pad_width


def unpad_to_original_shape(arr, pads, orig_shape):
    slices = []
    for (before, after), ori in zip(pads, orig_shape):
        start = before
        end = before + ori
        slices.append(slice(start, end))
    return arr[tuple(slices)]


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def compute_patch_start(center, full_shape, patch_size):
    starts = []
    for c, s, p in zip(center, full_shape, patch_size):
        st = int(round(c - p / 2))
        st = clamp(st, 0, s - p)
        starts.append(st)
    return np.array(starts, dtype=np.int32)


def get_positive_center_close_to_target_z(label_pad, z_pad):
    pos_2d = np.argwhere(label_pad[:, :, z_pad] > 0)
    if pos_2d.size > 0:
        cx = int(np.round(pos_2d[:, 0].mean()))
        cy = int(np.round(pos_2d[:, 1].mean()))
        cz = int(z_pad)
        return np.array([cx, cy, cz], dtype=np.int32)

    pos_3d = np.argwhere(label_pad > 0)
    if pos_3d.size > 0:
        center = np.round(pos_3d.mean(axis=0)).astype(np.int32)
        center[2] = int(z_pad)
        return center

    shape = np.array(label_pad.shape, dtype=np.int32)
    center = shape // 2
    center[2] = int(z_pad)
    return center


def crop_patch(arr, start, patch_size):
    end = start + np.array(patch_size, dtype=np.int32)
    return arr[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2],
    ]


def paste_patch(canvas, patch, start):
    end = start + np.array(patch.shape, dtype=np.int32)
    canvas[
        start[0]:end[0],
        start[1]:end[1],
        start[2]:end[2],
    ] = patch
    return canvas


def preprocess_case_trainlike_fixed_z(image_raw, label_raw, z_raw, config):
    patch_size = tuple(config.colon_loader.target_size)
    a_min = config.colon_loader.intensity.a_min
    a_max = config.colon_loader.intensity.a_max

    image_scaled = scale_intensity_range(image_raw, a_min, a_max)
    label_bin = (label_raw > 0).astype(np.float32)

    image_pad, pads = pad_to_min_shape(image_scaled, patch_size)
    label_pad, _ = pad_to_min_shape(label_bin, patch_size)

    z_pad = z_raw + pads[2][0]
    z_pad = clamp(z_pad, 0, image_pad.shape[2] - 1)

    center = get_positive_center_close_to_target_z(label_pad, z_pad)
    start = compute_patch_start(center, image_pad.shape, patch_size)

    image_patch = crop_patch(image_pad, start, patch_size)
    label_patch = crop_patch(label_pad, start, patch_size)

    meta = {
        "orig_shape": image_raw.shape,
        "pads": pads,
        "padded_shape": image_pad.shape,
        "patch_size": patch_size,
        "patch_start": start,
        "patch_end": start + np.array(patch_size, dtype=np.int32),
        "z_raw": z_raw,
        "z_pad": z_pad,
        "z_in_patch": int(z_pad - start[2]),
    }

    return image_scaled, label_bin, image_patch, label_patch, meta


def restore_patch_to_original_space(pred_patch, meta):
    full_pad = np.zeros(meta["padded_shape"], dtype=pred_patch.dtype)
    full_pad = paste_patch(full_pad, pred_patch, meta["patch_start"])
    full = unpad_to_original_shape(full_pad, meta["pads"], meta["orig_shape"])
    return full


@torch.no_grad()
def infer_one_case(model, device, image_patch, config):
    x = torch.from_numpy(image_patch[None, None]).float().to(device)  # [1,1,X,Y,Z]
    logits = maybe_extract_logits(model, x, config.trainer.choose_model)
    prob = torch.sigmoid(logits)
    pred = (prob > 0.5).float()

    logits_np = logits.squeeze().detach().cpu().numpy().astype(np.float32)
    prob_np = prob.squeeze().detach().cpu().numpy().astype(np.float32)
    pred_np = pred.squeeze().detach().cpu().numpy().astype(np.float32)
    return logits_np, prob_np, pred_np


def plot_gt_overlay(ax, img_slice, gt_slice, title="GT + Original"):
    gt_mask = gt_slice > 0
    ax.imshow(img_slice, cmap="gray")

    rgba = np.zeros((*gt_mask.shape, 4), dtype=np.float32)
    rgba[..., 0] = gt_mask.astype(np.float32)   # 红色通道
    rgba[..., 1] = 0.0
    rgba[..., 2] = 0.0
    rgba[..., 3] = gt_mask.astype(np.float32) * 0.45  # 半透明红色覆盖
    ax.imshow(rgba)

    # 不显示轮廓线
    ax.set_title(title)
    ax.axis("off")


def plot_gt_only(ax, gt_slice, title="GT"):
    gt_mask = gt_slice > 0

    # 先创建黑底 RGB 图
    rgb = np.zeros((*gt_mask.shape, 3), dtype=np.float32)

    # GT 区域填成红色
    rgb[..., 0] = gt_mask.astype(np.float32)   # R
    rgb[..., 1] = 0.0                          # G
    rgb[..., 2] = 0.0                          # B

    ax.imshow(rgb)

    # 不显示轮廓线
    ax.set_title(title)
    ax.axis("off")


def plot_pred_overlay(ax, img_slice, gt_slice, pred_slice, title="Pred@0.5"):
    gt_mask = gt_slice > 0
    pred_mask = pred_slice > 0

    tp = gt_mask & pred_mask
    fp = (~gt_mask) & pred_mask
    fn = gt_mask & (~pred_mask)
    err = fp | fn

    ax.imshow(img_slice, cmap="gray")

    rgba = np.zeros((*gt_mask.shape, 4), dtype=np.float32)

    # 一致 -> 绿色
    rgba[..., 1] = tp.astype(np.float32)
    rgba[..., 3] = np.maximum(rgba[..., 3], tp.astype(np.float32) * 0.45)

    # 不一致 -> 红色
    rgba[..., 0] = np.maximum(rgba[..., 0], err.astype(np.float32))
    rgba[..., 3] = np.maximum(rgba[..., 3], err.astype(np.float32) * 0.45)

    ax.imshow(rgba)

    if tp.sum() > 0:
        ax.contour(tp.astype(np.uint8), levels=[0.5], colors="red", linewidths=1.2)
    if err.sum() > 0:
        ax.contour(err.astype(np.uint8), levels=[0.5], colors="red", linewidths=1.2)

    ax.set_title(title)
    ax.axis("off")

def save_visualization(case_id, z_raw, image_full, gt_full, pred_full, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    if not (0 <= z_raw < image_full.shape[2]):
        raise ValueError(f"{case_id}: z_raw={z_raw} out of range, image shape={image_full.shape}")

    # ===== 原始切片 =====
    img_slice = image_full[:, :, z_raw].T
    gt_slice = gt_full[:, :, z_raw].T
    pred_slice = pred_full[:, :, z_raw].T

    # ===== 左右 + 上下翻转 =====
    img_slice = np.flip(img_slice, axis=(0, 1))
    gt_slice = np.flip(gt_slice, axis=(0, 1))
    pred_slice = np.flip(pred_slice, axis=(0, 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    plot_gt_overlay(axes[0], img_slice, gt_slice, title="GT + Original")
    plot_gt_only(axes[1], gt_slice, title="GT")
    plot_pred_overlay(axes[2], img_slice, gt_slice, pred_slice, title="Pred@0.5")

    plt.suptitle(f"{case_id} | original-space z={z_raw}")
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{case_id}_z{z_raw}.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    return out_path

@torch.no_grad()
def disperse_segmentation_preserve_connectivity(
    img_bcwhz: torch.Tensor,
    kernel_size: int = 3,
    prob_add: float = 0.12,
    iterations: int = 1,
    dep_epper: bool = True,
    lonely_thresh: int = 2,
):
    """
    对 (B, C, W, H, Z) 二值分割做“连通性保真”的边界分散：
    - 仅对外侧边界做 0->1 的随机外扩（不会 1->0），因此不会降低连通性；
    - 可多次迭代增强分散效果；
    - 可选去“椒盐尖刺”：只移除新增且极孤立的体素，不会破坏连通性。

    参数
    ----
    img_bcwhz : torch.Tensor  (B, C, W, H, Z)，元素为 {0,1}/{False,True}
    kernel_size : int         形态学邻域（建议 3 或 5）
    prob_add : float          外扩概率（越大越“散”，0.08~0.2 常用）
    iterations : int          外扩迭代次数，>1 会更明显
    dep_epper : bool          是否清理孤立新增尖刺
    lonely_thresh : int       孤立阈值：邻域计数 <= (1+lonely_thresh) 视为孤立
                              （kernel_size=3 时，<=2 表示仅自己+≤1邻居）

    返回
    ----
    与输入相同布局与 dtype 的张量 (B, C, W, H, Z)。
    """
    assert img_bcwhz.dim() == 5, f"Expect 5D (B,C,W,H,Z), got {tuple(img_bcwhz.shape)}"
    B, C, W, H, Z = img_bcwhz.shape
    device, dtype = img_bcwhz.device, img_bcwhz.dtype

    # 转到 (B, C, D, H, W) 以适配 conv3d
    x = (img_bcwhz > 0.5).to(torch.bool).permute(0, 1, 4, 3, 2).contiguous()  # bool
    p = kernel_size // 2
    weight = torch.ones((C, 1, kernel_size, kernel_size, kernel_size), device=device)
    win_size = kernel_size**3

    y = x.clone()
    added_accum = torch.zeros_like(y, dtype=torch.bool)  # 仅用于可选去尖刺
    import torch.nn.functional as F
    for _ in range(iterations):
        # 统计邻域内前景数量
        conv = F.conv3d(y.float(), weight, padding=p, groups=C)

        # 外侧壳层候选: 与前景相邻的背景体素
        outside_shell = (~y) & (conv > 0)

        # 随机选择一部分外侧体素外扩
        add_mask = outside_shell & (torch.rand_like(conv) < prob_add)

        # 应用外扩（只 0->1）
        y = y | add_mask
        added_accum |= add_mask  # 累加记录“新增”的体素

    # 可选：去掉极孤立的新增“尖刺”
    if dep_epper:
        neigh = F.conv3d(y.float(), weight, padding=p, groups=C)  # 包含自身
        lonely_new = added_accum & (neigh <= (1 + lonely_thresh))
        # 移除这些孤立新增体素（只作用于新增部分，不会破坏原始连通性）
        y = y & (~lonely_new)

    # 还原回 (B, C, W, H, Z)
    y = y.permute(0, 1, 4, 3, 2).contiguous()

    # 保持原 dtype
    if dtype is torch.bool:
        return y
    elif dtype.is_floating_point:
        return y.float()
    else:
        return y.long()

def load_case_nifti(case_id):
    ct_path = f"/mnt/liangjm/AbdomenAtlas2/AbdomenAtlas2.0/data/AbdomenAtlas2.0/{case_id}/ct.nii.gz"
    gt_path = f"/mnt/liangjm/AbdomenAtlas2/AbdomenAtlas2.0/data/AbdomenAtlas2.0/{case_id}/segmentations/colon_lesion.nii.gz"

    if not os.path.exists(ct_path):
        raise FileNotFoundError(ct_path)
    if not os.path.exists(gt_path):
        raise FileNotFoundError(gt_path)

    image = nib.load(ct_path).get_fdata().astype(np.float32)
    label = nib.load(gt_path).get_fdata().astype(np.float32)
    return image, label, ct_path, gt_path


def main():
    config_path = "/workspace/HWA/config.yml"
    model_store_root = "/workspace/HWA/model_store"
    save_root = "./visualize_cases"

    target_cases = [
        ("BDMAP_00000004", 14), # (数据Id,z坐标)
        ("BDMAP_00000069", 64),
        ("BDMAP_00000264", 55),
        
    ]

    # 添加需要生成图像的模型
    models_to_run = [
        # "CoTr",
        # "UNETR_PP",
        # "SwinUNETR",
        # "UXNET",
        # "SaBNet",
        # "ALIEN",
        "HWAUNETR",
        "UNETR_PP"
    ]

    base_config = EasyDict(
        yaml.load(open(config_path, "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )

    case_cache = {}

    print("[Info] models to run:")
    for m in models_to_run:
        print(f"  - {m}")

    for model_name in models_to_run:
        try:
            run_one_model(
                model_name=model_name,
                base_config=base_config,
                target_cases=target_cases,
                model_store_root=model_store_root,
                save_root=save_root,
                case_cache=case_cache,
            )
        except Exception as e:
            print(f"[Error] model {model_name} failed: {e}")

    print("\n[Info] all done.")


if __name__ == "__main__":
    main()