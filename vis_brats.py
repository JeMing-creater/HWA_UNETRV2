#!/usr/bin/env python3
"""
BraTS Segmentation Visualization
Usage: python vis_brats.py --sample_idx 0 --output vis.png
       python vis_brats.py --sample_idx 0 --model_path /path/to/model.safetensors --output vis.png
"""
import sys
sys.path.insert(0, '/workspace/HWA')

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from safetensors.torch import load_file
from monai.networks.nets import SwinUNETR
from src.BraTS_loader import get_dataloader_BraTS
from easydict import EasyDict
import yaml

# Default paths
DEFAULT_MODEL_PATH = '/workspace/HWA/model_store/SwinUNETR_20260406_075709/best/model.safetensors'
DEFAULT_OUTPUT = '/workspace/HWA/vis_output.png'

# Color scheme: ET=red, WT=green, TC=yellow
ET_COLOR = [1, 1, 0, 0.8]   # Yellow
WT_COLOR = [0, 1, 0, 0.5]   # Green (outer)
TC_COLOR = [1, 0, 0, 0.8]   # Red (inner)

def parse_args():
    parser = argparse.ArgumentParser(description='BraTS Segmentation Visualization')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index in validation set')
    parser.add_argument('--slice_z', type=int, default=None, help='Z slice to visualize (auto if None)')
    parser.add_argument('--modality', type=int, default=1, help='MRI modality index: 0=T1, 1=T1ce, 2=T2, 3=FLAIR')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to model weights')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT, help='Output image path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    parser.add_argument('--dpi', type=int, default=150, help='Output DPI')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set GPU
    torch.cuda.set_device(args.gpu)
    print(f"Using GPU {args.gpu}")
    
    # Load config
    config = EasyDict(yaml.load(open('/workspace/HWA/config.yml', 'r'), Loader=yaml.FullLoader))
    train_loader, val_loader, test_loader = get_dataloader_BraTS(config)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = SwinUNETR(img_size=128, in_channels=4, out_channels=3, 
                      feature_size=48, use_checkpoint=False, spatial_dims=3)
    ckpt = load_file(args.model_path, device='cpu')
    model.load_state_dict(ckpt, strict=False)
    model = model.cuda()
    model.eval()
    
    # Get specific sample
    print(f"Loading sample {args.sample_idx}...")
    val_iter = iter(val_loader)
    for i in range(args.sample_idx + 1):
        batch = next(val_iter)
    
    image = batch['image'][0].numpy()  # [4, 240, 240, 155]
    label = batch['label'][0].numpy()  # [3, 240, 240, 155]
    h, w, d = 240, 240, 155
    
    # Determine slice
    if args.slice_z is None:
        # Use slice with most ET
        et_per_z = label[2].sum(axis=(0,1))
        best_z = int(np.argmax(et_per_z))
    else:
        best_z = args.slice_z
    
    print(f"Using slice Z={best_z}")
    
    # Normalize input
    img_modality = image[args.modality].copy()
    img_modality = (img_modality - img_modality.min()) / (img_modality.max() - img_modality.min() + 1e-8)
    
    # Pad for SwinUNETR
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    pad_d = (32 - d % 32) % 32
    image_padded = np.pad(image, ((0,0),(0,pad_h),(0,pad_w),(0,pad_d)), mode='constant')
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        image_tensor = torch.from_numpy(image_padded).unsqueeze(0).cuda()
        output = model(image_tensor)
        output = torch.sigmoid(output).cpu().numpy()[0]
    
    output = output[:, :h, :w, :d]
    pred = (output > args.threshold).astype(np.float32)
    
    # Get 2D slices
    img_2d = img_modality[:, :, best_z].T
    wt_gt = label[1][:, :, best_z].T
    tc_gt = label[0][:, :, best_z].T
    et_gt = label[2][:, :, best_z].T
    wt_pred = pred[1][:, :, best_z].T
    tc_pred = pred[0][:, :, best_z].T
    et_pred = pred[2][:, :, best_z].T
    
    modality_names = ['T1', 'T1ce', 'T2', 'FLAIR']
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Input
    axes[0].imshow(img_2d, cmap='gray', origin='lower', vmin=0, vmax=1)
    axes[0].set_title(f'Input ({modality_names[args.modality]})', fontsize=16)
    axes[0].axis('off')
    
    # GT overlay
    axes[1].imshow(img_2d, cmap='gray', origin='lower', vmin=0, vmax=1)
    
    overlay = np.zeros((h, w, 4))
    overlay[wt_gt > 0.5] = WT_COLOR   # Green - outer
    overlay[et_gt > 0.5] = ET_COLOR   # Yellow - middle
    overlay[tc_gt > 0.5] = TC_COLOR   # Red - inner
    
    axes[1].imshow(overlay, origin='lower')
    legend_elements = [
        Patch(facecolor='green', alpha=0.8, label='WT (Whole Tumor)'),
        Patch(facecolor='yellow', alpha=0.8, label='ET (Enhancing)'),
        Patch(facecolor='red', alpha=0.8, label='TC (Tumor Core)'),
    ]
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=11)
    axes[1].set_title('Ground Truth', fontsize=16)
    axes[1].axis('off')
    
    # Pred overlay
    axes[2].imshow(img_2d, cmap='gray', origin='lower', vmin=0, vmax=1)
    
    overlay_p = np.zeros((h, w, 4))
    overlay_p[wt_pred > 0.5] = WT_COLOR
    overlay_p[et_pred > 0.5] = ET_COLOR
    overlay_p[tc_pred > 0.5] = TC_COLOR
    
    axes[2].imshow(overlay_p, origin='lower')
    axes[2].legend(handles=legend_elements, loc='upper right', fontsize=11)
    axes[2].set_title('Prediction', fontsize=16)
    axes[2].axis('off')
    
    plt.suptitle(f'Sample {args.sample_idx} | Slice Z={best_z}', fontsize=18)
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved: {args.output}")

if __name__ == '__main__':
    main()
