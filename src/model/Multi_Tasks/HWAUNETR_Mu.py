# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (
            sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i))
        )


class Swish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim, shallow=True):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU() if shallow else Swish()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class GMPBlock(nn.Module):
    def __init__(self, in_channles, shallow=True) -> None:
        super().__init__()

        act = nn.GELU if shallow else Swish
        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = act()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = act()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = act()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = act()

    def forward(self, x):
        x_residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual


class TDRMambaBlock(nn.Module):
    """
    Tri-directional Relation Aggregation block.

    - keeps tri-directional Mamba scanning
    - replaces the old q/k/v interaction with directional relation modeling
    - uses the learned directional routing weights to fuse fwd/bwd/slice experts
    - finishes with a light local refinement branch for lesion cohesion
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, head=4, num_slices=4, step=1):
        super().__init__()
        self.dim = dim
        self.step = step
        self.num_heads = head
        self.head_dim = dim // head
        self.norm = nn.LayerNorm(dim)

        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="v3",
            nslices=num_slices,
        )

        self.dir_inner_dim = self.mamba.d_inner
        self.dir_proj = nn.Conv3d(self.dir_inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.relation_mlp = nn.Sequential(
            nn.Linear(9, max(dim // 2, 8)),
            nn.GELU(),
            nn.Linear(max(dim // 2, 8), 3),
        )

        self.local_refine = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True),
            nn.InstanceNorm3d(dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.fusion = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def _to_3d(self, x_seq: torch.Tensor, img_dims: Tuple[int, int, int], project: bool = False) -> torch.Tensor:
        if x_seq.dim() != 3:
            raise ValueError(f"Expected a 3D tensor, got shape {tuple(x_seq.shape)}")

        b = x_seq.shape[0]
        num_voxels = img_dims[0] * img_dims[1] * img_dims[2]

        if x_seq.shape[1] == num_voxels:
            feat = x_seq.transpose(1, 2).reshape(b, x_seq.shape[2], *img_dims)
        elif x_seq.shape[2] == num_voxels:
            feat = x_seq.reshape(b, x_seq.shape[1], *img_dims)
        else:
            raise ValueError(
                f"Cannot reshape tensor with shape {tuple(x_seq.shape)} to 3D dims {img_dims}; "
                f"expected one dimension to equal {num_voxels}."
            )

        if project:
            feat = self.dir_proj(feat)
        return feat

    def _direction_descriptors(self, feats: List[torch.Tensor]) -> torch.Tensor:
        tokens = [F.adaptive_avg_pool3d(f, 1).flatten(1) for f in feats]
        tokens = torch.stack(tokens, dim=1)  # (B, 3, C)
        tokens = F.normalize(tokens, p=2, dim=-1)
        relation = torch.bmm(tokens, tokens.transpose(1, 2))  # (B, 3, 3)
        return relation

    def forward(self, x):
        x_skip = x
        b, c, h, w, z = x.shape
        assert c == self.dim
        img_dims = (h, w, z)

        x_flat = x.reshape(b, c, -1).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        out, fwd, bwd, slc = self.mamba(x_norm)

        out_m = self._to_3d(out, img_dims)
        fwd_m = self._to_3d(fwd, img_dims, project=True)
        bwd_m = self._to_3d(bwd, img_dims, project=True)
        slc_m = self._to_3d(slc, img_dims, project=True)

        relation = self._direction_descriptors([fwd_m, bwd_m, slc_m])
        route_logits = self.relation_mlp(relation.flatten(1))
        route = torch.softmax(route_logits, dim=1)

        fused_dir = (
            route[:, 0].view(b, 1, 1, 1, 1) * fwd_m
            + route[:, 1].view(b, 1, 1, 1, 1) * bwd_m
            + route[:, 2].view(b, 1, 1, 1, 1) * slc_m
        )

        cohesion = torch.sigmoid(self.local_refine(fused_dir))
        refined_dir = fused_dir * cohesion

        out = self.fusion(torch.cat([out_m, refined_dir], dim=1))
        out = out + x_skip
        return out

class ROICatFusionBlock(nn.Module):
    def __init__(self, dim: int, stage_index: int = 0):
        super().__init__()
        hidden_dim = max(dim // 2, 16)
        self.stage_index = int(stage_index)
        self.runtime_gate_scale = 1.0
        # Stage-wise prior fusion is initialized as an input-level soft guidance path.
        stage_gate_multipliers = (0.0, 0.0, 0.0, 0.0)
        direct_scales = (0.0, 0.0, 0.0, 0.0)
        max_residual_ratios = (0.060, 0.050, 0.040, 0.030)
        idx = max(0, min(self.stage_index, len(stage_gate_multipliers) - 1))
        self.stage_gate_multiplier = float(stage_gate_multipliers[idx])
        self.direct_residual_scale = float(direct_scales[idx])
        self.max_residual_ratio = float(max_residual_ratios[idx])
        self.self_prior_mix = 0.90 if self.stage_index <= 1 else 0.75
        init_guidance_scale = 0.02
        init_logit = math.log(init_guidance_scale / (1.0 - init_guidance_scale))
        self.guidance_scale_logit = nn.Parameter(
            torch.full((1, dim, 1, 1, 1), init_logit, dtype=torch.float32)
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=1, bias=False),
            nn.InstanceNorm3d(dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=1, bias=True),
        )
        self.gate_net = nn.Sequential(
            nn.Conv3d(dim * 2, hidden_dim, kernel_size=1, bias=True),
            nn.InstanceNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        final_proj = self.fusion[-1]
        nn.init.normal_(final_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final_proj.bias)

    @staticmethod
    def _feature_saliency(x: torch.Tensor) -> torch.Tensor:
        saliency = x.detach().float().abs().mean(dim=1, keepdim=True)
        sal_min = saliency.amin(dim=(2, 3, 4), keepdim=True)
        sal_max = saliency.amax(dim=(2, 3, 4), keepdim=True)
        saliency = (saliency - sal_min) / (sal_max - sal_min).clamp_min(1e-4)
        return saliency.to(dtype=x.dtype)

    @staticmethod
    def _roi_self_prior(x: torch.Tensor, quality_map: torch.Tensor) -> torch.Tensor:
        q = quality_map.to(dtype=x.dtype)
        denom = q.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)
        roi_mean = (x * q).sum(dim=(2, 3, 4), keepdim=True) / denom
        kernel = tuple(3 if int(size) >= 3 else 1 for size in x.shape[2:])
        padding = tuple(1 if k == 3 else 0 for k in kernel)
        context = F.avg_pool3d(x, kernel_size=kernel, stride=1, padding=padding)
        return x + q * (0.50 * (context - x) + 0.25 * (x - roi_mean))

    @staticmethod
    def _masked_match_stats(
        prior_feat: torch.Tensor,
        x: torch.Tensor,
        quality_map: torch.Tensor,
    ) -> torch.Tensor:
        q = quality_map.to(dtype=torch.float32)
        denom = q.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)

        x_f = x.float()
        prior_f = prior_feat.float()
        x_mean = (x_f * q).sum(dim=(2, 3, 4), keepdim=True) / denom
        p_mean = (prior_f * q).sum(dim=(2, 3, 4), keepdim=True) / denom
        x_var = ((x_f - x_mean) * q).pow(2).sum(dim=(2, 3, 4), keepdim=True) / denom
        p_var = ((prior_f - p_mean) * q).pow(2).sum(dim=(2, 3, 4), keepdim=True) / denom

        x_std = x_var.clamp_min(1e-6).sqrt()
        p_std = p_var.clamp_min(1e-6).sqrt()
        matched = (prior_f - p_mean) / p_std * x_std + x_mean
        return matched.to(dtype=x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        prior: torch.Tensor,
        return_debug: bool = False,
    ):
        if isinstance(prior, dict):
            prior_feat = prior.get("feat", None)
            quality_map = prior.get("quality_map", None)
            quality_score = prior.get("quality_score", None)
        else:
            prior_feat = prior
            quality_map = None
            quality_score = None

        if prior_feat is None:
            raise RuntimeError("ROICatFusionBlock received an empty prior feature.")

        if prior_feat.shape[2:] != x.shape[2:]:
            prior_feat = F.interpolate(
                prior_feat.float(),
                size=x.shape[2:],
                mode="trilinear",
                align_corners=False,
            ).to(dtype=x.dtype)

        if quality_map is None:
            quality_map = torch.ones(
                x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4],
                device=x.device, dtype=x.dtype,
            )
        else:
            if quality_map.shape[2:] != x.shape[2:]:
                quality_map = F.interpolate(
                    quality_map.float(),
                    size=x.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
            if quality_map.shape[1] != 1:
                quality_map = quality_map.mean(dim=1, keepdim=True)
            quality_map = quality_map.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)

        feature_saliency = self._feature_saliency(x)
        roi_mass = quality_map.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)
        roi_saliency = (feature_saliency * quality_map).sum(
            dim=(2, 3, 4), keepdim=True
        ) / roi_mass
        global_saliency = feature_saliency.mean(dim=(2, 3, 4), keepdim=True)
        feature_support = torch.sigmoid(6.0 * (roi_saliency - global_saliency))
        quality_map = (quality_map * (0.60 + 0.40 * feature_saliency)).clamp(0.0, 1.0)

        if quality_score is None:
            quality_score = quality_map.mean(dim=(2, 3, 4), keepdim=True)
        else:
            if not torch.is_tensor(quality_score):
                quality_score = torch.tensor(
                    float(quality_score), device=x.device, dtype=x.dtype
                )
            quality_score = quality_score.to(device=x.device, dtype=x.dtype)
            if quality_score.dim() == 1:
                quality_score = quality_score.view(-1, 1, 1, 1, 1)
            elif quality_score.dim() == 2:
                quality_score = quality_score.view(
                    quality_score.shape[0], quality_score.shape[1], 1, 1, 1
                )
            if quality_score.shape[1] != 1:
                quality_score = quality_score.mean(dim=1, keepdim=True)
            quality_score = quality_score.clamp(0.0, 1.0)
        quality_score = (quality_score * (0.40 + 0.60 * feature_support)).clamp(0.0, 1.0)

        gate_scale = float(max(0.0, min(1.0, getattr(self, "runtime_gate_scale", 1.0))))
        gate_scale *= self.stage_gate_multiplier
        if gate_scale <= 0.0 or self.max_residual_ratio <= 0.0:
            if not return_debug:
                return x
            empty_gate = x.new_zeros(x.shape[0], 1, *x.shape[2:])
            guidance_scale = torch.sigmoid(self.guidance_scale_logit).to(
                device=x.device, dtype=x.dtype
            )
            return x, empty_gate, quality_score.detach(), guidance_scale.detach()

        guidance_scale = torch.sigmoid(self.guidance_scale_logit).to(
            device=x.device, dtype=x.dtype
        )
        prior_feat = self._masked_match_stats(prior_feat, x, quality_map)
        self_prior = self._roi_self_prior(x, quality_map)
        mix = float(max(0.0, min(1.0, getattr(self, "self_prior_mix", 0.85))))
        prior_feat = mix * self_prior + (1.0 - mix) * prior_feat
        prior_delta = prior_feat - x
        guided_prior = (
            x
            + quality_map
            * quality_score
            * guidance_scale
            * prior_delta
        )
        fusion_input = torch.cat([x, guided_prior], dim=1)
        x_std = x.float().std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-4).to(dtype=x.dtype)
        direct_correction = self.direct_residual_scale * x_std * torch.tanh(
            prior_delta / x_std
        )
        correction = self.fusion(fusion_input)
        correction = correction + direct_correction
        correction = self.max_residual_ratio * x_std * torch.tanh(
            correction / x_std
        )
        gate_map = self.gate_net(fusion_input)
        gate_map = gate_map * quality_map * quality_score * gate_scale
        out = x + gate_map * correction

        if not return_debug:
            return out

        return out, gate_map, quality_score.detach(), guidance_scale.detach()


class Encoder(nn.Module):
    def __init__(
        self,
        in_chans=4,
        kernel_sizes=[4, 2, 2, 2],
        depths=[1, 1, 1, 1],
        dims=[48, 96, 192, 384],
        num_slices_list=[64, 32, 16, 8],
        out_indices=[0, 1, 2, 3],
        heads=[1, 2, 4, 4],
    ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(
                in_chans, dims[0], kernel_size=kernel_sizes[0], stride=kernel_sizes[0]
            )
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=kernel_sizes[i + 1],
                    stride=kernel_sizes[i + 1],
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        self.prior_fusions = nn.ModuleList()

        for i in range(4):
            shallow = i <= 1
            self.gscs.append(GMPBlock(dims[i], shallow))
            self.stages.append(
                nn.Sequential(
                    *[
                        TDRMambaBlock(
                            dim=dims[i],
                            num_slices=num_slices_list[i],
                            head=heads[i],
                            step=i,
                        )
                        for _ in range(depths[i])
                    ]
                )
            )
            self.prior_fusions.append(ROICatFusionBlock(dims[i], stage_index=i))

        self.out_indices = out_indices
        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            self.add_module(f"norm{i_layer}", layer)
            self.mlps.append(
                MlpChannel(dims[i_layer], 2 * dims[i_layer], shallow=(i_layer <= 1))
            )

    def forward_features(
        self,
        x,
        hwa_priors=None,
        hwa_prior_quality_maps=None,
        hwa_prior_quality_scores=None,
        return_debug: bool = False,
    ):
        feature_out = []
        prior_gates = []
        prior_alphas = []
        prior_guidance_scales = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            if hwa_priors is not None:
                prior_item = hwa_priors[i]
                if hwa_prior_quality_maps is not None or hwa_prior_quality_scores is not None:
                    prior_item = {
                        "feat": prior_item,
                        "quality_map": None if hwa_prior_quality_maps is None else hwa_prior_quality_maps[i],
                        "quality_score": None if hwa_prior_quality_scores is None else hwa_prior_quality_scores[i],
                    }
                if return_debug:
                    x, gate_map, alpha, guidance_scale = self.prior_fusions[i](
                        x, prior_item, return_debug=True
                    )
                    prior_gates.append(gate_map)
                    prior_alphas.append(alpha)
                    prior_guidance_scales.append(guidance_scale)
                else:
                    x = self.prior_fusions[i](x, prior_item, return_debug=False)
            x = self.gscs[i](x)
            x = self.stages[i](x)
            feature_out.append(x)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x = norm_layer(x)
                x = self.mlps[i](x)

        if not return_debug:
            return x, feature_out

        return x, feature_out, {
            "encoder_prior_gates": prior_gates,
            "encoder_prior_alphas": prior_alphas,
            "encoder_prior_guidance_scales": prior_guidance_scales,
        }

    def forward(
        self,
        x,
        hwa_priors=None,
        hwa_prior_quality_maps=None,
        hwa_prior_quality_scores=None,
        return_debug: bool = False,
    ):
        return self.forward_features(
            x,
            hwa_priors,
            hwa_prior_quality_maps=hwa_prior_quality_maps,
            hwa_prior_quality_scores=hwa_prior_quality_scores,
            return_debug=return_debug,
        )


class TransposedConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, head, r):
        super().__init__()
        self.transposed1 = nn.ConvTranspose3d(dim_in, dim_out, kernel_size=r, stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)
        self.transposed2 = nn.ConvTranspose3d(
            dim_out * 2, dim_out, kernel_size=1, stride=1
        )

    def forward(self, x, feature):
        x = self.transposed1(x)
        x = torch.cat((x, feature), dim=1)
        x = self.transposed2(x)
        x = self.norm(x)
        return x

class DetectorConvNormAct3d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (1, 1, 1),
        groups: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DetectorResidualBlock3d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block1 = DetectorConvNormAct3d(channels, channels)
        self.block2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block2(self.block1(x)))


class StageROILocalEncoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        hidden_ch = max(out_ch // 2, 24)
        self.block = nn.Sequential(
            DetectorConvNormAct3d(in_ch, hidden_ch),
            DetectorResidualBlock3d(hidden_ch),
            nn.Conv3d(hidden_ch, out_ch, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DetectorZOnlyMamba(nn.Module):
    """
    Keep detector context modeling along the Z axis only.
    Input / output shape: (B, C, H, W, Z)
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        nslices: int = 8,
    ):
        super().__init__()
        self.dim = int(dim)
        self.expand = int(expand)

        self.norm = nn.LayerNorm(self.dim)
        self.mamba = Mamba(
            d_model=self.dim,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=self.expand,
            bimamba_type="v3",
            nslices=int(nslices),
        )

        self.inner_dim = int(getattr(self.mamba, "d_inner", self.dim * self.expand))
        self.proj_dim_fwd = nn.Conv3d(self.dim, self.dim, kernel_size=1, bias=True)
        self.proj_dim_bwd = nn.Conv3d(self.dim, self.dim, kernel_size=1, bias=True)
        self.proj_dim_slc = nn.Conv3d(self.dim, self.dim, kernel_size=1, bias=True)
        self.proj_inner_fwd = nn.Conv3d(self.inner_dim, self.dim, kernel_size=1, bias=True)
        self.proj_inner_bwd = nn.Conv3d(self.inner_dim, self.dim, kernel_size=1, bias=True)
        self.proj_inner_slc = nn.Conv3d(self.inner_dim, self.dim, kernel_size=1, bias=True)

    def _reshape_back(self, t: torch.Tensor, b: int, h: int, w: int, z: int):
        if t.dim() != 3:
            raise RuntimeError(f"Unexpected detector mamba output shape: {tuple(t.shape)}")

        n = b * h * w
        if t.shape[0] != n:
            raise RuntimeError(
                f"Unexpected token count in detector mamba output: {t.shape[0]} vs {n}"
            )

        if t.shape[1] == z:
            y = t
        elif t.shape[2] == z:
            y = t.transpose(1, 2).contiguous()
        else:
            raise RuntimeError(
                f"Cannot infer detector mamba layout from {tuple(t.shape)} with z={z}"
            )

        c_in = int(y.shape[-1])
        y = y.reshape(b, h, w, z, c_in).permute(0, 4, 1, 2, 3).contiguous()
        return y, c_in

    def forward(self, x):
        b, c, h, w, z = x.shape
        if c != self.dim:
            raise RuntimeError(f"Expected detector context channels={self.dim}, got {c}")

        x_seq = x.permute(0, 2, 3, 4, 1).reshape(-1, z, c).contiguous()
        x_seq = self.norm(x_seq)

        _, fwd, bwd, slc = self.mamba(x_seq)

        fwd_3d, fwd_c = self._reshape_back(fwd, b, h, w, z)
        bwd_3d, bwd_c = self._reshape_back(bwd, b, h, w, z)
        slc_3d, slc_c = self._reshape_back(slc, b, h, w, z)

        if fwd_c == self.inner_dim:
            fwd_3d = self.proj_inner_fwd(fwd_3d)
        else:
            fwd_3d = self.proj_dim_fwd(fwd_3d)

        if bwd_c == self.inner_dim:
            bwd_3d = self.proj_inner_bwd(bwd_3d)
        else:
            bwd_3d = self.proj_dim_bwd(bwd_3d)

        if slc_c == self.inner_dim:
            slc_3d = self.proj_inner_slc(slc_3d)
        else:
            slc_3d = self.proj_dim_slc(slc_3d)

        return {
            "core": 0.5 * (fwd_3d + bwd_3d),
            "rim": slc_3d,
            "unc": torch.abs(fwd_3d - bwd_3d),
        }


class HWABlockV2(nn.Module):
    """
    Simplified linear HWA:
        1) Center Anchoring
        2) Multi-scale ROI Sampling
        3) Hierarchical Window Prior Generator
    """

    def __init__(
        self,
        in_chans=3,
        kernel_sizes=[4, 2, 2, 2],
        dims=[48, 96, 192, 384],
        det_channels=24,
        softargmax_beta=12.0,
        sigma_min=2.0,
        sigma_max_ratio=0.20,
        local_enhance=1.0,
        evidence_temperature=1.0,
        peak_kernel_size=5,
        peak_gate_gain=10.0,
        peak_gate_bias=0.35,
        center_refine_sigma_scale=0.06,
        anchor_topk=8,
        anchor_softmax_beta=8.0,
        center_offset_limit_ratio=0.10,
        offset_refine_blend=0.55,
        stage_roi_full_sizes=((36, 36, 18), (44, 44, 22), (52, 52, 26), (60, 60, 30)),
        stage_roi_sample_sizes=((24, 24, 12), (20, 20, 10), (16, 16, 8), (12, 12, 6)),
        stage_center_blends=(0.85, 0.70, 0.55, 0.40),
        roi_adapt_spread_lambdas=(2.0, 2.0, 2.0),
        roi_adapt_base_ref_ratio=0.12,
        roi_adapt_min_scale=0.85,
        roi_adapt_max_scale=1.25,
        roi_adapt_peak_ratio=0.55,
        stage_roi_conf_scales=(0.30, 0.22, 0.14, 0.08),
        stage_roi_mask_sharpness=(6.0, 5.5, 5.0, 4.5),
        stage_scope_factors=(0.70, 0.95, 1.10, 1.25),
        stage_center_strengths=(1.45, 1.15, 0.85, 0.60),
        stage_scope_strengths=(0.30, 0.42, 0.52, 0.60),
        stage_raw_strengths=(0.10, 0.14, 0.18, 0.22),
        stage_refined_gate_scales=(1.20, 1.05, 0.95, 0.90),
        stage_coarse_gate_scales=(1.70, 1.45, 1.25, 1.10),
        stage_context_gate_scales=(2.10, 1.85, 1.60, 1.35),
        stage_gate_floors=(0.38, 0.32, 0.26, 0.22),
        stage_gate_sharpness=(5.0, 5.5, 6.0, 6.5),
        stage_conf_expands=(1.45, 1.10, 0.80, 0.55),
    ):
        super().__init__()
        self.in_chans = int(in_chans)
        self.kernel_sizes = kernel_sizes
        self.dims = dims
        self.det_channels = int(det_channels)
        self.softargmax_beta = float(softargmax_beta)
        self.sigma_min = float(sigma_min)
        self.evidence_temperature = float(evidence_temperature)
        self.peak_kernel_size = max(3, int(peak_kernel_size))
        if self.peak_kernel_size % 2 == 0:
            self.peak_kernel_size += 1
        self.peak_gate_gain = float(peak_gate_gain)
        self.peak_gate_bias = float(peak_gate_bias)
        self.anchor_topk = max(1, int(anchor_topk))
        self.anchor_softmax_beta = float(anchor_softmax_beta)
        self.stage_roi_full_sizes = tuple(
            tuple(int(v) for v in size_xyz) for size_xyz in stage_roi_full_sizes
        )
        self.stage_roi_sample_sizes = tuple(
            tuple(int(v) for v in size_xyz) for size_xyz in stage_roi_sample_sizes
        )
        self.stage_center_blends = tuple(float(v) for v in stage_center_blends)
        self.base_roi_full_size = self.stage_roi_full_sizes[0]
        self.roi_adapt_spread_lambdas = tuple(float(v) for v in roi_adapt_spread_lambdas)
        self.roi_adapt_base_ref_ratio = float(roi_adapt_base_ref_ratio)
        self.roi_adapt_min_scale = float(roi_adapt_min_scale)
        self.roi_adapt_max_scale = float(roi_adapt_max_scale)
        self.roi_adapt_peak_ratio = float(roi_adapt_peak_ratio)
        self.roi_adapt_neighborhood_size = tuple(int(v) for v in self.base_roi_full_size)
        base_roi_tensor = torch.tensor(self.base_roi_full_size, dtype=torch.float32)
        self.stage_roi_scale_factors = tuple(
            tuple(
                float(cur_dim) / float(max(base_dim, 1.0))
                for cur_dim, base_dim in zip(stage_size, base_roi_tensor.tolist())
            )
            for stage_size in self.stage_roi_full_sizes
        )

        det_base = max(16, self.det_channels)
        self.det_feat_dim = int(max(64, self.det_channels * 4, dims[0]))

        cumulative = []
        scale = 1
        for k in kernel_sizes:
            scale *= k
            cumulative.append(scale)
        self.stage_windows = cumulative

        self.detector_stem = nn.Sequential(
            DetectorConvNormAct3d(self.in_chans, det_base),
            DetectorResidualBlock3d(det_base),
        )
        self.detector_down1 = nn.Sequential(
            DetectorConvNormAct3d(
                det_base,
                det_base * 2,
                stride=(2, 2, 2),
            ),
            DetectorResidualBlock3d(det_base * 2),
        )
        self.detector_down2 = nn.Sequential(
            DetectorConvNormAct3d(
                det_base * 2,
                self.det_feat_dim,
                stride=(2, 2, 2),
            ),
            DetectorResidualBlock3d(self.det_feat_dim),
        )
        self.modal_shared_proj = nn.Sequential(
            nn.Conv3d(self.det_feat_dim, self.det_feat_dim, kernel_size=1, bias=False),
            nn.InstanceNorm3d(self.det_feat_dim),
            nn.GELU(),
        )
        self.detector_context = DetectorZOnlyMamba(
            dim=self.det_feat_dim,
            d_state=16,
            d_conv=4,
            expand=2,
            nslices=8,
        )

        self.center_head = nn.Conv3d(self.det_feat_dim, 1, kernel_size=1, bias=True)
        self.core_head = nn.Conv3d(self.det_feat_dim, 1, kernel_size=1, bias=True)
        self.excl_head = nn.Conv3d(self.det_feat_dim, 1, kernel_size=1, bias=True)

        self.stage_roi_encoders = nn.ModuleList(
            [StageROILocalEncoder(self.in_chans, d) for d in dims]
        )
        self.input_pool_branches = nn.ModuleList(
            [
                nn.Conv3d(1, 1, kernel_size=k, stride=k, bias=False)
                for k in (1, 2, 4, 8)
            ]
        )
        self.input_agg_proj = nn.Conv3d(4, self.in_chans, kernel_size=3, padding=1, bias=True)
        self.input_channel_agg_proj = nn.Conv3d(
            4 * self.in_chans,
            self.in_chans,
            kernel_size=3,
            padding=1,
            groups=self.in_chans,
            bias=True,
        )
        self.input_modal_weights = nn.Parameter(torch.ones(self.in_chans))
        self.input_cross_mix_logit = nn.Parameter(torch.tensor(-8.00, dtype=torch.float32))
        self.input_gate_boost = nn.Parameter(torch.tensor(8.0, dtype=torch.float32))
        self.input_agg_gain = nn.Parameter(torch.tensor(0.28, dtype=torch.float32))
        self.input_detail_gain = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.output_logit_gain = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.input_roi_modulation = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        with torch.no_grad():
            for branch in self.input_pool_branches:
                branch.weight.fill_(1.0 / float(branch.weight.numel()))
            nn.init.zeros_(self.input_agg_proj.weight)
            nn.init.zeros_(self.input_agg_proj.bias)
            nn.init.zeros_(self.input_channel_agg_proj.weight)
            nn.init.zeros_(self.input_channel_agg_proj.bias)
            center = self.input_channel_agg_proj.kernel_size[0] // 2
            init_scale_weights = (0.70, 0.18, 0.08, 0.04)
            for branch_idx, scale_weight in enumerate(init_scale_weights):
                self.input_channel_agg_proj.weight[
                    :, branch_idx, center, center, center
                ] = float(scale_weight)
        self.last_prior_quality_maps = None
        self.last_prior_quality_scores = None
        self.last_stage_roi_centers = None
        self.last_stage_roi_sizes_full = None
        self.last_input_enhance_map = None
        self.last_fused_support_map = None
        self.last_input_aggregate = None
        self.last_input_prior_reliability = None
        self.last_logit_bias_map = None
        self.input_enhance_gain = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _soft_argmax_3d_multi(self, logits: torch.Tensor) -> torch.Tensor:
        b, m, h, w, z = logits.shape
        flat = logits.view(b, m, -1)
        prob = torch.softmax(self.softargmax_beta * flat, dim=-1).view(b, m, h, w, z)

        xs = torch.linspace(0, h - 1, h, device=logits.device, dtype=logits.dtype).view(1, 1, h, 1, 1)
        ys = torch.linspace(0, w - 1, w, device=logits.device, dtype=logits.dtype).view(1, 1, 1, w, 1)
        zs = torch.linspace(0, z - 1, z, device=logits.device, dtype=logits.dtype).view(1, 1, 1, 1, z)

        cx = (prob * xs).sum(dim=(2, 3, 4))
        cy = (prob * ys).sum(dim=(2, 3, 4))
        cz = (prob * zs).sum(dim=(2, 3, 4))
        return torch.stack([cx, cy, cz], dim=-1)

    def _topk_anchor_readout_multi(self, logits: torch.Tensor):
        b, m, h, w, z = logits.shape
        flat = logits.reshape(b, m, -1)
        k = min(self.anchor_topk, flat.shape[-1])
        vals, idx = torch.topk(flat, k=k, dim=-1)
        weights = torch.softmax(self.anchor_softmax_beta * vals, dim=-1)

        idx_x = torch.div(idx, w * z, rounding_mode="floor")
        idx_y = torch.div(idx, z, rounding_mode="floor") % w
        idx_z = idx % z
        coords = torch.stack(
            [
                idx_x.to(dtype=logits.dtype),
                idx_y.to(dtype=logits.dtype),
                idx_z.to(dtype=logits.dtype),
            ],
            dim=-1,
        )
        center = (weights.unsqueeze(-1) * coords).sum(dim=-2)

        sparse_flat = flat.new_zeros(flat.shape)
        sparse_flat.scatter_(-1, idx, weights)
        sparse_prob = sparse_flat.reshape(b, m, h, w, z)
        return center, sparse_prob

    def _peak_concentrate(self, prob: torch.Tensor) -> torch.Tensor:
        k = self.peak_kernel_size
        pad = k // 2
        local_avg = F.avg_pool3d(prob, kernel_size=k, stride=1, padding=pad)
        local_max = F.max_pool3d(prob, kernel_size=k, stride=1, padding=pad)
        local_denom = (local_max - local_avg).clamp_min(1e-4)
        peakness = (prob - local_avg) / local_denom
        peak_gate = torch.sigmoid(self.peak_gate_gain * (peakness - self.peak_gate_bias))
        concentrated = prob * (0.15 + 0.85 * peak_gate)
        return concentrated.clamp(1e-4, 1.0 - 1e-4)

    def _map_coords_to_fullres(
        self,
        coords_det: torch.Tensor,
        det_size: Tuple[int, int, int],
        full_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        scale = coords_det.new_tensor(
            [
                float(full_size[0]) / float(max(det_size[0], 1)),
                float(full_size[1]) / float(max(det_size[1], 1)),
                float(full_size[2]) / float(max(det_size[2], 1)),
            ]
        ).view(1, 1, 3)
        coords_full = coords_det * scale
        cx = coords_full[..., 0].clamp(0.0, float(full_size[0] - 1))
        cy = coords_full[..., 1].clamp(0.0, float(full_size[1] - 1))
        cz = coords_full[..., 2].clamp(0.0, float(full_size[2] - 1))
        return torch.stack([cx, cy, cz], dim=-1)

    def _map_coords_full_to_stage(
        self,
        coords_full: torch.Tensor,
        full_size: Tuple[int, int, int],
        stage_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        scale = coords_full.new_tensor(
            [
                float(stage_size[0]) / float(max(full_size[0], 1)),
                float(stage_size[1]) / float(max(full_size[1], 1)),
                float(stage_size[2]) / float(max(full_size[2], 1)),
            ]
        ).view(1, 3)
        coords_stage = coords_full * scale
        cx = coords_stage[:, 0].clamp(0.0, float(stage_size[0] - 1))
        cy = coords_stage[:, 1].clamp(0.0, float(stage_size[1] - 1))
        cz = coords_stage[:, 2].clamp(0.0, float(stage_size[2] - 1))
        return torch.stack([cx, cy, cz], dim=1)

    def _fixed_stage_roi_size_full(
        self,
        base_size_xyz: Tuple[int, int, int],
        full_size: Tuple[int, int, int],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        roi_size = ref.new_tensor(base_size_xyz).view(1, 3)
        min_size = ref.new_tensor([8.0, 8.0, 4.0]).view(1, 3)
        max_size = ref.new_tensor(full_size).view(1, 3)
        return roi_size.clamp(min=min_size, max=max_size)

    def _estimate_adaptive_base_roi_size_full(
        self,
        response_full: torch.Tensor,
        center_full: torch.Tensor,
        full_size: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w, z = response_full.shape
        base_size = response_full.new_tensor(self.base_roi_full_size).view(1, 3)
        neigh_size = response_full.new_tensor(self.roi_adapt_neighborhood_size).view(1, 3)
        spread_lambda = response_full.new_tensor(self.roi_adapt_spread_lambdas).view(1, 3)
        sigma_ref = self.roi_adapt_base_ref_ratio * base_size
        min_size = torch.maximum(
            base_size * self.roi_adapt_min_scale,
            response_full.new_tensor([8.0, 8.0, 4.0]).view(1, 3),
        )
        max_size = torch.minimum(
            base_size * self.roi_adapt_max_scale,
            response_full.new_tensor(full_size).view(1, 3),
        )

        adaptive_sizes = []
        spread_sigmas = []
        for bi in range(b):
            cx, cy, cz = [float(v) for v in center_full[bi].detach().cpu().tolist()]
            nx, ny, nz = [int(v) for v in neigh_size[0].detach().cpu().tolist()]
            x1, x2 = self._box_bounds(cx, nx, h)
            y1, y2 = self._box_bounds(cy, ny, w)
            z1, z2 = self._box_bounds(cz, nz, z)

            local_resp = response_full[bi, 0, x1:x2, y1:y2, z1:z2].clamp_min(0.0)
            if local_resp.numel() > 0:
                peak_thr = self.roi_adapt_peak_ratio * local_resp.max()
                local_resp = torch.where(
                    local_resp >= peak_thr,
                    local_resp,
                    torch.zeros_like(local_resp),
                )
            local_mass = local_resp.sum()
            if float(local_mass.item()) <= 1e-6:
                sigma = sigma_ref[0]
            else:
                weights = local_resp / local_mass.clamp_min(1e-6)
                xs = torch.arange(x1, x2, device=response_full.device, dtype=response_full.dtype).view(-1, 1, 1)
                ys = torch.arange(y1, y2, device=response_full.device, dtype=response_full.dtype).view(1, -1, 1)
                zs = torch.arange(z1, z2, device=response_full.device, dtype=response_full.dtype).view(1, 1, -1)
                var_x = (weights * (xs - center_full[bi, 0]) ** 2).sum()
                var_y = (weights * (ys - center_full[bi, 1]) ** 2).sum()
                var_z = (weights * (zs - center_full[bi, 2]) ** 2).sum()
                sigma = torch.stack(
                    [
                        torch.sqrt(var_x.clamp_min(1e-6)),
                        torch.sqrt(var_y.clamp_min(1e-6)),
                        torch.sqrt(var_z.clamp_min(1e-6)),
                    ],
                    dim=0,
                )

            roi_size = (base_size[0] + spread_lambda[0] * (sigma - sigma_ref[0])).clamp(
                min=min_size[0], max=max_size[0]
            )
            adaptive_sizes.append(roi_size)
            spread_sigmas.append(sigma)

        return torch.stack(adaptive_sizes, dim=0), torch.stack(spread_sigmas, dim=0)

    def _detector_prior_quality_score(
        self,
        response_full: torch.Tensor,
        center_topk_full: torch.Tensor,
        center_global_full: torch.Tensor,
        spread_sigma_full: torch.Tensor,
        base_roi_size_full: torch.Tensor,
    ) -> torch.Tensor:
        b = response_full.shape[0]
        flat = response_full.float().flatten(2)
        k = min(64, flat.shape[-1])
        top_vals = torch.topk(flat, k=k, dim=-1).values
        top_mean = top_vals.mean(dim=-1)
        resp_mean = flat.mean(dim=-1).clamp_min(1e-4)
        resp_std = flat.std(dim=-1).clamp_min(1e-4)

        peak_contrast = ((top_mean - resp_mean) / resp_std).clamp(0.0, 8.0)
        peak_quality = torch.sigmoid(0.75 * (peak_contrast - 1.0))

        focus_ratio = (top_mean / resp_mean).clamp(1.0, 6.0)
        focus_quality = ((focus_ratio - 1.0) / 5.0).clamp(0.0, 1.0)

        half_roi = (0.5 * base_roi_size_full.float()).clamp_min(1.0)
        center_dist = torch.linalg.vector_norm(
            (center_topk_full.float() - center_global_full.float()) / half_roi,
            dim=1,
            keepdim=True,
        )
        center_quality = torch.exp(-center_dist).clamp(0.0, 1.0)

        spread_ratio = (
            spread_sigma_full.float() / half_roi
        ).mean(dim=1, keepdim=True)
        spread_quality = torch.exp(-F.relu(spread_ratio - 0.9)).clamp(0.0, 1.0)

        quality_core = (
            0.45 * peak_quality
            + 0.35 * focus_quality
            + 0.20 * center_quality
        )
        quality = 0.15 + 0.85 * quality_core * spread_quality
        return quality.view(b, 1, 1, 1, 1).to(dtype=response_full.dtype)

    @staticmethod
    def _normalize_response_map(response: torch.Tensor) -> torch.Tensor:
        response_f = response.float()
        r_min = response_f.amin(dim=(2, 3, 4), keepdim=True)
        r_max = response_f.amax(dim=(2, 3, 4), keepdim=True)
        response_norm = (response_f - r_min) / (r_max - r_min).clamp_min(1e-4)
        return response_norm.to(dtype=response.dtype)

    def _stage_adaptive_roi_size_full(
        self,
        base_roi_size_full: torch.Tensor,
        stage_index: int,
        full_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        scale = base_roi_size_full.new_tensor(self.stage_roi_scale_factors[stage_index]).view(1, 3)
        roi_size = base_roi_size_full * scale
        min_size = roi_size.new_tensor([8.0, 8.0, 4.0]).view(1, 3)
        max_size = roi_size.new_tensor(full_size).view(1, 3)
        return roi_size.clamp(min=min_size, max=max_size)

    def _sample_roi_volume(
        self,
        volume: torch.Tensor,
        center_full: torch.Tensor,
        roi_size_full: torch.Tensor,
        out_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        b, _, h, w, z = volume.shape
        oh, ow, oz = [max(1, int(v)) for v in out_size]
        dtype = volume.dtype
        device = volume.device

        span_h = torch.linspace(-1.0, 1.0, oh, device=device, dtype=dtype).view(1, oh, 1, 1)
        span_w = torch.linspace(-1.0, 1.0, ow, device=device, dtype=dtype).view(1, 1, ow, 1)
        span_z = torch.linspace(-1.0, 1.0, oz, device=device, dtype=dtype).view(1, 1, 1, oz)

        half_h = 0.5 * roi_size_full[:, 0].view(b, 1, 1, 1)
        half_w = 0.5 * roi_size_full[:, 1].view(b, 1, 1, 1)
        half_z = 0.5 * roi_size_full[:, 2].view(b, 1, 1, 1)

        coord_h = center_full[:, 0].view(b, 1, 1, 1) + span_h * half_h
        coord_w = center_full[:, 1].view(b, 1, 1, 1) + span_w * half_w
        coord_z = center_full[:, 2].view(b, 1, 1, 1) + span_z * half_z

        denom_h = max(h - 1, 1)
        denom_w = max(w - 1, 1)
        denom_z = max(z - 1, 1)
        grid_h = 2.0 * coord_h / float(denom_h) - 1.0
        grid_w = 2.0 * coord_w / float(denom_w) - 1.0
        grid_z = 2.0 * coord_z / float(denom_z) - 1.0
        grid = torch.stack(
            [
                grid_z.expand(-1, oh, ow, oz),
                grid_w.expand(-1, oh, ow, oz),
                grid_h.expand(-1, oh, ow, oz),
            ],
            dim=-1,
        )
        return F.grid_sample(
            volume,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def _stage_box_size(
        self,
        roi_size_full: torch.Tensor,
        full_size: Tuple[int, int, int],
        stage_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        scale = roi_size_full.new_tensor(
            [
                float(stage_size[0]) / float(max(full_size[0], 1)),
                float(stage_size[1]) / float(max(full_size[1], 1)),
                float(stage_size[2]) / float(max(full_size[2], 1)),
            ]
        ).view(1, 3)
        box = torch.round(roi_size_full * scale).long()
        min_box = box.new_tensor([2, 2, 1]).view(1, 3)
        max_box = box.new_tensor(stage_size).view(1, 3)
        return box.clamp(min=min_box, max=max_box)

    def _box_bounds(self, center: float, size: int, limit: int) -> Tuple[int, int]:
        size = max(1, min(int(size), int(limit)))
        center = int(round(float(center)))
        start = center - size // 2
        start = max(0, min(start, int(limit) - size))
        end = start + size
        return int(start), int(end)

    def _paste_local_tensor(
        self,
        local_tensor: torch.Tensor,
        center_stage: torch.Tensor,
        box_size_stage: torch.Tensor,
        stage_size: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, _, _, _ = local_tensor.shape
        hs, ws, zs = [int(v) for v in stage_size]
        canvas = local_tensor.new_zeros(b, c, hs, ws, zs)
        mask = local_tensor.new_zeros(b, 1, hs, ws, zs)

        for bi in range(b):
            bx, by, bz = [int(v) for v in box_size_stage[bi].detach().cpu().tolist()]
            x1, x2 = self._box_bounds(center_stage[bi, 0].item(), bx, hs)
            y1, y2 = self._box_bounds(center_stage[bi, 1].item(), by, ws)
            z1, z2 = self._box_bounds(center_stage[bi, 2].item(), bz, zs)

            feat_b = F.interpolate(
                local_tensor[bi : bi + 1],
                size=(x2 - x1, y2 - y1, z2 - z1),
                mode="trilinear",
                align_corners=False,
            )
            local_mask = self._soft_roi_window(
                (x2 - x1, y2 - y1, z2 - z1),
                device=local_tensor.device,
                dtype=local_tensor.dtype,
            )
            canvas[bi : bi + 1, :, x1:x2, y1:y2, z1:z2] = feat_b * local_mask
            mask[bi : bi + 1, :, x1:x2, y1:y2, z1:z2] = local_mask

        return canvas, mask

    def _soft_roi_window(
        self,
        size_xyz: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        weights = []
        for size in size_xyz:
            if int(size) <= 1:
                weights.append(torch.ones(int(size), device=device, dtype=dtype))
                continue
            coord = torch.linspace(-1.0, 1.0, int(size), device=device, dtype=dtype)
            raised_cos = 0.5 * (1.0 + torch.cos(math.pi * coord.abs()))
            weights.append(0.25 + 0.75 * raised_cos)
        return (
            weights[0].view(1, 1, -1, 1, 1)
            * weights[1].view(1, 1, 1, -1, 1)
            * weights[2].view(1, 1, 1, 1, -1)
        )

    def _resize(self, x, size):
        return F.interpolate(x, size=size, mode="trilinear", align_corners=False)

    @staticmethod
    def _input_prior_reliability(
        x: torch.Tensor,
        enhance_map: torch.Tensor,
    ) -> torch.Tensor:
        saliency = x.detach().float().abs().mean(dim=1, keepdim=True)
        sal_min = saliency.amin(dim=(2, 3, 4), keepdim=True)
        sal_max = saliency.amax(dim=(2, 3, 4), keepdim=True)
        saliency = (saliency - sal_min) / (sal_max - sal_min).clamp_min(1e-4)
        support = enhance_map.detach().float().clamp(0.0, 1.0)
        mass = support.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)
        roi_saliency = (saliency * support).sum(dim=(2, 3, 4), keepdim=True) / mass
        global_saliency = saliency.mean(dim=(2, 3, 4), keepdim=True)
        reliability = torch.sigmoid(8.0 * (roi_saliency - global_saliency))
        return reliability.clamp(0.0, 1.0).to(dtype=x.dtype)

    def build_input_aggregate(
        self,
        x: torch.Tensor,
        enhance_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, m, h, w, z = x.shape
        modal_weights = torch.softmax(self.input_modal_weights, dim=0).to(
            device=x.device,
            dtype=x.dtype,
        )
        cross_aggregated = x.new_zeros(b, self.in_chans, h, w, z)
        per_modal_branches = []
        for modal_idx in range(m):
            channel = x[:, modal_idx : modal_idx + 1]
            branch_feats = []
            for branch in self.input_pool_branches:
                pooled = branch(channel)
                if pooled.shape[2:] != (h, w, z):
                    pooled = F.interpolate(
                        pooled,
                        size=(h, w, z),
                        mode="trilinear",
                        align_corners=False,
                    )
                branch_feats.append(pooled)
            branch_stack = torch.cat(branch_feats, dim=1)
            per_modal_branches.append(branch_stack)
            cross_aggregated = cross_aggregated + modal_weights[modal_idx] * self.input_agg_proj(
                branch_stack
            )

        channel_aggregated = x.new_zeros(b, self.in_chans, h, w, z)
        if len(per_modal_branches) == self.in_chans:
            channel_input = torch.cat(per_modal_branches, dim=1)
            channel_aggregated = self.input_channel_agg_proj(channel_input)

        # Input aggregation preserves each modality and keeps cross-modal mixing gated.
        cross_mix = torch.sigmoid(
            self.input_cross_mix_logit.to(device=x.device, dtype=x.dtype)
        )
        aggregated = channel_aggregated + cross_mix * cross_aggregated

        if enhance_map is not None:
            if enhance_map.shape[2:] != (h, w, z):
                enhance_map = F.interpolate(
                    enhance_map.float(),
                    size=(h, w, z),
                    mode="trilinear",
                    align_corners=False,
                )
            if enhance_map.shape[1] != 1:
                enhance_map = enhance_map.mean(dim=1, keepdim=True)
            enhance_map = enhance_map.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
            reliability = self._input_prior_reliability(x, enhance_map)
            self.last_input_prior_reliability = reliability

            kernel = tuple(3 if int(size) >= 3 else 1 for size in x.shape[2:])
            padding = tuple(1 if k == 3 else 0 for k in kernel)
            local_mean = F.avg_pool3d(x, kernel_size=kernel, stride=1, padding=padding)
            detail = x - local_mean
            detail_std = detail.detach().float().std(
                dim=(2, 3, 4), keepdim=True, unbiased=False
            ).to(dtype=x.dtype).clamp_min(1.0e-5)
            detail = torch.tanh(detail / (2.0 * detail_std)) * (2.0 * detail_std)
            detail_gain = torch.clamp(
                self.input_detail_gain.to(device=x.device, dtype=x.dtype),
                0.0,
                0.35,
            )
            aggregated = aggregated + detail_gain * reliability * enhance_map * detail

            roi_mod = torch.clamp(
                self.input_roi_modulation.to(device=x.device, dtype=x.dtype),
                0.0,
                0.25,
            )
            spatial_bias = enhance_map - enhance_map.mean(
                dim=(2, 3, 4), keepdim=True
            )
            support = (1.0 + roi_mod * reliability * spatial_bias).clamp(0.75, 1.25)
            aggregated = aggregated * support
        else:
            self.last_input_prior_reliability = None

        self.last_input_aggregate = aggregated
        return aggregated

    def forward(self, x, return_debug: bool = False):
        b, m, h, w, z = x.shape
        if m != self.in_chans:
            raise ValueError(f"Expected input channels {self.in_chans}, got {m}")

        det_feat = self.detector_stem(x)
        det_feat = self.detector_down1(det_feat)
        det_feat = self.detector_down2(det_feat)
        det_feat = self.modal_shared_proj(det_feat)
        ctx_pack = self.detector_context(det_feat)

        det_h, det_w, det_z = det_feat.shape[2:]
        det_size = (det_h, det_w, det_z)
        full_size = (h, w, z)

        ctx_rim = ctx_pack["rim"]
        ctx_core = ctx_pack["core"]
        ctx_unc = ctx_pack["unc"]

        anchor_logits_low = (
            self.center_head(ctx_rim) + self.core_head(ctx_core) - self.excl_head(ctx_unc)
        )
        anchor_prob_low = self._peak_concentrate(
            torch.sigmoid(anchor_logits_low / self.evidence_temperature)
        )
        anchor_logits_focus_low = torch.logit(anchor_prob_low.clamp(1e-4, 1.0 - 1e-4))
        anchor_prob_full = self._resize(anchor_prob_low, full_size).clamp(1e-4, 1.0 - 1.0e-4)

        center_topk_det, anchor_sparse_prob_low = self._topk_anchor_readout_multi(
            anchor_logits_focus_low
        )
        center_global_det = self._soft_argmax_3d_multi(anchor_logits_focus_low)

        center_topk_full = self._map_coords_to_fullres(
            center_topk_det, det_size, full_size
        )[:, 0, :]
        center_global_full = self._map_coords_to_fullres(
            center_global_det, det_size, full_size
        )[:, 0, :]
        adaptive_base_roi_size_full, response_spread_sigma_full = self._estimate_adaptive_base_roi_size_full(
            anchor_prob_full,
            center_topk_full,
            full_size,
        )
        detector_quality_score = self._detector_prior_quality_score(
            anchor_prob_full,
            center_topk_full,
            center_global_full,
            response_spread_sigma_full,
            adaptive_base_roi_size_full,
        )

        # Compatibility aliases used by downstream scripts.
        det_center_each_full = center_topk_full.unsqueeze(1).expand(-1, self.in_chans, -1)
        det_center_coarse_each_full = center_global_full.unsqueeze(1).expand(-1, self.in_chans, -1)
        det_sigma_each_full = response_spread_sigma_full.unsqueeze(1).expand(-1, self.in_chans, -1)
        det_conf_each_full = x.new_ones((b, self.in_chans))

        priors = []
        stage_window_evidence = []
        stage_prior_quality_maps = []
        stage_prior_quality_scores = []
        stage_roi_boxes = []
        stage_roi_centers = []
        stage_roi_sizes_full = []

        for i, roi_encoder in enumerate(self.stage_roi_encoders):
            hs = max(1, h // self.stage_windows[i])
            ws = max(1, w // self.stage_windows[i])
            zs = max(1, z // self.stage_windows[i])
            stage_size = (hs, ws, zs)

            blend = self.stage_center_blends[i]
            stage_center_full = (
                blend * center_topk_full + (1.0 - blend) * center_global_full
            )
            center_stage = self._map_coords_full_to_stage(
                stage_center_full, full_size, stage_size
            )
            roi_size_full = self._stage_adaptive_roi_size_full(
                adaptive_base_roi_size_full, i, full_size
            )
            sample_size = self.stage_roi_sample_sizes[i]
            box_size_stage = self._stage_box_size(
                roi_size_full, full_size, stage_size
            )

            image_roi = self._sample_roi_volume(
                x, stage_center_full, roi_size_full, sample_size
            )
            local_feat = roi_encoder(image_roi)
            prior_feat, hard_box_mask = self._paste_local_tensor(
                local_feat, center_stage, box_size_stage, stage_size
            )
            stage_response = self._resize(anchor_prob_full, stage_size)
            stage_response = self._normalize_response_map(stage_response)
            quality_map = hard_box_mask * (0.45 + 0.55 * stage_response)
            quality_score = detector_quality_score.clamp(0.05, 1.0)

            priors.append(prior_feat)
            stage_prior_quality_maps.append(quality_map)
            stage_prior_quality_scores.append(quality_score)
            stage_window_evidence.append(
                quality_map.expand(-1, self.in_chans, -1, -1, -1)
            )
            stage_roi_boxes.append(box_size_stage)
            stage_roi_centers.append(center_stage)
            stage_roi_sizes_full.append(roi_size_full)

        self.last_prior_quality_maps = stage_prior_quality_maps
        self.last_prior_quality_scores = stage_prior_quality_scores
        self.last_stage_roi_centers = stage_roi_centers
        self.last_stage_roi_sizes_full = stage_roi_sizes_full
        if stage_prior_quality_maps:
            input_quality = stage_prior_quality_scores[0].clamp(0.35, 1.0)
            stage0_support_full = self._resize(stage_prior_quality_maps[0], full_size)
            stage0_support_full = stage0_support_full.clamp(0.0, 1.0)
            fused_support_map = (stage0_support_full * input_quality).clamp(1e-4, 1.0 - 1e-4)
            input_enhance_map = stage0_support_full
            input_enhance_map = (
                torch.sqrt(input_enhance_map.clamp(0.0, 1.0).clamp_min(1e-6))
                * torch.sqrt(input_quality)
            ).clamp(0.0, 1.0)
            input_reliability = self._input_prior_reliability(x, input_enhance_map)
            input_enhance_map_for_raw = input_enhance_map * input_reliability
            self.last_input_prior_reliability = input_reliability
            self.last_input_enhance_map = input_enhance_map_for_raw
            self.last_fused_support_map = fused_support_map
            self.build_input_aggregate(x, input_enhance_map)
        else:
            self.last_input_enhance_map = None
            self.last_fused_support_map = None
            self.last_input_aggregate = None
            self.last_input_prior_reliability = None

        if not return_debug:
            return priors

        raw_evidence_prob_each = anchor_prob_full.expand(-1, self.in_chans, -1, -1, -1)
        fused_support_map = self.last_fused_support_map
        if fused_support_map is None:
            fused_support_map = anchor_prob_full
        seed_evidence_full = (0.70 * anchor_prob_full + 0.30 * fused_support_map).clamp(
            1e-4, 1.0 - 1e-4
        )
        seed_evidence_prob_each = seed_evidence_full.expand(-1, self.in_chans, -1, -1, -1)

        debug_dict = {
            "det_anchor_logits_full": self._resize(anchor_logits_low, full_size),
            "det_anchor_prob_full": anchor_prob_full,
            "det_raw_evidence_prob_each": raw_evidence_prob_each,
            "det_seed_evidence_prob_each": seed_evidence_prob_each,
            "det_fused_seed_prob_full": fused_support_map,
            "det_anchor_sparse_prob_full": self._resize(anchor_sparse_prob_low, full_size).clamp(1e-4, 1.0 - 1.0e-4),
            "det_center_topk_full": center_topk_full,
            "det_center_global_full": center_global_full,
            "det_center_coarse_each_full": det_center_coarse_each_full,
            "det_center_each_full": det_center_each_full,
            "det_sigma_each_full": det_sigma_each_full,
            "det_conf_each_full": det_conf_each_full,
            "det_fused_center_full": center_topk_full,
            "det_fused_center_coarse_full": center_global_full,
            "det_fused_sigma_full": det_sigma_each_full.mean(dim=1),
            "det_fused_conf_full": det_conf_each_full.mean(dim=1, keepdim=True),
            "det_response_spread_sigma_full": response_spread_sigma_full,
            "det_adaptive_base_roi_size_full": adaptive_base_roi_size_full,
            "det_prior_quality_score": detector_quality_score,
            "stage_window_evidence": stage_window_evidence,
            "stage_prior_quality_maps": self.last_prior_quality_maps,
            "stage_prior_quality_scores": self.last_prior_quality_scores,
            "input_enhance_map": self.last_input_enhance_map,
            "fused_support_map": self.last_fused_support_map,
            "input_aggregate": self.last_input_aggregate,
            "input_prior_reliability": self.last_input_prior_reliability,
            "stage_roi_boxes": stage_roi_boxes,
            "stage_roi_centers": stage_roi_centers,
            "stage_roi_sizes_full": stage_roi_sizes_full,
            "hwa_priors": priors,
        }
        return priors, debug_dict


class TODMClassificationHead3D(nn.Module):
    def __init__(
        self,
        in_chs=(48, 96, 192, 384),
        proj_ch=128,
        attn_hidden=256,
        cls_hidden=256,
        dropout=0.2,
        eps=1e-6,
    ):
        super().__init__()
        self.eps = eps

        self.proj = nn.ModuleList(
            [nn.Conv3d(c, proj_ch, kernel_size=1, bias=False) for c in in_chs]
        )
        self.norm = nn.ModuleList(
            [nn.InstanceNorm3d(proj_ch, affine=True) for _ in in_chs]
        )
        self.spatial_attn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(proj_ch, proj_ch // 4, kernel_size=1, bias=True),
                    nn.ReLU(inplace=False),
                    nn.Conv3d(proj_ch // 4, 1, kernel_size=1, bias=True),
                )
                for _ in in_chs
            ]
        )
        self.scale_fc = nn.Sequential(
            nn.Linear(proj_ch * len(in_chs), attn_hidden),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, len(in_chs)),
        )
        self.cls_fc = nn.Sequential(
            nn.Linear(proj_ch, cls_hidden),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, 1),
        )

    def _attn_pool(self, feat, attn):
        wfeat = feat * attn
        num = wfeat.sum(dim=(2, 3, 4))
        den = attn.sum(dim=(2, 3, 4)).clamp_min(self.eps)
        return num / den

    def forward(self, feats):
        assert len(feats) == 4, "Expect 4-scale features."

        vecs = []
        for f, p, n, a in zip(feats, self.proj, self.norm, self.spatial_attn):
            g = n(p(f))
            attn_i = torch.sigmoid(a(g))
            v_i = self._attn_pool(g, attn_i)
            vecs.append(v_i)

        v_cat = torch.cat(vecs, dim=1)
        scale_attn = torch.softmax(self.scale_fc(v_cat), dim=1)
        v_stack = torch.stack(vecs, dim=1)
        v_fused = (scale_attn.unsqueeze(-1) * v_stack).sum(dim=1)

        logit = self.cls_fc(v_fused)
        return logit


class HWAUNETRV2(nn.Module):
    def __init__(
        self,
        in_chans=4,
        out_chans=3,
        hwa_block=(1, 2, 4, 8),
        kernel_sizes=[4, 2, 2, 2],
        depths=[1, 1, 1, 1],
        dims=[48, 96, 192, 384],
        heads=[1, 2, 4, 4],
        hidden_size=768,
        num_slices_list=[64, 32, 16, 8],
        out_indices=[0, 1, 2, 3],
        hwa_det_channels=24,
        hwa_softargmax_beta=12.0,
        hwa_sigma_min=2.0,
        hwa_sigma_max_ratio=0.20,
        hwa_local_enhance=1.0,
        hwa_evidence_temperature=1.0,
        hwa_peak_kernel_size=5,
        hwa_peak_gate_gain=10.0,
        hwa_peak_gate_bias=0.35,
        hwa_center_refine_sigma_scale=0.06,
        hwa_stage_roi_full_sizes=((36, 36, 18), (44, 44, 22), (52, 52, 26), (60, 60, 30)),
        hwa_stage_roi_sample_sizes=((24, 24, 12), (20, 20, 10), (16, 16, 8), (12, 12, 6)),
        hwa_stage_roi_conf_scales=(0.30, 0.22, 0.14, 0.08),
        hwa_stage_roi_mask_sharpness=(6.0, 5.5, 5.0, 4.5),
        hwa_stage_scope_factors=(0.70, 0.95, 1.10, 1.25),
        hwa_stage_center_strengths=(1.45, 1.15, 0.85, 0.60),
        hwa_stage_scope_strengths=(0.30, 0.42, 0.52, 0.60),
        hwa_stage_raw_strengths=(0.10, 0.14, 0.18, 0.22),
        hwa_stage_refined_gate_scales=(1.20, 1.05, 0.95, 0.90),
        hwa_stage_coarse_gate_scales=(1.70, 1.45, 1.25, 1.10),
        hwa_stage_context_gate_scales=(2.10, 1.85, 1.60, 1.35),
        hwa_stage_gate_floors=(0.38, 0.32, 0.26, 0.22),
        hwa_stage_gate_sharpness=(5.0, 5.5, 6.0, 6.5),
        hwa_stage_conf_expands=(1.45, 1.10, 0.80, 0.55),
        use_hwa_prior_in_encoder=True,
        **kwargs,
    ):
        super().__init__()
        hwa_block = kwargs.pop("fussion", hwa_block)
        self.use_hwa_prior_in_encoder = bool(use_hwa_prior_in_encoder)
        self.runtime_hwa_gate_scale = 1.0
        self.hwa_input_enhance_gain_max = 0.12
        self.hwa_input_gate_scale_max = 1.0
        self.hwa_input_agg_gain_max = 0.35
        self.hwa_input_delta_std_clip = 1.5
        self.hwa_output_logit_gain_max = 4.0

        self.hwa_block = HWABlockV2(
            in_chans=in_chans,
            kernel_sizes=kernel_sizes,
            dims=dims,
            det_channels=hwa_det_channels,
            softargmax_beta=hwa_softargmax_beta,
            sigma_min=hwa_sigma_min,
            sigma_max_ratio=hwa_sigma_max_ratio,
            local_enhance=hwa_local_enhance,
            evidence_temperature=hwa_evidence_temperature,
            peak_kernel_size=hwa_peak_kernel_size,
            peak_gate_gain=hwa_peak_gate_gain,
            peak_gate_bias=hwa_peak_gate_bias,
            center_refine_sigma_scale=hwa_center_refine_sigma_scale,
            stage_roi_full_sizes=hwa_stage_roi_full_sizes,
            stage_roi_sample_sizes=hwa_stage_roi_sample_sizes,
            stage_roi_conf_scales=hwa_stage_roi_conf_scales,
            stage_roi_mask_sharpness=hwa_stage_roi_mask_sharpness,
            stage_scope_factors=hwa_stage_scope_factors,
            stage_center_strengths=hwa_stage_center_strengths,
            stage_scope_strengths=hwa_stage_scope_strengths,
            stage_raw_strengths=hwa_stage_raw_strengths,
            stage_refined_gate_scales=hwa_stage_refined_gate_scales,
            stage_coarse_gate_scales=hwa_stage_coarse_gate_scales,
            stage_context_gate_scales=hwa_stage_context_gate_scales,
            stage_gate_floors=hwa_stage_gate_floors,
            stage_gate_sharpness=hwa_stage_gate_sharpness,
            stage_conf_expands=hwa_stage_conf_expands,
        )

        self.Encoder = Encoder(
            in_chans=in_chans,
            kernel_sizes=kernel_sizes,
            depths=depths,
            dims=dims,
            num_slices_list=num_slices_list,
            out_indices=out_indices,
            heads=heads,
        )

        self.hidden_downsample = nn.Conv3d(
            dims[3], hidden_size, kernel_size=2, stride=2
        )
        self.TSconv1 = TransposedConvLayer(
            dim_in=hidden_size, dim_out=dims[3], head=heads[3], r=2
        )
        self.TSconv2 = TransposedConvLayer(
            dim_in=dims[3], dim_out=dims[2], head=heads[2], r=kernel_sizes[3]
        )
        self.TSconv3 = TransposedConvLayer(
            dim_in=dims[2], dim_out=dims[1], head=heads[1], r=kernel_sizes[2]
        )
        self.TSconv4 = TransposedConvLayer(
            dim_in=dims[1], dim_out=dims[0], head=heads[0], r=kernel_sizes[1]
        )

        self.todm_seg_head = nn.ConvTranspose3d(
            dims[0], out_chans, kernel_size=kernel_sizes[0], stride=kernel_sizes[0]
        )
        self.todm_cls_head = TODMClassificationHead3D(
            in_chs=dims[::-1],
            proj_ch=128,
            attn_hidden=256,
            cls_hidden=256,
            dropout=0.2,
        )
    def _apply_hwa_input_enhancement(
        self,
        x: torch.Tensor,
        enhance_map: Optional[torch.Tensor],
        input_aggregate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if enhance_map is None and input_aggregate is None:
            return x
        if enhance_map is not None and enhance_map.shape[2:] != x.shape[2:]:
            enhance_map = F.interpolate(
                enhance_map.float(),
                size=x.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        if enhance_map is not None and enhance_map.shape[1] != 1:
            enhance_map = enhance_map.mean(dim=1, keepdim=True)
        if enhance_map is not None:
            enhance_map = enhance_map.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
        gate_scale = float(max(0.0, min(1.0, getattr(self, "runtime_hwa_gate_scale", 1.0))))
        if gate_scale <= 0.0:
            return x
        gain_source = getattr(self.hwa_block, "input_enhance_gain", None)
        if gain_source is None:
            gain_source = x.new_tensor(0.08)
        enhance_gain_max = float(max(0.0, getattr(self, "hwa_input_enhance_gain_max", 0.12)))
        gain = torch.clamp(
            gain_source.to(device=x.device, dtype=x.dtype),
            0.0,
            enhance_gain_max,
        )
        boost_source = getattr(self.hwa_block, "input_gate_boost", None)
        if boost_source is None:
            boost_source = x.new_tensor(4.0)
        gate_scale_max = float(max(0.0, getattr(self, "hwa_input_gate_scale_max", 1.0)))
        input_gate_scale = torch.clamp(
            gate_scale * boost_source.to(device=x.device, dtype=x.dtype),
            0.0,
            gate_scale_max,
        )
        out = x
        if enhance_map is not None:
            out = out * (1.0 + input_gate_scale * gain * enhance_map)

        if input_aggregate is not None:
            if input_aggregate.shape[2:] != x.shape[2:]:
                input_aggregate = F.interpolate(
                    input_aggregate.float(),
                    size=x.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
            input_aggregate = input_aggregate.to(device=x.device, dtype=x.dtype)
            agg_gain_source = getattr(self.hwa_block, "input_agg_gain", None)
            if agg_gain_source is None:
                agg_gain_source = x.new_tensor(0.12)
            agg_gain_max = float(max(0.0, getattr(self, "hwa_input_agg_gain_max", 0.35)))
            agg_gain = torch.clamp(
                agg_gain_source.to(device=x.device, dtype=x.dtype), 0.0, agg_gain_max
            )
            raw_std = x.std(dim=(2, 3, 4), keepdim=True, unbiased=False).clamp_min(1.0e-5)
            delta = input_aggregate - x
            delta_clip = float(max(1.0e-3, getattr(self, "hwa_input_delta_std_clip", 1.5)))
            delta = torch.tanh(delta / (delta_clip * raw_std)) * (delta_clip * raw_std)
            self.last_input_aggregate_delta_mean = delta.detach().float().abs().mean()
            self.last_input_aggregate_delta_ratio = (
                delta.detach().float().abs().mean()
                / x.detach().float().abs().mean().clamp_min(1.0e-5)
            )
            out = out + input_gate_scale * agg_gain * delta
        else:
            self.last_input_aggregate_delta_mean = None
            self.last_input_aggregate_delta_ratio = None
        return out

    def _apply_hwa_logit_bias(
        self,
        seg_logits: torch.Tensor,
        enhance_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if enhance_map is None:
            self.last_logit_bias_map = None
            return seg_logits

        gate_scale = float(max(0.0, min(1.0, getattr(self, "runtime_hwa_gate_scale", 1.0))))
        if gate_scale <= 0.0:
            self.last_logit_bias_map = None
            return seg_logits

        if enhance_map.shape[2:] != seg_logits.shape[2:]:
            enhance_map = F.interpolate(
                enhance_map.float(),
                size=seg_logits.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        if enhance_map.shape[1] != 1:
            enhance_map = enhance_map.mean(dim=1, keepdim=True)
        enhance_map = enhance_map.to(
            device=seg_logits.device,
            dtype=seg_logits.dtype,
        ).clamp(0.0, 1.0)

        # Convert the soft ROI support into a bounded logit bias.
        support = torch.sqrt(enhance_map.clamp_min(1.0e-6))
        gain_source = getattr(self.hwa_block, "output_logit_gain", None)
        if gain_source is None:
            gain_source = seg_logits.new_tensor(0.25)
        logit_gain_max = float(max(0.0, getattr(self, "hwa_output_logit_gain_max", 4.0)))
        gain = torch.clamp(
            gain_source.to(device=seg_logits.device, dtype=seg_logits.dtype),
            0.0,
            logit_gain_max,
        )
        if float(gain.detach().max().cpu()) <= 0.0:
            self.last_logit_bias_map = None
            return seg_logits
        bias = gate_scale * gain * support
        self.last_logit_bias_map = bias.detach()
        return seg_logits + bias.expand(-1, seg_logits.shape[1], -1, -1, -1)

    def forward(
        self,
        x,
        return_debug: bool = False,
        detector_only: bool = False,
    ):
        if detector_only:
            _, det_debug = self.hwa_block(x, return_debug=True)
            return None, None, det_debug

        if return_debug or self.use_hwa_prior_in_encoder:
            if return_debug:
                priors, det_debug = self.hwa_block(x, return_debug=True)
                prior_quality_maps = det_debug.get("stage_prior_quality_maps", None)
                prior_quality_scores = det_debug.get("stage_prior_quality_scores", None)
                input_enhance_map = det_debug.get("input_enhance_map", None)
                input_aggregate = det_debug.get("input_aggregate", None)
            else:
                priors = self.hwa_block(x, return_debug=False)
                det_debug = None
                prior_quality_maps = getattr(self.hwa_block, "last_prior_quality_maps", None)
                prior_quality_scores = getattr(self.hwa_block, "last_prior_quality_scores", None)
                input_enhance_map = getattr(self.hwa_block, "last_input_enhance_map", None)
                input_aggregate = getattr(self.hwa_block, "last_input_aggregate", None)
        else:
            priors = None
            det_debug = None
            prior_quality_maps = None
            prior_quality_scores = None
            input_enhance_map = None
            input_aggregate = None

        encoder_priors = priors if self.use_hwa_prior_in_encoder else None
        encoder_quality_maps = prior_quality_maps if self.use_hwa_prior_in_encoder else None
        encoder_quality_scores = prior_quality_scores if self.use_hwa_prior_in_encoder else None
        encoder_input = self._apply_hwa_input_enhancement(
            x,
            input_enhance_map if self.use_hwa_prior_in_encoder else None,
            input_aggregate if self.use_hwa_prior_in_encoder else None,
        )

        if return_debug:
            outs, feature_out, encoder_debug = self.Encoder(
                encoder_input,
                encoder_priors,
                hwa_prior_quality_maps=encoder_quality_maps,
                hwa_prior_quality_scores=encoder_quality_scores,
                return_debug=True,
            )
        else:
            outs, feature_out = self.Encoder(
                encoder_input,
                encoder_priors,
                hwa_prior_quality_maps=encoder_quality_maps,
                hwa_prior_quality_scores=encoder_quality_scores,
                return_debug=False,
            )
            encoder_debug = None

        deep_feature = self.hidden_downsample(outs)
        up_feature = []

        seg_x = self.TSconv1(deep_feature, feature_out[-1])
        up_feature.append(seg_x)

        seg_x = self.TSconv2(seg_x, feature_out[-2])
        up_feature.append(seg_x)

        seg_x = self.TSconv3(seg_x, feature_out[-3])
        up_feature.append(seg_x)

        seg_x = self.TSconv4(seg_x, feature_out[-4])
        up_feature.append(seg_x)

        seg_logits = self.todm_seg_head(seg_x)
        seg_logits = self._apply_hwa_logit_bias(
            seg_logits,
            input_enhance_map if self.use_hwa_prior_in_encoder else None,
        )
        cls_logits = self.todm_cls_head(up_feature)

        if return_debug:
            debug_dict = det_debug if det_debug is not None else {}
            if encoder_debug is not None:
                debug_dict.update(encoder_debug)
            debug_dict["hwa_priors"] = priors
            return cls_logits, seg_logits, debug_dict

        return cls_logits, seg_logits

    @property
    def fussion(self):
        return self.hwa_block

    @property
    def SegHead(self):
        return self.todm_seg_head

    @property
    def Class_Decoder(self):
        return self.todm_cls_head
