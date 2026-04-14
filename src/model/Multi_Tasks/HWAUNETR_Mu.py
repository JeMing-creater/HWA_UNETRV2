# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import time
from typing import List, Tuple

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
        return grad_output * (sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i)))


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


class TriDRABlock(nn.Module):
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
        stem = nn.Sequential(nn.Conv3d(in_chans, dims[0], kernel_size=kernel_sizes[0], stride=kernel_sizes[0]))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=kernel_sizes[i + 1], stride=kernel_sizes[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        self.hwa_scales = nn.ParameterList()

        for i in range(4):
            shallow = i <= 1
            self.gscs.append(GMPBlock(dims[i], shallow))
            self.stages.append(
                nn.Sequential(
                    *[
                        TriDRABlock(dim=dims[i], num_slices=num_slices_list[i], head=heads[i], step=i)
                        for _ in range(depths[i])
                    ]
                )
            )
            self.hwa_scales.append(nn.Parameter(torch.tensor(1.0)))

        self.out_indices = out_indices
        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            self.add_module(f'norm{i_layer}', layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer], shallow=(i_layer <= 1)))

    def forward_features(self, x, hwa_priors=None):
        feature_out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            if hwa_priors is not None:
                x = x + self.hwa_scales[i] * hwa_priors[i]
            x = self.gscs[i](x)
            x = self.stages[i](x)
            feature_out.append(x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x = norm_layer(x)
                x = self.mlps[i](x)

        return x, feature_out

    def forward(self, x, hwa_priors=None):
        x, feature_out = self.forward_features(x, hwa_priors)
        return x, feature_out


class TransposedConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, head, r):
        super().__init__()
        self.transposed1 = nn.ConvTranspose3d(dim_in, dim_out, kernel_size=r, stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)
        self.transposed2 = nn.ConvTranspose3d(dim_out * 2, dim_out, kernel_size=1, stride=1)

    def forward(self, x, feature):
        x = self.transposed1(x)
        x = torch.cat((x, feature), dim=1)
        x = self.transposed2(x)
        x = self.norm(x)
        return x


class HierarchicalWindowAggregateBlock(nn.Module):
    """
    Stage-wise HWA block.

    Input remains the raw multi-modal MRI.
    Output is a list of stage-aligned priors, one for each encoder stage.
    Each stage uses spatial modality routing instead of a fixed global modality weight.
    """

    def __init__(self, in_chans=3, kernel_sizes=[4, 2, 2, 2], dims=[48, 96, 192, 384]):
        super().__init__()
        self.in_chans = in_chans
        self.kernel_sizes = kernel_sizes
        self.dims = dims

        cumulative = []
        scale = 1
        for k in kernel_sizes:
            scale *= k
            cumulative.append(scale)
        self.stage_windows = cumulative

        self.window_aggs = nn.ModuleList(
            [
                nn.Conv3d(in_chans, in_chans, kernel_size=s, stride=s, groups=in_chans, bias=True)
                for s in self.stage_windows
            ]
        )
        self.route_convs = nn.ModuleList(
            [nn.Conv3d(in_chans, in_chans, kernel_size=1, stride=1, padding=0, bias=True) for _ in self.stage_windows]
        )
        self.prior_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(1, d, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.InstanceNorm3d(d),
                    nn.GELU(),
                )
                for d in dims
            ]
        )

    def forward(self, x):
        priors = []
        for agg, router, proj in zip(self.window_aggs, self.route_convs, self.prior_projs):
            stage_feat = agg(x)  # (B, M, Hs, Ws, Ds), modality-specific via depthwise conv
            route = torch.softmax(router(stage_feat), dim=1)
            fused = (stage_feat * route).sum(dim=1, keepdim=True)
            priors.append(proj(fused))
        return priors


class ScaleAttnSelfROIClsHead3D(nn.Module):
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

        self.proj = nn.ModuleList([nn.Conv3d(c, proj_ch, kernel_size=1, bias=False) for c in in_chs])
        self.norm = nn.ModuleList([nn.InstanceNorm3d(proj_ch, affine=True) for _ in in_chs])
        self.spatial_attn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(proj_ch, proj_ch // 4, kernel_size=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(proj_ch // 4, 1, kernel_size=1, bias=True),
                )
                for _ in in_chs
            ]
        )
        self.scale_fc = nn.Sequential(
            nn.Linear(proj_ch * len(in_chs), attn_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, len(in_chs)),
        )
        self.cls_fc = nn.Sequential(
            nn.Linear(proj_ch, cls_hidden),
            nn.ReLU(inplace=True),
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
        fussion=[1, 2, 4, 8],
        kernel_sizes=[4, 2, 2, 2],
        depths=[1, 1, 1, 1],
        dims=[48, 96, 192, 384],
        heads=[1, 2, 4, 4],
        hidden_size=768,
        num_slices_list=[64, 32, 16, 8],
        out_indices=[0, 1, 2, 3],
    ):
        super().__init__()
        self.fussion = HierarchicalWindowAggregateBlock(in_chans=in_chans, kernel_sizes=kernel_sizes, dims=dims)
        self.Encoder = Encoder(
            in_chans=in_chans,
            kernel_sizes=kernel_sizes,
            depths=depths,
            dims=dims,
            num_slices_list=num_slices_list,
            out_indices=out_indices,
            heads=heads,
        )

        self.hidden_downsample = nn.Conv3d(dims[3], hidden_size, kernel_size=2, stride=2)
        self.TSconv1 = TransposedConvLayer(dim_in=hidden_size, dim_out=dims[3], head=heads[3], r=2)
        self.TSconv2 = TransposedConvLayer(dim_in=dims[3], dim_out=dims[2], head=heads[2], r=kernel_sizes[3])
        self.TSconv3 = TransposedConvLayer(dim_in=dims[2], dim_out=dims[1], head=heads[1], r=kernel_sizes[2])
        self.TSconv4 = TransposedConvLayer(dim_in=dims[1], dim_out=dims[0], head=heads[0], r=kernel_sizes[1])

        self.SegHead = nn.ConvTranspose3d(dims[0], out_chans, kernel_size=kernel_sizes[0], stride=kernel_sizes[0])
        self.Class_Decoder = ScaleAttnSelfROIClsHead3D(
            in_chs=dims[::-1], proj_ch=128, attn_hidden=256, cls_hidden=256, dropout=0.2
        )

    def forward(self, x):
        priors = self.fussion(x)
        outs, feature_out = self.Encoder(x, priors)

        deep_feature = self.hidden_downsample(outs)
        up_feature = []

        x = self.TSconv1(deep_feature, feature_out[-1])
        up_feature.append(x)

        x = self.TSconv2(x, feature_out[-2])
        up_feature.append(x)

        x = self.TSconv3(x, feature_out[-3])
        up_feature.append(x)

        x = self.TSconv4(x, feature_out[-4])
        up_feature.append(x)

        x = self.SegHead(x)
        c_x = self.Class_Decoder(up_feature)
        # return c_x, x
        return x


def test_weight(model, x):
    for _ in range(0, 3):
        _ = model(x)
    start_time = time.time()
    output = model(x)
    end_time = time.time()
    need_time = end_time - start_time
    from thop import profile

    flops, params = profile(model, inputs=(x,))
    throughout = round(x.shape[0] / (need_time / 1), 3)
    return flops, params, throughout


def Unitconversion(flops, params, throughout):
    print('params : {} M'.format(round(params / (1000**2), 2)))
    print('flop : {} G'.format(round(flops / (1000**3), 2)))
    print('throughout: {} FPS'.format(throughout))


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(size=(2, 3, 128, 128, 64)).to(device)

    model = HWAUNETRV2(
        in_chans=3,
        out_chans=3,
        fussion=[1, 2, 4, 8],
        kernel_sizes=[4, 2, 2, 2],
        depths=[2, 2, 2, 2],
        dims=[48, 96, 192, 384],
        heads=[1, 2, 4, 4],
        hidden_size=768,
        num_slices_list=[64, 32, 16, 8],
        out_indices=[0, 1, 2, 3],
    ).to(device)

    classx, segx = model(x)
    print(classx.shape)
    print(segx.shape)
