from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from batchgenerators.augmentations.utils import pad_nd_image
from scipy.ndimage import gaussian_filter
from typing import Union, Tuple, List

from torch.cuda.amp import autocast
import copy
from torch.nn.init import xavier_uniform_, constant_, normal_


import math
import torch
from torch import nn
import torch.nn.functional as F


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=[64, 64, 64], temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        bs, c, d, h, w = x.shape
        mask = torch.zeros(bs, d, h, w, dtype=torch.bool, device=x.device)
        assert mask is not None
        not_mask = ~mask
        d_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            d_embed = (d_embed - 0.5) / (d_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = (y_embed - 0.5) / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_tx = torch.arange(self.num_pos_feats[0], dtype=torch.float32, device=x.device)
        dim_tx = self.temperature ** (3 * (dim_tx // 3) / self.num_pos_feats[0])

        dim_ty = torch.arange(self.num_pos_feats[1], dtype=torch.float32, device=x.device)
        dim_ty = self.temperature ** (3 * (dim_ty // 3) / self.num_pos_feats[1])

        dim_td = torch.arange(self.num_pos_feats[2], dtype=torch.float32, device=x.device)
        dim_td = self.temperature ** (3 * (dim_td // 3) / self.num_pos_feats[2])

        pos_x = x_embed[:, :, :, :, None] / dim_tx
        pos_y = y_embed[:, :, :, :, None] / dim_ty
        pos_d = d_embed[:, :, :, :, None] / dim_td

        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_d = torch.stack((pos_d[:, :, :, :, 0::2].sin(), pos_d[:, :, :, :, 1::2].cos()), dim=5).flatten(4)

        pos = torch.cat((pos_d, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3)
        return pos


def build_position_encoding(mode, hidden_dim):
    N_steps = hidden_dim // 3
    if (hidden_dim % 3) != 0:
        N_steps = [N_steps, N_steps, N_steps + hidden_dim % 3]
    else:
        N_steps = [N_steps, N_steps, N_steps]

    if mode in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(num_pos_feats=N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {mode}")

    return position_embedding

def ms_deform_attn_core_pytorch_3D(value, value_spatial_shapes, sampling_locations, attention_weights):
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([T_ * H_ * W_ for T_, H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    # sampling_grids = 3 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (T_, H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, T_, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)[:,None,:,:,:]
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_.to(dtype=value_l_.dtype), mode='bilinear', padding_mode='zeros', align_corners=False)[:,:,0]
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()
class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 3)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()*thetas.cos(), thetas.sin()*thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 3).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 3)
        :param input_flatten               (N, \sum_{l=0}^{L-1} D_l \cdot H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 3), [(D_0, H_0, W_0), (D_1, H_1, W_1), ..., (D_{L-1}, H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, D_0*H_0*W_0, D_0*H_0*W_0+D_1*H_1*W_1, D_0*H_0*W_0+D_1*H_1*W_1+D_2*H_2*W_2, ..., D_0*H_0*W_0+D_1*H_1*W_1+...+D_{L-1}*H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} D_l \cdot H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1] * input_spatial_shapes[:, 2]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 3)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 3:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 0], input_spatial_shapes[..., 2], input_spatial_shapes[..., 1]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        output = ms_deform_attn_core_pytorch_3D(value, input_spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)
        return output
class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", num_feature_levels=4, enc_n_points=4):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, D, H, W = mask.shape
        valid_D = torch.sum(~mask[:, :, 0, 0], 1)
        valid_H = torch.sum(~mask[:, 0, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, 0, :], 1)

        valid_ratio_d = valid_D.float() / D
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_d, valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds):

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, d, h, w = src.shape
            spatial_shape = (d, h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        return memory



class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (D_, H_, W_) in enumerate(spatial_shapes):

            ref_d, ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, D_ - 0.5, D_, dtype=torch.float32, device=device),
                                                 torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                                 torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))

            ref_d = ref_d.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * D_)
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 2] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * W_)

            ref = torch.stack((ref_d, ref_x, ref_y), -1)   # D W H
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class no_op(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=True)
    return data

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

    def get_device(self):
        if next(self.parameters()).device == "cpu":
            return "cpu"
        else:
            return next(self.parameters()).device.index

    def set_device(self, device):
        if device == "cpu":
            self.cpu()
        else:
            self.cuda(device)

    def forward(self, x):
        raise NotImplementedError


class SegmentationNetwork(NeuralNetwork):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # if we have 5 pooling then our patch size must be divisible by 2**5
        self.input_shape_must_be_divisible_by = None  # for example in a 2d network that does 5 pool in x and 6 pool
        # in y this would be (32, 64)

        # we need to know this because we need to know if we are a 2d or a 3d netowrk
        self.conv_op = None  # nn.Conv2d or nn.Conv3d

        # this tells us how many channely we have in the output. Important for preallocation in inference
        self.num_classes = None  # number of channels in the output

        # depending on the loss, we do not hard code a nonlinearity into the architecture. To aggregate predictions
        # during inference, we need to apply the nonlinearity, however. So it is important to let the newtork know what
        # to apply in inference. For the most part this will be softmax
        self.inference_apply_nonlin = lambda x: x  # softmax_helper

        # This is for saving a gaussian importance map for inference. It weights voxels higher that are closer to the
        # center. Prediction at the borders are often less accurate and are thus downweighted. Creating these Gaussians
        # can be expensive, so it makes sense to save and reuse them.
        self._gaussian_3d = self._patch_size_for_gaussian_3d = None
        self._gaussian_2d = self._patch_size_for_gaussian_2d = None

    def predict_3D(self, x: np.ndarray, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
                   use_sliding_window: bool = False,
                   step_size: float = 0.5, patch_size: Tuple[int, ...] = None, regions_class_order: Tuple[int, ...] = None,
                   use_gaussian: bool = False, pad_border_mode: str = "constant",
                   pad_kwargs: dict = None, all_in_gpu: bool = False,
                   verbose: bool = True, mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use this function to predict a 3D image. It does not matter whether the network is a 2D or 3D U-Net, it will
        detect that automatically and run the appropriate code.

        When running predictions, you need to specify whether you want to run fully convolutional of sliding window
        based inference. We very strongly recommend you use sliding window with the default settings.

        It is the responsibility of the user to make sure the network is in the proper mode (eval for inference!). If
        the network is not in eval mode it will print a warning.

        :param x: Your input data. Must be a nd.ndarray of shape (c, x, y, z).
        :param do_mirroring: If True, use test time data augmentation in the form of mirroring
        :param mirror_axes: Determines which axes to use for mirroing. Per default, mirroring is done along all three
        axes
        :param use_sliding_window: if True, run sliding window prediction. Heavily recommended! This is also the default
        :param step_size: When running sliding window prediction, the step size determines the distance between adjacent
        predictions. The smaller the step size, the denser the predictions (and the longer it takes!). Step size is given
        as a fraction of the patch_size. 0.5 is the default and means that wen advance by patch_size * 0.5 between
        predictions. step_size cannot be larger than 1!
        :param patch_size: The patch size that was used for training the network. Do not use different patch sizes here,
        this will either crash or give potentially less accurate segmentations
        :param regions_class_order: Fabian only
        :param use_gaussian: (Only applies to sliding window prediction) If True, uses a Gaussian importance weighting
         to weigh predictions closer to the center of the current patch higher than those at the borders. The reason
         behind this is that the segmentation accuracy decreases towards the borders. Default (and recommended): True
        :param pad_border_mode: leave this alone
        :param pad_kwargs: leave this alone
        :param all_in_gpu: experimental. You probably want to leave this as is it
        :param verbose: Do you want a wall of text? If yes then set this to True
        :param mixed_precision: if True, will run inference in mixed precision with autocast()
        :return:
        """
        torch.cuda.empty_cache()

        assert step_size <= 1, 'step_size must be smaller than 1. Otherwise there will be a gap between consecutive ' \
                               'predictions'

        if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        assert self.get_device() != "cpu", "CPU not implemented"

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3, 4) for a 3d network. This is just to intercept any old
        # code that uses this convention
        if len(mirror_axes):
            if self.conv_op == nn.Conv2d:
                if max(mirror_axes) > 1:
                    raise ValueError("mirror axes. duh")
            if self.conv_op == nn.Conv3d or self.conv_op == Conv3d_wd:
                if max(mirror_axes) > 2:
                    raise ValueError("mirror axes. duh")

        if self.training:
            print('WARNING! Network is in train mode during inference. This may be intended, or not...')

        assert len(x.shape) == 4, "data must have shape (c,x,y,z)"

        if mixed_precision:
            context = autocast
        else:
            context = no_op

        with context():
            with torch.no_grad():
                if self.conv_op == nn.Conv3d or self.conv_op == Conv3d_wd:
                    if use_sliding_window:
                        res = self._internal_predict_3D_3Dconv_tiled(x, step_size, do_mirroring, mirror_axes, patch_size,
                                                                     regions_class_order, use_gaussian, pad_border_mode,
                                                                     pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                     verbose=verbose)
                    else:
                        res = self._internal_predict_3D_3Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs=pad_kwargs, verbose=verbose)
                elif self.conv_op == nn.Conv2d:
                    if use_sliding_window:
                        res = self._internal_predict_3D_2Dconv_tiled(x, patch_size, do_mirroring, mirror_axes, step_size,
                                                                     regions_class_order, use_gaussian, pad_border_mode,
                                                                     pad_kwargs, all_in_gpu, False)
                    else:
                        res = self._internal_predict_3D_2Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs, all_in_gpu, False)
                else:
                    raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")

        return res

    def predict_2D(self, x, do_mirroring: bool, mirror_axes: tuple = (0, 1, 2), use_sliding_window: bool = False,
                   step_size: float = 0.5, patch_size: tuple = None, regions_class_order: tuple = None,
                   use_gaussian: bool = False, pad_border_mode: str = "constant",
                   pad_kwargs: dict = None, all_in_gpu: bool = False,
                   verbose: bool = True, mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use this function to predict a 2D image. If this is a 3D U-Net it will crash because you cannot predict a 2D
        image with that (you dummy).

        When running predictions, you need to specify whether you want to run fully convolutional of sliding window
        based inference. We very strongly recommend you use sliding window with the default settings.

        It is the responsibility of the user to make sure the network is in the proper mode (eval for inference!). If
        the network is not in eval mode it will print a warning.

        :param x: Your input data. Must be a nd.ndarray of shape (c, x, y).
        :param do_mirroring: If True, use test time data augmentation in the form of mirroring
        :param mirror_axes: Determines which axes to use for mirroing. Per default, mirroring is done along all three
        axes
        :param use_sliding_window: if True, run sliding window prediction. Heavily recommended! This is also the default
        :param step_size: When running sliding window prediction, the step size determines the distance between adjacent
        predictions. The smaller the step size, the denser the predictions (and the longer it takes!). Step size is given
        as a fraction of the patch_size. 0.5 is the default and means that wen advance by patch_size * 0.5 between
        predictions. step_size cannot be larger than 1!
        :param patch_size: The patch size that was used for training the network. Do not use different patch sizes here,
        this will either crash or give potentially less accurate segmentations
        :param regions_class_order: Fabian only
        :param use_gaussian: (Only applies to sliding window prediction) If True, uses a Gaussian importance weighting
         to weigh predictions closer to the center of the current patch higher than those at the borders. The reason
         behind this is that the segmentation accuracy decreases towards the borders. Default (and recommended): True
        :param pad_border_mode: leave this alone
        :param pad_kwargs: leave this alone
        :param all_in_gpu: experimental. You probably want to leave this as is it
        :param verbose: Do you want a wall of text? If yes then set this to True
        :return:
        """
        torch.cuda.empty_cache()

        assert step_size <= 1, 'step_size must be smaler than 1. Otherwise there will be a gap between consecutive ' \
                               'predictions'

        if self.conv_op == nn.Conv3d:
            raise RuntimeError("Cannot predict 2d if the network is 3d. Dummy.")

        if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        assert self.get_device() != "cpu", "CPU not implemented"

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3) for a 2d network. This is just to intercept any old
        # code that uses this convention
        if len(mirror_axes):
            if max(mirror_axes) > 1:
                raise ValueError("mirror axes. duh")

        if self.training:
            print('WARNING! Network is in train mode during inference. This may be intended, or not...')

        assert len(x.shape) == 3, "data must have shape (c,x,y)"

        if mixed_precision:
            context = autocast
        else:
            context = no_op

        with context():
            with torch.no_grad():
                if self.conv_op == nn.Conv2d:
                    if use_sliding_window:
                        res = self._internal_predict_2D_2Dconv_tiled(x, step_size, do_mirroring, mirror_axes, patch_size,
                                                                     regions_class_order, use_gaussian, pad_border_mode,
                                                                     pad_kwargs, all_in_gpu, verbose)
                    else:
                        res = self._internal_predict_2D_2Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs, verbose)
                else:
                    raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")

        return res

    @staticmethod
    def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])

        return gaussian_importance_map

    @staticmethod
    def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> List[List[int]]:
        assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
        assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

        # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
        # 110, patch size of 32 and step_size of 0.5, then we want to make 4 steps starting at coordinate 0, 27, 55, 78
        target_step_sizes_in_voxels = [i * step_size for i in patch_size]

        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

        steps = []
        for dim in range(len(patch_size)):
            # the highest step value for this dimension is
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999  # does not matter because there is only one step at 0

            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

            steps.append(steps_here)

        return steps

    def _internal_predict_3D_3Dconv_tiled(self, x: np.ndarray, step_size: float, do_mirroring: bool, mirror_axes: tuple,
                                          patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
                                          pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
                                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        # better safe than sorry
        assert len(x.shape) == 4, "x must be (c, x, y, z)"
        assert self.get_device() != "cpu"
        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape  # still c, x, y, z

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

        if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self._gaussian_3d is None or not all(
                    [i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_3d)]):
                if verbose: print('computing Gaussian')
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)

                self._gaussian_3d = gaussian_importance_map
                self._patch_size_for_gaussian_3d = patch_size
            else:
                if verbose: print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_3d

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map).cuda(self.get_device(),
                                                                                     non_blocking=True)

        else:
            gaussian_importance_map = None

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                    gaussian_importance_map != 0].min()

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(data.shape[1:], device=self.get_device())

            if verbose: print("initializing result array (on GPU)")
            aggregated_results = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                             device=self.get_device())

            if verbose: print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                                       device=self.get_device())
        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_3d
            else:
                # add_for_nb_of_preds = np.ones(data.shape[1:], dtype=np.float32)
                add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]

                    predicted_patch = self._internal_maybe_mirror_and_pred_3D(
                        data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], mirror_axes, do_mirroring,
                        gaussian_importance_map)[0]

                    if all_in_gpu:
                        predicted_patch = predicted_patch.half()
                    else:
                        predicted_patch = predicted_patch.cpu().numpy()

                    aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                    aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
             range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        class_probabilities = aggregated_results / aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = class_probabilities.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = class_probabilities.detach().cpu().numpy()
            else:
                class_probabilities_here = class_probabilities
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if all_in_gpu:
            if verbose: print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            class_probabilities = class_probabilities.detach().cpu().numpy()

        if verbose: print("prediction done")
        return predicted_segmentation, class_probabilities

    def _internal_predict_2D_2Dconv(self, x: np.ndarray, min_size: Tuple[int, int], do_mirroring: bool,
                                    mirror_axes: tuple = (0, 1, 2), regions_class_order: tuple = None,
                                    pad_border_mode: str = "constant", pad_kwargs: dict = None,
                                    verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        This one does fully convolutional inference. No sliding window
        """
        assert len(x.shape) == 3, "x must be (c, x, y)"
        assert self.get_device() != "cpu"
        assert self.input_shape_must_be_divisible_by is not None, 'input_shape_must_be_divisible_by must be set to ' \
                                                                  'run _internal_predict_2D_2Dconv'
        if verbose: print("do mirror:", do_mirroring)

        data, slicer = pad_nd_image(x, min_size, pad_border_mode, pad_kwargs, True,
                                    self.input_shape_must_be_divisible_by)

        predicted_probabilities = self._internal_maybe_mirror_and_pred_2D(data[None], mirror_axes, do_mirroring,
                                                                          None)[0]

        slicer = tuple(
            [slice(0, predicted_probabilities.shape[i]) for i in range(len(predicted_probabilities.shape) -
                                                                       (len(slicer) - 1))] + slicer[1:])
        predicted_probabilities = predicted_probabilities[slicer]

        if regions_class_order is None:
            predicted_segmentation = predicted_probabilities.argmax(0)
            predicted_segmentation = predicted_segmentation.detach().cpu().numpy()
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
        else:
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
            predicted_segmentation = np.zeros(predicted_probabilities.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[predicted_probabilities[i] > 0.5] = c

        return predicted_segmentation, predicted_probabilities

    def _internal_predict_3D_3Dconv(self, x: np.ndarray, min_size: Tuple[int, ...], do_mirroring: bool,
                                    mirror_axes: tuple = (0, 1, 2), regions_class_order: tuple = None,
                                    pad_border_mode: str = "constant", pad_kwargs: dict = None,
                                    verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        This one does fully convolutional inference. No sliding window
        """
        assert len(x.shape) == 4, "x must be (c, x, y, z)"
        assert self.get_device() != "cpu"
        assert self.input_shape_must_be_divisible_by is not None, 'input_shape_must_be_divisible_by must be set to ' \
                                                                  'run _internal_predict_3D_3Dconv'
        if verbose: print("do mirror:", do_mirroring)

        data, slicer = pad_nd_image(x, min_size, pad_border_mode, pad_kwargs, True,
                                    self.input_shape_must_be_divisible_by)

        predicted_probabilities = self._internal_maybe_mirror_and_pred_3D(data[None], mirror_axes, do_mirroring,
                                                                          None)[0]

        slicer = tuple(
            [slice(0, predicted_probabilities.shape[i]) for i in range(len(predicted_probabilities.shape) -
                                                                       (len(slicer) - 1))] + slicer[1:])
        predicted_probabilities = predicted_probabilities[slicer]

        if regions_class_order is None:
            predicted_segmentation = predicted_probabilities.argmax(0)
            predicted_segmentation = predicted_segmentation.detach().cpu().numpy()
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
        else:
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
            predicted_segmentation = np.zeros(predicted_probabilities.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[predicted_probabilities[i] > 0.5] = c

        return predicted_segmentation, predicted_probabilities

    def _internal_maybe_mirror_and_pred_3D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult: np.ndarray or torch.tensor = None) -> torch.tensor:
        assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'
        # everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        # we now return a cuda tensor! Not numpy array!

        x = to_cuda(maybe_to_torch(x), gpu_id=self.get_device())
        result_torch = torch.zeros([1, self.num_classes] + list(x.shape[2:]),
                                   dtype=torch.float).cuda(self.get_device(), non_blocking=True)

        if mult is not None:
            mult = to_cuda(maybe_to_torch(mult), gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 8
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                pred = self.inference_apply_nonlin(self(x))
                result_torch += 1 / num_results * pred

            if m == 1 and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, ))))
                result_torch += 1 / num_results * torch.flip(pred, (4,))

            if m == 2 and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, ))))
                result_torch += 1 / num_results * torch.flip(pred, (3,))

            if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3))

            if m == 4 and (0 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (2, ))))
                result_torch += 1 / num_results * torch.flip(pred, (2,))

            if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 2))

            if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

            if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch

    def _internal_maybe_mirror_and_pred_2D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult: np.ndarray or torch.tensor = None) -> torch.tensor:
        # everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        # we now return a cuda tensor! Not numpy array!
        assert len(x.shape) == 4, 'x must be (b, c, x, y)'

        x = to_cuda(maybe_to_torch(x), gpu_id=self.get_device())
        result_torch = torch.zeros([x.shape[0], self.num_classes] + list(x.shape[2:]),
                                   dtype=torch.float).cuda(self.get_device(), non_blocking=True)

        if mult is not None:
            mult = to_cuda(maybe_to_torch(mult), gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 4
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                pred = self.inference_apply_nonlin(self(x))
                result_torch += 1 / num_results * pred

            if m == 1 and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, ))))
                result_torch += 1 / num_results * torch.flip(pred, (3, ))

            if m == 2 and (0 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (2, ))))
                result_torch += 1 / num_results * torch.flip(pred, (2, ))

            if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch

    def _internal_predict_2D_2Dconv_tiled(self, x: np.ndarray, step_size: float, do_mirroring: bool, mirror_axes: tuple,
                                          patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
                                          pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
                                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        # better safe than sorry
        assert len(x.shape) == 3, "x must be (c, x, y)"
        assert self.get_device() != "cpu"
        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape  # still c, x, y

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1])

        if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self._gaussian_2d is None or not all(
                    [i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_2d)]):
                if verbose: print('computing Gaussian')
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)

                self._gaussian_2d = gaussian_importance_map
                self._patch_size_for_gaussian_2d = patch_size
            else:
                if verbose: print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_2d

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map).cuda(self.get_device(),
                                                                                     non_blocking=True)
        else:
            gaussian_importance_map = None

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                    gaussian_importance_map != 0].min()

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(data.shape[1:], device=self.get_device())

            if verbose: print("initializing result array (on GPU)")
            aggregated_results = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                             device=self.get_device())

            if verbose: print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                                       device=self.get_device())
        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_2d
            else:
                add_for_nb_of_preds = np.ones(data.shape[1:], dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]

                predicted_patch = self._internal_maybe_mirror_and_pred_2D(
                    data[None, :, lb_x:ub_x, lb_y:ub_y], mirror_axes, do_mirroring,
                    gaussian_importance_map)[0]

                if all_in_gpu:
                    predicted_patch = predicted_patch.half()
                else:
                    predicted_patch = predicted_patch.cpu().numpy()

                aggregated_results[:, lb_x:ub_x, lb_y:ub_y] += predicted_patch
                aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
             range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        class_probabilities = aggregated_results / aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = class_probabilities.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = class_probabilities.detach().cpu().numpy()
            else:
                class_probabilities_here = class_probabilities
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if all_in_gpu:
            if verbose: print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            class_probabilities = class_probabilities.detach().cpu().numpy()

        if verbose: print("prediction done")
        return predicted_segmentation, class_probabilities

    def _internal_predict_3D_2Dconv(self, x: np.ndarray, min_size: Tuple[int, int], do_mirroring: bool,
                                    mirror_axes: tuple = (0, 1), regions_class_order: tuple = None,
                                    pad_border_mode: str = "constant", pad_kwargs: dict = None,
                                    all_in_gpu: bool = False, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError
        assert len(x.shape) == 4, "data must be c, x, y, z"
        predicted_segmentation = []
        softmax_pred = []
        for s in range(x.shape[1]):
            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv(
                x[:, s], min_size, do_mirroring, mirror_axes, regions_class_order, pad_border_mode, pad_kwargs, verbose)
            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])
        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))
        return predicted_segmentation, softmax_pred

    def predict_3D_pseudo3D_2Dconv(self, x: np.ndarray, min_size: Tuple[int, int], do_mirroring: bool,
                                   mirror_axes: tuple = (0, 1), regions_class_order: tuple = None,
                                   pseudo3D_slices: int = 5, all_in_gpu: bool = False,
                                   pad_border_mode: str = "constant", pad_kwargs: dict = None,
                                   verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError
        assert len(x.shape) == 4, "data must be c, x, y, z"
        assert pseudo3D_slices % 2 == 1, "pseudo3D_slices must be odd"
        extra_slices = (pseudo3D_slices - 1) // 2

        shp_for_pad = np.array(x.shape)
        shp_for_pad[1] = extra_slices

        pad = np.zeros(shp_for_pad, dtype=np.float32)
        data = np.concatenate((pad, x, pad), 1)

        predicted_segmentation = []
        softmax_pred = []
        for s in range(extra_slices, data.shape[1] - extra_slices):
            d = data[:, (s - extra_slices):(s + extra_slices + 1)]
            d = d.reshape((-1, d.shape[-2], d.shape[-1]))
            pred_seg, softmax_pres = \
                self._internal_predict_2D_2Dconv(d, min_size, do_mirroring, mirror_axes,
                                                 regions_class_order, pad_border_mode, pad_kwargs, verbose)
            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])
        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))

        return predicted_segmentation, softmax_pred

    def _internal_predict_3D_2Dconv_tiled(self, x: np.ndarray, patch_size: Tuple[int, int], do_mirroring: bool,
                                          mirror_axes: tuple = (0, 1), step_size: float = 0.5,
                                          regions_class_order: tuple = None, use_gaussian: bool = False,
                                          pad_border_mode: str = "edge", pad_kwargs: dict =None,
                                          all_in_gpu: bool = False,
                                          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError

        assert len(x.shape) == 4, "data must be c, x, y, z"

        predicted_segmentation = []
        softmax_pred = []

        for s in range(x.shape[1]):
            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv_tiled(
                x[:, s], step_size, do_mirroring, mirror_axes, patch_size, regions_class_order, use_gaussian,
                pad_border_mode, pad_kwargs, all_in_gpu, verbose)

            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])

        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))

        return predicted_segmentation, softmax_pred

class Conv3d_wd(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d_wd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        # std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)


def Norm_layer(norm_cfg, inplanes):

    if norm_cfg == 'BN':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm3d(inplanes,affine=True)

    return out


def Activation_layer(activation_cfg, inplace=True):

    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, stride=(1, 1, 1), downsample=None, weight_std=False):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, planes)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.nonlin(out)

        return out


class Backbone(nn.Module):

    arch_settings = {
        9: (ResBlock, (3, 3, 2))
    }


    def __init__(self,
                 depth,
                 in_channels=1,
                 norm_cfg='BN',
                 activation_cfg='ReLU',
                 weight_std=False):
        super(Backbone, self).__init__()

        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        block, layers = self.arch_settings[depth]
        self.inplanes = 64
        self.conv1 = conv3x3x3(in_channels, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=False, weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, 64)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.layer1 = self._make_layer(block, 192, layers[0], stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer2 = self._make_layer(block, 384, layers[1], stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer3 = self._make_layer(block, 384, layers[2], stride=(2, 2, 2), norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        self.layers = []

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd)):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=(1, 1, 1), norm_cfg='BN', activation_cfg='ReLU', weight_std=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv3x3x3(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False, weight_std=weight_std), Norm_layer(norm_cfg, planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, stride=stride, downsample=downsample, weight_std=weight_std))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, weight_std=weight_std))

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlin(x)
        out.append(x)

        x = self.layer1(x)
        out.append(x)
        x = self.layer2(x)
        out.append(x)
        x = self.layer3(x)
        out.append(x)

        return out


class Conv3d_wd(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
                 groups=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4,
                                                                                                                keepdim=True)
        weight = weight - weight_mean
        # std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1,
              bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)


def Norm_layer(norm_cfg, inplanes):
    if norm_cfg == 'BN':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm3d(inplanes, affine=True)

    return out


def Activation_layer(activation_cfg, inplace=True):
    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg, activation_cfg, kernel_size, stride=(1, 1, 1),
                 padding=(0, 0, 0), dilation=(1, 1, 1), bias=False, weight_std=False):
        super(Conv3dBlock, self).__init__()
        self.conv = conv3x3x3(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=bias, weight_std=weight_std)
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, weight_std=False):
        super(ResBlock, self).__init__()
        self.resconv1 = Conv3dBlock(inplanes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1,
                                    bias=False, weight_std=weight_std)
        self.resconv2 = Conv3dBlock(planes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1,
                                    bias=False, weight_std=weight_std)

    def forward(self, x):
        residual = x

        out = self.resconv1(x)
        out = self.resconv2(out)
        out = out + residual

        return out


class U_ResTran3D(nn.Module):
    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', img_size=None, num_classes=None, weight_std=False,in_channels=1):
        super(U_ResTran3D, self).__init__()

        self.MODEL_NUM_CLASSES = num_classes

        self.upsamplex2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')

        self.transposeconv_stage2 = nn.ConvTranspose3d(384, 384, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_stage1 = nn.ConvTranspose3d(384, 192, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)
        self.transposeconv_stage0 = nn.ConvTranspose3d(192, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False)

        self.stage2_de = ResBlock(384, 384, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage1_de = ResBlock(192, 192, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage0_de = ResBlock(64, 64, norm_cfg, activation_cfg, weight_std=weight_std)

        self.ds2_cls_conv = nn.Conv3d(384, self.MODEL_NUM_CLASSES, kernel_size=1)
        self.ds1_cls_conv = nn.Conv3d(192, self.MODEL_NUM_CLASSES, kernel_size=1)
        self.ds0_cls_conv = nn.Conv3d(64, self.MODEL_NUM_CLASSES, kernel_size=1)

        self.cls_conv = nn.Conv3d(64, self.MODEL_NUM_CLASSES, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd, nn.ConvTranspose3d)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.backbone = Backbone(depth=9,in_channels=in_channels, norm_cfg=norm_cfg, activation_cfg=activation_cfg,
                                             weight_std=weight_std)
        total = sum([param.nelement() for param in self.backbone.parameters()])
        print('  + Number of Backbone Params: %.2f(e6)' % (total / 1e6))

        self.position_embed = build_position_encoding(mode='v2', hidden_dim=384)
        self.encoder_Detrans = DeformableTransformer(d_model=384, dim_feedforward=1536, dropout=0.1, activation='gelu',
                                                     num_feature_levels=2, nhead=6, num_encoder_layers=6,
                                                     enc_n_points=4)
        total = sum([param.nelement() for param in self.encoder_Detrans.parameters()])
        print('  + Number of Transformer Params: %.2f(e6)' % (total / 1e6))

    def posi_mask(self, x):

        x_fea = []
        x_posemb = []
        masks = []
        for lvl, fea in enumerate(x):
            if lvl > 1:
                x_fea.append(fea)
                x_posemb.append(self.position_embed(fea))
                masks.append(
                    torch.zeros(
                        (fea.shape[0], fea.shape[2], fea.shape[3], fea.shape[4]),
                        dtype=torch.bool,
                        device=fea.device
                    )
                )

        return x_fea, masks, x_posemb

    def forward(self, inputs):
        x_convs = self.backbone(inputs)
        x_fea, masks, x_posemb = self.posi_mask(x_convs)
        x_trans = self.encoder_Detrans(x_fea, masks, x_posemb)

        # # Single_scale
        # # x = self.transposeconv_stage2(x_trans.transpose(-1, -2).view(x_convs[-1].shape))
        # # skip2 = x_convs[-2]
        # Multi-scale
        x = self.transposeconv_stage2(
            x_trans[:, x_fea[0].shape[-3] * x_fea[0].shape[-2] * x_fea[0].shape[-1]::].transpose(-1, -2).view(
                x_convs[-1].shape))  # x_trans length: 12*24*24+6*12*12=7776
        skip2 = x_trans[:, 0:x_fea[0].shape[-3] * x_fea[0].shape[-2] * x_fea[0].shape[-1]].transpose(-1, -2).view(
            x_convs[-2].shape)

        x = x + skip2
        x = self.stage2_de(x)
        ds2 = self.ds2_cls_conv(x)

        x = self.transposeconv_stage1(x)
        skip1 = x_convs[-3]
        x = x + skip1
        x = self.stage1_de(x)
        ds1 = self.ds1_cls_conv(x)

        x = self.transposeconv_stage0(x)
        skip0 = x_convs[-4]
        x = x + skip0
        x = self.stage0_de(x)
        ds0 = self.ds0_cls_conv(x)

        result = self.upsamplex2(x)
        result = self.cls_conv(result)

        return [result, ds0, ds1, ds2]


class ResTranUnet(SegmentationNetwork):
    """
    ResTran-3D Unet
    """

    def __init__(self, in_channels=1, norm_cfg='BN', activation_cfg='ReLU', img_size=None, num_classes=None, weight_std=False,
                 deep_supervision=False):
        super().__init__()
        self.do_ds = False
        self.U_ResTran3D = U_ResTran3D(norm_cfg, activation_cfg, img_size, num_classes, weight_std,in_channels=in_channels)  # U_ResTran3D

        if weight_std == False:
            self.conv_op = nn.Conv3d
        else:
            self.conv_op = Conv3d_wd
        if norm_cfg == 'BN':
            self.norm_op = nn.BatchNorm3d
        if norm_cfg == 'SyncBN':
            self.norm_op = nn.SyncBatchNorm
        if norm_cfg == 'GN':
            self.norm_op = nn.GroupNorm
        if norm_cfg == 'IN':
            self.norm_op = nn.InstanceNorm3d
        self.dropout_op = nn.Dropout3d
        self.num_classes = num_classes
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

    def forward(self, x):
        seg_output = self.U_ResTran3D(x)
        if self._deep_supervision and self.do_ds:
            return seg_output
        else:
            return seg_output[0]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    B = 2
    in_channels = 3
    out_channels = 3
    D, H, W = 128, 128, 64

    model = ResTranUnet(
        in_channels=in_channels,
        num_classes=out_channels,
        img_size=(D, H, W),
        deep_supervision=False,
    ).to(device)

    x = torch.randn(B, in_channels, D, H, W, device=device)

    model.eval()
    with torch.no_grad():
        y = model(x)

    print("y.shape =", tuple(y.shape))