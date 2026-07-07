import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util
import sys
from pathlib import Path


def _load_base_module():
    module_path = Path(__file__).with_name("HWAUNETR_Mu.py")
    module_name = "_hwav2_center_prior_base_hwaunetr_mu"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base model module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_base_module()
DetectorConvNormAct3d = _base.DetectorConvNormAct3d
DetectorResidualBlock3d = _base.DetectorResidualBlock3d
DetectorZOnlyMamba = _base.DetectorZOnlyMamba
Encoder = _base.Encoder
TODMClassificationHead3D = _base.TODMClassificationHead3D
TransposedConvLayer = _base.TransposedConvLayer


class ISPHWABlockV2(nn.Module):
    """
    Center-conditioned HWA variant.

    This block intentionally removes ROI cropping, local ROI encoders, and
    paste-back fusion. The detector predicts a stable center response, which is
    converted into multi-scale soft center priors and a weak input residual.
    """

    encoder_prior_enabled = False

    def __init__(
        self,
        in_chans=3,
        kernel_sizes=(4, 2, 2, 2),
        dims=(48, 96, 192, 384),
        det_channels=24,
        softargmax_beta=12.0,
        anchor_topk=8,
        anchor_softmax_beta=8.0,
        evidence_temperature=1.0,
        peak_kernel_size=5,
        peak_gate_gain=10.0,
        peak_gate_bias=0.35,
        stage_scope_factors=(0.90, 1.35, 1.90, 2.60),
    ):
        super().__init__()
        self.in_chans = int(in_chans)
        self.kernel_sizes = tuple(int(v) for v in kernel_sizes)
        self.dims = tuple(int(v) for v in dims)
        self.det_channels = int(det_channels)
        self.softargmax_beta = float(softargmax_beta)
        self.anchor_topk = max(1, int(anchor_topk))
        self.anchor_softmax_beta = float(anchor_softmax_beta)
        self.evidence_temperature = float(evidence_temperature)
        self.peak_kernel_size = max(3, int(peak_kernel_size))
        if self.peak_kernel_size % 2 == 0:
            self.peak_kernel_size += 1
        self.peak_gate_gain = float(peak_gate_gain)
        self.peak_gate_bias = float(peak_gate_bias)
        self.stage_scope_factors = tuple(float(v) for v in stage_scope_factors)

        cumulative = []
        scale = 1
        for k in self.kernel_sizes:
            scale *= k
            cumulative.append(scale)
        self.stage_windows = tuple(cumulative)

        det_base = max(16, self.det_channels)
        self.det_feat_dim = int(max(64, self.det_channels * 4, self.dims[0]))
        self.detector_stem = nn.Sequential(
            DetectorConvNormAct3d(self.in_chans, det_base),
            DetectorResidualBlock3d(det_base),
        )
        self.detector_down1 = nn.Sequential(
            DetectorConvNormAct3d(det_base, det_base * 2, stride=(2, 2, 2)),
            DetectorResidualBlock3d(det_base * 2),
        )
        self.detector_down2 = nn.Sequential(
            DetectorConvNormAct3d(det_base * 2, self.det_feat_dim, stride=(2, 2, 2)),
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

        self.input_pool_scales = (1, 2, 4, 8)
        self.input_channel_agg_proj = nn.Conv3d(
            len(self.input_pool_scales) * self.in_chans,
            self.in_chans,
            kernel_size=3,
            padding=1,
            groups=self.in_chans,
            bias=True,
        )
        self.input_agg_gain = nn.Parameter(torch.tensor(0.24, dtype=torch.float32))
        self.input_gate_boost = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.center_prior_gain = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.input_enhance_gain = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.output_logit_gain = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

        with torch.no_grad():
            nn.init.zeros_(self.input_channel_agg_proj.weight)
            nn.init.zeros_(self.input_channel_agg_proj.bias)
            center = self.input_channel_agg_proj.kernel_size[0] // 2
            for branch_idx, scale_weight in enumerate((0.70, 0.18, 0.08, 0.04)):
                self.input_channel_agg_proj.weight[
                    :, branch_idx, center, center, center
                ] = float(scale_weight)

        self.last_input_enhance_map = None
        self.last_fused_support_map = None
        self.last_input_aggregate = None
        self.last_input_prior_reliability = None
        self.last_prior_quality_maps = None
        self.last_prior_quality_scores = None
        self.last_stage_roi_centers = None
        self.last_stage_roi_sizes_full = None

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size=size, mode="trilinear", align_corners=False)

    @staticmethod
    def _normalize_response_map(response: torch.Tensor) -> torch.Tensor:
        response_f = response.float()
        r_min = response_f.amin(dim=(2, 3, 4), keepdim=True)
        r_max = response_f.amax(dim=(2, 3, 4), keepdim=True)
        return ((response_f - r_min) / (r_max - r_min).clamp_min(1e-4)).to(
            dtype=response.dtype
        )

    def _peak_concentrate(self, prob: torch.Tensor) -> torch.Tensor:
        k = self.peak_kernel_size
        pad = k // 2
        local_avg = F.avg_pool3d(prob, kernel_size=k, stride=1, padding=pad)
        local_max = F.max_pool3d(prob, kernel_size=k, stride=1, padding=pad)
        peakness = (prob - local_avg) / (local_max - local_avg).clamp_min(1e-4)
        gate = torch.sigmoid(self.peak_gate_gain * (peakness - self.peak_gate_bias))
        return (prob * (0.15 + 0.85 * gate)).clamp(1e-4, 1.0 - 1e-4)

    def _soft_argmax_3d(self, logits: torch.Tensor) -> torch.Tensor:
        b, _, h, w, z = logits.shape
        flat = logits.view(b, 1, -1)
        prob = torch.softmax(self.softargmax_beta * flat, dim=-1).view(b, 1, h, w, z)
        xs = torch.linspace(0, h - 1, h, device=logits.device, dtype=logits.dtype).view(
            1, 1, h, 1, 1
        )
        ys = torch.linspace(0, w - 1, w, device=logits.device, dtype=logits.dtype).view(
            1, 1, 1, w, 1
        )
        zs = torch.linspace(0, z - 1, z, device=logits.device, dtype=logits.dtype).view(
            1, 1, 1, 1, z
        )
        return torch.stack(
            [
                (prob * xs).sum(dim=(2, 3, 4)),
                (prob * ys).sum(dim=(2, 3, 4)),
                (prob * zs).sum(dim=(2, 3, 4)),
            ],
            dim=-1,
        )[:, 0, :]

    def _topk_anchor_readout(self, logits: torch.Tensor):
        b, _, h, w, z = logits.shape
        flat = logits.reshape(b, 1, -1)
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
        center = (weights.unsqueeze(-1) * coords).sum(dim=-2)[:, 0, :]
        sparse = flat.new_zeros(flat.shape)
        sparse.scatter_(-1, idx, weights)
        return center, sparse.reshape(b, 1, h, w, z)

    @staticmethod
    def _map_coords_to_fullres(
        coords_det: torch.Tensor,
        det_size,
        full_size,
    ) -> torch.Tensor:
        scale = coords_det.new_tensor(
            [
                float(full_size[0]) / float(max(det_size[0], 1)),
                float(full_size[1]) / float(max(det_size[1], 1)),
                float(full_size[2]) / float(max(det_size[2], 1)),
            ]
        ).view(1, 3)
        coords = coords_det * scale
        return torch.stack(
            [
                coords[:, 0].clamp(0.0, float(full_size[0] - 1)),
                coords[:, 1].clamp(0.0, float(full_size[1] - 1)),
                coords[:, 2].clamp(0.0, float(full_size[2] - 1)),
            ],
            dim=1,
        )

    @staticmethod
    def _map_coords_full_to_stage(coords_full: torch.Tensor, full_size, stage_size):
        scale = coords_full.new_tensor(
            [
                float(stage_size[0]) / float(max(full_size[0], 1)),
                float(stage_size[1]) / float(max(full_size[1], 1)),
                float(stage_size[2]) / float(max(full_size[2], 1)),
            ]
        ).view(1, 3)
        coords = coords_full * scale
        return torch.stack(
            [
                coords[:, 0].clamp(0.0, float(stage_size[0] - 1)),
                coords[:, 1].clamp(0.0, float(stage_size[1] - 1)),
                coords[:, 2].clamp(0.0, float(stage_size[2] - 1)),
            ],
            dim=1,
        )

    def _response_sigma_full(
        self,
        response_full: torch.Tensor,
        center_full: torch.Tensor,
        min_sigma=(3.0, 3.0, 2.0),
        max_sigma=(16.0, 16.0, 10.0),
    ) -> torch.Tensor:
        b, _, h, w, z = response_full.shape
        weights = self._normalize_response_map(response_full).clamp_min(1e-6)
        weights = weights / weights.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
        xs = torch.arange(h, device=response_full.device, dtype=response_full.dtype).view(
            1, 1, h, 1, 1
        )
        ys = torch.arange(w, device=response_full.device, dtype=response_full.dtype).view(
            1, 1, 1, w, 1
        )
        zs = torch.arange(z, device=response_full.device, dtype=response_full.dtype).view(
            1, 1, 1, 1, z
        )
        cx = center_full[:, 0].view(b, 1, 1, 1, 1)
        cy = center_full[:, 1].view(b, 1, 1, 1, 1)
        cz = center_full[:, 2].view(b, 1, 1, 1, 1)
        sigma = torch.stack(
            [
                torch.sqrt((weights * (xs - cx).pow(2)).sum(dim=(2, 3, 4)).clamp_min(1e-6)),
                torch.sqrt((weights * (ys - cy).pow(2)).sum(dim=(2, 3, 4)).clamp_min(1e-6)),
                torch.sqrt((weights * (zs - cz).pow(2)).sum(dim=(2, 3, 4)).clamp_min(1e-6)),
            ],
            dim=1,
        )[:, :, 0]
        min_s = response_full.new_tensor(min_sigma).view(1, 3)
        max_s = response_full.new_tensor(max_sigma).view(1, 3)
        return sigma.clamp(min=min_s, max=max_s)

    @staticmethod
    def _render_gaussian(center: torch.Tensor, sigma: torch.Tensor, size_hwz) -> torch.Tensor:
        b = center.shape[0]
        h, w, z = [int(v) for v in size_hwz]
        device = center.device
        dtype = center.dtype
        xs = torch.arange(h, device=device, dtype=dtype).view(1, h, 1, 1)
        ys = torch.arange(w, device=device, dtype=dtype).view(1, 1, w, 1)
        zs = torch.arange(z, device=device, dtype=dtype).view(1, 1, 1, z)
        cx = center[:, 0].view(b, 1, 1, 1)
        cy = center[:, 1].view(b, 1, 1, 1)
        cz = center[:, 2].view(b, 1, 1, 1)
        sx = sigma[:, 0].clamp_min(1.0).view(b, 1, 1, 1)
        sy = sigma[:, 1].clamp_min(1.0).view(b, 1, 1, 1)
        sz = sigma[:, 2].clamp_min(1.0).view(b, 1, 1, 1)
        dist = ((xs - cx) / sx).pow(2) + ((ys - cy) / sy).pow(2) + ((zs - cz) / sz).pow(2)
        return torch.exp(-0.5 * dist).unsqueeze(1)

    @staticmethod
    def _input_prior_reliability(x: torch.Tensor, enhance_map: torch.Tensor) -> torch.Tensor:
        saliency = x.detach().float().abs().mean(dim=1, keepdim=True)
        sal_min = saliency.amin(dim=(2, 3, 4), keepdim=True)
        sal_max = saliency.amax(dim=(2, 3, 4), keepdim=True)
        saliency = (saliency - sal_min) / (sal_max - sal_min).clamp_min(1e-4)
        support = enhance_map.detach().float().clamp(0.0, 1.0)
        mass = support.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)
        prior_saliency = (saliency * support).sum(dim=(2, 3, 4), keepdim=True) / mass
        global_saliency = saliency.mean(dim=(2, 3, 4), keepdim=True)
        return torch.sigmoid(6.0 * (prior_saliency - global_saliency)).to(dtype=x.dtype)

    def build_input_aggregate(
        self,
        x: torch.Tensor,
        enhance_map: torch.Tensor = None,
    ) -> torch.Tensor:
        b, m, h, w, z = x.shape
        per_modal = []
        for modal_idx in range(m):
            channel = x[:, modal_idx : modal_idx + 1]
            branches = []
            for scale in self.input_pool_scales:
                if scale == 1:
                    pooled = channel
                else:
                    pooled = F.avg_pool3d(
                        channel,
                        kernel_size=scale,
                        stride=scale,
                        ceil_mode=True,
                    )
                    pooled = F.interpolate(
                        pooled,
                        size=(h, w, z),
                        mode="trilinear",
                        align_corners=False,
                    )
                branches.append(pooled)
            per_modal.append(torch.cat(branches, dim=1))
        aggregate = self.input_channel_agg_proj(torch.cat(per_modal, dim=1))

        if enhance_map is not None:
            if enhance_map.shape[2:] != (h, w, z):
                enhance_map = self._resize(enhance_map.float(), (h, w, z))
            enhance_map = enhance_map.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
            if enhance_map.shape[1] != 1:
                enhance_map = enhance_map.mean(dim=1, keepdim=True)
            reliability = self._input_prior_reliability(x, enhance_map)
            self.last_input_prior_reliability = reliability
            prior_gain = torch.clamp(
                self.center_prior_gain.to(device=x.device, dtype=x.dtype),
                0.0,
                0.30,
            )
            centered_prior = enhance_map - enhance_map.mean(dim=(2, 3, 4), keepdim=True)
            aggregate = aggregate * (1.0 + prior_gain * reliability * centered_prior).clamp(
                0.85, 1.15
            )
        else:
            self.last_input_prior_reliability = None

        self.last_input_aggregate = aggregate
        return aggregate

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        b, m, h, w, z = x.shape
        if m != self.in_chans:
            raise ValueError(f"Expected input channels {self.in_chans}, got {m}")

        det_feat = self.detector_stem(x)
        det_feat = self.detector_down1(det_feat)
        det_feat = self.detector_down2(det_feat)
        det_feat = self.modal_shared_proj(det_feat)
        ctx = self.detector_context(det_feat)
        anchor_logits_low = self.center_head(ctx["rim"]) + self.core_head(ctx["core"]) - self.excl_head(ctx["unc"])
        anchor_prob_low = self._peak_concentrate(
            torch.sigmoid(anchor_logits_low / self.evidence_temperature)
        )
        anchor_logits_focus_low = torch.logit(anchor_prob_low.clamp(1e-4, 1.0 - 1e-4))
        det_size = tuple(anchor_prob_low.shape[2:])
        full_size = (h, w, z)
        anchor_prob_full = self._resize(anchor_prob_low, full_size).clamp(1e-4, 1.0 - 1e-4)
        anchor_sparse_low = None

        center_topk_det, anchor_sparse_low = self._topk_anchor_readout(anchor_logits_focus_low)
        center_global_det = self._soft_argmax_3d(anchor_logits_focus_low)
        center_topk_full = self._map_coords_to_fullres(center_topk_det, det_size, full_size)
        center_global_full = self._map_coords_to_fullres(center_global_det, det_size, full_size)
        sigma_full = self._response_sigma_full(anchor_prob_full, center_topk_full)

        det_center_each_full = center_topk_full.unsqueeze(1).expand(-1, self.in_chans, -1)
        det_center_coarse_each_full = center_global_full.unsqueeze(1).expand(-1, self.in_chans, -1)
        det_sigma_each_full = sigma_full.unsqueeze(1).expand(-1, self.in_chans, -1)
        det_conf_each_full = x.new_ones((b, self.in_chans))

        stage_window_evidence = []
        stage_centers = []
        stage_sizes = []
        stage_quality_maps = []
        for idx, scope in enumerate(self.stage_scope_factors):
            hs = max(1, h // self.stage_windows[idx])
            ws = max(1, w // self.stage_windows[idx])
            zs = max(1, z // self.stage_windows[idx])
            stage_size = (hs, ws, zs)
            prior_full = self._render_gaussian(
                center_topk_full,
                (sigma_full * float(scope)).clamp_min(1.0),
                full_size,
            )
            prior_stage = self._resize(prior_full, stage_size).clamp(0.0, 1.0)
            prior_stage = self._normalize_response_map(prior_stage)
            stage_window_evidence.append(prior_stage.expand(-1, self.in_chans, -1, -1, -1))
            stage_quality_maps.append(prior_stage)
            stage_centers.append(self._map_coords_full_to_stage(center_topk_full, full_size, stage_size))
            stage_sizes.append((sigma_full * float(scope) * 4.0).clamp_min(4.0))

        input_prior = self._render_gaussian(
            center_topk_full,
            (sigma_full * self.stage_scope_factors[0]).clamp_min(1.0),
            full_size,
        ).clamp(0.0, 1.0)
        fused_support = torch.maximum(anchor_prob_full, input_prior).clamp(1e-4, 1.0 - 1e-4)
        self.last_input_enhance_map = input_prior
        self.last_fused_support_map = fused_support
        self.last_prior_quality_maps = stage_quality_maps
        self.last_prior_quality_scores = [x.new_ones(b, 1, 1, 1, 1) for _ in stage_quality_maps]
        self.last_stage_roi_centers = stage_centers
        self.last_stage_roi_sizes_full = stage_sizes
        self.build_input_aggregate(x, input_prior)

        if not return_debug:
            return None

        evidence_each = anchor_prob_full.expand(-1, self.in_chans, -1, -1, -1)
        seed_each = fused_support.expand(-1, self.in_chans, -1, -1, -1)
        debug = {
            "det_anchor_logits_full": self._resize(anchor_logits_low, full_size),
            "det_anchor_prob_full": anchor_prob_full,
            "det_raw_evidence_prob_each": evidence_each,
            "det_seed_evidence_prob_each": seed_each,
            "det_fused_seed_prob_full": fused_support,
            "det_anchor_sparse_prob_full": self._resize(anchor_sparse_low, full_size).clamp(1e-4, 1.0 - 1e-4),
            "det_center_topk_full": center_topk_full,
            "det_center_global_full": center_global_full,
            "det_center_each_full": det_center_each_full,
            "det_center_coarse_each_full": det_center_coarse_each_full,
            "det_sigma_each_full": det_sigma_each_full,
            "det_conf_each_full": det_conf_each_full,
            "det_fused_center_full": center_topk_full,
            "det_fused_center_coarse_full": center_global_full,
            "det_fused_sigma_full": sigma_full,
            "det_fused_conf_full": det_conf_each_full.mean(dim=1, keepdim=True),
            "stage_window_evidence": stage_window_evidence,
            "stage_prior_quality_maps": stage_quality_maps,
            "stage_prior_quality_scores": self.last_prior_quality_scores,
            "stage_roi_centers": stage_centers,
            "stage_roi_sizes_full": stage_sizes,
            "input_enhance_map": self.last_input_enhance_map,
            "fused_support_map": self.last_fused_support_map,
            "input_aggregate": self.last_input_aggregate,
            "input_prior_reliability": self.last_input_prior_reliability,
            "hwa_priors": None,
        }
        return None, debug


class HWAUNETRCenterPriorV2(nn.Module):
    def __init__(
        self,
        in_chans=3,
        out_chans=3,
        hwa_block=(1, 2, 4, 8),
        kernel_sizes=(4, 2, 2, 2),
        depths=(2, 2, 2, 2),
        dims=(48, 96, 192, 384),
        heads=(1, 2, 4, 4),
        hidden_size=768,
        num_slices_list=(64, 32, 16, 8),
        out_indices=(0, 1, 2, 3),
        hwa_det_channels=24,
        hwa_softargmax_beta=12.0,
        hwa_evidence_temperature=1.0,
        hwa_peak_kernel_size=5,
        hwa_peak_gate_gain=10.0,
        hwa_peak_gate_bias=0.35,
        hwa_stage_scope_factors=(0.90, 1.35, 1.90, 2.60),
        use_hwa_prior_in_encoder=True,
        **kwargs,
    ):
        super().__init__()
        hwa_block = kwargs.pop("fussion", hwa_block)
        self.use_hwa_prior_in_encoder = bool(use_hwa_prior_in_encoder)
        self.runtime_hwa_gate_scale = 1.0
        self.runtime_hwa_gain_scale = 1.0
        self.hwa_input_agg_gain_max = 0.35
        self.hwa_input_enhance_gain_max = 0.12
        self.hwa_input_gate_scale_max = 1.0
        self.hwa_input_delta_std_clip = 1.5
        self.hwa_output_logit_gain_max = 0.25
        self.hwa_block = ISPHWABlockV2(
            in_chans=in_chans,
            kernel_sizes=kernel_sizes,
            dims=dims,
            det_channels=hwa_det_channels,
            softargmax_beta=hwa_softargmax_beta,
            evidence_temperature=hwa_evidence_temperature,
            peak_kernel_size=hwa_peak_kernel_size,
            peak_gate_gain=hwa_peak_gate_gain,
            peak_gate_bias=hwa_peak_gate_bias,
            stage_scope_factors=hwa_stage_scope_factors,
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
        self.hidden_downsample = nn.Conv3d(dims[3], hidden_size, kernel_size=2, stride=2)
        self.TSconv1 = TransposedConvLayer(dim_in=hidden_size, dim_out=dims[3], head=heads[3], r=2)
        self.TSconv2 = TransposedConvLayer(dim_in=dims[3], dim_out=dims[2], head=heads[2], r=kernel_sizes[3])
        self.TSconv3 = TransposedConvLayer(dim_in=dims[2], dim_out=dims[1], head=heads[1], r=kernel_sizes[2])
        self.TSconv4 = TransposedConvLayer(dim_in=dims[1], dim_out=dims[0], head=heads[0], r=kernel_sizes[1])
        self.todm_seg_head = nn.ConvTranspose3d(dims[0], out_chans, kernel_size=kernel_sizes[0], stride=kernel_sizes[0])
        self.todm_cls_head = TODMClassificationHead3D(
            in_chs=dims[::-1],
            proj_ch=128,
            attn_hidden=256,
            cls_hidden=256,
            dropout=0.2,
        )
        self.last_input_aggregate_delta_mean = None
        self.last_input_aggregate_delta_ratio = None
        self.last_logit_bias_map = None

    def _apply_center_prior_input(self, x: torch.Tensor) -> torch.Tensor:
        gate_scale = float(max(0.0, min(1.0, getattr(self, "runtime_hwa_gate_scale", 1.0))))
        if not self.use_hwa_prior_in_encoder or gate_scale <= 0.0:
            self.last_input_aggregate_delta_mean = None
            self.last_input_aggregate_delta_ratio = None
            return x

        input_aggregate = self.hwa_block.last_input_aggregate
        if input_aggregate is None:
            self.last_input_aggregate_delta_mean = None
            self.last_input_aggregate_delta_ratio = None
            return x
        if input_aggregate.shape[2:] != x.shape[2:]:
            input_aggregate = F.interpolate(
                input_aggregate.float(),
                size=x.shape[2:],
                mode="trilinear",
                align_corners=False,
            ).to(dtype=x.dtype)

        gain_max = float(max(0.0, getattr(self, "hwa_input_agg_gain_max", 0.35)))
        gain = torch.clamp(
            self.hwa_block.input_agg_gain.to(device=x.device, dtype=x.dtype),
            0.0,
            gain_max,
        )
        gain_scale = float(max(0.0, min(1.0, getattr(self, "runtime_hwa_gain_scale", 1.0))))
        gain = gain * gain_scale
        gate_max = float(max(0.0, getattr(self, "hwa_input_gate_scale_max", 1.0)))
        boost = torch.clamp(
            self.hwa_block.input_gate_boost.to(device=x.device, dtype=x.dtype),
            0.0,
            gate_max,
        )
        eff_gate = torch.clamp(gate_scale * boost, 0.0, gate_max)
        raw_std = x.std(dim=(2, 3, 4), keepdim=True, unbiased=False).clamp_min(1e-5)
        delta = input_aggregate.to(device=x.device, dtype=x.dtype) - x
        delta_clip = float(max(1.0e-3, getattr(self, "hwa_input_delta_std_clip", 1.5)))
        delta = torch.tanh(delta / (delta_clip * raw_std)) * (delta_clip * raw_std)
        self.last_input_aggregate_delta_mean = delta.detach().float().abs().mean()
        self.last_input_aggregate_delta_ratio = (
            delta.detach().float().abs().mean()
            / x.detach().float().abs().mean().clamp_min(1e-5)
        )
        out = x
        enhance_map = getattr(self.hwa_block, "last_input_enhance_map", None)
        if enhance_map is not None:
            if enhance_map.shape[2:] != x.shape[2:]:
                enhance_map = F.interpolate(
                    enhance_map.float(),
                    size=x.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
            if enhance_map.shape[1] != 1:
                enhance_map = enhance_map.mean(dim=1, keepdim=True)
            enhance_map = enhance_map.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
            enhance_gain_max = float(
                max(0.0, getattr(self, "hwa_input_enhance_gain_max", 0.12))
            )
            enhance_gain = torch.clamp(
                self.hwa_block.input_enhance_gain.to(device=x.device, dtype=x.dtype),
                0.0,
                enhance_gain_max,
            )
            out = out * (1.0 + eff_gate * enhance_gain * torch.sqrt(enhance_map.clamp_min(1e-6)))
        return out + eff_gate * gain * delta

    def _apply_hwa_input_enhancement(
        self,
        x: torch.Tensor,
        enhance_map=None,
        input_aggregate=None,
    ) -> torch.Tensor:
        if input_aggregate is not None:
            self.hwa_block.last_input_aggregate = input_aggregate
        return self._apply_center_prior_input(x)

    def _apply_hwa_logit_bias(self, seg_logits: torch.Tensor, enhance_map=None) -> torch.Tensor:
        if enhance_map is None:
            self.last_logit_bias_map = None
            return seg_logits

        gate_scale = float(max(0.0, min(1.0, getattr(self, "runtime_hwa_gate_scale", 1.0))))
        if not self.use_hwa_prior_in_encoder or gate_scale <= 0.0:
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

        support = torch.sqrt(enhance_map.clamp_min(1.0e-6))
        logit_gain_max = float(max(0.0, getattr(self, "hwa_output_logit_gain_max", 0.25)))
        gain = torch.clamp(
            self.hwa_block.output_logit_gain.to(device=seg_logits.device, dtype=seg_logits.dtype),
            0.0,
            logit_gain_max,
        )
        if float(gain.detach().max().cpu()) <= 0.0:
            self.last_logit_bias_map = None
            return seg_logits

        bias = gate_scale * gain * support
        self.last_logit_bias_map = bias.detach()
        return seg_logits + bias.expand(-1, seg_logits.shape[1], -1, -1, -1)

    def forward(self, x, return_debug: bool = False, detector_only: bool = False):
        if detector_only:
            _, debug = self.hwa_block(x, return_debug=True)
            return None, None, debug

        debug = None
        if self.use_hwa_prior_in_encoder or return_debug:
            if return_debug:
                _, debug = self.hwa_block(x, return_debug=True)
            else:
                self.hwa_block(x, return_debug=False)
        encoder_input = self._apply_center_prior_input(x)
        outs, feature_out = self.Encoder(encoder_input, None, return_debug=False)

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
        enhance_map = getattr(self.hwa_block, "last_input_enhance_map", None)
        seg_logits = self._apply_hwa_logit_bias(
            seg_logits,
            enhance_map if self.use_hwa_prior_in_encoder else None,
        )
        cls_logits = self.todm_cls_head(up_feature)

        if return_debug:
            return cls_logits, seg_logits, debug or {}
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
