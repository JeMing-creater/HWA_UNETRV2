import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_center_prior_module():
    module_path = Path(__file__).with_name("HWAUNETR_CenterPrior.py")
    module_name = "_hwav2_soft_prior_center_prior_base"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load center-prior model module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_center_prior_module()
ISPHWABlockV2 = _base.ISPHWABlockV2
HWAUNETRCenterPriorV2 = _base.HWAUNETRCenterPriorV2


class PGHWABlockV2(ISPHWABlockV2):
    """
    Segmentation-oriented HWA block.

    The detector still estimates a lesion-centered soft support, but the
    segmentation path receives only a bounded per-modality 1/2/4/8 input prior.
    Strong ROI paste-back, encoder feature correction, cross-modal mixing, and
    logit bias are intentionally excluded.
    """

    encoder_prior_enabled = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_soft_prior_gains()

    def reset_soft_prior_gains(self):
        """Restore mid-strength soft-prior gains after detector warm-starts."""
        with torch.no_grad():
            self.input_agg_gain.fill_(0.26)
            self.input_gate_boost.fill_(1.10)
            self.center_prior_gain.fill_(0.05)
            self.input_enhance_gain.fill_(0.0)
            self.output_logit_gain.fill_(0.0)

    def build_input_aggregate(self, x: torch.Tensor, enhance_map=None):
        aggregate = super().build_input_aggregate(x, enhance_map)
        if enhance_map is None:
            return aggregate
        if enhance_map.shape[2:] != x.shape[2:]:
            enhance_map = self._resize(enhance_map.float(), x.shape[2:])
        enhance_map = enhance_map.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
        if enhance_map.shape[1] != 1:
            enhance_map = enhance_map.mean(dim=1, keepdim=True)

        reliability = self._input_prior_reliability(x, enhance_map)
        self.last_input_prior_reliability = reliability
        local_support_floor = 0.08
        soft_support = (
            local_support_floor
            + (1.0 - local_support_floor)
            * reliability
            * torch.sqrt(enhance_map.clamp_min(1e-6))
        ).clamp(local_support_floor, 1.0)
        aggregate = x + soft_support * (aggregate - x)
        self.last_input_aggregate = aggregate
        return aggregate


class HWAUNETRSoftPriorV2(HWAUNETRCenterPriorV2):
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
        hwa_block = kwargs.pop("fussion", hwa_block)
        super().__init__(
            in_chans=in_chans,
            out_chans=out_chans,
            hwa_block=hwa_block,
            kernel_sizes=kernel_sizes,
            depths=depths,
            dims=dims,
            heads=heads,
            hidden_size=hidden_size,
            num_slices_list=num_slices_list,
            out_indices=out_indices,
            hwa_det_channels=hwa_det_channels,
            hwa_softargmax_beta=hwa_softargmax_beta,
            hwa_evidence_temperature=hwa_evidence_temperature,
            hwa_peak_kernel_size=hwa_peak_kernel_size,
            hwa_peak_gate_gain=hwa_peak_gate_gain,
            hwa_peak_gate_bias=hwa_peak_gate_bias,
            hwa_stage_scope_factors=hwa_stage_scope_factors,
            use_hwa_prior_in_encoder=use_hwa_prior_in_encoder,
            **kwargs,
        )
        self.hwa_block = PGHWABlockV2(
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

    def _apply_center_prior_input(self, x: torch.Tensor) -> torch.Tensor:
        gate_scale = float(max(0.0, min(1.0, getattr(self, "runtime_hwa_gate_scale", 1.0))))
        gain_scale = float(max(0.0, min(1.0, getattr(self, "runtime_hwa_gain_scale", 1.0))))
        if not self.use_hwa_prior_in_encoder or gate_scale <= 0.0 or gain_scale <= 0.0:
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

        gain_max = float(max(0.0, getattr(self, "hwa_input_agg_gain_max", 0.55)))
        gain = torch.clamp(
            self.hwa_block.input_agg_gain.to(device=x.device, dtype=x.dtype),
            0.0,
            gain_max,
        )
        gate_max = float(max(0.0, getattr(self, "hwa_input_gate_scale_max", 1.50)))
        boost = torch.clamp(
            self.hwa_block.input_gate_boost.to(device=x.device, dtype=x.dtype),
            0.0,
            gate_max,
        )
        delta = input_aggregate.to(device=x.device, dtype=x.dtype) - x
        raw_std = x.std(dim=(2, 3, 4), keepdim=True, unbiased=False).clamp_min(1e-5)
        delta_clip = float(max(1.0e-3, getattr(self, "hwa_input_delta_std_clip", 1.0)))
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
            out = out * (
                1.0
                + gate_scale
                * gain_scale
                * boost
                * enhance_gain
                * torch.sqrt(enhance_map.clamp_min(1e-6))
            )
        return out + gate_scale * gain_scale * boost * gain * delta
