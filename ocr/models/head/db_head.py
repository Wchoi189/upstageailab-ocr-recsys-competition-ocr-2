"""
*****************************************************************************************
* Modified from https://github.com/MhLiao/DB/blob/master/decoders/seg_detector.py
*
* 참고 논문:
* Real-time Scene Text Detection with Differentiable Binarization
* https://arxiv.org/pdf/1911.08947.pdf
*
* 참고 Repository:
* https://github.com/MhLiao/DB/
*****************************************************************************************
"""

import math

import torch
import torch.nn as nn

from ..core import BaseHead
from .db_postprocess import DBPostProcessor


class DBHead(BaseHead):
    """DBNet head for text detection with differentiable binarization.

    Implements the DBNet prediction head that produces probability maps,
    threshold maps, and binary maps for text detection.
    """

    def __init__(
        self,
        in_channels: int,
        upscale: int = 4,
        k: int = 50,
        bias: bool = False,
        smooth: bool = False,
        postprocess: dict | None = None,
        **kwargs,
    ):
        """Initialize the DB head.

        Args:
            in_channels: Number of input channels from decoder
            upscale: Upscaling factor for output maps
            k: Steepness parameter for step function
            bias: Whether to use bias in convolutions
            smooth: Whether to use smooth upsampling
            postprocess: Configuration for postprocessing
            **kwargs: Additional arguments
        """
        super().__init__(in_channels=in_channels, **kwargs)

        if postprocess is None:
            postprocess = {}
        self.postprocess = DBPostProcessor(**postprocess)

        self.inner_channels = in_channels // 4
        self.k = k
        self.upscale = int(math.log2(upscale))

        # Output of Probability map
        # Upscale에 따라 ConvTranspose2d Layer를 동적으로 생성
        binarize_layers = [
            nn.Conv2d(
                self.in_channels,
                self.inner_channels,
                kernel_size=3,
                padding=1,
                bias=bias,
            ),
            nn.BatchNorm2d(self.inner_channels),
            nn.ReLU(inplace=True),
        ]
        for i in range(self.upscale):
            if i == self.upscale - 1:
                binarize_layers.append(nn.ConvTranspose2d(self.inner_channels, 1, 2, 2))
            else:
                binarize_layers.append(nn.ConvTranspose2d(self.inner_channels, self.inner_channels, 2, 2))
                binarize_layers.append(nn.BatchNorm2d(self.inner_channels))
                binarize_layers.append(nn.ReLU(inplace=True))
        self.binarize = nn.Sequential(*binarize_layers)
        self.binarize.apply(self.weights_init)
        self.prob_activation = nn.Sigmoid()

        # Output of Threshold map
        self.thresh = self._init_thresh(smooth=smooth, bias=bias)
        self.thresh.apply(self.weights_init)

    def weights_init(self, m: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(m, nn.Conv2d | nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(1e-4)
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                m.weight.data.fill_(1.0)
            if m.bias is not None:
                m.bias.data.fill_(1e-4)

    def _init_thresh(self, smooth=False, bias=False):
        # Upscale에 따라 Upsample Layer를 동적으로 생성
        thresh_layers = [
            nn.Conv2d(
                self.in_channels,
                self.inner_channels,
                kernel_size=3,
                padding=1,
                bias=bias,
            ),
            nn.BatchNorm2d(self.inner_channels),
            nn.ReLU(inplace=True),
        ]
        for i in range(self.upscale):
            if i == self.upscale - 1:
                thresh_layers.append(self._init_upsample(self.inner_channels, out_channels=1, smooth=smooth, bias=bias))
            else:
                thresh_layers.append(
                    self._init_upsample(
                        self.inner_channels,
                        self.inner_channels,
                        smooth=smooth,
                        bias=bias,
                    )
                )
                thresh_layers.append(nn.BatchNorm2d(self.inner_channels))
                thresh_layers.append(nn.ReLU(inplace=True))
        thresh_layers.append(nn.Sigmoid())
        thresh = nn.Sequential(*thresh_layers)

        return thresh

    def _init_upsample(self, in_channels, out_channels, smooth=False, bias=False):
        # Smooth 가 True인 경우, ConvTranspose2d 대신 Upsample을 사용
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, inter_out_channels, 3, padding=1, bias=bias),
            ]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    )
                )
            return nn.Sequential(*module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def _step_function(self, x, y):
        """
        Differentiable step function for binarization.

        BUG-20251110-002: Fixed numerical instability causing NaN gradients.
        Original: torch.reciprocal(1 + torch.exp(-k * (x - y))) with k=50
        - Caused exp overflow when x - y is negative (e.g., exp(50) ≈ 5e21)
        - Led to NaN gradients propagating through backprop at step ~122

        Fix: Use torch.sigmoid which is mathematically equivalent but numerically stable.
        sigmoid(k*z) = 1 / (1 + exp(-k*z)) but with built-in overflow protection.

        See: docs/bug_reports/BUG-20251110-002-nan-gradients-from-step-function-overflow.md

        Args:
            x: Probability maps from sigmoid, range [0, 1], shape (B, 1, H, W)
            y: Threshold maps from sigmoid, range [0, 1], shape (B, 1, H, W)

        Returns:
            Binary map from differentiable binarization, range [0, 1], shape (B, 1, H, W)
        """
        # BUG-20251110-002: Clamp inputs to prevent extreme values
        # Even though x and y are from sigmoid (range [0,1]), numerical errors could produce
        # values slightly outside this range, which get amplified by k=50
        x_clamped = torch.clamp(x, 0.0, 1.0)
        y_clamped = torch.clamp(y, 0.0, 1.0)

        # BUG-20251110-002: Use sigmoid instead of reciprocal + exp for numerical stability
        # This is mathematically equivalent but handles extreme values gracefully
        # sigmoid(k*(x-y)) = 1 / (1 + exp(-k*(x-y)))
        result = torch.sigmoid(self.k * (x_clamped - y_clamped))

        # BUG-20251110-002: Validate output to catch any remaining numerical issues
        # This should never trigger with sigmoid, but acts as a safety check
        if torch.isnan(result).any() or torch.isinf(result).any():
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"NaN/Inf detected in step function output. "
                f"x range: [{x.min().item():.6f}, {x.max().item():.6f}], "
                f"y range: [{y.min().item():.6f}, {y.max().item():.6f}], "
                f"result range: [{result.min().item():.6f}, {result.max().item():.6f}]"
            )
            # Clamp to valid range as fallback
            result = torch.clamp(result, 0.0, 1.0)

        return result

    def forward(self, x: torch.Tensor, return_loss: bool = True) -> dict[str, torch.Tensor]:
        """Produce predictions from decoded features.

        Args:
            x: Input tensor from decoder of shape (B, C, H, W)
            return_loss: Whether to include loss computation in output

        Returns:
            Dictionary containing prediction maps
        """
        # Input feature concat - handle both single tensor and list of tensors
        if isinstance(x, list):
            fuse = torch.cat(x, dim=1)
        else:
            fuse = x

        # Validate input tensor before convolution operations
        # CUDA illegal instruction errors often occur due to invalid input data
        if torch.isnan(fuse).any():
            raise ValueError(f"NaN values detected in input tensor to DBHead.forward(). Shape: {fuse.shape}, Device: {fuse.device}")
        if torch.isinf(fuse).any():
            raise ValueError(f"Inf values detected in input tensor to DBHead.forward(). Shape: {fuse.shape}, Device: {fuse.device}")
        if fuse.numel() == 0:
            raise ValueError(f"Empty tensor passed to DBHead.forward(). Shape: {fuse.shape}")
        if fuse.shape[2] < 1 or fuse.shape[3] < 1:
            raise ValueError(f"Invalid spatial dimensions in input tensor. Shape: {fuse.shape}, expected (B, C, H, W) with H, W >= 1")

        # Probability logits and map (alias prob_maps for downstream post-processing)
        binary_logits = self.binarize(fuse)
        prob_maps = self.prob_activation(binary_logits)

        if return_loss:
            # Threshold map
            thresh = self.thresh(fuse)

            # BUG-20251110-002: Validate thresh map before step function to catch numerical issues early
            if torch.isnan(thresh).any():
                raise ValueError(
                    f"NaN values detected in thresh map. "
                    f"Shape: {thresh.shape}, Device: {thresh.device}, "
                    f"Range: [{thresh.min().item():.6f}, {thresh.max().item():.6f}]"
                )
            if torch.isinf(thresh).any():
                raise ValueError(
                    f"Inf values detected in thresh map. "
                    f"Shape: {thresh.shape}, Device: {thresh.device}, "
                    f"Range: [{thresh.min().item():.6f}, {thresh.max().item():.6f}]"
                )

            # BUG-20251110-002: Validate prob_maps before step function
            if torch.isnan(prob_maps).any():
                raise ValueError(
                    f"NaN values detected in prob_maps. "
                    f"Shape: {prob_maps.shape}, Device: {prob_maps.device}, "
                    f"Range: [{prob_maps.min().item():.6f}, {prob_maps.max().item():.6f}]"
                )
            if torch.isinf(prob_maps).any():
                raise ValueError(
                    f"Inf values detected in prob_maps. "
                    f"Shape: {prob_maps.shape}, Device: {prob_maps.device}, "
                    f"Range: [{prob_maps.min().item():.6f}, {prob_maps.max().item():.6f}]"
                )

            # Approximate Binary map
            thresh_binary = self._step_function(prob_maps, thresh)
            result = {
                "binary_logits": binary_logits,
                "binary_map": prob_maps,
                "prob_maps": prob_maps,
                "thresh_map": thresh,
                "thresh_binary_map": thresh_binary,
            }
        else:
            # Probability map only - Inference mode
            result = {"binary_map": prob_maps, "prob_maps": prob_maps}

        return result

    def get_polygons_from_maps(self, batch, pred):
        return self.postprocess.represent(batch, pred)
