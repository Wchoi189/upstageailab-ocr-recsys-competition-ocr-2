"""Shared layer components for OCR models."""

import torch.nn as nn


def conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> nn.Sequential:
    """Create a Sequential container with Conv2d, BatchNorm2d, and ReLU.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to all four sides of the input.

    Returns:
        nn.Sequential container with Conv2d, BatchNorm2d, and ReLU layers.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
