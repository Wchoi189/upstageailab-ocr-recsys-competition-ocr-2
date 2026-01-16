import torch.nn as nn

class SVTRMerging(nn.Module):
    """
    Converts SVTR 2D feature maps [B, C, H, W] to 1D sequence [B, W, C].

    This layer performs the critical 'Merging' step in SVTR where the vertical dimension
    is pooled to create a 1D text feature sequence, suitable for CTC or Transformer Decoders.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Pool the Height (H) dimension completely to get a 1D strip
        # AdaptiveAvgPool2d((1, None)) keeps W but forces H=1
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]

        Returns:
            Sequence tensor [B, W, out_channels]
        """
        # x: [B, C, H, W]
        x = self.pool(x) # -> [B, C, 1, W]
        x = x.squeeze(2).permute(0, 2, 1) # -> [B, W, C]

        # Projection and Norm
        x = self.proj(x)
        x = self.norm(x)

        return x
