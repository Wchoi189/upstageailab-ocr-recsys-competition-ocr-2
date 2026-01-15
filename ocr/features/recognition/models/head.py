import torch.nn as nn
from ocr.core.interfaces.models import BaseHead

class PARSeqHead(BaseHead):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__(in_channels=in_channels)
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x, return_loss=False):
        # x is [B, T, D]
        return self.fc(x)


