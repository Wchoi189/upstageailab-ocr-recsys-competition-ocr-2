import torch.nn as nn
from ocr.models.core.base_classes import BaseHead

class PARSeqHead(BaseHead):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__(in_channels=in_channels)
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x, return_loss=False):
        # x is [B, T, D]
        return self.fc(x)

    def get_polygons_from_maps(self, batch, pred):
        # PARSeq is recognition only, no polygons to extract from maps
        # Return empty lists matching batch size
        B = len(batch['image_filename'])
        return [[] for _ in range(B)], [[] for _ in range(B)]
