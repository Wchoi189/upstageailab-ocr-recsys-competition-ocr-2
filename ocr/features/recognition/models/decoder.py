import math
import torch
import torch.nn as nn
from ocr.core.interfaces.models import BaseDecoder


class PARSeqDecoder(BaseDecoder):
    """
    Transformer Decoder capable of Autoregressive decoding.
    """

    def __init__(
        self,
        in_channels,  # From BaseDecoder signature
        d_model=384,
        nhead=12,
        num_layers=12,
        dim_feedforward=1536,
        dropout=0.1,
        vocab_size=100,
        max_len=25,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,  # Accept extra kwargs
    ):
        super().__init__(in_channels=in_channels)
        self.d_model = d_model
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="gelu", batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Positional Embeddings
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len + 1, d_model))

        # Token Embeddings
        self.embed_tokens = nn.Embedding(vocab_size, d_model)

        # Normalization
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_encoder, std=0.02)
        nn.init.normal_(self.embed_tokens.weight, std=0.02)

    @property
    def out_channels(self) -> int:
        return self.d_model

    def forward(self, features, targets=None, **kwargs):
        """
        Args:
            features: List of feature tensors from encoder OR pre-flattened memory [B, S, D].
                      If list, we assume it's from TimmBackbone and process it.
            targets: [B, T] Token indices
        """
        # Handle BaseDecoder contract: features is list[torch.Tensor]
        if isinstance(features, list):
            # Take the last feature map
            visual_feat = features[-1]  # [B, C, H, W]
            # Flatten: [B, C, H, W] -> [B, S, C]
            memory = visual_feat.permute(0, 2, 3, 1).flatten(1, 2)
        else:
            # Assume it's already processed memory [B, S, C]
            memory = features

        device = memory.device

        if targets is None:
             # If targets are not provided, we cannot perform AR decoding in this module alone.
             # The OCRModel is responsible for the generation loop.
             raise ValueError("PARSeqDecoder requires 'targets' to be passed. "
                              "For inference, use OCRModel.generate() which handles iterative decoding.")

        device = memory.device

        # Training Mode (AR) or One Step of Inference
        # targets usually include BOS/EOS or we just prepend BOS
        # Let's assume targets are raw tokens.

        B, T = targets.shape

        # Create input sequence
        # Usually input is BOS + targets (excluding EOS if present at end? or just length constraint)
        # Simplified: Use targets directly as input (assuming it starts with BOS)
        tgt_emb = self.embed_tokens(targets) * math.sqrt(self.d_model)

        # Add positional encoding
        # Use T positions
        pos_emb = self.pos_encoder[:, :T, :]
        tgt = tgt_emb + pos_emb

        # Causal Mask (Upper triangular)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        # Padding Mask (cast to float to match tgt_mask dtype - fixes PyTorch 2.x deprecation)
        tgt_key_padding_mask = (targets == self.pad_token_id).float()

        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        return self.norm(output)
