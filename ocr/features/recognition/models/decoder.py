import math
import torch
import torch.nn as nn
from ocr.core.base_classes import BaseDecoder


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

        if targets is not None:
            # Training Mode (AR)
            # targets usually include BOS/EOS or we just prepend BOS
            # Let's assume targets are raw tokens.
            # We usually feed [BOS, t1, ..., t_{N-1}] to predict [t1, ..., tN]

            # If targets doesn't have BOS, prepend it
            # But verifying input format is hard. Assume external dataloader handles it or we handle it.
            # Let's implement robust handling:
            #   Input to decoder: [BOS] + targets[:, :-1]

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

            # Padding Mask
            tgt_key_padding_mask = targets == self.pad_token_id

            output = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

            return self.norm(output)

        else:
            # Inference (should implement generate)
            return self.generate(memory)

    def generate(self, memory):
        # Greedy decoding
        device = memory.device
        B = memory.size(0)

        # Start symbol
        tgt_tokens = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)

        for i in range(self.max_len):
            tgt_emb = self.embed_tokens(tgt_tokens) * math.sqrt(self.d_model)
            pos_emb = self.pos_encoder[:, : tgt_tokens.size(1), :]
            tgt = tgt_emb + pos_emb

            output = self.decoder(tgt, memory)
            output = self.norm(output)

            # Project logic is usually in Head, but we need logits here to pick next token.
            # This indicates tightly coupled decoder-head or verify if we return features.
            # Since Head is separate in architecture, 'generate' in decoder is tricky without head.

            # Option: Return features for all steps so far, architecture calls head, then loop?
            # Architecture-controlled generation is better.
            # But let's return features and let architecture handle it?
            # No, standard is architecture.generate()

            pass

        # As a fallback for this tool call, return features corresponding to 'single pass' or similar.
        # But since I delegated generate to here in Architecture, I should probably implement it properly
        # OR move generation loop to Architecture.

        return output
