"""
Two-Stream Transformer Decoder (XLNet-style)

Extracted from original PARSeq implementation to support separate query and content streams.
Required for proper autoregressive decoding with position-specific queries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from torch.nn.modules import transformer


class TwoStreamDecoderLayer(nn.Module):
    """Transformer decoder layer with two-stream attention (XLNet-style).

    Pre-LN architecture with separate query and content streams.

    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        dim_feedforward: FFN dimension
        dropout: Dropout rate
        activation: Activation function name
        layer_norm_eps: LayerNorm epsilon
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def forward_stream(
        self,
        tgt: Tensor,
        tgt_norm: Tensor,
        tgt_kv: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor],
        tgt_key_padding_mask: Optional[Tensor],
    ):
        """Forward pass for a single stream (query or content).

        Args:
            tgt: Stream input [B, L, D]
            tgt_norm: LayerNorm'd tgt (for efficiency)
            tgt_kv: Key/value stream [B, L, D] (LayerNorm'd)
            memory: Encoder output [B, S, D] (LayerNorm'd)
            tgt_mask: Self-attention mask [L, L]
            tgt_key_padding_mask: Padding mask [B, L]

        Returns:
            Updated stream [B, L, D]
        """
        # Self-attention: query attends to content
        tgt2, _ = self.self_attn(
            tgt_norm, tgt_kv, tgt_kv,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)

        # Cross-attention: attend to encoder memory
        tgt2, _ = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)

        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

    def forward(
        self,
        query: Tensor,
        content: Tensor,
        memory: Tensor,
        query_mask: Optional[Tensor] = None,
        content_mask: Optional[Tensor] = None,
        content_key_padding_mask: Optional[Tensor] = None,
        update_content: bool = True,
    ):
        """Two-stream forward pass.

        Args:
            query: Query stream [B, L, D] (position-specific)
            content: Content stream [B, L, D] (token embeddings)
            memory: Encoder output [B, S, D]
            query_mask: Query attention mask [L, L]
            content_mask: Content attention mask [L, L]
            content_key_padding_mask: Padding mask [B, L]
            update_content: Whether to update content stream

        Returns:
            (query, content) updated streams
        """
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)

        # Update query stream (always)
        query = self.forward_stream(
            query, query_norm, content_norm, memory,
            query_mask, content_key_padding_mask
        )

        # Update content stream (optional, disabled in last layer)
        if update_content:
            content = self.forward_stream(
                content, content_norm, content_norm, memory,
                content_mask, content_key_padding_mask
            )

        return query, content


class TwoStreamDecoder(nn.Module):
    """Two-stream Transformer decoder.

    Args:
        decoder_layer: TwoStreamDecoderLayer instance
        num_layers: Number of decoder layers
        norm: Final LayerNorm
    """

    def __init__(self, decoder_layer: TwoStreamDecoderLayer, num_layers: int, norm: nn.Module):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        query: Tensor,
        content: Tensor,
        memory: Tensor,
        query_mask: Optional[Tensor] = None,
        content_mask: Optional[Tensor] = None,
        content_key_padding_mask: Optional[Tensor] = None,
    ):
        """Forward pass through all decoder layers.

        Args:
            query: Query stream [B, L, D]
            content: Content stream [B, L, D]
            memory: Encoder output [B, S, D]
            query_mask: Query attention mask [L, L]
            content_mask: Content attention mask [L, L]
            content_key_padding_mask: Padding mask [B, L]

        Returns:
            Final query stream [B, L, D] (after LayerNorm)
        """
        for i, layer in enumerate(self.layers):
            last_layer = (i == len(self.layers) - 1)
            query, content = layer(
                query, content, memory,
                query_mask, content_mask, content_key_padding_mask,
                update_content=not last_layer  # Don't update content in last layer
            )

        return self.norm(query)
