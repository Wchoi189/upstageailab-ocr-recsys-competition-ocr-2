import math
import torch
import torch.nn as nn
from ocr.core.interfaces.models import BaseDecoder
from ocr.domains.recognition.models.two_stream_decoder import TwoStreamDecoderLayer, TwoStreamDecoder


class PARSeqDecoder(BaseDecoder):
    """
    Transformer Decoder capable of Autoregressive decoding and PLM training.

    Supports two training modes:
    - Standard AR: Causal autoregressive decoding (when plm_config=None)
    - PLM: Permutation Language Modeling (when plm_config provided)
    """

    def __init__(
        self,
        in_channels,  # From BaseDecoder signature
        d_model=384,
        nhead=12,
        num_layers=12,
        dim_feedforward=1536,
        dropout=0.1,
        vocab_size=None,
        max_len=25,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        plm_config=None,  # PLM configuration
        use_flash_attention=False,  # NEW: Flash Attention support
        **kwargs,  # Accept extra kwargs
    ):
        if vocab_size is None:
            raise ValueError("PARSeqDecoder requires 'vocab_size' to be specified.")
        super().__init__(in_channels=in_channels)
        self.d_model = d_model
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.use_flash_attention = use_flash_attention

        # Two-Stream Decoder (like original PARSeq)
        decoder_layer = TwoStreamDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu"
        )
        self.decoder = TwoStreamDecoder(decoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))

        # Positional queries (learned, like original PARSeq)
        self.pos_queries = nn.Parameter(torch.zeros(1, max_len + 1, d_model))

        # Token Embeddings
        self.embed_tokens = nn.Embedding(vocab_size, d_model)

        # Input projection: project encoder features (in_channels) to decoder dimension (d_model)
        # This is necessary when encoder output != d_model
        self.input_proj = nn.Linear(in_channels, d_model) if in_channels != d_model else nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # PLM integration (after other components initialized)
        self.plm = None
        if plm_config is not None:
            from ocr.domains.recognition.models.plm import PermutationLanguageModeling
            # Set device to 'cpu' initially, will be moved with model.to(device)
            plm_config_with_device = {**plm_config, 'device': 'cpu'}
            self.plm = PermutationLanguageModeling(**plm_config_with_device)

        self._init_weights()

    def _init_weights(self):
        # Initialize like original PARSeq
        nn.init.xavier_uniform_(self.embed_tokens.weight)
        # Positional queries initialized with small std
        nn.init.trunc_normal_(self.pos_queries, std=0.02)

    @property
    def out_channels(self) -> int:
        return self.d_model

    def to(self, *args, **kwargs):
        """Override to() to move PLM module to the correct device."""
        super().to(*args, **kwargs)
        if self.plm is not None:
            # Extract device from args/kwargs
            device = None
            if args:
                if isinstance(args[0], torch.device):
                    device = str(args[0])
                elif isinstance(args[0], str):
                    device = args[0]
            if device is None and 'device' in kwargs:
                device = str(kwargs['device'])
            if device is not None:
                self.plm.to(device)
        return self

    def forward(self, features, targets=None, memory_key_padding_mask=None,
                tgt_mask=None, tgt_query_mask=None, **kwargs):
        """
        Args:
            features: List of feature tensors from encoder OR pre-flattened memory [B, S, D].
                      If list, we assume it's from TimmBackbone and process it.
            targets: [B, T] Token indices
            memory_key_padding_mask: [B, S] Boolean mask (True = ignore, False = attend).
                                      If None, assumes all visual tokens are valid.
            tgt_mask: Optional [T, T] attention mask for PLM training (overrides causal mask)
            tgt_query_mask: Optional [T, T] query mask for PLM training
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
        B, S, C = memory.shape

        # Project encoder features to decoder dimension
        memory = self.input_proj(memory)  # [B, S, in_channels] -> [B, S, d_model]

        # Generate default memory mask if not provided
        # Default: all visual tokens are valid (no padding)
        if memory_key_padding_mask is None:
            memory_key_padding_mask = torch.zeros(B, S, dtype=torch.bool, device=device)

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

        # Two-stream architecture (like original PARSeq)
        # Content stream: token embeddings with position info
        # Query stream: positional queries only

        # Scale token embeddings
        scaled_emb = self.embed_tokens(targets) * math.sqrt(self.d_model)

        # BOS token: just token embedding (null context)
        null_ctx = scaled_emb[:, :1]

        # BUGFIX (BUG-001): Shift content stream to prevent information leakage
        # Original: content[i] = pos_queries[i-1] + embed(token[i])
        # Fixed: content[i] = pos_queries[i] + embed(token[i-1])
        # This ensures Query[i] cannot see token[i] when predicting next token
        if T > 1:
            # Shift embeddings: use tokens [0:T-1] instead of [1:T]
            # Use positions [1:T] to maintain proper positional encoding alignment
            tgt_emb_rest = self.pos_queries[:, 1:T] + scaled_emb[:, :T-1]
            tgt_emb = torch.cat([null_ctx, tgt_emb_rest], dim=1)
        else:
            tgt_emb = null_ctx

        # Apply dropout to content stream
        tgt_emb = self.dropout(tgt_emb)

        # Query stream: just positional queries
        tgt_query = self.pos_queries[:, :T].expand(B, -1, -1)
        tgt_query = self.dropout(tgt_query)

        # Attention masks
        if tgt_mask is None and tgt_query_mask is None:
            # Standard causal mask for AR decoding
            causal_mask = torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), 1)
            tgt_mask = causal_mask
            tgt_query_mask = causal_mask

        # Padding mask
        tgt_key_padding_mask = (targets == self.pad_token_id)

        # Two-stream decoder forward
        output = self.decoder(
            query=tgt_query,
            content=tgt_emb,
            memory=memory,
            query_mask=tgt_query_mask,
            content_mask=tgt_mask,
            content_key_padding_mask=tgt_key_padding_mask
        )

        return output
