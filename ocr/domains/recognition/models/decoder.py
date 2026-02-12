import math
import torch
import torch.nn as nn
from ocr.core.interfaces.models import BaseDecoder


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
        plm_config=None,  # NEW: PLM configuration
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

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="gelu", batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Positional Embeddings
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len + 1, d_model))

        # Token Embeddings
        self.embed_tokens = nn.Embedding(vocab_size, d_model)

        # Input projection: project encoder features (in_channels) to decoder dimension (d_model)
        # This is necessary when encoder output != d_model
        self.input_proj = nn.Linear(in_channels, d_model) if in_channels != d_model else nn.Identity()

        # Normalization
        self.norm = nn.LayerNorm(d_model)

        # PLM integration (after other components initialized)
        self.plm = None
        if plm_config is not None:
            from ocr.domains.recognition.models.plm import PermutationLanguageModeling
            # Set device to 'cpu' initially, will be moved with model.to(device)
            plm_config_with_device = {**plm_config, 'device': 'cpu'}
            self.plm = PermutationLanguageModeling(**plm_config_with_device)

        self._init_weights()

    def _init_weights(self):
        # FIX: standard transformer initialization (Xavier) works better for Post-Norm
        # trunc_normal(std=0.02) is too small and causes vanishing gradients without warmup
        nn.init.xavier_uniform_(self.embed_tokens.weight)

        # Init Pos Encoder with Sinusoidal
        # self.pos_encoder: [1, max_len, d_model]
        max_len = self.max_len + 1 # Account for the +1 in pos_encoder definition
        d_model = self.d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        with torch.no_grad():
            self.pos_encoder.copy_(pe.unsqueeze(0))

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

        # Create input sequence
        # Usually input is BOS + targets (excluding EOS if present at end? or just length constraint)
        # Simplified: Use targets directly as input (assuming it starts with BOS)
        tgt_emb = self.embed_tokens(targets) * math.sqrt(self.d_model)

        # Add positional encoding
        # Use T positions
        # FIX: Scale pos_emb too, or use Sinusoidal.
        # If we use Learned with Xavier, it must be scaled to match tgt_emb.
        pos_emb = self.pos_encoder[:, :T, :] * math.sqrt(self.d_model)
        tgt = tgt_emb + pos_emb

        # Attention Masks
        # Use custom masks if provided (for PLM), otherwise use causal mask
        if tgt_mask is None:
            # Standard causal mask for AR decoding
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        # Padding Mask (Boolean: True = Ignore, False = Keep)
        # Using boolean mask is often more stable with Flash Attention backends
        tgt_key_padding_mask = (targets == self.pad_token_id)

        output = self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask  # FIX: Prevent attention to padded visual features
        )

        return self.norm(output)
