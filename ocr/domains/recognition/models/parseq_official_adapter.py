"""
PARSeq Official Implementation Adapter

This module integrates the official PARSeq implementation from baudm/parseq
with our training pipeline. It uses the core permutation language modeling logic
from the official implementation while adapting to our data format and tokenizer.
"""

import math
from itertools import permutations
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Setup vendor path for strhub imports
import ocr.vendor  # This adds vendor dir to sys.path

# Import official PARSeq components from vendored code
from strhub.models.parseq.model import PARSeq as OfficialPARSeqModel
from strhub.models.parseq.modules import Decoder, DecoderLayer, Encoder, TokenEmbedding


class PARSeqOfficial(nn.Module):
    """
    Adapter for official PARSeq that integrates with our training pipeline.

    This class uses the official PARSeq model architecture and permutation logic
    while adapting to our tokenizer and data format.
    """

    def __init__(
        self,
        num_tokens: int,
        max_label_length: int = 25,
        img_size: tuple = (32, 128),
        patch_size: tuple = (4, 8),
        embed_dim: int = 384,
        enc_num_heads: int = 6,
        enc_mlp_ratio: int = 4,
        enc_depth: int = 12,
        dec_num_heads: int = 12,
        dec_mlp_ratio: int = 4,
        dec_depth: int = 1,
        perm_num: int = 6,
        perm_forward: bool = True,
        perm_mirrored: bool = True,
        decode_ar: bool = True,
        refine_iters: int = 1,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        # Store config
        self.max_label_length = max_label_length
        self.perm_num = perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored

        # Create the official PARSeq model
        self.model = OfficialPARSeqModel(
            num_tokens=num_tokens,
            max_label_length=max_label_length,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            enc_num_heads=enc_num_heads,
            enc_mlp_ratio=enc_mlp_ratio,
            enc_depth=enc_depth,
            dec_num_heads=dec_num_heads,
            dec_mlp_ratio=dec_mlp_ratio,
            dec_depth=dec_depth,
            decode_ar=decode_ar,
            refine_iters=refine_iters,
            dropout=dropout,
        )

        # Permutation/attention mask generation
        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def gen_tgt_perms(self, tgt):
        """
        Generate shared permutations for the whole batch.
        From official PARSeq implementation.
        """
        # We don't permute the position of BOS, we permute EOS separately
        max_num_chars = tgt.shape[1] - 2
        # Special handling for 1-character sequences
        if max_num_chars == 1:
            return torch.arange(3, device=self._device).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=self._device)] if self.perm_forward else []
        # Additional permutations if needed
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
        # For 4-char sequences and shorter, generate all permutations
        if max_num_chars < 5:
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(
                list(permutations(range(max_num_chars), max_num_chars)),
                device=self._device,
            )[selector]
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(perms), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend(
                [torch.randperm(max_num_chars, device=self._device) for _ in range(num_gen_perms - len(perms))]
            )
            perms = torch.stack(perms)
        if self.perm_mirrored:
            # Add complementary pairs
            comp = perms.flip(-1)
            perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        # Add position indices of BOS and EOS
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        # Special handling for reverse direction
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self._device)
        return perms

    def generate_attn_masks(self, perm):
        """
        Generate attention masks given a sequence permutation.
        From official PARSeq implementation.
        """
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), dtype=torch.bool, device=self._device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = True
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = True  # mask "self"
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def forward(self, images, text_tokens=None, return_loss=True, **kwargs):
        """
        Forward pass with automatic mode detection.

        Args:
            images: Input images [B, C, H, W]
            text_tokens: Ground truth tokens for training [B, L] (optional)
            return_loss: Whether to compute loss (for validation split)

        Returns:
            If training (text_tokens provided and return_loss=True):
                dict{"loss": scalar, "loss_dict": {"parseq_loss": scalar}}
            If inference (return_loss=False):
                dict{"logits": [B, L, C], "tokens": [B, L]}
        """
        if text_tokens is not None and return_loss:
            # Training mode - compute loss with permutation language modeling
            loss = self.forward_train(images, text_tokens)
            return {
                "loss": loss,
                "loss_dict": {"parseq_loss": loss.detach()},
            }
        else:
            # Inference mode - need to create a minimal tokenizer wrapper
            # For now, return logits directly and let module handle decoding

            # Create a minimal tokenizer-like object for inference
            # Our tokenizer uses same special tokens: BOS=1, EOS=2, PAD=0
            class TokenizerAdapter:
                def __init__(self, vocab_size):
                    self.pad_id = 0
                    self.bos_id = 1
                    self.eos_id = 2
                    self.vocab_size = vocab_size

            tokenizer = TokenizerAdapter(self.model.head.out_features + 2)  # +2 for BOS and PAD
            logits = self.model.forward(tokenizer, images)

            # Greedy decode to get token IDs
            tokens = logits.argmax(dim=-1)

            return {
                "logits": logits,
                "tokens": tokens,
            }

    def forward_train(self, images, text_tokens):
        """
        Training forward pass with permutation language modeling.

        Args:
            images: Input images [B, C, H, W]
            text_tokens: Ground truth tokens [B, L] where tokens are [BOS, char1, ..., charN, EOS, PAD, ...]

        Returns:
            loss: Scalar loss value
        """
        # Encode visual features
        memory = self.model.encode(images)

        # Generate permutations for this batch
        tgt_perms = self.gen_tgt_perms(text_tokens)

        # Prepare target sequences
        tgt_in = text_tokens[:, :-1]  # Input: [BOS, char1, ..., charN, EOS]
        tgt_out = text_tokens[:, 1:]  # Output: [char1, ..., charN, EOS, PAD]

        # Token IDs from our tokenizer (matches official: PAD=0, BOS=1, EOS=2)
        pad_id = 0
        eos_id = 2

        tgt_padding_mask = (tgt_in == pad_id) | (tgt_in == eos_id)

        # Compute loss across all permutations
        loss = 0
        loss_numel = 0
        n = (tgt_out != pad_id).sum().item()

        for i,perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm)
            out = self.model.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
            logits = self.model.head(out).flatten(end_dim=1)
            loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=pad_id)
            loss_numel += n
            # After second iteration, remove [EOS] tokens for succeeding perms
            if i == 1:
                tgt_out = torch.where(tgt_out == eos_id, torch.tensor(pad_id, device=tgt_out.device), tgt_out)
                n = (tgt_out != pad_id).sum().item()

        loss /= loss_numel
        return loss
