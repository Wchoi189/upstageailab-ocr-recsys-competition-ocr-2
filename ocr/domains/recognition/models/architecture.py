import torch
import math
from omegaconf import DictConfig
from ocr.core.models.architecture import OCRModel




class PARSeq(OCRModel):
    """
    PARSeq Architecture (approximated).

    This class orchestrates the Encoder (ViT/ResNet), Decoder (Transformer), and Head.
    It currently supports standard Autoregressive (AR) training.
    """
    def __init__(
        self,
        cfg=None,
        encoder=None,
        decoder=None,
        head=None,
        loss=None,
        **kwargs
    ):
        # Handle Atomic Instantiation where 'cfg' might be passed as kwargs or object
        if cfg is None:
            if encoder is not None:
                # Atomic Instantiation via Hydra: create minimal config from kwargs
                # This supports instantiation where components are passed directly
                cfg = DictConfig(kwargs)
            else:
                raise ValueError("PARSeq requires a valid 'cfg' argument. Defaults are no longer supported.")

        if encoder:
            # Atomic Mode: Bypass OCRModel.__init__ component loading
            # We must manually call nn.Module's init
            super(OCRModel, self).__init__()
            self.cfg = cfg
            self.encoder = encoder
            self.decoder = decoder
            self.head = head
            self.loss = loss

            # Encoder Positional Embedding (Atomic)
            self.encoder_pos_embed = torch.nn.Parameter(torch.zeros(1, 256, 2, 8))
            torch.nn.init.trunc_normal_(self.encoder_pos_embed, std=0.2)

            # Visual Feature Normalization (FIX: Balance visual/positional scales)
            self.visual_norm = torch.nn.LayerNorm(256)  # Normalize channel dimension

            return

        # Legacy Mode: Rely on OCRModel to load from config
        super().__init__(cfg)

        self.image_size = cfg.get("image_size", [32, 128]) # H, W

        # Encoder Positional Embedding (Legacy)
        self.encoder_pos_embed = torch.nn.Parameter(torch.zeros(1, 256, 2, 8))
        torch.nn.init.trunc_normal_(self.encoder_pos_embed, std=0.2)

        # Visual Feature Normalization (FIX: Balance visual/positional scales)
        self.visual_norm = torch.nn.LayerNorm(256)  # Normalize channel dimension

    def forward(self, images, return_loss=True, **kwargs):
        # 1. Encoder
        # images: [B, C, H, W]
        # DEBUG: Check Input Images
        if True:
             print(f"DEBUG: Input Images: {images.shape}, Mean: {images.mean():.4f}, Std: {images.std():.4f}, Min: {images.min():.4f}, Max: {images.max():.4f}")

        features = self.encoder(images)

        # Handle features structure
        # TimmBackbone returns a list of tensors. We usually want the last one for PARSeq.
        if isinstance(features, (list, tuple)):
            visual_feat = features[-1] # [B, C, H', W']
        else:
            visual_feat = features

        # Flatten visual features for Transformer
        if visual_feat.ndim == 4:
            # CNN output: [B, C, H, W]
            # Add 2D Positional Embeddings
            if not hasattr(self, "encoder_pos_embed"):
                 # Lazy init if not in __init__ (safety fallback)
                 # Ideally should be in __init__ but we need to know C, H, W.
                 # Since C is known (256) and H, W are roughly fixed (2, 8), we can verify.
                 pass

            # [B, C, H, W] -> Extract dimensions for PE generation
            b, c, h, w = visual_feat.shape

            # Use Fixed Sinusoidal 2D PE instead of Learned
            pos_embed = self._generate_2d_sincos_pos_embed(
                h, w, c, device=visual_feat.device
            )

            # FIX: Normalize visual features BEFORE adding positional encoding
            # This balances the signal magnitudes (visual ~1.0 vs pos ~8.0 before)
            # LayerNorm operates on channel dimension [B, C, H, W] -> normalize C
            visual_feat_normalized = visual_feat.permute(0, 2, 3, 1)  # [B, H, W, C]
            visual_feat_normalized = self.visual_norm(visual_feat_normalized)  # Normalize C
            visual_feat = visual_feat_normalized.permute(0, 3, 1, 2)  # Back to [B, C, H, W]

            # Scale Sinusoidal PE to comparable magnitude as normalized visual features
            # visual_feat (normalized) ~ 0.0 mean, 1.0 std
            # pos_embed (raw) * sqrt(256) = ~8.0 mean (TOO LARGE)
            # FIX: Scale down by 0.1 to get ~0.8 mean, comparable to visual
            pos_embed = pos_embed * math.sqrt(c) * 0.1  # Balanced visual/positional signals

            # DEBUG
            if True:
                 print(f"DEBUG: Visual Feat (normalized): {visual_feat.shape}, Mean: {visual_feat.mean():.4f}, Std: {visual_feat.std():.4f}")
                 print(f"DEBUG: Pos Embed (scaled 0.1×): {pos_embed.shape}, Mean: {pos_embed.mean():.4f}, Std: {pos_embed.std():.4f}")

            # Standard Addition
            visual_feat = visual_feat + pos_embed

            # [B, C, H, W] -> [B, H*W, C] -> [B, S, C]
            visual_memory = visual_feat.permute(0, 2, 3, 1).flatten(1, 2) # [B, S, C]
        elif visual_feat.ndim == 3:
            # ViT output: [B, S, C]
            # TIMM ViT includes [CLS] token at index 0. We must remove it for PARSeq.
            visual_memory = visual_feat[:, 1:, :]
        else:
            raise ValueError(f"Unexpected visual features shape: {visual_feat.shape}")

        # 2. Decoder
        # Prepare targets
        targets = kwargs.get("text_tokens", None)

        if return_loss and targets is not None:
            # Training: Forward with targets
            # output: [B, T, D_model]
            # AR Training: Input is targets[:, :-1], Gold is targets[:, 1:]
            tgt_in = targets[:, :-1]
            tgt_out = targets[:, 1:]

            # output: [B, T-1, D_model]
            decoded_output = self.decoder(visual_memory, targets=tgt_in)

            # 3. Head
            logits = self.head(decoded_output) # [B, T, V]

            # 4. Loss
            loss_val = self.loss(logits.permute(0, 2, 1), tgt_out)
            loss_dict = {"loss": loss_val}

            return {
                "logits": logits,
                "loss": loss_val,
                "loss_dict": loss_dict
            }

        else:
            # Inference: Greedy Decoding
            # We explicitly implement the loop here since Architecture owns Encoder, Decoder, and Head.

            device = visual_memory.device
            B = visual_memory.size(0)

            # Start tokens: [B, 1] filled with BOS
            bos_token = self.decoder.bos_token_id
            eos_token = self.decoder.eos_token_id

            tgt_tokens = torch.full((B, 1), bos_token, dtype=torch.long, device=device)

            # Track which samples in the batch have finished
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            # Store logits for each step: list of [B, 1, V]
            logits_list = []

            # Decoding loop
            # Max length limited by decoder.max_len
            max_len = self.decoder.max_len

            for i in range(max_len):
                # Wrapped step
                step_logits = self._decode_step(visual_memory, tgt_tokens) # [B, 1, V]
                logits_list.append(step_logits)

                # Greedy selection
                next_token = step_logits.argmax(dim=-1) # [B, 1]

                # SAFETY: Clamp tokens to valid range [0, vocab_size-1]
                vocab_size = step_logits.size(-1)
                next_token = torch.clamp(next_token, 0, vocab_size - 1)

                # Update finished status
                finished |= (next_token.squeeze(1) == eos_token)

                # Append
                tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)

                # BREAK: If every image in the batch has predicted EOS, stop early
                if finished.all():
                    break

            # Combine logits
            logits = torch.cat(logits_list, dim=1) # [B, T, V]

            return {"logits": logits, "tokens": tgt_tokens}

    def _generate_2d_sincos_pos_embed(self, h, w, embed_dim, device, temperature=10000.0):
        """Generate 2D sin-cos positional embedding."""
        grid_w = torch.arange(w, dtype=torch.float32, device=device)
        grid_h = torch.arange(h, dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='xy')

        assert embed_dim % 2 == 0, 'Embed dimension must be divisible by 2 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 2

        omega = torch.arange(pos_dim // 2, dtype=torch.float32, device=device) / (pos_dim // 2)
        omega = 1. / (temperature**omega)

        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])

        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        # [1, H*W, C] -> [1, C, H, W]
        return pos_emb.permute(0, 2, 1).reshape(1, embed_dim, h, w)

    def _decode_step(self, visual_memory, tokens):
        """
        Helper to perform a single decode step.
        Wraps decoder forward + head projection used in both greedy and beam search.

        Args:
            visual_memory: [B, S, C]
            tokens: [B, T]

        Returns:
            step_logits: [B, 1, V] (Last step logits)
        """
        decoded_output = self.decoder(visual_memory, targets=tokens)
        last_step_output = decoded_output[:, -1:, :] # [B, 1, C]
        step_logits = self.head(last_step_output) # [B, 1, V]
        return step_logits

    @torch.no_grad()
    def beam_search_inference(self, visual_memory, beam_width=3):
        """
        Beam search inference for better accuracy.

        Args:
            visual_memory: [B, S, C] Encoded visual features
            beam_width: Number of beams to keep

        Returns:
            dict: {"tokens": [B, T]} Best token sequence for each image
        """
        device = visual_memory.device
        B = visual_memory.size(0)
        max_len = self.decoder.max_len
        bos_token = self.decoder.bos_token_id

        # 1. Setup initial beams: [B * beam_width, 1]
        # We expand the visual memory to match the beam width
        # [B, S, C] -> [B * beam_width, S, C]
        visual_memory_expanded = visual_memory.repeat_interleave(beam_width, dim=0)

        tgt_tokens = torch.full((B * beam_width, 1), bos_token, dtype=torch.long, device=device)
        beam_scores = torch.zeros(B * beam_width, device=device)
        # Mask out all beams except the first one for each batch item at step 0
        beam_scores.view(B, beam_width)[:, 1:] = -1e9

        for i in range(max_len):
            # Decoder forward pass using unified step
            # Note: _decode_step wraps head() too.
            # Output: [B*K, 1, V]
            # Since _decode_step does the head projection, we just take it.
            # But wait, our _decode_step expects (visual_memory, tokens)

            # Using unified helper
            logits = self._decode_step(visual_memory_expanded, tgt_tokens) # [B*K, 1, V]
            log_probs = torch.log_softmax(logits.squeeze(1), dim=-1) # [B*K, V]

            # Calculate scores for all possible next tokens
            # Current score + new log probability
            vocab_size = log_probs.size(-1)
            next_scores = beam_scores.unsqueeze(1) + log_probs # [B*K, V]

            # Flatten to find the top K across the whole vocabulary for each batch item
            next_scores = next_scores.view(B, beam_width * vocab_size)
            topk_scores, topk_indices = next_scores.topk(beam_width, dim=1)

            # Map indices back to beam index and token index
            beam_indices = topk_indices // vocab_size  # Which beam did it come from?
            token_indices = topk_indices % vocab_size  # Which character is it?

            # We need to reconstruct the sequence for each batch item
            # Vectorized implementation of re-arranging
            # Calculate batch offsets
            batch_offsets = torch.arange(B, device=device) * beam_width
            batch_offsets = batch_offsets.unsqueeze(1).repeat(1, beam_width).flatten() # [B*K]

            # Adjust beam_indices to be global indices
            global_beam_indices = beam_indices.flatten() + batch_offsets

            # Select the winning beams (previous tokens)
            selected_sequences = tgt_tokens[global_beam_indices]

            # Helper for token appending
            selected_new_tokens = token_indices.flatten().unsqueeze(1)

            # Update tgt_tokens
            tgt_tokens = torch.cat([selected_sequences, selected_new_tokens], dim=1)

            # Update scores
            beam_scores = topk_scores.flatten()

        # At the end, just pick the top beam for each batch
        # buffer is [B*K, T]. view as [B, K, T]
        best_beams = tgt_tokens.view(B, beam_width, -1)[:, 0, :]

        return {"tokens": best_beams}


