import torch
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
        # Handle Atomic Instantiation where 'cfg' might be missing or minimal
        if cfg is None:
            # Create a DictConfig from kwargs if needed, or empty
            # We ensure "architectures" key exists to avoid OCRModel registry lookup if components are passed
            cfg_dict = kwargs.copy()
            if encoder is not None:
                # Mark as 'atomic' in a way OCRModel respects?
                # Actually, OCRModel checks self.architecture_name.
                pass
            cfg = DictConfig(cfg_dict)

        if encoder:
            # Atomic Mode: Bypass OCRModel.__init__ component loading
            # We must manually call nn.Module's init
            super(OCRModel, self).__init__()
            self.cfg = cfg
            self.encoder = encoder
            self.decoder = decoder
            self.head = head
            self.loss = loss
            return

        # Legacy Mode: Rely on OCRModel to load from config
        super().__init__(cfg)

        self.image_size = cfg.get("image_size", [32, 128]) # H, W

    def forward(self, images, return_loss=True, **kwargs):
        # 1. Encoder
        # images: [B, C, H, W]
        features = self.encoder(images)

        # Handle features structure
        # TimmBackbone returns a list of tensors. We usually want the last one for PARSeq.
        if isinstance(features, (list, tuple)):
            visual_feat = features[-1] # [B, C, H', W']
        else:
            visual_feat = features

        # Flatten visual features for Transformer
        if visual_feat.ndim == 4:
            # CNN output: [B, C, H, W] -> [B, H*W, C] -> [B, S, C]
            b, c, h, w = visual_feat.shape
            visual_memory = visual_feat.permute(0, 2, 3, 1).flatten(1, 2) # [B, S, C]
        elif visual_feat.ndim == 3:
            # ViT output: [B, S, C]
            # TIMM ViT includes [CLS] token at index 0. We must remove it for PARSeq.
            visual_memory = visual_feat[:, 1:, :]
            # DEBUG: Confirm shape
            if torch.rand(1).item() < 0.001:
                print(f"[DEBUG] Visual Memory Shape (After CLS Removal): {visual_memory.shape}")
        else:
            raise ValueError(f"Unexpected visual features shape: {visual_feat.shape}")

        # DEBUG: Check if features are dead
        if torch.rand(1).item() < 0.01: # 1% chance to print (or first batch if we could track it)
             pass
             # We rely on the lightning module loop for printing mainly,
             # preventing spam here. BUT, let's print once if mean is suspicious.

        if visual_memory.abs().mean() < 1e-6:
             print(f"[WARNING] Visual Memory seems dead! Mean: {visual_memory.abs().mean().item()}")

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
            # Target for loss usually excludes BOS (if input included it) or depends on shift
            # Providing loss calculation here or determining it via self.loss

            # Often, we pass logits and targets to self.loss
            # Typically targets need simple alignment.
            # If decoder inputs included [BOS, t1, ... tn], output corresponds to [t1, ... tn, EOS]

            # Let's assume prediction aligns with targets for now
            loss_val, loss_dict = self.loss(logits, tgt_out)

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
                # Decoder Forward
                # tgt_tokens: [B, current_len]
                # visual_memory: [B, S, C]
                # Output: [B, current_len, C]
                decoded_output = self.decoder(visual_memory, targets=tgt_tokens)

                # We only care about the last token's output for prediction
                last_step_output = decoded_output[:, -1:, :] # [B, 1, C]

                # Head: Project to logits
                step_logits = self.head(last_step_output) # [B, 1, V]
                logits_list.append(step_logits)

                # Greedy selection
                next_token = step_logits.argmax(dim=-1) # [B, 1]

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
        # to avoid starting with 3 identical beams.
        beam_scores.view(B, beam_width)[:, 1:] = -1e9

        # finished_beams = [[] for _ in range(B)] # TODO: Handle finished beams more strictly if needed

        for i in range(max_len):
            # Decoder forward pass
            decoded_output = self.decoder(visual_memory_expanded, targets=tgt_tokens)
            logits = self.head(decoded_output[:, -1:, :]) # [B*K, 1, V]
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

            # Re-arrange tgt_tokens and beam_scores based on topk
            # new_tokens = []
            # new_scores = []

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

            # (Optional: Add EOS check here to stop finished beams and move to finished_beams list)
            # For simplicity, we run for max_len or could optimize similar to greedy

        # At the end, just pick the top beam for each batch
        # buffer is [B*K, T]. view as [B, K, T]
        best_beams = tgt_tokens.view(B, beam_width, -1)[:, 0, :]

        return {"tokens": best_beams}


