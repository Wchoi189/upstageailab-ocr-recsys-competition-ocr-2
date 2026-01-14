import torch
from ocr.core.models.architecture import OCRModel
from ocr.core import registry
from ocr.features.recognition.models.decoder import PARSeqDecoder
from ocr.features.recognition.models.head import PARSeqHead
from ocr.core.models.encoder.timm_backbone import TimmBackbone
from ocr.core.models.loss.cross_entropy_loss import CrossEntropyLoss

def register_parseq_components():
    # Register components if not already registered
    try:
        registry.register_encoder("timm_backbone", TimmBackbone)
    except KeyError:
        pass # Already registered? Registry raises? No, dict assignment. But just in case.

    registry.register_decoder("parseq_decoder", PARSeqDecoder)
    registry.register_head("parseq_head", PARSeqHead)
    registry.register_loss("cross_entropy", CrossEntropyLoss)

    registry.register_architecture(
        name="parseq",
        encoder="timm_backbone",
        decoder="parseq_decoder",
        head="parseq_head",
        loss="cross_entropy"
    )

class PARSeq(OCRModel):
    """
    PARSeq Architecture (approximated).

    This class orchestrates the Encoder (ViT/ResNet), Decoder (Transformer), and Head.
    It currently supports standard Autoregressive (AR) training.
    """
    def __init__(self, cfg):
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
            visual_memory = visual_feat
        else:
            raise ValueError(f"Unexpected visual features shape: {visual_feat.shape}")

        # 2. Decoder
        # Prepare targets
        targets = kwargs.get("text_tokens", None)

        if return_loss and targets is not None:
            # Training: Forward with targets
            # output: [B, T, D_model]
            decoded_output = self.decoder(visual_memory, targets=targets)

            # 3. Head
            logits = self.head(decoded_output) # [B, T, V]

            # 4. Loss
            # Target for loss usually excludes BOS (if input included it) or depends on shift
            # Providing loss calculation here or determining it via self.loss

            # Often, we pass logits and targets to self.loss
            # Typically targets need simple alignment.
            # If decoder inputs included [BOS, t1, ... tn], output corresponds to [t1, ... tn, EOS]

            # Let's assume prediction aligns with targets for now
            loss_val, loss_dict = self.loss(logits, targets)

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

            # Start tokens: [B, 1] filled with BOS (1)
            # We assume BOS=1 based on decoder default. Should come from cfg/dataset.
            bos_token = 1
            eos_token = 2
            pad_token = 0

            tgt_tokens = torch.full((B, 1), bos_token, dtype=torch.long, device=device)

            # Decoding loop
            # Max length limited by decoder.max_len
            max_len = self.decoder.max_len

            # Cache could be optimized but we do naive re-forward for simplicity
            for i in range(max_len):
                # Decoder Forward
                # tgt_tokens: [B, current_len]
                # visual_memory: [B, S, C]
                # Output: [B, current_len, C]
                decoded_output = self.decoder(visual_memory, targets=tgt_tokens)

                # We only care about the last token's output for prediction
                last_step_output = decoded_output[:, -1:, :] # [B, 1, C]

                # Head: Project to logits
                logits = self.head(last_step_output) # [B, 1, V]

                # Greedy selection
                next_token = logits.argmax(dim=-1) # [B, 1]

                # Append
                tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)

                # Stop if all finished? (Batch optimization omitted for simplicity)

            return {"logits": self.head(self.decoder(visual_memory, targets=tgt_tokens))}

register_parseq_components()
