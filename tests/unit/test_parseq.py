import pytest
import torch
from unittest.mock import Mock, MagicMock
from ocr.domains.recognition.models.architecture import PARSeq
from omegaconf import OmegaConf

class TestPARSeq:
    @pytest.fixture
    def mock_config(self):
        return OmegaConf.create({
            "image_size": [32, 128],
            "encoder": {},
            "decoder": {},
            "head": {},
            "loss": {}
        })

    @pytest.fixture
    def mock_components(self):
        encoder = MagicMock()
        # Mock encoder output: [B, C, H, W] -> [B, 64, 4, 32]
        encoder.return_value = torch.randn(2, 64, 4, 32)

        decoder = MagicMock()
        decoder.bos_token_id = 0
        decoder.eos_token_id = 1
        decoder.max_len = 10
        # Mock decoder output: [B, T, D]
        decoder.return_value = torch.randn(2, 1, 128)

        head = MagicMock()
        # Mock head output: [B, T, V] where V=100
        head.return_value = torch.randn(2, 1, 100)

        loss = MagicMock()
        loss.return_value = torch.tensor(0.5)

        return encoder, decoder, head, loss

    def test_init_raises_without_config(self):
        """Test that PARSeq raises TypeError or ValueError if cfg is missing/None."""
        # Depending on implementation, might raise TypeError due to missing arg
        # or AttributeError later if it tries to access None.
        # Our plan is to remove default None, so it should raise TypeError for missing arg.
        with pytest.raises(ValueError):
            PARSeq()

    def test_init_with_config(self, mock_config, mock_components):
        encoder, decoder, head, loss = mock_components
        # We need to mock OCRModel init if we want to isolate PARSeq logic,
        # but PARSeq inherits. Ideally we strictly test PARSeq logic.
        # Since PARSeq calls super().__init__(cfg), we need to ensure that doesn't explode.
        # We can pass components directly to bypass super's component loading (Atomic Mode logic we preserved/cleaned).

        model = PARSeq(
            cfg=mock_config,
            encoder=encoder,
            decoder=decoder,
            head=head,
            loss=loss
        )
        assert model.encoder is encoder
        assert model.decoder is decoder

    def test_forward_train(self, mock_config, mock_components):
        encoder, decoder, head, loss = mock_components
        model = PARSeq(
            cfg=mock_config,
            encoder=encoder,
            decoder=decoder,
            head=head,
            loss=loss
        )

        images = torch.randn(2, 3, 32, 128)
        text_tokens = torch.randint(0, 100, (2, 10))

        # Setup mocks specifically for this call
        # Encoder returns visual features
        encoder.return_value = torch.randn(2, 64, 4, 32)
        # Decoder returns features [B, T-1, D]
        decoder.return_value = torch.randn(2, 9, 128)
        # Head returns logits [B, T-1, V]
        head.return_value = torch.randn(2, 9, 100)

        out = model(images, return_loss=True, text_tokens=text_tokens)

        assert "loss" in out
        assert "logits" in out
        # Verify decoder called with target tokens (shifted)
        # We don't strictly check args here, just flow

    def test_forward_inference_greedy(self, mock_config, mock_components):
        encoder, decoder, head, loss = mock_components
        model = PARSeq(
            cfg=mock_config,
            encoder=encoder,
            decoder=decoder,
            head=head,
            loss=loss
        )

        images = torch.randn(2, 3, 32, 128)

        # Mock head to return predictable logits to avoid infinite loop or random eos
        # We want to ensure it stops eventually.
        # Let's mock return value of head to have specific shape
        head.return_value = torch.randn(2, 1, 100)

        out = model(images, return_loss=False)

        assert "logits" in out
        assert "tokens" in out
        assert out["tokens"].shape[0] == 2
        # Should start with BOS
        assert (out["tokens"][:, 0] == decoder.bos_token_id).all()

    def test_beam_search(self, mock_config, mock_components):
        encoder, decoder, head, loss = mock_components
        model = PARSeq(
            cfg=mock_config,
            encoder=encoder,
            decoder=decoder,
            head=head,
            loss=loss
        )

        images = torch.randn(2, 3, 32, 128)

        # Mock visual memory shape [B, S, C]
        encoder.return_value = torch.randn(2, 64, 4, 32)
        # Note: PARSeq forward flatten this. We might need to handle that inside mock or assume architecture does it.
        # Architecture does it.

        # Mock decoder/head for beam search
        # Beam search expands batch size.
        # Input B=2, Beam=3 -> Effective B=6
        decoder.return_value = torch.randn(6, 1, 128)
        head.return_value = torch.randn(6, 1, 100)

        out = model.beam_search_inference(model.encoder(images), beam_width=3)

        assert "tokens" in out
        assert out["tokens"].shape[0] == 2
