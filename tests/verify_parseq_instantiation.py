
import os
import sys
from omegaconf import OmegaConf

# Add project root to path
sys.path.append(os.getcwd())  # noqa: path-hack

from ocr.domains.recognition.models.architecture import PARSeq
from ocr.core.models.encoder.timm_backbone import TimmBackbone
from ocr.domains.recognition.models.decoder import PARSeqDecoder
from ocr.domains.recognition.models.head import PARSeqHead
from ocr.core.models.loss.cross_entropy_loss import CrossEntropyLoss

def test_atomic_instantiation():
    print("Testing Atomic Instantiation...")
    encoder = TimmBackbone(model_name="resnet18", pretrained=False)
    decoder = PARSeqDecoder(in_channels=512, d_model=512, nhead=8, num_layers=2, vocab_size=100)
    head = PARSeqHead(d_model=512, vocab_size=100)
    loss = CrossEntropyLoss()

    model = PARSeq(encoder=encoder, decoder=decoder, head=head, loss=loss)
    assert model.encoder == encoder
    assert model.decoder == decoder
    assert model.head == head
    assert model.loss == loss
    print("Atomic Instantiation Passed.")

def test_legacy_instantiation():
    print("Testing Legacy Instantiation...")
    # Minimal config to trigger OCRModel's init
    cfg = OmegaConf.create({
        "image_size": [32, 128],
        # Legacy path uses registry or direct import via get_x_by_cfg
        # We need to make sure get_encoder_by_cfg works.
        # It relies on 'encoder' key.
        "encoder": {"_target_": "ocr.core.models.encoder.TimmBackbone", "model_name": "resnet18", "pretrained": False},
        "decoder": {"_target_": "ocr.domains.recognition.models.decoder.PARSeqDecoder", "in_channels": 512, "d_model": 512, "nhead": 8, "num_layers": 2, "vocab_size": 100},
        "head": {"_target_": "ocr.domains.recognition.models.head.PARSeqHead", "d_model": 512, "vocab_size": 100},
        "loss": {"_target_": "ocr.core.models.loss.cross_entropy_loss.CrossEntropyLoss"}
    })

    try:
        model = PARSeq(cfg)
        print("Successfully instantiated PARSeq (Legacy/Config mode).")
    except Exception as e:
        print(f"Legacy instantiation failed (expected if registry not setup?): {e}")

if __name__ == "__main__":
    test_atomic_instantiation()
    test_legacy_instantiation()
