"""
Device Placement Tests for PARSeq with PLM and Flash Attention

Tests verify that all tensors remain on the correct device throughout forward/backward passes.
Uses forward hooks to monitor tensor devices at each layer.

Reference: Phase 6.2 - Memory Safety (Device Placement)
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, List


class DeviceTracker:
    """Forward hook to track tensor devices during forward pass."""

    def __init__(self):
        self.device_logs: List[Dict] = []

    def __call__(self, module, input, output):
        """Hook function called during forward pass."""
        entry = {
            "module": module.__class__.__name__,
            "input_devices": [],
            "output_devices": [],
        }

        # Track input devices
        if isinstance(input, tuple):
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor):
                    entry["input_devices"].append((i, inp.device))
        elif isinstance(input, torch.Tensor):
            entry["input_devices"].append((0, input.device))

        # Track output devices
        if isinstance(output, tuple):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    entry["output_devices"].append((i, out.device))
        elif isinstance(output, torch.Tensor):
            entry["output_devices"].append((0, output.device))
        elif isinstance(output, dict):
            for key, val in output.items():
                if isinstance(val, torch.Tensor):
                    entry["output_devices"].append((key, val.device))

        self.device_logs.append(entry)

    def verify_all_cuda(self, expected_device: torch.device) -> bool:
        """Verify all tensors are on expected CUDA device."""
        mismatches = []

        for log in self.device_logs:
            for idx, device in log["input_devices"]:
                if device != expected_device:
                    mismatches.append({
                        "module": log["module"],
                        "tensor": f"input[{idx}]",
                        "expected": str(expected_device),
                        "actual": str(device),
                    })

            for idx, device in log["output_devices"]:
                if device != expected_device:
                    mismatches.append({
                        "module": log["module"],
                        "tensor": f"output[{idx}]",
                        "expected": str(expected_device),
                        "actual": str(device),
                    })

        if mismatches:
            print("\n❌ Device Mismatches Found:")
            for m in mismatches:
                print(f"  {m['module']}.{m['tensor']}: {m['actual']} != {m['expected']}")
            return False

        return True


def register_device_hooks(model: nn.Module) -> DeviceTracker:
    """Register forward hooks on all modules to track device placement."""
    tracker = DeviceTracker()

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            module.register_forward_hook(tracker)

    return tracker


@pytest.fixture
def device():
    """Get CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def parseq_model(device):
    """Create PARSeq model for testing."""
    from ocr.domains.recognition.models.architecture import PARSeqModel

    model = PARSeqModel(
        d_model=384,
        nhead=12,
        num_layers=12,
        vocab_size=100,
        max_len=25,
        use_flash_attention=False,  # Test without Flash first
        plm_config=None,  # Standard AR mode
    )
    model = model.to(device)
    model.eval()

    return model


@pytest.fixture
def parseq_plm_model(device):
    """Create PARSeq model with PLM for testing."""
    from ocr.domains.recognition.models.architecture import PARSeqModel

    plm_config = {
        "max_label_length": 25,
        "perm_num": 6,
        "perm_forward": True,
        "perm_mirrored": True,
    }

    model = PARSeqModel(
        d_model=384,
        nhead=12,
        num_layers=12,
        vocab_size=100,
        max_len=25,
        use_flash_attention=False,
        plm_config=plm_config,
    )
    model = model.to(device)
    model.eval()

    return model


@pytest.fixture
def parseq_flash_model(device):
    """Create PARSeq model with Flash Attention for testing."""
    from ocr.domains.recognition.models.architecture import PARSeqModel

    model = PARSeqModel(
        d_model=384,
        nhead=12,
        num_layers=12,
        vocab_size=100,
        max_len=25,
        use_flash_attention=True,
        plm_config=None,
    )
    model = model.to(device)
    model.eval()

    return model


@pytest.fixture
def sample_batch(device):
    """Create sample batch for testing."""
    B, C, H, W = 8, 3, 32, 128
    L = 10

    return {
        "images": torch.randn(B, C, H, W, device=device),
        "text_tokens": torch.randint(1, 99, (B, L), device=device),
    }


class TestDevicePlacement:
    """Test suite for device placement verification."""

    def test_baseline_ar_device_consistency(self, parseq_model, sample_batch, device):
        """Test device consistency in baseline autoregressive mode."""
        tracker = register_device_hooks(parseq_model)

        with torch.no_grad():
            output = parseq_model(**sample_batch)

        # Verify all tensors on correct device
        assert tracker.verify_all_cuda(device), "Device mismatch detected in baseline AR"

        # Verify output device
        assert output["loss"].device == device
        for key, val in output["loss_dict"].items():
            assert val.device == device, f"loss_dict[{key}] on wrong device"

    def test_plm_device_consistency(self, parseq_plm_model, sample_batch, device):
        """Test device consistency with PLM training."""
        tracker = register_device_hooks(parseq_plm_model)

        with torch.no_grad():
            output = parseq_plm_model(**sample_batch)

        # Verify all tensors on correct device
        assert tracker.verify_all_cuda(device), "Device mismatch detected in PLM mode"

        # Verify output device
        assert output["loss"].device == device
        for key, val in output["loss_dict"].items():
            assert val.device == device, f"loss_dict[{key}] on wrong device"

    def test_flash_attention_device_consistency(self, parseq_flash_model, sample_batch, device):
        """Test device consistency with Flash Attention."""
        if not torch.cuda.is_available():
            pytest.skip("Flash Attention requires CUDA")

        tracker = register_device_hooks(parseq_flash_model)

        with torch.no_grad():
            output = parseq_flash_model(**sample_batch)

        # Verify all tensors on correct device
        assert tracker.verify_all_cuda(device), "Device mismatch detected in Flash Attention"

        # Verify output device
        assert output["loss"].device == device
        for key, val in output["loss_dict"].items():
            assert val.device == device, f"loss_dict[{key}] on wrong device"

    def test_plm_mask_device_placement(self, parseq_plm_model, device):
        """Test that PLM masks are created on correct device."""
        # Access PLM module
        plm = parseq_plm_model.decoder.plm
        assert plm is not None, "PLM module not found"

        # Generate permutations (should be on device)
        targets = torch.randint(1, 99, (4, 10), device=device)
        perms = plm.gen_tgt_perms(targets)

        assert perms.device == device, f"Permutations on wrong device: {perms.device} != {device}"

        # Generate attention masks
        for perm in perms:
            masks = plm.generate_attn_masks(perm)
            assert masks.content_mask.device == device, "content_mask on wrong device"
            assert masks.query_mask.device == device, "query_mask on wrong device"

    def test_padding_mask_device_consistency(self, parseq_model, sample_batch, device):
        """Test that padding masks are on correct device."""
        # Create padding mask
        targets = sample_batch["text_tokens"]
        padding_mask = (targets == 0)  # Assuming 0 is pad_token_id

        assert padding_mask.device == device, "Padding mask on wrong device"

        # Verify through forward pass
        with torch.no_grad():
            output = parseq_model(**sample_batch)

        assert output["loss"].device == device

    def test_mixed_precision_device_consistency(self, parseq_model, sample_batch, device):
        """Test device consistency with mixed precision (autocast)."""
        if not torch.cuda.is_available():
            pytest.skip("Mixed precision requires CUDA")

        tracker = register_device_hooks(parseq_model)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                output = parseq_model(**sample_batch)

        # Verify all tensors on correct device (dtype may vary)
        assert tracker.verify_all_cuda(device), "Device mismatch in mixed precision"
        assert output["loss"].device == device


class TestDeviceMigration:
    """Test device migration (CPU to CUDA and vice versa)."""

    def test_model_to_cuda(self):
        """Test moving model from CPU to CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from ocr.domains.recognition.models.architecture import PARSeqModel

        plm_config = {
            "max_label_length": 25,
            "perm_num": 6,
        }

        # Create on CPU
        model = PARSeqModel(
            d_model=384,
            nhead=12,
            num_layers=2,  # Small model for testing
            vocab_size=100,
            max_len=25,
            plm_config=plm_config,
        )

        # Move to CUDA
        device = torch.device("cuda")
        model = model.to(device)

        # Verify PLM device was updated
        assert model.decoder.plm.device == "cuda", "PLM device not updated"

        # Test forward pass
        batch = {
            "images": torch.randn(2, 3, 32, 128, device=device),
            "text_tokens": torch.randint(1, 99, (2, 10), device=device),
        }

        with torch.no_grad():
            output = model(**batch)

        assert output["loss"].device == device

    def test_model_to_cpu(self):
        """Test moving model from CUDA to CPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from ocr.domains.recognition.models.architecture import PARSeqModel

        plm_config = {
            "max_label_length": 25,
            "perm_num": 6,
        }

        # Create on CUDA
        device_cuda = torch.device("cuda")
        model = PARSeqModel(
            d_model=384,
            nhead=12,
            num_layers=2,
            vocab_size=100,
            max_len=25,
            plm_config=plm_config,
        )
        model = model.to(device_cuda)

        # Move to CPU
        device_cpu = torch.device("cpu")
        model = model.to(device_cpu)

        # Verify PLM device was updated
        assert model.decoder.plm.device == "cpu", "PLM device not updated to CPU"

        # Test forward pass
        batch = {
            "images": torch.randn(2, 3, 32, 128, device=device_cpu),
            "text_tokens": torch.randint(1, 99, (2, 10), device=device_cpu),
        }

        with torch.no_grad():
            output = model(**batch)

        assert output["loss"].device == device_cpu


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
