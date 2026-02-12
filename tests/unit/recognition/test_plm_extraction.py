"""
Tests for PLM module extraction - Numerical Equivalence with Reference Adapter.

CRITICAL: These tests verify that the extracted PLM logic in plm.py
matches the reference implementation in parseq_official_adapter.py EXACTLY.

Success Criteria:
    - All permutations must match (ε ≤ 1e-5)
    - All attention masks must match exactly
    - Edge cases (1-char, 4-char) must work identically
"""

import numpy as np
import pytest
import torch

from ocr.core.interfaces.plm import AttentionMasks
from ocr.domains.recognition.models.parseq_official_adapter import PARSeqOfficial
from ocr.domains.recognition.models.plm import PermutationLanguageModeling


class TestPLMExtraction:
    """Test PLM extraction against reference adapter."""

    @pytest.fixture
    def plm_module(self):
        """Create PLM module with default settings."""
        # Synchronize PyTorch RNG for torch.randperm() calls
        torch.manual_seed(42)
        plm = PermutationLanguageModeling(
            max_label_length=25,
            perm_num=6,
            perm_forward=True,
            perm_mirrored=True,
            device="cpu",
            seed=42,  # NumPy RNG seed
        )
        return plm

    @pytest.fixture
    def reference_adapter(self):
        """Create reference adapter for comparison."""
        # Synchronize PyTorch RNG for torch.randperm() calls
        torch.manual_seed(42)
        model = PARSeqOfficial(
            num_tokens=95,  # Standard charset size
            max_label_length=25,
            perm_num=6,
            perm_forward=True,
            perm_mirrored=True,
        )
        # Synchronize NumPy RNG for rng.choice() calls
        model.rng = np.random.default_rng(42)
        model.eval()
        return model

    def test_gen_tgt_perms_1char_equivalence(self, plm_module, reference_adapter):
        """Test 1-character sequence permutation generation."""
        # 1-char sequence: [BOS, char, EOS]
        tgt = torch.tensor([[1, 5, 2]], dtype=torch.long)  # BOS=1, char=5, EOS=2

        # Generate permutations
        plm_perms = plm_module.gen_tgt_perms(tgt)
        ref_perms = reference_adapter.gen_tgt_perms(tgt)

        # Must match exactly
        assert torch.equal(plm_perms, ref_perms), (
            f"1-char permutations don't match!\n"
            f"PLM: {plm_perms}\n"
            f"Ref: {ref_perms}"
        )

        # Verify shape
        assert plm_perms.shape == (1, 3), f"Expected shape (1, 3), got {plm_perms.shape}"

    def test_gen_tgt_perms_4char_equivalence(self, plm_module, reference_adapter):
        """Test 4-character sequence with hardcoded selector."""
        # 4-char sequence: [BOS, a, b, c, d, EOS]
        tgt = torch.tensor([[1, 5, 10, 15, 20, 2]], dtype=torch.long)

        # Generate permutations
        plm_perms = plm_module.gen_tgt_perms(tgt)
        ref_perms = reference_adapter.gen_tgt_perms(tgt)

        # Must match exactly (same seed, same selector logic)
        assert torch.equal(plm_perms, ref_perms), (
            f"4-char permutations don't match!\n"
            f"PLM shape: {plm_perms.shape}\n"
            f"Ref shape: {ref_perms.shape}\n"
            f"Diff:\n{plm_perms - ref_perms}"
        )

        # Verify shape: 6 permutations (3 base + 3 mirrored)
        assert plm_perms.shape == (6, 6), f"Expected shape (6, 6), got {plm_perms.shape}"

        # Verify BOS positions (all must start with BOS=0)
        assert torch.all(plm_perms[:, 0] == 0), "All permutations must start with BOS=0"

        # Note: EOS position varies due to special reverse handling (perms[1])
        # so we don't check all ending with same EOS

    def test_gen_tgt_perms_long_sequence_equivalence(self, plm_module, reference_adapter):
        """Test longer sequence (>4 chars) with random sampling."""
        # 10-char sequence
        tgt = torch.tensor([[1] + list(range(5, 15)) + [2]], dtype=torch.long)

        # Generate permutations with synchronized RNG state
        # Reset both RNGs before each call to ensure deterministic comparison
        torch.manual_seed(42)
        plm_perms = plm_module.gen_tgt_perms(tgt)

        torch.manual_seed(42)
        ref_perms = reference_adapter.gen_tgt_perms(tgt)

        # Must match exactly (same RNG seed)
        assert torch.equal(plm_perms, ref_perms), (
            f"Long sequence permutations don't match!\n"
            f"PLM shape: {plm_perms.shape}\n"
            f"Ref shape: {ref_perms.shape}\n"
            f"First perm diff: {plm_perms[0] - ref_perms[0]}"
        )

        # Verify shape
        assert plm_perms.shape[0] == 6, "Should have 6 permutations"
        assert plm_perms.shape[1] == 12, "Sequence length should be 12 (10 chars + BOS + EOS)"

    def test_generate_attn_masks_equivalence(self, plm_module, reference_adapter):
        """Test attention mask generation."""
        # Create a permutation
        perm = torch.tensor([0, 3, 1, 2, 4], dtype=torch.long)  # BOS, c, a, b, EOS

        # Generate masks
        plm_masks = plm_module.generate_attn_masks(perm)
        ref_content, ref_query = reference_adapter.generate_attn_masks(perm)

        # Verify type
        assert isinstance(plm_masks, AttentionMasks), "PLM must return AttentionMasks dataclass"

        # Must match exactly
        assert torch.equal(plm_masks.content_mask, ref_content), (
            f"Content masks don't match!\n"
            f"PLM:\n{plm_masks.content_mask}\n"
            f"Ref:\n{ref_content}"
        )

        assert torch.equal(plm_masks.query_mask, ref_query), (
            f"Query masks don't match!\n"
            f"PLM:\n{plm_masks.query_mask}\n"
            f"Ref:\n{ref_query}"
        )

        # Verify shapes
        assert plm_masks.content_mask.shape == (4, 4), "Content mask should be [L-1, L-1]"
        assert plm_masks.query_mask.shape == (4, 4), "Query mask should be [L-1, L-1]"

    def test_full_pipeline_equivalence(self, plm_module, reference_adapter):
        """Test full PLM pipeline: gen_tgt_perms → generate_attn_masks."""
        # 5-char sequence
        tgt = torch.tensor([[1, 5, 10, 15, 20, 25, 2]], dtype=torch.long)

        # Generate permutations with synchronized RNG
        torch.manual_seed(42)
        plm_perms = plm_module.gen_tgt_perms(tgt)

        torch.manual_seed(42)
        ref_perms = reference_adapter.gen_tgt_perms(tgt)

        assert torch.equal(plm_perms, ref_perms), "Permutations must match"

        # Generate masks for each permutation
        for i, (plm_perm, ref_perm) in enumerate(zip(plm_perms, ref_perms)):
            plm_masks = plm_module.generate_attn_masks(plm_perm)
            ref_content, ref_query = reference_adapter.generate_attn_masks(ref_perm)

            assert torch.equal(plm_masks.content_mask, ref_content), (
                f"Permutation {i} content masks don't match"
            )
            assert torch.equal(plm_masks.query_mask, ref_query), (
                f"Permutation {i} query masks don't match"
            )

    def test_device_handling(self):
        """Test device parameter is respected."""
        plm_cpu = PermutationLanguageModeling(device="cpu")
        tgt = torch.tensor([[1, 5, 10, 2]], dtype=torch.long)

        perms = plm_cpu.gen_tgt_perms(tgt)
        assert perms.device.type == "cpu", "Permutations should be on CPU"

        perm = perms[0]
        masks = plm_cpu.generate_attn_masks(perm)
        assert masks.content_mask.device.type == "cpu", "Masks should be on CPU"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_equivalence(self):
        """Test CUDA device works and matches CPU results."""
        plm_cpu = PermutationLanguageModeling(device="cpu", seed=42)
        plm_cuda = PermutationLanguageModeling(device="cuda", seed=42)

        tgt_cpu = torch.tensor([[1, 5, 10, 15, 2]], dtype=torch.long)
        tgt_cuda = tgt_cpu.cuda()

        # Generate on both devices
        perms_cpu = plm_cpu.gen_tgt_perms(tgt_cpu)
        perms_cuda = plm_cuda.gen_tgt_perms(tgt_cuda)

        # Move to same device for comparison
        assert torch.equal(perms_cpu, perms_cuda.cpu()), "CPU and CUDA results must match"

    def test_mirrored_permutations(self, plm_module, reference_adapter):
        """Test mirrored (complementary) permutation generation."""
        tgt = torch.tensor([[1, 5, 10, 15, 2]], dtype=torch.long)

        plm_perms = plm_module.gen_tgt_perms(tgt)
        ref_perms = reference_adapter.gen_tgt_perms(tgt)

        # Verify mirrored structure: K/2 base + K/2 complementary
        assert plm_perms.shape[0] == 6, "Should have 6 permutations (3 base + 3 mirrored)"

        # Verify exact match with reference
        assert torch.equal(plm_perms, ref_perms), "Mirrored permutations must match reference"

    def test_forward_permutation_included(self, plm_module):
        """Test that forward (left-to-right) permutation is included when perm_forward=True."""
        tgt = torch.tensor([[1, 5, 10, 15, 2]], dtype=torch.long)
        perms = plm_module.gen_tgt_perms(tgt)

        # First permutation should be forward (identity ordering)
        # [BOS=0, 1, 2, 3, EOS=4]
        expected_forward = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        assert torch.equal(perms[0], expected_forward), (
            f"First permutation should be forward ordering\n"
            f"Got: {perms[0]}\n"
            f"Expected: {expected_forward}"
        )

    def test_special_reverse_handling(self, plm_module, reference_adapter):
        """Test special handling for reverse direction (perms[1])."""
        tgt = torch.tensor([[1, 5, 10, 2]], dtype=torch.long)

        plm_perms = plm_module.gen_tgt_perms(tgt)
        ref_perms = reference_adapter.gen_tgt_perms(tgt)

        # Verify reverse permutation (second one)
        assert torch.equal(plm_perms[1], ref_perms[1]), (
            f"Reverse permutation doesn't match!\n"
            f"PLM: {plm_perms[1]}\n"
            f"Ref: {ref_perms[1]}"
        )


class TestPLMEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_plm_config_validation(self):
        """Test PLM config is validated on init."""
        # Valid config
        plm = PermutationLanguageModeling(max_label_length=25, perm_num=6, perm_mirrored=True)
        assert plm.config.max_len == 25

        # Invalid: odd perm_num with mirrored
        with pytest.raises(ValueError, match="perm_num must be even"):
            PermutationLanguageModeling(perm_num=5, perm_mirrored=True)

    def test_attention_mask_properties(self):
        """Test attention mask properties and validation."""
        plm = PermutationLanguageModeling(device="cpu")
        perm = torch.tensor([0, 2, 1, 3], dtype=torch.long)

        masks = plm.generate_attn_masks(perm)

        # Verify dataclass properties
        assert masks.sequence_length == 3, "Sequence length should be L-1"
        assert masks.content_mask.dtype == torch.bool, "Masks must be boolean"
        assert masks.query_mask.dtype == torch.bool, "Masks must be boolean"

    def test_permutation_invariants(self):
        """Test permutation generation invariants."""
        plm = PermutationLanguageModeling(max_label_length=25, device="cpu")
        tgt = torch.tensor([[1, 5, 10, 15, 20, 2]], dtype=torch.long)

        perms = plm.gen_tgt_perms(tgt)

        # All permutations must:
        # 1. Start with BOS=0
        assert torch.all(perms[:, 0] == 0), "All perms must start with BOS=0"

        # 2. First permutation ends with EOS (others may differ due to reverse handling)
        max_num_chars = tgt.shape[1] - 2
        expected_eos = max_num_chars + 1
        assert perms[0, -1] == expected_eos, f"First perm must end with EOS={expected_eos}"

        # 3. Have correct length
        expected_len = tgt.shape[1]
        assert perms.shape[1] == expected_len, f"Permutations should have length {expected_len}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
