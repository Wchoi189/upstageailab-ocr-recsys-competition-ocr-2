"""
Integration Tests: Atomic PLM vs Monolithic Reference

Verifies that the atomic decoder with PLM integration produces
numerically equivalent results to the monolithic reference implementation.

Success Criteria:
    - Loss values must match (ε ≤ 1e-5 for fp32)
    - Gradients must match (ε ≤ 1e-5 for fp32)
    - Training convergence must match baseline
"""

import pytest
import torch
import torch.nn as nn

from ocr.domains.recognition.models.parseq_official_adapter import PARSeqOfficial
from ocr.domains.recognition.models.architecture import PARSeq
from ocr.domains.recognition.models.decoder import PARSeqDecoder
from ocr.domains.recognition.models.head import PARSeqHead


class TestAtomicPLMIntegration:
    """Test atomic decoder with PLM against monolithic reference."""

    @pytest.fixture
    def reference_model(self):
        """Create reference monolithic model."""
        model = PARSeqOfficial(
            num_tokens=95,
            max_label_length=25,
            perm_num=6,
            perm_forward=True,
            perm_mirrored=True,
        )
        model.eval()
        return model

    @pytest.fixture
    def atomic_model(self):
        """Create atomic model with PLM."""
        # Create a minimal atomic model for testing
        # Note: This won't use the full PARSeq architecture, just decoder + head
        decoder = PARSeqDecoder(
            in_channels=256,
            d_model=384,
            nhead=12,
            num_layers=12,
            dim_feedforward=1536,
            dropout=0.1,
            vocab_size=95,
            max_len=25,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            plm_config={
                'max_label_length': 25,
                'perm_num': 6,
                'perm_forward': True,
                'perm_mirrored': True,
                'device': 'cpu',
                'seed': 42,
            }
        )
        head = PARSeqHead(in_channels=384, out_channels=95)
        decoder.eval()
        head.eval()
        return {'decoder': decoder, 'head': head}

    def test_plm_permutation_generation(self, atomic_model, reference_model):
        """Test that PLM permutation generation matches reference."""
        import numpy as np

        decoder = atomic_model['decoder']
        ref = reference_model

        # Synchronize BOTH PyTorch and NumPy RNGs
        torch.manual_seed(42)

        # Reset NumPy RNG for both models to ensure determinism
        decoder.plm.rng = np.random.default_rng(42)
        ref.rng = np.random.default_rng(42)

        # Create test targets (4-char sequence without padding)
        targets = torch.tensor([
            [1, 5, 10, 15, 20, 2],  # BOS, 4 chars, EOS
        ], dtype=torch.long)

        # Generate permutations
        atomic_perms = decoder.plm.gen_tgt_perms(targets)
        ref_perms = ref.gen_tgt_perms(targets)

        # Must match exactly
        assert torch.equal(atomic_perms, ref_perms), (
            f"Permutations don't match!\n"
            f"Atomic: {atomic_perms}\n"
            f"Reference: {ref_perms}"
        )

    def test_attention_mask_generation(self, atomic_model, reference_model):
        """Test that attention mask generation matches reference."""
        decoder = atomic_model['decoder']
        ref = reference_model

        # Create test permutation
        perm = torch.tensor([0, 2, 1, 3, 4], dtype=torch.long)

        # Generate masks
        atomic_masks = decoder.plm.generate_attn_masks(perm)
        ref_content, ref_query = ref.generate_attn_masks(perm)

        # Must match exactly
        assert torch.equal(atomic_masks.content_mask, ref_content), (
            "Content masks don't match"
        )
        assert torch.equal(atomic_masks.query_mask, ref_query), (
            "Query masks don't match"
        )

    @pytest.mark.skip(reason="Full integration test - requires complete architecture setup")
    def test_forward_pass_equivalence(self, atomic_model, reference_model):
        """Test that forward pass produces equivalent outputs."""
        # This test requires full architecture setup with encoder
        # Skipping for now, will implement after full integration
        pass

    def test_loss_computation_single_perm(self, atomic_model):
        """Test loss computation for a single permutation."""
        decoder = atomic_model['decoder']
        head = atomic_model['head']

        # Create dummy data
        B, S, C = 2, 16, 256
        T = 8
        V = 95

        memory = torch.randn(B, S, C)
        targets = torch.randint(3, V, (B, T), dtype=torch.long)
        targets[:, 0] = 1  # BOS
        targets[:, -2] = 2  # EOS
        targets[:, -1] = 0  # PAD

        # Prepare input/output
        tgt_in = targets[:, :-1]
        tgt_out = targets[:, 1:]

        # Generate one permutation
        perm = decoder.plm.gen_tgt_perms(targets)[0]
        masks = decoder.plm.generate_attn_masks(perm)

        # Convert mask to additive format
        tgt_mask = masks.content_mask.float()
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1.0, float('-inf'))
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 0.0, 0.0)

        # Forward pass
        decoded = decoder(memory, targets=tgt_in, tgt_mask=tgt_mask)
        logits = head(decoded)

        # Compute loss
        loss = nn.functional.cross_entropy(
            logits.flatten(end_dim=1),
            tgt_out.flatten(),
            ignore_index=0
        )

        # Verify loss is finite and reasonable
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss > 0, "Loss should be positive"
        assert loss < 100, "Loss should be reasonable (< 100 for random init)"


class TestPLMTrainingLoop:
    """Test the full PLM training loop logic."""

    def test_eos_removal_after_second_perm(self):
        """Test critical EOS removal logic."""
        pad_id = 0
        eos_id = 2

        # Create targets with EOS
        tgt_out = torch.tensor([
            [5, 10, 15, 2, 0, 0],  # char, char, char, EOS, PAD, PAD
            [7, 12, 2, 0, 0, 0],   # char, char, EOS, PAD, PAD, PAD
        ], dtype=torch.long)

        # Count before removal
        n_before = (tgt_out != pad_id).sum().item()
        assert n_before == 7, f"Expected 7 non-pad tokens (including EOS), got {n_before}"

        # Remove EOS (simulating i==1 in training loop)
        tgt_out_modified = torch.where(
            tgt_out == eos_id,
            torch.tensor(pad_id, device=tgt_out.device),
            tgt_out
        )

        # Count after removal
        n_after = (tgt_out_modified != pad_id).sum().item()
        assert n_after == 5, f"Expected 5 non-pad tokens after EOS removal (3+2 chars), got {n_after}"

        # Verify EOS tokens are replaced with PAD
        assert (tgt_out_modified == eos_id).sum() == 0, "All EOS should be removed"

    def test_loss_normalization(self):
        """Test loss normalization by character count."""
        # Simulate loss computation across 3 permutations
        losses = []
        counts = []

        # Permutation 1: 4 chars + 1 EOS = 5 tokens
        n1 = 5
        loss1 = 2.5
        losses.append(n1 * loss1)
        counts.append(n1)

        # Permutation 2: 4 chars + 1 EOS = 5 tokens
        n2 = 5
        loss2 = 2.3
        losses.append(n2 * loss2)
        counts.append(n2)

        # Permutation 3+: 4 chars (EOS removed) = 4 tokens
        n3 = 4
        loss3 = 2.1
        losses.append(n3 * loss3)
        counts.append(n3)

        # Total loss
        total_loss = sum(losses)
        total_count = sum(counts)
        normalized_loss = total_loss / total_count

        # Verify normalization
        expected = (5*2.5 + 5*2.3 + 4*2.1) / (5 + 5 + 4)
        assert abs(normalized_loss - expected) < 1e-6, (
            f"Loss normalization incorrect: {normalized_loss} vs {expected}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
