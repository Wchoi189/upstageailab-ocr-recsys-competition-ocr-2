
import pytest
import torch
import torch.nn as nn
from ocr.features.detection.models.heads.db_head import DBHead

@pytest.mark.regression
@pytest.mark.bug_001
def test_db_head_initialization_prior():
    """
    Regression Test for BUG-001.
    Verifies that the DBHead initializes with a low probability prior (p=0.01)
    to prevent probability map saturation in uninitialized models.
    """
    # Initialize DBHead
    head = DBHead(in_channels=256, upscale=4)

    # Check bias of the final layer explicitly
    # The fix sets bias to -4.595 (approx -log((1-0.01)/0.01))
    final_conv = head.binarize[-1]
    assert isinstance(final_conv, nn.ConvTranspose2d), "Final layer should be ConvTranspose2d"

    # Check if bias is close to expected value
    # We check the mean bias, though it should be constant
    bias_mean = final_conv.bias.mean().item()

    expected_bias = -4.595
    assert abs(bias_mean - expected_bias) < 1e-2, \
        f"Final layer bias initialized to {bias_mean:.4f}, expected ~{expected_bias} for p=0.01 prior."

    # Functional Check: Run forward pass and check stats
    dummy_input = torch.randn(1, 256, 32, 32) # Small input
    out = head(dummy_input, return_loss=False)
    prob_map = out['prob_maps']

    # Sigmoid(-4.595) is approx 0.01. With random weights input, output should be somewhat centered around there but definitely low.
    p_mean = prob_map.mean().item()

    # Threshold for failure was ~0.5. We expect < 0.2 safely.
    assert p_mean < 0.25, f"Mean probability {p_mean:.4f} is too high! Risk of saturation (BUG-001 regression)."

    print(f"BUG-001 Verified: Bias={bias_mean:.4f}, Mean Prob={p_mean:.4f}")
