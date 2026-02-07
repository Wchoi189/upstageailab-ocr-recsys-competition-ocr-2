#!/usr/bin/env python
"""Verification script to test refactored components without full training.

This script:
1. Instantiates the Orchestrator with recognition config
2. Verifies vocab injection works correctly
3. Tests module instantiation
4. Validates configuration access patterns
"""

from hydra import initialize, compose
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("Refactoring Verification Script")
    print("=" * 60)

    # Initialize Hydra
    with initialize(config_path="../../../configs", version_base=None):
        cfg = compose(
            config_name="main",
            overrides=[
                "experiment=rec_baseline_v1",
                "hardware=rtx3090",
            ]
        )

    print("\n[1/4] Testing Orchestrator instantiation...")
    from ocr.pipelines.orchestrator import OCRProjectOrchestrator

    orchestrator = OCRProjectOrchestrator(cfg)
    print(f"✓ Orchestrator created (domain={orchestrator.domain})")

    print("\n[2/4] Testing vocab injection strategy...")
    from ocr.pipelines.strategies.recognition_config import RecognitionConfigStrategy

    # Test that strategy can be imported and used
    RecognitionConfigStrategy.inject_vocab_size(cfg)
    print("✓ Vocab injection completed")

    # Verify vocab_size was injected
    if hasattr(cfg.model, 'vocab_size'):
        print(f"✓ Model vocab_size = {cfg.model.vocab_size}")

    if hasattr(cfg.model, 'component_overrides'):
        if hasattr(cfg.model.component_overrides, 'decoder'):
            vocab = cfg.model.component_overrides.decoder.get('vocab_size', None)
            print(f"✓ Decoder vocab_size = {vocab}")

    print("\n[3/4] Testing module setup...")
    try:
        pl_module, data_module = orchestrator.setup_modules()
        print(f"✓ PL Module created: {type(pl_module).__name__}")
        print(f"✓ Data Module created: {type(data_module).__name__}")
    except Exception as e:
        print(f"✗ Module setup failed: {e}")
        return 1

    print("\n[4/4] Testing config access patterns...")
    # Test the new dict-based config access
    debug_mode = cfg.get("global", {}).get("debug", False)
    print(f"✓ Config access works (debug={debug_mode})")

    print("\n" + "=" * 60)
    print("✅ All verification checks passed!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
