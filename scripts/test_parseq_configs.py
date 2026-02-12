#!/usr/bin/env python3
"""
Smoke test for PARSeq variant configurations.
Tests Hydra composition and basic instantiation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
# Import path_utils to register Hydra resolvers
import ocr.core.utils.path_utils  # noqa: F401

def test_config_composition(experiment_name: str):
    """Test if a config can be composed successfully."""
    print(f"\n{'='*80}")
    print(f"Testing: {experiment_name}")
    print(f"{'='*80}")

    try:
        # Initialize Hydra (resolvers registered by importing path_utils)
        config_dir = PROJECT_ROOT / "configs"
        with initialize_config_dir(
            version_base="1.3",
            config_dir=str(config_dir),
            job_name="test"
        ):
            # Compose config
            cfg = compose(
                config_name="main",
                overrides=[
                    f"experiment={experiment_name}",
                    "hydra.job.name=test",
                    "hydra.job.num=0",
                ],
                return_hydra_config=False
            )

            print(f"\n✓ Config composition successful")

            # Check key components
            print(f"\nArchitecture: {cfg.model.architectures.name}")
            print(f"Decoder d_model: {cfg.model.architectures.decoder.d_model}")
            print(f"Decoder use_flash_attention: {cfg.model.architectures.decoder.use_flash_attention}")
            print(f"Decoder plm_config: {cfg.model.architectures.decoder.plm_config}")
            print(f"Batch size: {cfg.data.batch_size}")
            print(f"Max epochs: {cfg.trainer.max_epochs}")

            return True

    except Exception as e:
        print(f"\n✗ Config composition failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test all PARSeq variant configs."""
    experiments = [
        "parseq_baseline",
        "parseq_flash",
        "parseq_plm",
        "parseq_plm_flash",
    ]

    print("="*80)
    print("PARSeq Configuration Smoke Tests")
    print("="*80)

    results = {}
    for exp in experiments:
        results[exp] = test_config_composition(exp)

    # Summary
    print(f"\n{'='*80}")
    print("Test Summary")
    print(f"{'='*80}")

    passed = sum(results.values())
    total = len(results)

    for exp, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {exp}")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
