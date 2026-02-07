#!/usr/bin/env python
"""
Diagnostic script to audit the 0% accuracy failure.
Checks Data Integrity, Tokenizer Alignment, and Model Initialization.
"""

import sys
import torch
from hydra import initialize, compose
from omegaconf import OmegaConf

def main():
    print("=" * 60)
    print("Recognition Pipeline Diagnostic")
    print("=" * 60)

    # 1. Initialize Hydra & Config
    print("\n[1/5] Loading Configuration...")
    with initialize(config_path="../../../configs", version_base=None):
        cfg = compose(
            config_name="main",
            overrides=[
                "experiment=rec_baseline_v1",
                "hardware=rtx3090",
                "global.debug=true", # Enable debug checks
                "global.paths.root_dir=/workspaces" # Force root dir
            ]
        )
    print("✓ Configuration loaded")

    # 2. Setup Orchestrator & Modules
    print("\n[2/5] Setting up Modules...")
    from ocr.pipelines.orchestrator import OCRProjectOrchestrator
    orchestrator = OCRProjectOrchestrator(cfg)
    try:
        pl_module, data_module = orchestrator.setup_modules()
        print(f"✓ PL Module: {type(pl_module).__name__}")
        print(f"✓ Data Module: {type(data_module).__name__}")
    except Exception as e:
        print(f"✗ Module setup failed: {e}")
        return 1

    # 3. Data Audit
    print("\n[3/5] Auditing Data Pipeline...")
    try:
        data_module.setup("fit") # Ensure datasets are loaded
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))

        images = batch["images"]
        text_tokens = batch.get("text_tokens")
        labels = batch.get("labels")

        print(f"  Batch Size: {images.size(0)}")
        print(f"  Image Shape: {images.shape}")
        print(f"  Image Stats: Min={images.min():.4f}, Max={images.max():.4f}, Mean={images.mean():.4f}, Std={images.std():.4f}")

        if images.max() > 1.0:
            print("  ⚠️ WARNING: Images appear unnormalized (Max > 1.0). Expected [0, 1] or [-1, 1].")
        else:
            print("  ✓ Images appear normalized.")

        tokenizer = pl_module._get_tokenizer()
        if tokenizer:
            print(f"  Tokenizer Vocab Size: {tokenizer.vocab_size}")

            # Check First Sample
            sample_tokens = text_tokens[0]
            decoded_text = tokenizer.decode(sample_tokens.tolist())
            raw_label = labels[0] if labels else "N/A"

            print(f"  Sample 0 Tokens: {sample_tokens.tolist()}")
            print(f"  Sample 0 Decoded: '{decoded_text}'")
            print(f"  Sample 0 Raw Label: '{raw_label}'")

            # Check BOS/EOS
            if sample_tokens[0] == tokenizer.BOS:
                print("  ✓ Sample starts with BOS token.")
            else:
                print(f"  ❌ ERROR: Sample DOES NOT start with BOS. Found ID {sample_tokens[0]}.")

            if tokenizer.EOS in sample_tokens:
                 print("  ✓ Sample contains EOS token.")
            else:
                 print("  ⚠️ WARNING: Sample does not contain EOS token (might be truncated).")

    except Exception as e:
        print(f"✗ Data Audit failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 4. Model Forward Pass
    print("\n[4/5] Running Model Forward Pass...")
    try:
        pl_module.eval()
        with torch.no_grad():
            outputs = pl_module.model(**batch, return_loss=True)
            loss = outputs["loss"]
            logits = outputs["logits"]

            print(f"  Loss: {loss.item():.6f}")
            print(f"  Logits Shape: {logits.shape}")
            print(f"  Logits Stats: Min={logits.min():.4f}, Max={logits.max():.4f}, Mean={logits.mean():.4f}")

            if torch.isnan(loss):
                print("  ❌ ERROR: Loss is NaN!")
            elif loss.item() == 0.0:
                print("  ⚠️ WARNING: Loss is exactly 0.0!")
            else:
                print("  ✓ Loss seems valid.")

    except Exception as e:
        print(f"✗ Model Forward failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("Diagnostic Complete")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    if "ocr" not in sys.modules:
        # Add project root to sys.path if not running as package
        project_root = "/workspaces"
        if project_root not in sys.path:
            sys.path.append(project_root)

    sys.exit(main())
