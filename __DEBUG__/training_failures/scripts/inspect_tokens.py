#!/usr/bin/env python3
"""Quick BOS/EOS Token Diagnostic - Leverages existing train.py infrastructure"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from ocr.core.utils.path_utils import PROJECT_ROOT

@hydra.main(config_path=str(PROJECT_ROOT / "configs"), config_name="main", version_base=None)
def main(cfg: DictConfig):
    print("=" * 80)
    print("BOS/EOS Token Diagnostic")
    print("=" * 80)

    OmegaConf.set_struct(cfg, False)

    from ocr.pipelines.orchestrator import OCRProjectOrchestrator

    orchestrator = OCRProjectOrchestrator(cfg)

    # Get dataset
    dataset = orchestrator.data_module.dataset_dict
    val_dataset = dataset.get("val") or dataset.get("test")

    if not val_dataset:
        print("\n❌ No validation dataset found")
        return

    tokenizer = getattr(val_dataset, "tokenizer", None)
    if not tokenizer:
        print("\n❌ No tokenizer found in dataset")
        return

    print(f"\n✓ Tokenizer: {tokenizer.__class__.__name__}")
    print(f"  BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}, PAD={tokenizer.pad_token_id}")

    # Get dataloader
    val_loader = orchestrator.data_module.val_dataloader()
    batch = next(iter(val_loader))

    print(f"\n✓ Batch: images={batch['images'].shape}, tokens={batch['text_tokens'].shape}")

    # Check first 10 samples
    print("\nChecking first 10 samples...")
    tokens_batch = batch["text_tokens"]

    bos_count = 0
    eos_count = 0

    for i in range(min(10, tokens_batch.size(0))):
        tokens = tokens_batch[i]
        non_pad = (tokens != tokenizer.pad_token_id).nonzero(as_tuple=False).squeeze()

        if len(non_pad) == 0:
            continue

        first = tokens[0].item()
        last = tokens[non_pad[-1] if non_pad.ndim > 0 else non_pad].item()

        has_bos = (first == tokenizer.bos_token_id)
        has_eos = (last == tokenizer.eos_token_id)

        if has_bos:
            bos_count += 1
        if has_eos:
            eos_count += 1

        print(f"  [{i}] BOS={'✓' if has_bos else '❌'} EOS={'✓' if has_eos else '❌'} | tokens={tokens.tolist()[:10]}...")

    print(f"\nResult: BOS={bos_count}/10, EOS={eos_count}/10")

    if bos_count == 10 and eos_count == 10:
        print("\n✅ PASS: All samples have BOS/EOS")
    else:
        print("\n❌ FAIL: Missing BOS/EOS tokens!")

if __name__ == "__main__":
    main()
