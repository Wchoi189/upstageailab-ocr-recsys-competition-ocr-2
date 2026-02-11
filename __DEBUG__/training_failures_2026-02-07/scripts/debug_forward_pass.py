import hydra
from hydra.core.global_hydra import GlobalHydra
import torch
import time

def main():
    # 1. Setup
    GlobalHydra.instance().clear()
    hydra.initialize(version_base=None, config_path="../../../configs")
    cfg = hydra.compose(config_name="main", overrides=["domain=recognition", "global.paths.root_dir=/workspaces"])

    # 2. Instantiate Model
    print("Creating Model...")
    # Fix: Inject vocab_size manually since we are skipping full orchestration
    # The config expects ${model.vocab_size} which is usually set by constants or orchestrator
    cfg.model.architectures.decoder.vocab_size = 1000
    cfg.model.architectures.head.out_features = 1000

    model = hydra.utils.instantiate(cfg.model.architectures)
    print(f"Model created: {type(model).__name__}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Moving to device: {device}")
    model = model.to(device)
    model.eval() # Test inference first

    # 3. Create Dummy Input
    print("Creating dummy input...")
    B, C, H, W = 2, 3, 32, 128
    images = torch.randn(B, C, H, W).to(device)

    # 4. Forward Pass
    # Create dummy targets (B, T)
    targets = torch.randint(1, 1000, (B, 26)).to(device)

    print("Running forward pass...")
    start_time = time.time()
    with torch.no_grad():
        # Fix: targets must be passed as 'text_tokens' kwarg, and return_loss explicitly if needed
        output = model(images, text_tokens=targets, return_loss=True)
        print(f"Output key: {output.keys()}")
        print(f"Logits shape: {output['logits'].shape}")

    print(f"Forward pass finished in {time.time() - start_time:.4f}s")

if __name__ == "__main__":
    main()
