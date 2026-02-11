import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

def main():
    # 1. Setup Hydra
    GlobalHydra.instance().clear()
    hydra.initialize(version_base=None, config_path="../../../configs")
    cfg = hydra.compose(config_name="main", overrides=["domain=recognition", "global.paths.root_dir=/workspaces"])

    # 2. Instantiate Dataset directly
    print("Creating Dataset...")
    # Fix: Access dataset config from correct location (likely cfg.data.train for DataModule)
    # The config composed is 'main', which loads domain 'recognition'.
    # In 'recognition/data', the dataset is likely under 'train'.
    dataset = hydra.utils.instantiate(cfg.data.val_dataset)
    print(f"Dataset size: {len(dataset)}")

    print("Creating DataLoader...")
    # Fix: Correctly access collate_fn config
    collate_cfg = cfg.data.collate_fn
    collate_fn = hydra.utils.instantiate(collate_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    # 4. Iterate and Time
    print("Starting iteration...")
    start_time = time.time()
    for i, batch in enumerate(tqdm(dataloader, total=50)):
        if i == 0:
            print(f"Batch 0 keys: {batch.keys()}")
            print(f"Sample 0 Label: {batch['labels'][0]}")
            print(f"Sample 0 Tokens: {batch['text_tokens'][0]}")

        if i >= 10: break
        # Simulate basic GPU transfer -> OFF for now to test CPU speed
        images = batch['images']
        # if torch.cuda.is_available():
        #     images = images.cuda()

    print(f"Iterated 10 batches in {time.time() - start_time:.2f}s")
    print("Data loading is healthy if this finishes quickly.")

if __name__ == "__main__":
    main()
