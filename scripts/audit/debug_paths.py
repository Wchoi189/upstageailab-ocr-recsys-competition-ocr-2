import hydra
from omegaconf import OmegaConf

@hydra.main(version_base=None, config_path="../../configs", config_name="main")
def debug_paths(cfg):
    print("--- 🔍 Namespace Trace ---")
    # Check if the data.transforms key exists
    if hasattr(cfg, 'data') and "transforms" in cfg.data:
        print(f"✅ Found 'data.transforms'")
        print(f"Keys available: {list(cfg.data.transforms.keys())}")
    else:
        if hasattr(cfg, 'data'):
            print(f"❌ Missing 'data.transforms'. Current data keys: {list(cfg.data.keys())}")
        else:
            print("❌ Missing 'data' key entirely.")

    # Verify the specific interpolation used by the dataset
    try:
        if hasattr(cfg, 'data') and hasattr(cfg.data, 'train_dataset'):
             val = cfg.data.train_dataset.transform
             print(f"✅ Dataset transform resolved to: {val._target_}")
        else:
             print("❌ plain 'data.train_dataset' missing")
    except Exception as e:
        print(f"❌ Dataset transform failed: {e}")

    print("--- 🔍 Model Trace ---")
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'architectures'):
         print(f"✅ Found 'model.architectures'")
         print(f"Keys: {list(cfg.model.architectures.keys())}")
         if 'max_len' in cfg.model.architectures:
             print(f"✅ Found max_len: {cfg.model.architectures.max_len}")
    else:
         print(f"❌ Missing 'model.architectures'")
         if hasattr(cfg, 'model'):
             print(f"Model keys: {list(cfg.model.keys())}")
