import hydra
from omegaconf import OmegaConf, DictConfig
import sys
import os


# Mock hydra args
sys.argv = ["scripts/runners/train.py", "experiment=rec_baseline_v1"]

# Resolve path to configs
# Script is in /workspaces/__DEBUG__/training_failures/scripts/
# Configs are in /workspaces/configs
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(current_dir, "../../../"))
config_path_rel = os.path.join(workspace_root, "configs")
print(f"Debugger: Workspace Root: {workspace_root}")
print(f"Debugger: Config Path: {config_path_rel}")

def debug_config():
    with hydra.initialize(config_path="../../../configs", version_base=None):
        cfg = hydra.compose(config_name="main", overrides=["experiment=rec_baseline_v1"])

        print("-" * 40)
        print(f"Domain Raw: {cfg.get('domain')}")
        print(f"Task Raw: {cfg.get('task')}")

        domain_cfg = cfg.get("domain", cfg.get("task", "detection"))
        print(f"Resolved Domain Config Type: {type(domain_cfg)}")
        print(f"Resolved Domain Config Value: {domain_cfg}")

        if isinstance(domain_cfg, (dict, DictConfig)):
            domain = domain_cfg.get("task", "detection")
        else:
            domain = domain_cfg

        print(f"Final Domain: {domain}")
        print("-" * 40)

if __name__ == "__main__":
    debug_config()
