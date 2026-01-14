#!/usr/bin/env python3
"""
Show effective Hydra configuration.
Usage: uv run python scripts/utils/show_config.py [overrides]
Example: uv run python scripts/utils/show_config.py train experiment=my_exp
"""

import sys

from AgentQMS.tools.utils.paths import get_project_root

# Add project root to path for resolving top-level modules like 'runners'
PROJECT_ROOT = get_project_root()
sys.path.append(str(PROJECT_ROOT))  # noqa: path-hack

from omegaconf import OmegaConf

# Default to 'train' config if not specified, but allow overrides
# We use a trick to make this work with hydra.main which expects a specific config name
# or we can just use the compose API which is more flexible for a tool like this.

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig

def show_config():
    # Parse arguments to separate config name and overrides
    args = sys.argv[1:]
    if not args:
        print("Usage: python scripts/utils/show_config.py [config_name] [overrides...]")
        print("Example: python scripts/utils/show_config.py train experiment=base")
        sys.exit(1)

    config_name = args[0]
    overrides = args[1:]

    # If the first arg looks like an override, assume config_name="train"
    if "=" in config_name:
        overrides = [config_name] + overrides
        config_name = "train"

    # Inject standard Hydra runtime overrides to satisfy interpolations
    overrides.extend([
        "hydra.job.name=job",
        "hydra.job.num=0",
        "hydra.job.id=0",
        "hydra.job.override_dirname=''",
        "hydra.runtime.output_dir=.",
    ])

    try:
        # Use absolute path for configs
        config_dir = PROJECT_ROOT / "configs"
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir), job_name="show_config"):
            cfg = compose(config_name=config_name, overrides=overrides, return_hydra_config=True)
            HydraConfig.instance().set_config(cfg)
            print(OmegaConf.to_yaml(cfg, resolve=True))
    except Exception as e:
        print(f"Error composing config: {e}")
        sys.exit(1)

if __name__ == "__main__":
    show_config()
