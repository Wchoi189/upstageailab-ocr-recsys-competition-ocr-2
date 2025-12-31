import os
import yaml
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "base.yaml"

def load_config() -> dict[str, Any]:
    """Load configuration from YAML file."""
    if not CONFIG_PATH.exists():
        # Fallback defaults if config file missing
        return {
            "api": {
                "base_url": "https://api.upstage.ai/v1",
                "endpoints": {
                    "document_parse": {
                        "submit": "https://api.upstage.ai/v1/document-digitization/async",
                        "status": "https://api.upstage.ai/v1/document-digitization/requests"
                    },
                    "prebuilt_extraction": "https://api.upstage.ai/v1/information-extraction"
                }
            },
            "processing": {
                "default_concurrency": 3,
                "default_batch_size": 500,
                "request_delay": 0.05,
                "poll_delay": 5.0,
                "poll_concurrency": 2,
                "poll_max_wait": 300
            }
        }
    
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    # Resolve simple variable substitution (e.g., ${api.base_url})
    # This is a very basic implementation to support the base.yaml structure
    base_url = config.get("api", {}).get("base_url", "")
    if base_url:
        for key, subsection in config.get("api", {}).get("endpoints", {}).items():
            if isinstance(subsection, dict):
                for subkey, value in subsection.items():
                    if isinstance(value, str) and "${api.base_url}" in value:
                        subsection[subkey] = value.replace("${api.base_url}", base_url)
            elif isinstance(subsection, str) and "${api.base_url}" in subsection:
                 config["api"]["endpoints"][key] = subsection.replace("${api.base_url}", base_url)
                 
    return config

# Global config object
cfg = load_config()
