from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CatalogOptions:
    outputs_dir: Path
    hydra_config_filenames: tuple[str, ...]


try:
    c = CatalogOptions(outputs_dir=Path("/tmp"), hydra_config_filenames=("config.yaml",))
    print("Success")
except Exception as e:
    print(f"Error: {e}")
