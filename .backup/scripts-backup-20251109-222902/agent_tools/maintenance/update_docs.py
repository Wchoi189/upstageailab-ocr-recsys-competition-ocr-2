#!/usr/bin/env python3
"""Quick documentation update workflow."""

import importlib.util
import sys
from pathlib import Path


def _load_bootstrap():
    if "scripts._bootstrap" in sys.modules:
        return sys.modules["scripts._bootstrap"]

    current_dir = Path(__file__).resolve().parent
    for directory in (current_dir, *tuple(current_dir.parents)):
        candidate = directory / "_bootstrap.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location(
                "scripts._bootstrap", candidate
            )
            if spec is None or spec.loader is None:  # pragma: no cover - defensive
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise RuntimeError("Could not locate scripts/_bootstrap.py")


_BOOTSTRAP = _load_bootstrap()
setup_project_paths = _BOOTSTRAP.setup_project_paths

setup_project_paths()

from scripts.agent_tools.maintenance.regenerate_docs import main as regenerate_main


def main() -> None:
    """Quick documentation update."""
    print("🚀 Quick Documentation Update")
    print("This will regenerate the AI handbook index and validate the structure.")
    print()

    # Run the regeneration workflow
    regenerate_main()


if __name__ == "__main__":
    main()
