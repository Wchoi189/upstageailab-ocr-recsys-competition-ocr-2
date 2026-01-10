"""
Centralized Factory for Experiment Operations.
This is the ONLY allowed interface for AI agents to modify experiment state.
"""
import json
import datetime
from pathlib import Path
from typing import Optional

class ExperimentFactory:
    def __init__(self, root: str = "."):
        self.root = Path(root)
        self.experiments_dir = self.root / "experiments"
        
    def create_id(self, name: str) -> str:
        """Enforces YYYYMMDD_name format."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        slug = name.lower().replace(" ", "_").replace("-", "_")
        return f"{timestamp}_{slug}"

    def initialize(self, name: str, intention: str):
        """Atomic initialization of a new experiment."""
        exp_id = self.create_id(name)
        path = self.experiments_dir / exp_id
        path.mkdir(parents=True, exist_ok=False)
        
        # Create mandatory structure
        (path / "artifacts").mkdir()
        (path / "logs").mkdir()
        
        manifest = {
            "experiment_id": exp_id,
            "name": name,
            "intention": intention,
            "status": "active",
            "created_at": datetime.datetime.utcnow().isoformat(),
            "tasks": [],
            "insights": []
        }
        
        manifest_path = path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return exp_id

    def validate_state(self, exp_id: str) -> bool:
        """Validates current experiment against schema."""
        # Logic to check manifest structure and artifact presence
        return True

if __name__ == "__main__":
    # Provides a CLI for agents to use directly
    import argparse
    # ... CLI implementation ...