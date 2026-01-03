
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


class ExperimentReconciler:
    """
    Handles reconciliation between the filesystem and the manifest.json.
    """

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.metadata_dir = experiment_dir / ".metadata"
        self.manifest_path = experiment_dir / "manifest.json"

    def reconcile(self) -> dict[str, Any]:
        """
        Scans .metadata/ for artifacts, updates manifest.json, and returns stats.
        """
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")

        # Load current manifest
        with open(self.manifest_path) as f:
            manifest_data = json.load(f)

        # Scan for artifacts
        found_artifacts = self._scan_artifacts()

        # Update manifest artifacts list
        # We replace the list entirely based on disk state to ensure sync
        # But we might want to preserve some metadata if it exists?
        # For now, let's treat the disk frontmatter as truth for basics.

        manifest_data["artifacts"] = found_artifacts
        manifest_data["last_reconciled"] = datetime.now().isoformat()

        # Save updated manifest atomically (tempfile + os.replace pattern)
        json_content = json.dumps(manifest_data, indent=2)
        temp_path = self.manifest_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(json_content)
            os.replace(temp_path, self.manifest_path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise OSError(f"Failed to save manifest: {e}")

        return {
            "status": "success",
            "artifacts_count": len(found_artifacts),
            "last_reconciled": manifest_data["last_reconciled"]
        }

    def _scan_artifacts(self) -> list[dict[str, Any]]:
        """
        Recursively scans .metadata directory for markdown artifacts.
        """
        artifacts = []
        if not self.metadata_dir.exists():
            return artifacts

        for root, _, files in os.walk(self.metadata_dir):
            for file in files:
                if file.endswith(".md"):
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(self.experiment_dir)

                    frontmatter = self._extract_frontmatter(full_path)
                    if frontmatter:
                        artifact_entry = {
                            "path": str(rel_path),
                            "type": frontmatter.get("type", "other"),
                        }
                        if "subtype" in frontmatter:
                            artifact_entry["subtype"] = frontmatter["subtype"]

                        # Store other useful metadata if needed, but keep schema minimal
                        artifacts.append(artifact_entry)

        return artifacts

    def _extract_frontmatter(self, file_path: Path) -> dict[str, Any] | None:
        """
        Extracts YAML frontmatter from a markdown file.
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                     return yaml.safe_load(parts[1])
        except Exception as e:
            print(f"Warning: Failed to parse frontmatter for {file_path}: {e}")

        return None
