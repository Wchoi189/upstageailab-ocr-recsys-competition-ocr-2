#!/usr/bin/env python3
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import yaml

EXPERIMENTS_ROOT = Path("/workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment_manager/experiments")
TARGET_EXPERIMENTS = [
    "20251122_172313_perspective_correction",
    "20251128_005231_perspective_correction",
    "20251128_220100_perspective_correction",
]


def patch_frontmatter(file_path, experiment_id, artifact_type):
    """Adds or updates EDS v1.0 frontmatter to a markdown file."""
    if not file_path.exists():
        return

    content = file_path.read_text(encoding="utf-8")

    # Check if frontmatter exists
    pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
    match = re.match(pattern, content, re.DOTALL)

    now = datetime.now().isoformat()
    title = file_path.stem.replace("_", " ").title()

    if match:
        fm_str, body = match.groups()
        try:
            fm = yaml.safe_load(fm_str) or {}
        except:
            fm = {}
    else:
        fm = {}
        body = content

    # Standardize frontmatter
    fm.update(
        {
            "ads_version": "1.0",
            "type": artifact_type,
            "experiment_id": experiment_id,
            "status": fm.get("status", "complete"),
            "created": fm.get("created", now),
            "updated": now,
            "tags": fm.get("tags", []),
            "title": fm.get("title", title),
        }
    )

    # Add type-specific defaults if missing
    if artifact_type == "assessment":
        fm.setdefault("phase", "phase_1")
        fm.setdefault("priority", "medium")
        fm.setdefault("evidence_count", 0)
    elif artifact_type == "report":
        fm.setdefault("metrics", [])
        fm.setdefault("baseline", "none")
        fm.setdefault("comparison", "neutral")

    new_content = "---\n" + yaml.dump(fm, sort_keys=False) + "---\n" + body
    file_path.write_text(new_content, encoding="utf-8")


def migrate_experiment(exp_id):
    print(f"Migrating {exp_id}...")
    exp_path = EXPERIMENTS_ROOT / exp_id
    if not exp_path.exists():
        print(f"Error: {exp_path} not found")
        return

    # 1. Directory Reorganization
    meta_path = exp_path / ".metadata"
    meta_path.mkdir(exist_ok=True)

    mapping = {"assessments": "assessments", "incident_reports": "reports", "scripts": "scripts", "logs": "logs"}

    for old_name, new_name in mapping.items():
        old_dir = exp_path / old_name
        new_dir = meta_path / new_name
        if old_dir.exists() and old_dir.is_dir():
            print(f"  Moving {old_name} -> .metadata/{new_name}")
            new_dir.mkdir(parents=True, exist_ok=True)
            for item in old_dir.iterdir():
                shutil.move(str(item), str(new_dir / item.name))
            old_dir.rmdir()

    # 2. Manifest Conversion
    state_file = exp_path / "state.json"
    manifest_file = exp_path / "manifest.json"

    if state_file.exists():
        print("  Converting state.json to manifest.json")
        with open(state_file) as f:
            state = json.load(f)

        # Map state fields to manifest
        manifest = {
            "experiment_id": exp_id,
            "name": state.get("type", exp_id).replace("_", " ").title(),
            "status": "complete" if state.get("status") == "complete" else "active",
            "created_at": state.get("timestamp", datetime.now().isoformat()),
            "updated_at": state.get("last_updated", datetime.now().isoformat()),
            "intention": state.get("intention", ""),
            "tasks": [],
            "insights": [],
            "artifacts": [],
        }

        # Convert artifacts
        for art in state.get("artifacts", []):
            art_path = art.get("path", "")
            if not art_path:
                continue

            # Remap paths to .metadata/ if they were in the moved directories
            parts = art_path.split("/")
            if parts[0] in mapping:
                parts[0] = f".metadata/{mapping[parts[0]]}"
                art_path = "/".join(parts)

            manifest["artifacts"].append(
                {"path": art_path, "type": art.get("type", "other"), "timestamp": art.get("timestamp", datetime.now().isoformat())}
            )

            # Patch metadata if it's a markdown file
            full_path = exp_path / art_path
            if full_path.suffix == ".md" and "artifacts" in art_path:  # Legacy artifacts folder
                pass  # We'll handle this in the recursive walk

        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

        state_file.unlink()

    # 3. Patch markdown artifacts (Recursive walk)
    for md_file in exp_path.rglob("*.md"):
        if md_file.name == "README.md":
            continue

        # Determine type based on parents
        rel_path = md_file.relative_to(exp_path)
        art_type = "assessment"
        if ".metadata/reports" in str(rel_path):
            art_type = "report"
        elif ".metadata/guides" in str(rel_path):
            art_type = "guide"
        elif ".metadata/scripts" in str(rel_path):
            art_type = "script"

        print(f"  Patching {rel_path} as {art_type}")
        patch_frontmatter(md_file, exp_id, art_type)


if __name__ == "__main__":
    for exp_id in TARGET_EXPERIMENTS:
        migrate_experiment(exp_id)
    print("Done!")
