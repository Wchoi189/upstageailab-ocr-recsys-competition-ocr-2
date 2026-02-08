#!/usr/bin/env python3
"""Batch update Type B and Type C bundle references"""

import csv
import yaml
from pathlib import Path

BUNDLES_DIR = Path("AgentQMS/.agentqms/plugins/context_bundles")
TRIAGE_CSV = Path("broken_refs_triage.csv")

def load_triage_results():
    """Load CSV and filter Type B and C refs"""
    refs_to_update = []

    with open(TRIAGE_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["category"] in ["B", "C"]:
                # Parse affected bundles
                bundles = row["affected_bundles"].split(";")
                refs_to_update.append({
                    "old_ref": row["broken_ref"],
                    "new_ref": row["new_location"],
                    "category": row["category"],
                    "bundles": bundles
                })

    return refs_to_update

def update_bundle_reference(bundle_path, old_ref, new_ref):
    """Update single reference in bundle YAML"""

    with open(bundle_path) as f:
        data = yaml.safe_load(f)

    modified = False

    # Recursively find and replace
    def replace_in_dict(obj):
        nonlocal modified
        if isinstance(obj, dict):
            if "path" in obj and obj["path"] == old_ref:
                obj["path"] = new_ref
                modified = True
            for value in obj.values():
                replace_in_dict(value)
        elif isinstance(obj, list):
            for item in obj:
                replace_in_dict(item)

    replace_in_dict(data)

    if modified:
        with open(bundle_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return modified

def main():
    print("=== Batch Update Type B & C Bundle References ===\n")

    refs_to_update = load_triage_results()

    print(f"Found {len(refs_to_update)} references to update\n")

    total_updates = 0

    for ref_info in refs_to_update:
        old_ref = ref_info["old_ref"]
        new_ref = ref_info["new_ref"]
        category = ref_info["category"]

        # Skip if new_ref is a placeholder
        if new_ref.startswith("RESTORE_FROM") or new_ref == "DELETE_REF":
            # Need to map Type C refs to actual spec paths
            if category == "C":
                # Map to extracted specs (Phase 1)
                basename = Path(old_ref).name
                if "python-core" in basename:
                    new_ref = "AgentQMS/specs/tier2-framework/core-infra/python-core.spec.md"
                elif "discovery-rules" in basename:
                    new_ref = "AgentQMS/specs/tier2-framework/discovery/discovery-rules.spec.md"
                elif "hydra-v5-patterns" in basename:
                    new_ref = "AgentQMS/specs/tier2-framework/patterns/hydra-v5-patterns.spec.md"
                elif "prohibited-actions" in basename:
                    new_ref = "AgentQMS/specs/tier1-contracts/prohibited-actions.spec.md"
                elif "workflow-requirements" in basename:
                    new_ref = "AgentQMS/specs/tier1-contracts/workflow-requirements.spec.md"

                # Map to extracted specs (Phase 2 - new)
                elif "artifact-types" in old_ref:
                    new_ref = "AgentQMS/specs/tier1-contracts/artifact-types.spec.md"
                elif "file-placement-rules" in old_ref:
                    new_ref = "AgentQMS/specs/tier1-contracts/file-placement-rules.spec.md"
                elif "configuration-standards" in old_ref:
                    new_ref = "AgentQMS/specs/tier2-framework/configuration/configuration-standards.spec.md"
                elif "hydra-configuration-architecture" in old_ref:
                    new_ref = "AgentQMS/specs/tier2-framework/configuration/hydra-architecture.spec.md"
                elif "ollama-models" in old_ref:
                    new_ref = "AgentQMS/specs/tier2-framework/agent-infra/ollama-models.spec.md"
                elif "api-contracts" in old_ref:
                    new_ref = "AgentQMS/specs/tier2-framework/api/contracts.spec.md"
                elif "coordinate-transforms" in old_ref:
                    new_ref = "AgentQMS/specs/tier2-framework/ocr-engine/coordinate-transforms.spec.md"
                elif "image-loading" in old_ref:
                    new_ref = "AgentQMS/specs/tier2-framework/ocr-engine/image-loading.spec.md"
                elif "model-management" in old_ref:
                    new_ref = "AgentQMS/specs/tier2-framework/ocr-engine/model-management.spec.md"
                elif "orchestration-flow" in old_ref:
                    new_ref = "AgentQMS/specs/tier2-framework/ocr-engine/orchestration-flow.spec.md"
                elif "pipeline-contracts" in old_ref:
                    new_ref = "AgentQMS/specs/tier2-framework/ocr-engine/pipeline-contracts.spec.md"
                elif "postprocessing-logic" in old_ref:
                    new_ref = "AgentQMS/specs/tier2-framework/ocr-engine/postprocessing-logic.spec.md"
                elif "preprocessing-logic" in old_ref:
                    new_ref = "AgentQMS/specs/tier2-framework/ocr-engine/preprocessing-logic.spec.md"
                else:
                    print(f"⚠️  No mapping for Type C: {old_ref}")
                    continue
            else:
                print(f"⚠️  Skipping placeholder: {old_ref}")
                continue


        print(f"Updating [{category}]: {Path(old_ref).name}")
        print(f"  → {new_ref}")

        bundle_updates = 0
        for bundle_name in ref_info["bundles"]:
            bundle_path = BUNDLES_DIR / bundle_name
            if not bundle_path.exists():
                print(f"  ⚠️  Bundle not found: {bundle_name}")
                continue

            if update_bundle_reference(bundle_path, old_ref, new_ref):
                bundle_updates += 1

        print(f"  ✅ Updated {bundle_updates} bundle(s)\n")
        total_updates += bundle_updates

    print(f"✅ Total updates: {total_updates}")
    print("\nNext: Run delete_bundle_refs.py to remove Type A refs")

if __name__ == "__main__":
    main()
