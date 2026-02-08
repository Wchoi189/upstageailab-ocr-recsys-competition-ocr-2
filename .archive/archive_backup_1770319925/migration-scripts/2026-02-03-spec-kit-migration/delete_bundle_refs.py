#!/usr/bin/env python3
"""Batch delete Type A bundle references (intentionally pruned)"""

import csv
import yaml
from pathlib import Path

BUNDLES_DIR = Path("AgentQMS/.agentqms/plugins/context_bundles")
TRIAGE_CSV = Path("broken_refs_triage.csv")

def load_type_a_refs():
    """Load CSV and filter Type A refs"""
    refs_to_delete = []

    with open(TRIAGE_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["category"] == "A":
                bundles = row["affected_bundles"].split(";")
                refs_to_delete.append({
                    "ref": row["broken_ref"],
                    "bundles": bundles
                })

    return refs_to_delete

def delete_bundle_reference(bundle_path, ref_to_delete):
    """Delete single reference from bundle YAML"""

    with open(bundle_path) as f:
        data = yaml.safe_load(f)

    modified = False

    # Recursively find and remove
    def remove_from_structure(obj, parent=None, parent_key=None):
        nonlocal modified

        if isinstance(obj, dict):
            # Check if this is a file entry with matching path
            if "path" in obj and obj["path"] == ref_to_delete:
                # Mark for removal - will be handled by parent
                return "DELETE"

            # Process nested structures
            keys_to_delete = []
            for key, value in list(obj.items()):
                result = remove_from_structure(value, obj, key)
                if result == "DELETE":
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del obj[key]
                modified = True

            # Clean up empty tier structures
            if "files" in obj and isinstance(obj["files"], list):
                obj["files"] = [f for f in obj["files"] if f is not None]
                if not obj["files"]:
                    return "DELETE"

        elif isinstance(obj, list):
            items_to_remove = []
            for i, item in enumerate(obj):
                # Check if item is a dict with matching path
                if isinstance(item, dict) and item.get("path") == ref_to_delete:
                    items_to_remove.append(i)
                elif isinstance(item, str) and item == ref_to_delete:
                    items_to_remove.append(i)
                else:
                    result = remove_from_structure(item, obj, i)
                    if result == "DELETE":
                        items_to_remove.append(i)

            # Remove in reverse order to preserve indices
            for i in sorted(items_to_remove, reverse=True):
                obj.pop(i)
                modified = True

        return None

    remove_from_structure(data)

    if modified:
        with open(bundle_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return modified

def main():
    print("=== Batch Delete Type A Bundle References ===\n")

    refs_to_delete = load_type_a_refs()

    print(f"Found {len(refs_to_delete)} Type A references to delete\n")

    total_deletions = 0

    for ref_info in refs_to_delete:
        ref = ref_info["ref"]

        print(f"Deleting: {Path(ref).name}")

        bundle_deletions = 0
        for bundle_name in ref_info["bundles"]:
            bundle_path = BUNDLES_DIR / bundle_name
            if not bundle_path.exists():
                print(f"  ⚠️  Bundle not found: {bundle_name}")
                continue

            if delete_bundle_reference(bundle_path, ref):
                bundle_deletions += 1

        print(f"  ✅ Deleted from {bundle_deletions} bundle(s)\n")
        total_deletions += bundle_deletions

    print(f"✅ Total deletions: {total_deletions}")
    print("\nNext: Verify with audit_bundles.py")

if __name__ == "__main__":
    main()
