#!/usr/bin/env python3
"""Delete code glob references from context bundles (not appropriate for bundles)"""

import yaml
from pathlib import Path

BUNDLES_DIR = Path("AgentQMS/.agentqms/plugins/context_bundles")

# Code glob patterns to delete (should use code analysis tools instead)
CODE_REFS_TO_DELETE = [
    "ocr/core/analysis/*.py",
    "ocr/core/evaluation/*.py",
    "ocr/core/inference/pipeline.py",
    "ocr/core/metrics/*.py",
    "ocr/core/utils/*.py",
    "ocr/core/lightning/*.py",
    "ocr/features/detection/*.py",
    "ocr/features/kie/*.py",
    "ocr/features/layout/*.py",
    "ocr/features/recognition/*.py",
    "ocr/inference/pipeline.py",
    "ocr/models/",
    "ocr/postprocessing/",
    "ocr/preprocessing/"
]

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
    print("=== Deleting Code Glob References from Bundles ===\n")
    print("Rationale: Context bundles should reference specs/docs, not code files\n")

    total_deletions = 0
    bundles_modified = set()

    for ref in CODE_REFS_TO_DELETE:
        print(f"Deleting: {ref}")

        ref_deletions = 0
        for bundle_path in BUNDLES_DIR.glob("*.yaml"):
            if delete_bundle_reference(bundle_path, ref):
                ref_deletions += 1
                bundles_modified.add(bundle_path.name)

        if ref_deletions > 0:
            print(f"  ✅ Deleted from {ref_deletions} bundle(s)")
        else:
            print(f"  ℹ️  Not found in any bundles")

        total_deletions += ref_deletions

    print(f"\n✅ Total deletions: {total_deletions}")
    print(f"📦 Modified bundles: {len(bundles_modified)}")
    if bundles_modified:
        for bundle in sorted(bundles_modified):
            print(f"  - {bundle}")

    print("\nNext: Run update_bundle_refs.py to update refs to new specs")

if __name__ == "__main__":
    main()
