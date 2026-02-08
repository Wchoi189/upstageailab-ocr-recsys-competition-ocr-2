#!/usr/bin/env python3
"""Clean up final 9 broken refs (non-existent files)"""

import yaml
from pathlib import Path

BUNDLES_DIR = Path("AgentQMS/.agentqms/plugins/context_bundles")

# Final broken refs to delete (files don't exist)
REFS_TO_DELETE = [
    # Tier3-agents configs (don't exist - only agent_identities.spec.md exists)
    "AgentQMS/standards/tier3-agents/claude/config.yaml",
    "AgentQMS/standards/tier3-agents/copilot/config.yaml",
    "AgentQMS/standards/tier3-agents/cursor/config.yaml",
    "AgentQMS/standards/tier3-agents/gemini/config.yaml",

    # Config files (don't exist)
    "configs/train.yaml",
    "configs/_foundation/*.yaml",
    "configs/hydra/*.yaml",

    # Documentation (doesn't exist)
    "docs/index.md"
]

def delete_bundle_reference(bundle_path, ref_to_delete):
    """Delete single reference from bundle YAML"""

    with open(bundle_path) as f:
        data = yaml.safe_load(f)

    modified = False

    # Recursively find and remove
    def remove_from_structure(obj):
        nonlocal modified

        if isinstance(obj, dict):
            # Check if this is a file entry with matching path
            if "path" in obj and obj["path"] == ref_to_delete:
                return "DELETE"

            # Process nested structures
            keys_to_delete = []
            for key, value in list(obj.items()):
                result = remove_from_structure(value)
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
                    result = remove_from_structure(item)
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
    print("=== Cleaning Up Final 9 Broken References ===\n")
    print("Deleting refs to non-existent files\n")

    total_deletions = 0
    bundles_modified = set()

    for ref in REFS_TO_DELETE:
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

    print("\nNext: Run audit_bundles.py to verify 0 broken refs")

if __name__ == "__main__":
    main()
