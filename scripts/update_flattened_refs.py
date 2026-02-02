#!/usr/bin/env python3
"""Update bundle references for flattened specs"""

from pathlib import Path

BUNDLES_DIR = Path("AgentQMS/.agentqms/plugins/context_bundles")

mappings = {
    "AgentQMS/specs/tier2-framework/agent-infra/ollama-models.spec.md":
        "AgentQMS/specs/tier2-framework/agent-infra.spec.md",
    "AgentQMS/specs/tier2-framework/api/contracts.spec.md":
        "AgentQMS/specs/tier2-framework/api.spec.md",
}

updated = []
for bundle_file in BUNDLES_DIR.glob("*.yaml"):
    content = bundle_file.read_text()
    original = content

    for old_path, new_path in mappings.items():
        if old_path in content:
            content = content.replace(old_path, new_path)

    if content != original:
        bundle_file.write_text(content)
        updated.append(bundle_file.name)
        print(f"✓ Updated: {bundle_file.name}")

print(f"\nTotal: {len(updated)} bundles updated")
