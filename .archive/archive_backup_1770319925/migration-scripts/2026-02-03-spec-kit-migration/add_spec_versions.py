#!/usr/bin/env python3
"""
Add spec_version frontmatter to all tier2 spec files
"""

from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).parent.parent
SPECS_DIR = PROJECT_ROOT / "AgentQMS" / "specs" / "tier2-framework"

# Specs that need frontmatter added
SPECS_WITHOUT_FRONTMATTER = [
    "configuration.spec.md",
    "constraints.spec.md",
    "framework_specs.spec.md",
    "misc.spec.md",
    "runtime.spec.md",
]

# Specs with frontmatter that need spec_version added
SPECS_NEED_VERSION = [
    "agent-infra.spec.md",
    "api.spec.md",
    "core-infra.spec.md",
    "discovery.spec.md",
    "patterns.spec.md",
]

def add_frontmatter(spec_path: Path):
    """Add frontmatter with spec_version to specs without frontmatter"""
    content = spec_path.read_text(encoding='utf-8')

    # Extract title from first heading
    title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
    title = title_match.group(1) if title_match else spec_path.stem.replace('-', ' ').title()

    # Generate spec_id from filename
    spec_id = f"FW-{spec_path.stem.upper().replace('-', '_')}"

    frontmatter = f"""---
ads_version: '2.0'
id: '{spec_id}'
type: 'rule_set'
tier: 2
priority: 'high'
spec_version: '1.0.0'
updated: '2026-02-03'
description: '{title} for framework tier'
---

"""

    new_content = frontmatter + content
    spec_path.write_text(new_content, encoding='utf-8')
    print(f"✅ Added frontmatter to {spec_path.name}")

def add_spec_version(spec_path: Path):
    """Add spec_version field to existing frontmatter"""
    content = spec_path.read_text(encoding='utf-8')

    # Check if spec_version already exists
    if 'spec_version:' in content:
        print(f"⏭️  {spec_path.name} already has spec_version")
        return

    # Add spec_version after updated field in frontmatter
    pattern = r"(updated: '[^']+'\n)"
    replacement = r"\1spec_version: '1.0.0'\n"

    new_content = re.sub(pattern, replacement, content)

    if new_content != content:
        spec_path.write_text(new_content, encoding='utf-8')
        print(f"✅ Added spec_version to {spec_path.name}")
    else:
        print(f"⚠️  Could not add spec_version to {spec_path.name} (pattern not found)")

def main():
    print(f"Processing tier2 specs in: {SPECS_DIR}\n")

    # Add frontmatter to specs without it
    for spec_name in SPECS_WITHOUT_FRONTMATTER:
        spec_path = SPECS_DIR / spec_name
        if spec_path.exists():
            add_frontmatter(spec_path)

    # Add spec_version to specs with frontmatter
    for spec_name in SPECS_NEED_VERSION:
        spec_path = SPECS_DIR / spec_name
        if spec_path.exists():
            add_spec_version(spec_path)

    print(f"\n✅ Spec versioning complete!")

if __name__ == "__main__":
    main()
