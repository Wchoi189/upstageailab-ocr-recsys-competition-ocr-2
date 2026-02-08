#!/usr/bin/env python3
"""
Consolidate fragmented specs into unified specs per original design intent.

This script fixes the architecture misalignment from Phase 2 by merging
7 OCR engine specs → 1 consolidated spec
2 configuration specs → 1 consolidated spec
"""

import yaml
from pathlib import Path
from datetime import datetime

# Paths
SPECS_DIR = Path("AgentQMS/specs/tier2-framework")
BUNDLES_DIR = Path("AgentQMS/.agentqms/plugins/context_bundles")

# === OCR Engine Consolidation ===

def consolidate_ocr_engine():
    """Consolidate 7 OCR engine specs into ocr_engine.spec.md"""

    ocr_specs = [
        "coordinate-transforms.spec.md",  # FW-012
        "image-loading.spec.md",          # FW-020
        "model-management.spec.md",       # FW-023
        "orchestration-flow.spec.md",     # FW-024
        "pipeline-contracts.spec.md",     # FW-026
        "postprocessing-logic.spec.md",   # FW-027
        "preprocessing-logic.spec.md",    # FW-028
    ]

    # Read all fragmented specs
    sections = {}
    for spec_file in ocr_specs:
        spec_path = SPECS_DIR / "ocr-engine" / spec_file
        content = spec_path.read_text()

        # Extract frontmatter and content
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = yaml.safe_load(parts[1])
            body = parts[2].strip()

            # Store by ID
            spec_id = frontmatter.get('id', spec_file)
            sections[spec_id] = {
                'title': body.split('\n')[0].strip('# '),
                'description': frontmatter.get('description', ''),
                'content': body
            }

    # Build consolidated spec
    consolidated = f"""---
ads_version: '2.0'
id: 'FW-OCR-ENGINE'
type: 'rule_set'
tier: 2
priority: 'high'
spec_version: '1.0.0'
updated: '{datetime.now().strftime("%Y-%m-%d")}'
description: 'OCR Engine standards consolidated from pipeline contracts, preprocessing, postprocessing, model management, orchestration, coordinates, and image loading.'
---

# OCR Engine Standards

> Consolidated orchestration, preprocessing, postprocessing, pipeline contracts, model management, coordinate transforms, and image loading standards.

## Overview

This specification consolidates all OCR engine component standards into a unified document for efficient agent consumption.

---

"""

    # Add sections in logical order
    order = [
        'FW-026',  # Pipeline Contracts
        'FW-024',  # Orchestration Flow
        'FW-028',  # Preprocessing Logic
        'FW-027',  # Postprocessing Logic
        'FW-023',  # Model Management
        'FW-012',  # Coordinate Transforms
        'FW-020',  # Image Loading
    ]

    for spec_id in order:
        if spec_id in sections:
            section = sections[spec_id]
            # Remove the first-level heading from content (we'll add our own)
            lines = section['content'].split('\n')
            body_lines = []
            skip_first_heading = True
            for line in lines:
                if skip_first_heading and line.startswith('# '):
                    skip_first_heading = False
                    continue
                body_lines.append(line)

            consolidated += f"## {section['title']} (from {spec_id})\n\n"
            consolidated += f"> {section['description']}\n\n"
            consolidated += '\n'.join(body_lines).strip() + "\n\n---\n\n"

    # Write consolidated spec
    output_path = SPECS_DIR / "ocr-engine.spec.md"
    output_path.write_text(consolidated)
    print(f"✓ Created consolidated: {output_path}")
    print(f"  Merged {len(sections)} specs: {', '.join(order)}")

    return list(sections.keys())


def consolidate_configuration():
    """Consolidate 2 configuration specs into configuration.spec.md"""

    config_specs = [
        "configuration-standards.spec.md",  # FW-011
        "hydra-architecture.spec.md",       # FW-017
    ]

    sections = {}
    for spec_file in config_specs:
        spec_path = SPECS_DIR / "configuration" / spec_file
        content = spec_path.read_text()

        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = yaml.safe_load(parts[1])
            body = parts[2].strip()

            spec_id = frontmatter.get('id', spec_file)
            sections[spec_id] = {
                'title': body.split('\n')[0].strip('# '),
                'description': frontmatter.get('description', ''),
                'content': body
            }

    consolidated = f"""---
ads_version: '2.0'
id: 'FW-CONFIGURATION'
type: 'rule_set'
tier: 2
priority: 'high'
spec_version: '1.0.0'
updated: '{datetime.now().strftime("%Y-%m-%d")}'
description: 'Configuration management standards consolidated from configuration standards and Hydra architecture rules.'
---

# Configuration Standards

> Consolidated configuration management, Hydra architecture, and externalization standards.

## Overview

This specification consolidates all configuration and Hydra-related standards for efficient agent consumption.

---

"""

    order = ['FW-011', 'FW-017']  # Configuration Standards, then Hydra Architecture

    for spec_id in order:
        if spec_id in sections:
            section = sections[spec_id]
            lines = section['content'].split('\n')
            body_lines = []
            skip_first_heading = True
            for line in lines:
                if skip_first_heading and line.startswith('# '):
                    skip_first_heading = False
                    continue
                body_lines.append(line)

            consolidated += f"## {section['title']} (from {spec_id})\n\n"
            consolidated += f"> {section['description']}\n\n"
            consolidated += '\n'.join(body_lines).strip() + "\n\n---\n\n"

    output_path = SPECS_DIR / "configuration.spec.md"
    output_path.write_text(consolidated)
    print(f"✓ Created consolidated: {output_path}")
    print(f"  Merged {len(sections)} specs: {', '.join(order)}")

    return list(sections.keys())


def update_bundle_references():
    """Update bundle references to point to consolidated specs"""

    # Mapping of old → new paths
    mappings = {
        # OCR Engine specs
        "AgentQMS/specs/tier2-framework/ocr-engine/coordinate-transforms.spec.md":
            "AgentQMS/specs/tier2-framework/ocr-engine.spec.md",
        "AgentQMS/specs/tier2-framework/ocr-engine/image-loading.spec.md":
            "AgentQMS/specs/tier2-framework/ocr-engine.spec.md",
        "AgentQMS/specs/tier2-framework/ocr-engine/model-management.spec.md":
            "AgentQMS/specs/tier2-framework/ocr-engine.spec.md",
        "AgentQMS/specs/tier2-framework/ocr-engine/orchestration-flow.spec.md":
            "AgentQMS/specs/tier2-framework/ocr-engine.spec.md",
        "AgentQMS/specs/tier2-framework/ocr-engine/pipeline-contracts.spec.md":
            "AgentQMS/specs/tier2-framework/ocr-engine.spec.md",
        "AgentQMS/specs/tier2-framework/ocr-engine/postprocessing-logic.spec.md":
            "AgentQMS/specs/tier2-framework/ocr-engine.spec.md",
        "AgentQMS/specs/tier2-framework/ocr-engine/preprocessing-logic.spec.md":
            "AgentQMS/specs/tier2-framework/ocr-engine.spec.md",

        # Configuration specs
        "AgentQMS/specs/tier2-framework/configuration/configuration-standards.spec.md":
            "AgentQMS/specs/tier2-framework/configuration.spec.md",
        "AgentQMS/specs/tier2-framework/configuration/hydra-architecture.spec.md":
            "AgentQMS/specs/tier2-framework/configuration.spec.md",
    }

    updated_bundles = []

    for bundle_file in BUNDLES_DIR.glob("*.yaml"):
        content = bundle_file.read_text()
        original = content

        for old_path, new_path in mappings.items():
            if old_path in content:
                content = content.replace(old_path, new_path)

        if content != original:
            bundle_file.write_text(content)
            updated_bundles.append(bundle_file.name)
            print(f"✓ Updated: {bundle_file.name}")

    print(f"\nTotal bundles updated: {len(updated_bundles)}")
    return updated_bundles


def deduplicate_bundle_references():
    """Remove duplicate references to the same consolidated spec in bundles"""

    for bundle_file in BUNDLES_DIR.glob("*.yaml"):
        try:
            with open(bundle_file) as f:
                data = yaml.safe_load(f)

            if not data or 'tiers' not in data:
                continue

            modified = False
            for tier_name, tier_data in data.get('tiers', {}).items():
                if 'files' not in tier_data:
                    continue

                # Track seen paths
                seen_paths = set()
                new_files = []

                for file_entry in tier_data['files']:
                    if isinstance(file_entry, dict):
                        path = file_entry.get('path')
                    else:
                        path = file_entry

                    if path and path not in seen_paths:
                        seen_paths.add(path)
                        new_files.append(file_entry)
                    elif path in seen_paths:
                        modified = True
                        print(f"  Removed duplicate: {path}")

                tier_data['files'] = new_files

            if modified:
                with open(bundle_file, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
                print(f"✓ Deduplicated: {bundle_file.name}")

        except Exception as e:
            print(f"⚠ Error processing {bundle_file.name}: {e}")


def main():
    print("=" * 70)
    print("SPEC CONSOLIDATION - Fixing Architecture Misalignment")
    print("=" * 70)
    print()

    # Step 1: Consolidate OCR Engine
    print("[1/4] Consolidating OCR Engine specs...")
    ocr_ids = consolidate_ocr_engine()
    print()

    # Step 2: Consolidate Configuration
    print("[2/4] Consolidating Configuration specs...")
    config_ids = consolidate_configuration()
    print()

    # Step 3: Update bundle references
    print("[3/4] Updating bundle references...")
    updated = update_bundle_references()
    print()

    # Step 4: Deduplicate references
    print("[4/4] Deduplicating bundle references...")
    deduplicate_bundle_references()
    print()

    print("=" * 70)
    print("CONSOLIDATION COMPLETE")
    print("=" * 70)
    print(f"✓ OCR Engine: 7 specs → 1 (FW-OCR-ENGINE)")
    print(f"✓ Configuration: 2 specs → 1 (FW-CONFIGURATION)")
    print(f"✓ Bundles updated: {len(updated)}")
    print()
    print("Next steps:")
    print("  1. Delete fragmented specs: rm -rf AgentQMS/specs/tier2-framework/ocr-engine/")
    print("  2. Delete fragmented specs: rm -rf AgentQMS/specs/tier2-framework/configuration/")
    print("  3. Run audit: uv run python audit_bundles.py")


if __name__ == "__main__":
    main()
