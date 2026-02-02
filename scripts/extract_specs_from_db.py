#!/usr/bin/env python3
"""Extract specs from standards_db.json and convert to .spec.md format"""

import json
import yaml
from pathlib import Path
from datetime import datetime

STANDARDS_DB = Path("AgentQMS/.agentqms/standards_db.json")
SPECS_DIR = Path("AgentQMS/specs")

# Specs to extract with target locations
SPECS_TO_EXTRACT = {
    "tier2-framework/core-infra/python-core.yaml": {
        "target": "tier2-framework/core-infra/python-core.spec.md",
        "id": "FW-030",
        "title": "Python Core Standards"
    },
    "tier2-framework/discovery/discovery-rules.yaml": {
        "target": "tier2-framework/discovery/discovery-rules.spec.md",
        "id": "FW-037",
        "title": "Context Bundle Discovery Rules"
    },
    "tier2-framework/specs/hydra-v5-patterns-reference.yaml": {
        "target": "tier2-framework/patterns/hydra-v5-patterns.spec.md",
        "id": "FW-018",
        "title": "Hydra V5 Patterns Reference"
    },
    "tier1-sst/prohibited-actions.yaml": {
        "target": "tier1-contracts/prohibited-actions.spec.md",
        "id": "SC-006",
        "title": "Prohibited Actions"
    },
    "tier1-sst/constraints/workflow-requirements.yaml": {
        "target": "tier1-contracts/workflow-requirements.spec.md",
        "id": "SC-010",
        "title": "Workflow Requirements"
    }
}

def load_standards_db():
    """Load standards_db.json"""
    with open(STANDARDS_DB) as f:
        return json.load(f)

def yaml_to_markdown(data, title, spec_id):
    """Convert YAML data to AI-optimized Markdown spec"""

    # Parse YAML string from DB
    if isinstance(data, str):
        spec_data = yaml.safe_load(data)
    else:
        spec_data = data

    # Build frontmatter
    frontmatter = {
        "ads_version": spec_data.get("ads_version", "2.0"),
        "id": spec_data.get("id", spec_id),
        "type": spec_data.get("type", "rule_set"),
        "tier": spec_data.get("tier", 2),
        "priority": spec_data.get("priority", "high"),
        "updated": datetime.now().strftime("%Y-%m-%d")
    }

    # Add optional fields
    if "description" in spec_data:
        frontmatter["description"] = spec_data["description"]
    if "dependencies" in spec_data:
        frontmatter["dependencies"] = spec_data["dependencies"]

    # Build markdown content
    md_lines = ["---"]
    for key, value in frontmatter.items():
        if isinstance(value, list):
            md_lines.append(f"{key}:")
            for item in value:
                md_lines.append(f"  - {item}")
        else:
            md_lines.append(f"{key}: {repr(value) if isinstance(value, str) else value}")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append(f"# {title}")
    md_lines.append("")

    # Convert YAML structure to concise Markdown
    # Remove metadata fields already in frontmatter
    content_data = {k: v for k, v in spec_data.items()
                    if k not in ["ads_version", "id", "type", "tier", "priority",
                                 "validates_with", "compliance_status", "memory_footprint",
                                 "fuzzy_threshold", "keywords"]}

    # Add description if present
    if "description" in content_data:
        md_lines.append(f"> {content_data['description']}")
        md_lines.append("")
        del content_data["description"]

    # Convert remaining content to YAML code block for AI parsing
    md_lines.append("## Specification")
    md_lines.append("")
    md_lines.append("```yaml")
    md_lines.append(yaml.dump(content_data, default_flow_style=False, sort_keys=False))
    md_lines.append("```")

    return "\n".join(md_lines)

def extract_spec(db_key, target_info):
    """Extract single spec from DB and save as .spec.md"""

    db = load_standards_db()

    # Find spec in DB
    if db_key not in db:
        print(f"⚠️  {db_key} not found in standards_db.json")
        return False

    spec_yaml = db[db_key]

    # Convert to markdown
    md_content = yaml_to_markdown(
        spec_yaml,
        target_info["title"],
        target_info["id"]
    )

    # Write to file
    target_path = SPECS_DIR / target_info["target"]
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with open(target_path, "w") as f:
        f.write(md_content)

    print(f"✅ Extracted {target_info['id']}: {target_path}")
    return True

def main():
    print("=== Extracting Specs from standards_db.json ===\n")

    extracted = 0
    for db_key, target_info in SPECS_TO_EXTRACT.items():
        if extract_spec(db_key, target_info):
            extracted += 1

    print(f"\n✅ Extracted {extracted}/{len(SPECS_TO_EXTRACT)} specs")

    # Update bundle references will happen in next script
    print("\nNext: Run update_bundle_refs.py to update Type C references")

if __name__ == "__main__":
    main()
