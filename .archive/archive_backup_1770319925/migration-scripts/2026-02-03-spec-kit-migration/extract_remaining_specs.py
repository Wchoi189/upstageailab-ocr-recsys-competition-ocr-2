#!/usr/bin/env python3
"""Extract remaining 13 specs from standards_db.json and convert to .spec.md format"""

import json
import yaml
from pathlib import Path
from datetime import datetime

STANDARDS_DB = Path("AgentQMS/.agentqms/standards_db.json")
SPECS_DIR = Path("AgentQMS/specs")

# Remaining specs to extract (Phase 2)
SPECS_TO_EXTRACT = {
    # Tier 1 Contracts
    "tier1-sst/specs/artifact-types-reference.yaml": {
        "target": "tier1-contracts/artifact-types.spec.md",
        "id": "SC-002",
        "title": "Artifact Types Reference"
    },
    "tier1-sst/file-placement-rules.yaml": {
        "target": "tier1-contracts/file-placement-rules.spec.md",
        "id": "SC-004",
        "title": "File Placement Rules"
    },

    # Tier 2 Framework - Configuration
    "tier2-framework/configuration/configuration-standards.yaml": {
        "target": "tier2-framework/configuration/configuration-standards.spec.md",
        "id": "FW-011",
        "title": "Configuration Standards"
    },
    "tier2-framework/configuration/hydra-configuration-architecture.yaml": {
        "target": "tier2-framework/configuration/hydra-architecture.spec.md",
        "id": "FW-017",
        "title": "Hydra Configuration Architecture"
    },

    # Tier 2 Framework - Agent Infrastructure
    "tier2-framework/agent-infra/ollama-models.yaml": {
        "target": "tier2-framework/agent-infra/ollama-models.spec.md",
        "id": "AG-006",
        "title": "Ollama Models Configuration"
    },

    # Tier 2 Framework - API
    "tier2-framework/specs/api-contracts.yaml": {
        "target": "tier2-framework/api/contracts.spec.md",
        "id": "FW-004",
        "title": "API Contracts"
    },

    # Tier 2 Framework - OCR Engine Components
    "tier2-framework/runtime/coordinate-transforms.yaml": {
        "target": "tier2-framework/ocr-engine/coordinate-transforms.spec.md",
        "id": "FW-012",
        "title": "Coordinate Transforms"
    },
    "tier2-framework/runtime/image-loading.yaml": {
        "target": "tier2-framework/ocr-engine/image-loading.spec.md",
        "id": "FW-020",
        "title": "Image Loading Standards"
    },
    "tier2-framework/ocr-engine/model-management.yaml": {
        "target": "tier2-framework/ocr-engine/model-management.spec.md",
        "id": "FW-023",
        "title": "Model Management"
    },
    "tier2-framework/runtime/orchestration-flow.yaml": {
        "target": "tier2-framework/ocr-engine/orchestration-flow.spec.md",
        "id": "FW-024",
        "title": "Orchestration Flow"
    },
    "tier2-framework/ocr-engine/pipeline-contracts.yaml": {
        "target": "tier2-framework/ocr-engine/pipeline-contracts.spec.md",
        "id": "FW-026",
        "title": "Pipeline Contracts"
    },
    "tier2-framework/ocr-engine/postprocessing-logic.yaml": {
        "target": "tier2-framework/ocr-engine/postprocessing-logic.spec.md",
        "id": "FW-027",
        "title": "Postprocessing Logic"
    },
    "tier2-framework/ocr-engine/preprocessing-logic.yaml": {
        "target": "tier2-framework/ocr-engine/preprocessing-logic.spec.md",
        "id": "FW-028",
        "title": "Preprocessing Logic"
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
    print("=== Extracting Remaining Specs from standards_db.json (Phase 2) ===\n")

    extracted = 0
    failed = []

    for db_key, target_info in SPECS_TO_EXTRACT.items():
        if extract_spec(db_key, target_info):
            extracted += 1
        else:
            failed.append(db_key)

    print(f"\n✅ Extracted {extracted}/{len(SPECS_TO_EXTRACT)} specs")

    if failed:
        print(f"\n⚠️  Failed to extract {len(failed)} specs:")
        for key in failed:
            print(f"  - {key}")

    print("\nNext: Run delete_code_refs.py to remove code glob references")

if __name__ == "__main__":
    main()
