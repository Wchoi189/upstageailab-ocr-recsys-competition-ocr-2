import re
from pathlib import Path
from typing import Dict, Any, List

class SpecParser:
    def __init__(self, project_root: Path):
        self.specs_root = project_root / "AgentQMS" / "specs"
        self.compliance_spec = self.specs_root / "tier1-contracts" / "compliance.spec.md"
        self.validation_spec = self.specs_root / "tier1-contracts" / "validation.spec.md"

    def parse_artifact_types(self) -> Dict[str, Dict[str, str]]:
        """
        Parses the Artifact Types table from compliance.spec.md.
        Returns: { 'implementation_plan': {'prefix': 'implementation_plan_', 'directory': 'implementation_plans/'} }
        """
        types = {}
        if not self.compliance_spec.exists():
            return types

        content = self.compliance_spec.read_text(encoding="utf-8")
        lines = content.splitlines()

        in_table = False
        for line in lines:
            if "| TypeKey | Prefix | Directory" in line:
                in_table = True
                continue
            if in_table and "|" in line and "---" not in line:
                # Row: | **implementation_plan** | `implementation_plan_` | `implementation_plans/` | ... |
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 3:
                    type_key = parts[0].replace("*", "").strip()
                    prefix = parts[1].replace("`", "").strip()
                    directory = parts[2].replace("`", "").strip()

                    # Handle special paths (e.g. .../session_notes/)
                    if directory.startswith(".../"):
                        # This implies a relative path suffix. For now, treat literally or map.
                        # Legacy session_notes were in completed_plans/completion_summaries/session_notes/
                        # We will just capture what's there.
                        pass

                    types[type_key] = {
                        "prefix": prefix,
                        "directory": directory,
                        "frontmatter_type": type_key
                    }
            elif in_table and not line.strip():
                in_table = False

        return types

    def parse_required_fields(self) -> List[str]:
        """
        Parses the YAML code block under 'Required Fields' in validation.spec.md.
        """
        fields = []
        if not self.validation_spec.exists():
            return fields

        content = self.validation_spec.read_text(encoding="utf-8")

        # Regex to find the code block after "Required Fields"
        # Search for ### Required Fields\n```yaml\n(.*?)\n```
        match = re.search(r"### Required Fields\s+```yaml\n(.*?)\n```", content, re.DOTALL)
        if match:
            block = match.group(1)
            # Parse lines: "title: String" -> "title"
            for line in block.splitlines():
                if ":" in line:
                    key = line.split(":")[0].strip()
                    fields.append(key)

        return fields
