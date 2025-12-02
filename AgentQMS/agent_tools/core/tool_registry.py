#!/usr/bin/env python3
"""
Tool Registry Generator

Scans AgentQMS tools and generates machine-readable (JSON) and human-readable
(markdown) registries for auto-discovery by AI agents.

Usage:
    python tool_registry.py
    python tool_registry.py --output-dir .copilot/context
"""

from __future__ import annotations

import ast
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

from AgentQMS.agent_tools.utils.paths import get_project_root
from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()

PROJECT_ROOT = get_project_root()
AGENT_TOOLS_ROOT = PROJECT_ROOT / "AgentQMS" / "agent_tools"
INTERFACE_ROOT = PROJECT_ROOT / "AgentQMS" / "interface"
COPLOT_CONTEXT_DIR = PROJECT_ROOT / ".copilot" / "context"


@dataclass
class ToolMetadata:
    """Metadata for a single tool."""
    name: str
    category: str
    description: str
    path: str
    is_cli: bool
    usage_hint: str
    tags: list[str]
    make_target: str | None = None
    parameters: dict[str, Any] | None = None


@dataclass
class WorkflowMetadata:
    """Metadata for a Makefile workflow."""
    name: str
    description: str
    usage: str | None = None
    category: str = "workflow"


EXCLUDE_DIRS = {"__pycache__", "_deprecated", "plugins"}
CATEGORIES = {"core", "documentation", "compliance", "utilities", "audit"}


def discover_python_files(base: Path) -> list[Path]:
    """Discover all Python tool files."""
    tools: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base):
        rel = Path(dirpath).relative_to(base)
        parts = set(rel.parts)
        if parts & EXCLUDE_DIRS:
            continue
        for fn in filenames:
            if fn.endswith(".py") and not fn.startswith("_"):
                tools.append(Path(dirpath) / fn)
    return tools


def infer_category(p: Path) -> str:
    """Infer tool category from path."""
    try:
        rel = p.relative_to(AGENT_TOOLS_ROOT)
        if rel.parts:
            top = rel.parts[0]
            return top if top in CATEGORIES else "utilities"
    except ValueError:
        pass
    return "utilities"


def parse_metadata_block(src: str) -> tuple[str | None, str | None, list[str]]:
    """Parse metadata from top-of-file comments."""
    lines = src.split("\n")[:30]
    desc = None
    usage = None
    tags: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("#"):
            break

        if "@tool:" in stripped:
            if "description=" in stripped:
                desc = stripped.split("description=", 1)[1].strip()
            elif "usage=" in stripped:
                usage = stripped.split("usage=", 1)[1].strip()
            elif "tags=" in stripped:
                tag_str = stripped.split("tags=", 1)[1].strip()
                tags.extend(t.strip() for t in tag_str.split(",") if t.strip())
        elif stripped.startswith("# Tool:"):
            desc = stripped[7:].strip()
        elif stripped.startswith("# Usage:"):
            usage = stripped[8:].strip()
        elif stripped.startswith("# Tags:"):
            tag_str = stripped[7:].strip()
            tags.extend(t.strip() for t in tag_str.split(",") if t.strip())

    return (desc, usage, tags)


def parse_module(p: Path) -> tuple[str, bool, str, list[str]]:
    """Parse module and return (doc, is_cli, usage_hint, tags)."""
    try:
        src = p.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src)
    except Exception:
        return ("", False, "", [])

    meta_desc, meta_usage, meta_tags = parse_metadata_block(src)

    doc = meta_desc or (ast.get_docstring(tree) or "")
    if "\n" in doc:
        doc = doc.split("\n", 1)[0]
    doc = doc.strip()[:100]

    is_cli = ("argparse" in src) or ('if __name__ == "__main__"' in src)
    usage_hint = meta_usage or ""
    if not usage_hint and "add_argument(" in src:
        usage_hint = "argparse CLI"

    tags: list[str] = []
    if meta_tags:
        tags.extend(meta_tags)
    else:
        low = src.lower()
        if "streamlit" in low:
            tags.append("ui")
        if "sqlite3" in low or "sqlite" in low:
            tags.append("sqlite")
        if "ast.parse" in low:
            tags.append("ast")
        if "requests" in low:
            tags.append("network")
        if "yaml" in low:
            tags.append("yaml")

    return (doc, is_cli, usage_hint, tags)


def gather_tools() -> list[ToolMetadata]:
    """Gather all tool metadata."""
    tools: list[ToolMetadata] = []
    
    for f in discover_python_files(AGENT_TOOLS_ROOT):
        if f.name == "tool_registry.py":
            continue
        
        cat = infer_category(f)
        name = f.stem
        doc, is_cli, usage_hint, tags = parse_module(f)
        
        if not doc:
            doc = f"{name} tool"
        
        rel_path = str(f.relative_to(PROJECT_ROOT))
        
        tools.append(ToolMetadata(
            name=name,
            category=cat,
            description=doc,
            path=rel_path,
            is_cli=is_cli,
            usage_hint=usage_hint,
            tags=tags,
        ))
    
    tools.sort(key=lambda t: (t.category, t.name))
    return tools


def parse_makefile_targets() -> list[WorkflowMetadata]:
    """Parse Makefile to extract workflow targets."""
    workflows: list[WorkflowMetadata] = []
    makefile_path = INTERFACE_ROOT / "Makefile"
    
    if not makefile_path.exists():
        return workflows
    
    try:
        content = makefile_path.read_text(encoding="utf-8")
        lines = content.split("\n")
        
        for i, line in enumerate(lines):
            # Match pattern: target: ## Description
            if "##" in line and ":" in line:
                parts = line.split("##", 1)
                if len(parts) == 2:
                    target_part = parts[0].strip()
                    desc = parts[1].strip()
                    
                    if target_part.endswith(":"):
                        target_name = target_part[:-1].strip()
                        
                        # Extract usage if present
                        usage = None
                        if "usage:" in desc.lower():
                            usage_match = desc.split("usage:", 1)
                            if len(usage_match) > 1:
                                usage = usage_match[1].strip()
                        
                        # Categorize workflow
                        category = "workflow"
                        if target_name.startswith("create-"):
                            category = "artifact_creation"
                        elif target_name.startswith("validate") or target_name.startswith("compliance"):
                            category = "validation"
                        elif target_name.startswith("context"):
                            category = "context_loading"
                        elif target_name.startswith("docs-"):
                            category = "documentation"
                        elif target_name.startswith("audit-"):
                            category = "audit"
                        
                        workflows.append(WorkflowMetadata(
                            name=target_name,
                            description=desc,
                            usage=usage,
                            category=category,
                        ))
    except Exception as e:
        print(f"Warning: Could not parse Makefile: {e}", file=sys.stderr)
    
    return workflows


def load_architecture_capabilities() -> dict[str, Any]:
    """Load capabilities from architecture.yaml."""
    arch_path = PROJECT_ROOT / ".agentqms" / "state" / "architecture.yaml"
    
    if not arch_path.exists():
        return {}
    
    try:
        with arch_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("capabilities", [])
    except Exception:
        return {}


def generate_json_registry(tools: list[ToolMetadata], workflows: list[WorkflowMetadata]) -> dict[str, Any]:
    """Generate JSON registry."""
    capabilities = load_architecture_capabilities()
    
    registry = {
        "version": "1.0",
        "framework": "AgentQMS",
        "tools": [asdict(t) for t in tools],
        "workflows": [asdict(w) for w in workflows],
        "capabilities": capabilities,
        "categories": {
            cat: [t.name for t in tools if t.category == cat]
            for cat in CATEGORIES
        },
    }
    
    return registry


def generate_markdown_catalog(tools: list[ToolMetadata], workflows: list[WorkflowMetadata]) -> str:
    """Generate markdown catalog."""
    lines: list[str] = []
    lines.append("# AgentQMS Tool Catalog")
    lines.append("")
    lines.append("Auto-generated tool registry for AI agent discovery.")
    lines.append("")
    
    # Tools by category
    lines.append("## Tools by Category")
    lines.append("")
    
    current_cat = None
    for tool in tools:
        if tool.category != current_cat:
            current_cat = tool.category
            lines.append(f"### {current_cat.title()}")
            lines.append("")
            lines.append("| Tool | Description | CLI | Usage |")
            lines.append("|---|---|---|---|")
        
        cli_mark = "✓" if tool.is_cli else "✗"
        rel_path = tool.path.replace("\\", "/")
        lines.append(f"| **{tool.name}** | {tool.description} | {cli_mark} | `{rel_path}` |")
    
    lines.append("")
    lines.append("## Workflows (Makefile Targets)")
    lines.append("")
    lines.append("| Workflow | Description | Category |")
    lines.append("|---|---|---|")
    
    for workflow in workflows:
        lines.append(f"| `{workflow.name}` | {workflow.description} | {workflow.category} |")
    
    return "\n".join(lines)


def write_registry_files(output_dir: Path) -> None:
    """Generate and write all registry files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tools = gather_tools()
    workflows = parse_makefile_targets()
    
    # Generate JSON registry
    json_registry = generate_json_registry(tools, workflows)
    json_path = output_dir / "tool-registry.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_registry, f, indent=2)
    print(f"✓ Generated {json_path}")
    
    # Generate markdown catalog
    md_catalog = generate_markdown_catalog(tools, workflows)
    md_path = output_dir / "tool-catalog.md"
    md_path.write_text(md_catalog, encoding="utf-8")
    print(f"✓ Generated {md_path}")


def main() -> int:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate AgentQMS tool registry")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(COPLOT_CONTEXT_DIR),
        help="Output directory for registry files",
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    try:
        write_registry_files(output_dir)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

